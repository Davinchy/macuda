// SPDX-License-Identifier: GPL-2.0-only
// RTX 4090 (AD102) tensor-speed unlock transaction on the unchanged TinyGPU transport. See tensor_unlock.h.
//
// Adapted from JustAResearcher/gpu-unlock-ada @ 8855e7bca8ff142204e02301331fd56b26699819 (GPL-2.0-only),
// kernel/gpu_unlock/unlock_610.c from the pinned, independently exercised revision. The pure parsers are
// vendored verbatim in unlock610/. Adaptations, each forced by the transport audit (transport-audit.md):
//   - Linux coherent DMA -> pci->dma_alloc (server shm + DART). Every buffer's IOVAs are checked < 4 GiB,
//     contiguous and 4 KiB-aligned BEFORE any address is published to firmware (audit §3, inputs F6).
//   - One dma mapping per radix page is impossible (server cap: 128 mappings/session); the radix tables and GSP
//     image pages live in ONE validated-contiguous buffer, sub-allocated at 4 KiB (audit §3 rule 3).
//   - pcie_flr -> pci->reset, whose RPC status is meaningless (the dext discards Reset's return value, audit §4):
//     only verified effects count. Apple's FunctionReset saves/restores config itself; BARs/COMMAND are still
//     verified and repaired from the claim-time snapshot (dev.c pattern).
//   - dma_wmb -> atomic_thread_fence(memory_order_release); ktime -> tinynv_now_s; bounded polls everywhere.
//   - Module-reference retention is impossible in userspace. firmware_started && !quiesced => recovery_required:
//     the caller keeps process/socket/mappings alive until an enclosure power-off (O14
//     crash retention is a known limitation, not a guarantee claimed here).
//   - GA102 paths and the AD103 canary are dropped: this file admits 10de:2684 / AD102 only, per spec.
//   - gpu_unlock_remove_apertures (Linux framebuffer handoff) has no macOS counterpart and is not ported.
#include "tensor_unlock.h"
#include "internal.h"
#include "sha256.h"
#include "unlock610/prep_610.h"
#include "unlock610/layout_610.h"
#include "unlock610/vbios_610.h"
#include "unlock610/gfw_eval.h"
#include <errno.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#define TU_PCI_ID 0x268410deu // 10de:2684, desktop RTX 4090 only
#define TU_AD102_ARCH 0x192u  // PMC_BOOT_0 bits 28:20

#define SEC2_BASE 0x00840000u
#define SEC2_FBIF 0x00840600u
#define SEC2_FALCON2 0x00841000u
#define GSP_BASE 0x00110000u
#define GSP_FBIF 0x00110600u
#define GSP_FALCON2 0x00111000u
#define GU_SS0_UNLOCK 0x88888888u
#define GU_SS1_UNLOCK 0x8u
// The public post-boot verifier (tensor_unlock.h) must agree with the transaction's own gates.
_Static_assert(GU_SS0_UNLOCK == TU_UNLOCK_SS0 && GU_SS1_UNLOCK == TU_UNLOCK_SS1, "public unlock selector constants drifted");
#define GU_AD102_MARKER 0x67b00001u
#define GU_SPEED_READOUT_MASK 0x07ffffffu
_Static_assert(GU_SPEED_READOUT_MASK == TU_UNLOCK_READOUT_MASK, "public readout mask drifted");
#define BOOTER_MAGIC 0x314c5442u
#define TU_BOOT_IMAGE_SIZE 36864u
#define TU_DMA_LIMIT (1ull << 32) // the recipe's descriptor fields are 32-bit (inputs F6)

#define PCI_COMMAND 0x04
#define PCI_COMMAND_MEMORY 0x2
#define PCI_COMMAND_MASTER 0x4

volatile sig_atomic_t tensor_unlock_abort;
static volatile sig_atomic_t tu_got_term;
static volatile sig_atomic_t tu_holding;

// Offline-test seam: scales every deadline and poll sleep, never the gates themselves. 1.0 on the live path.
static double tu_time_scale = 1.0;
void tensor_unlock_set_time_scale(double scale) { tu_time_scale = scale > 0 ? scale : 1.0; }

const tensor_unlock_pins_t tensor_unlock_pins = {
  .booter = {"booter_load_ad102.bin", 57672, "6ee05b20e1dc244c08c052f36ef28469b8d09f4955d1fe33e48a179e11a7fd49"},
  .payload = {"v67-ad102-610.43.03-four-plm.bin", 64000, "44c36b2d6b3424e53e18f3e55db85c833655de071ce66ce42749588a4a7dc8cb"},
  .gsp = {"gsp_ga10x.bin", 84277400, "572373620a37418f24dc16b5031c39338778c3257e48e8408de9a57291b24f3a"},
  .boot = {"gsp_boot_ad102.bin", 36864, "45d6335a8185d6ff13e8d1484cea73f68d3d13aa1c19dc080ed1c3482dd5bac0"},
  .desc = {"gsp_boot_desc_ad102.bin", 84, "83f8f3f1f5f9496e0ea8fc7103c2beb64d0b6de2f5ffe725f13514de5a95fa42"},
};

const char *tensor_unlock_stage_name(int stage) {
  switch (stage) {
    case TU_STAGE_NONE: return "none";
    case TU_STAGE_ASSETS: return "assets";
    case TU_STAGE_CLAIM: return "claim";
    case TU_STAGE_LAYOUT: return "layout";
    case TU_STAGE_DMA: return "dma";
    case TU_STAGE_GFW: return "gfw";
    case TU_STAGE_FWSEC: return "fwsec";
    case TU_STAGE_PUBLISH: return "publish";
    case TU_STAGE_PAYLOAD: return "payload";
    case TU_STAGE_RESTORE_SIG: return "restore-signature";
    case TU_STAGE_SELECTORS: return "selectors";
    case TU_STAGE_RESET: return "reset";
    case TU_STAGE_DONE: return "done";
    default: return "?";
  }
}

#ifndef TINYNV_UNLOCK_DEFAULT_DIR
#define TINYNV_UNLOCK_DEFAULT_DIR ""
#endif

const char *tensor_unlock_bundle_dir(void) {
  const char *e = getenv("TINYNV_UNLOCK_BUNDLE");
  if (e && *e) return e;
  return TINYNV_UNLOCK_DEFAULT_DIR;
}

// ---- assets ---------------------------------------------------------------------------------------------------------

static int tu_load_blob(const char *dir, const tensor_unlock_pin_t *pin, tensor_unlock_blob_t *out) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", dir, pin->name);
  struct stat st;
  if (lstat(path, &st)) return tinynv_fail("unlock asset %s: %s", path, strerror(errno));
  if (!S_ISREG(st.st_mode)) return tinynv_fail("unlock asset %s is not a regular file (symlinks refused)", path);
  if ((uint64_t)st.st_size != pin->size)
    return tinynv_fail("unlock asset %s is %lld bytes, pinned %llu", path, (long long)st.st_size,
                       (unsigned long long)pin->size);
  FILE *f = fopen(path, "rb");
  if (!f) return tinynv_fail("unlock asset %s: %s", path, strerror(errno));
  uint8_t *buf = malloc(pin->size);
  if (!buf) { fclose(f); return tinynv_fail("out of memory for %s", path); }
  size_t n = fread(buf, 1, pin->size, f);
  fclose(f);
  if (n != pin->size) { free(buf); return tinynv_fail("unlock asset %s read short", path); }
  uint8_t digest[32];
  char hex[65];
  tinynv_sha256(buf, pin->size, digest);
  tinynv_sha256_hex(digest, hex);
  if (strcmp(hex, pin->sha256)) {
    free(buf);
    return tinynv_fail("unlock asset %s hashes to %s, not the pinned %s: refusing to run it", path, hex, pin->sha256);
  }
  out->data = buf;
  out->size = pin->size;
  return 0;
}

static void tu_blob_free(tensor_unlock_blob_t *b) { free(b->data); b->data = NULL; b->size = 0; }

void tensor_unlock_assets_free(struct tu_assets *a) {
  if (!a) return;
  tu_blob_free(&a->booter);
  tu_blob_free(&a->payload);
  tu_blob_free(&a->gsp);
  tu_blob_free(&a->boot);
  tu_blob_free(&a->desc);
  free(a);
}

int tensor_unlock_assets_load(const char *dir, const tensor_unlock_pins_t *pins, struct tu_assets **out) {
  *out = NULL;
  if (!dir || !*dir) return tinynv_fail("no unlock bundle directory (set TINYNV_UNLOCK_BUNDLE or prepare the bundle)");
  struct tu_assets *a = calloc(1, sizeof(*a));
  if (!a) return tinynv_fail("out of memory");
  int rc = -1;
  if (tu_load_blob(dir, &pins->booter, &a->booter)) goto out;
  if (tu_load_blob(dir, &pins->payload, &a->payload)) goto out;
  if (tu_load_blob(dir, &pins->gsp, &a->gsp)) goto out;
  if (tu_load_blob(dir, &pins->boot, &a->boot)) goto out;
  if (tu_load_blob(dir, &pins->desc, &a->desc)) goto out;

  // BTL1 container: magic, version, exact pinned AD102 geometry, ranges inside the file.
  const uint8_t *d = a->booter.data;
  uint32_t magic, version, image_size, sig_bytes;
  memcpy(&magic, d, 4);
  memcpy(&version, d + 4, 4);
  if (magic != BOOTER_MAGIC || version != 1) { tinynv_fail("unlock booter: bad BTL1 magic/version"); goto out; }
  memcpy(&a->hdr, d + 8, sizeof(a->hdr));
  memcpy(&image_size, d + 8 + sizeof(a->hdr), 4);
  memcpy(&sig_bytes, d + 12 + sizeof(a->hdr), 4);
  if (image_size > a->booter.size - 72 || sig_bytes > a->booter.size - 72 - image_size ||
      a->booter.size != 72u + image_size + sig_bytes) { tinynv_fail("unlock booter: container sizes disagree"); goto out; }
  if (!gpu_unlock_booter_geometry_match(&a->hdr, image_size, sig_bytes) ||
      gpu_unlock_booter_validate(GU_BOOTER_AD10X, &a->hdr, image_size, sig_bytes)) {
    tinynv_fail("unlock booter: not the pinned AD102 geometry");
    goto out;
  }
  a->booter_image_size = image_size;

  if (a->payload.size != GU_V67_SIZE) { tinynv_fail("unlock payload: %zu bytes, want %u", a->payload.size, GU_V67_SIZE); goto out; }
  if (gpu_unlock_elf_section(a->gsp.data, a->gsp.size, ".fwimage", &a->image_span) ||
      gpu_unlock_elf_section(a->gsp.data, a->gsp.size, ".fwsignature_ad10x", &a->sig_span)) {
    tinynv_fail("unlock GSP ELF: .fwimage/.fwsignature_ad10x section missing or malformed");
    goto out;
  }
  if (!a->image_span.size || a->image_span.size > (128ull << 20) || a->sig_span.size != 4096) {
    tinynv_fail("unlock GSP ELF: image/signature span out of bounds");
    goto out;
  }
  if (gpu_unlock_radix_plan(a->image_span.size, &a->radix)) { tinynv_fail("unlock radix plan rejected"); goto out; }
  if (a->boot.size != TU_BOOT_IMAGE_SIZE) { tinynv_fail("unlock GSP boot: %zu bytes, want %u", a->boot.size, TU_BOOT_IMAGE_SIZE); goto out; }
  if (gpu_unlock_boot_desc_parse(a->desc.data, a->desc.size, (uint32_t)a->boot.size, &a->boot_desc)) {
    tinynv_fail("unlock boot descriptor rejected");
    goto out;
  }
  rc = 0;
  *out = a;
out:
  if (rc) tensor_unlock_assets_free(a);
  return rc;
}

// ---- transaction state ----------------------------------------------------------------------------------------------

typedef struct {
  tinynv_pci_t *pci;
  tinynv_mmio_t bar;
  int mapped;
  uint64_t bar_len;
  uint32_t pcie_cap, pm_cap;
  uint16_t original_cmd;
  int cmd_valid, master;
  uint32_t bars[6];
  struct tu_assets *assets;
  tensor_unlock_result_t *r;
  int stage, last_ok;
  int error; // first error, positive errno; never overwritten

  struct gpu_unlock_layout_in layout_in;
  struct gpu_unlock_layout_out layout;

  // transaction buffers: CPU pointer + validated IOVA base. One dma mapping each (radix+image is one big mapping).
  struct {
    const char *name;
    tinynv_dma_t dma;
    uint64_t iova;
    int live;
  } buf[TU_MAX_BUFFERS];
  int nbuf;

  uint32_t chip_entry, plm_entry;
  uint32_t permission_before[4];
  int permission_saved, permission_sampled, payload_verified;
  int sec2_started, sec2_halt_verified;
  int firmware_started, quiesced, published;
  unsigned signature_calls;
  uint64_t stock_size;

  uint32_t last_offset, last_value;
  int last_valid;
} tu_work_t;

static const struct {
  const char *name;
  uint32_t offset, expected;
  int required;
} ad102_permissions[] = {
  {"WPR_CFG", 0x001fa7cc, 0xfffff0ff, 1},
  {"FBPA", 0x009a0148, 0xffffffff, 0}, // observed unchanged 0xffffff0f on the measured card; optional
  {"WPR", 0x001fa7c4, 0xffffffff, 1},
  {"FEAT", GU_FEAT_PLM_OFFSET, 0xffffffff, 1},
};

static void tu_sleep_us(long us) {
  double s = us * tu_time_scale;
  if (s < 1) s = 1;
  struct timespec ts = {.tv_sec = (time_t)(s / 1e6), .tv_nsec = (long)((long)s % 1000000) * 1000};
  nanosleep(&ts, NULL);
}

// Record the FIRST failure with its stage and site; later cleanup errors never overwrite it (spec req 8).
static int tu_fail(tu_work_t *w, int err, const char *func, int line, const char *fmt, ...) __attribute__((format(printf, 5, 6)));
static int tu_fail(tu_work_t *w, int err, const char *func, int line, const char *fmt, ...) {
  if (!err) return 0;
  if (!w->error) {
    w->error = err;
    char tmp[400];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    snprintf(w->r->first_error, sizeof(w->r->first_error), "%s (stage %s, %s:%d, errno %d)", tmp,
             tensor_unlock_stage_name(w->stage), func, line, err);
    fprintf(stderr, "tensor-unlock: FAILURE %s\n", w->r->first_error);
  }
  return err;
}
#define TU_FAIL(w, err, ...) tu_fail((w), (err), __func__, __LINE__, __VA_ARGS__)

static int tu_inaccessible(uint32_t v) { return v == ~0u || (v & 0xffff0000u) == 0xbadf0000u; }

static int tu_in_range(uint64_t length, uint64_t off, uint64_t bytes) {
  return off <= length && bytes <= length - off;
}

static int tu_rd(tu_work_t *w, uint32_t off, uint32_t *value) {
  w->last_offset = off;
  w->last_valid = 0;
  if (!w->mapped || off > w->bar_len || w->bar_len - off < 4) return TU_FAIL(w, EINVAL, "BAR read outside window: %#x", off);
  *value = nv_rd32(&w->bar, off);
  w->last_value = *value;
  w->last_valid = 1;
  return 0;
}

static void tu_wr(tu_work_t *w, uint32_t off, uint32_t v) { nv_wr32(&w->bar, off, v); }

// Bounded poll. All-ones / 0xbadf____ reads are inaccessibility, never a matched value. A tolerated timeout returns
// ETIMEDOUT without recording a failure (the pre-reset HW-bug-3419321 wait); every other timeout is a failure.
static int tu_poll(tu_work_t *w, uint32_t off, uint32_t mask, uint32_t expect, long budget_us, int timeout_is_failure) {
  double deadline = tinynv_now_s() + budget_us * tu_time_scale / 1e6;
  uint32_t v = 0;
  for (;;) {
    if (tensor_unlock_abort) return TU_FAIL(w, EINTR, "abort requested while polling %#x", off);
    if (tu_rd(w, off, &v) || tu_inaccessible(v))
      return TU_FAIL(w, EIO, "poll inaccessible: reg=%#x last=%#x mask=%#x expected=%#x", off, v, mask, expect);
    if ((v & mask) == expect) return 0;
    if (tinynv_now_s() >= deadline) {
      if (!timeout_is_failure) return ETIMEDOUT;
      return TU_FAIL(w, ETIMEDOUT, "poll timeout: reg=%#x last=%#x mask=%#x expected=%#x budget_us=%ld", off, v, mask,
                     expect, budget_us);
    }
    tu_sleep_us(100);
  }
}

static uint32_t tu_cfg(tu_work_t *w, uint32_t off, uint32_t size) { return w->pci->cfg_read(w->pci, off, size); }

static void tu_cfg_write(tu_work_t *w, uint32_t off, uint32_t size, uint32_t val) {
  w->pci->cfg_write(w->pci, off, size, val);
  w->pci->cfg_read(w->pci, off, size); // read back so the write has landed (dev.c cfg_write_flush)
}

// Walk the capability list the standard way (dev.c pattern). Returns the offset of `id`, or 0.
static uint32_t tu_find_cap(tu_work_t *w, uint32_t id) {
  uint32_t cap = tu_cfg(w, 0x34, 1) & 0xfc;
  for (int guard = 0; cap && cap != 0xfc && guard < 48; guard++) {
    if (tu_cfg(w, cap, 1) == id) return cap;
    cap = tu_cfg(w, cap + 1, 1) & 0xfc;
  }
  return 0;
}

// ---- transaction memory (audit §3: validate before publication, fail closed) -----------------------------------------

static int tu_dma_alloc(tu_work_t *w, const char *name, size_t size, uint64_t *iova, tinynv_dma_t **out) {
  if (w->nbuf >= TU_MAX_BUFFERS) return TU_FAIL(w, ENOMEM, "too many transaction buffers");
  tinynv_dma_t *dma = &w->buf[w->nbuf].dma;
  if (w->pci->dma_alloc(w->pci, size, dma))
    return TU_FAIL(w, EIO, "dma allocation failed: %s (%zu bytes): %s", name, size, tinynv_last_error());
  w->r->ndma_alloc++;
  // The server rounds up (16 KiB minimum); validate the MAPPED size, page by page, before anything is published.
  // A rejected buffer never reached the device, so it is reclaimable right here (and must be: the failure report
  // counts allocations against releases).
  int bad = 0;
  if (!dma->npages || dma->npages * 0x1000 != dma->size || dma->size < size) {
    TU_FAIL(w, EIO, "%s: mapping describes %zu pages of %zu bytes (requested %zu)", name, dma->npages, dma->size, size);
    bad = 1;
  } else if (dma->pages[0] & 0xfff) {
    TU_FAIL(w, EIO, "%s: first IOVA %#llx is not 4 KiB-aligned", name, (unsigned long long)dma->pages[0]);
    bad = 1;
  } else {
    for (size_t i = 0; i < dma->npages; i++) {
      if (dma->pages[i] >= TU_DMA_LIMIT) {
        TU_FAIL(w, EIO, "%s: page %zu IOVA %#llx is above 4 GiB; the recipe's descriptors are 32-bit", name, i,
                (unsigned long long)dma->pages[i]);
        bad = 1;
        break;
      }
      if (i && dma->pages[i] != dma->pages[i - 1] + 0x1000) {
        TU_FAIL(w, EIO, "%s: IOVA run breaks at page %zu (%#llx after %#llx); falcon DMA needs one contiguous run",
                name, i, (unsigned long long)dma->pages[i], (unsigned long long)dma->pages[i - 1]);
        bad = 1;
        break;
      }
    }
  }
  if (bad) {
    w->pci->dma_free(w->pci, dma);
    w->r->ndma_freed++;
    return w->error;
  }
  w->buf[w->nbuf].name = name;
  w->buf[w->nbuf].iova = dma->pages[0];
  w->buf[w->nbuf].live = 1;
  w->nbuf++;
  w->r->buffers[w->r->nbuffer++] = (tensor_unlock_iova_t){.name = name, .iova = dma->pages[0], .size = dma->size};
  fprintf(stderr, "tensor-unlock: buffer %s: %zu bytes, iova %#llx..%#llx (validated: <4GiB, contiguous, aligned)\n", name,
          dma->size, (unsigned long long)dma->pages[0], (unsigned long long)(dma->pages[0] + dma->size));
  *iova = dma->pages[0];
  *out = dma;
  return 0;
}

static void tu_dma_free_all(tu_work_t *w) {
  for (int i = 0; i < w->nbuf; i++) {
    if (!w->buf[i].live) continue;
    w->pci->dma_free(w->pci, &w->buf[i].dma);
    w->buf[i].live = 0;
    w->r->ndma_freed++;
  }
}

// ---- falcon execution (ported gates: a mailbox marker alone never completes a stage) ----------------------------------

struct falcon_eng { uint32_t base, fbif, falcon2; };
static const struct falcon_eng gu_sec2 = {SEC2_BASE, SEC2_FBIF, SEC2_FALCON2};
static const struct falcon_eng gu_gsp = {GSP_BASE, GSP_FBIF, GSP_FALCON2};

static int falcon_reset(tu_work_t *w, const struct falcon_eng *eng) {
  uint32_t engine, hwcfg, bcr, chip;
  // 610 kflcnPreResetWait_GA102 permits this timeout (HW bug 3419321).
  int ret = tu_poll(w, eng->base + 0xf4, 1u << 31, 1u << 31, 1000, 0);
  if (ret && ret != ETIMEDOUT) return TU_FAIL(w, ret, "falcon pre-reset wait failed");
  if (tu_rd(w, eng->base + 0x3c0, &engine) || tu_inaccessible(engine))
    return TU_FAIL(w, ENODEV, "falcon engine register inaccessible");
  tu_wr(w, eng->base + 0x3c0, engine | 1);
  for (int i = 0; i < 10; i++) nv_rd32(&w->bar, eng->base + 0x3c0);
  tu_wr(w, eng->base + 0x3c0, engine & ~1u);
  for (int i = 0; i < 10; i++) nv_rd32(&w->bar, eng->base + 0x3c0);
  if (tu_poll(w, eng->base + 0xf4, 1u << 12, 0, 1000000, 1)) return w->error ? w->error : EIO;
  if (tu_rd(w, eng->base + 0xf4, &hwcfg) || tu_inaccessible(hwcfg)) return TU_FAIL(w, EIO, "falcon hwcfg2 inaccessible");
  if (hwcfg & (1u << 10)) {
    if (tu_rd(w, eng->falcon2 + 0x668, &bcr) || tu_inaccessible(bcr)) return TU_FAIL(w, ENODEV, "falcon bcr inaccessible");
    if (bcr & (1u << 4)) {
      tu_wr(w, eng->falcon2 + 0x668, 0);
      if (tu_poll(w, eng->falcon2 + 0x668, 0x11, 1, 1000000, 1)) return w->error ? w->error : EIO;
    }
  }
  if (tu_rd(w, 0, &chip) || tu_inaccessible(chip)) return TU_FAIL(w, ENODEV, "chip id inaccessible during falcon reset");
  tu_wr(w, eng->base + 0x84, chip);
  return 0;
}

// s_dmaTransfer_GA102 moves ceil(size/256) blocks; the last block reads up to 255 bytes past the length.
static uint64_t falcon_dma_span(uint32_t size) { return ((uint64_t)size + 255) & ~255ull; }

static int falcon_dma_valid(uint32_t dest, uint32_t mem_off, uint64_t src, uint32_t size) {
  uint64_t span = falcon_dma_span(size);
  return size && !((dest | mem_off | src) & 255) && (uint64_t)dest + span <= (1ull << 24) &&
         (uint64_t)mem_off + span <= (1ull << 32) && src < (1ull << 49) && (uint64_t)mem_off + span <= (1ull << 49) - src;
}

static uint32_t gu_dma_cmd(int imem) { return (6u << 8) | (imem ? (1u << 4) | (1u << 2) : 0); }

static int falcon_dma(tu_work_t *w, uint32_t base, uint32_t dest, uint32_t mem_off, uint64_t src, uint32_t size, uint32_t cmd) {
  if (!falcon_dma_valid(dest, mem_off, src, size))
    return TU_FAIL(w, EINVAL, "falcon DMA geometry rejected: dest=%#x off=%#x src=%#llx size=%#x", dest, mem_off,
                   (unsigned long long)src, size);
  // From here the engine may read the buffer, so it must outlive a timeout (recovery, not release).
  w->firmware_started = 1;
  if (tu_poll(w, base + 0x118, 1, 0, 1000000, 1)) return w->error ? w->error : EIO;
  tu_wr(w, base + 0x110, (uint32_t)(src >> 8));
  tu_wr(w, base + 0x128, (uint32_t)((src >> 8) >> 32) & 0x1ff);
  for (uint32_t done = 0; done < size; done += 256) {
    if (tu_poll(w, base + 0x118, 1, 0, 1000000, 1)) return w->error ? w->error : EIO;
    tu_wr(w, base + 0x114, dest + done);
    tu_wr(w, base + 0x11c, mem_off + done);
    tu_wr(w, base + 0x118, cmd);
  }
  return tu_poll(w, base + 0x118, 2, 2, 1000000, 1) ? (w->error ? w->error : EIO) : 0;
}

// A halted falcon leaves its reason in the mailboxes, IRQ status and handshake words. Read them while BAR0 is ours.
static void log_falcon_state(tu_work_t *w, const char *name, const struct falcon_eng *eng) {
  uint32_t base = eng->base, a2 = eng->falcon2;
  uint32_t cpuctl = 0, irqstat = 0, mb0 = 0, mb1 = 0, os = 0, rm = 0, debug = 0, bootvec = 0, engine = 0, hwcfg = 0;
  uint32_t pc_lo = 0, pc_hi = 0, riscv = 0, rpc = 0, priv = 0, priv_info = 0, priv_addr = 0, priv_addr_hi = 0, hub = 0;
  tu_rd(w, base + 0x100, &cpuctl); tu_rd(w, base + 0x08, &irqstat);
  tu_rd(w, base + 0x40, &mb0); tu_rd(w, base + 0x44, &mb1);
  tu_rd(w, base + 0x80, &os); tu_rd(w, base + 0x84, &rm);
  tu_rd(w, base + 0x94, &debug); tu_rd(w, base + 0x104, &bootvec);
  tu_rd(w, base + 0x3c0, &engine); tu_rd(w, base + 0x108, &hwcfg);
  tu_rd(w, a2 + 0x40c, &pc_lo); tu_rd(w, a2 + 0x410, &pc_hi);
  tu_rd(w, a2 + 0x388, &riscv); tu_rd(w, a2 + 0x3ec, &rpc);
  tu_rd(w, a2 + 0x500, &priv); tu_rd(w, a2 + 0x504, &priv_info);
  tu_rd(w, a2 + 0x508, &priv_addr); tu_rd(w, a2 + 0x50c, &priv_addr_hi);
  tu_rd(w, a2 + 0x510, &hub);
  fprintf(stderr, "tensor-unlock: falcon %s base=%#x cpuctl=%#x irqstat=%#x mailbox=%#x:%#x os=%#x rm=%#x debug=%#x "
                  "bootvec=%#x engine=%#x hwcfg=%#x tracepc=%#x:%#x\n", name, base, cpuctl, irqstat, mb1, mb0, os, rm,
          debug, bootvec, engine, hwcfg, pc_hi, pc_lo);
  fprintf(stderr, "tensor-unlock: falcon %s riscv=%#x rpc=%#x priv_err=%#x info=%#x addr=%#x:%#x hub_err=%#x\n", name,
          riscv, rpc, priv, priv_info, priv_addr_hi, priv_addr, hub);
}

static int falcon_hs(tu_work_t *w, const struct falcon_eng *eng, uint32_t code_off, uint32_t code_size, uint32_t code_va,
                     uint32_t code_pa, uint32_t data_off, uint32_t data_size, uint32_t data_pa, uint32_t sig_dmem,
                     uint32_t engine_mask, uint32_t ucode_id, uint64_t image, uint32_t mailbox0, uint32_t mailbox1,
                     int wait_plm) {
  uint32_t trans, cpuctl;
  if (image > ~0ull - code_off || image + code_off < code_va || image > ~0ull - data_off)
    return TU_FAIL(w, ERANGE, "falcon image offsets wrap");
  uint64_t code_src = image + code_off - code_va;
  if (!falcon_dma_valid(code_pa, code_va, code_src, code_size) ||
      (data_size && !falcon_dma_valid(data_pa, 0, image + data_off, data_size)))
    return TU_FAIL(w, EINVAL, "falcon HS geometry rejected");
  int ret = falcon_reset(w, eng);
  if (ret) return ret;
  if (wait_plm) {
    // The SEC2 payload copy must still be byte-exact at start; a changed copy is not the validated recipe.
    tinynv_dma_t *v67 = &w->buf[1].dma; // allocation order is fixed: booter, v67, ...
    if (!w->buf[1].live || v67->size < GU_V67_SIZE || memcmp(v67->va, w->assets->payload.data, GU_V67_SIZE))
      return TU_FAIL(w, EIO, "AD102 payload copy changed before SEC2 start; aborting");
  }
  if (tu_rd(w, eng->fbif + 0x24, &trans) || tu_inaccessible(trans)) return TU_FAIL(w, EIO, "FBIF+0x24 inaccessible");
  tu_wr(w, eng->fbif + 0x24, trans | (1u << 7));
  tu_wr(w, eng->base + 0x10c, 0);
  if (tu_rd(w, eng->fbif, &trans) || tu_inaccessible(trans)) return TU_FAIL(w, ENODEV, "FBIF TRANSCFG inaccessible");
  tu_wr(w, eng->fbif, (trans & ~7u) | 5u); // sysmem, physical (IOVA behind DART), no context
  atomic_thread_fence(memory_order_release);
  ret = falcon_dma(w, eng->base, code_pa, code_va, code_src, code_size, gu_dma_cmd(1));
  if (ret) return ret;
  if (data_size) {
    ret = falcon_dma(w, eng->base, data_pa, 0, image + data_off, data_size, gu_dma_cmd(0));
    if (ret) return ret;
  }
  tu_wr(w, eng->falcon2 + 0x210, sig_dmem);
  tu_wr(w, eng->falcon2 + 0x19c, engine_mask);
  tu_wr(w, eng->falcon2 + 0x198, ucode_id & 0xff);
  tu_wr(w, eng->falcon2 + 0x180, 1);
  tu_wr(w, eng->base + 0x104, code_va);
  tu_wr(w, eng->base + 0x40, mailbox0);
  tu_wr(w, eng->base + 0x44, mailbox1);
  if (tu_rd(w, eng->base + 0x100, &cpuctl) || tu_inaccessible(cpuctl)) return TU_FAIL(w, EIO, "CPUCTL inaccessible before start");
  fprintf(stderr, "tensor-unlock: falcon start: base=%#x cpuctl=%#x code_va=%#x sig_dmem=%#x engine_mask=%#x ucode=%u "
                  "mailbox=%#x:%#x\n", eng->base, cpuctl, code_va, sig_dmem, engine_mask, ucode_id, mailbox1, mailbox0);
  if (eng->base == SEC2_BASE) {
    w->sec2_started = 1;
    w->sec2_halt_verified = 0;
  }
  if (cpuctl & (1u << 6)) tu_wr(w, eng->base + 0x130, 1u << 1);
  else tu_wr(w, eng->base + 0x100, 1u << 1);
  if (wait_plm) {
    double deadline = tinynv_now_s() + 5.0 * tu_time_scale;
    for (;;) {
      uint32_t plm, engine, marker = 0;
      if (tensor_unlock_abort) return TU_FAIL(w, EINTR, "abort requested while SEC2 runs");
      if (tu_cfg(w, 0, 4) != TU_PCI_ID || tu_rd(w, eng->base + 0x100, &engine) || tu_inaccessible(engine) ||
          tu_rd(w, GU_FEAT_PLM_OFFSET, &plm) || tu_rd(w, eng->base + 0x40, &marker))
        return TU_FAIL(w, ENODEV, "SEC2 wait: device or registers inaccessible");
      // Ada's explicit marker is independent evidence that all-ones PLM is the payload result. Full permission and
      // HALT checks follow in the payload stage; this alone is never success.
      if (plm == ~0u && w->plm_entry != ~0u && marker == GU_AD102_MARKER) return 0;
      if ((engine & (1u << 4)) || tinynv_now_s() >= deadline) {
        fprintf(stderr, "tensor-unlock: early PLM failed: cpuctl=%#x plm=%#x entry=%#x\n", engine, plm, w->plm_entry);
        uint32_t lo = 0, hi = 0;
        tu_rd(w, 0x001fa824, &lo);
        tu_rd(w, 0x001fa828, &hi);
        fprintf(stderr, "tensor-unlock: wpr2=%#x:%#x\n", hi, lo);
        log_falcon_state(w, "sec2", &gu_sec2);
        log_falcon_state(w, "gsp", &gu_gsp);
      }
      if (engine & (1u << 4)) return TU_FAIL(w, EIO, "SEC2 halted before the payload completed (cpuctl=%#x plm=%#x)", engine, plm);
      if (tinynv_now_s() >= deadline) return TU_FAIL(w, ETIMEDOUT, "SEC2 payload did not complete in 5 s");
      tu_sleep_us(100);
    }
  }
  return tu_poll(w, eng->base + 0x100, 1u << 4, 1u << 4, 5000000, 1) ? (w->error ? w->error : EIO) : 0;
}

// ---- AD102 permission / readout gates ---------------------------------------------------------------------------------

// A PLM may legitimately contain all ones. Independent PCI and BAR identity checks distinguish that value from a
// generally inaccessible device. All four samples are logged even when an earlier one fails.
static int ad102_sample_permissions(tu_work_t *w, const char *phase, uint32_t values[4], int before) {
  uint32_t identity = tu_cfg(w, 0, 4), chip = 0;
  int accessible = identity == TU_PCI_ID && !tu_rd(w, 0, &chip) && chip && !tu_inaccessible(chip) &&
                   (before || chip == w->chip_entry);
  if (before && accessible) w->chip_entry = chip;
  int ret = 0;
  for (int i = 0; i < 4; i++) {
    values[i] = 0;
    int valid = !tu_rd(w, ad102_permissions[i].offset, &values[i]) && (values[i] & 0xffff0000u) != 0xbadf0000u && accessible;
    fprintf(stderr, "tensor-unlock: permission %s: %s[%#x]=%#x expected=%#x required=%d valid=%d pci=%#x chip=%#x\n",
            phase, ad102_permissions[i].name, ad102_permissions[i].offset, values[i], ad102_permissions[i].expected,
            ad102_permissions[i].required, valid, identity, chip);
    if (!valid) ret = ENODEV;
  }
  if (ret) return TU_FAIL(w, ret, "permission sampling failed at %s", phase);
  return 0;
}

static int ad102_readout(tu_work_t *w, const char *phase, int require_full, int slot) {
  uint32_t value = 0;
  int ret = tu_rd(w, GU_FEAT_READOUT_1_OFFSET, &value);
  int valid = !ret && !tu_inaccessible(value);
  if (slot >= 0 && slot < TU_READOUT_PHASES) {
    w->r->readout[slot] = value;
    w->r->readout_valid[slot] = valid;
  }
  fprintf(stderr, "tensor-unlock: %s: readout1=%#x fields=%u %u %u %u %u %u %u %u %u valid=%d full=%d\n", phase, value,
          value & 7, (value >> 3) & 7, (value >> 6) & 7, (value >> 9) & 7, (value >> 12) & 7, (value >> 15) & 7,
          (value >> 18) & 7, (value >> 21) & 7, (value >> 24) & 7, valid, valid && !(value & GU_SPEED_READOUT_MASK));
  if (!valid) return TU_FAIL(w, ENODEV, "effective readout inaccessible at %s", phase);
  if (require_full && (value & GU_SPEED_READOUT_MASK))
    return TU_FAIL(w, EIO, "effective readout not full-rate at %s: %#x", phase, value);
  return 0;
}

// ---- stages ----------------------------------------------------------------------------------------------------------

static int tu_claim(tu_work_t *w) {
  uint32_t identity = tu_cfg(w, 0, 4);
  w->r->identity = identity;
  if (identity != TU_PCI_ID)
    return TU_FAIL(w, ENODEV, "device %04x:%04x is not the desktop RTX 4090 (10de:2684); unlock refused", identity & 0xffff,
                   identity >> 16);
  w->pm_cap = tu_find_cap(w, 0x01);
  if (w->pm_cap) {
    uint32_t pmcsr = tu_cfg(w, w->pm_cap + 4, 2);
    if (pmcsr == 0xffff || (pmcsr & 3) != 0) return TU_FAIL(w, EOPNOTSUPP, "device is not in D0 (pmcsr=%#x)", pmcsr);
  }
  w->pcie_cap = tu_find_cap(w, 0x10);
  if (!w->pcie_cap) return TU_FAIL(w, EOPNOTSUPP, "no PCI Express capability; FLR cannot be verified");
  uint32_t devcap = tu_cfg(w, w->pcie_cap + 0x04, 4);
  if (devcap == ~0u || !(devcap & (1u << 28))) return TU_FAIL(w, EOPNOTSUPP, "FLR not supported (devcap=%#x)", devcap);

  uint32_t command = tu_cfg(w, PCI_COMMAND, 2);
  if (command == 0xffff) return TU_FAIL(w, ENODEV, "PCI command register inaccessible");
  w->original_cmd = (uint16_t)command;
  w->cmd_valid = 1;
  // Some board firmware leaves BME set before any client connects (the dext sets it at provider start). Clear it;
  // the WPR2 cold gate below still rejects a previously initialized card.
  if (command & PCI_COMMAND_MASTER) {
    tu_cfg_write(w, PCI_COMMAND, 2, command & ~PCI_COMMAND_MASTER);
    command = tu_cfg(w, PCI_COMMAND, 2);
    if (command == 0xffff || (command & PCI_COMMAND_MASTER)) return TU_FAIL(w, EBUSY, "could not clear firmware BME");
    w->original_cmd = (uint16_t)command;
    fprintf(stderr, "tensor-unlock: claim: cleared firmware-set bus master\n");
  }
  if (!(command & PCI_COMMAND_MEMORY)) {
    tu_cfg_write(w, PCI_COMMAND, 2, command | PCI_COMMAND_MEMORY);
    command = tu_cfg(w, PCI_COMMAND, 2);
    if (!(command & PCI_COMMAND_MEMORY)) return TU_FAIL(w, EIO, "could not enable the memory window");
  }
  for (int i = 0; i < 6; i++) {
    w->bars[i] = tu_cfg(w, 0x10 + 4 * i, 4);
    if (w->bars[i] == ~0u) return TU_FAIL(w, EIO, "BAR%d inaccessible at claim", i);
  }
  uint64_t base = 0, size = 0;
  if (w->pci->bar_info(w->pci, 0, &base, &size)) return TU_FAIL(w, EIO, "BAR0 size unknown: %s", tinynv_last_error());
  w->bar_len = size;
  if (size < SEC2_FALCON2 + 0x1000 || size < 0x300000 + 0x1a)
    return TU_FAIL(w, EINVAL, "BAR0 too small for the transaction: %#llx", (unsigned long long)size);
  if (w->pci->bar_map(w->pci, 0, 0, 0, &w->bar)) return TU_FAIL(w, EIO, "BAR0 window failed: %s", tinynv_last_error());
  w->mapped = 1;

  uint32_t chip = 0;
  if (tu_rd(w, 0, &chip) || !chip || tu_inaccessible(chip))
    return TU_FAIL(w, ENODEV, "PMC_BOOT_0 reads %#x: the card is not answering", chip);
  w->r->chip = chip;
  if (((chip >> 20) & 0x1ff) != TU_AD102_ARCH)
    return TU_FAIL(w, ENODEV, "chip arch %#x is not AD102 (0x192); unlock refused", (chip >> 20) & 0x1ff);
  fprintf(stderr, "tensor-unlock: claim: identity=%#x chip=%#x command=%#x flr_cap=%#x bar0=%#llx bytes\n", identity, chip,
          command, devcap, (unsigned long long)size);
  return 0;
}

static int tu_layout(tu_work_t *w) {
  uint32_t mb = 0, vga = 0, lo = 0, hi = 0, plm = 0;
  if (tu_rd(w, 0x001183a4, &mb) || !mb || mb == ~0u || tu_rd(w, 0x00625f04, &vga) || vga == ~0u ||
      tu_rd(w, 0x001fa7c8, &plm) || plm == ~0u || !(plm & 1) || tu_rd(w, 0x001fa82c, &lo) || lo == ~0u ||
      tu_rd(w, 0x001fa830, &hi) || hi == ~0u)
    return TU_FAIL(w, ENODEV, "cold layout inputs inaccessible (fb=%#x vga=%#x plm=%#x lo=%#x hi=%#x)", mb, vga, plm, lo, hi);
  fprintf(stderr, "tensor-unlock: layout inputs: fb_mb=%#x vga=%#x mmu_plm=%#x mmu_lo=%#x mmu_hi=%#x image=%llu boot=%u\n",
          mb, vga, plm, lo, hi, (unsigned long long)w->assets->image_span.size, w->assets->boot_desc.image_size);
  struct gpu_unlock_layout_in *in = &w->layout_in;
  in->usable_fb = (uint64_t)mb << 20;
  if (in->usable_fb < (256ull << 20)) return TU_FAIL(w, ERANGE, "usable FB below 256 MiB");
  in->vga_base = in->usable_fb - (1ull << 20); // kgspPopulateWprMeta_TU102: PRAMIN top-1MiB fallback
  if (vga & (1u << 3)) {
    in->vga_base = (uint64_t)(vga >> 8) << 16;
    if (in->vga_base < in->usable_fb - (1ull << 20)) in->vga_base = in->usable_fb - 0x20000;
  }
  if (in->vga_base >= in->usable_fb) return TU_FAIL(w, ERANGE, "VGA base at or above usable FB");
  in->mmu_lock_low = (uint64_t)(lo >> 4) << 12;
  uint64_t mmu_hi = (uint64_t)(hi >> 4) << 12;
  uint64_t mmu_top = mmu_hi > ~0ull - 0xffffull ? ~0ull : (mmu_hi + 0xffffull) & ~0xffffull;
  in->mmu_lock_valid = mmu_hi >= in->mmu_lock_low;
  int top_lock = in->mmu_lock_valid && (mmu_top >> 20) == (in->usable_fb >> 20);
  if (!top_lock) in->mmu_lock_valid = 0; // a lock elsewhere is not the VBIOS end-of-FB reservation
  in->frts_size = 1ull << 20;
  in->boot_size = w->assets->boot_desc.image_size;
  in->image_size = w->assets->image_span.size;
  in->nonwpr_size = 1ull << 20;
  in->heap_request = 88ull << 20; // GA102's supported minimum heap; no GSP-RM clients start here
  if (gpu_unlock_layout(in, &w->layout)) return TU_FAIL(w, ERANGE, "reservation calculation rejected");
  fprintf(stderr, "tensor-unlock: layout ranges: nonwpr=%#llx wpr=%#llx heap=%#llx gsp=%#llx boot=%#llx frts=%#llx end=%#llx\n",
          (unsigned long long)w->layout.nonwpr_start, (unsigned long long)w->layout.wpr_start,
          (unsigned long long)w->layout.heap_offset, (unsigned long long)w->layout.gsp_offset,
          (unsigned long long)w->layout.boot_offset, (unsigned long long)w->layout.frts_offset,
          (unsigned long long)w->layout.wpr_end);
  // GA102 has no scrubber: all reservations must fit the pre-scrubbed top 256 MiB.
  if (w->layout.nonwpr_start < in->usable_fb - (256ull << 20))
    return TU_FAIL(w, ERANGE, "reservations below the pre-scrubbed top 256 MiB");
  return 0;
}

static int tu_patch_booter(tu_work_t *w, tinynv_dma_t *booter) {
  const struct gpu_unlock_booter_hdr *h = &w->assets->hdr;
  if (!h->ucode_id || h->ucode_id > 16) return TU_FAIL(w, EINVAL, "booter ucode id %u is outside 1..16", h->ucode_id);
  uint32_t fuse_raw, index = h->ucode_id - 1;
  if (tu_rd(w, GU_FUSE_SEC2_UCODE1 + index * 4, &fuse_raw)) return TU_FAIL(w, EIO, "SEC2 fuse register inaccessible");
  uint32_t fuse_ver = gpu_unlock_fuse_version(fuse_raw);
  fprintf(stderr, "tensor-unlock: booter signature: fuse_raw=%#x version=%u signatures=%u size=%u\n", fuse_raw, fuse_ver,
          h->num_sigs, h->sig_bytes);
  if (gpu_unlock_booter_patch(booter->va, w->assets->booter_image_size, h,
                              w->assets->booter.data + 72 + w->assets->booter_image_size, h->sig_bytes, fuse_ver))
    return TU_FAIL(w, EINVAL, "booter signature patch rejected (fuse version %u)", fuse_ver);
  return 0;
}

// Buffer allocation order is fixed and relied on by falcon_hs's payload-integrity check: 0 booter, 1 v67, 2 meta,
// 3 stock, 4 boot, 5 radix+image, 6 fwsec (allocated later, in the FWSEC stage).
static int tu_dma(tu_work_t *w) {
  tinynv_dma_t *booter, *v67, *meta, *stock, *boot, *radix;
  uint64_t iova;
  if (tu_dma_alloc(w, "booter", w->assets->booter_image_size, &iova, &booter)) return w->error;
  memcpy(booter->va, w->assets->booter.data + 72, w->assets->booter_image_size);
  if (tu_dma_alloc(w, "v67", GU_V67_SIZE, &iova, &v67)) return w->error;
  memcpy(v67->va, w->assets->payload.data, GU_V67_SIZE);
  if (tu_dma_alloc(w, "wpr-meta", GU_WPR_META_SIZE, &iova, &meta)) return w->error;
  if (tu_dma_alloc(w, "stock-signature", w->assets->sig_span.size, &iova, &stock)) return w->error;
  memcpy(stock->va, w->assets->gsp.data + w->assets->sig_span.offset, w->assets->sig_span.size);
  w->stock_size = w->assets->sig_span.size;
  if (tu_dma_alloc(w, "gsp-boot", w->assets->boot_desc.image_size, &iova, &boot)) return w->error;
  memcpy(boot->va, w->assets->boot.data, w->assets->boot_desc.image_size);
  int ret = tu_patch_booter(w, booter);
  if (ret) return ret;

  // Radix tables + GSP image pages in ONE contiguous validated mapping (the server allows 128 mappings/session;
  // one mapping per 4 KiB page would need ~20600). Sub-allocations stay 4 KiB-aligned (audit §3 rule 3).
  const struct gpu_unlock_radix_plan *r = &w->assets->radix;
  uint64_t page_count = r->table_pages + r->pages[3];
  if (tu_dma_alloc(w, "radix+image", (size_t)(page_count * GU_RADIX_PAGE), &iova, &radix)) return w->error;
  memset(radix->va, 0, (size_t)(page_count * GU_RADIX_PAGE));
  for (uint64_t i = r->table_pages; i < page_count; i++) {
    uint64_t offset = (i - r->table_pages) * GU_RADIX_PAGE;
    size_t n = (size_t)(w->assets->image_span.size - offset < GU_RADIX_PAGE ? w->assets->image_span.size - offset : GU_RADIX_PAGE);
    memcpy((uint8_t *)radix->va + i * GU_RADIX_PAGE, w->assets->gsp.data + w->assets->image_span.offset + offset, n);
  }
  uint64_t first = 0;
  for (uint32_t level = 0; level < 3; level++) {
    uint64_t child = first + r->pages[level];
    for (uint64_t i = 0; i < r->pages[level]; i++) {
      uint64_t at = i * (GU_RADIX_PAGE / 8);
      uint64_t n = r->pages[level + 1] - at < GU_RADIX_PAGE / 8 ? r->pages[level + 1] - at : GU_RADIX_PAGE / 8;
      // children are IOVAs into this same validated buffer, never CPU pointers
      uint64_t children[GU_RADIX_PAGE / 8];
      for (uint64_t j = 0; j < n; j++) children[j] = radix->pages[child + at + j];
      if (gpu_unlock_radix_page((uint8_t *)radix->va + (first + i) * GU_RADIX_PAGE, children, (size_t)n))
        return TU_FAIL(w, EINVAL, "radix page build rejected at level %u", level);
    }
    first = child;
  }
  return 0;
}

static int tu_gfw(tu_work_t *w) {
  double deadline = tinynv_now_s() + GU_GFW_TIMEOUT_US * tu_time_scale / 1e6;
  for (;;) {
    if (tensor_unlock_abort) return TU_FAIL(w, EINTR, "abort requested while waiting for ROM boot");
    struct gpu_unlock_gfw_sample s = {0};
    s.config_ok = tu_cfg(w, 0, 4) == TU_PCI_ID;
    if (tu_rd(w, GU_GFW_PLM_OFFSET, &s.plm) || tu_rd(w, GU_GFW_BOOT_OFFSET, &s.boot) || tu_rd(w, GU_GFW_CPUCTL_OFFSET, &s.cpuctl))
      return TU_FAIL(w, EIO, "GFW registers inaccessible");
    int assessed = gpu_unlock_gfw_assess(&s);
    if (!assessed) break;
    if (assessed != -EAGAIN) return TU_FAIL(w, -assessed, "ROM boot state rejected (plm=%#x boot=%#x cpuctl=%#x)", s.plm, s.boot, s.cpuctl);
    if (tinynv_now_s() >= deadline) return TU_FAIL(w, ETIMEDOUT, "ROM boot did not complete in 2.05 s");
    tu_sleep_us(1000);
  }
  uint32_t ss0, ss1, wpr_hi;
  if (tu_rd(w, GU_FEAT_PLM_OFFSET, &w->plm_entry) || tu_rd(w, GU_FEAT_SS0_OFFSET, &ss0) || ss0 == ~0u ||
      tu_rd(w, GU_FEAT_SS1_OFFSET, &ss1) || ss1 == ~0u || tu_rd(w, 0x001fa828, &wpr_hi) || tu_inaccessible(wpr_hi))
    return TU_FAIL(w, ENODEV, "cold-entry feature registers inaccessible");
  w->r->feat_entry = w->plm_entry;
  w->r->ss0_entry = ss0;
  w->r->ss1_entry = ss1;
  fprintf(stderr, "tensor-unlock: entry: plm=%#x ss0=%#x ss1=%#x wpr2_hi=%#x\n", w->plm_entry, ss0, ss1, wpr_hi);
  if (w->plm_entry == ~0u) {
    // Retained override. Spec req 3: skipping reapplication requires effective readout AND a usable cold handoff
    // state, not just an open PLM (upstream returns -EALREADY here without those checks; inputs F2/F4).
    if (wpr_hi >> 4) return TU_FAIL(w, EBUSY, "override retained but WPR2 is up: not a cold handoff; power off first");
    if (ss0 != GU_SS0_UNLOCK || ss1 != GU_SS1_UNLOCK)
      return TU_FAIL(w, EALREADY, "FEAT open but selectors are %#x/%#x: ambiguous retained state", ss0, ss1);
    int ret = ad102_readout(w, "verified-existing", 1, 0);
    if (ret) return ret;
    w->r->verified_existing = 1;
    w->r->feat_final = w->plm_entry;
    w->r->ss0_final = ss0;
    w->r->ss1_final = ss1;
    fprintf(stderr, "tensor-unlock: verified-existing: retained override with full effective readout and cold handoff\n");
    return 0;
  }
  // WPR2 is down from power-on until FWSEC-FRTS runs. An enabled WPR2 means a driver already used the card this
  // power cycle; starting FWSEC and Booter again from here is untested.
  if (wpr_hi >> 4) return TU_FAIL(w, EBUSY, "card already initialised since power on (WPR2 enabled); power off first");
  int ret = ad102_sample_permissions(w, "before-firmware", w->permission_before, 1);
  if (ret) return ret;
  memcpy(w->r->perm_before, w->permission_before, sizeof(w->permission_before));
  w->permission_saved = 1;
  return ad102_readout(w, "before-firmware", 0, 0);
}

static int tu_fwsec(tu_work_t *w) {
  if (w->bar_len < 0x300000 + 0x1a) return TU_FAIL(w, ENODEV, "BAR0 too small for the PROM window");
  size_t mapped = (size_t)(w->bar_len - 0x300000 < (1024 * 1024) ? w->bar_len - 0x300000 : (1024 * 1024));
  uint8_t *rom = malloc(mapped);
  if (!rom) return TU_FAIL(w, ENOMEM, "out of memory for the PROM copy");
  fprintf(stderr, "tensor-unlock: FWSEC: reading %zu PROM bytes at 0x300000\n", mapped);
  nv_rd_block(&w->bar, 0x300000, rom, mapped);
  struct gpu_unlock_fwsec fw;
  int ret = 0;
  if (gpu_unlock_vbios_fwsec(rom, mapped, &fw)) {
    ret = TU_FAIL(w, ENOENT, "FWSEC-FRTS not found in the board ROM");
    goto out;
  }
  // The parser validates this too, but keep the copy's device-read source bounded at the point of use.
  if (!tu_in_range(mapped, fw.image_off, fw.image_size)) {
    ret = TU_FAIL(w, EINVAL, "FWSEC image outside the mapped PROM (rom=%#x+%#x, mapped=%#zx)",
                  fw.image_off, fw.image_size, mapped);
    goto out;
  }
  if (fw.engine_mask != 0x400) {
    ret = TU_FAIL(w, EOPNOTSUPP, "FWSEC: unsupported GSP engine mask %#x", fw.engine_mask);
    goto out;
  }
  tinynv_dma_t *image;
  uint64_t image_dma;
  if (tu_dma_alloc(w, "fwsec", fw.image_size, &image_dma, &image)) { ret = w->error; goto out; }
  memcpy(image->va, rom + fw.image_off, fw.image_size);
  uint64_t frts = w->layout.frts_offset;
  if (gpu_unlock_fwsec_patch_frts(image->va, &fw, frts)) {
    ret = TU_FAIL(w, EINVAL, "FWSEC FRTS patch rejected");
    goto out;
  }
  uint32_t fuse_raw;
  if (!fw.ucode_id || tu_rd(w, 0x008241c0 + (fw.ucode_id - 1) * 4, &fuse_raw)) {
    ret = TU_FAIL(w, EIO, "FWSEC fuse register inaccessible");
    goto out;
  }
  uint32_t version = gpu_unlock_fuse_version(fuse_raw), sig_at = 0;
  fprintf(stderr, "tensor-unlock: FWSEC fuse: raw=%#x version=%u\n", fuse_raw, version);
  if (gpu_unlock_fwsec_sig_offset(fw.sig_versions, version, 384, &sig_at)) {
    ret = TU_FAIL(w, ERANGE, "FWSEC signature version %u not present (versions=%#x)", version, fw.sig_versions);
    goto out;
  }
  uint64_t image_copy = (uint64_t)fw.data_off + fw.pkc_off;
  uint64_t signature_copy = (uint64_t)fw.sig_off + sig_at;
  if (!tu_in_range(mapped, signature_copy, 384) || !tu_in_range(fw.image_size, image_copy, 384)) {
    ret = TU_FAIL(w, EINVAL, "FWSEC signature copy outside its validated image/ROM ranges "
                             "(rom=%#x+%#x, signature=%#llx+384, image=%#x+%#llx+384)",
                  fw.image_off, fw.image_size, (unsigned long long)signature_copy, fw.image_size,
                  (unsigned long long)image_copy);
    goto out;
  }
  memcpy((uint8_t *)image->va + fw.data_off + fw.pkc_off, rom + fw.sig_off + sig_at, 384);

  // Bus master ON, verified, BEFORE the engine may DMA. The dext already masters at provider start; this makes the
  // state explicit and checked rather than assumed.
  uint32_t command = tu_cfg(w, PCI_COMMAND, 2);
  tu_cfg_write(w, PCI_COMMAND, 2, command | PCI_COMMAND_MASTER);
  command = tu_cfg(w, PCI_COMMAND, 2);
  if (command == 0xffff || !(command & PCI_COMMAND_MASTER)) {
    ret = TU_FAIL(w, EIO, "bus master did not take before FWSEC launch");
    goto out;
  }
  w->master = 1;
  // The rounded tail of the last DMA block must stay inside the allocated image.
  if (falcon_dma_span(fw.imem_size) > fw.image_size || (uint64_t)fw.data_off + falcon_dma_span(fw.dmem_size) > fw.image_size) {
    ret = TU_FAIL(w, EINVAL, "FWSEC DMA tail outside image: imem=%#x data=%#x dmem=%#x image=%#x", fw.imem_size, fw.data_off,
                  fw.dmem_size, fw.image_size);
    goto out;
  }
  ret = falcon_hs(w, &gu_gsp, 0, fw.imem_size, fw.imem_va, fw.imem_pa, fw.data_off, fw.dmem_size, fw.dmem_pa, fw.pkc_off,
                  fw.engine_mask, fw.ucode_id, image_dma, 0, 0, 0);
  if (ret) goto out;
  uint32_t scratch, wpr_lo, wpr_hi;
  if (tu_rd(w, 0x00001400 + 0x0e * 4, &scratch)) { ret = TU_FAIL(w, EIO, "ACR scratch inaccessible"); goto out; }
  fprintf(stderr, "tensor-unlock: FWSEC result: scratch=%#x expected_hi16=0\n", scratch);
  if (scratch >> 16) { ret = TU_FAIL(w, EIO, "FWSEC-FRTS failed: ACR scratch=%#x", scratch); goto out; }
  if (tu_rd(w, 0x001fa824, &wpr_lo) || wpr_lo == ~0u || tu_rd(w, 0x001fa828, &wpr_hi) || wpr_hi == ~0u) {
    ret = TU_FAIL(w, EIO, "WPR2 inaccessible after FWSEC");
    goto out;
  }
  fprintf(stderr, "tensor-unlock: FWSEC WPR2: low=%#x high=%#x expected_page=%#llx end=%#llx\n", wpr_lo, wpr_hi,
          (unsigned long long)(frts >> 12), (unsigned long long)w->layout.wpr_end);
  // NVIDIA's initialized-WPR2 check plus the expected low address. Never publish Booter metadata for alien WPR2.
  if (!(wpr_hi >> 4) || (wpr_hi >> 4) < (wpr_lo >> 4) || ((uint64_t)(wpr_hi >> 4) << 12) >= w->layout.wpr_end ||
      (wpr_lo >> 4) != (uint32_t)(frts >> 12)) {
    ret = TU_FAIL(w, EIO, "FWSEC WPR2 does not match the FRTS reservation");
    goto out;
  }
out:
  free(rom);
  return ret;
}

static void put_le64(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static int tu_publish_v67(tu_work_t *w) {
  struct gpu_unlock_wpr_args args = {
    .fb_size = w->layout_in.usable_fb,
    .vga_offset = w->layout_in.vga_base,
    .frts_size = w->layout_in.frts_size,
    .signature_dma = w->buf[1].iova, .signature_size = GU_V67_SIZE,
    .radix_dma = w->buf[5].iova,
    .image_size = w->assets->image_span.size,
    .boot_dma = w->buf[4].iova, .boot = w->assets->boot_desc,
  };
  if (gpu_unlock_wpr_meta_pack(w->buf[2].dma.va, &w->layout, &args))
    return TU_FAIL(w, EINVAL, "WPR meta packing rejected");
  atomic_thread_fence(memory_order_release); // stores visible before any doorbell can reach the device
  w->published = 1;
  return 0;
}

static int tu_signature(tu_work_t *w) {
  uint8_t *meta = w->buf[2].dma.va;
  if (!meta || !w->buf[3].live || !w->buf[5].live || w->signature_calls >= 2)
    return TU_FAIL(w, EINVAL, "signature stage preconditions not met");
  if (w->signature_calls == 0) {
    int ret = tu_publish_v67(w);
    if (ret) return ret;
  } else {
    if (w->sec2_started && !w->sec2_halt_verified)
      return TU_FAIL(w, EBUSY, "stock signature restore deferred: SEC2 HALT unverified; retaining live metadata");
    put_le64(meta + 72, w->buf[3].iova);
    put_le64(meta + 80, w->stock_size);
    atomic_thread_fence(memory_order_release);
  }
  w->signature_calls++;
  return 0;
}

static int tu_boot(tu_work_t *w) {
  const struct gpu_unlock_booter_hdr *h = &w->assets->hdr;
  if (!w->published || !w->master || !w->firmware_started) return TU_FAIL(w, EINVAL, "SEC2 launch preconditions not met");
  if (!w->permission_saved) return TU_FAIL(w, EINVAL, "no before-firmware permission snapshot");
  int ret = falcon_hs(w, &gu_sec2, h->app_code_off, h->app_code_size, h->app_code_off, 0, h->os_data_off, h->os_data_size, 0,
                      h->patch_loc - h->os_data_off, h->engine_id, h->ucode_id, w->buf[0].iova,
                      (uint32_t)w->buf[2].iova, (uint32_t)(w->buf[2].iova >> 32), 1);
  // The marker precedes the cleanup tail. Wait for its exit before changing firmware-visible metadata or applying
  // selectors; a live SEC2 retains DMA.
  int halt_ret = ret ? ret : tu_poll(w, SEC2_BASE + 0x100, 0x10, 0x10, 5000000, 1);
  uint32_t marker = 0, cpuctl = 0, debug = 0, ss0 = 0, ss1 = 0;
  int state_ret = tu_rd(w, SEC2_BASE + 0x40, &marker);
  state_ret |= tu_rd(w, SEC2_BASE + 0x100, &cpuctl);
  state_ret |= tu_rd(w, SEC2_BASE + 0x94, &debug);
  w->sec2_halt_verified = !state_ret && !tu_inaccessible(cpuctl) && !!(cpuctl & 0x10);
  w->permission_sampled = 1;
  uint32_t after[4];
  int sample_ret = ad102_sample_permissions(w, "after-payload", after, 0);
  memcpy(w->r->perm_after, after, sizeof(after));
  int required = 1;
  for (int i = 0; i < 4; i++)
    if (ad102_permissions[i].required && after[i] != ad102_permissions[i].expected) required = 0;
  // Independently readable feature words reject a disconnected or protected BAR view even when every sampled PLM
  // happens to be all ones.
  state_ret |= tu_rd(w, GU_FEAT_SS0_OFFSET, &ss0);
  state_ret |= tu_rd(w, GU_FEAT_SS1_OFFSET, &ss1);
  if (tu_inaccessible(ss0) || tu_inaccessible(ss1)) state_ret = ENODEV;
  if (ad102_readout(w, "after-payload", 0, 1)) state_ret = ENODEV;
  fprintf(stderr, "tensor-unlock: payload result: boot=%d halt=%d marker=%#x expected=%#x marker_ok=%d required=%d "
                  "cpuctl=%#x halted=%d debug=%#x ss0=%#x ss1=%#x\n", ret, halt_ret, marker, GU_AD102_MARKER,
          marker == GU_AD102_MARKER, required, cpuctl, w->sec2_halt_verified, debug, ss0, ss1);
  if (ret || halt_ret || state_ret || sample_ret || !required || marker != GU_AD102_MARKER || !w->sec2_halt_verified)
    return TU_FAIL(w, ret ? ret : halt_ret ? halt_ret : state_ret ? ENODEV : sample_ret ? sample_ret : EIO,
                   "SEC2 payload stage gates failed (marker=%#x halted=%d required=%d)", marker, w->sec2_halt_verified, required);
  w->payload_verified = 1;
  return 0;
}

static int tu_write_selector(tu_work_t *w, const char *name, uint32_t offset, uint32_t unlock) {
  uint32_t value;
  if (tu_rd(w, offset, &value) || tu_inaccessible(value)) return TU_FAIL(w, ENODEV, "selector %s inaccessible", name);
  fprintf(stderr, "tensor-unlock: selector read: %s=%#x expected_unlock=%#x\n", name, value, unlock);
  if (value == unlock) return 0;
  tu_wr(w, offset, unlock);
  if (tu_rd(w, offset, &value) || tu_inaccessible(value)) return TU_FAIL(w, ENODEV, "selector %s readback inaccessible", name);
  fprintf(stderr, "tensor-unlock: selector result: %s=%#x expected=%#x matched=%d\n", name, value, unlock, value == unlock);
  if (value != unlock) return TU_FAIL(w, EIO, "selector %s write did not take: %#x", name, value);
  return 0;
}

static int tu_selectors(tu_work_t *w) {
  if (!w->payload_verified) return TU_FAIL(w, EPROTO, "selectors requested before the payload was verified");
  int ret = tu_write_selector(w, "SS0", GU_FEAT_SS0_OFFSET, GU_SS0_UNLOCK);
  if (!ret) ret = tu_write_selector(w, "SS1", GU_FEAT_SS1_OFFSET, GU_SS1_UNLOCK);
  if (!ret) ret = ad102_readout(w, "selectors-applied", 1, 2);
  return ret;
}

// Reset SEC2 after FLR and leave it halted with reset released (ksec2ResetHw_TU102 ends RESET_FALSE; leaving
// RESET_TRUE makes later host reads return FECS_PRI_RESET and RM's scrub wait never finishes).
static int hold_sec2_reset(tu_work_t *w, const char *phase) {
  uint32_t engine, hwcfg;
  if (tu_rd(w, SEC2_BASE + 0x3c0, &engine) || tu_inaccessible(engine)) return TU_FAIL(w, ENODEV, "SEC2 engine register inaccessible");
  tu_wr(w, SEC2_BASE + 0x3c0, engine | 1);
  for (int j = 0; j < 10; j++) {
    if (tu_rd(w, SEC2_BASE + 0x3c0, &engine) || tu_inaccessible(engine))
      return TU_FAIL(w, EIO, "engine reset inaccessible at %s cycle %d", phase, j + 1);
  }
  int ret = tu_poll(w, SEC2_BASE + 0x3c0, 1, 1, 1000, 1);
  if (ret) return ret;
  if (tu_rd(w, SEC2_BASE + 0x3c0, &engine) || tu_inaccessible(engine)) return TU_FAIL(w, EIO, "SEC2 engine read failed");
  tu_wr(w, SEC2_BASE + 0x3c0, engine & ~1u);
  for (int j = 0; j < 10; j++) {
    if (tu_rd(w, SEC2_BASE + 0x3c0, &engine) || tu_inaccessible(engine))
      return TU_FAIL(w, EIO, "engine reset release inaccessible at cycle %d", j + 1);
  }
  ret = tu_poll(w, SEC2_BASE + 0x3c0, 1, 0, 1000, 1);
  if (ret) return ret;
  ret = tu_poll(w, SEC2_BASE + 0xf4, 1u << 12, 0, 1000000, 1); // HWCFG2 bit12 clears when memory scrubbing finishes
  if (ret) return ret;
  if (tu_rd(w, SEC2_BASE + 0xf4, &hwcfg) || tu_inaccessible(hwcfg)) return TU_FAIL(w, EIO, "SEC2 hwcfg2 inaccessible");
  fprintf(stderr, "tensor-unlock: engine reset released: phase=%s engine=%#x hwcfg2=%#x\n", phase, engine, hwcfg);
  return 0;
}

// PCI config can be ready before BAR0; transient badf/all-ones reads are expected only inside this bounded wait.
static int wait_post_flr_bar(tu_work_t *w, uint32_t chip) {
  double deadline = tinynv_now_s() + 1.0 * tu_time_scale;
  uint32_t value = 0;
  for (;;) {
    if (tensor_unlock_abort) return TU_FAIL(w, EINTR, "abort requested after FLR");
    if (tu_rd(w, 0, &value)) return TU_FAIL(w, EIO, "BAR0 read failed after FLR");
    if (value == chip) return 0;
    if (tinynv_now_s() >= deadline) return TU_FAIL(w, ETIMEDOUT, "post-FLR BAR timeout: chip=%#x expected=%#x", value, chip);
    tu_sleep_us(1000);
  }
}

static int wait_post_flr_gfw(tu_work_t *w) {
  double deadline = tinynv_now_s() + GU_GFW_TIMEOUT_US * tu_time_scale / 1e6;
  struct gpu_unlock_gfw_sample s = {0};
  int ret;
  for (;;) {
    if (tensor_unlock_abort) return TU_FAIL(w, EINTR, "abort requested after FLR");
    s = (struct gpu_unlock_gfw_sample){0};
    s.config_ok = tu_cfg(w, 0, 4) == TU_PCI_ID;
    if (!s.config_ok) { ret = ENODEV; break; }
    if (tu_rd(w, GU_GFW_CPUCTL_OFFSET, &s.cpuctl) || tu_rd(w, GU_GFW_PLM_OFFSET, &s.plm)) { ret = EIO; break; }
    if (!tu_inaccessible(s.plm) && (s.plm & GU_GFW_PLM_LEVEL0) && tu_rd(w, GU_GFW_BOOT_OFFSET, &s.boot)) { ret = EIO; break; }
    ret = tu_inaccessible(s.plm) || tu_inaccessible(s.boot) || tu_inaccessible(s.cpuctl) ? -EAGAIN : gpu_unlock_gfw_assess(&s);
    if (!ret) break;
    if (ret != -EAGAIN) break;
    if (tinynv_now_s() >= deadline) { ret = ETIMEDOUT; break; }
    tu_sleep_us(1000);
  }
  fprintf(stderr, "tensor-unlock: post-FLR GFW result: error=%d plm=%#x boot=%#x cpuctl=%#x\n", ret, s.plm, s.boot, s.cpuctl);
  if (ret == -EAGAIN) ret = ETIMEDOUT;
  return ret ? TU_FAIL(w, ret, "post-FLR ROM boot did not complete") : 0;
}

static int tu_reset(tu_work_t *w) {
  uint32_t ss0, ss1, chip, cpuctl, lo, hi, plm;
  if (w->signature_calls != 2) return TU_FAIL(w, EPROTO, "reset before both signature stages completed");
  if (tu_rd(w, GU_FEAT_SS0_OFFSET, &ss0) || ss0 == ~0u || tu_rd(w, GU_FEAT_SS1_OFFSET, &ss1) || ss1 == ~0u)
    return TU_FAIL(w, ENODEV, "pre-FLR selectors inaccessible");
  fprintf(stderr, "tensor-unlock: pre-FLR selectors: ss0=%#x ss1=%#x\n", ss0, ss1);
  if (ss0 != GU_SS0_UNLOCK || ss1 != GU_SS1_UNLOCK) return TU_FAIL(w, EIO, "pre-FLR selectors are not the unlock values");
  int ret = ad102_readout(w, "pre-FLR", 1, 3);
  if (ret) return ret;
  w->pci->quiesce(w->pci); // BME clear + readback (the write lands before anything depends on it)
  w->master = 0;
  uint32_t command = tu_cfg(w, PCI_COMMAND, 2);
  if (command == 0xffff || (command & PCI_COMMAND_MASTER)) return TU_FAIL(w, EIO, "bus master still set after quiesce");
  if (tu_rd(w, 0, &chip) || !chip || tu_inaccessible(chip)) return TU_FAIL(w, ENODEV, "chip id inaccessible before FLR");
  fprintf(stderr, "tensor-unlock: FLR begin: BME=0, DMA retained, chip=%#x\n", chip);
  // The RPC status is meaningless (the dext discards Reset's return value, audit §4): only verified effects count.
  // Apple's FunctionReset saves/restores config space itself and blocks until the link is up.
  w->pci->reset(w->pci);
  uint32_t identity = tu_cfg(w, 0, 4);
  if (identity != TU_PCI_ID) return TU_FAIL(w, ENODEV, "identity after FLR is %#x, not %#x", identity, TU_PCI_ID);
  // Verify and repair from the claim-time snapshot (dev.c pattern): a reset may clear BARs/COMMAND on this tunnel.
  for (int i = 0; i < 6; i++) {
    uint32_t bar = tu_cfg(w, 0x10 + 4 * i, 4);
    if (bar != w->bars[i]) {
      fprintf(stderr, "tensor-unlock: post-FLR BAR%d=%#x, restoring %#x\n", i, bar, w->bars[i]);
      tu_cfg_write(w, 0x10 + 4 * i, 4, w->bars[i]);
      if (tu_cfg(w, 0x10 + 4 * i, 4) != w->bars[i]) return TU_FAIL(w, EIO, "BAR%d did not restore after FLR", i);
    }
  }
  command = tu_cfg(w, PCI_COMMAND, 2);
  if (command == 0xffff) return TU_FAIL(w, ENODEV, "command register inaccessible after FLR");
  if ((command & PCI_COMMAND_MASTER) || !(command & PCI_COMMAND_MEMORY)) {
    tu_cfg_write(w, PCI_COMMAND, 2, (command & ~PCI_COMMAND_MASTER) | PCI_COMMAND_MEMORY);
    command = tu_cfg(w, PCI_COMMAND, 2);
    if ((command & PCI_COMMAND_MASTER) || !(command & PCI_COMMAND_MEMORY))
      return TU_FAIL(w, EIO, "command register did not restore after FLR: %#x", command);
  }
  ret = wait_post_flr_bar(w, chip);
  if (ret) return ret;
  ret = wait_post_flr_gfw(w);
  if (ret) return ret;
  ret = hold_sec2_reset(w, "post-FLR");
  if (ret) return ret;
  uint32_t pending = tu_cfg(w, w->pcie_cap + 0x0a, 2); // DEVSTA
  if (pending == 0xffff || (pending & 0x20))
    return TU_FAIL(w, EBUSY, "transactions still pending after FLR (devsta=%#x)", pending);
  if (tu_rd(w, GU_GFW_CPUCTL_OFFSET, &cpuctl) || tu_inaccessible(cpuctl) || !(cpuctl & GU_GFW_CPUCTL_HALTED) ||
      tu_rd(w, 0x001fa824, &lo) || tu_inaccessible(lo) || tu_rd(w, 0x001fa828, &hi) || tu_inaccessible(hi) ||
      tu_rd(w, GU_FEAT_SS0_OFFSET, &ss0) || tu_inaccessible(ss0) || tu_rd(w, GU_FEAT_SS1_OFFSET, &ss1) || tu_inaccessible(ss1) ||
      tu_rd(w, GU_FEAT_PLM_OFFSET, &plm) || (plm & 0xffff0000u) == 0xbadf0000u)
    return TU_FAIL(w, ENODEV, "post-FLR containment registers inaccessible");
  fprintf(stderr, "tensor-unlock: post-FLR: SEC2 reset released, GSP halted, BME=0 pending=0 WPR2=%08x:%08x SS0=%08x SS1=%08x\n",
          lo, hi, ss0, ss1);
  // Stock RM refuses a preexisting WPR2 (kgspIsWpr2Up_TU102 uses HI.VAL != 0).
  if (hi >> 4) return TU_FAIL(w, EBUSY, "post-FLR WPR2 remains enabled: lo=%#x hi=%#x", lo, hi);
  w->quiesced = 1; // containment proven: engines halted, region down, no pending transactions, BME off
  w->r->feat_final = plm;
  w->r->ss0_final = ss0;
  w->r->ss1_final = ss1;
  if (ss0 != GU_SS0_UNLOCK || ss1 != GU_SS1_UNLOCK)
    return TU_FAIL(w, EIO, "override lost across FLR: ss0=%#x ss1=%#x", ss0, ss1);
  ret = ad102_sample_permissions(w, "post-FLR", w->r->perm_after, 0);
  if (ret) return ret;
  if (plm != ~0u) return TU_FAIL(w, EIO, "FEAT PLM not retained across FLR: %#x", plm);
  return ad102_readout(w, "post-FLR", 1, 4);
}

// ---- cleanup / recovery ----------------------------------------------------------------------------------------------

static void tu_unwind(tu_work_t *w) {
  // A timeout never authorizes freeing firmware-visible memory. Keep the buffers, the window and (caller-owned)
  // socket alive; the probe holds the process until an enclosure power-off. Best-effort BME clear stops NEW DMA;
  // it is not proof that pending DMA drained, so nothing is released.
  if (w->firmware_started && !w->quiesced) {
    w->r->recovery_required = 1;
    if (w->master) {
      w->pci->quiesce(w->pci);
      w->master = 0;
    }
    fprintf(stderr, "tensor-unlock: firmware did not provably stop; KEEPING its DMA buffers, window and socket alive "
                    "until the enclosure is powered off (recovery state)\n");
    return;
  }
  if (w->master) {
    w->pci->quiesce(w->pci);
    w->master = 0;
  }
  if (w->mapped) {
    w->pci->bar_unmap(w->pci, &w->bar);
    w->mapped = 0;
  }
  tu_dma_free_all(w);
  if (!w->cmd_valid) return;
  uint32_t now = tu_cfg(w, PCI_COMMAND, 2);
  if (now == 0xffff) {
    fprintf(stderr, "tensor-unlock: command register inaccessible at unwind; nothing to restore\n");
    return;
  }
  uint32_t mask = PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER;
  uint32_t want = (now & ~mask) | (w->original_cmd & mask);
  if (want != now) tu_cfg_write(w, PCI_COMMAND, 2, want);
  now = tu_cfg(w, PCI_COMMAND, 2);
  fprintf(stderr, "tensor-unlock: restore PCI command: actual=%#x original=%#x\n", now, w->original_cmd);
  if ((now & mask) != (w->original_cmd & mask))
    fprintf(stderr, "tensor-unlock: WARNING: command restore did not take (actual=%#x)\n", now);
}

// ---- entry points -----------------------------------------------------------------------------------------------------

static int tu_run(tu_work_t *w) {
  // GFW/cold inspection runs BEFORE layout and DMA so a retained override (verified-existing) or a warm card is
  // refused without allocating any transaction memory. Upstream inspects after allocation; nothing in the GFW gate
  // depends on the layout, and the spec's retained path must not allocate.
  static const int order[] = {TU_STAGE_CLAIM, TU_STAGE_GFW, TU_STAGE_LAYOUT, TU_STAGE_DMA, TU_STAGE_FWSEC,
                              TU_STAGE_PUBLISH, TU_STAGE_PAYLOAD, TU_STAGE_RESTORE_SIG, TU_STAGE_SELECTORS, TU_STAGE_RESET};
  int (*const fns[])(tu_work_t *) = {tu_claim, tu_gfw, tu_layout, tu_dma, tu_fwsec, tu_signature, tu_boot,
                                     tu_signature, tu_selectors, tu_reset};
  int ret = 0;
  for (size_t i = 0; i < sizeof(order) / sizeof(order[0]); i++) {
    w->stage = order[i];
    if (tensor_unlock_abort) { ret = TU_FAIL(w, EINTR, "abort requested before stage %s", tensor_unlock_stage_name(w->stage)); break; }
    ret = fns[i](w);
    if (ret) break;
    w->last_ok = order[i];
    if (w->r->verified_existing) { // retained override verified at the GFW gate: no transaction needed
      w->last_ok = TU_STAGE_DONE;
      break;
    }
  }
  // Restore the stock signature when SEC2 no longer consumes the metadata. An unverified live engine retains the
  // V67 page and all DMA buffers (upstream failure-unwind rule).
  if (ret && w->signature_calls == 1) {
    int save_stage = w->stage;
    w->stage = TU_STAGE_RESTORE_SIG;
    tu_signature(w);
    w->stage = save_stage;
  }
  if (w->permission_saved && !w->permission_sampled && w->mapped)
    ad102_sample_permissions(w, "abort", w->r->perm_after, 0);
  tu_unwind(w);
  return ret;
}

int tensor_unlock_run_with(tinynv_pci_t *pci, struct tu_assets *assets, tensor_unlock_result_t *out) {
  memset(out, 0, sizeof(*out));
  out->classification = "transaction";
  tensor_unlock_abort = 0;
  tu_got_term = 0;
  tu_holding = 0;
  double t0 = tinynv_now_s();
  tu_work_t w = {0};
  w.pci = pci;
  w.assets = assets;
  w.r = out;
  w.stage = TU_STAGE_ASSETS;
  w.last_ok = TU_STAGE_ASSETS; // the caller loaded them; run_with receives them parsed
  int ret = tu_run(&w);
  out->error = ret;
  out->failed_at = ret ? w.stage : TU_STAGE_NONE;
  out->last_ok = w.last_ok;
  out->elapsed_s = tinynv_now_s() - t0;
  if (!ret) out->last_ok = TU_STAGE_DONE;
  if (!ret)
    fprintf(stderr, "tensor-unlock: %s in %.2f s (chip=%#x)\n", out->verified_existing ? "verified-existing" : "applied-and-verified",
            out->elapsed_s, out->chip);
  return ret ? -1 : 0;
}

int tensor_unlock_run(tinynv_pci_t *pci, const char *bundle_dir, tensor_unlock_result_t *out) {
  memset(out, 0, sizeof(*out));
  out->classification = "transaction";
  double t0 = tinynv_now_s();
  struct tu_assets *assets = NULL;
  if (tensor_unlock_assets_load(bundle_dir, &tensor_unlock_pins, &assets)) {
    out->error = EIO;
    out->failed_at = TU_STAGE_ASSETS;
    out->last_ok = TU_STAGE_NONE;
    snprintf(out->first_error, sizeof(out->first_error), "%s", tinynv_last_error());
    out->elapsed_s = tinynv_now_s() - t0;
    return -1;
  }
  int ret = tensor_unlock_run_with(pci, assets, out);
  out->elapsed_s = tinynv_now_s() - t0;
  tensor_unlock_assets_free(assets); // CPU-side copies only; device-visible buffers follow the containment rules
  return ret;
}

// Pure post-boot classification, shared by the live read in tinynv.c and the offline tests. The gates
// are the transaction's own: FEAT open + both selectors at the unlock values + effective readout all zero means
// unlocked; anything inaccessible is valid=0 and never evidence of success. FEAT == ~0 is the OPEN value, so the
// all-ones accessibility check applies to the chip/readout/selectors, and 0xbadf____ to every field.
void tensor_unlock_state_classify(tensor_unlock_state_t *s) {
  s->valid = s->chip != 0 && !tu_inaccessible(s->chip) &&
             !tu_inaccessible(s->readout) && !tu_inaccessible(s->ss0) && !tu_inaccessible(s->ss1) &&
             (s->feat & 0xffff0000u) != 0xbadf0000u;
  s->unlocked = s->valid && s->feat == ~0u && s->ss0 == GU_SS0_UNLOCK && s->ss1 == GU_SS1_UNLOCK &&
                !(s->readout & GU_SPEED_READOUT_MASK);
  s->stock = s->valid && s->feat != ~0u && !(s->ss0 == GU_SS0_UNLOCK && s->ss1 == GU_SS1_UNLOCK);
}

int tensor_unlock_preflight(tinynv_pci_t *pci, tensor_unlock_result_t *out) {
  memset(out, 0, sizeof(*out));
  out->preflight = 1;
  out->last_ok = TU_STAGE_NONE;
  double t0 = tinynv_now_s();
  uint32_t identity = pci->cfg_read(pci, 0, 4);
  out->identity = identity;
  if (identity == ~0u || identity == 0) {
    out->classification = "inaccessible";
    out->error = ENODEV;
    out->failed_at = TU_STAGE_CLAIM;
    snprintf(out->first_error, sizeof(out->first_error), "preflight identity is inaccessible (%#x)", identity);
    out->elapsed_s = tinynv_now_s() - t0;
    return -1;
  }
  if (identity != TU_PCI_ID) {
    out->classification = "wrong-device";
    out->error = ENODEV;
    out->failed_at = TU_STAGE_CLAIM;
    snprintf(out->first_error, sizeof(out->first_error), "preflight identity %#x is not RTX 4090 (%#x)",
             identity, TU_PCI_ID);
    out->elapsed_s = tinynv_now_s() - t0;
    return -1;
  }
  tu_work_t w = {0};
  w.pci = pci;
  w.r = out;
  w.stage = TU_STAGE_CLAIM;
  w.pcie_cap = tu_find_cap(&w, 0x10);
  uint64_t size = 0;
  if (pci->bar_info(pci, 0, NULL, &size) || pci->bar_map(pci, 0, 0, 0, &w.bar)) {
    out->classification = "inaccessible";
    out->error = EIO;
    out->failed_at = TU_STAGE_CLAIM;
    snprintf(out->first_error, sizeof(out->first_error), "preflight cannot map BAR0: %s", tinynv_last_error());
    out->elapsed_s = tinynv_now_s() - t0;
    return -1;
  }
  w.mapped = 1;
  w.bar_len = size;
  uint32_t chip = 0, plm = 0, boot = 0, cpuctl = 0, ss0 = 0, ss1 = 0, wpr_hi = 0, readout = 0;
  if (tu_rd(&w, 0, &chip) ||
      tu_rd(&w, GU_GFW_PLM_OFFSET, &plm) ||
      tu_rd(&w, GU_GFW_BOOT_OFFSET, &boot) ||
      tu_rd(&w, GU_GFW_CPUCTL_OFFSET, &cpuctl) ||
      tu_rd(&w, GU_FEAT_PLM_OFFSET, &plm) ||
      tu_rd(&w, GU_FEAT_SS0_OFFSET, &ss0) ||
      tu_rd(&w, GU_FEAT_SS1_OFFSET, &ss1) ||
      tu_rd(&w, GU_FEAT_READOUT_1_OFFSET, &readout) ||
      tu_rd(&w, 0x001fa828, &wpr_hi)) {
    out->classification = "inaccessible";
    out->error = out->error ? out->error : EIO;
    out->failed_at = TU_STAGE_CLAIM;
    pci->bar_unmap(pci, &w.bar);
    out->elapsed_s = tinynv_now_s() - t0;
    return -1;
  }
  out->chip = chip;
  out->feat_entry = plm;
  out->ss0_entry = ss0;
  out->ss1_entry = ss1;
  out->readout[0] = readout;
  out->readout_valid[0] = !tu_inaccessible(readout);
  uint32_t devcap = w.pcie_cap ? tu_cfg(&w, w.pcie_cap + 0x04, 4) : 0;
  fprintf(stderr, "tensor-unlock: preflight: identity=%#x chip=%#x feat=%#x readout=%#x ss0=%#x ss1=%#x wpr2_hi=%#x "
                  "gfw_cpuctl=%#x flr=%d\n", identity, chip, plm, readout, ss0, ss1, wpr_hi, cpuctl,
          !!(devcap & (1u << 28)));
  if (tu_inaccessible(chip) || !chip) {
    out->classification = "inaccessible";
    out->error = ENODEV;
    out->failed_at = TU_STAGE_CLAIM;
    snprintf(out->first_error, sizeof(out->first_error), "preflight chip identity is inaccessible (%#x)", chip);
  } else if (((chip >> 20) & 0x1ff) != TU_AD102_ARCH) {
    out->classification = "wrong-chip";
    out->error = ENODEV;
    out->failed_at = TU_STAGE_CLAIM;
    snprintf(out->first_error, sizeof(out->first_error), "preflight chip %#x is not AD102 (%#x)", chip, TU_AD102_ARCH);
  } else if (tu_inaccessible(wpr_hi)) {
    out->classification = "inaccessible";
    out->error = EIO;
    out->failed_at = TU_STAGE_CLAIM;
    snprintf(out->first_error, sizeof(out->first_error), "preflight WPR2 register is inaccessible (%#x)", wpr_hi);
  }
  else if ((wpr_hi >> 4)) out->classification = "warm (WPR2 up: a driver used the card this power cycle)";
  else if (plm == ~0u && ss0 == GU_SS0_UNLOCK && ss1 == GU_SS1_UNLOCK && !(readout & GU_SPEED_READOUT_MASK))
    out->classification = "retained-unlock (verified by readout; cold handoff)";
  else if (plm == ~0u) out->classification = "ambiguous retained override (selectors/readout disagree)";
  else out->classification = "stock-cold";
  pci->bar_unmap(pci, &w.bar);
  out->elapsed_s = tinynv_now_s() - t0;
  if (out->error) return -1;
  out->last_ok = TU_STAGE_CLAIM;
  return 0;
}

// ---- option parsing and status naming (runtime integration) ---------------------------------------------------

int tensor_unlock_option_parse(const char *value) {
  if (!value || !*value) return 0;
  if (!strcmp(value, "0")) return 0;
  if (!strcmp(value, "1")) return 1;
  return -1;
}

const char *tensor_unlock_status(const tensor_unlock_result_t *r) {
  if (r->preflight) {
    if (r->error == ENODEV && r->failed_at == TU_STAGE_CLAIM) return "unsupported";
    if (r->error) return "failed-preflight";
    return "preflight-ok";
  }
  if (!r->error) return r->verified_existing ? "verified-existing" : "applied-and-verified";
  // ENODEV at claim is the eligibility refusal: wrong PCI identity, wrong chip arch, or a card not answering.
  if (r->failed_at == TU_STAGE_CLAIM && r->error == ENODEV) return "unsupported";
  // Everything up to and including DMA completes before any firmware is launched (FWSEC is the first launch).
  if (r->failed_at >= TU_STAGE_NONE && r->failed_at <= TU_STAGE_DMA) return "failed-preflight";
  return "failed-after-start";
}

// ---- shared signal handling and the recovery hold ---------------------------------------------------------------------
// Shared with the standalone probe (tools/rtx4090-probe/probe.c) so the tool and runtime integration
// integration run the SAME tested code. test_probe_recovery.py drives the probe through every scenario below.

// SIGUSR1 is the owner's power-off confirmation and is ONLY honored inside the recovery hold. A SIGUSR1 that arrives
// earlier (stray, scripted, sent twice) must not pre-confirm a power-off that has not happened: that would disconnect
// under a possibly-live firmware the moment the hold is entered.

static void tu_on_term(int sig) {
  (void)sig;
  tensor_unlock_abort = 1;
  if (tu_got_term != 2) tu_got_term = 1;
}
static void tu_on_usr1(int sig) { (void)sig; if (tu_holding) tu_got_term = 2; }

void tensor_unlock_signals_install(tensor_unlock_sigstate_t *s) {
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = tu_on_term;
  sigaction(SIGINT, &sa, &s->prev[0]);
  sigaction(SIGTERM, &sa, &s->prev[1]);
  sigaction(SIGHUP, &sa, &s->prev[2]);
  sa.sa_handler = tu_on_usr1;
  sigaction(SIGUSR1, &sa, &s->prev[3]);
  s->installed = 1;
}

void tensor_unlock_signals_restore(tensor_unlock_sigstate_t *s) {
  if (!s->installed) return;
  sigaction(SIGINT, &s->prev[0], NULL);
  sigaction(SIGTERM, &s->prev[1], NULL);
  sigaction(SIGHUP, &s->prev[2], NULL);
  sigaction(SIGUSR1, &s->prev[3], NULL);
  s->installed = 0;
}

int tensor_unlock_recovery_hold(const tensor_unlock_result_t *r, const char *who) {
  fprintf(stderr,
          "\n============================== RECOVERY REQUIRED ==============================\n"
          "The unlock firmware did not provably stop. Its DMA buffers, BAR window and the\n"
          "TinyGPU socket are being kept ALIVE by this process (pid %ld). Do not kill it.\n"
          "First failure: %s\n"
          "Last verified stage: %s\n"
          "\n"
          "  1. POWER OFF THE ENCLOSURE now (physical switch, >= 30 s off).\n"
          "  2. Only then confirm here: type 'off' + Enter, or send SIGUSR1 to pid %ld.\n"
          "\n"
          "SIGINT/SIGTERM will NOT disconnect while containment is unproven. A crash or\n"
          "kill -9 of this process force-unmaps the buffers under a possibly-live firmware\n"
          "(known limitation O14); if that happens, power-cycle before any further use.\n"
          "=============================================================================\n",
          (long)getpid(), r->first_error, tensor_unlock_stage_name(r->last_ok), (long)getpid());
  fflush(stderr);
  sigset_t block, orig;
  sigemptyset(&block);
  sigaddset(&block, SIGINT);
  sigaddset(&block, SIGTERM);
  sigaddset(&block, SIGHUP);
  sigaddset(&block, SIGUSR1);
  // Block first, then publish/reset the hold state. pselect below atomically restores `orig` for the wait, so a
  // SIGUSR1 at any point after this state becomes visible is either observed by pselect or already recorded.
  if (sigprocmask(SIG_BLOCK, &block, &orig)) {
    fprintf(stderr, "%s: cannot block recovery signals: %s\n", who, strerror(errno));
    return -1;
  }
  tu_got_term = 0;
  tu_holding = 1;
  int fd = fileno(stdin);
  int stdin_open = fd >= 0 && fd < FD_SETSIZE;
  char line[64];
  size_t used = 0;
  int confirmed = 0;
  for (;;) {
    if (tu_got_term == 2) { confirmed = 1; break; } // SIGUSR1 during the hold: the owner's power-off confirmation
    if (tu_got_term == 1) {
      tu_got_term = 0;
      fprintf(stderr, "%s: termination refused while containment is unproven; power off the enclosure, "
                      "then type 'off' or send SIGUSR1\n", who);
      fflush(stderr);
    }
    if (stdin_open) {
      fd_set readfds;
      FD_ZERO(&readfds);
      FD_SET(fd, &readfds);
      int ready = pselect(fd + 1, &readfds, NULL, NULL, NULL, &orig);
      if (ready < 0 && errno == EINTR) continue;
      if (ready < 0) {
        fprintf(stderr, "%s: pselect failed while holding containment: %s; stdin confirmation disabled\n", who,
                strerror(errno));
        fflush(stderr);
        stdin_open = 0;
        continue;
      }
      if (ready > 0 && FD_ISSET(fd, &readfds)) {
        char buf[128];
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n > 0) {
          for (ssize_t i = 0; i < n; i++) {
            char ch = buf[i];
            if (ch == '\n') {
              line[used] = 0;
              while (used && (line[used - 1] == '\r' || line[used - 1] == ' ' || line[used - 1] == '\t'))
                line[--used] = 0;
              if (!strcasecmp(line, "off")) {
                confirmed = 1;
                break;
              }
              fprintf(stderr, "%s: still holding. Type 'off' ONLY after the enclosure is powered off.\n", who);
              fflush(stderr);
              used = 0;
            } else if (used + 1 < sizeof(line)) {
              line[used++] = ch;
            } else {
              used = 0; // a line longer than the accepted command cannot be "off"
            }
          }
          if (confirmed) break;
          continue;
        }
        if (n < 0 && errno == EINTR) continue;
        stdin_open = 0; // EOF or an unusable stdin: wait only for the signals below
      }
    }
    // stdin closed (non-interactive) or unusable: sigsuspend atomically restores the caller's mask and waits for a
    // signal. There is no check-then-pause race and no timer that could release mappings automatically.
    if (!stdin_open && !tu_got_term) sigsuspend(&orig);
  }
  tu_holding = 0;
  tu_got_term = 0;
  sigprocmask(SIG_SETMASK, &orig, NULL);
  fprintf(stderr, "%s: power-off confirmed by the owner; releasing the socket and mappings now\n", who);
  return 3;
}
