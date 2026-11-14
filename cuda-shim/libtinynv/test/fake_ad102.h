// The fake AD102 card and synthetic asset builders shared by the offline unlock suites.
//
// Extracted VERBATIM from test_tensor_unlock.c so the wire-protocol mock server (mock_ad102_server.c) drives the
// SAME register-level semantics the unit suite does. Two additions, both inert in unit mode (wire_mode = 0):
//   - wire_mode: the SEC2 payload-integrity gate compares against the published v67 DMA buffer (allocation slot 1)
//     instead of the synthetic payload, so the mock can serve the REAL pinned bundle through the probe's loader.
//     The adapter's own pre-start memcmp against the hash-verified asset is unchanged and still the real gate.
//   - K_STALL_SEC2: after SEC2 starts, the next read reply is delayed (server-side), so a test can deliver a signal
//     while the client is provably blocked in recv. Control flow only; no hardware behavior is modeled by the delay.
#ifndef TINYNV_TEST_FAKE_AD102_H
#define TINYNV_TEST_FAKE_AD102_H
#include "tensor_unlock.h"
#include "internal.h"
#include "unlock610/prep_610.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---- synthetic assets (built once; the loader hash-checks them through the same code the probe uses) ------------------

#define BOOTER_IMAGE 56832u
#define PAYLOAD_SIZE 64000u
#define GSP_IMAGE_SIZE 8192u
#define SIG_SIZE 4096u
#define BOOT_SIZE 36864u

static uint8_t synth_payload[PAYLOAD_SIZE];
static uint8_t synth_stock[SIG_SIZE];
static uint8_t synth_sigs[768];

static void wr32le(uint8_t *p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }
static void wr16le(uint8_t *p, uint16_t v) { p[0] = v; p[1] = v >> 8; }
static uint32_t rd32le(const uint8_t *p) { return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t rd64le(const uint8_t *p) { return rd32le(p) | (uint64_t)rd32le(p + 4) << 32; }

static uint8_t *build_booter(size_t *len, int bad_geometry) {
  *len = 72 + BOOTER_IMAGE + 768;
  uint8_t *b = calloc(1, *len);
  wr32le(b, 0x314c5442);
  wr32le(b + 4, 1);
  const uint32_t hdr[14] = {0, 256, 32256, 24576, 1, 256, 32000, 256, 0, 32272, 3, 1, 2, 768};
  for (int i = 0; i < 14; i++) wr32le(b + 8 + 4 * i, hdr[i]);
  if (bad_geometry) wr32le(b + 8 + 4 * 2, 32255); // os_data_off wrong by one
  wr32le(b + 64, BOOTER_IMAGE);
  wr32le(b + 68, 768);
  memset(b + 72, 0xa5, BOOTER_IMAGE);
  for (int i = 0; i < 384; i++) { synth_sigs[i] = 0x11; synth_sigs[384 + i] = 0x22; }
  memcpy(b + 72 + BOOTER_IMAGE, synth_sigs, 768);
  return b;
}

static void build_payload(void) {
  for (size_t i = 0; i < PAYLOAD_SIZE; i += 4) wr32le(synth_payload + i, 0x0000019c);
  // a few recipe-shaped patch words; content is pinned by hash, the fake checks byte integrity only
  wr32le(synth_payload + 0xf960, 0x001fa7cc);
  wr32le(synth_payload + 0xf948, 0xfffff0ff);
  wr32le(synth_payload + 0xf9e0, 0x67b00001);
}

// minimal ELF64 with .fwimage, .fwsignature_ad10x and .shstrtab, accepted by gpu_unlock_elf_section
static uint8_t *build_gsp_elf(size_t *len) {
  const char *names = "\0.fwimage\0.fwsignature_ad10x\0.shstrtab";
  size_t names_len = 1 + 9 + 19 + 10;
  size_t off_img = 64, off_sig = off_img + GSP_IMAGE_SIZE, off_str = off_sig + SIG_SIZE;
  size_t off_sh = (off_str + names_len + 7) & ~7ull;
  *len = off_sh + 4 * 64;
  uint8_t *b = calloc(1, *len);
  b[0] = 0x7f; b[1] = 'E'; b[2] = 'L'; b[3] = 'F'; b[4] = 2; b[5] = 1; b[6] = 1;
  wr16le(b + 16, 2);      // ET_EXEC
  wr16le(b + 18, 62);     // EM_X86_64 (the parser does not care)
  wr32le(b + 20, 1);      // e_version
  wr16le(b + 52, 64);     // e_ehsize
  wr32le(b + 40, (uint32_t)off_sh); // e_shoff (low; file < 4 GiB)
  wr16le(b + 58, 64);     // e_shentsize
  wr16le(b + 60, 4);      // e_shnum
  wr16le(b + 62, 3);      // e_shstrndx
  memset(b + off_img, 0x3c, GSP_IMAGE_SIZE);
  memset(b + off_sig, 0x5a, SIG_SIZE);
  memcpy(synth_stock, b + off_sig, SIG_SIZE);
  memcpy(b + off_str, names, names_len);
  uint8_t *sh = b + off_sh;
  // section 1: .fwimage
  wr32le(sh + 64, 1);
  wr32le(sh + 64 + 24, (uint32_t)off_img);
  wr32le(sh + 64 + 32, GSP_IMAGE_SIZE);
  // section 2: .fwsignature_ad10x
  wr32le(sh + 128, 10);
  wr32le(sh + 128 + 24, (uint32_t)off_sig);
  wr32le(sh + 128 + 32, SIG_SIZE);
  // section 3: .shstrtab
  wr32le(sh + 192, 29);
  wr32le(sh + 192 + 24, (uint32_t)off_str);
  wr32le(sh + 192 + 32, (uint32_t)names_len);
  return b;
}

static uint8_t *build_boot(size_t *len) {
  *len = BOOT_SIZE;
  uint8_t *b = malloc(BOOT_SIZE);
  memset(b, 0x77, BOOT_SIZE);
  return b;
}

static uint8_t *build_desc(size_t *len) {
  *len = 84;
  uint8_t *b = calloc(1, 84);
  wr32le(b + 0, 5);          // version
  wr32le(b + 4, 0);          // range 1: off
  wr32le(b + 8, 2048);       //          len
  wr32le(b + 12, 2048);      // range 3: off
  wr32le(b + 16, 16384);     //          len
  wr32le(b + 32, 0);         // manifest offset (word 8)
  wr32le(b + 36, 2048);
  wr32le(b + 40, 2048);      // data offset (word 10)
  wr32le(b + 44, 16384);
  wr32le(b + 48, 18432);     // code offset (word 12)
  wr32le(b + 52, 18432);
  wr32le(b + 56, 1);
  wr32le(b + 76, BOOT_SIZE);
  wr32le(b + 80, 0);
  return b;
}

// synthetic board ROM: one expansion image, PCIR, BIT token -> ucode table -> FWSEC-FRTS v3 descriptor (engine 0x400)
#define ROM_SIZE 0x8000
static void build_rom(uint8_t *rom) {
  memset(rom, 0, ROM_SIZE);
  rom[0] = 0x55; rom[1] = 0xaa;
  wr16le(rom + 0x18, 0x40);            // PCIR offset
  memcpy(rom + 0x40, "PCIR", 4);
  wr16le(rom + 0x40 + 0x0a, 0);        // no NPDE
  wr16le(rom + 0x40 + 0x10, 16);       // image length, 512-byte units
  rom[0x40 + 0x14] = 0;                // type: base image
  rom[0x40 + 0x15] = 0x80;             // last image
  rom[0x100] = 0xff; rom[0x101] = 0xb8;
  memcpy(rom + 0x102, "BIT\0", 4);
  rom[0x108] = 12;                     // BIT size
  rom[0x109] = 8;                      // token stride
  rom[0x10a] = 1;                      // tokens
  unsigned sum = 0;
  for (int i = 0; i < 11; i++) sum += rom[0x100 + i];
  rom[0x10b] = (uint8_t)(256 - (sum & 0xff));
  rom[0x10c] = 0x70; rom[0x10d] = 2;   // ucode token
  wr16le(rom + 0x10e, 8);
  wr32le(rom + 0x110, 0x180);          // pointer to the table pointer
  wr32le(rom + 0x180, 0x200);          // ucode table offset
  rom[0x200] = 1; rom[0x201] = 6; rom[0x202] = 6; rom[0x203] = 1;
  rom[0x206] = 0x85;                   // FWSEC entry
  wr32le(rom + 0x208, 0x300);          // descriptor offset
  uint8_t *d = rom + 0x300;
  wr32le(d, (812u << 16) | (3u << 8) | 1u); // size 44+768, version 3, valid
  wr32le(d + 4, 0x1580);               // stored image size: imem 0x800 + dmem 0xd80
  wr32le(d + 8, 0x900);                // pkc offset in DMEM
  wr32le(d + 12, 0x100);               // interface offset in DMEM
  wr32le(d + 16, 0);                   // imem pa
  wr32le(d + 20, 0x800);               // imem size
  wr32le(d + 24, 0);                   // imem va
  wr32le(d + 28, 0);                   // dmem pa
  wr32le(d + 32, 0xd80);               // dmem size (the real AD102 FWSEC DMEM length)
  wr16le(d + 36, 0x400);               // GSP engine mask
  d[38] = 1;                           // ucode id
  d[39] = 2;                           // signature count
  wr16le(d + 40, 0x3);                 // versions 0 and 1
  memset(d + 44, 0x66, 768);           // signatures
  uint8_t *img = rom + 0x300 + 812;
  memset(img, 0x99, 0x800);            // IMEM
  uint8_t *dmem = img + 0x800;
  memset(dmem, 0, 0xd80);
  dmem[0x100] = 1; dmem[0x101] = 4; dmem[0x102] = 8; dmem[0x103] = 2; // interface v1, 2 entries
  wr32le(dmem + 0x104, 4);             // entry: mapper
  wr32le(dmem + 0x108, 0x200);         // mapper destination in DMEM
  uint8_t *mapper = dmem + 0x200;
  wr16le(mapper + 4, 3);               // mapper version 3
  wr16le(mapper + 6, 64);
  wr32le(mapper + 8, 0x300);           // command offset
  wr32le(mapper + 12, 64);             // command size
}

// ---- the fake card ----------------------------------------------------------------------------------------------------

#define FB_MB 24576u // 24 GiB, like the real card
#define BAR0_SIZE 0x1000000u

enum {
  K_NONE = 0,
  K_WRONG_DEVID, K_WRONG_CHIP, K_CHIP_ALLONES, K_NO_FLR, K_NOT_D0, K_CMD_DROP,
  K_WPR2_UP, K_GFW_HANG, K_HIGH_IOVA, K_NONCONTIG, K_MISALIGN,
  K_ROM_GARBAGE, K_FWSEC_FAIL, K_FWSEC_WPR2_WRONG,
  K_MARKER_ONLY, K_NO_PERMS, K_WPR_STUCK, K_SELECTOR_DROP, K_READOUT_STUCK,
  K_RESET_DEAD, K_RESET_WPR2_UP, K_RETENTION_LOSS, K_RESET_NO_GFW, K_RESET_CLEARS_BARS,
  K_CORRUPT_PAYLOAD, K_ABORT_ON_SEC2,
  K_DEVSTA_BIT0, K_DEVSTA_PENDING,
  K_FWSEC_BAD_PKC, K_FWSEC_BAD_SIGNATURE,
  K_STALL_SEC2, // wire mode: delay the next read reply after SEC2 starts (signal-delivery window)
  K_RETAINED,   // The card already carries a verified override (retained-unlock cold handoff state)
};

typedef struct {
  tinynv_pci_t pci;
  uint32_t regs[BAR0_SIZE / 4];
  uint8_t cfg[256];
  uint8_t rom[ROM_SIZE];
  int rom_valid;
  struct { uint8_t *va; size_t size; uint64_t iova; uint64_t *pages; size_t npages; int live; } dma[128];
  int ndma;
  uint64_t next_iova;
  uint8_t imem[2][0x20000], dmem[2][0x20000]; // 0 = GSP, 1 = SEC2
  uint64_t src_base[2];
  int dma_fault[2];
  int nwrites, nstarts, nreset, ndma_alloc, ndma_free;
  int sec2_applied, sec2_saw_magic, sec2_saw_bme, booter_patched;
  uint64_t meta_sig_at_selector;
  int knob;
  int wire_mode;      // serve the real pinned bundle: payload gate reads the published v67 buffer (slot 1)
  int stall_read_ms;  // set by K_STALL_SEC2; the wire server applies and clears it before the next read reply
} fake_t;

static fake_t F;

static uint32_t cfg_rd(tinynv_pci_t *p, uint32_t off, uint32_t size) {
  (void)p;
  if (off + size > sizeof(F.cfg)) return ~0u;
  uint32_t v = 0;
  for (uint32_t i = 0; i < size; i++) v |= (uint32_t)F.cfg[off + i] << (8 * i);
  return v;
}
static void cfg_wr(tinynv_pci_t *p, uint32_t off, uint32_t size, uint32_t val) {
  (void)p;
  if (F.knob == K_CMD_DROP && off == 0x04) return;
  if (off + size > sizeof(F.cfg)) return;
  for (uint32_t i = 0; i < size; i++) F.cfg[off + i] = (uint8_t)(val >> (8 * i));
}

static int eng_of(uint32_t off) { return (off & 0xfff000) == 0x840000 ? 1 : 0; }

static int buf_contains(uint64_t iova, size_t n) {
  for (int i = 0; i < F.ndma; i++)
    if (F.dma[i].live && iova >= F.dma[i].iova && iova + n <= F.dma[i].iova + F.dma[i].size) return i;
  return -1;
}
static uint8_t *buf_ptr(uint64_t iova) {
  int i = buf_contains(iova, 1);
  return i < 0 ? NULL : F.dma[i].va + (iova - F.dma[i].iova);
}
static uint32_t cold_readout(void) { return 0x00410080; }

static uint32_t fake_rd32(tinynv_mmio_t *m, uint64_t off) {
  (void)m;
  if (off >= BAR0_SIZE) return ~0u;
  uint32_t o = (uint32_t)off;
  if (o == 0x823818) { // effective readout: derived, never directly writable
    if (F.knob == K_READOUT_STUCK) return cold_readout();
    int full = F.regs[0x823804 / 4] == ~0u && F.regs[0x82381c / 4] == 0x88888888 && F.regs[0x823820 / 4] == 0x8;
    return full ? 0 : cold_readout();
  }
  if (o == 0x840118 || o == 0x110118) return 2; // DMA idle, not full
  return F.regs[o / 4];
}

static void gsp_execute(void) {
  // FWSEC-FRTS: find the patched FRTS command in the loaded DMEM and establish WPR2 from it.
  uint32_t frts_page = 0;
  int found = 0;
  for (uint32_t at = 0; at + 44 <= 0xd80; at += 4) {
    const uint8_t *c = F.dmem[0] + at;
    if (rd32le(c) == 1 && rd32le(c + 4) == 24 && rd32le(c + 20) == 2 && rd32le(c + 24) == 1 && rd32le(c + 28) == 20 &&
        rd32le(c + 36) == 0x100 && rd32le(c + 40) == 2) {
      frts_page = rd32le(c + 32);
      found = 1;
      break;
    }
  }
  int bme = (cfg_rd(NULL, 4, 2) & 4) != 0;
  if (!found || F.dma_fault[0] || !bme || F.knob == K_FWSEC_FAIL) {
    F.regs[0x1438 / 4] = 0x12340000; // ACR scratch hi16 != 0: FWSEC failed
  } else {
    F.regs[0x1438 / 4] = 0;
    uint64_t frts = (uint64_t)frts_page << 12;
    uint32_t lo = (uint32_t)(frts_page << 4);
    uint32_t hi = (uint32_t)((((frts + 0x100000 - 0x1000) >> 12) << 4) | 1);
    if (F.knob == K_FWSEC_WPR2_WRONG) lo ^= 0x10; // a WPR2 that is not the FRTS reservation
    F.regs[0x1fa824 / 4] = lo;
    F.regs[0x1fa828 / 4] = hi;
  }
  F.regs[0x110100 / 4] = 0x10; // halted
}

static void sec2_execute(void) {
  uint64_t meta_iova = F.regs[0x840040 / 4] | ((uint64_t)F.regs[0x840044 / 4] << 32);
  uint8_t *meta = buf_ptr(meta_iova);
  F.sec2_saw_magic = meta && rd64le(meta) == 0xdc3aae21371a60b3ull && rd64le(meta + 8) == 1;
  F.sec2_saw_bme = (cfg_rd(NULL, 4, 2) & 4) != 0;
  if (!F.wire_mode) // the synthetic-signature check only means something against the synthetic booter
    for (int i = 0; i < F.ndma; i++)
      if (F.dma[i].live && F.dma[i].size >= BOOTER_IMAGE && !memcmp(F.dma[i].va + 32272, synth_sigs, 384))
        F.booter_patched = 1;
  if (F.knob == K_ABORT_ON_SEC2) tensor_unlock_abort = 1;
  if (F.knob == K_MARKER_ONLY) {
    // the false-success shape: marker set, engine never halts, permissions never applied
    F.regs[0x840040 / 4] = 0x67b00001;
    F.regs[0x840100 / 4] = 0;
    return;
  }
  uint64_t sig_iova = meta ? rd64le(meta + 72) : 0;
  uint64_t sig_size = meta ? rd64le(meta + 80) : 0;
  uint8_t *sig = buf_ptr(sig_iova);
  const uint8_t *expect = synth_payload;
  size_t expect_size = PAYLOAD_SIZE;
  if (F.wire_mode) { // real bundle: the published copy must equal the v67 buffer the adapter integrity-checked
    if (!F.dma[1].live) return;
    expect = F.dma[1].va;
    expect_size = GU_V67_SIZE;
  }
  int is_v67 = sig && sig_size == expect_size && !memcmp(sig, expect, expect_size);
  // publication gates: the radix root must point at the next page of the same validated buffer, boot image live
  uint64_t radix_iova = meta ? rd64le(meta + 16) : 0;
  uint8_t *radix = buf_ptr(radix_iova);
  int radix_ok = radix && rd64le(radix) == radix_iova + 0x1000;
  int boot_ok = meta && buf_ptr(rd64le(meta + 32)) != NULL;
  if (F.knob != K_NO_PERMS && F.sec2_saw_magic && F.sec2_saw_bme && is_v67 && radix_ok && boot_ok && !F.dma_fault[1]) {
    F.regs[0x1fa7cc / 4] = 0xfffff0ff;                    // WPR_CFG
    if (F.knob != K_WPR_STUCK) F.regs[0x1fa7c4 / 4] = ~0u; // WPR
    F.regs[0x823804 / 4] = ~0u;                            // FEAT
    // FBPA stays at its cold observed value: the optional write, exercised on every happy path
    F.regs[0x840040 / 4] = 0x67b00001;
    F.sec2_applied = 1;
  } else if (F.knob == K_NO_PERMS) {
    F.regs[0x840040 / 4] = 0x67b00001; // marker + halt, but no privileged writes
  }
  F.regs[0x840100 / 4] = 0x10; // halted
  if (F.knob == K_STALL_SEC2) F.stall_read_ms = 2000;
}

static void fake_wr32(tinynv_mmio_t *m, uint64_t off, uint32_t v) {
  (void)m;
  if (off >= BAR0_SIZE) return;
  uint32_t o = (uint32_t)off;
  F.nwrites++;
  uint32_t old = F.regs[o / 4];
  F.regs[o / 4] = v;
  int e = eng_of(o);
  uint32_t base = e ? 0x840000 : 0x110000;
  uint32_t rel = o - base;
  if (o == 0x82381c || o == 0x823820) {
    // host selector writes are inert until the privileged stage opened FEAT; SS1 may also be dropped (fault inject)
    if (F.regs[0x823804 / 4] != ~0u || (F.knob == K_SELECTOR_DROP && o == 0x823820)) F.regs[o / 4] = old;
    if (o == 0x82381c) {
      // record what the WPR meta points at when the selectors are applied: must be the STOCK span, never the V67
      for (int i = 0; i < F.ndma; i++) {
        if (!F.dma[i].live || F.dma[i].size < 256) continue;
        if (rd64le(F.dma[i].va) == 0xdc3aae21371a60b3ull) F.meta_sig_at_selector = rd64le(F.dma[i].va + 72);
      }
    }
    return;
  }
  if (rel == 0x110) F.src_base[e] = (F.src_base[e] & ~0xffffffffull) | v;
  else if (rel == 0x128) F.src_base[e] = (F.src_base[e] & 0xffffffffull) | ((uint64_t)(v & 0x1ff) << 32);
  else if (rel == 0x118 && v) {
    uint64_t src = (F.src_base[e] << 8) + F.regs[(base + 0x11c) / 4];
    uint32_t dest = F.regs[(base + 0x114) / 4];
    int b = buf_contains(src, 256);
    if (b < 0 || dest + 256 > 0x20000) F.dma_fault[e] = 1;
    else memcpy((v & 0x10) ? F.imem[e] + dest : F.dmem[e] + dest, F.dma[b].va + (src - F.dma[b].iova), 256);
  } else if ((rel == 0x100 || rel == 0x130) && (v & 2)) {
    F.nstarts++;
    if (e) sec2_execute();
    else gsp_execute();
  } else if (rel == 0x3c0 && e == 1 && (v & 1) && F.knob == K_CORRUPT_PAYLOAD) {
    // corrupt the published payload copy while SEC2 is in reset: the adapter's pre-start integrity check must catch it
    for (int i = 0; i < F.ndma; i++)
      if (F.dma[i].live && F.dma[i].size >= PAYLOAD_SIZE && !memcmp(F.dma[i].va, synth_payload, PAYLOAD_SIZE)) {
        F.dma[i].va[123] ^= 0xff;
        break;
      }
  }
}

static void fake_rd_block(tinynv_mmio_t *m, uint64_t off, void *dst, size_t n) {
  (void)m;
  if (off == 0x300000 && F.rom_valid) {
    size_t k = n < ROM_SIZE ? n : ROM_SIZE;
    memcpy(dst, F.rom, k);
    if (k < n) memset((uint8_t *)dst + k, 0xff, n - k);
    return;
  }
  memset(dst, 0xff, n);
}
static void fake_wr_block(tinynv_mmio_t *m, uint64_t off, const void *src, size_t n) { (void)m; (void)off; (void)src; (void)n; }

static int fake_bar_info(tinynv_pci_t *p, int bar, uint64_t *base, uint64_t *size) {
  (void)p;
  if (bar != 0) return -1;
  if (base) *base = 0;
  if (size) *size = BAR0_SIZE;
  return 0;
}
static int fake_bar_map(tinynv_pci_t *p, int bar, uint64_t off, uint64_t size, tinynv_mmio_t *out) {
  (void)p;
  if (bar != 0) return -1;
  *out = (tinynv_mmio_t){.ptr = NULL, .ctx = NULL, .bar = 0, .off = off, .size = size ? size : BAR0_SIZE,
                         .rd32 = fake_rd32, .wr32 = fake_wr32, .rd_block = fake_rd_block, .wr_block = fake_wr_block};
  return 0;
}
static void fake_bar_unmap(tinynv_pci_t *p, tinynv_mmio_t *m) { (void)p; memset(m, 0, sizeof(*m)); }

static int fake_dma_alloc(tinynv_pci_t *p, size_t size, tinynv_dma_t *out) {
  (void)p;
  if (F.ndma >= 128) return -1;
  size_t mapped = (size + 0xfff) & ~0xfffull;
  if (mapped < 0x4000) mapped = 0x4000; // the server's 16 KiB minimum
  uint8_t *va = NULL;
  if (posix_memalign((void **)&va, 0x1000, mapped)) return -1;
  memset(va, 0, mapped);
  size_t npages = mapped / 0x1000;
  uint64_t *pages = malloc(npages * sizeof(uint64_t));
  uint64_t base = F.next_iova;
  if (F.knob == K_MISALIGN) base += 0x100;
  for (size_t i = 0; i < npages; i++) pages[i] = base + i * 0x1000;
  if (F.knob == K_NONCONTIG && npages > 4)
    for (size_t i = 3; i < npages; i++) pages[i] += 0x2000; // a broken IOVA run
  F.next_iova = base + mapped;
  int i = F.ndma++;
  F.dma[i].va = va;
  F.dma[i].size = mapped;
  F.dma[i].iova = pages[0];
  F.dma[i].pages = pages;
  F.dma[i].npages = npages;
  F.dma[i].live = 1;
  F.ndma_alloc++;
  *out = (tinynv_dma_t){.va = va, .size = mapped, .pages = pages, .npages = npages, .ctx = NULL};
  tinynv_dma_bind_host(out);
  return 0;
}
static void fake_dma_free(tinynv_pci_t *p, tinynv_dma_t *dma) {
  (void)p;
  for (int i = 0; i < F.ndma; i++)
    if (F.dma[i].live && F.dma[i].va == dma->va) {
      F.dma[i].live = 0;
      free(F.dma[i].va);
      free(F.dma[i].pages);
      F.ndma_free++;
    }
  dma->pages = NULL;
  dma->npages = 0;
}

static void fake_quiesce(tinynv_pci_t *p) {
  uint32_t cmd = cfg_rd(p, 4, 2);
  if (cmd == 0xffff) return;
  cfg_wr(p, 4, 2, cmd & ~4u);
  cfg_rd(p, 4, 2);
}

static int fake_reset(tinynv_pci_t *p) {
  (void)p;
  F.nreset++;
  // the RPC always reports success, like the installed dext (it discards Reset's return value). Effects only.
  if (F.knob == K_RESET_DEAD) {
    memset(F.cfg, 0xff, 16);
    return 0;
  }
  if (F.knob == K_RESET_CLEARS_BARS) {
    memset(F.cfg + 0x10, 0, 24);
    cfg_wr(NULL, 4, 2, 0);
  }
  // Apple's FunctionReset restores config itself (simulated: left as-is unless cleared above).
  if (F.knob != K_RESET_WPR2_UP) {
    F.regs[0x1fa824 / 4] = 0;
    F.regs[0x1fa828 / 4] = 0;
  }
  F.regs[0x118128 / 4] = 1;
  F.regs[0x118234 / 4] = F.knob == K_RESET_NO_GFW ? 0 : 0xff;
  F.regs[0x110100 / 4] = 0x10;
  F.regs[0x840100 / 4] = 0x10;
  F.regs[0x8403c0 / 4] = 0;
  F.regs[0x1438 / 4] = 0;
  if (F.knob == K_RETENTION_LOSS) {
    F.regs[0x823804 / 4] = 0xffffff8f;
    F.regs[0x82381c / 4] = 0x44023025;
    F.regs[0x823820 / 4] = 0x7;
  }
  return 0;
}
static void fake_close(tinynv_pci_t *p) {
  (void)p;
  for (int i = 0; i < F.ndma; i++)
    if (F.dma[i].live) {
      free(F.dma[i].va);
      free(F.dma[i].pages);
      F.dma[i].live = 0;
    }
}

static void fake_init(int knob) {
  memset(&F, 0, sizeof(F));
  F.knob = knob;
  F.rom_valid = 1;
  F.next_iova = 0x84000000; // low IOVAs, as observed on this Mac
  if (knob == K_HIGH_IOVA) F.next_iova = 0x100000000ull;
  F.pci = (tinynv_pci_t){.ctx = NULL, .name = "fake-ad102", .live = 1,
                         .cfg_read = cfg_rd, .cfg_write = cfg_wr, .bar_info = fake_bar_info, .bar_map = fake_bar_map,
                         .bar_unmap = fake_bar_unmap, .dma_alloc = fake_dma_alloc, .dma_free = fake_dma_free,
                         .reset = fake_reset, .quiesce = fake_quiesce, .close = fake_close};
  build_rom(F.rom);
  if (knob == K_ROM_GARBAGE) { memset(F.rom, 0, ROM_SIZE); F.rom_valid = 1; }
  wr32le(F.cfg, knob == K_WRONG_DEVID ? 0x270510de : 0x268410de);
  wr16le(F.cfg + 4, 0x7); // IO|MEM|MASTER: the dext masters at provider start, before any client
  const uint32_t bars[6] = {0x90000004, 0x1, 0x8000000c, 0x2, 0, 0};
  for (int i = 0; i < 6; i++) wr32le(F.cfg + 0x10 + 4 * i, bars[i]);
  F.cfg[0x34] = 0x40;
  F.cfg[0x40] = 0x01; F.cfg[0x41] = 0x60;       // PM cap -> PCIe cap
  wr16le(F.cfg + 0x44, knob == K_NOT_D0 ? 0x3 : 0x8); // PMCSR
  F.cfg[0x60] = 0x10; F.cfg[0x61] = 0x00;
  wr32le(F.cfg + 0x64, knob == K_NO_FLR ? 0 : 0x10000000); // DEVCAP: FLR
  wr16le(F.cfg + 0x6a, 0);                      // DEVSTA: no pending transactions
  if (knob == K_DEVSTA_BIT0) wr16le(F.cfg + 0x6a, 0x1);      // correctable error only; not transactions pending
  if (knob == K_DEVSTA_PENDING) wr16le(F.cfg + 0x6a, 0x20);  // Transactions Pending
  F.regs[0] = knob == K_WRONG_CHIP ? 0x193000a1 : knob == K_CHIP_ALLONES ? ~0u : 0x192000a1;
  F.regs[0x1183a4 / 4] = FB_MB;
  F.regs[0x625f04 / 4] = 0;
  F.regs[0x1fa7c8 / 4] = 1;
  F.regs[0x1fa82c / 4] = 0;
  F.regs[0x1fa830 / 4] = 0;
  F.regs[0x118128 / 4] = 1;
  F.regs[0x118234 / 4] = knob == K_GFW_HANG ? 0 : 0xff;
  F.regs[0x110100 / 4] = 0x10;
  F.regs[0x823804 / 4] = 0xffffff8f; // cold FEAT, as measured
  F.regs[0x82381c / 4] = 0x44023025;
  F.regs[0x823820 / 4] = 0x7;
  F.regs[0x9a0148 / 4] = 0xffffff0f; // FBPA cold observed value
  F.regs[0x824148 / 4] = 0x1;        // SEC2 ucode 3 fuse -> version 1 -> signature index 0
  F.regs[0x8241c0 / 4] = 0x1;        // FWSEC ucode 1 fuse
  F.regs[0x8400f4 / 4] = 0x80000000;
  F.regs[0x1100f4 / 4] = 0x80000000;
  if (knob == K_WPR2_UP) F.regs[0x1fa828 / 4] = 0x100;
  if (knob == K_RETAINED) { // a power cycle's worth of retained override: FEAT open, selectors set, readout derived 0
    F.regs[0x823804 / 4] = ~0u;
    F.regs[0x82381c / 4] = 0x88888888;
    F.regs[0x823820 / 4] = 0x8;
  }
  if (knob == K_FWSEC_BAD_PKC) wr32le(F.rom + 0x308, 0xe00); // signature slot extends past FWSEC DMEM
  if (knob == K_FWSEC_BAD_SIGNATURE) F.rom[0x327] = 3;       // count disagrees with the two version bits
}

#endif
