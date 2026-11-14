// The 610 boot-metadata adapters under test, with no GPU.
//
// Drives the pure boot610 functions with synthetic fixtures built from the pinned AD102 geometries, and - when the
// prepared bundle is passed as argv[1] - with the real hash-pinned assets. Covers the four fixture classes the card
// requires: FIELD (every field at its offset, serialization round-trips), RANGE (offsets/sizes inside their image),
// OVERFLOW (size/offset arithmetic that would wrap is refused), and WRONG-PROFILE (a 570 NVFW image or a wrong-chip
// geometry cannot pass as a 610 AD102 asset). Nothing here runs on a GPU or proves firmware boots.
#include "boot610.h"
#include "fw_layout_610.h"
#include "fw.h"
#include "mmu.h"
#include "runtime_profile.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static void wr32(uint8_t *p, uint32_t v) { p[0]=v; p[1]=v>>8; p[2]=v>>16; p[3]=v>>24; }
static uint32_t rd32(const uint8_t *p) { return p[0] | (p[1]<<8) | (p[2]<<16) | ((uint32_t)p[3]<<24); }

// Pack a BTL1 v1 container exactly as the extractor does (magic+ver, 14 header words, image_size+sig_bytes,
// image, signatures). The caller frees the result.
static uint8_t *pack_btl1(const tinynv_btl1_hdr_t *h, uint32_t image_size, const uint8_t *image, const uint8_t *sigs,
                          size_t *len_out) {
  size_t total = TINYNV_BTL1_PREFIX + image_size + h->sig_bytes;
  uint8_t *b = calloc(1, total);
  wr32(b, TINYNV_BTL1_MAGIC); wr32(b + 4, TINYNV_BTL1_VERSION);
  const uint32_t *w = (const uint32_t *)(const void *)h;
  for (int i = 0; i < 14; i++) wr32(b + 8 + i * 4, w[i]);
  wr32(b + 64, image_size); wr32(b + 68, h->sig_bytes);
  memcpy(b + TINYNV_BTL1_PREFIX, image, image_size);
  memcpy(b + TINYNV_BTL1_PREFIX + image_size, sigs, h->sig_bytes);
  *len_out = total;
  return b;
}

// A synthetic image + two distinguishable signatures for one pinned geometry.
typedef struct { uint8_t *image, *sigs, *container; size_t len; tinynv_btl1_hdr_t h; uint32_t image_size; } btl1_fix_t;
static btl1_fix_t make_fix(const char *which) {
  btl1_fix_t f;
  const tinynv_btl1_hdr_t *g = tinynv_boot610_geom(which, &f.image_size);
  f.h = *g;
  f.image = malloc(f.image_size);
  for (uint32_t i = 0; i < f.image_size; i++) f.image[i] = (uint8_t)(i * 7 + 1);
  f.sigs = malloc(f.h.sig_bytes);
  for (uint32_t s = 0; s < f.h.num_sigs; s++)
    for (uint32_t i = 0; i < TINYNV_BTL1_SIG_SIZE; i++) f.sigs[s * TINYNV_BTL1_SIG_SIZE + i] = (uint8_t)(0xA0 + s);
  f.container = pack_btl1(&f.h, f.image_size, f.image, f.sigs, &f.len);
  return f;
}
static void free_fix(btl1_fix_t *f) { free(f->image); free(f->sigs); free(f->container); }

// ---- BTL1: valid parse + geometry pin ----------------------------------------------------------------------------
static void btl1_valid(void) {
  const char *names[] = {"booter_load", "booter_unload", "scrubber"};
  for (size_t k = 0; k < 3; k++) {
    btl1_fix_t f = make_fix(names[k]);
    tinynv_btl1_t b;
    CHECK(tinynv_boot610_btl1_parse(f.container, f.len, &f.h, f.image_size, &b) == 0,
          "%s: valid container refused: %s", names[k], tinynv_last_error());
    CHECK(b.image_size == f.image_size && b.hdr.ucode_id == f.h.ucode_id, "%s: parsed geometry wrong", names[k]);
    CHECK(b.sig_dmem_off == f.h.patch_loc - f.h.os_data_off, "%s: sig DMEM offset wrong", names[k]);
    // Without the geometry pin the same container still parses (the pin is an extra defence, not the only check).
    CHECK(tinynv_boot610_btl1_parse(f.container, f.len, NULL, 0, &b) == 0, "%s: unpinned parse refused", names[k]);
    free_fix(&f);
  }
  printf("btl1: the three pinned AD102 geometries parse, pinned and unpinned\n");
}

// ---- BTL1: wrong-profile / wrong-chip / corrupt ------------------------------------------------------------------
static void btl1_refusals(void) {
  btl1_fix_t f = make_fix("booter_load");
  tinynv_btl1_t b;

  // WRONG-PROFILE: a 570 NVFW image (magic 0x10de) must be refused explicitly, not parsed as BTL1.
  uint8_t *nvfw = malloc(f.len); memcpy(nvfw, f.container, f.len); wr32(nvfw, 0x000010deu);
  CHECK(tinynv_boot610_btl1_parse(nvfw, f.len, &f.h, f.image_size, &b) != 0, "a 570 NVFW image passed as a 610 booter");
  CHECK(strstr(tinynv_last_error(), "570 NVFW") != NULL, "the NVFW refusal does not name the container: %s", tinynv_last_error());
  free(nvfw);

  // Wrong magic that is neither BTL1 nor NVFW.
  uint8_t *bad = malloc(f.len); memcpy(bad, f.container, f.len); wr32(bad, 0xdeadbeefu);
  CHECK(tinynv_boot610_btl1_parse(bad, f.len, NULL, 0, &b) != 0, "a garbage magic passed"); free(bad);

  // Truncated header.
  CHECK(tinynv_boot610_btl1_parse(f.container, 40, NULL, 0, &b) != 0, "a truncated header passed");
  // Truncated payload (one byte short of the declared total).
  CHECK(tinynv_boot610_btl1_parse(f.container, f.len - 1, NULL, 0, &b) != 0, "a truncated payload passed");
  // Trailing data (one byte over).
  uint8_t *over = malloc(f.len + 1); memcpy(over, f.container, f.len); over[f.len] = 0;
  CHECK(tinynv_boot610_btl1_parse(over, f.len + 1, NULL, 0, &b) != 0, "trailing data passed"); free(over);

  // Bad version.
  bad = malloc(f.len); memcpy(bad, f.container, f.len); wr32(bad + 4, 2);
  CHECK(tinynv_boot610_btl1_parse(bad, f.len, NULL, 0, &b) != 0, "BTL1 v2 passed"); free(bad);

  // Signature size fields disagree.
  bad = malloc(f.len); memcpy(bad, f.container, f.len); wr32(bad + 68, f.h.sig_bytes + 384);
  CHECK(tinynv_boot610_btl1_parse(bad, f.len, NULL, 0, &b) != 0, "disagreeing sig sizes passed"); free(bad);

  // num_apps != 1.
  bad = malloc(f.len); memcpy(bad, f.container, f.len); wr32(bad + 8 + 4 * 4, 2);
  CHECK(tinynv_boot610_btl1_parse(bad, f.len, NULL, 0, &b) != 0, "num_apps=2 passed"); free(bad);

  // num_sigs implausible (0 and 9).
  for (uint32_t ns = 0; ns <= 9; ns += 9) {
    bad = malloc(f.len); memcpy(bad, f.container, f.len); wr32(bad + 8 + 12 * 4, ns);
    CHECK(tinynv_boot610_btl1_parse(bad, f.len, NULL, 0, &b) != 0, "num_sigs=%u passed", ns); free(bad);
  }

  // RANGE: os_data misaligned (off not a multiple of 256).
  bad = malloc(f.len); memcpy(bad, f.container, f.len); wr32(bad + 8 + 2 * 4, f.h.os_data_off + 1);
  CHECK(tinynv_boot610_btl1_parse(bad, f.len, NULL, 0, &b) != 0, "a misaligned os_data passed"); free(bad);

  // RANGE/OVERFLOW: os_data off+size wraps past the image (off near UINT32_MAX).
  bad = malloc(f.len); memcpy(bad, f.container, f.len); wr32(bad + 8 + 2 * 4, 0xfffffff0u);
  CHECK(tinynv_boot610_btl1_parse(bad, f.len, NULL, 0, &b) != 0, "an os_data range that wraps passed"); free(bad);

  // RANGE: patch_loc outside the DMEM range (before os_data_off).
  bad = malloc(f.len); memcpy(bad, f.container, f.len); wr32(bad + 8 + 9 * 4, f.h.os_data_off - 256);
  CHECK(tinynv_boot610_btl1_parse(bad, f.len, NULL, 0, &b) != 0, "a patch_loc before DMEM passed"); free(bad);
  // patch_loc too close to the end of DMEM for a full signature.
  bad = malloc(f.len); memcpy(bad, f.container, f.len); wr32(bad + 8 + 9 * 4, f.h.os_data_off + f.h.os_data_size - 16);
  CHECK(tinynv_boot610_btl1_parse(bad, f.len, NULL, 0, &b) != 0, "a patch_loc without room for a signature passed"); free(bad);

  // WRONG-CHIP: the GA10x booter_load geometry (image 60416) must not match the AD102 pin. Build a container with the
  // GA10x geometry and try to parse it against the AD102 expectation.
  tinynv_btl1_hdr_t ga = f.h; ga.os_data_off = 35328; ga.os_data_size = 25088; ga.app_code_size = 35072; ga.patch_loc = 35344;
  uint32_t ga_img = 60416;
  uint8_t *ga_image = calloc(1, ga_img);
  uint8_t *ga_c = pack_btl1(&ga, ga_img, ga_image, f.sigs, &f.len);
  CHECK(tinynv_boot610_btl1_parse(ga_c, f.len, &f.h, f.image_size, &b) != 0, "a GA10x geometry passed the AD102 pin");
  CHECK(strstr(tinynv_last_error(), "wrong chip") != NULL, "the wrong-chip refusal is not named: %s", tinynv_last_error());
  free(ga_image); free(ga_c);

  free_fix(&f);
  printf("btl1: NVFW/wrong-chip/truncated/overflow/range fixtures all refused\n");
}

// ---- BTL1: fuse-selected signature patch -------------------------------------------------------------------------
static void btl1_patch(void) {
  btl1_fix_t f = make_fix("booter_load");
  tinynv_btl1_t b;
  CHECK(tinynv_boot610_btl1_parse(f.container, f.len, &f.h, f.image_size, &b) == 0, "patch: parse failed");
  // num_sigs=2: index = 1 - fuse_ver. fuse 0 -> sig 1 (0xA1), fuse 1 -> sig 0 (0xA0).
  for (uint32_t fuse = 0; fuse < 2; fuse++) {
    uint8_t *img = NULL; size_t n = 0;
    CHECK(tinynv_boot610_btl1_image(&b, f.container, fuse, &img, &n) == 0, "patch fuse %u failed", fuse);
    CHECK(n == f.image_size, "patch fuse %u: wrong image size", fuse);
    uint32_t want = 1 - fuse;
    int ok = 1;
    for (uint32_t i = 0; i < TINYNV_BTL1_SIG_SIZE; i++)
      if (img[f.h.patch_loc + i] != (uint8_t)(0xA0 + want)) ok = 0;
    CHECK(ok, "patch fuse %u selected the wrong signature", fuse);
    // The rest of the image is untouched.
    CHECK(img[0] == f.image[0] && img[f.h.patch_loc - 1] == f.image[f.h.patch_loc - 1], "patch fuse %u corrupted the image", fuse);
    free(img);
  }
  // OVERFLOW/RANGE: fuse_ver >= num_sigs is refused.
  uint8_t *img = NULL; size_t n = 0;
  CHECK(tinynv_boot610_btl1_image(&b, f.container, 2, &img, &n) != 0, "fuse_ver past the signature count passed");
  CHECK(img == NULL, "a refused patch left an image allocated");
  free_fix(&f);

  // fuse_version derivation: highest set bit of the low 16 bits, plus one.
  CHECK(tinynv_boot610_fuse_version(0) == 0, "fuse_version(0)");
  CHECK(tinynv_boot610_fuse_version(1) == 1, "fuse_version(1)");
  CHECK(tinynv_boot610_fuse_version(2) == 2, "fuse_version(2)");
  CHECK(tinynv_boot610_fuse_version(3) == 2, "fuse_version(3)");
  CHECK(tinynv_boot610_fuse_version(0x8000) == 16, "fuse_version(0x8000)");
  CHECK(tinynv_boot610_fuse_version(0xffff0000u) == 0, "fuse_version ignores the high bits");
  printf("btl1: fuse selects index num_sigs-1-fuse_ver; out-of-range fuse refused; fuse_version derived\n");
}

// ---- the 84-byte v5 boot descriptor ------------------------------------------------------------------------------
static void desc_parse(void) {
  // A valid descriptor for a 36864-byte image, mirroring the pinned AD102 wire form.
  uint32_t image_size = 36864;
  uint8_t d[84]; memset(d, 0, sizeof d);
  wr32(d + 0, 5);              // version
  wr32(d + 4, 32768); wr32(d + 8, 2176);    // bootloader off/size
  wr32(d + 12, 34944); wr32(d + 16, 16);    // params off/size
  wr32(d + 32, 0); wr32(d + 36, 2048);      // manifest off/size
  wr32(d + 40, 2048); wr32(d + 44, 16384);  // monitor data off/size
  wr32(d + 48, 18432); wr32(d + 52, 10496); // monitor code off/size
  wr32(d + 56, 1);             // bIsMonitorEnabled
  wr32(d + 76, image_size);    // fbReservedSize == image size
  wr32(d + 80, 0);             // bSignedAsCode
  tinynv_boot610_desc_t out;
  CHECK(tinynv_boot610_desc_parse(d, sizeof d, image_size, &out) == 0, "valid descriptor refused: %s", tinynv_last_error());
  CHECK(out.manifest_offset == 0 && out.data_offset == 2048 && out.code_offset == 18432, "descriptor offsets wrong");
  CHECK(out.image_size == image_size, "descriptor image size wrong");

  // WRONG-PROFILE: the 92-byte 610 struct form is NOT the wire form; it must be refused.
  uint8_t d92[92]; memset(d92, 0, sizeof d92); memcpy(d92, d, 84);
  CHECK(tinynv_boot610_desc_parse(d92, sizeof d92, image_size, &out) != 0, "a 92-byte descriptor passed as the 84-byte wire form");
  // Wrong size (83).
  CHECK(tinynv_boot610_desc_parse(d, 83, image_size, &out) != 0, "an 83-byte descriptor passed");
  // Wrong version.
  uint8_t bad[84]; memcpy(bad, d, 84); wr32(bad, 4);
  CHECK(tinynv_boot610_desc_parse(bad, sizeof bad, image_size, &out) != 0, "a v4 descriptor passed");
  // RANGE: monitor code runs past the image.
  memcpy(bad, d, 84); wr32(bad + 48, image_size - 16); wr32(bad + 52, 10496);
  CHECK(tinynv_boot610_desc_parse(bad, sizeof bad, image_size, &out) != 0, "a monitor-code range past the image passed");
  // OVERFLOW: a range whose off+size wraps.
  memcpy(bad, d, 84); wr32(bad + 40, 0xfffffff0u); wr32(bad + 44, 16384);
  CHECK(tinynv_boot610_desc_parse(bad, sizeof bad, image_size, &out) != 0, "a wrapping monitor-data range passed");
  // Feature field nonzero (riscvElfOffset).
  memcpy(bad, d, 84); wr32(bad + 20, 1);
  CHECK(tinynv_boot610_desc_parse(bad, sizeof bad, image_size, &out) != 0, "a nonzero riscvElf feature passed");
  // image-size mismatch.
  CHECK(tinynv_boot610_desc_parse(d, sizeof d, image_size + 1, &out) != 0, "a descriptor/image size mismatch passed");
  printf("descriptor: 84-byte v5 wire form parsed; 92-byte/wrong-version/range/overflow/feature refused\n");
}

// ---- the 610 startup args ----------------------------------------------------------------------------------------
static void args_fill(void) {
  tinynv_boot610_args_cached_t a;
  tinynv_boot610_fill_args_cached(&a, 0x1234567890ull, 130, 0x2000, 0x42000);
  CHECK(sizeof a == 152, "args cached is not 152 bytes");
  CHECK(a.messageQueueInitArguments.sharedMemPhysAddr == 0x1234567890ull, "sharedMemPhysAddr wrong");
  CHECK(a.messageQueueInitArguments.pageTableEntryCount == 130, "pageTableEntryCount wrong");
  CHECK(a.messageQueueInitArguments.cmdQueueOffset == 0x2000, "cmdQueueOffset wrong");
  CHECK(a.messageQueueInitArguments.statQueueOffset == 0x42000, "statQueueOffset wrong");
  CHECK(a.messageQueueInitArguments.queueElementHdrSize == 16, "queueElementHdrSize wrong");
  CHECK(a.messageQueueInitArguments.queueElementSizeMin == 4096, "queueElementSizeMin wrong");
  CHECK(a.messageQueueInitArguments.queueElementSizeMax == 65536, "queueElementSizeMax wrong");
  CHECK(a.messageQueueInitArguments.queueHeaderAlign == 4, "queueHeaderAlign wrong");
  CHECK(a.messageQueueInitArguments.queueElementAlign == 12, "queueElementAlign wrong");
  CHECK(a.bDmemStack == 1, "bDmemStack not set");
  // Everything the boot path does not set is explicitly zero (the new arg blocks, srInitArguments, gpuInstance).
  CHECK(a.gpuInstance == 0 && a.profilerArgs[0] == 0 && a.sysmemHeapArgs[0] == 0 &&
        a.rmStateMonitorBufferArgs[0] == 0 && a.bindataArgs[0] == 0, "an unset arg block is nonzero");
  for (size_t i = 0; i < sizeof a.srInitArguments; i++) CHECK(a.srInitArguments[i] == 0, "srInitArguments nonzero");
  // FIELD: the wire offsets match the static assertions (serialize and read back at the 610 offsets).
  uint8_t wire[152]; memcpy(wire, &a, sizeof a);
  CHECK(rd32(wire + 76) == 0, "gpuInstance not at 76");
  CHECK(wire[80] == 1, "bDmemStack not at 80");
  printf("args: 152-byte cached args filled at the 610 offsets, queue geometry set, tail zeroed\n");
}

// ---- the 610 vbios carveout --------------------------------------------------------------------------------------
static void layout_valid(void) {
  // The RTX 4090's usable FB (~32607 MiB) with the real pinned image and bootloader sizes.
  uint64_t vram = 32607ull << 20;
  tinynv_boot610_layout_t l;
  CHECK(tinynv_boot610_vbios_layout(vram, TINYNV_FW610_IMAGE_SIZE, TINYNV_FW610_BOOTBIN_SIZE, &l) == 0,
        "the real 610 layout refused: %s", tinynv_last_error());
  // Boundaries strictly ordered, top down.
  CHECK(l.nonwpr_offset < l.wpr_start && l.wpr_start < l.heap_offset && l.heap_offset < l.gsp_offset &&
        l.gsp_offset < l.boot_offset && l.boot_offset < l.frts_offset && l.frts_offset < l.wpr_end && l.wpr_end <= vram,
        "the 610 layout boundaries are not ordered");
  // The whole carveout fits the reservation, and the manager (which stops at vram - reserve_top) is below the non-WPR heap.
  CHECK(vram - l.nonwpr_offset <= TINYNV_FW610_RESERVE_TOP, "the 610 carveout exceeds the reservation");
  CHECK(vram - TINYNV_FW610_RESERVE_TOP < l.nonwpr_offset, "the manager's top is not below the firmware's non-WPR heap");
  // The image fits between gsp_offset and boot_offset; the boot binary between boot_offset and frts_offset.
  CHECK(l.boot_offset - l.gsp_offset >= TINYNV_FW610_IMAGE_SIZE, "the image does not fit its slot");
  CHECK(l.frts_offset - l.boot_offset >= TINYNV_FW610_BOOTBIN_SIZE, "the boot binary does not fit its slot");
  CHECK(l.nonwpr_size == TINYNV_FW610_NONWPR_HEAP, "the non-WPR heap is not the matching AD102 value");

  // Fill the WPR meta and check the fields the firmware reads.
  tinynv_boot610_desc_t desc = {.manifest_offset = 0, .data_offset = 2048, .code_offset = 18432, .image_size = 36864};
  tinynv_wpr_meta_t m;
  tinynv_boot610_fill_wpr_meta(&m, &l, 0xaaaa000, TINYNV_FW610_IMAGE_SIZE, 0xbbbb000, 36864, &desc, 0xcccc000);
  CHECK(m.magic == TINYNV_GSP_FW_WPR_META_MAGIC && m.revision == 1, "wpr meta magic/revision wrong");
  CHECK(m.sysmemAddrOfRadix3Elf == 0xaaaa000 && m.sizeOfRadix3Elf == TINYNV_FW610_IMAGE_SIZE, "wpr meta radix wrong");
  CHECK(m.sysmemAddrOfBootloader == 0xbbbb000 && m.sizeOfBootloader == 36864, "wpr meta bootloader wrong");
  CHECK(m.bootloaderCodeOffset == 18432 && m.bootloaderDataOffset == 2048 && m.bootloaderManifestOffset == 0,
        "wpr meta descriptor offsets wrong");
  CHECK(m.nonWprHeapOffset == l.nonwpr_offset && m.gspFwWprStart == l.wpr_start && m.fbSize == vram, "wpr meta layout wrong");
  printf("layout: the real 610 image/boot fit, boundaries ordered, reservation covers, wpr meta filled\n");
}

static void layout_refusals(void) {
  tinynv_boot610_layout_t l;
  // OVERFLOW: a tiny FB cannot hold the carveout - every underflow is refused, not wrapped.
  CHECK(tinynv_boot610_vbios_layout(1ull << 20, TINYNV_FW610_IMAGE_SIZE, TINYNV_FW610_BOOTBIN_SIZE, &l) != 0,
        "a 1 MiB FB passed");
  CHECK(tinynv_boot610_vbios_layout(64ull << 20, TINYNV_FW610_IMAGE_SIZE, TINYNV_FW610_BOOTBIN_SIZE, &l) != 0,
        "a 64 MiB FB passed");
  // Zero inputs refused.
  CHECK(tinynv_boot610_vbios_layout(0, 1, 1, &l) != 0, "a zero FB passed");
  CHECK(tinynv_boot610_vbios_layout(32607ull << 20, 0, 1, &l) != 0, "a zero image passed");
  CHECK(tinynv_boot610_vbios_layout(32607ull << 20, 1, 0, &l) != 0, "a zero boot passed");
  // OVERFLOW: an image past the load bound is refused (the 570 64 MiB bound would have refused the real image; the 610
  // bound accepts it, but something larger still must not pass).
  CHECK(tinynv_boot610_vbios_layout(32607ull << 20, TINYNV_FW610_IMAGE_BOUND + 1, TINYNV_FW610_BOOTBIN_SIZE, &l) != 0,
        "an image past the 610 load bound passed");
  CHECK(tinynv_boot610_vbios_layout(32607ull << 20, TINYNV_FW610_IMAGE_SIZE, TINYNV_FW610_BOOTBIN_BOUND + 1, &l) != 0,
        "a boot binary past the bound passed");
  // A FB just large enough to underflow the non-WPR heap but not the earlier regions: build a FB where the carveout
  // reaches below zero. 200 MiB is enough for vga+frts+boot+image but not the heap.
  CHECK(tinynv_boot610_vbios_layout(200ull << 20, TINYNV_FW610_IMAGE_SIZE, TINYNV_FW610_BOOTBIN_SIZE, &l) != 0,
        "a 200 MiB FB (too small for the heap) passed");
  printf("layout: underflow/zero/oversize/too-small-FB fixtures all refused\n");
}

// ---- the post-boot WPR/non-WPR carveout check, with the 610 profile ----------------------------------------------
// The spec requires the placed WPR/non-WPR bounds to be checked against the allocator after boot. That is
// tinynv_mm_check_fw_carveout, now profile-routed; here it is driven with the 610 policy and the boundaries the 610
// layout actually places, plus the nonsense/overlap cases it must refuse. No GPU: the wpr2 values are synthetic.
static void carveout_610(void) {
  uint64_t vram = 32607ull << 20;
  tinynv_boot610_layout_t l;
  CHECK(tinynv_boot610_vbios_layout(vram, TINYNV_FW610_IMAGE_SIZE, TINYNV_FW610_BOOTBIN_SIZE, &l) == 0,
        "carveout: layout failed");
  // A hand-built manager that stops where the 610 reservation says (vram - reserve_top), as tinynv_mm_init would carve.
  tinynv_dev_t dev; memset(&dev, 0, sizeof dev);
  dev.vram_size = vram;
  dev.runtime = &tinynv_runtime_610;
  tinynv_mm_t mm; memset(&mm, 0, sizeof mm);
  mm.dev = &dev;
  mm.pa.base = 0x4200000ull;
  mm.pa.size = (vram - TINYNV_FW610_RESERVE_TOP) - mm.pa.base;
  char line[256];
  // The placed WPR2 region (wpr_start .. wpr_end) with the manager below the non-WPR heap: accepted.
  CHECK(tinynv_mm_check_fw_carveout(&mm, l.wpr_start, l.wpr_end, line, sizeof line) == 0,
        "the 610 carveout check refused the layout it placed: %s", tinynv_last_error());
  // The manager's top pushed up into the WPR region (above the non-WPR heap): refused.
  mm.pa.size = l.wpr_start - mm.pa.base;
  CHECK(tinynv_mm_check_fw_carveout(&mm, l.wpr_start, l.wpr_end, line, sizeof line) != 0,
        "the 610 carveout check accepted a manager inside the firmware's non-WPR heap");
  mm.pa.size = (vram - TINYNV_FW610_RESERVE_TOP) - mm.pa.base;
  // Nonsense registers refused: a tiny region, a region past FB, an inverted region.
  CHECK(tinynv_mm_check_fw_carveout(&mm, vram - 30 * (1ull << 20), vram - 25 * (1ull << 20), line, sizeof line) != 0,
        "a 5 MB wpr2 was believed under the 610 profile");
  CHECK(tinynv_mm_check_fw_carveout(&mm, vram - 900 * (1ull << 20), vram, line, sizeof line) != 0,
        "a 900 MB wpr2 was believed under the 610 profile");
  CHECK(tinynv_mm_check_fw_carveout(&mm, l.wpr_end, l.wpr_start, line, sizeof line) != 0,
        "an inverted wpr2 was believed under the 610 profile");
  // Both registers zero: the chip exposes no region, reported as unverified rather than refused (the 570 contract).
  CHECK(tinynv_mm_check_fw_carveout(&mm, 0, 0, line, sizeof line) == 0 && strstr(line, "unverified") != NULL,
        "a hidden wpr2 was not reported as unverified");
  printf("carveout: the 610 profile's placed WPR/non-WPR bounds pass; overlap/nonsense refused; hidden region unverified\n");
}

// ---- the real hash-pinned assets, when the bundle is present -----------------------------------------------------
static void real_assets(const char *dir) {
  char path[1024];
  const char *names[] = {"booter_load", "booter_unload", "scrubber"};
  for (size_t k = 0; k < 3; k++) {
    snprintf(path, sizeof path, "%s/nvidia/ad102/gsp-610.43.03/%s-610.43.03.bin", dir, names[k]);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("  FAIL: cannot open %s\n", path); fails++; continue; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(n); if (fread(buf, 1, n, f) != (size_t)n) { printf("  FAIL: short read %s\n", path); fails++; }
    fclose(f);
    uint32_t image_size; const tinynv_btl1_hdr_t *g = tinynv_boot610_geom(names[k], &image_size);
    tinynv_btl1_t b;
    CHECK(tinynv_boot610_btl1_parse(buf, n, g, image_size, &b) == 0, "real %s refused: %s", names[k], tinynv_last_error());
    // Patch with fuse 0 and 1 (both valid for num_sigs=2); the image must come out the pinned size.
    for (uint32_t fuse = 0; fuse < 2; fuse++) {
      uint8_t *img = NULL; size_t len = 0;
      CHECK(tinynv_boot610_btl1_image(&b, buf, fuse, &img, &len) == 0, "real %s patch fuse %u failed", names[k], fuse);
      CHECK(len == image_size, "real %s patched size wrong", names[k]);
      free(img);
    }
    free(buf);
  }
  // The real 84-byte descriptor against the real bootloader image.
  snprintf(path, sizeof path, "%s/nvidia/ad102/gsp-610.43.03/bootloader-610.43.03.bin", dir);
  FILE *f = fopen(path, "rb");
  if (f) {
    fseek(f, 0, SEEK_END); long bln = ftell(f); fclose(f);
    snprintf(path, sizeof path, "%s/nvidia/ad102/gsp-610.43.03/bootloader_desc-610.43.03.bin", dir);
    f = fopen(path, "rb");
    if (f) {
      uint8_t d[84]; size_t got = fread(d, 1, sizeof d, f); fclose(f);
      tinynv_boot610_desc_t out;
      CHECK(got == 84 && tinynv_boot610_desc_parse(d, got, (uint32_t)bln, &out) == 0,
            "the real descriptor refused: %s", tinynv_last_error());
      CHECK(bln == TINYNV_FW610_BOOTBIN_SIZE, "the real bootloader is not the pinned size");
    }
  }
  // The real GSP ELF: the shared section reader must find .fwimage at the pinned size (inside the 610 load bound)
  // and the AD10x signature section. This is the "610 ELF handling" the reservation was re-sized for.
  snprintf(path, sizeof path, "%s/nvidia/ad102/gsp-610.43.03/gsp-610.43.03.bin", dir);
  f = fopen(path, "rb");
  if (f) {
    fseek(f, 0, SEEK_END); long en = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *eb = malloc(en);
    if (fread(eb, 1, en, f) != (size_t)en) { printf("  FAIL: short read %s\n", path); fails++; }
    fclose(f);
    tinynv_blob_t blob = {.data = eb, .size = (size_t)en};
    const uint8_t *img = NULL, *sig = NULL; size_t img_len = 0, sig_len = 0;
    CHECK(tinynv_elf_section(&blob, ".fwimage", &img, &img_len) == 0, "the real 610 ELF has no .fwimage");
    CHECK(img_len == TINYNV_FW610_IMAGE_SIZE, "the real .fwimage is %zu, not the pinned %llu", img_len,
          (unsigned long long)TINYNV_FW610_IMAGE_SIZE);
    CHECK(img_len <= TINYNV_FW610_IMAGE_BOUND, "the real .fwimage exceeds the 610 load bound");
    CHECK(tinynv_elf_section(&blob, ".fwsignature_ad10x", &sig, &sig_len) == 0 && sig_len == 0x1000,
          "the real 610 ELF AD10x signature section is missing or not 4096 bytes");
    free(eb);
  }
  printf("real assets: the prepared bundle's booters/descriptor parse and patch; the GSP ELF .fwimage fits the 610 bound\n");
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  btl1_valid();
  btl1_refusals();
  btl1_patch();
  desc_parse();
  args_fill();
  layout_valid();
  layout_refusals();
  carveout_610();
  if (argc > 1) real_assets(argv[1]);
  else printf("real assets: SKIPPED - no bundle dir given (pass cuda-shim/build/gsp610-04/bundle); an environmental gap\n");
  printf(fails ? "boot610: %d checks FAILED\n" : "boot610: all checks passed\n", fails);
  return fails ? 1 : 0;
}
