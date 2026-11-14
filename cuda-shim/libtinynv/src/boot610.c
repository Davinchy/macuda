// The 610.43.03 boot-metadata adapters. See boot610.h for the contract and the audit citations. Pure functions: no
// device, no GPU, no allocation except the one image copy the caller frees. Every refusal goes through tinynv_fail so
// the offline suite can assert on the reason, and every range/size computation is overflow-checked rather than wrapped.
#include "boot610.h"
#include "fw_layout_610.h"
#include "internal.h"
#include <stdlib.h>
#include <string.h>

#define PAGE 0x1000ull

// --- little-endian readers that stay inside the buffer -----------------------------------------------------------
static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// --- the pinned AD102 geometries (BTL1_GEOMETRY, from the hash-verified bindata) -------------------------
static const tinynv_btl1_hdr_t GEOM_LOAD = {
  .os_code_off = 0, .os_code_size = 256, .os_data_off = 32256, .os_data_size = 24576,
  .num_apps = 1, .app_code_off = 256, .app_code_size = 32000, .app_data_off = 256, .app_data_size = 0,
  .patch_loc = 32272, .ucode_id = 3, .engine_id = 1, .num_sigs = 2, .sig_bytes = 768,
};
static const tinynv_btl1_hdr_t GEOM_UNLOAD = {
  .os_code_off = 0, .os_code_size = 256, .os_data_off = 20480, .os_data_size = 19968,
  .num_apps = 1, .app_code_off = 256, .app_code_size = 20224, .app_data_off = 256, .app_data_size = 0,
  .patch_loc = 20496, .ucode_id = 3, .engine_id = 1, .num_sigs = 2, .sig_bytes = 768,
};
static const tinynv_btl1_hdr_t GEOM_SCRUB = {
  .os_code_off = 0, .os_code_size = 256, .os_data_off = 6656, .os_data_size = 768,
  .num_apps = 1, .app_code_off = 256, .app_code_size = 6400, .app_data_off = 256, .app_data_size = 0,
  .patch_loc = 6656, .ucode_id = 13, .engine_id = 1, .num_sigs = 2, .sig_bytes = 768,
};

const tinynv_btl1_hdr_t *tinynv_boot610_geom(const char *which, uint32_t *image_size_out) {
  if (!strcmp(which, "booter_load")) { *image_size_out = 56832; return &GEOM_LOAD; }
  if (!strcmp(which, "booter_unload")) { *image_size_out = 40704; return &GEOM_UNLOAD; }
  if (!strcmp(which, "scrubber")) { *image_size_out = 7424; return &GEOM_SCRUB; }
  return NULL;
}

// One 256-aligned falcon transfer range inside the image. `size` must be nonzero; off+size is computed in 64 bits so a
// hostile header cannot wrap it past the image.
static int range_ok(uint32_t off, uint32_t size, uint32_t image_size, int allow_empty) {
  if (!size) return allow_empty;
  if (off % 256 || size % 256) return 0;
  return (uint64_t)off + size <= image_size;
}

int tinynv_boot610_btl1_parse(const uint8_t *data, size_t len, const tinynv_btl1_hdr_t *expect,
                              uint32_t expect_image_size, tinynv_btl1_t *out) {
  memset(out, 0, sizeof *out);
  if (!data || len < TINYNV_BTL1_PREFIX)
    return tinynv_fail("the 610 booter container is truncated (%zu bytes, header needs %u)", len, TINYNV_BTL1_PREFIX);
  uint32_t magic = rd32(data), version = rd32(data + 4);
  if (magic == 0x000010deu)
    return tinynv_fail("the 610 booter is a 570 NVFW image (magic 0x10de): the 610 profile needs the BTL1 container, "
                       "not a renamed 570 file");
  if (magic != TINYNV_BTL1_MAGIC)
    return tinynv_fail("the 610 booter is not a BTL1 container (magic %#x)", magic);
  if (version != TINYNV_BTL1_VERSION)
    return tinynv_fail("unsupported BTL1 version %u (this build parses v1)", version);

  tinynv_btl1_hdr_t h;
  uint32_t *w = (uint32_t *)(void *)&h;
  for (int i = 0; i < 14; i++) w[i] = rd32(data + 8 + (size_t)i * 4);
  uint32_t image_size = rd32(data + 64), sig_bytes2 = rd32(data + 68);
  if (h.sig_bytes != sig_bytes2)
    return tinynv_fail("the BTL1 signature size fields disagree (%u vs %u)", h.sig_bytes, sig_bytes2);
  uint64_t total = (uint64_t)TINYNV_BTL1_PREFIX + image_size + h.sig_bytes;
  if ((uint64_t)len != total)
    return tinynv_fail("the BTL1 file is %zu bytes, its header declares %llu (truncated or trailing data)",
                       len, (unsigned long long)total);
  if (h.num_apps != 1)
    return tinynv_fail("the BTL1 container declares %u apps, the 610 booter path requires 1", h.num_apps);
  if (h.num_sigs < 1 || h.num_sigs > 8)
    return tinynv_fail("implausible BTL1 signature count %u", h.num_sigs);
  if (h.sig_bytes != h.num_sigs * TINYNV_BTL1_SIG_SIZE)
    return tinynv_fail("BTL1 signature bytes %u != %u x %u", h.sig_bytes, h.num_sigs, TINYNV_BTL1_SIG_SIZE);
  if (!image_size || image_size > (128u << 20))
    return tinynv_fail("implausible BTL1 image size %u", image_size);
  if (!range_ok(h.os_code_off, h.os_code_size, image_size, 0) ||
      !range_ok(h.os_data_off, h.os_data_size, image_size, 0) ||
      !range_ok(h.app_code_off, h.app_code_size, image_size, 0) ||
      !range_ok(h.app_data_off, h.app_data_size, image_size, 1))
    return tinynv_fail("a BTL1 code/data range is misaligned or outside the %u byte image", image_size);
  if (!h.ucode_id || !h.engine_id)
    return tinynv_fail("the BTL1 container has a zero ucode/engine id");
  // The signature is patched into DMEM: patch_loc must sit inside os_data with room for one full signature.
  if (h.patch_loc < h.os_data_off ||
      (uint64_t)h.patch_loc + TINYNV_BTL1_SIG_SIZE > (uint64_t)h.os_data_off + h.os_data_size)
    return tinynv_fail("the BTL1 signature patch location %#x is outside the DMEM range %#x+%#x",
                       h.patch_loc, h.os_data_off, h.os_data_size);

  if (expect) {
    if (memcmp(&h, expect, sizeof h) || image_size != expect_image_size)
      return tinynv_fail("the BTL1 geometry does not match the pinned AD102 asset (wrong chip or wrong image for the "
                         "610 profile)");
  }

  out->hdr = h;
  out->image_size = image_size;
  out->sig_bytes = h.sig_bytes;
  out->sig_dmem_off = h.patch_loc - h.os_data_off;
  out->image_off = TINYNV_BTL1_PREFIX;
  out->sig_off = TINYNV_BTL1_PREFIX + image_size;
  return 0;
}

uint32_t tinynv_boot610_fuse_version(uint32_t raw) {
  uint32_t index = 0, value = raw & 0xffffu;
  if (!value) return 0;
  while (value > 1) { value >>= 1; index++; }
  return index + 1;
}

int tinynv_boot610_btl1_image(const tinynv_btl1_t *b, const uint8_t *data, uint32_t fuse_ver,
                              uint8_t **image_out, size_t *len_out) {
  *image_out = NULL; *len_out = 0;
  uint32_t sig_size = b->sig_bytes / b->hdr.num_sigs;
  uint32_t index = 0;
  if (b->hdr.num_sigs > 1) {
    if (fuse_ver >= b->hdr.num_sigs)
      return tinynv_fail("the SEC2 fuse version %u is past the %u signatures the booter carries", fuse_ver, b->hdr.num_sigs);
    index = b->hdr.num_sigs - 1 - fuse_ver;
  }
  uint8_t *image = malloc(b->image_size);
  if (!image) return tinynv_fail("no memory for the %u byte 610 booter image", b->image_size);
  memcpy(image, data + b->image_off, b->image_size);
  memcpy(image + b->hdr.patch_loc, data + b->sig_off + (size_t)index * sig_size, sig_size);
  *image_out = image;
  *len_out = b->image_size;
  return 0;
}

// --- the 84-byte v5 boot descriptor ------------------------------------------------------------------------------
int tinynv_boot610_desc_parse(const uint8_t *desc, size_t len, uint32_t image_size, tinynv_boot610_desc_t *out) {
  memset(out, 0, sizeof *out);
  if (!desc || len != TINYNV_BOOT610_DESC_SIZE)
    return tinynv_fail("the 610 boot descriptor is %zu bytes, the AD102 v5 wire form is %u", len, TINYNV_BOOT610_DESC_SIZE);
  if (!image_size) return tinynv_fail("the 610 boot descriptor names an empty bootloader image");
  if (rd32(desc) != 5)
    return tinynv_fail("the 610 boot descriptor version is %u, not 5", rd32(desc));
  if (rd32(desc + 56) != 1 || rd32(desc + 76) != image_size || rd32(desc + 80) != 0)
    return tinynv_fail("the 610 boot descriptor flags/image-size disagree with the %u byte bootloader", image_size);
  // The ranges the boot path reads: bootloader, params, manifest, monitor data, monitor code.
  static const unsigned ranges[] = {1, 3, 8, 10, 12};
  for (unsigned i = 0; i < sizeof(ranges) / sizeof(*ranges); i++) {
    uint32_t off = rd32(desc + ranges[i] * 4), size = rd32(desc + (ranges[i] + 1) * 4);
    if (!size || off > image_size || size > image_size - off)
      return tinynv_fail("the 610 boot descriptor range [%u] %#x+%#x is outside the %u byte image", ranges[i], off, size,
                         image_size);
  }
  // This path boots no secure partition, loads no ITCM and enables no SMP: those feature fields must be zero.
  for (unsigned i = 5; i <= 7; i++)
    if (rd32(desc + i * 4)) return tinynv_fail("the 610 boot descriptor enables an unsupported feature (field %u)", i);
  for (unsigned i = 15; i <= 18; i++)
    if (rd32(desc + i * 4)) return tinynv_fail("the 610 boot descriptor enables an unsupported feature (field %u)", i);
  out->manifest_offset = rd32(desc + 32); out->manifest_size = rd32(desc + 36);
  out->data_offset = rd32(desc + 40); out->data_size = rd32(desc + 44);
  out->code_offset = rd32(desc + 48); out->code_size = rd32(desc + 52);
  out->image_size = image_size;
  return 0;
}

// --- the 610 startup args ----------------------------------------------------------------------------------------
void tinynv_boot610_fill_args_cached(tinynv_boot610_args_cached_t *a, uint64_t shared_mem_pa, uint32_t pte_count,
                                     uint64_t cmd_off, uint64_t stat_off) {
  memset(a, 0, sizeof *a);
  a->bDmemStack = 1;
  a->messageQueueInitArguments.sharedMemPhysAddr = shared_mem_pa;
  a->messageQueueInitArguments.pageTableEntryCount = pte_count;
  a->messageQueueInitArguments.cmdQueueOffset = cmd_off;
  a->messageQueueInitArguments.statQueueOffset = stat_off;
  // The five fields 610 added (message_queue_cpu.c defaults, no Confidential Compute). The element header is the
  // 16-byte MCTP/NVDM prefix; the element codec itself is , but the boot args must name its geometry now.
  a->messageQueueInitArguments.queueElementHdrSize = TINYNV_BOOT610_ELEM_HDR_SIZE;
  a->messageQueueInitArguments.queueElementSizeMin = TINYNV_BOOT610_ELEM_SIZE_MIN;
  a->messageQueueInitArguments.queueElementSizeMax = TINYNV_BOOT610_ELEM_SIZE_MAX;
  a->messageQueueInitArguments.queueHeaderAlign = TINYNV_BOOT610_HDR_ALIGN;
  a->messageQueueInitArguments.queueElementAlign = TINYNV_BOOT610_ELEM_ALIGN;
}

// --- the 610 vbios carveout --------------------------------------------------------------------------------------
static uint64_t round_down(uint64_t v, uint64_t a) { return v / a * a; }

// Subtract with an underflow check: a layout that would wrap is refused, not computed.
static int sub(uint64_t v, uint64_t amount, uint64_t *out, const char *what) {
  if (v < amount) return tinynv_fail("the 610 carveout does not fit: %s would underflow (%#llx < %#llx)", what,
                                     (unsigned long long)v, (unsigned long long)amount);
  *out = v - amount;
  return 0;
}

int tinynv_boot610_vbios_layout(uint64_t vram, uint64_t image_size, uint64_t boot_size, tinynv_boot610_layout_t *out) {
  memset(out, 0, sizeof *out);
  if (!vram || !image_size || !boot_size)
    return tinynv_fail("the 610 carveout needs a nonzero FB, image and boot size");
  if (image_size > TINYNV_FW610_IMAGE_BOUND)
    return tinynv_fail("the 610 GSP image is %llu bytes, past the %llu MB load bound", (unsigned long long)image_size,
                       (unsigned long long)(TINYNV_FW610_IMAGE_BOUND >> 20));
  if (boot_size > TINYNV_FW610_BOOTBIN_BOUND)
    return tinynv_fail("the 610 bootloader is %llu bytes, past the %llu MB load bound", (unsigned long long)boot_size,
                       (unsigned long long)(TINYNV_FW610_BOOTBIN_BOUND >> 20));

  tinynv_boot610_layout_t l;
  memset(&l, 0, sizeof l);
  l.fb_size = vram;
  l.vga_size = TINYNV_FW610_VGA_SIZE;
  if (sub(vram, l.vga_size, &l.vga_offset, "the VGA workspace")) return -1;
  l.frts_size = TINYNV_FW610_FRTS_SIZE;
  if (sub(l.vga_offset, l.frts_size, &l.frts_offset, "the FRTS region")) return -1;
  if (sub(l.frts_offset, boot_size, &l.boot_offset, "the boot binary")) return -1;
  uint64_t t;
  if (sub(l.boot_offset, image_size, &t, "the GSP image")) return -1;
  l.gsp_offset = round_down(t, 0x10000);
  if (sub(l.gsp_offset, TINYNV_FW610_VBIOS_HEAP, &t, "the firmware heap")) return -1;
  l.heap_offset = round_down(t, 0x100000);
  l.heap_size = TINYNV_FW610_VBIOS_HEAP;
  if (sub(l.heap_offset, PAGE, &t, "the LSB/meta page")) return -1;
  l.wpr_start = round_down(t, 0x100000);
  l.nonwpr_size = TINYNV_FW610_NONWPR_HEAP;
  if (sub(l.wpr_start, l.nonwpr_size, &t, "the non-WPR heap")) return -1;
  l.nonwpr_offset = round_down(t, 0x100000);
  l.wpr_end = l.vga_offset;

  // The boundaries must be strictly ordered and inside FB, and the whole carveout must fit the reservation. A layout
  // that passes the subtractions but collapses two regions is still refused here rather than handed to the firmware.
  if (l.nonwpr_offset >= l.wpr_start || l.wpr_start >= l.heap_offset || l.heap_offset >= l.gsp_offset ||
      l.gsp_offset >= l.boot_offset || l.boot_offset >= l.frts_offset || l.frts_offset >= l.wpr_end ||
      l.wpr_end > vram)
    return tinynv_fail("the 610 carveout boundaries are not strictly ordered inside FB");
  if (vram - l.nonwpr_offset > TINYNV_FW610_RESERVE_TOP)
    return tinynv_fail("the 610 carveout reaches %llu MB down from the top, past the %llu MB reservation",
                       (unsigned long long)((vram - l.nonwpr_offset) >> 20),
                       (unsigned long long)(TINYNV_FW610_RESERVE_TOP >> 20));
  *out = l;
  return 0;
}

void tinynv_boot610_fill_wpr_meta(tinynv_wpr_meta_t *m, const tinynv_boot610_layout_t *l, uint64_t radix3_pa,
                                  uint64_t image_size, uint64_t bootloader_pa, uint64_t bootloader_size,
                                  const tinynv_boot610_desc_t *desc, uint64_t signature_pa) {
  memset(m, 0, sizeof *m);
  m->magic = TINYNV_GSP_FW_WPR_META_MAGIC;
  m->revision = TINYNV_GSP_FW_WPR_META_REVISION;
  m->sysmemAddrOfRadix3Elf = radix3_pa;
  m->sizeOfRadix3Elf = image_size;
  m->sysmemAddrOfBootloader = bootloader_pa;
  m->sizeOfBootloader = bootloader_size;
  m->sysmemAddrOfSignature = signature_pa;
  m->sizeOfSignature = 0x1000;
  m->bootloaderCodeOffset = desc->code_offset;
  m->bootloaderDataOffset = desc->data_offset;
  m->bootloaderManifestOffset = desc->manifest_offset;
  m->vgaWorkspaceSize = l->vga_size;
  m->vgaWorkspaceOffset = l->vga_offset;
  m->gspFwWprEnd = l->wpr_end;
  m->frtsSize = l->frts_size;
  m->frtsOffset = l->frts_offset;
  m->bootBinOffset = l->boot_offset;
  m->gspFwOffset = l->gsp_offset;
  m->gspFwHeapSize = l->heap_size;
  m->gspFwHeapOffset = l->heap_offset;
  m->gspFwWprStart = l->wpr_start;
  m->nonWprHeapSize = l->nonwpr_size;
  m->nonWprHeapOffset = l->nonwpr_offset;
  m->gspFwRsvdStart = l->nonwpr_offset;
  m->fbSize = l->fb_size;
}
