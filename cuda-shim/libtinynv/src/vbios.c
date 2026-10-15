// Finding FWSEC in the card's own VBIOS, and patching it to place the write-protected region.
//
// An Ampere card is not released from lockdown by a separate security processor the way a Blackwell one is. Instead the
// driver runs a signed image that is already ON the card - FWSEC, which NVIDIA ships inside the VBIOS - and asks it to
// place the firmware's write-protected region (FRTS). Only then can booter_load unpack GSP-RM. So this file's job is to
// read a megabyte of ROM, walk three nested tables to find that image, and rewrite four fields in it.
//
// Every structure here is packed and none of the offsets are naturally aligned, which is why they are read with memcpy
// into the generated declarations in vbios_structs.h rather than by casting a pointer into the buffer.
//
// The oracle (tinygrad's NV_FLCN.prep_ucode) does not bounds-check the walk, because a VBIOS that fails these reads is
// a card that is not going to boot anyway. It is checked here: the difference between "the ROM does not contain what
// this driver expects" and a read three megabytes past a one-megabyte buffer is the difference between a message and a
// crash, and the walk is driven entirely by numbers that came off the card.
#include "vbios.h"
#include "gpu.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The ROM appears in the register window at this offset, a megabyte of it, and that is where the oracle reads it from -
// not the PCI expansion ROM BAR. The whole megabyte is taken in ONE block read: the recording of a boot contains a
// single operation for it, and reading it in pieces would diverge from that on the first piece.
#define VBIOS_MMIO_BASE 0x00300000u
#define VBIOS_SIZE 0x00100000u
#define BIT_HEADER_OFFSET 0x1b0     // where the BIT table is, which is not discoverable: it is this address
#define BIT_SIGNATURE 0x00544942u   // "BIT\0", checked so a wrong offset is reported rather than walked
#define FWSEC_FRTS_CMD 0x15         // the command id that means "place the write-protected region"
#define FWSEC_SB_CMD 0x19           // and the one that means "put the pre-os applications back", run at unload
#define FWSEC_SIG_BYTES 0x180       // how much of the descriptor's signature block the boot ROM reads

// read a packed structure out of the rom buffer, refusing anything that would run off the end
static int rom_get(const uint8_t *rom, size_t rom_len, size_t off, void *dst, size_t n, const char *what) {
  if (off > rom_len || n > rom_len - off) return tinynv_fail("the vbios %s is at %#zx+%#zx, past the %#x byte rom", what, off, n, (unsigned)rom_len);
  memcpy(dst, rom + off, n);
  return 0;
}

// Walk the PCI expansion ROM chain to find where the extended image begins. Reproduced from the oracle including its
// arithmetic: the base image's length is remembered, and the extended image's offset is measured back from it.
static int find_expansion_rom(const uint8_t *rom, size_t rom_len, size_t *out) {
  size_t off = 0, block_size = 0;
  int saw_base = 0;
  for (int image = 0; image < 64; image++) { // a real rom has a handful; the bound is so a malformed one ends
    uint16_t pci_blck, img_blocks;
    uint8_t code_type;
    if (rom_get(rom, rom_len, off + TINYNV_OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR, &pci_blck, 2, "rom data structure pointer")) return -1;
    if (rom_get(rom, rom_len, off + pci_blck + TINYNV_OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN, &img_blocks, 2, "image length")) return -1;
    if (rom_get(rom, rom_len, off + pci_blck + TINYNV_OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE, &code_type, 1, "code type")) return -1;
    size_t img_len = (size_t)img_blocks * TINYNV_PCI_ROM_IMAGE_BLOCK_SIZE;

    if (code_type == TINYNV_NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE) { block_size = img_len; saw_base = 1; }
    else if (code_type == TINYNV_NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT) {
      // the oracle measures the extended image back from the base's length. reaching this before a base image would
      // subtract a zero it never set, so that is refused rather than producing a plausible offset.
      if (!saw_base) return tinynv_fail("the vbios extended image comes before the base image, so its offset cannot be measured");
      if (block_size > off) return tinynv_fail("the vbios base image is %#zx bytes but the extended one is at %#zx", block_size, off);
      *out = off - block_size;
      return 0;
    }
    if (!img_len) return tinynv_fail("a vbios image declares zero length, so the walk cannot advance");
    off += img_len;
  }
  return tinynv_fail("no extended image in the first 64 images of the vbios");
}

// Find the production FWSEC entry through the BIT table and its falcon ucode table, and return where its descriptor is.
static int find_fwsec_desc(const uint8_t *rom, size_t rom_len, size_t exp_off, size_t *desc_off, size_t *desc_size) {
  tinynv_bit_header_t bit;
  if (rom_get(rom, rom_len, BIT_HEADER_OFFSET, &bit, sizeof bit, "bit header")) return -1;
  if (bit.Signature != BIT_SIGNATURE)
    return tinynv_fail("the bit table at %#x has signature %#x, not %#x: this is not a vbios this driver can read",
                       BIT_HEADER_OFFSET, bit.Signature, BIT_SIGNATURE);

  for (unsigned i = 0; i < bit.TokenEntries; i++) {
    tinynv_bit_token_t tok;
    if (rom_get(rom, rom_len, BIT_HEADER_OFFSET + bit.HeaderSize + (size_t)i * bit.TokenSize, &tok, sizeof tok, "bit token")) return -1;
    if (tok.TokenId != TINYNV_BIT_TOKEN_FALCON_DATA || tok.DataVersion != 2 || tok.DataSize < TINYNV_BIT_DATA_FALCON_DATA_V2_SIZE_4) continue;

    tinynv_bit_falcon_data_t fd;
    if (rom_get(rom, rom_len, tok.DataPtr & 0xffff, &fd, sizeof fd, "falcon data token")) return -1;
    size_t table = exp_off + fd.FalconUcodeTablePtr;

    tinynv_falcon_ucode_table_hdr_t hdr;
    if (rom_get(rom, rom_len, table, &hdr, sizeof hdr, "falcon ucode table header")) return -1;
    for (unsigned j = 0; j < hdr.EntryCount; j++) {
      tinynv_falcon_ucode_table_entry_t ent;
      if (rom_get(rom, rom_len, table + hdr.HeaderSize + (size_t)j * hdr.EntrySize, &ent, sizeof ent, "falcon ucode table entry")) return -1;
      if (ent.ApplicationID != TINYNV_FALCON_UCODE_ENTRY_APPID_FWSEC_PROD) continue;

      tinynv_falcon_ucode_desc_hdr_t dh;
      if (rom_get(rom, rom_len, exp_off + ent.DescPtr, &dh, sizeof dh, "falcon ucode descriptor header")) return -1;
      *desc_off = exp_off + ent.DescPtr;
      *desc_size = dh.vDesc >> 16; // the descriptor's own length, signature block included
      if (*desc_size < TINYNV_FALCON_UCODE_DESC_V3_SIZE_44)
        return tinynv_fail("the fwsec descriptor declares %#zx bytes, less than the %#x byte v3 layout", *desc_size,
                           TINYNV_FALCON_UCODE_DESC_V3_SIZE_44);
      return 0;
    }
  }
  return tinynv_fail("no production fwsec image in the vbios falcon ucode table");
}

// Patch one copy of FWSEC for one command. One image, two commands: FRTS takes the whole command (where the region
// goes); SB takes only its first part, the vbios descriptor, exactly as NVIDIA's kgspPrepareForFwsec builds the two.
// Three writes into the image: the command id the mapper is to run, the command itself, and the signature the boot
// ROM will check. A wrong offset for any of them is a card that refuses the image and says nothing about why.
static int fwsec_patch(uint8_t *image, size_t image_len, const tinynv_falcon_ucode_desc_v3_t *desc, uint32_t cmd_id,
                       uint64_t frts_offset, const uint8_t *sig) {
  // The command, and where in the image's data segment it goes. The region is named in 4K units and the firmware
  // decides its own contents; the driver only says where and how big.
  tinynv_fwsec_frts_cmd_t cmd;
  memset(&cmd, 0, sizeof cmd);
  cmd.readVbiosDesc.version = 1;
  cmd.readVbiosDesc.size = sizeof cmd.readVbiosDesc;
  cmd.readVbiosDesc.flags = 2;
  cmd.frtsRegionDesc.version = 1;
  cmd.frtsRegionDesc.size = sizeof cmd.frtsRegionDesc;
  cmd.frtsRegionDesc.frtsRegionOffset4K = (uint32_t)(frts_offset >> 12);
  cmd.frtsRegionDesc.frtsRegionSize = 0x100;
  cmd.frtsRegionDesc.frtsRegionMediaType = 2;

  // Where the image keeps its interface table, and in it, the entry that says where its command buffer is. Everything
  // is relative to the end of the code segment, because that is where the data segment starts.
  size_t app_hdr_off = desc->IMEMLoadSize + desc->InterfaceOffset;
  tinynv_falcon_app_if_hdr_t app;
  if (app_hdr_off > image_len || sizeof app > image_len - app_hdr_off) return tinynv_fail("the fwsec interface table is outside its own image");
  memcpy(&app, image + app_hdr_off, sizeof app);

  size_t dmem_offset = 0;
  int found_mapper = 0;
  for (unsigned i = 0; i < app.entryCount; i++) {
    tinynv_falcon_app_if_entry_t e;
    size_t eo = app_hdr_off + sizeof app + (size_t)i * sizeof e;
    if (eo > image_len || sizeof e > image_len - eo) return tinynv_fail("a fwsec interface entry is outside its own image");
    memcpy(&e, image + eo, sizeof e);
    if (e.id == TINYNV_FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER) { dmem_offset = e.dmemOffset; found_mapper = 1; }
  }
  if (!found_mapper) return tinynv_fail("the fwsec image declares no dmem mapper, so there is nowhere to put the command");

  tinynv_falcon_dmem_mapper_t mapper;
  size_t mapper_off = desc->IMEMLoadSize + dmem_offset;
  if (mapper_off > image_len || sizeof mapper > image_len - mapper_off) return tinynv_fail("the fwsec dmem mapper is outside its own image");
  memcpy(&mapper, image + mapper_off, sizeof mapper);
  mapper.init_cmd = cmd_id;
  memcpy(image + mapper_off, &mapper, sizeof mapper);

  size_t cmd_off = desc->IMEMLoadSize + mapper.cmd_in_buffer_offset;
  size_t cmd_len = cmd_id == FWSEC_FRTS_CMD ? sizeof cmd : sizeof cmd.readVbiosDesc;
  if (cmd_off > image_len || cmd_len > image_len - cmd_off) return tinynv_fail("the fwsec command buffer is outside its own image");
  memcpy(image + cmd_off, &cmd, cmd_len);

  size_t pkc_off = desc->IMEMLoadSize + desc->PKCDataOffset;
  if (pkc_off > image_len || FWSEC_SIG_BYTES > image_len - pkc_off) return tinynv_fail("the fwsec signature slot is outside its own image");
  memcpy(image + pkc_off, sig, FWSEC_SIG_BYTES);
  return 0;
}

int tinynv_vbios_fwsec_frts(tinynv_gpu_t *g, uint64_t frts_offset, tinynv_fwsec_t *out) {
  memset(out, 0, sizeof *out);
  uint8_t *rom = malloc(VBIOS_SIZE);
  if (!rom) return tinynv_fail("no memory for the vbios image");
  nv_rd_block(&g->dev.mmio, VBIOS_MMIO_BASE, rom, VBIOS_SIZE);

  int rc = -1;
  size_t exp_off = 0, desc_off = 0, desc_size = 0;
  uint8_t *image = NULL, *sb = NULL;
  if (find_expansion_rom(rom, VBIOS_SIZE, &exp_off)) goto done;
  if (find_fwsec_desc(rom, VBIOS_SIZE, exp_off, &desc_off, &desc_size)) goto done;
  if (rom_get(rom, VBIOS_SIZE, desc_off, &out->desc, sizeof out->desc, "fwsec descriptor")) goto done;

  // The signature block sits directly after the v3 layout, and the image directly after that. Only the LAST 0x180
  // bytes of the block are the production signature the boot ROM checks.
  size_t sig_len = desc_size - TINYNV_FALCON_UCODE_DESC_V3_SIZE_44;
  size_t sig_off = desc_off + TINYNV_FALCON_UCODE_DESC_V3_SIZE_44;
  if (sig_len < FWSEC_SIG_BYTES) { tinynv_fail("the fwsec signature block is %#zx bytes, under the %#x the boot rom reads", sig_len, FWSEC_SIG_BYTES); goto done; }
  const uint8_t *sig = rom + sig_off + sig_len - FWSEC_SIG_BYTES;

  size_t image_len = (out->desc.StoredSize + 0xff) & ~(size_t)0xff;
  if (!(image = malloc(image_len))) { tinynv_fail("no memory for the fwsec image"); goto done; }
  if (rom_get(rom, VBIOS_SIZE, desc_off + desc_size, image, image_len, "fwsec image")) goto done;

  // The SB variant is built now, from the same bytes, and kept in host memory until the driver unloads. NVIDIA's driver
  // and nouveau both prepare it at init rather than read the ROM again at unload, after GSP-RM has had the card; and it
  // costs the recorded boot nothing, because nothing about it reaches the device until then. A failure here only
  // means there will be no SB step at unload, so it is recorded rather than allowed to fail the boot.
  if ((sb = malloc(image_len))) {
    memcpy(sb, image, image_len);
    if (fwsec_patch(sb, image_len, &out->desc, FWSEC_SB_CMD, 0, sig)) {
      snprintf(out->sb_error, sizeof out->sb_error, "%s", tinynv_last_error());
      free(sb);
      sb = NULL;
    }
  } else snprintf(out->sb_error, sizeof out->sb_error, "no memory for the fwsec-sb image");

  if (fwsec_patch(image, image_len, &out->desc, FWSEC_FRTS_CMD, frts_offset, sig)) goto done;

  // Video memory, not host memory: the falcon's own DMA engine fetches this, and it is told a physical address.
  if (tinynv_alloc_boot_mem(&g->mm, image_len, image, 0, &out->image)) goto done;
  out->image_len = image_len;
  out->sb_image = sb;
  sb = NULL;
  rc = 0;

done:
  free(sb);
  free(image);
  free(rom);
  return rc;
}

void tinynv_vbios_fwsec_free(tinynv_gpu_t *g, tinynv_fwsec_t *f) {
  if (f->image.size) tinynv_free_boot_mem(&g->mm, &f->image);
  free(f->sb_image);
  memset(f, 0, sizeof *f);
}
