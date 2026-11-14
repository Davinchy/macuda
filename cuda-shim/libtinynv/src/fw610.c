// The 610 runtime bundle's asset table and preflight. See fw610.h. The pins below are the derived-asset
// pins (tools/gsp610-runtime/runtime_assets.py ASSET_PINS), copied here as the C side of the same manifest: the
// bundle validator and the driver must agree on every hash, and test_lifecycle610 cross-checks this table against
// the prepared manifest so the two cannot drift silently.
#include "fw610.h"
#include "internal.h"
#include <stdio.h>
#include <string.h>

static const struct { const char *which, *file, *sha; } ASSETS[] = {
  {"gsp",             "gsp-610.43.03.bin",             "572373620a37418f24dc16b5031c39338778c3257e48e8408de9a57291b24f3a"},
  {"bootloader",      "bootloader-610.43.03.bin",      "45d6335a8185d6ff13e8d1484cea73f68d3d13aa1c19dc080ed1c3482dd5bac0"},
  {"bootloader_desc", "bootloader_desc-610.43.03.bin", "83f8f3f1f5f9496e0ea8fc7103c2beb64d0b6de2f5ffe725f13514de5a95fa42"},
  {"booter_load",     "booter_load-610.43.03.bin",     "6ee05b20e1dc244c08c052f36ef28469b8d09f4955d1fe33e48a179e11a7fd49"},
  {"booter_unload",   "booter_unload-610.43.03.bin",   "bf1495f15d00c3a41a6371747510be3d4803b98addf9097cf9e55bb3f7c2f6b3"},
  {"scrubber",        "scrubber-610.43.03.bin",        "1cfc78136a4dff4993b7e5a1a09345f6a67d0762555e5bd9660e406831379348"},
  {"generic_bl",      "generic_bl-610.43.03.bin",      "f21f1cfbb8fffa8067685314367fc4c50f3f84dd99eda7c19db4292d9269b9d9"},
  {"generic_bl_desc", "generic_bl_desc-610.43.03.bin", "3196d0c5e8817736651f0d888951d4f7aca10dfa4dedd908bac760e8c3335faf"},
};
#define NASSETS (sizeof(ASSETS) / sizeof(*ASSETS))

int tinynv_fw610_load(const char *which, tinynv_blob_t *out) {
  for (size_t i = 0; i < NASSETS; i++) {
    if (strcmp(ASSETS[i].which, which)) continue;
    char rel[256];
    snprintf(rel, sizeof rel, "%s/%s", TINYNV_FW610_DIR, ASSETS[i].file);
    if (tinynv_fw_load_rel(rel, ASSETS[i].sha, out))
      return tinynv_fail("the 610.43.03 runtime asset '%s' failed validation: %s (prepare the bundle with "
                         "tools/gsp610-runtime/run_prepare.sh; the 570 firmware is not a substitute)",
                         which, tinynv_last_error());
    return 0;
  }
  return tinynv_fail("'%s' is not an asset of the 610.43.03 runtime bundle", which);
}

int tinynv_fw610_validate_all(void) {
  for (size_t i = 0; i < NASSETS; i++) {
    tinynv_blob_t b;
    if (tinynv_fw610_load(ASSETS[i].which, &b)) return -1;
    tinynv_fw_free(&b);
  }
  return 0;
}

int tinynv_fw610_bl_desc(const tinynv_blob_t *desc, uint32_t *start_tag, uint32_t *code_size) {
  if (start_tag) *start_tag = 0;
  if (code_size) *code_size = 0;
  // RM_FLCN_BL_DESC packed: blStartTag, blDmemDescLoadOff, then {blCodeOffset, blCodeSize, blDataOffset, blDataSize}.
  if (!desc || desc->size != 24)
    return tinynv_fail("the 610 generic-BL descriptor is %zu bytes, the packed RM_FLCN_BL_DESC is 24",
                       desc ? desc->size : 0);
  uint32_t w[6];
  for (int i = 0; i < 6; i++)
    w[i] = (uint32_t)desc->data[i * 4] | ((uint32_t)desc->data[i * 4 + 1] << 8) |
           ((uint32_t)desc->data[i * 4 + 2] << 16) | ((uint32_t)desc->data[i * 4 + 3] << 24);
  if (!w[3]) return tinynv_fail("the 610 generic-BL descriptor declares zero code size");
  if (start_tag) *start_tag = w[0];
  if (code_size) *code_size = w[3];
  return 0;
}