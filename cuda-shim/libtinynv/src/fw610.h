// The 610.43.03 runtime bundle's asset resolution (docs/driver/gsp-610-runtime plan §4, design "Assets and
// filesystem"). The bundle is the prepared, hash-pinned set under <TINYNV_FW_DIR>/nvidia/ad102/gsp-610.43.03/
// (tools/gsp610-runtime/run_prepare.sh). Every logical asset resolves to its exact filename and SHA256 here; nothing
// is fetched at device open, and a missing or wrong-hash asset is an error, never a reason to substitute another
// version's file. The 570 paths (fw.c tinynv_fw_load) are untouched.
#ifndef TINYNV_FW610_H
#define TINYNV_FW610_H
#include "fw.h"

// The bundle subtree under the firmware root (the ASSET_DIR; also runtime_profile.asset_dir).
#define TINYNV_FW610_DIR "nvidia/ad102/gsp-610.43.03"

// Load one logical asset ("gsp", "bootloader", "bootloader_desc", "booter_load", "booter_unload", "scrubber",
// "generic_bl", "generic_bl_desc") from the bundle, hash-pinned. An unknown name is a refusal, not a path guess.
int tinynv_fw610_load(const char *which, tinynv_blob_t *out);

// Preflight the WHOLE selected lifecycle bundle - boot AND unload AND the host-request ucode - before any firmware
// is placed (design: "a preflight validates the entire selected lifecycle bundle, not only the assets used before
// the first successful boot"). Loads and hash-checks all eight assets, then frees them. Offline-safe: files only.
int tinynv_fw610_validate_all(void);

// The generic-bootloader descriptor (24-byte packed RM_FLCN_BL_DESC, ): the two fields the host-request
// executor needs. Bounds-checked; the pinned geometry is asserted by the caller's tests, not assumed here.
int tinynv_fw610_bl_desc(const tinynv_blob_t *desc, uint32_t *start_tag, uint32_t *code_size);

#endif
