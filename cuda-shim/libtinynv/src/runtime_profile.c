// The runtime profiles themselves, and the selection gate. See runtime_profile.h for the contract.
//
// The 570 descriptor is initialized FROM the existing constants (fw.h, fw_layout.h), not from copies of them: the
// profile owns the policy, and the values are the ones the recorded boot already proved. The per-chip hash pins
// stay where they have always lived - gsp.c (GSP_SHA, bl_sha) and flcn.c (FMC_SHA, hs_sha) - because they are the
// 570 profile's hash tables, keyed by chip; the 610 profile will resolve its hashes from the bundle manifest
// instead, and nothing here may substitute one version's hash for another's.
#include "runtime_profile.h"
#include "fw.h"
#include "fw_layout.h"
#include "fw_layout_610.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const tinynv_runtime_profile_t tinynv_runtime_570 = {
  .version = TINYNV_FW_VER,
  .implemented = 1,
  .device_ids = NULL, // every chip dev.c already names: ga102, ad102, gb202 - no new restriction
  .ndevice_ids = 0,
  .headers = "open-gpu-kernel-modules 81fe4fb417c8ac3b9bdcc1d56827d116743892a5 (570.86.16)",
  .asset_dir = "nvidia/<chip>/gsp",
  .image_bound = TINYNV_FW_IMAGE_BOUND,
  .bootbin_bound = TINYNV_FW_BOOTBIN_BOUND,
  .reserve_top = TINYNV_FW_RESERVE_TOP,
  .fw_heap_size = TINYNV_FW_HEAP_SIZE,
  .nonwpr_heap = TINYNV_FW_NONWPR_HEAP,
};

// The owner's desktop RTX 4090 only (10de:2684, AD102). No other device is in scope, and a successful boot on this
// one would not generalize the allowlist - that is a separate, evidence-based decision (spec).
static const uint32_t devices_610[] = {0x10de2684u};

const tinynv_runtime_profile_t tinynv_runtime_610 = {
  .version = "610.43.03",
  // implemented - the audited boot/RPC adapters are wired into the boot, resource, retry and
  // unload paths, and the whole lifecycle bundle is preflighted. The profile remains opt-in and
  // RTX-4090-only; 570 stays the default.
  .implemented = 1,
  .device_ids = devices_610,
  .ndevice_ids = sizeof(devices_610) / sizeof(*devices_610),
  .headers = "NVIDIA-kernel-module-source-610.43.03 (sha256 7e118923c7a23edc36114d63273a46e3e04e9af98695a42203e7ac2dfe9fc1dc)",
  .asset_dir = "nvidia/ad102/gsp-610.43.03",
  // sized from the matching pinned 610 definitions (fw_layout_610.h), not copied from 570. The image bound
  // covers the 80.32 MiB .fwimage; the AD102 non-WPR heap is the matching 1 MiB HAL (the audit's 2 MiB was a misread).
  .image_bound = TINYNV_FW610_IMAGE_BOUND,
  .bootbin_bound = TINYNV_FW610_BOOTBIN_BOUND,
  .reserve_top = TINYNV_FW610_RESERVE_TOP,
  .fw_heap_size = TINYNV_FW610_HEAP_SIZE,
  .nonwpr_heap = TINYNV_FW610_NONWPR_HEAP,
};

int tinynv_runtime_parse(const char *env, const tinynv_runtime_profile_t **out) {
  *out = NULL;
  if (!env || !*env) { *out = &tinynv_runtime_570; return 0; } // unset or empty: the default, as every knob here reads
  if (!strcmp(env, tinynv_runtime_570.version)) { *out = &tinynv_runtime_570; return 0; }
  if (!strcmp(env, tinynv_runtime_610.version)) { *out = &tinynv_runtime_610; return 0; }
  return tinynv_fail("TINYNV_GSP_RUNTIME must be 570.144 or 610.43.03 (got '%s'); the runtime selection is refused", env);
}

int tinynv_runtime_check(const tinynv_runtime_profile_t *p, tinynv_pci_t *pci) {
  if (!p) return tinynv_fail("no runtime profile was selected");
  // The allowlist, when the profile has one. ONE read of the PCI identity, read-only, and only on a live backend:
  // the 570 profile has no allowlist and never reaches this, so the default boot makes no extra device access.
  if (p->device_ids && p->ndevice_ids) {
    // an allowlisted profile is validated against real silicon only. A recording/replay backend carries the
    // 570 boot; letting a 610 request consume recorded operations would mix versions, which the design forbids.
    if (!pci || !pci->live || !pci->cfg_read)
      return tinynv_fail("TINYNV_GSP_RUNTIME=%s requires a live device: the profile is validated only against the "
                         "RTX 4090's real PCI identity, and a recording/replay backend is not a device. Use the "
                         "default 570.144 runtime for recorded boots", p->version);
    uint32_t raw = pci->cfg_read(pci, 0x00, 4);
    // (first live attempt): config dword 0 is the WIRE value - vendor in the LOW half, device in the HIGH
    // half (the 4090 reads 0x268410de). The allowlist is
    // written vendor:device (0x10de2684), so swap before comparing. The offline fakes used to answer the already
    // swapped value, which hid this until the real card was read; they now answer the wire value too.
    uint32_t id = (raw >> 16) | (raw << 16);
    int ok = 0;
    for (size_t i = 0; i < p->ndevice_ids; i++) ok |= id == p->device_ids[i];
    if (!ok) {
      char allow[128] = "";
      size_t used = 0;
      for (size_t i = 0; i < p->ndevice_ids && used < sizeof(allow) - 16; i++)
        used += (size_t)snprintf(allow + used, sizeof(allow) - used, "%s%04x:%04x", i ? ", " : "",
                                 p->device_ids[i] >> 16, p->device_ids[i] & 0xffff);
      return tinynv_fail("TINYNV_GSP_RUNTIME=%s is not supported on this device: the profile is validated only for "
                         "%s, and this device reads %04x:%04x", p->version, allow, id >> 16, id & 0xffff);
    }
  }
  if (!p->implemented)
    return tinynv_fail("TINYNV_GSP_RUNTIME=%s: the %s runtime profile is not implemented in this build; refusing "
                       "before any firmware is placed. 570.144 (the default) is the supported runtime",
                       p->version, p->version);
  return 0;
}

int tinynv_runtime_select(tinynv_pci_t *pci, const tinynv_runtime_profile_t **out) {
  const char *env = getenv("TINYNV_GSP_RUNTIME");
  if (tinynv_runtime_parse(env, out)) return -1;
  if (tinynv_runtime_check(*out, pci)) { *out = NULL; return -1; }
  // An explicit selection names itself; an unset variable stays silent so the default boot's output is unchanged.
  if (env && *env)
    fprintf(stderr, "tinynv: GSP runtime profile %s (TINYNV_GSP_RUNTIME; headers %s)\n", (*out)->version, (*out)->headers);
  return 0;
}
