// The selectable GSP runtime profile (docs/driver/gsp-610-runtime plan §3).
//
// One immutable descriptor per device, chosen from TINYNV_GSP_RUNTIME before any device state exists and frozen for
// the whole open - boot, every retry, and unload. The 570 profile delegates to the existing logic: same paths under
// TINYNV_FW_DIR, same pinned hashes, same allocation order, same recorded MMIO behavior. It is the default, and an
// unset variable and an explicit "570.144" select the very same descriptor without reading or writing the device.
//
// The 610 profile is implemented (boot/RPC adapters wired, lifecycle bundle preflighted, offline and live
// lifecycle checks green) but remains opt-in and allowlisted to the RTX 4090. The 570 profile stays the default.
//
// Runtime selection is independent of the application using the driver and never changes
// tensor state.
#ifndef TINYNV_RUNTIME_PROFILE_H
#define TINYNV_RUNTIME_PROFILE_H
#include "tinynv_pci.h"
#include <stddef.h>
#include <stdint.h>

typedef struct tinynv_runtime_profile {
  const char *version;       // the TINYNV_GSP_RUNTIME value, and the infix of every firmware filename
  int implemented;           // whether THIS build can boot it. Both profiles are implemented as of .
  // PCI identity allowlist, vendor<<16 | device. NULL means "every device this driver already supports" - the 570
  // profile keeps the driver's existing chip table (dev.c) as its allowlist and adds no new restriction.
  const uint32_t *device_ids;
  size_t ndevice_ids;
  const char *headers;       // the pinned NVIDIA header revision this profile's ABI is audited against
  const char *asset_dir;     // layout under TINYNV_FW_DIR, for diagnostics; 570 keeps nvidia/<chip>/gsp
  // Reservation policy (fw_layout.h for 570, fw_layout_610.h for 610): the load-time image bounds and the hold-back
  // from the top of video memory the manager must never hand out. The 610 values come from the matching
  // pinned definitions (the .fwimage grew to 80.32 MiB, past the 570 64 MiB bound) and are reachable
  // only through the 610 profile.
  uint64_t image_bound, bootbin_bound, reserve_top;
  // The firmware-heap terms the post-boot carveout check reasons about (mmu.c). Profile-owned so the check is correct
  // for each runtime rather than hardcoded to 570: fw_heap_size is the minimum-span heap term, nonwpr_heap the
  // unprotected heap below wpr2. 570 carries exactly the fw_layout.h constants, so its recorded behavior is unchanged.
  uint64_t fw_heap_size, nonwpr_heap;
} tinynv_runtime_profile_t;

extern const tinynv_runtime_profile_t tinynv_runtime_570;
extern const tinynv_runtime_profile_t tinynv_runtime_610;

// Strict parsing, no trimming, no prefixes, no case folding: NULL or "" selects the 570 default; exactly "570.144"
// or "610.43.03" select that profile; anything else is refused through tinynv_fail with *out left NULL. Pure: it
// reads no environment and touches no device, so the test drives it directly.
int tinynv_runtime_parse(const char *env, const tinynv_runtime_profile_t **out);

// Whether this build can run the profile on this device. The 570 profile returns 0 without any device access. A
// profile with an allowlist is checked against the PCI identity with ONE read-only config-space read, and only on a
// live backend - a recording is not a device and is never read from here. An unimplemented profile is refused after
// the device check, so a 610 request on a non-RTX-4090 gets the unsupported-device message and one on the 4090 gets
// the not-implemented message. No firmware is loaded, placed or published on any failure.
int tinynv_runtime_check(const tinynv_runtime_profile_t *p, tinynv_pci_t *pci);

// Parse $TINYNV_GSP_RUNTIME and check it against the build and the device. On success *out is the frozen descriptor
// and an EXPLICIT selection is named on stderr (an unset variable stays silent, so the default boot's output is
// unchanged). On failure *out is NULL and the reason is in tinynv_last_error.
int tinynv_runtime_select(tinynv_pci_t *pci, const tinynv_runtime_profile_t **out);

#endif
