// RTX 4090 (AD102) tensor-speed unlock: the pre-boot transaction, on the unchanged TinyGPU transport.
//
// Experimental, supervised operation; see docs/driver/rtx4090-tensor-unlock/spec.md.
// This is an adaptation of the pinned, independently exercised upstream transaction
// (JustAResearcher/gpu-unlock-ada @ 8855e7bca8ff142204e02301331fd56b26699819, GPL-2.0-only,
// kernel/gpu_unlock/unlock_610.c) to tinynv_pci_t. The pure parsers are vendored verbatim in unlock610/.
//
// It runs BEFORE any normal macuda device/MMU/allocator state exists (transport-audit.md D1): raw pci ops plus a
// temporary BAR0 window only. Callers: the standalone probe (tools/rtx4090-probe) and the optional
// default-off runtime gate (tinynv_tensor_unlock_gate in tinynv.c, TINYNV_TENSOR_UNLOCK=1). Off by default.
//
// Containment model: ordinary cleanup goes through quiesce + reset-effect verification.
// If firmware may still access host buffers and containment cannot be proven, the result says recovery_required:
// the caller MUST keep this process, its socket and its mappings alive, stop issuing GPU work, and request an
// enclosure power-off before disconnecting. That is best-effort recovery, NOT crash-proof retention: SIGKILL, a
// client crash, server death or host failure still force-unmap (transport-audit O14, a known limitation).
#ifndef TINYNV_TENSOR_UNLOCK_H
#define TINYNV_TENSOR_UNLOCK_H
#include "tinynv_pci.h"
#include "unlock610/prep_610.h"
#include <signal.h>
#include <stddef.h>
#include <stdint.h>

// Stages, in execution order. failed_at names the stage that did not complete; last_ok the one before it.
typedef enum {
  TU_STAGE_NONE = 0,
  TU_STAGE_ASSETS,      // bundle discovered, hash-verified, parsed
  TU_STAGE_CLAIM,       // identity 10de:2684, AD102 arch 0x192, D0, FLR cap, BME clear, BAR0 window
  TU_STAGE_GFW,         // ROM boot complete, cold gates (WPR2 down, FEAT not already open) - before any allocation
  TU_STAGE_LAYOUT,      // cold board inputs -> WPR2/FRTS/heap/GSP/boot reservations
  TU_STAGE_DMA,         // transaction buffers, IOVA < 4 GiB + contiguous + aligned, booter patch, radix tables
  TU_STAGE_FWSEC,       // FWSEC-FRTS from board ROM on GSP, WPR2 established
  TU_STAGE_PUBLISH,     // V67 signature published in WPR meta
  TU_STAGE_PAYLOAD,     // SEC2 privileged writes, completion + permission + readout gates
  TU_STAGE_RESTORE_SIG, // stock signature span restored before anything else touches the card
  TU_STAGE_SELECTORS,   // SS0/SS1 host writes + readbacks + full effective readout
  TU_STAGE_RESET,       // FLR + effect verification + retention re-read
  TU_STAGE_DONE
} tensor_unlock_stage_t;

const char *tensor_unlock_stage_name(int stage);

// Asset pins. The production pins below come from tools/rtx4090-unlock/manifest.json; the loader takes
// them as a parameter so the offline tests can drive the same code paths with synthetic, self-hashed fixtures.
typedef struct {
  const char *name;     // file name inside the bundle directory
  uint64_t size;        // exact byte count
  const char *sha256;   // lowercase hex
} tensor_unlock_pin_t;

typedef struct {
  tensor_unlock_pin_t booter, payload, gsp, boot, desc;
} tensor_unlock_pins_t;

extern const tensor_unlock_pins_t tensor_unlock_pins; // the pinned 610.43.03 AD102 bundle

// One IOVA record per transaction buffer, for the log: a DART fault is latched as an IOVA and nothing else.
typedef struct {
  const char *name;
  uint64_t iova, size;
} tensor_unlock_iova_t;

#define TU_MAX_BUFFERS 8
#define TU_READOUT_PHASES 5 // before-firmware, after-payload, selectors-applied, pre-FLR, post-FLR

typedef struct {
  int error;              // 0 = success, else a positive errno (for a preflight, success only means the read succeeded)
  int failed_at;          // tensor_unlock_stage_t that did not complete (TU_STAGE_NONE on success)
  int last_ok;            // last fully completed stage
  int preflight;          // this result came from the read-only probe and applied no state
  int recovery_required;  // firmware may still DMA: keep process/socket/mappings alive, power-off before disconnect
  int verified_existing;  // retained override found and verified (effective readout + selectors + cold handoff)
  char first_error[512];  // the FIRST failure, preserved through cleanup
  uint32_t identity, chip;
  uint32_t feat_entry, ss0_entry, ss1_entry;       // cold snapshot
  uint32_t perm_before[4], perm_after[4];          // WPR_CFG, FBPA, WPR, FEAT (inputs.md order)
  uint32_t readout[TU_READOUT_PHASES];             // raw 0x823818 at each phase
  int readout_valid[TU_READOUT_PHASES];
  uint32_t feat_final, ss0_final, ss1_final;       // post-FLR retention
  tensor_unlock_iova_t buffers[TU_MAX_BUFFERS];
  int nbuffer;
  int ndma_alloc, ndma_freed;                      // ownership evidence: recovery frees nothing
  double elapsed_s;
  const char *classification;                    // preflight: stock-cold / retained-unlock / warm / inaccessible / ...
} tensor_unlock_result_t;

// Set by a signal handler (async-signal-safe: a plain flag store, no socket work). Every bounded poll checks it and
// fails with EINTR, which enters the same containment cleanup as any other failure. Catchable termination therefore
// lands in the recovery state instead of tearing mappings down under a possibly-live firmware.
extern volatile sig_atomic_t tensor_unlock_abort;

// The transaction. `pci` is an open backend (tinygpu on the Mac); the caller keeps ownership of it. On
// recovery_required the caller must NOT close it or exit until the enclosure is powered off. `bundle_dir` is the
// prepared, validated bundle (third_party/unlock-610.43.03); nothing is downloaded here.
int tensor_unlock_run(tinynv_pci_t *pci, const char *bundle_dir, tensor_unlock_result_t *out);

// Read-only preflight snapshot: identity, chip, power/FLR capability, GFW state, FEAT/readout/SS0/SS1, WPR2.
// Issues no writes and allocates no DMA. Fills identity/chip/feat_entry/ss0/ss1/readout[0]/classification.
int tensor_unlock_preflight(tinynv_pci_t *pci, tensor_unlock_result_t *out);

// Default bundle discovery: $TINYNV_UNLOCK_BUNDLE, else the build-time third_party/unlock-610.43.03 directory.
const char *tensor_unlock_bundle_dir(void);

// ---- option parsing and status naming (runtime integration; pure, offline-tested) ---------------------------
// TINYNV_TENSOR_UNLOCK: NULL/empty/"0" -> 0 (off, the default), "1" -> 1, anything else -> -1 (refused).
int tensor_unlock_option_parse(const char *value);
// Transaction status: verified-existing / applied-and-verified / unsupported / failed-preflight (nothing launched:
// assets, claim, cold gates, layout, DMA) / failed-after-start (FWSEC onward). A read-only preflight is reported
// separately as preflight-ok; a failed preflight uses unsupported or failed-preflight.
const char *tensor_unlock_status(const tensor_unlock_result_t *r);

// ---- shared signal handling and the recovery hold (shared with the standalone probe) -----------------
// Flag-only handlers for SIGINT/SIGTERM/SIGHUP (set tensor_unlock_abort) and SIGUSR1 (the owner's power-off
// confirmation, honored ONLY inside the hold). Async-signal-safe: no socket work, no fprintf, no free. The previous
// dispositions are saved so a library caller can restore them after the transaction window; the normal boot installs
// its own teardown handlers afterwards and must not chain into these.
typedef struct { struct sigaction prev[4]; int installed; } tensor_unlock_sigstate_t;
void tensor_unlock_signals_install(tensor_unlock_sigstate_t *s);
void tensor_unlock_signals_restore(tensor_unlock_sigstate_t *s);
// The containment hold: firmware may still DMA and the installed transport force-unmaps on disconnect (O14), so the
// caller keeps process/socket/mappings alive, issues no GPU work, and waits for the owner to power the enclosure off
// ('off' + Enter on stdin, or SIGUSR1). SIGINT/SIGTERM are refused while holding. No timer, no automatic exit. Returns
// 3 after the confirmed power-off. Requires tensor_unlock_signals_install first. `who` prefixes the log lines.
// SIGKILL, a client crash, server death or host failure defeat this — the documented crash limitation, not a guarantee.
int tensor_unlock_recovery_hold(const tensor_unlock_result_t *r, const char *who);

// The runtime integration gate (implemented in tinynv.c, next to device_boot): parses TINYNV_TENSOR_UNLOCK, and when
// it is 1 runs the transaction on the open TinyGPU backend BEFORE any normal device state exists, reports the spec
// status, and holds recovery ownership if containment cannot be proven. Returns 0 to proceed with the normal boot
// (*active=1 only when the override is established), 1 for a contained refusal, 3 after a confirmed recovery power-off.
// With the option absent/0 it is inert: no asset reads, no allocations, no unlock MMIO. Exposed for the offline
// end-to-end tests (test/runtime_unlock_harness.c) which drive it against the mock server.
int tinynv_tensor_unlock_gate(tinynv_pci_t *pci, int *active);

// ---- post-boot verification: the handoff readout on a live normal runtime boot -------------------------
// The override values the pinned recipe applies (inputs.md register manifest). tensor_unlock.c static-asserts its
// private copies equal these, so the transaction and the verifier cannot drift apart.
#define TU_UNLOCK_SS0 0x88888888u
#define TU_UNLOCK_SS1 0x8u
#define TU_UNLOCK_READOUT_MASK 0x07ffffffu

typedef struct {
  uint32_t chip;      // PMC_BOOT_0: the accessibility canary (a dead bus reads all-ones everywhere)
  uint32_t feat;      // 0x823804 FEAT PLM: 0xffffffff is the OPEN (unlocked) value, not an error pattern here
  uint32_t readout;   // 0x823818 raw; effective state is readout & TU_UNLOCK_READOUT_MASK (0 = no speed limit)
  uint32_t ss0, ss1;  // 0x82381c / 0x823820 selectors
  int valid;          // every read accessible (chip sane, no all-ones/badf on readout/selectors)
  int unlocked;       // valid && FEAT open && both selectors at the unlock values && effective readout all zero
  int stock;          // valid && FEAT not open && selectors not the unlock pair (a stock or partial state)
} tensor_unlock_state_t;

// Read-only post-boot snapshot through the NORMAL device path (tinynv_rd32 on the booted device's BAR0 window).
// Requires an already-booted, non-failed runtime device: it never boots implicitly and issues no writes, no reset,
// no DMA. Returns 0 and fills *out (check valid/unlocked), or -1 when there is no live booted device to read.
// An inaccessible register is reported as valid=0, never as evidence of success (spec requirement 3).
struct tinynv_device;
int tensor_unlock_postboot(struct tinynv_device *d, tensor_unlock_state_t *out);

// Pure classification of the raw reads (the same gates the transaction uses): fills valid/unlocked/stock.
// FEAT == 0xffffffff is the recipe's OPEN value, so all-ones is an accessibility error only on the OTHER
// registers; the chip id is the canary that separates "unlocked" from "dead bus". Exposed for the offline tests.
void tensor_unlock_state_classify(tensor_unlock_state_t *s);

// ---- test seams (documented, not used by the probe) ------------------------------------------------------------------
// The parsed bundle: five hash-verified blobs plus the geometry the transaction needs. Exposed so the offline tests
// can assert what the loader accepted; the probe only uses tensor_unlock_run.
typedef struct { uint8_t *data; size_t size; } tensor_unlock_blob_t;
struct tu_assets {
  tensor_unlock_blob_t booter, payload, gsp, boot, desc;
  struct gpu_unlock_booter_hdr hdr;
  uint32_t booter_image_size;
  struct gpu_unlock_span image_span, sig_span;
  struct gpu_unlock_boot_desc boot_desc;
  struct gpu_unlock_radix_plan radix;
};
// Load + hash-verify + parse a bundle with explicit pins. Returns 0 or -1 (tinynv_last_error has the reason).
int tensor_unlock_assets_load(const char *dir, const tensor_unlock_pins_t *pins, struct tu_assets **out);
void tensor_unlock_assets_free(struct tu_assets *a);
// The transaction with caller-supplied pins and preloaded assets (synthetic fixtures in the offline tests).
int tensor_unlock_run_with(tinynv_pci_t *pci, struct tu_assets *assets, tensor_unlock_result_t *out);
// Scale every bounded wait (offline tests only; 1.0 = the real budgets). Never changes what a timeout MEANS.
void tensor_unlock_set_time_scale(double scale);

#endif
