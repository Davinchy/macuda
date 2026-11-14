// The runtime-profile gate (, docs/driver/gsp-610-runtime plan §3), with no GPU.
//
// What is checked here:
//   1. The descriptors: 570 is the implemented default carrying exactly the existing fw.h/fw_layout.h values; 610 is
//      implemented and allowlisted to the RTX 4090.
//   2. Strict TINYNV_GSP_RUNTIME parsing: unset/empty and the two exact names, and a table of near-misses that must
//      all be refused (no trimming, no prefixes, no case folding, no "latest").
//   3. Dispatch: the 570 check makes ZERO device accesses; a 610 request reads the PCI identity once (read-only) on a
//      live backend, accepts the 4090, refuses a non-4090 with the unsupported-device message, and refuses a
//      recording/replay backend (610 is validated against real silicon only) - before any firmware is placed.
//   4. tinynv_gpu_open selects and freezes the profile BEFORE any device state: an invalid or unsupported request
//      fails with no BAR mapped, no config write, no reset; unset and explicit 570 reach the device identically and
//      freeze the same descriptor, which survives the gsp/flcn wipe a boot retry performs (unmake_boot).
//   5. With a recorded boot (optional argv[1]): unset and explicit 570 reproduce the SAME recorded operations with
//      the same stats, through gpu_open and (when firmware is present) the software init that loads every image by
//      its profile-owned name; 610 and invalid requests consume ZERO recorded operations.
//
// This is a configuration/dispatch suite. It is NOT evidence that any firmware boots: nothing here runs on a GPU.
// The wired 610 lifecycle (software init, mock queue peer, failure injection) is test_lifecycle610.
#include "gpu.h"
#include "internal.h"
#include "runtime_profile.h"
#include "fw.h"
#include "fw_layout.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

// ---- a counting fake backend ---------------------------------------------------------------------------------------
// Every entry point is counted; cfg_read answers a chosen PCI identity. bar_map fails, so gpu_open under test stops
// at the first device touch - which is exactly the boundary the selection gate must sit in front of.

typedef struct {
  int cfg_reads, cfg_writes, bar_maps, bar_infos, resets, quiesces, dma_allocs;
  uint32_t id;
} fake_counts_t;

static fake_counts_t FC;
static tinynv_pci_t fake_pci;

static uint32_t fk_cfg_read(tinynv_pci_t *p, uint32_t off, uint32_t size) {
  (void)p; (void)size;
  FC.cfg_reads++;
  // answer the WIRE dword (vendor low, device high), like the real card does. FC.id stays in the
  // vendor:device notation the callers write; returning it unswapped is what hid the allowlist's endian bug.
  return off == 0x00 ? (FC.id >> 16) | (FC.id << 16) : 0;
}
static void fk_cfg_write(tinynv_pci_t *p, uint32_t off, uint32_t size, uint32_t val) {
  (void)p; (void)off; (void)size; (void)val;
  FC.cfg_writes++;
}
static int fk_bar_map(tinynv_pci_t *p, int bar, uint64_t off, uint64_t size, tinynv_mmio_t *out) {
  (void)p; (void)bar; (void)off; (void)size; (void)out;
  FC.bar_maps++;
  return tinynv_fail("fake backend: no BAR is mapped here (the selection gate is what is under test)");
}
static int fk_bar_info(tinynv_pci_t *p, int bar, uint64_t *base, uint64_t *size) {
  (void)p; (void)bar; (void)base; (void)size;
  FC.bar_infos++;
  return -1;
}
static int fk_reset(tinynv_pci_t *p) { (void)p; FC.resets++; return -1; }
static void fk_quiesce(tinynv_pci_t *p) { (void)p; FC.quiesces++; }
static int fk_dma_alloc(tinynv_pci_t *p, size_t size, tinynv_dma_t *out) {
  (void)p; (void)size; (void)out;
  FC.dma_allocs++;
  return -1;
}

static void fake_init(uint32_t id, int live) {
  memset(&FC, 0, sizeof FC);
  FC.id = id;
  fake_pci = (tinynv_pci_t){.name = "fake-profile", .live = live,
                            .cfg_read = fk_cfg_read, .cfg_write = fk_cfg_write,
                            .bar_map = fk_bar_map, .bar_info = fk_bar_info,
                            .reset = fk_reset, .quiesce = fk_quiesce, .dma_alloc = fk_dma_alloc};
}

static int no_device_work(void) {
  return !FC.cfg_writes && !FC.bar_maps && !FC.bar_infos && !FC.resets && !FC.quiesces && !FC.dma_allocs;
}

// ---- 1. the descriptors --------------------------------------------------------------------------------------------

static void descriptors(void) {
  CHECK(!strcmp(tinynv_runtime_570.version, TINYNV_FW_VER), "the 570 profile version drifted from TINYNV_FW_VER");
  CHECK(tinynv_runtime_570.implemented == 1, "the 570 profile must be implemented: it is the default");
  CHECK(tinynv_runtime_570.device_ids == NULL, "the 570 profile must keep the driver's existing chip support, "
        "not a new PCI allowlist");
  CHECK(tinynv_runtime_570.image_bound == TINYNV_FW_IMAGE_BOUND &&
        tinynv_runtime_570.bootbin_bound == TINYNV_FW_BOOTBIN_BOUND &&
        tinynv_runtime_570.reserve_top == TINYNV_FW_RESERVE_TOP,
        "the 570 reservation policy drifted from fw_layout.h: every recorded address depends on it");
  CHECK(!strcmp(tinynv_runtime_610.version, "610.43.03"), "the 610 version string is wrong");
  CHECK(tinynv_runtime_610.implemented == 1, "610 must be implemented with its adapters and lifecycle wired");
  CHECK(tinynv_runtime_610.ndevice_ids == 1 && tinynv_runtime_610.device_ids[0] == 0x10de2684u,
        "the 610 allowlist must be exactly the owner's RTX 4090 (10de:2684)");
  // the wire dword the real card answers at config 0x00 (vendor low, device high).
  // The gate must compare against THIS, not the vendor:device notation.
  CHECK(((tinynv_runtime_610.device_ids[0] >> 16) | (tinynv_runtime_610.device_ids[0] << 16)) == 0x268410deu,
        "the 610 allowlist entry does not swap to the 4090's wire identity 0x268410de");
  printf("descriptors: 570 implemented/default with the fw_layout policy; 610 implemented, opt-in, 4090-only\n");
}

// ---- 2. strict parsing ---------------------------------------------------------------------------------------------

static void parsing(void) {
  const struct { const char *env; int rc; const tinynv_runtime_profile_t *want; } cases[] = {
    {NULL, 0, &tinynv_runtime_570},
    {"", 0, &tinynv_runtime_570},
    {"570.144", 0, &tinynv_runtime_570},
    {"610.43.03", 0, &tinynv_runtime_610},
    // near-misses: all refused. A typo must never silently select the default, and a prefix must never match.
    {"570", -1, NULL}, {"610", -1, NULL}, {"570.144.0", -1, NULL}, {"610.43", -1, NULL},
    {" 570.144", -1, NULL}, {"570.144 ", -1, NULL}, {"570.144\n", -1, NULL}, {"\t570.144", -1, NULL},
    {"610.43.03 ", -1, NULL}, {" 610.43.03", -1, NULL}, {"0", -1, NULL}, {"1", -1, NULL},
    {"default", -1, NULL}, {"latest", -1, NULL}, {"LATEST", -1, NULL}, {"570.86.16", -1, NULL},
    {"610.43.3", -1, NULL}, {"570.144,610.43.03", -1, NULL}, {"570.144;610.43.03", -1, NULL},
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
    const tinynv_runtime_profile_t *p = (void *)1; // poisoned: a refused parse must leave NULL, not hold a stale pick
    int rc = tinynv_runtime_parse(cases[i].env, &p);
    CHECK(rc == cases[i].rc, "parse('%s') returned %d, wanted %d", cases[i].env ? cases[i].env : "(unset)", rc, cases[i].rc);
    CHECK(p == cases[i].want, "parse('%s') selected the wrong profile", cases[i].env ? cases[i].env : "(unset)");
    if (rc) CHECK(strstr(tinynv_last_error(), "TINYNV_GSP_RUNTIME") != NULL,
                  "the refusal of '%s' does not name the variable: %s", cases[i].env, tinynv_last_error());
  }
  printf("parsing: %zu cases - unset/empty default to 570, the two exact names select, every near-miss refused\n",
         sizeof(cases) / sizeof(*cases));
}

// ---- 3. the dispatch gate ------------------------------------------------------------------------------------------

static void dispatch(void) {
  // 570: no device access, whatever the device is - including one that cannot answer at all.
  const uint32_t ids[] = {0x10de2684u, 0x10de2204u, 0x10de2b85u, 0xffffffffu, 0};
  for (size_t i = 0; i < sizeof(ids) / sizeof(*ids); i++) {
    fake_init(ids[i], 1);
    CHECK(tinynv_runtime_check(&tinynv_runtime_570, &fake_pci) == 0, "the 570 check refused device %#x", ids[i]);
    CHECK(FC.cfg_reads == 0 && no_device_work(), "the 570 check touched the device (%d reads)", FC.cfg_reads);
  }
  fake_init(0x10de2684u, 1);
  CHECK(tinynv_runtime_check(&tinynv_runtime_570, NULL) == 0, "the 570 check needs no backend");

  // 610 on the live 4090: the device matches and the build implements it - accepted after ONE read-only identity read.
  fake_init(0x10de2684u, 1);
  CHECK(tinynv_runtime_check(&tinynv_runtime_610, &fake_pci) == 0, "610 was refused on the live 4090: %s", tinynv_last_error());
  CHECK(FC.cfg_reads == 1 && no_device_work(), "the 610 check did more than one identity read");

  // 610 on anything else: the unsupported-device message, naming both the allowlist and what the device reads.
  const uint32_t other[] = {0x10de2204u, 0x10de2b85u, 0x10de2685u, 0xffffffffu};
  for (size_t i = 0; i < sizeof(other) / sizeof(*other); i++) {
    fake_init(other[i], 1);
    CHECK(tinynv_runtime_check(&tinynv_runtime_610, &fake_pci) != 0, "610 was accepted on device %#x", other[i]);
    CHECK(strstr(tinynv_last_error(), "not supported on this device") != NULL &&
          strstr(tinynv_last_error(), "10de:2684") != NULL,
          "the non-4090 refusal is not the unsupported-device message: %s", tinynv_last_error());
    CHECK(FC.cfg_reads == 1 && no_device_work(), "the device check did more than one identity read");
  }

  // A recording is not a device: no identity read, and 610 is refused - it is validated against real silicon only,
  // so a recorded 570 boot can never be consumed by a 610 request (mixed-version state is forbidden).
  fake_init(0x10de2684u, 0);
  CHECK(tinynv_runtime_check(&tinynv_runtime_610, &fake_pci) != 0, "610 was accepted on a recording");
  CHECK(strstr(tinynv_last_error(), "requires a live device") != NULL,
        "the recording refusal does not say why: %s", tinynv_last_error());
  CHECK(FC.cfg_reads == 0 && no_device_work(), "the check read from a non-live backend");
  CHECK(tinynv_runtime_check(&tinynv_runtime_610, NULL) != 0, "610 was accepted with no backend at all");
  printf("dispatch: 570 makes zero device accesses; 610 accepted on the live 4090 only (device message off-4090, "
         "live-device message on recordings)\n");
}

// ---- 4. selection inside tinynv_gpu_open ----------------------------------------------------------------------------

static void open_gate(void) {
  tinynv_gpu_t g;

  // Invalid: refused before the first device touch of any kind.
  fake_init(0x10de2684u, 1);
  setenv("TINYNV_GSP_RUNTIME", "610", 1);
  memset(&g, 0xa5, sizeof g);
  CHECK(tinynv_gpu_open(&g, &fake_pci) != 0, "gpu_open accepted an invalid runtime name");
  CHECK(FC.cfg_reads == 0 && no_device_work(), "an invalid runtime name reached the device");
  CHECK(strstr(tinynv_last_error(), "TINYNV_GSP_RUNTIME") != NULL, "the refusal does not name the variable");

  // 610 on the live 4090: selection passes (one identity read), the profile is frozen, and the open proceeds to the
  // first device touch (bar_map, which this fake refuses) - no config write, no reset, no firmware before that.
  fake_init(0x10de2684u, 1);
  setenv("TINYNV_GSP_RUNTIME", "610.43.03", 1);
  CHECK(tinynv_gpu_open(&g, &fake_pci) != 0, "the fake backend was supposed to stop this open at bar_map");
  CHECK(g.profile == &tinynv_runtime_610, "the 610 open did not freeze the 610 profile");
  CHECK(FC.cfg_reads == 1 && FC.bar_maps == 1 && FC.cfg_writes == 0 && FC.resets == 0 && FC.dma_allocs == 0,
        "the 610 open did more than the identity read and the first bar_map before stopping");

  // 610 on a non-4090: the unsupported-device message, still no device work beyond the identity read.
  fake_init(0x10de2204u, 1);
  CHECK(tinynv_gpu_open(&g, &fake_pci) != 0, "gpu_open accepted 610 on a non-4090");
  CHECK(strstr(tinynv_last_error(), "not supported on this device") != NULL,
        "the non-4090 open refusal is not the device message: %s", tinynv_last_error());
  CHECK(no_device_work(), "the non-4090 refusal came after device work");

  // Unset and explicit 570: the same frozen descriptor, and the first device touch is the same one (bar_map).
  const tinynv_runtime_profile_t *picked[2] = {NULL, NULL};
  int reads[2] = {0, 0}, maps[2] = {0, 0};
  for (int i = 0; i < 2; i++) {
    fake_init(0x10de2684u, 1);
    if (i) setenv("TINYNV_GSP_RUNTIME", "570.144", 1);
    else unsetenv("TINYNV_GSP_RUNTIME");
    int rc = tinynv_gpu_open(&g, &fake_pci); // stops at the fake bar_map, after selection
    CHECK(rc != 0, "the fake backend was supposed to stop this open at bar_map");
    CHECK(g.profile == &tinynv_runtime_570, "open did not freeze the 570 profile (%s)", i ? "explicit" : "unset");
    CHECK(g.dev.runtime == NULL || g.dev.runtime == &tinynv_runtime_570, "dev carries a different profile");
    picked[i] = g.profile; reads[i] = FC.cfg_reads; maps[i] = FC.bar_maps;
    CHECK(FC.cfg_reads == 0 && FC.cfg_writes == 0 && FC.resets == 0,
          "the 570 open made selection-time device accesses (%s)", i ? "explicit" : "unset");
  }
  CHECK(picked[0] == picked[1] && reads[0] == reads[1] && maps[0] == maps[1] && maps[0] == 1,
        "unset and explicit 570 did not reach the device identically");
  unsetenv("TINYNV_GSP_RUNTIME");

  // The freeze across a retry: unmake_boot wipes gsp and flcn and nothing else. Mirror exactly that wipe here and
  // require the profile to survive it; the live guard in tinynv_gpu_boot_firmware asserts the same on every retry.
  memset(&g.gsp, 0, sizeof g.gsp);
  memset(&g.flcn, 0, sizeof g.flcn);
  CHECK(g.profile == &tinynv_runtime_570, "the gsp/flcn wipe a retry performs disturbed the frozen profile");
  // The same wipe under a frozen 610 profile: a retry must reboot the SAME runtime, never fall back.
  g.profile = &tinynv_runtime_610;
  memset(&g.gsp, 0, sizeof g.gsp);
  memset(&g.flcn, 0, sizeof g.flcn);
  CHECK(g.profile == &tinynv_runtime_610, "the retry wipe disturbed a frozen 610 profile");
  printf("open gate: invalid refused before device work; 610 frozen on the live 4090; unset == explicit 570; "
         "profile survives the retry wipe\n");
}

// ---- 5. the recorded boot, when one is present ----------------------------------------------------------------------

typedef struct {
  int rc, sw;
  const tinynv_runtime_profile_t *profile;
  size_t consumed, ops_done, skipped, divergences, cursor;
} replay_run_t;

static replay_run_t replay_once(const char *prefix, const char *env, int with_sw) {
  replay_run_t r = {.sw = -1};
  if (env) setenv("TINYNV_GSP_RUNTIME", env, 1);
  else unsetenv("TINYNV_GSP_RUNTIME");
  tinynv_pci_t pci;
  if (tinynv_pci_open_replay(prefix, &pci)) {
    printf("  FAIL: replay open: %s\n", tinynv_last_error());
    fails++;
    return r;
  }
  tinynv_gpu_t g;
  r.rc = tinynv_gpu_open(&g, &pci);
  r.profile = g.profile;
  if (!r.rc && with_sw) r.sw = tinynv_gpu_init_sw(&g);
  tinynv_replay_stats_t s;
  tinynv_replay_stats(&pci, &s);
  r.consumed = s.consumed; r.ops_done = s.ops_done; r.skipped = s.skipped;
  r.divergences = s.divergences; r.cursor = s.cursor;
  if (pci.close) pci.close(&pci);
  return r;
}

static void same_run(const replay_run_t *a, const replay_run_t *b, const char *what) {
  CHECK(a->rc == b->rc && a->sw == b->sw, "%s: unset and explicit 570 returned different results", what);
  CHECK(a->profile == b->profile && a->profile == &tinynv_runtime_570, "%s: different profiles selected", what);
  CHECK(a->consumed == b->consumed && a->ops_done == b->ops_done && a->skipped == b->skipped &&
        a->divergences == b->divergences && a->cursor == b->cursor,
        "%s: unset and explicit 570 diverged from each other (%zu/%zu ops, %zu/%zu divergences)",
        what, a->ops_done, b->ops_done, a->divergences, b->divergences);
}

static void replay_arm(const char *prefix) {
  // The recording began at a card with no resident firmware; the reset-on-every-open policy sits above it (see the
  // Makefile's test target), so both runs say TINYNV_START_RESET=0 and reproduce the recorded boot itself.
  setenv("TINYNV_START_RESET", "0", 1);
  // Is the firmware present? Then the comparison extends through the software init, which loads every image by its
  // profile-owned name and checks every hash - the strongest offline statement that 570 routing changed nothing.
  tinynv_blob_t probe;
  int have_fw = !tinynv_fw_load("gb202", "fmc-" TINYNV_FW_VER ".bin", NULL, &probe);
  if (have_fw) tinynv_fw_free(&probe);

  replay_run_t unset_run = replay_once(prefix, NULL, have_fw);
  replay_run_t explicit_run = replay_once(prefix, "570.144", have_fw);
  same_run(&unset_run, &explicit_run, "replay");
  CHECK(unset_run.rc == 0, "the recorded bring-up failed under the default profile: %s", tinynv_last_error());
  if (have_fw) CHECK(unset_run.sw == 0, "the recorded software init failed under the default profile: %s", tinynv_last_error());
  printf("replay: unset == explicit 570 over %zu recorded operations%s (%zu divergences each)\n",
         unset_run.ops_done, have_fw ? " including the firmware software init" : " (bring-up only; no firmware)",
         unset_run.divergences);

  // 610 and an invalid name: refused with ZERO recorded operations consumed - before firmware publication, before
  // anything the recording contains.
  const char *refused[] = {"610.43.03", "610"};
  for (size_t i = 0; i < sizeof(refused) / sizeof(*refused); i++) {
    replay_run_t r = replay_once(prefix, refused[i], 0);
    CHECK(r.rc != 0, "the replay accepted TINYNV_GSP_RUNTIME=%s", refused[i]);
    CHECK(r.consumed == 0 && r.cursor == 0 && r.divergences == 0,
          "TINYNV_GSP_RUNTIME=%s consumed %zu recorded operations before refusing", refused[i], r.consumed);
  }
  unsetenv("TINYNV_GSP_RUNTIME");
  unsetenv("TINYNV_START_RESET");
  printf("replay: 610 and invalid requests refused with zero recorded operations\n");
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  descriptors();
  parsing();
  dispatch();
  open_gate();
  if (argc > 1) replay_arm(argv[1]);
  else printf("replay: SKIPPED - no trace prefix given (pass one to compare unset vs explicit 570 on a recorded boot); "
              "an environmental gap, not a pass\n");
  printf(fails ? "runtime profile: %d checks FAILED\n" : "runtime profile: all checks passed\n", fails);
  return fails ? 1 : 0;
}
