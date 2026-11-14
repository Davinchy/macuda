// The AD102 unlock transaction against a fake card: every stage gate, every failure path, no GPU.
//
// The fake backend below simulates the installed TinyGPU transport's semantics at register level: config space with
// PM/PCIe capabilities, a BAR0 whose reads/writes behave like the AD102 fields the recipe touches, server-style DMA
// mappings (page-granular IOVAs, rounded sizes), and a Reset whose STATUS IS MEANINGLESS (the dext discards it) —
// only its effects are simulated. Falcons execute semantically: FWSEC-FRTS establishes WPR2 from the FRTS command
// actually patched into its DMEM; SEC2 applies the privileged PLM writes only when the published WPR meta points at
// an intact V67 payload copy and bus mastering is on. Host selector writes do nothing until FEAT is open — the
// direct-write shortcut the spec forbids is impossible here, exactly as on the card.
//
// Mock tests prove CONTROL FLOW, not hardware completion or DMA safety (plan Phase 3). Assets are synthetic and
// self-hashed through the same loader the probe uses; when the prepared real bundle is present (argv[1]) its bytes
// are hash/geometry-checked offline as well.
#include "tensor_unlock.h"
#include "internal.h"
#include "sha256.h"
#include "unlock610/prep_610.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

#include "fake_ad102.h"

// ---- environment: synthetic bundle on disk + test pins ------------------------------------------------------------------

typedef struct {
  char dir[256];
  tensor_unlock_pins_t pins;
  char sha[5][65];
  struct tu_assets *assets;
} env_t;

static void put_file(const char *dir, const char *name, const uint8_t *data, size_t n) {
  char path[512];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  if (!f) { perror(path); exit(2); }
  if (fwrite(data, 1, n, f) != n) { perror("fwrite"); exit(2); }
  fclose(f);
}

static void env_pin(env_t *e, int i, tensor_unlock_pin_t *pin, const char *name, const uint8_t *data, size_t n) {
  uint8_t d[32];
  tinynv_sha256(data, n, d);
  tinynv_sha256_hex(d, e->sha[i]);
  *pin = (tensor_unlock_pin_t){.name = name, .size = n, .sha256 = e->sha[i]};
  put_file(e->dir, name, data, n);
}

static void env_setup(env_t *e) {
  memset(e, 0, sizeof(*e));
  snprintf(e->dir, sizeof(e->dir), "/tmp/tu-test-XXXXXX");
  if (!mkdtemp(e->dir)) { perror("mkdtemp"); exit(2); }
  size_t n;
  uint8_t *booter = build_booter(&n, 0);
  env_pin(e, 0, &e->pins.booter, "booter_load_ad102.bin", booter, n);
  free(booter);
  build_payload();
  env_pin(e, 1, &e->pins.payload, "v67-ad102-610.43.03-four-plm.bin", synth_payload, PAYLOAD_SIZE);
  uint8_t *elf = build_gsp_elf(&n);
  env_pin(e, 2, &e->pins.gsp, "gsp_ga10x.bin", elf, n);
  free(elf);
  uint8_t *boot = build_boot(&n);
  env_pin(e, 3, &e->pins.boot, "gsp_boot_ad102.bin", boot, n);
  free(boot);
  uint8_t *desc = build_desc(&n);
  env_pin(e, 4, &e->pins.desc, "gsp_boot_desc_ad102.bin", desc, n);
  free(desc);
}

static void env_teardown(env_t *e) {
  tensor_unlock_assets_free(e->assets);
  const char *names[] = {"booter_load_ad102.bin", "v67-ad102-610.43.03-four-plm.bin", "gsp_ga10x.bin",
                         "gsp_boot_ad102.bin", "gsp_boot_desc_ad102.bin"};
  char path[512];
  for (int i = 0; i < 5; i++) {
    snprintf(path, sizeof(path), "%s/%s", e->dir, names[i]);
    unlink(path);
  }
  rmdir(e->dir);
}

static int env_load(env_t *e) {
  int rc = tensor_unlock_assets_load(e->dir, &e->pins, &e->assets);
  CHECK(rc == 0, "synthetic bundle failed to load: %s", tinynv_last_error());
  return rc;
}

// run the transaction on a fresh fake card; returns the result code and fills r
static int run_case(env_t *e, int knob, tensor_unlock_result_t *r) {
  fake_init(knob);
  tensor_unlock_abort = 0;
  int rc = tensor_unlock_run_with(&F.pci, e->assets, r);
  F.pci.close(&F.pci);
  return rc;
}

// ---- the tests -----------------------------------------------------------------------------------------------------------

static void test_happy_path(env_t *e) {
  printf("happy path: full transaction, every gate\n");
  tensor_unlock_result_t r;
  int rc = run_case(e, K_NONE, &r);
  CHECK(rc == 0 && r.error == 0, "transaction failed: %s", r.first_error);
  CHECK(!r.recovery_required && !r.verified_existing, "unexpected recovery/existing flags");
  CHECK(r.last_ok == TU_STAGE_DONE && r.failed_at == TU_STAGE_NONE, "stages: last_ok=%d failed=%d", r.last_ok, r.failed_at);
  CHECK(r.chip == 0x192000a1, "chip %#x", r.chip);
  CHECK(r.feat_entry == 0xffffff8f && r.ss0_entry == 0x44023025 && r.ss1_entry == 0x7, "cold snapshot wrong");
  CHECK(r.readout_valid[2] && r.readout[2] == 0, "selectors-applied readout %#x", r.readout[2]);
  CHECK(r.readout_valid[3] && r.readout[3] == 0, "pre-FLR readout %#x", r.readout[3]);
  CHECK(r.readout_valid[4] && r.readout[4] == 0, "post-FLR readout %#x", r.readout[4]);
  CHECK(r.ss0_final == 0x88888888 && r.ss1_final == 0x8 && r.feat_final == ~0u, "retention lost: %x/%x/%x",
        r.ss0_final, r.ss1_final, r.feat_final);
  CHECK(r.perm_after[0] == 0xfffff0ff && r.perm_after[2] == ~0u && r.perm_after[3] == ~0u, "required permissions wrong");
  CHECK(r.perm_after[1] == 0xffffff0f, "optional FBPA should stay at its cold observed value, got %#x", r.perm_after[1]);
  CHECK(F.sec2_applied && F.sec2_saw_magic && F.sec2_saw_bme, "SEC2 publication gates not exercised");
  CHECK(F.nreset == 1, "expected exactly one FLR, got %d", F.nreset);
  CHECK(r.ndma_alloc == 7 && r.ndma_freed == 7, "buffers: alloc %d freed %d", r.ndma_alloc, r.ndma_freed);
  CHECK(F.meta_sig_at_selector == F.dma[3].iova, "selectors applied while the V67 was still published (sig=%#llx, stock=%#llx)",
        (unsigned long long)F.meta_sig_at_selector, (unsigned long long)F.dma[3].iova);
  for (int i = 0; i < r.nbuffer; i++)
    CHECK(r.buffers[i].iova < (1ull << 32), "buffer %s IOVA above 4 GiB", r.buffers[i].name);
  CHECK(cfg_rd(NULL, 4, 2) == 0x3, "command not restored to the claim-time state: %#x", cfg_rd(NULL, 4, 2));
  CHECK(F.regs[0x1fa828 / 4] == 0, "WPR2 still up at the end");
  // the booter copy carried the fuse-selected signature (index 0 for fuse version 1) when SEC2 ran
  CHECK(F.booter_patched, "booter signature patch missing/wrong");
}

static void test_verified_existing(env_t *e) {
  printf("retained override: verified-existing, no transaction\n");
  fake_init(K_NONE);
  F.regs[0x823804 / 4] = ~0u;
  F.regs[0x82381c / 4] = 0x88888888;
  F.regs[0x823820 / 4] = 0x8;
  tensor_unlock_result_t r;
  tensor_unlock_abort = 0;
  int rc = tensor_unlock_run_with(&F.pci, e->assets, &r);
  CHECK(rc == 0 && r.verified_existing, "verified-existing not taken: %s", r.first_error);
  CHECK(r.ndma_alloc == 0 && F.nwrites == 0 && F.nreset == 0, "verified-existing must not allocate, write or reset");
  CHECK(r.readout_valid[0] && r.readout[0] == 0, "existing readout not verified");
  F.pci.close(&F.pci);
}

static void test_existing_bad_readout(env_t *e) {
  printf("retained override with a stuck readout is NOT verified-existing\n");
  fake_init(K_READOUT_STUCK);
  F.regs[0x823804 / 4] = ~0u;
  F.regs[0x82381c / 4] = 0x88888888;
  F.regs[0x823820 / 4] = 0x8;
  tensor_unlock_result_t r;
  tensor_unlock_abort = 0;
  int rc = tensor_unlock_run_with(&F.pci, e->assets, &r);
  CHECK(rc != 0 && !r.verified_existing, "a non-full readout was accepted as verified-existing");
  CHECK(r.error == EIO, "expected EIO, got %d", r.error);
  F.pci.close(&F.pci);
}

static void test_existing_warm(env_t *e) {
  printf("retained override with WPR2 up: not a cold handoff\n");
  fake_init(K_NONE);
  F.regs[0x823804 / 4] = ~0u;
  F.regs[0x1fa828 / 4] = 0x100;
  tensor_unlock_result_t r;
  tensor_unlock_abort = 0;
  int rc = tensor_unlock_run_with(&F.pci, e->assets, &r);
  CHECK(rc != 0 && r.error == EBUSY && !r.verified_existing, "warm retained card accepted (err %d)", r.error);
  F.pci.close(&F.pci);
}

static void expect_early_fail(env_t *e, int knob, int err, int stage, const char *what) {
  tensor_unlock_result_t r;
  int rc = run_case(e, knob, &r);
  CHECK(rc != 0 && r.error == err, "%s: expected errno %d at %s, got %d (%s)", what, err,
        tensor_unlock_stage_name(stage), r.error, r.first_error);
  CHECK(r.failed_at == stage, "%s: failed at %s, expected %s", what, tensor_unlock_stage_name(r.failed_at),
        tensor_unlock_stage_name(stage));
  CHECK(!r.recovery_required, "%s: no firmware started, so no recovery may be required", what);
  CHECK(r.ndma_freed == r.ndma_alloc, "%s: pre-firmware failure must release every buffer", what);
  CHECK(F.nstarts == 0, "%s: a falcon was started", what);
}

static void expect_recovery(env_t *e, int knob, int stage, const char *what) {
  tensor_unlock_result_t r;
  int rc = run_case(e, knob, &r);
  CHECK(rc != 0, "%s: unexpectedly succeeded", what);
  CHECK(r.failed_at == stage, "%s: failed at %s, expected %s (%s)", what, tensor_unlock_stage_name(r.failed_at),
        tensor_unlock_stage_name(stage), r.first_error);
  CHECK(r.recovery_required, "%s: firmware started and containment unproven, recovery must be required", what);
  CHECK(r.ndma_alloc > 0 && r.ndma_freed == 0, "%s: recovery must retain every buffer (alloc %d freed %d)", what,
        r.ndma_alloc, r.ndma_freed);
  CHECK(r.first_error[0] != 0, "%s: first error not preserved", what);
}

static void test_preflight(env_t *e) {
  (void)e;
  printf("preflight: read-only classification\n");
  tensor_unlock_result_t r;
  fake_init(K_NONE);
  CHECK(tensor_unlock_preflight(&F.pci, &r) == 0, "preflight failed");
  CHECK(!strcmp(r.classification, "stock-cold"), "classification %s", r.classification);
  CHECK(r.preflight && r.error == 0 && r.last_ok == TU_STAGE_CLAIM, "successful preflight contract fields are wrong");
  CHECK(!strcmp(tensor_unlock_status(&r), "preflight-ok"), "successful preflight status is %s",
        tensor_unlock_status(&r));
  CHECK(F.nwrites == 0 && F.ndma_alloc == 0 && F.nreset == 0, "preflight wrote/allocated/reset");
  CHECK(r.feat_entry == 0xffffff8f && r.readout[0] == 0x00410080, "cold snapshot wrong");
  F.pci.close(&F.pci);

  fake_init(K_NONE);
  F.regs[0x823804 / 4] = ~0u;
  F.regs[0x82381c / 4] = 0x88888888;
  F.regs[0x823820 / 4] = 0x8;
  CHECK(tensor_unlock_preflight(&F.pci, &r) == 0, "preflight failed");
  CHECK(!strncmp(r.classification, "retained-unlock", 15), "classification %s", r.classification);
  F.pci.close(&F.pci);

  fake_init(K_WRONG_DEVID);
  CHECK(tensor_unlock_preflight(&F.pci, &r) != 0, "wrong device accepted");
  CHECK(!strcmp(r.classification, "wrong-device"), "classification %s", r.classification);
  CHECK(r.preflight && r.error == ENODEV && r.failed_at == TU_STAGE_CLAIM && r.first_error[0],
        "wrong-device preflight did not retain its refusal");
  CHECK(!strcmp(tensor_unlock_status(&r), "unsupported"), "wrong-device preflight status is %s",
        tensor_unlock_status(&r));
  F.pci.close(&F.pci);

  fake_init(K_NONE);
  F.regs[0x1fa828 / 4] = ~0u; // published WPR2 high word is an inaccessible MMIO read
  CHECK(tensor_unlock_preflight(&F.pci, &r) != 0, "inaccessible MMIO accepted");
  CHECK(!strcmp(r.classification, "inaccessible"), "classification %s", r.classification);
  CHECK(r.preflight && r.error == EIO && r.failed_at == TU_STAGE_CLAIM && r.first_error[0],
        "inaccessible preflight did not retain its I/O error");
  CHECK(!strcmp(tensor_unlock_status(&r), "failed-preflight"), "inaccessible preflight status is %s",
        tensor_unlock_status(&r));
  F.pci.close(&F.pci);

  fake_init(K_WPR2_UP);
  CHECK(tensor_unlock_preflight(&F.pci, &r) == 0, "preflight failed");
  CHECK(!strncmp(r.classification, "warm", 4), "classification %s", r.classification);
  F.pci.close(&F.pci);
}

static void test_loader(env_t *e) {
  printf("loader: hash, size, geometry, symlink refusals\n");
  char path[512];
  struct tu_assets *a = NULL;
  // a flipped payload byte under unchanged pins
  snprintf(path, sizeof(path), "%s/%s", e->dir, e->pins.payload.name);
  FILE *f = fopen(path, "r+b");
  CHECK(f != NULL, "cannot reopen payload");
  if (f) {
    uint8_t b;
    if (fread(&b, 1, 1, f) == 1) {
      rewind(f);
      b ^= 0xff;
      fwrite(&b, 1, 1, f);
    }
    fclose(f);
  }
  CHECK(tensor_unlock_assets_load(e->dir, &e->pins, &a) != 0, "corrupt payload accepted");
  CHECK(a == NULL, "assets leaked on failure");
  // restore
  put_file(e->dir, e->pins.payload.name, synth_payload, PAYLOAD_SIZE);
  CHECK(tensor_unlock_assets_load(e->dir, &e->pins, &a) == 0, "restored bundle rejected: %s", tinynv_last_error());
  tensor_unlock_assets_free(a);
  a = NULL;
  // missing file
  snprintf(path, sizeof(path), "%s/%s", e->dir, e->pins.desc.name);
  unlink(path);
  CHECK(tensor_unlock_assets_load(e->dir, &e->pins, &a) != 0, "missing descriptor accepted");
  size_t n;
  uint8_t *desc = build_desc(&n);
  put_file(e->dir, e->pins.desc.name, desc, n);
  free(desc);
  // symlink refused even when the target hashes correctly
  snprintf(path, sizeof(path), "%s/%s", e->dir, e->pins.payload.name);
  char link[512];
  snprintf(link, sizeof(link), "%s/link.bin", e->dir);
  CHECK(symlink(path, link) == 0, "symlink setup failed");
  tensor_unlock_pins_t p2 = e->pins;
  p2.payload.name = "link.bin";
  CHECK(tensor_unlock_assets_load(e->dir, &p2, &a) != 0, "symlinked asset accepted");
  unlink(link);
  // wrong geometry under RE-HASHED pins: structure checks must be hash-independent
  uint8_t *bad = build_booter(&n, 1);
  put_file(e->dir, e->pins.booter.name, bad, n);
  tensor_unlock_pins_t p3 = e->pins;
  uint8_t d[32];
  char hex[65];
  tinynv_sha256(bad, n, d);
  tinynv_sha256_hex(d, hex);
  p3.booter.sha256 = hex;
  CHECK(tensor_unlock_assets_load(e->dir, &p3, &a) != 0, "re-hashed wrong-geometry booter accepted");
  free(bad);
  uint8_t *good = build_booter(&n, 0);
  put_file(e->dir, e->pins.booter.name, good, n);
  free(good);
  // wrong payload size under re-hashed pins
  put_file(e->dir, e->pins.payload.name, synth_payload, PAYLOAD_SIZE - 4);
  tensor_unlock_pins_t p4 = e->pins;
  tinynv_sha256(synth_payload, PAYLOAD_SIZE - 4, d);
  tinynv_sha256_hex(d, hex);
  p4.payload.sha256 = hex;
  p4.payload.size = PAYLOAD_SIZE - 4;
  CHECK(tensor_unlock_assets_load(e->dir, &p4, &a) != 0, "short payload accepted");
  put_file(e->dir, e->pins.payload.name, synth_payload, PAYLOAD_SIZE);
  // and the restored bundle loads again
  CHECK(tensor_unlock_assets_load(e->dir, &e->pins, &a) == 0, "restored bundle rejected: %s", tinynv_last_error());
  tensor_unlock_assets_free(a);
}

static void test_real_bundle(const char *dir) {
  if (!dir || !*dir) { printf("  (no bundle path given, real-bundle arm skipped)\n"); return; }
  struct stat st;
  if (stat(dir, &st) || !S_ISDIR(st.st_mode)) {
    printf("  SKIPPED: prepared bundle not present at %s (run tools/rtx4090-unlock/prepare_bundle.py)\n", dir);
    return;
  }
  printf("real bundle: production pins, offline hash + geometry check\n");
  struct tu_assets *a = NULL;
  int rc = tensor_unlock_assets_load(dir, &tensor_unlock_pins, &a);
  CHECK(rc == 0, "real bundle rejected by the C loader: %s", tinynv_last_error());
  if (!rc) {
    CHECK(a->booter_image_size == 56832, "booter image size %u", a->booter_image_size);
    CHECK(a->sig_span.size == 4096, "signature span %llu", (unsigned long long)a->sig_span.size);
    CHECK(a->boot_desc.code_offset == 18432 && a->boot_desc.data_offset == 2048, "boot descriptor offsets wrong");
    tensor_unlock_assets_free(a);
  }
}

static void test_parsers(void) {
  printf("vendored parsers: spot checks\n");
  CHECK(gpu_unlock_fuse_version(0) == 0, "fuse 0");
  CHECK(gpu_unlock_fuse_version(1) == 1, "fuse 1");
  CHECK(gpu_unlock_fuse_version(2) == 2, "fuse 2");
  CHECK(gpu_unlock_fuse_version(0x8000) == 16, "fuse 0x8000");
  struct gpu_unlock_radix_plan plan;
  CHECK(gpu_unlock_radix_plan(8192, &plan) == 0 && plan.pages[3] == 2 && plan.table_pages == 3, "radix plan 8192");
  CHECK(gpu_unlock_radix_plan(0, &plan) != 0, "zero-size radix plan accepted");
  size_t n;
  uint8_t *desc = build_desc(&n);
  struct gpu_unlock_boot_desc bd;
  CHECK(gpu_unlock_boot_desc_parse(desc, n, BOOT_SIZE, &bd) == 0, "synthetic descriptor rejected");
  desc[0] = 4; // wrong version
  CHECK(gpu_unlock_boot_desc_parse(desc, n, BOOT_SIZE, &bd) != 0, "version-4 descriptor accepted");
  free(desc);
}

// The post-boot handoff classifier (pure; the live read in tinynv.c feeds it the same fields).
// The gates must match the transaction's: FEAT open + selectors + FULL effective readout, and an inaccessible
// register is never evidence of success — including the dead-bus case where FEAT's legitimate OPEN value (~0)
// looks exactly like an all-ones read.
static void test_postboot_classify(void) {
  printf("post-boot handoff classifier\n");
  tensor_unlock_state_t s;
#define PB(chip_, feat_, readout_, ss0_, ss1_) \
  do { memset(&s, 0, sizeof s); s.chip = (chip_); s.feat = (feat_); s.readout = (readout_); s.ss0 = (ss0_); s.ss1 = (ss1_); \
       tensor_unlock_state_classify(&s); } while (0)
  PB(0x192000a1u, ~0u, 0xf8000000u, TU_UNLOCK_SS0, TU_UNLOCK_SS1);
  CHECK(s.valid && s.unlocked && !s.stock, "unlocked state not recognized (readout bits outside the mask are legal)");
  PB(0x192000a1u, ~0u, 0u, TU_UNLOCK_SS0, TU_UNLOCK_SS1);
  CHECK(s.valid && s.unlocked, "unlocked state with zero readout not recognized");
  PB(0x192000a1u, 0xffffff8fu, 0x00410080u, 0x54414174u, 0u);
  CHECK(s.valid && s.stock && !s.unlocked, "stock state not recognized");
  PB(~0u, ~0u, ~0u, ~0u, ~0u);
  CHECK(!s.valid && !s.unlocked && !s.stock, "dead bus (all-ones everywhere) classified as a state");
  PB(0x192000a1u, ~0u, 0xbadf0000u, TU_UNLOCK_SS0, TU_UNLOCK_SS1);
  CHECK(!s.valid && !s.unlocked, "badf readout accepted");
  PB(0x192000a1u, 0xbadf1234u, 0u, TU_UNLOCK_SS0, TU_UNLOCK_SS1);
  CHECK(!s.valid && !s.unlocked, "badf FEAT accepted");
  PB(0x192000a1u, ~0u, 0u, TU_UNLOCK_SS0, 0x7u);
  CHECK(s.valid && !s.unlocked && !s.stock, "wrong SS1 still called unlocked");
  PB(0x192000a1u, ~0u, 0x00410080u, TU_UNLOCK_SS0, TU_UNLOCK_SS1);
  CHECK(s.valid && !s.unlocked, "stuck effective readout still called unlocked");
  PB(0x192000a1u, ~0u, 0u, 0x54414174u, 0u);
  CHECK(s.valid && !s.unlocked && !s.stock, "ambiguous retained state (FEAT open, stock selectors) called stock");
  PB(0u, ~0u, 0u, TU_UNLOCK_SS0, TU_UNLOCK_SS1);
  CHECK(!s.valid && !s.unlocked, "zero chip id accepted");
  PB(0xbadf0000u, ~0u, 0u, TU_UNLOCK_SS0, TU_UNLOCK_SS1);
  CHECK(!s.valid && !s.unlocked, "badf chip id accepted");
#undef PB
}

// The runtime option parser and the specification's five final statuses (pure; the gate in tinynv.c uses these).
static void test_option_and_status(void) {
  printf("runtime option parsing and status naming\n");
  CHECK(tensor_unlock_option_parse(NULL) == 0, "unset option is off");
  CHECK(tensor_unlock_option_parse("") == 0, "empty option is absence (off)");
  CHECK(tensor_unlock_option_parse("0") == 0, "0 is off");
  CHECK(tensor_unlock_option_parse("1") == 1, "1 is on");
  CHECK(tensor_unlock_option_parse("2") == -1, "2 refused");
  CHECK(tensor_unlock_option_parse("01") == -1, "01 refused (exact strings only)");
  CHECK(tensor_unlock_option_parse("yes") == -1, "yes refused");
  CHECK(tensor_unlock_option_parse("true") == -1, "true refused");
  CHECK(tensor_unlock_option_parse(" 1") == -1, "whitespace-padded 1 refused");
  CHECK(tensor_unlock_option_parse("1 ") == -1, "trailing-space 1 refused");
  tensor_unlock_result_t r;
#define ST(err_, stage_, existing_) \
  do { memset(&r, 0, sizeof r); r.error = (err_); r.failed_at = (stage_); r.verified_existing = (existing_); } while (0)
  ST(0, TU_STAGE_NONE, 0);
  CHECK(!strcmp(tensor_unlock_status(&r), "applied-and-verified"), "success maps to applied-and-verified");
  ST(0, TU_STAGE_NONE, 1);
  CHECK(!strcmp(tensor_unlock_status(&r), "verified-existing"), "retained success maps to verified-existing");
  ST(ENODEV, TU_STAGE_CLAIM, 0);
  CHECK(!strcmp(tensor_unlock_status(&r), "unsupported"), "wrong device/chip maps to unsupported");
  ST(EIO, TU_STAGE_ASSETS, 0);
  CHECK(!strcmp(tensor_unlock_status(&r), "failed-preflight"), "asset failure maps to failed-preflight");
  ST(EBUSY, TU_STAGE_GFW, 0);
  CHECK(!strcmp(tensor_unlock_status(&r), "failed-preflight"), "warm-card refusal maps to failed-preflight");
  ST(EIO, TU_STAGE_DMA, 0);
  CHECK(!strcmp(tensor_unlock_status(&r), "failed-preflight"), "DMA placement refusal maps to failed-preflight (no firmware yet)");
  ST(EIO, TU_STAGE_FWSEC, 0);
  CHECK(!strcmp(tensor_unlock_status(&r), "failed-after-start"), "FWSEC failure maps to failed-after-start");
  ST(ETIMEDOUT, TU_STAGE_PAYLOAD, 0);
  CHECK(!strcmp(tensor_unlock_status(&r), "failed-after-start"), "payload timeout maps to failed-after-start");
  ST(EIO, TU_STAGE_RESET, 0);
  CHECK(!strcmp(tensor_unlock_status(&r), "failed-after-start"), "reset failure maps to failed-after-start");
  ST(EOPNOTSUPP, TU_STAGE_CLAIM, 0);
  CHECK(!strcmp(tensor_unlock_status(&r), "failed-preflight"), "no-FLR capability is a preflight failure, not unsupported-device");
#undef ST
}

int main(int argc, char **argv) {
  tensor_unlock_set_time_scale(0.002); // 5 s budgets become 10 ms; the gates themselves are unchanged
  env_t e;
  env_setup(&e);
  if (env_load(&e)) { printf("cannot continue without the synthetic bundle\n"); return 1; }

  test_parsers();
  test_postboot_classify();
  test_option_and_status();
  test_happy_path(&e);
  test_verified_existing(&e);
  test_existing_bad_readout(&e);
  test_existing_warm(&e);
  test_preflight(&e);

  printf("eligibility and cold gates\n");
  expect_early_fail(&e, K_WRONG_DEVID, ENODEV, TU_STAGE_CLAIM, "wrong PCI device (4070 Ti SUPER id)");
  expect_early_fail(&e, K_WRONG_CHIP, ENODEV, TU_STAGE_CLAIM, "wrong chip arch");
  expect_early_fail(&e, K_CHIP_ALLONES, ENODEV, TU_STAGE_CLAIM, "all-ones PMC");
  expect_early_fail(&e, K_NO_FLR, EOPNOTSUPP, TU_STAGE_CLAIM, "no FLR capability");
  expect_early_fail(&e, K_NOT_D0, EOPNOTSUPP, TU_STAGE_CLAIM, "not in D0");
  expect_early_fail(&e, K_CMD_DROP, EBUSY, TU_STAGE_CLAIM, "command writes dropped");
  expect_early_fail(&e, K_WPR2_UP, EBUSY, TU_STAGE_GFW, "WPR2 up at entry");
  expect_early_fail(&e, K_GFW_HANG, ETIMEDOUT, TU_STAGE_GFW, "ROM boot never completes");
  {
    // DEVSTA bit 0 is Correctable Error Detected, not Transactions Pending. It must not block containment.
    tensor_unlock_result_t r;
    int rc = run_case(&e, K_DEVSTA_BIT0, &r);
    CHECK(rc == 0, "post-FLR DEVSTA correctable-error bit was mistaken for a pending transaction: %s", r.first_error);
  }
  expect_recovery(&e, K_DEVSTA_PENDING, TU_STAGE_RESET, "post-FLR Transactions Pending bit");

  printf("DMA placement gates (audit §3): refuse before publication\n");
  expect_early_fail(&e, K_HIGH_IOVA, EIO, TU_STAGE_DMA, "IOVA above 4 GiB");
  expect_early_fail(&e, K_NONCONTIG, EIO, TU_STAGE_DMA, "non-contiguous IOVA run");
  expect_early_fail(&e, K_MISALIGN, EIO, TU_STAGE_DMA, "misaligned IOVA");

  printf("FWSEC stage\n");
  expect_early_fail(&e, K_ROM_GARBAGE, ENOENT, TU_STAGE_FWSEC, "no FWSEC in ROM");
  expect_early_fail(&e, K_FWSEC_BAD_PKC, ENOENT, TU_STAGE_FWSEC, "FWSEC signature slot outside DMEM");
  expect_early_fail(&e, K_FWSEC_BAD_SIGNATURE, ENOENT, TU_STAGE_FWSEC, "FWSEC signature count mismatch");
  expect_recovery(&e, K_FWSEC_FAIL, TU_STAGE_FWSEC, "FWSEC ACR scratch failure");
  expect_recovery(&e, K_FWSEC_WPR2_WRONG, TU_STAGE_FWSEC, "FWSEC WPR2 not the FRTS reservation");

  printf("SEC2 payload stage: marker-only is never success\n");
  expect_recovery(&e, K_MARKER_ONLY, TU_STAGE_PAYLOAD, "marker without halt or permissions");
  expect_recovery(&e, K_NO_PERMS, TU_STAGE_PAYLOAD, "halt + marker without permissions");
  expect_recovery(&e, K_WPR_STUCK, TU_STAGE_PAYLOAD, "required WPR permission not applied");
  expect_recovery(&e, K_CORRUPT_PAYLOAD, TU_STAGE_PAYLOAD, "payload copy corrupted before SEC2 start");

  printf("selectors and effective readout\n");
  expect_recovery(&e, K_SELECTOR_DROP, TU_STAGE_SELECTORS, "SS1 write does not take");
  expect_recovery(&e, K_READOUT_STUCK, TU_STAGE_SELECTORS, "effective readout stuck after selectors");

  printf("reset: status is meaningless, effects are the evidence\n");
  expect_recovery(&e, K_RESET_DEAD, TU_STAGE_RESET, "identity lost after FLR");
  expect_recovery(&e, K_RESET_WPR2_UP, TU_STAGE_RESET, "WPR2 still up after FLR");
  expect_recovery(&e, K_RESET_NO_GFW, TU_STAGE_RESET, "ROM boot does not return after FLR");
  {
    // retention loss is detected AFTER containment was proven (upstream ordering): contained failure, not recovery
    tensor_unlock_result_t r;
    int rc = run_case(&e, K_RETENTION_LOSS, &r);
    CHECK(rc != 0 && r.error == EIO && r.failed_at == TU_STAGE_RESET, "retention loss not caught");
    CHECK(!r.recovery_required && r.ndma_freed == r.ndma_alloc, "post-quiesce failure should be contained");
  }
  {
    // The booter id is never used as `id - 1` unless it is in the one-based fuse range.
    tensor_unlock_result_t r;
    uint8_t original = e.assets->hdr.ucode_id;
    fake_init(K_NONE);
    e.assets->hdr.ucode_id = 0;
    int rc = tensor_unlock_run_with(&F.pci, e.assets, &r);
    e.assets->hdr.ucode_id = original;
    CHECK(rc != 0 && r.error == EINVAL && r.failed_at == TU_STAGE_DMA,
          "booter ucode id 0 was not refused before launch (err=%d stage=%s)", r.error,
          tensor_unlock_stage_name(r.failed_at));
    CHECK(!r.recovery_required && F.nstarts == 0 && F.ndma_free == F.ndma_alloc,
          "booter ucode id 0 refusal did not stay contained");
    F.pci.close(&F.pci);
  }
  {
    // a reset that clears BARs/COMMAND is repaired from the claim snapshot and the transaction still completes
    tensor_unlock_result_t r;
    int rc = run_case(&e, K_RESET_CLEARS_BARS, &r);
    CHECK(rc == 0, "BAR-clearing reset not repaired: %s", r.first_error);
  }

  printf("abort: catchable termination enters containment, never a silent disconnect\n");
  {
    // A signal flag from an earlier interruption is stale state, not an abort of the next transaction.
    tensor_unlock_result_t r;
    fake_init(K_NONE);
    tensor_unlock_abort = 1;
    int rc = tensor_unlock_run_with(&F.pci, e.assets, &r);
    CHECK(rc == 0, "stale abort flag prevented the next transaction: %s", r.first_error);
    CHECK(tensor_unlock_abort == 0, "run_with did not clear the stale abort flag");
    CHECK(F.nreset == 1 && F.nstarts == 2, "stale abort run did not execute the full transaction");
    F.pci.close(&F.pci);
    tensor_unlock_abort = 0;
  }
  expect_recovery(&e, K_ABORT_ON_SEC2, TU_STAGE_PAYLOAD, "abort while SEC2 runs");
  tensor_unlock_abort = 0;

  test_loader(&e);
  env_teardown(&e);
  test_real_bundle(argc > 1 ? argv[1] : NULL);

  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
