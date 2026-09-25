// Does a launch get the barriers its kernel declares? (card; B, 2026-09-24; the C8 FP8 fault, docs log 5764d199)
//
// C8's CUTLASS sm120 FP8 GEMM faulted every SM at BAR.SYNC 7 with "Illegal Instruction Parameter"; it declares
// EIATTR_NUM_BARRIERS 8 and libtinynv granted every launch BARRIER_COUNT 1. The alternative was a block too small for
// bar.sync's thread count. This program separates them with a block whose size is KNOWN (64) and a kernel that syncs on
// NAMED barrier 1 (bar.sync 1, 64; NUM_BARRIERS 2): fixture test/fixtures/namedbar.sm120a.cubin (a3c3a408).
//   fix       N0 hooks at 0; N1 nb_named, 64 threads: out[t] == 0xB0000000 + (t ^ 32) + 1 for all 64 (the named barrier
//             worked); N2 nb_plain (barrier 0, the control): the same values
//   mustfail  BARRIER_COUNT forced back to 1 (the pre-fix grant), nb_named once: PREDICTED to fault with Illegal Instruction
//             Parameter (read from the daemon/GSP log). A clean result here refutes the barrier explanation. Run LAST, in its
//             own process: it faults the card by design (chip reset at the next open, no replug).
#include "exec.h"
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static tinynv_status_t st;
#define T(c) do { if ((st = (c)) != TINYNV_OK) { printf("  STOP %s: %s\n    %s\n", #c, tinynv_status_str(st), tinynv_last_error()); return 1; } } while (0)

static int run(tinynv_stream_t s, tinynv_kernel_t k, tinynv_devptr_t out, const char *name) {
  static unsigned r[64];
  for (int i = 0; i < 64; i++) r[i] = 0xdeadbeefu;
  T(tinynv_memcpy_htod(s, out, r, sizeof r));
  uint8_t p[16]; memset(p, 0, sizeof p); uint64_t o = (uint64_t)out; memcpy(p, &o, 8);
  T(tinynv_launch(s, k, 1, 1, 1, 64, 1, 1, 0, p, 8));
  T(tinynv_stream_sync(s));
  T(tinynv_memcpy_dtoh(s, r, out, sizeof r));
  int bad = 0, first = -1;
  for (unsigned t = 0; t < 64; t++) if (r[t] != 0xB0000000u + (t ^ 32u) + 1u) { bad++; if (first < 0) first = (int)t; }
  printf("%s %s: %d of 64 wrong%s", name, bad ? "MISMATCH" : "MATCH", bad, bad ? "" : "\n");
  if (bad) printf(" (first t=%d read %#x, want %#x)\n", first, r[first], 0xB0000000u + ((unsigned)first ^ 32u) + 1u);
  return bad ? 2 : 0;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *mode = argc > 1 ? argv[1] : "fix";
  const char *path = argc > 2 ? argv[2] : "test/fixtures/namedbar.sm120a.cubin";
  if (!getenv("TINYNV_HW") || !getenv("TINYNV_SOCKET")) { printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s fix|mustfail [cubin]\n", argv[0]); return 2; }
  if (strcmp(mode, "fix") && strcmp(mode, "mustfail")) { printf("  STOP: mode is fix or mustfail\n"); return 2; }
  FILE *f = fopen(path, "rb"); if (!f) { printf("  STOP cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
  void *b = malloc((size_t)n); if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { printf("  STOP cannot read %s\n", path); return 1; }
  fclose(f);
  printf("libtinynv build %s\nmode %s\n", tinynv_build_id(), mode);
  printf("N0 ENTER: hooks at process start\n");
  if (tinynv_exec_test_barriers_force) { printf("  STOP N0: the barrier force hook is %u at start\n", tinynv_exec_test_barriers_force); return 1; }
  printf("N0 MATCH: barriers force 0\n");
  tinynv_device_t dev; tinynv_module_t mod; tinynv_kernel_t kn, kp; tinynv_stream_t s; tinynv_devptr_t out;
  T(tinynv_init()); T(tinynv_device_get(&dev, 0)); T(tinynv_module_load(dev, b, (size_t)n, &mod));
  T(tinynv_get_kernel(mod, "nb_named", &kn)); T(tinynv_get_kernel(mod, "nb_plain", &kp));
  T(tinynv_stream_create(dev, &s)); T(tinynv_malloc(dev, 4096, &out));
  int rc = 0;
  if (!strcmp(mode, "fix")) {
    printf("N1 ENTER: nb_named, 64 threads, named barrier 1, the declared grant\n"); rc |= run(s, kn, out, "N1");
    printf("N2 ENTER: nb_plain, 64 threads, barrier 0 (control)\n"); rc |= run(s, kp, out, "N2");
    printf("namedbar fix: %s\n", rc ? "FAILED" : "all rows met; mustfail may run");
  } else {
    tinynv_exec_test_barriers_force = 1;
    printf("M1 ENTER: nb_named with BARRIER_COUNT forced to 1 (the pre-fix grant) - PREDICTED: Illegal Instruction Parameter fault\n");
    fflush(stdout); fsync(fileno(stdout));
    rc = run(s, kn, out, "M1");
    printf("M1 returned rc %d: a clean MATCH here REFUTES the barrier explanation\n", rc);
  }
  free(b);
  return rc;
}
