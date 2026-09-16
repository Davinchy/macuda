// What a launch costs the processor, with no GPU and no card time.
//
// This exists because the number it measures was being guessed at. Record-once-replay (HANDOFF item 4) is worth
// exactly the host work it removes per launch, and nobody had measured that work: the null device cannot, because its
// launch path prints a line and returns without ever calling into exec.c, so a null-device rate measures fprintf and
// the shim above it, not the descriptor building underneath.
//
// So this times the descriptor building on its own, which is the bulk of what exec.c does per launch on the processor:
// build the program's half of the QMD, apply the launch, attach the release, link it to its predecessor, and copy the
// constant buffer and parameters that go with it. It deliberately does NOT time the arena bookkeeping or the flush,
// which are per batch rather than per launch, so the number it prints is a floor on the host cost rather than the whole
// of it - and a floor is the honest shape for sizing a change that can only ever remove some of it.
//
// Reported as microseconds per launch next to the measured per-launch total on the card, because that ratio is the
// whole question: if the processor is a tenth of a launch, removing it cannot be the lever.
#include "qmd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + t.tv_nsec * 1e-9; }

// What a real decode launch carries, taken from the 27B's own kernels rather than invented: mul_mat_vec_q passes 160
// bytes of parameters, and the driver injects a 896 byte constant buffer zero for Blackwell.
#define PARAM_BYTES 160
#define CBUF0_BYTES 896
#define REPS 200000

int main(int argc, char **argv) {
  unsigned reps = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 0) : REPS;
  static uint8_t cbuf0[CBUF0_BYTES], params[PARAM_BYTES];
  static uint8_t dest[CBUF0_BYTES + PARAM_BYTES];
  memset(cbuf0, 0x5a, sizeof(cbuf0));
  memset(params, 0x3c, sizeof(params));

  tinynv_qmd_program_t prog;
  memset(&prog, 0, sizeof(prog));
  prog.regs = 40;
  prog.shmem = 0;
  prog.slm_per_thread = 0;
  prog.prog_size = 4096;
  prog.sass_version = tinynv_sass_version(0xa04);   // sm_120, this card
  prog.constbuf_size[0] = CBUF0_BYTES;
  prog.constbuf_used[0] = 1;

  tinynv_qmd_launch_t l;
  memset(&l, 0, sizeof(l));
  l.grid[0] = 14336; l.grid[1] = 1; l.grid[2] = 1;
  l.block[0] = 32; l.block[1] = 4; l.block[2] = 1;
  l.program_addr = 0x7f0000000000ull;
  l.constbuf_addr[0] = 0x7f0000100000ull;
  l.constbuf_set[0] = 1;

  static tinynv_qmd_t q[2];
  double t0 = now();
  for (unsigned i = 0; i < reps; i++) {
    tinynv_qmd_t *cur = &q[i & 1], *prev = &q[(i & 1) ^ 1];
    memset(cur, 0, sizeof(*cur));
    if (tinynv_qmd_program(cur, &prog)) return printf("  FAIL: program\n"), 1;
    if (tinynv_qmd_launch(cur, &l)) return printf("  FAIL: launch\n"), 1;
    if (tinynv_qmd_release(cur, 0x7f0000200000ull, i + 1, 0) < 0) return printf("  FAIL: release\n"), 1;
    if (i) tinynv_qmd_chain(prev, 0x7f0000300000ull + (i << 9), 1);
    // the two copies that ride with every launch
    memcpy(dest, cbuf0, sizeof(cbuf0));
    memcpy(dest + CBUF0_BYTES, params, sizeof(params));
  }
  double us = (now() - t0) * 1e6 / reps;

  printf("host cost of building one launch: %.3f us (%u reps, descriptor + release + chain + %d byte constant buffer "
         "+ %d byte parameters)\n", us, reps, CBUF0_BYTES, PARAM_BYTES);
  printf("  for scale, a 27B decode launch on the card currently takes 10.9 us end to end, of which about 4.6 is the\n"
         "  memory the kernel actually moves, so the processor is %.0f%% of a launch and %.0f%% of what is not memory.\n",
         us / 10.9 * 100.0, us / 6.3 * 100.0);
  printf("  a change that removes ALL host work per launch is therefore worth at most %.3f us, which at 1992 launches\n"
         "  a token moves 46.2 tok/s to %.1f.\n", us, 1000.0 / (21.65 - us * 1992 / 1000.0));
  return 0;
}
