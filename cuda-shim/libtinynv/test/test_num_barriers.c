// The barriers a kernel declares reach the descriptor (B, 2026-09-24; the C8 FP8 fault, docs log 5764d199).
// CUTLASS's sm120 FP8 GEMM declares EIATTR_NUM_BARRIERS 8 and syncs on NAMED barrier 7; libtinynv granted every launch
// BARRIER_COUNT 1 and the card faulted every SM with "Illegal Instruction Parameter". Rows, card-free:
//   P1  the parser reads EIATTR_NUM_BARRIERS (0x4c) from captured compiler output: namedbar.sm120a.cubin (a3c3a408,
//       CUDA 13.0 nvcc -arch=sm_120a on the 3090): nb_named (bar.sync 1, 64) declares 2, nb_plain (__syncthreads) declares 1
//   Q1  the descriptor carries it: tinynv_qmd_program writes BARRIER_COUNT 2 for 2 and 8 for 8, read back from the bits
//   Q2  a program that declares nothing (0) is still granted 1 - every kernel before this change, and the oracle's value
#include "cubin.h"
#include "internal.h"
#include "qmd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static uint64_t granted(uint32_t barriers) {
  tinynv_qmd_t q;
  memset(&q, 0, sizeof q);
  tinynv_qmd_program_t p;
  memset(&p, 0, sizeof p);
  p.regs = 32; p.shmem = 0x400; p.prog_size = 0x100; p.sass_version = tinynv_sass_version(120); p.barriers = barriers;
  if (tinynv_qmd_program(&q, &p)) { printf("  qmd_program: %s\n", tinynv_last_error()); return ~0ull; }
  return tinynv_qmd_get(&q, TINYNV_QMD_F(BARRIER_COUNT));
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "test/fixtures/namedbar.sm120a.cubin";
  FILE *f = fopen(path, "rb");
  if (!f) { printf("  cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
  uint8_t *b = malloc((size_t)n);
  if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { printf("  cannot read %s\n", path); return 1; }
  fclose(f);
  tinynv_cubin_t c;
  if (tinynv_cubin_parse(b, (size_t)n, &c)) { printf("  parse: %s\n", tinynv_last_error()); return 1; }
  int seen = 0;
  for (int i = 0; i < c.nkernels; i++) {
    const tinynv_kernel_desc_t *k = &c.kernels[i];
    if (!strcmp(k->name, "nb_named")) { seen++; CHECK(k->num_barriers == 2, "P1: nb_named declares %u barriers; want 2", k->num_barriers); }
    if (!strcmp(k->name, "nb_plain")) { seen++; CHECK(k->num_barriers == 1, "P1: nb_plain declares %u barriers; want 1", k->num_barriers); }
    printf("  P1: %s num_barriers %u\n", k->name, k->num_barriers);
  }
  CHECK(seen == 2, "P1: found %d of the two kernels", seen);
  uint64_t g2 = granted(2), g8 = granted(8), g0 = granted(0);
  CHECK(g2 == 2, "Q1: 2 declared, BARRIER_COUNT %llu", (unsigned long long)g2);
  CHECK(g8 == 8, "Q1: 8 declared (the CUTLASS FP8 kernel), BARRIER_COUNT %llu", (unsigned long long)g8);
  CHECK(g0 == 1, "Q2: none declared, BARRIER_COUNT %llu; want 1", (unsigned long long)g0);
  printf("  Q1/Q2: BARRIER_COUNT %llu for 2, %llu for 8, %llu for none\n", (unsigned long long)g2, (unsigned long long)g8, (unsigned long long)g0);
  tinynv_cubin_free(&c);
  free(b);
  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
