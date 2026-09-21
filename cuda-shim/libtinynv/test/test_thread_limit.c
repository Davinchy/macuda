// The most threads one block of a kernel can have - its __launch_bounds__, capped by its registers.
//
// A kernel with no bound used to be treated as good for 1024 threads. Torch's guest library has 1387 such kernels whose
// registers do not allow 1024 (B's sweep, D's rerun on the guest's own file), and a block too large for the register
// file is not refused by the hardware in any way this driver sees: like the shared-memory case measured on this card,
// it is accepted and never scheduled. So the limit is enforced here, before a card is involved.
//
// Two halves. The table checks tinynv_kernel_thread_limit against values worked from NVIDIA's rule by hand (cubin.c
// cites cuda_occupancy.h line by line). The fixture half runs the PUBLIC API on the null device against a real torch
// sm_120 image whose kernels declare no bound and carry 128 and 254 registers: kernel_info must report the limit, a
// launch at it must pass, and a launch one warp above it must be refused by name. A kernel that declares a bound its
// registers can meet must come out exactly as before.
#include "cubin.h"
#include "internal.h"
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static uint32_t lim(uint32_t regs, uint32_t bound) {
  tinynv_kernel_desc_t d; memset(&d, 0, sizeof d); d.regs = regs; d.max_threads = bound;
  return tinynv_kernel_thread_limit(&d);
}

int main(int argc, char **argv) {
  // regs -> per-warp allocation rounded to 256 -> the largest W with alloc * roundUp(W,4) <= 65536 and alloc * W <= 65536
  static const struct { uint32_t regs, bound, want; const char *why; } t[] = {
    {0, 0, 1024, "no registers counted: nothing to cap"},
    {32, 0, 1024, "32 regs: 1024 per warp, 32 warps = 32768"},
    {64, 0, 1024, "64 regs: 2048 per warp, 32 warps = 65536 exactly"},
    {65, 0, 896, "65 regs: 2080 -> 2304 per warp, 28 warps (65536/2304 = 28.4, rounded down to a multiple of 4)"},
    {72, 0, 896, "72 regs: 2304 per warp, 28 warps"},
    {80, 0, 768, "80 regs: 2560 per warp, 25.6 -> 24 warps"},
    {128, 0, 512, "128 regs: 4096 per warp, 16 warps"},
    {254, 0, 256, "254 regs: 8128 -> 8192 per warp, 8 warps"},
    {255, 0, 256, "255 regs: 8160 -> 8192 per warp, 8 warps"},
    {256, 0, 256, "256 regs: 8192 per warp, 8 warps - the per-thread maximum"},
    {257, 0, 0, "257 regs: over NVIDIA's per-thread maximum, no block fits"},
    {255, 256, 256, "a 256 bound the registers meet: unchanged"},
    {64, 128, 128, "a 128 bound with room to spare: unchanged"},
    {128, 1024, 512, "a 1024 bound the registers cannot meet: capped"},
    {64, 1000, 1000, "a bound that is not a warp multiple, registers allow 1024: the bound"},
  };
  printf("== the limit, against NVIDIA's rule worked by hand\n");
  for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
    uint32_t got = lim(t[i].regs, t[i].bound);
    CHECK(got == t[i].want, "regs %u bound %u: got %u, want %u (%s)", t[i].regs, t[i].bound, got, t[i].want, t[i].why);
  }
  printf("  %zu cases\n", sizeof t / sizeof t[0]);

  const char *path = argc > 1 ? argv[1] : "test/fixtures/torch-cu128-unbounded-regs.sm120.cubin";
  FILE *f = fopen(path, "rb");
  if (!f) { printf("  cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
  uint8_t *blob = malloc((size_t)len);
  if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) { printf("  cannot read %s\n", path); return 1; }
  fclose(f);
  tinynv_cubin_t cb;
  if (tinynv_cubin_parse(blob, (size_t)len, &cb)) { printf("  %s: %s\n", path, tinynv_last_error()); return 1; }

  printf("== %s through the public API on the null device\n", path);
  tinynv_device_t dev; tinynv_module_t mod; tinynv_stream_t s;
  CHECK(tinynv_init() == TINYNV_OK, "init");
  CHECK(tinynv_device_get(&dev, 0) == TINYNV_OK, "device_get");
  CHECK(tinynv_module_load(dev, blob, (size_t)len, &mod) == TINYNV_OK, "module_load: %s", tinynv_last_error());
  CHECK(tinynv_stream_create(dev, &s) == TINYNV_OK, "stream_create");
  static unsigned char params[4096];
  int capped = 0, bounded = 0;
  int before = fails;
  for (int i = 0; i < cb.nkernels && fails - before < 5; i++) {
    const tinynv_kernel_desc_t *d = &cb.kernels[i];
    uint32_t want = tinynv_kernel_thread_limit(d);
    tinynv_kernel_t k; tinynv_kernel_info_t info; memset(&info, 0, sizeof info);
    CHECK(tinynv_get_kernel(mod, d->name, &k) == TINYNV_OK, "get_kernel %s", d->name);
    CHECK(tinynv_kernel_info(k, &info) == TINYNV_OK, "kernel_info %s: %s", d->name, tinynv_last_error());
    CHECK(info.max_threads == (int)want, "%s: kernel_info reports %d, the limit is %u", d->name, info.max_threads, want);
    if (!d->max_threads && want < 1024) {
      capped++;
      CHECK(tinynv_launch(s, k, 1, 1, 1, want, 1, 1, 0, params, d->param_size) == TINYNV_OK,
            "%s (%u regs): a launch AT the limit %u was refused: %s", d->name, d->regs, want, tinynv_last_error());
      tinynv_status_t st = tinynv_launch(s, k, 1, 1, 1, want + 32, 1, 1, 0, params, d->param_size);
      CHECK(st != TINYNV_OK && strstr(tinynv_last_error(), "registers per thread allow at most"),
            "%s (%u regs): %u threads, one warp over the limit, was not refused by name: \"%s\"", d->name, d->regs,
            want + 32, st == TINYNV_OK ? "(accepted)" : tinynv_last_error());
      if (capped == 1) printf("  %u registers, no bound: kernel_info says %u; %u threads refused: %.160s...\n", d->regs,
                              want, want + 32, tinynv_last_error());
    } else if (d->max_threads) {
      bounded++;
      CHECK(want == d->max_threads, "%s: bound %u, registers %u fit it, yet the limit came out %u", d->name,
            d->max_threads, d->regs, want);
    }
  }
  CHECK(capped > 0, "the fixture has no unbounded kernel over 64 registers, so the launch half proved nothing");
  printf("  %d kernels capped by their registers and refused one warp over; %d bounded kernels unchanged\n", capped, bounded);
  tinynv_cubin_free(&cb);
  free(blob);
  printf(fails ? "%d of %d checks failed\n" : "all %d checks passed\n", fails ? fails : checks, checks);
  return fails ? 1 : 0;
}
