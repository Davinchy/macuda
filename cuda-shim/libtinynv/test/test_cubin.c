// Reads a real cubin (the sm_120 vecadd from cuda-shim/spike) and checks every number the launch path depends on against
// what cuobjdump reports for the same file. The parameter base is the one that matters: 0x380 here, 0x160 on sm_8x.
#include "cubin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

const char *tinynv_last_error(void);

// the same kernel built for three architectures: the parameter base moves every time, which is why it is read per cubin
static const struct { const char *suffix; uint32_t arch, param_base; } ARCHES[] = {
  {"sm86", 86, 0x160}, {"sm90", 90, 0x210}, {"sm120", 120, 0x380},
};

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <cubin>\n", argv[0]); return 2; }
  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror("open"); return 1; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *buf = malloc((size_t)len);
  if (fread(buf, 1, (size_t)len, f) != (size_t)len) { perror("read"); return 1; }
  fclose(f);

  tinynv_cubin_t c;
  if (tinynv_cubin_parse(buf, (size_t)len, &c)) { fprintf(stderr, "parse: %s\n", tinynv_last_error()); return 1; }

  if (argc > 2 && !strcmp(argv[2], "-")) { // survey mode: no hand-checked expectations, just report what the reader makes of it
    printf("sm_%u, %d kernels\n", c.sm_arch, c.nkernels);
    size_t longest = 0;
    // Every kernel is checked; only the first few are printed. Checking just the ones that fit on screen is how a defect
    // in the three hundredth kernel of a module stays invisible, which is exactly what happened with long names.
    for (int i = 0; i < c.nkernels; i++) {
      const tinynv_kernel_desc_t *k = &c.kernels[i];
      if (strlen(k->name) > longest) longest = strlen(k->name);
      if (i < 6)
        printf("  %-42s text %#6llx  regs %3u  params %d at c[0][%#x]+%#x  smem %u\n", k->name,
               (unsigned long long)k->text_size, k->regs, k->nparams, k->param_base, k->param_size, k->static_smem);
      CHECK(k->text_size > 0, "%s has no code", k->name);
      // a kernel whose describing sections were not found reports no parameter base at all, which is the symptom a
      // truncated name produces: the sections exist, but under a name the reader built wrongly
      CHECK(k->param_base == 0x160 || k->param_base == 0x210 || k->param_base == 0x380,
            "kernel %d (%zu characters) has param_base %#x: its .nv.info section was not found", i, strlen(k->name), k->param_base);
      CHECK(k->param_base + k->param_size <= k->const0_size || !k->const0_size, "%s parameters overflow its constant bank", k->name);
      // and the symptom session A actually hit: the kernel is there, but asking for it by its own name does not find it
      CHECK(tinynv_cubin_kernel(&c, k->name) == k, "kernel %d cannot be looked up by the name the reader stored for it", i);
    }
    if (c.nkernels > 6) printf("  ... and %d more\n", c.nkernels - 6);
    printf("  longest kernel name: %zu characters\n", longest);
    tinynv_cubin_free(&c);
    free(buf);
    printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
    return fails ? 1 : 0;
  }

  uint32_t want_base = 0;
  for (size_t i = 0; i < sizeof(ARCHES) / sizeof(*ARCHES); i++)
    if (strstr(argv[1], ARCHES[i].suffix)) { CHECK(c.sm_arch == ARCHES[i].arch, "sm_%u for %s", c.sm_arch, argv[1]); want_base = ARCHES[i].param_base; }
  CHECK(want_base != 0, "%s is not one of the known test cubins", argv[1]);
  CHECK(c.nkernels == 1, "%d kernels", c.nkernels);

  const tinynv_kernel_desc_t *k = tinynv_cubin_kernel(&c, "vecadd");
  CHECK(k != NULL, "vecadd not found: %s", tinynv_last_error());
  if (k) {
    // cuobjdump: PARAM_CBANK 0x9 0x1c0380, CBANK_PARAM_SIZE 0x1c, four KPARAM_INFO records, REGCOUNT 12
    CHECK(k->param_base == want_base, "param_base %#x, expected %#x: it moves per architecture, so hardcoding one is the bug", k->param_base, want_base);
    CHECK(k->param_size == 0x1c, "param_size %#x", k->param_size);
    CHECK(k->nparams == 4, "%d params", k->nparams);
    const uint32_t off[] = {0, 8, 16, 24}, sz[] = {8, 8, 8, 4}; // three pointers then an int, as the shim marshals them
    for (int i = 0; i < 4 && i < k->nparams; i++) {
      CHECK(k->params[i].offset == off[i], "param %d offset %#x", i, k->params[i].offset);
      CHECK(k->params[i].size == sz[i], "param %d size %u", i, k->params[i].size);
    }
    CHECK(k->regs > 0 && k->regs < 256, "regs %u", k->regs);
    CHECK(k->text_size > 0, "text size %#llx", (unsigned long long)k->text_size);
    CHECK(k->param_base + k->param_size <= k->const0_size, "the parameters do not fit the constant bank");
    CHECK(k->static_smem == 0, "static smem %u", k->static_smem);
    printf("vecadd: sm_%u regs=%u text=%#llx params at c[0][%#x]+%#x:", c.sm_arch, k->regs, (unsigned long long)k->text_size,
           k->param_base, k->param_size);
    for (int i = 0; i < k->nparams; i++) printf(" [%d]=%#x/%u", i, k->params[i].offset, k->params[i].size);
    printf("\n");
  }

  // a truncated cubin must be refused, not read past its end
  tinynv_cubin_t bad;
  CHECK(tinynv_cubin_parse(buf, 0x40, &bad) != 0, "a truncated cubin was accepted");

  tinynv_cubin_free(&c);
  free(buf);
  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
