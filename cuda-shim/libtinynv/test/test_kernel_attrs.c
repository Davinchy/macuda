// cuFuncGetAttribute values that come from the image, through tinynv_kernel_info on the null device.
//
// The expected values are NOT worked out here. They are what NVIDIA's own driver (580.126.18, RTX 3090) reported via
// cuFuncGetAttribute for these same cubins on 2026-09-21 (B, cuda-shim-b 68b6070 vm/rmtrace/kernel-attrs-20260921/). The
// cubins are fixtures built by CUDA 13.0 from attrs-props.cu / attrs-clusters.cu, one property per kernel:
//   attrs-p80.cubin     nvcc -cubin -arch=sm_80 attrs-props.cu
//   attrs-p75v86.cubin  nvcc -cubin -gencode arch=compute_75,code=sm_86 attrs-props.cu   (PTX 75, binary 86)
//   attrs-c120.cubin    nvcc -cubin -arch=sm_120 attrs-clusters.cu
// The sm_120 cluster values cannot be loaded on that card, so for those the expectation is the SOURCE's own
// __cluster_dims__ - decoded, not driver-checked, and include/tinynv.h says so.
//
// And the refusal: the same sm_80 cubin with its PTX note renamed carries no PTX version, so bit 5 of attr_reported
// must be CLEAR (the caller then refuses attribute 5 by name) rather than a 0 or a guess being reported.
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static unsigned char *slurp(const char *path, long *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END); *len = ftell(f); rewind(f);
  unsigned char *b = malloc((size_t)*len);
  if (b && fread(b, 1, (size_t)*len, f) != (size_t)*len) { free(b); b = NULL; }
  fclose(f);
  return b;
}

static tinynv_device_t dev;
static int info_of(const unsigned char *img, long len, const char *kernel, tinynv_kernel_info_t *out) {
  tinynv_module_t m; tinynv_kernel_t k;
  memset(out, 0, sizeof(*out));
  if (tinynv_module_load(dev, img, (size_t)len, &m) != TINYNV_OK) return -1;
  int rc = tinynv_get_kernel(m, kernel, &k) == TINYNV_OK && tinynv_kernel_info(k, out) == TINYNV_OK ? 0 : -1;
  tinynv_module_unload(m);
  return rc;
}

int main(int argc, char **argv) {
  const char *dir = argc > 1 ? argv[1] : "test/fixtures";
  char path[512];
  CHECK(tinynv_init() == TINYNV_OK && tinynv_device_get(&dev, 0) == TINYNV_OK, "init");

  // attrs-p80.cubin: the driver's answers for attributes 0, 2, 3, 5, 7
  static const struct { const char *k; int a0, a2, a3, a5, a7; } p80[] = {
    {"k_plain", 1024, 2048, 0, 80, 0},   {"k_local", 1024, 2048, 0, 80, 0},
    {"k_const", 1024, 2048, 0, 80, 0},   {"k_bounds", 128, 2048, 0, 80, 0},
    {"k_bigstack", 1024, 2048, 4096, 80, 0}, {"k_calls", 512, 2048, 1024, 80, 0},
    {"k_printf", 1024, 2048, 16, 80, 0},
  };
  long len; snprintf(path, sizeof path, "%s/attrs-p80.cubin", dir);
  unsigned char *b = slurp(path, &len);
  CHECK(b != NULL, "cannot read %s", path);
  printf("== %s against the driver's answers\n", path);
  for (size_t i = 0; b && i < sizeof p80 / sizeof p80[0]; i++) {
    tinynv_kernel_info_t ki;
    CHECK(!info_of(b, len, p80[i].k, &ki), "%s: kernel_info refused: %s", p80[i].k, tinynv_last_error());
    unsigned want_bits = (1u << 2) | (1u << 3) | (1u << 5) | (1u << 7) | (1u << 10) | (1u << 11) | (1u << 12) | (1u << 13);
    CHECK(ki.attr_reported == want_bits, "%s: attr_reported %#x, want %#x", p80[i].k, ki.attr_reported, want_bits);
    CHECK(ki.max_threads == p80[i].a0, "%s: max_threads %d, the driver said %d", p80[i].k, ki.max_threads, p80[i].a0);
    CHECK(ki.const_size_bytes == p80[i].a2, "%s: const_size %d, the driver said %d", p80[i].k, ki.const_size_bytes, p80[i].a2);
    CHECK(ki.local_size_bytes == p80[i].a3, "%s: local_size %d, the driver said %d", p80[i].k, ki.local_size_bytes, p80[i].a3);
    CHECK(ki.ptx_version == p80[i].a5, "%s: ptx_version %d, the driver said %d", p80[i].k, ki.ptx_version, p80[i].a5);
    CHECK(ki.cache_mode_ca == p80[i].a7, "%s: cache_mode_ca %d, the driver said %d", p80[i].k, ki.cache_mode_ca, p80[i].a7);
    CHECK(!ki.cluster_size_must_be_set && !ki.required_cluster[0] && !ki.required_cluster[1] && !ki.required_cluster[2],
          "%s: sm_80 carries no cluster settings, yet some were reported", p80[i].k);
  }

  // the refusal: rename the note (same length), so the image no longer carries a PTX version. The name is in the
  // fixture twice, measured: once in .shstrtab (the section's own name, which the parser reads) and once in .strtab
  // (its section symbol). Both are renamed; a count other than 2 means the fixture changed under this test.
  if (b) {
    int renamed = 0;
    for (long i = 0; i + 15 <= len; i++)
      if (!memcmp(b + i, ".note.nv.cuinfo", 15)) { memcpy(b + i, ".note.nv.xxxxxx", 15); renamed++; }
    CHECK(renamed == 2, "the note name was found %d times, not the 2 measured in this fixture", renamed);
    tinynv_kernel_info_t ki;
    CHECK(!info_of(b, len, "k_plain", &ki), "k_plain without its note: %s", tinynv_last_error());
    CHECK(!(ki.attr_reported & (1u << 5)), "with no PTX note, attribute 5 is still marked reported (value %d)", ki.ptx_version);
    CHECK(ki.attr_reported & (1u << 2), "the other attributes stopped being reported too");
    printf("  with the PTX note renamed: attr_reported %#x - attribute 5 is left for the caller to refuse\n", ki.attr_reported);
    free(b);
  }

  // attrs-p75v86.cubin: the same source as PTX 75 against a binary of 86. The driver gave the p80 answers for 0, 2, 3
  // and 7 and 75 for attribute 5 (its attribute 6 is 86), kernel for kernel.
  snprintf(path, sizeof path, "%s/attrs-p75v86.cubin", dir);
  b = slurp(path, &len);
  CHECK(b != NULL, "cannot read %s", path);
  printf("== %s against the driver's answers (PTX 75, binary 86)\n", path);
  for (size_t i = 0; b && i < sizeof p80 / sizeof p80[0]; i++) {
    tinynv_kernel_info_t ki;
    CHECK(!info_of(b, len, p80[i].k, &ki), "p75v86 %s: %s", p80[i].k, tinynv_last_error());
    CHECK(ki.ptx_version == 75, "p75v86 %s: ptx_version %d, the driver said 75", p80[i].k, ki.ptx_version);
    CHECK(ki.max_threads == p80[i].a0 && ki.const_size_bytes == p80[i].a2 && ki.local_size_bytes == p80[i].a3 &&
          ki.cache_mode_ca == p80[i].a7, "p75v86 %s: max_threads %d const %d local %d cache %d, the driver said %d %d %d %d",
          p80[i].k, ki.max_threads, ki.const_size_bytes, ki.local_size_bytes, ki.cache_mode_ca, p80[i].a0, p80[i].a2,
          p80[i].a3, p80[i].a7);
  }
  free(b);

  // attrs-c120.cubin: compile-time clusters, against the source's own __cluster_dims__
  static const struct { const char *k; int must, w, h, d; } c120[] = {
    {"c_plain", 0, 0, 0, 0}, {"c_dims2", 1, 2, 1, 1}, {"c_dims22", 1, 2, 2, 1}, {"c_maxrank", 0, 0, 0, 0},
  };
  snprintf(path, sizeof path, "%s/attrs-c120.cubin", dir);
  b = slurp(path, &len);
  CHECK(b != NULL, "cannot read %s", path);
  printf("== %s against attrs-clusters.cu (decoded, not driver-checked)\n", path);
  for (size_t i = 0; b && i < sizeof c120 / sizeof c120[0]; i++) {
    tinynv_kernel_info_t ki;
    CHECK(!info_of(b, len, c120[i].k, &ki), "%s: %s", c120[i].k, tinynv_last_error());
    CHECK(ki.cluster_size_must_be_set == c120[i].must && ki.required_cluster[0] == c120[i].w &&
          ki.required_cluster[1] == c120[i].h && ki.required_cluster[2] == c120[i].d,
          "%s: must_be_set %d dims (%d,%d,%d), the source says %d (%d,%d,%d)", c120[i].k, ki.cluster_size_must_be_set,
          ki.required_cluster[0], ki.required_cluster[1], ki.required_cluster[2], c120[i].must, c120[i].w, c120[i].h, c120[i].d);
    CHECK(ki.ptx_version == 120, "%s: ptx_version %d, want 120", c120[i].k, ki.ptx_version);
  }
  free(b);
  printf(fails ? "%d of %d checks failed\n" : "kernel attributes: all %d checks passed\n", fails ? fails : checks, checks);
  return fails ? 1 : 0;
}
