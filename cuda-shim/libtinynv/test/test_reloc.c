// What a cubin references and does not contain.
//
// There is exactly one such symbol in practice - `vprintf`, CUDA's device-side printf, which nvcc emits a relocation
// for whenever a kernel can print, including from an assert path that never runs. Across all 183 cubins of a full ggml
// build it is the only one, 115 times. This driver has no device runtime to point it at, so it resolves to zero: a
// kernel that actually prints then faults on a null call, which is honest, where refusing the cubin cost every
// quantised model's decode path.
//
// The negative half is the point of the file. Resolving anything undefined to zero would be a blanket that hides the
// next symbol that matters, so the same cubin is presented with the name changed and must be refused - and refused by
// name, because "relocation against a symbol this driver cannot resolve (section 0)" cost session A a bisection.
#include "cubin.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static const char *kernel_of(const tinynv_cubin_t *c) { return c->nkernels ? c->kernels[0].name : NULL; }

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "../spike/printf_kernel.sm120.cubin";
  FILE *f = fopen(path, "rb");
  if (!f) { printf("  cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  rewind(f);
  uint8_t *blob = malloc((size_t)len);
  if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) { printf("  cannot read %s\n", path); return 1; }
  fclose(f);

  tinynv_cubin_t cb;
  if (tinynv_cubin_parse(blob, (size_t)len, &cb)) { printf("  %s: %s\n", path, tinynv_last_error()); return 1; }
  const char *k = kernel_of(&cb);
  CHECK(k != NULL, "%s holds no kernels", path);
  if (!k) return 1;

  tinynv_image_t im;
  CHECK(!tinynv_cubin_image(&cb, k, 1, &im), "a cubin that can print was refused: %s", tinynv_last_error());
  if (!fails) {
    CHECK(im.unresolved == 1, "%d symbols resolved to zero, expected exactly the one printf", im.unresolved);
    // and the word it points at really is zero, rather than left holding whatever was there
    int at_zero = 1;
    for (int i = 0; i < im.nrelocs; i++)
      if (!im.relocs[i].target && im.relocs[i].at + 8 <= im.len)
        for (int b = 0; b < 8; b++) if (im.bytes[im.relocs[i].at + b]) at_zero = 0;
    CHECK(at_zero, "the unresolved relocation's place does not hold zero");
    printf("  a cubin that can print loads: %d relocations, %d resolved to zero, bank %d bound\n", im.nrelocs,
           im.unresolved, im.constbuf[4].used ? 4 : 0);
    tinynv_image_free(&im);
  }
  tinynv_cubin_free(&cb);

  // The same cubin with the symbol renamed. Nothing else about it changes, so a loader that accepts this one is
  // accepting undefined symbols in general rather than the one it has evidence about.
  uint8_t *renamed = malloc((size_t)len);
  memcpy(renamed, blob, (size_t)len);
  int hits = 0;
  for (long i = 0; i + 8 <= len; i++)
    if (!memcmp(renamed + i, "vprintf", 8)) { renamed[i + 6] = 'q'; hits++; }   // vprintf -> vprintq, same length
  CHECK(hits > 0, "the test could not find the symbol name to change, so it proved nothing");

  tinynv_cubin_t cb2;
  if (!tinynv_cubin_parse(renamed, (size_t)len, &cb2)) {
    tinynv_image_t im2;
    int refused = tinynv_cubin_image(&cb2, kernel_of(&cb2), 1, &im2) != 0;
    CHECK(refused, "a cubin referencing an unknown symbol was loaded anyway");
    if (refused) {
      CHECK(strstr(tinynv_last_error(), "vprintq") != NULL,
            "the refusal does not name the symbol: \"%s\"", tinynv_last_error());
      printf("  and with the symbol renamed it is refused: %s\n", tinynv_last_error());
    } else tinynv_image_free(&im2);
    tinynv_cubin_free(&cb2);
  }

  free(renamed);
  free(blob);
  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
