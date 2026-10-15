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
//
// `__assertfail` is the second allowed name (torch's core ops; image.c says why). Pass a cubin that references it as
// the argument and the same arms run against it: the rename arm renames whichever allowed name the cubin carries.
//
// ZERO IS CHECKED WHERE THE GPU SEES IT, AFTER tinynv_image_relocate. This file used to check the image bytes BEFORE
// relocation, where an unresolved word is zero by construction, and passed while the relocation step wrote the
// image's own base address there. The post-relocation arm below fails against that code (the printf cubin's word
// read 0x100000000000, the VA passed in) and is the reason the arm exists.
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
  CHECK(!tinynv_cubin_image(&cb, k, 1, &im), "a cubin with an allowed undefined symbol was refused: %s", tinynv_last_error());
  if (!fails) {
    int flagged = 0;
    for (int i = 0; i < im.nrelocs; i++) flagged += im.relocs[i].undefined != 0;
    CHECK(im.unresolved > 0, "no symbol resolved to zero, so this cubin exercised nothing here");
    CHECK(flagged == im.unresolved, "%d relocations marked undefined but %d counted", flagged, im.unresolved);
    // the word it points at really is zero before relocation, rather than left holding whatever was there
    int at_zero = 1;
    for (int i = 0; i < im.nrelocs; i++)
      if (im.relocs[i].undefined && im.relocs[i].at + 8 <= im.len)
        for (int b = 0; b < 8; b++) if (im.bytes[im.relocs[i].at + b]) at_zero = 0;
    CHECK(at_zero, "an unresolved relocation's place does not hold zero before relocation");
    // ...AND AFTER, which is what the GPU is given: an absolute zero plus the addend, never the image's address.
    const uint64_t VA = 0x100000000000ull;
    CHECK(!tinynv_image_relocate(&im, VA), "relocation failed: %s", tinynv_last_error());
    int after_ok = 1;
    for (int i = 0; i < im.nrelocs; i++) {
      if (!im.relocs[i].undefined || im.relocs[i].type != 2 || im.relocs[i].at + 8 > im.len) continue;
      uint64_t w = 0;
      for (int b = 7; b >= 0; b--) w = w << 8 | im.bytes[im.relocs[i].at + b];
      if (w != (uint64_t)im.relocs[i].addend) {
        after_ok = 0;
        printf("  unresolved word at image+%#llx holds %#llx after relocation at %#llx%s\n",
               (unsigned long long)im.relocs[i].at, (unsigned long long)w, (unsigned long long)VA,
               w == VA + (uint64_t)im.relocs[i].addend ? " - THE IMAGE'S OWN BASE" : "");
      }
    }
    CHECK(after_ok, "an unresolved relocation is not zero where the GPU sees it");
    printf("  a cubin with %d unresolved symbol reference%s loads: %d relocations, zero before AND after relocation, "
           "bank %d bound\n", im.unresolved, im.unresolved == 1 ? "" : "s", im.nrelocs, im.constbuf[4].used ? 4 : 0);
    tinynv_image_free(&im);
  }
  tinynv_cubin_free(&cb);

  // The same cubin with an allowed symbol renamed. Nothing else about it changes, so a loader that accepts this one is
  // accepting undefined symbols in general rather than the ones it has evidence about. Each allowed name the cubin
  // carries gets its own arm; at least one must be present or the file proved nothing.
  static const struct { const char *from, *to; size_t n; } names[] = {
    {"vprintf", "vprintq", 8}, {"__assertfail", "__assertfaiq", 13}};   // same length, NUL included
  int tried = 0;
  for (size_t v = 0; v < sizeof names / sizeof names[0]; v++) {
    uint8_t *renamed = malloc((size_t)len);
    memcpy(renamed, blob, (size_t)len);
    int hits = 0;
    for (long i = 0; i + (long)names[v].n <= len; i++)
      if (!memcmp(renamed + i, names[v].from, names[v].n)) { memcpy(renamed + i, names[v].to, names[v].n); hits++; }
    if (!hits) { free(renamed); continue; }
    tried++;
    // The other allowed name, if present, stays allowed; the refusal must name the renamed one, which is checked.
    tinynv_cubin_t cb2;
    if (!tinynv_cubin_parse(renamed, (size_t)len, &cb2)) {
      tinynv_image_t im2;
      int refused = tinynv_cubin_image(&cb2, kernel_of(&cb2), 1, &im2) != 0;
      CHECK(refused, "a cubin referencing an unknown symbol (%s) was loaded anyway", names[v].to);
      if (refused) {
        CHECK(strstr(tinynv_last_error(), names[v].to) != NULL,
              "the refusal does not name the symbol: \"%s\"", tinynv_last_error());
        printf("  and with %s renamed it is refused: %s\n", names[v].from, tinynv_last_error());
      } else tinynv_image_free(&im2);
      tinynv_cubin_free(&cb2);
    }
    free(renamed);
  }
  CHECK(tried > 0, "the test found no allowed symbol name to change, so it proved nothing");

  free(blob);
  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
