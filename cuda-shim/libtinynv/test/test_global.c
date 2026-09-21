// A module's data object by name - what cuModuleGetGlobal answers - on the real cuBLAS image that cublasCreate asks it of.
//
// Fixture: libcublasLt.so.12 from nvidia-cublas-cu12 12.8.4.1 (the bench's x86 copy), fatbin 0 entry 10, sm_120,
// 13,616 bytes - the size C's H2 run reported for the image it asked about. It defines
// `_ZN6cublas8internal15deviceConstantsE` as an OBJECT, LOCAL, 112 bytes at offset 0 of .nv.global.init.
//
// What is proved, on the null device, where "device memory" is host memory:
//   the global is found and its size is the symbol's 112;
//   the 112 bytes at the returned address ARE that symbol's bytes in the file, so the offset is right, not merely plausible;
//   some relocation in the image targets exactly that offset, so the address a caller gets is the one kernels use;
//   a misspelt name, and a kernel's name, are refused by name.
#include "cubin.h"
#include "internal.h"
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)
static uint16_t r16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t r32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t r64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }

// The symbol's bytes straight out of the file, by a route that shares nothing with image.c: section header ->
// file offset of its section -> + st_value.
static const uint8_t *file_bytes_of(const uint8_t *e, const char *name, uint64_t *size) {
  uint64_t shoff = r64(e + 0x28); uint16_t se = r16(e + 0x3a), sn = r16(e + 0x3c);
  for (int i = 0; i < sn; i++) {
    const uint8_t *s = e + shoff + (uint64_t)i * se;
    if (r32(s + 4) != 2) continue;
    uint64_t o = r64(s + 0x18), sz = r64(s + 0x20);
    const uint8_t *str = e + r64(e + shoff + (uint64_t)r32(s + 0x28) * se + 0x18);
    for (uint64_t k = 0; k + 24 <= sz; k += 24) {
      const uint8_t *sy = e + o + k;
      if (strcmp((const char *)str + r32(sy), name)) continue;
      const uint8_t *tgt = e + shoff + (uint64_t)r16(sy + 6) * se;
      *size = r64(sy + 16);
      return e + r64(tgt + 0x18) + r64(sy + 8);
    }
  }
  return NULL;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "test/fixtures/cublaslt-cu128-deviceconstants.sm120.cubin";
  const char *want = "_ZN6cublas8internal15deviceConstantsE";
  FILE *f = fopen(path, "rb");
  if (!f) { printf("  cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
  uint8_t *blob = malloc((size_t)len);
  if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) { printf("  cannot read %s\n", path); return 1; }
  fclose(f);
  printf("== %s (%ld bytes)\n", path, len);

  uint64_t fsz = 0;
  const uint8_t *fbytes = file_bytes_of(blob, want, &fsz);
  CHECK(fbytes && fsz == 112, "the fixture itself does not define %s as 112 bytes", want);

  tinynv_device_t dev = NULL; tinynv_module_t mod = NULL;
  CHECK(tinynv_init() == TINYNV_OK, "init");
  CHECK(tinynv_device_get(&dev, 0) == TINYNV_OK, "device_get");
  CHECK(tinynv_module_load(dev, blob, (size_t)len, &mod) == TINYNV_OK, "module_load: %s", tinynv_last_error());
  tinynv_devptr_t addr = 0; size_t size = 0;
  CHECK(tinynv_get_global(mod, want, &addr, &size) == TINYNV_OK, "get_global: %s", tinynv_last_error());
  CHECK(size == 112, "size %zu, the symbol says 112", size);
  CHECK(addr && fbytes && !memcmp((const void *)(uintptr_t)addr, fbytes, 112),
        "the 112 bytes at the global's address are not the symbol's bytes in the file");
  printf("  %s: %zu bytes, and they are the symbol's own bytes from the file\n", want, size);

  // the offset against the relocations: an image-level check through the internal layout
  tinynv_cubin_t cb; tinynv_image_t im; uint64_t off = 0, sz = 0;
  CHECK(!tinynv_cubin_parse(blob, (size_t)len, &cb), "parse");
  CHECK(!tinynv_cubin_global(&cb, want, &off, &sz), "cubin_global: %s", tinynv_last_error());
  CHECK(!tinynv_cubin_image(&cb, cb.kernels[0].name, 1, &im), "image: %s", tinynv_last_error());
  int hits = 0;
  for (int i = 0; i < im.nrelocs; i++) hits += !im.relocs[i].undefined && im.relocs[i].target == off;
  CHECK(off + sz <= im.len, "offset %llu + %llu is outside the %zu byte image", (unsigned long long)off, (unsigned long long)sz, im.len);
  CHECK(hits > 0, "no relocation in the image targets the global's offset %llu, so nothing ties it to what kernels use",
        (unsigned long long)off);
  printf("  image offset %llu of %zu; %d relocation%s target exactly that offset\n", (unsigned long long)off, im.len, hits,
         hits == 1 ? "" : "s");

  tinynv_status_t st = tinynv_get_global(mod, "_ZN6cublas8internal15deviceConstantsX", &addr, &size);
  CHECK(st != TINYNV_OK && strstr(tinynv_last_error(), "deviceConstantsX"), "a misspelt name: status %d, \"%s\"", (int)st,
        tinynv_last_error());
  st = tinynv_get_global(mod, cb.kernels[0].name, &addr, &size);
  CHECK(st != TINYNV_OK && strstr(tinynv_last_error(), "not a data object"), "a kernel's name: status %d, \"%.120s\"",
        (int)st, tinynv_last_error());
  tinynv_image_free(&im);
  tinynv_cubin_free(&cb);
  CHECK(tinynv_module_unload(mod) == TINYNV_OK, "unload");
  free(blob);
  printf(fails ? "%d of %d checks failed\n" : "module globals: all %d checks passed\n", fails ? fails : checks, checks);
  return fails ? 1 : 0;
}
