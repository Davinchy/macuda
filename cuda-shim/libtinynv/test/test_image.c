// Checks the loadable image against the one tinygrad's elf_loader builds for the same cubin.
//
// This is the stage between reading a cubin and launching from it, and it is the one where a mistake does not announce
// itself: a kernel whose constant bank is placed at the wrong offset loads fine, launches fine, and reads a lookup table
// that is not there. Every IQ-quantised llama.cpp kernel depends on it - eleven relocated pointers each into a 26 KB
// table of __device__ data - which is why the corpus here is a real ggml cubin rather than the vector add.
//
// The image is compared by hash rather than byte by byte because it is 150 KB of code and constants; what is compared
// piece by piece is everything a launch is built out of - each kernel's program offset, each constant bank's offset and
// size, and every relocation's place, target and kind.
#include "cubin.h"
#include "sha256.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static long num_after(const char *s, const char *key) {
  const char *p = strstr(s, key);
  return p ? strtol(p + strlen(key), NULL, 0) : -1;
}

int main(int argc, char **argv) {
  const char *ref_path = argc > 1 ? argv[1] : "test/reference-image.txt";
  const char *cubin_path = argc > 2 ? argv[2] : "../spike/norm.cubin";

  FILE *f = fopen(cubin_path, "rb");
  if (!f) { printf("  cannot open %s\n", cubin_path); return 1; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  rewind(f);
  uint8_t *blob = malloc((size_t)len);
  if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) { printf("  cannot read %s\n", cubin_path); return 1; }
  fclose(f);

  tinynv_cubin_t cb;
  if (tinynv_cubin_parse(blob, (size_t)len, &cb)) { printf("  %s: %s\n", cubin_path, tinynv_last_error()); return 1; }

  FILE *r = fopen(ref_path, "r");
  if (!r) { printf("  cannot open %s: run tools/nv_reference_image.py\n", ref_path); return 1; }

  // The image is the same whichever kernel it is laid out for, so it is built once here from the first kernel named in
  // the reference and every kernel's offsets are read out of that one layout - which is also what module_load will do.
  tinynv_image_t im = {0};
  int have_image = 0, relocs_seen = 0, kernels_seen = 0;
  char line[4096];
  while (fgets(line, sizeof(line), r)) {
    line[strcspn(line, "\n")] = 0;
    if (line[0] == '#' || !line[0]) continue;

    if (!strncmp(line, "kernel ", 7)) {
      const char *nm = strstr(line, "name=");
      if (!nm) continue;
      char name[1024];
      snprintf(name, sizeof(name), "%s", nm + 5);
      name[strcspn(name, " ")] = 0;
      kernels_seen++;

      tinynv_image_t k = {0};
      if (tinynv_cubin_image(&cb, name, 1, &k)) { CHECK(0, "%s: %s", name, tinynv_last_error()); continue; }
      if (!have_image) { im = k; have_image = 1; } // keep the first, for the image and relocation checks below

      CHECK(k.text_off == (uint64_t)num_after(line, "text="), "%s: code at %#llx, the oracle placed it at %#llx",
            name, (unsigned long long)k.text_off, (unsigned long long)num_after(line, "text="));
      for (int b = 0; b < TINYNV_MAX_CONSTBUFS; b++) {
        char key[16];
        snprintf(key, sizeof(key), "cbuf%d=", b);
        const char *p = strstr(line, key);
        CHECK(!!p == !!k.constbuf[b].used, "%s: bank %d is %s here and %s in the oracle", name, b,
              k.constbuf[b].used ? "bound" : "unbound", p ? "bound" : "unbound");
        if (!p || !k.constbuf[b].used) continue;
        uint64_t off = strtoull(p + strlen(key), NULL, 0), size = strtoull(strchr(p, ':') + 1, NULL, 0);
        CHECK(k.constbuf[b].off == off && k.constbuf[b].size == size,
              "%s: bank %d at %#llx size %#llx, the oracle has %#llx size %#llx", name, b,
              (unsigned long long)k.constbuf[b].off, (unsigned long long)k.constbuf[b].size,
              (unsigned long long)off, (unsigned long long)size);
      }
      if (have_image && k.bytes != im.bytes) tinynv_image_free(&k);
      continue;
    }

    if (!strncmp(line, "image ", 6)) {
      // built from the first kernel in the file, which the loop above has not reached yet, so do it here
      if (!have_image) {
        const char *first = NULL;
        long pos = ftell(r);
        char peek[4096];
        while (fgets(peek, sizeof(peek), r)) if (!strncmp(peek, "kernel ", 7)) { first = strstr(peek, "name="); break; }
        fseek(r, pos, SEEK_SET);
        if (!first) { CHECK(0, "%s names no kernels", ref_path); continue; }
        char name[1024];
        snprintf(name, sizeof(name), "%s", first + 5);
        name[strcspn(name, " \n")] = 0;
        if (tinynv_cubin_image(&cb, name, 1, &im)) { CHECK(0, "%s: %s", name, tinynv_last_error()); continue; }
        have_image = 1;
      }
      // THE ORACLE LAYS OUT SHT_PROGBITS ONLY, AND THIS LOADER NOW DOES NOT, DELIBERATELY. Zero-initialised `.nv.global`
      // (SHT_NOBITS) used to be placed nowhere - here and in the oracle alike - so relocations against it resolved into
      // another section's bytes (image.c place()). It is now appended as zeroes after every PROGBITS section. So the
      // comparison is: the first `len=` bytes are the oracle's image byte for byte (the hash), and what follows is
      // exactly the NOBITS sections at their alignment - a length worked out here from the section table, not read back
      // from the loader - and nothing but zeroes.
      size_t olen = (size_t)num_after(line, "len="), want_len = olen;
      for (int i = 0; i < cb.nsections; i++) {
        const tinynv_section_t *z = &cb.sections[i];
        if (z->type != 8 /* SHT_NOBITS */ || !z->size || z->addr) continue;
        uint64_t al = z->align > 128 ? z->align : 128;
        want_len += (size_t)((al - want_len % al) % al) + (size_t)z->size;
      }
      CHECK(im.len == want_len, "the image is %zu bytes; the oracle built %zu and the NOBITS tail adds %zu", im.len, olen,
            want_len - olen);
      uint8_t digest[32];
      char hex[65];
      tinynv_sha256(im.bytes, im.len < olen ? im.len : olen, digest);
      tinynv_sha256_hex(digest, hex);
      const char *want = strstr(line, "sha256=");
      CHECK(want && !strcmp(hex, want + 7), "the image's first %zu bytes hash to %s, the oracle's image to %s", olen, hex,
            want ? want + 7 : "?");
      size_t nonzero = 0;
      for (size_t b = olen; b < im.len; b++) nonzero += im.bytes[b] != 0;
      CHECK(nonzero == 0, "%zu of the %zu bytes past the oracle's image are not zero", nonzero, im.len - olen);
      if (im.len > olen) printf("  the oracle's %zu bytes match; %zu bytes of zeroed .nv.global follow\n", olen, im.len - olen);
      continue;
    }

    if (!strncmp(line, "relocs ", 7)) {
      CHECK(have_image && im.nrelocs == num_after(line, "n="), "%d relocations, the oracle resolved %ld",
            have_image ? im.nrelocs : -1, num_after(line, "n="));
      continue;
    }

    if (!strncmp(line, "reloc ", 6) && have_image) {
      uint64_t at = (uint64_t)num_after(line, "at="), target = (uint64_t)num_after(line, "target=");
      uint32_t type = (uint32_t)num_after(line, "type=");
      long addend = num_after(line, "addend=");
      int found = 0;
      for (int i = 0; i < im.nrelocs && !found; i++)
        found = im.relocs[i].at == at && im.relocs[i].target == target && im.relocs[i].type == type &&
                im.relocs[i].addend == addend;
      CHECK(found, "the oracle resolves %#llx to %#llx (type %#x) and this loader does not",
            (unsigned long long)at, (unsigned long long)target, type);
      relocs_seen++;
      continue;
    }
  }
  fclose(r);

  // and the addresses actually land in the image once it has one
  if (have_image && im.nrelocs) {
    uint64_t va = 0x7f0011220000ull;
    uint64_t at = im.relocs[0].at, expect = va + im.relocs[0].target + (uint64_t)im.relocs[0].addend;
    CHECK(!tinynv_image_relocate(&im, va), "applying the relocations failed: %s", tinynv_last_error());
    uint64_t got = 0;
    for (int i = 0; i < 8; i++) got |= (uint64_t)im.bytes[at + i] << (8 * i);
    CHECK(im.relocs[0].type != 2 || got == expect, "the first relocation wrote %#llx where %#llx belongs",
          (unsigned long long)got, (unsigned long long)expect);
  }

  printf("  %zu byte image, %d relocations and %d kernels, all as the oracle lays them out\n",
         have_image ? im.len : 0, relocs_seen, kernels_seen);
  CHECK(kernels_seen > 0 && relocs_seen > 0, "%s covered %d kernels and %d relocations", ref_path, kernels_seen, relocs_seen);

  if (have_image) tinynv_image_free(&im);
  tinynv_cubin_free(&cb);
  free(blob);
  printf(fails ? "%d of %d checks failed\n" : "all %d checks passed\n", fails ? fails : checks, checks);
  return fails ? 1 : 0;
}
