#include "fw.h"
#include "internal.h"
#include "sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TINYNV_FW_DEFAULT_DIR
#define TINYNV_FW_DEFAULT_DIR "" // no build-time root: then the environment has to say, because guessing is what bit us
#endif

static const char *fw_root_override;

void tinynv_fw_set_root(const char *dir) { fw_root_override = dir; }

const char *tinynv_fw_root(void) {
  if (fw_root_override && *fw_root_override) return fw_root_override;
  const char *e = getenv("TINYNV_FW_DIR");
  if (e && *e) return e;
  return TINYNV_FW_DEFAULT_DIR;
}

int tinynv_fw_load(const char *chip_dir, const char *name, const char *sha256_hex, tinynv_blob_t *out) {
  char path[1024];
  const char *root = tinynv_fw_root();
  if (!*root) return tinynv_fail("no firmware directory is configured: set TINYNV_FW_DIR, or build with TINYNV_FW_DEFAULT_DIR");
  snprintf(path, sizeof(path), "%s/nvidia/%s/gsp/%s", root, chip_dir, name);
  FILE *f = fopen(path, "rb");
  if (!f) return tinynv_fail("firmware %s is missing (run tools/fetch_firmware.sh, or set TINYNV_FW_DIR)", path);

  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n <= 0) { fclose(f); return tinynv_fail("firmware %s is empty", path); }
  uint8_t *buf = malloc((size_t)n);
  if (!buf) { fclose(f); return tinynv_fail("out of memory for %ld bytes of firmware", n); }
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return tinynv_fail("firmware %s is short", path); }
  fclose(f);

  uint8_t digest[32];
  char hex[65];
  tinynv_sha256(buf, (size_t)n, digest);
  tinynv_sha256_hex(digest, hex);
  if (sha256_hex && strcmp(hex, sha256_hex)) {
    free(buf);
    return tinynv_fail("firmware %s hashes to %s, not the pinned %s: refusing to run it", path, hex, sha256_hex);
  }
  *out = (tinynv_blob_t){.data = buf, .size = (size_t)n};
  return 0;
}

void tinynv_fw_free(tinynv_blob_t *b) {
  free(b->data);
  b->data = NULL;
  b->size = 0;
}

// little endian reads that stay inside the blob, because a truncated firmware file must be an error rather than a walk
// off the end of an allocation
static int rd(const tinynv_blob_t *b, size_t off, size_t width, uint64_t *out) {
  if (off + width > b->size) return -1;
  uint64_t v = 0;
  for (size_t i = 0; i < width; i++) v |= (uint64_t)b->data[off + i] << (i * 8);
  *out = v;
  return 0;
}

int tinynv_elf_section(const tinynv_blob_t *b, const char *name, const uint8_t **out, size_t *out_len) {
  if (b->size < 64 || memcmp(b->data, "\x7f" "ELF", 4)) return tinynv_fail("firmware image is not an ELF");
  int is64 = b->data[4] == 2;
  uint64_t shoff, shentsize, shnum, shstrndx;
  if (rd(b, is64 ? 0x28 : 0x20, is64 ? 8 : 4, &shoff) || rd(b, is64 ? 0x3a : 0x2e, 2, &shentsize) ||
      rd(b, is64 ? 0x3c : 0x30, 2, &shnum) || rd(b, is64 ? 0x3e : 0x32, 2, &shstrndx))
    return tinynv_fail("firmware image has no readable ELF header");
  if (!shnum || shstrndx >= shnum) return tinynv_fail("firmware image has no section name table");

  const size_t o_name = 0, o_off = is64 ? 24 : 16, o_size = is64 ? 32 : 20;
  const size_t w = is64 ? 8 : 4;

  uint64_t strtab_off, strtab_size;
  if (rd(b, shoff + shstrndx * shentsize + o_off, w, &strtab_off) ||
      rd(b, shoff + shstrndx * shentsize + o_size, w, &strtab_size) || strtab_off + strtab_size > b->size)
    return tinynv_fail("firmware image's section name table is outside the file");

  for (uint64_t i = 0; i < shnum; i++) {
    uint64_t nm, off, size;
    if (rd(b, shoff + i * shentsize + o_name, 4, &nm) || rd(b, shoff + i * shentsize + o_off, w, &off) ||
        rd(b, shoff + i * shentsize + o_size, w, &size))
      return tinynv_fail("firmware image's section table is outside the file");
    if (nm >= strtab_size) continue;
    const char *sname = (const char *)b->data + strtab_off + nm;
    size_t avail = (size_t)(strtab_size - nm);
    if (!memchr(sname, 0, avail) || strcmp(sname, name)) continue; // unterminated inside the table is not a match
    if (off + size > b->size) return tinynv_fail("firmware section %s runs past the end of the file", name);
    *out = b->data + off;
    *out_len = (size_t)size;
    return 0;
  }
  return tinynv_fail("firmware image has no section called %s", name);
}
