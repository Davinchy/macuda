// The hash and the firmware reader, checked on their own.
//
// The hash matters more than it looks: it is the only thing standing between a wrong or tampered file on disk and code
// the GPU's secure boot executes. So it is checked against the published NIST vectors rather than against itself, and
// the loader is checked to refuse a file whose contents do not match what was asked for.
#include "fw.h"
#include "sha256.h"
#include "tinynv_pci.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static const struct { const char *in; int repeat; const char *want; } VECTORS[] = {
  {"", 1, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
  {"abc", 1, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
  {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 1,
   "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
  {"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu", 1,
   "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"},
  {"a", 1000000, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
};

static void hash_vectors(void) {
  for (size_t i = 0; i < sizeof(VECTORS) / sizeof(*VECTORS); i++) {
    tinynv_sha256_t c;
    uint8_t d[32];
    char hex[65];
    tinynv_sha256_init(&c);
    for (int r = 0; r < VECTORS[i].repeat; r++) tinynv_sha256_update(&c, VECTORS[i].in, strlen(VECTORS[i].in));
    tinynv_sha256_final(&c, d);
    tinynv_sha256_hex(d, hex);
    CHECK(!strcmp(hex, VECTORS[i].want), "vector %zu hashed to %s, expected %s", i, hex, VECTORS[i].want);
  }
  printf("sha-256: %zu published vectors, including the one million byte one\n", sizeof(VECTORS) / sizeof(*VECTORS));
}

// Each of the three images the boot path uses, loaded through the same call the driver makes, then taken apart.
static void firmware(void) {
  printf("firmware root: %s\n", *tinynv_fw_root() ? tinynv_fw_root() : "(nowhere configured)");
  struct { const char *dir, *name, *sha, *section; } want[] = {
    {"gb202", "fmc-" TINYNV_FW_VER ".bin", "cb59a35c1d4bd1274d7267fd10243c29f843ff41c851b9cbd59f5af2ddd7fece", "image"},
    {"gb202", "bootloader-" TINYNV_FW_VER ".bin", "d40b48e431d1707dc77af3605db358ed7a32ebfc2830eb74de2eddb4d3025071", NULL},
    {"ga102", "gsp-" TINYNV_FW_VER ".bin", "a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f", ".fwimage"},
  };
  for (size_t i = 0; i < sizeof(want) / sizeof(*want); i++) {
    tinynv_blob_t b;
    if (tinynv_fw_load(want[i].dir, want[i].name, want[i].sha, &b)) {
      printf("%-28s %s\n", want[i].name, tinynv_last_error());
      // not finding the firmware is a failure unless the reduced run was actually asked for
      CHECK(getenv("TINYNV_ALLOW_NO_FIRMWARE"), "%s is missing; run tools/fetch_firmware.sh or set TINYNV_ALLOW_NO_FIRMWARE=1",
            want[i].name);
      continue;
    }
    printf("%-28s %9zu bytes", want[i].name, b.size);
    if (want[i].section) {
      const uint8_t *p;
      size_t n;
      CHECK(!tinynv_elf_section(&b, want[i].section, &p, &n), "%s has no %s section", want[i].name, want[i].section);
      printf(", section %s %zu bytes", want[i].section, n);
      const uint8_t *q;
      size_t m;
      // a name that is a prefix of a real section must not match it: firmware sections are picked by exact name
      CHECK(tinynv_elf_section(&b, ".fwimag", &q, &m), "a prefix matched a section name in %s", want[i].name);
    }
    printf("\n");

    // the version is the content, so a wrong hash has to be refused however plausible the file looks
    tinynv_blob_t again;
    CHECK(tinynv_fw_load(want[i].dir, want[i].name, "00000000000000000000000000000000000000000000000000000000deadbeef", &again),
          "%s was accepted under the wrong hash", want[i].name);
    tinynv_fw_free(&b);
  }
}

int main(void) {
  hash_vectors();
  firmware();
  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
