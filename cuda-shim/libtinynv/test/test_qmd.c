// Checks the kernel descriptor writer against the one the oracle builds, byte for byte.
//
// The reference comes from tools/nv_reference_qmd.py, which runs tinygrad's own NVProgramData and QMD classes against
// the same cubin. That is the oracle itself rather than a second reading of it, which matters here more than it did for
// the command buffers: a descriptor is a hundred and thirty fields at bit positions that mostly do not fall on byte
// boundaries, and several are named for a shift they are not given.
//
// This side is not handed the answers. It reads the same cubin with libtinynv's own reader and derives the register
// count, the shared memory size and the program size itself, so a divergence in the cubin reader shows up here too. Only
// the card's own properties - local memory per thread, sass version, the two windows - come from the reference file,
// because those are facts about the GPU rather than about the kernel.
#include "qmd.h"
#include "cubin.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

// the generated field list, so a differing byte is reported as the field that lives there
typedef struct { const char *name; uint16_t lo, hi; } field_info_t;
#define X(n, lo, hi) {n, lo, hi},
static const field_info_t FIELDS[] = {TINYNV_QMD_FIELDS(X)};
#undef X

static const char *field_at(uint32_t byte) {
  static char buf[128];
  buf[0] = 0;
  for (size_t i = 0; i < sizeof(FIELDS) / sizeof(*FIELDS); i++)
    if (FIELDS[i].lo / 8 <= byte && byte <= FIELDS[i].hi / 8) {
      size_t n = strlen(buf);
      snprintf(buf + n, sizeof(buf) - n, "%s%s", n ? ", " : "", FIELDS[i].name);
      if (strlen(buf) > sizeof(buf) - 24) break;
    }
  return buf[0] ? buf : "no field";
}

// the reference file: "key rest-of-line", with the blocks written as hex
#define MAXLINE 8192
static char *value_of(const char *path, const char *key, char *out, size_t cap) {
  FILE *f = fopen(path, "r");
  if (!f) return NULL;
  static char line[MAXLINE];
  char *found = NULL;
  size_t klen = strlen(key);
  while (fgets(line, sizeof(line), f))
    if (!strncmp(line, key, klen) && (line[klen] == ' ' || line[klen] == '\t')) {
      snprintf(out, cap, "%s", line + klen + 1);
      out[strcspn(out, "\n")] = 0;
      found = out;
      break;
    }
  fclose(f);
  return found;
}

static uint64_t field_from(const char *line, const char *name) {
  const char *p = strstr(line, name);
  if (!p) return (uint64_t)-1;
  return strtoull(p + strlen(name), NULL, 0);
}

static int unhex(const char *hex, uint8_t *out, size_t cap) {
  size_t n = strlen(hex) / 2;
  if (n > cap) return -1;
  for (size_t i = 0; i < n; i++) {
    char b[3] = {hex[2 * i], hex[2 * i + 1], 0};
    out[i] = (uint8_t)strtoul(b, NULL, 16);
  }
  return (int)n;
}

static void compare(const char *what, const tinynv_qmd_t *got, const uint8_t *want, uint32_t n) {
  for (uint32_t i = 0; i < n; i++)
    if (got->b[i] != want[i]) {
      printf("  FAIL: %s differs at byte %u: %#04x, the oracle builds %#04x (bits %u-%u: %s)\n",
             what, i, got->b[i], want[i], i * 8, i * 8 + 7, field_at(i));
      fails++;
      return; // the first difference is the one worth reading
    }
  printf("  %-10s %3u bytes, identical to the oracle\n", what, n);
}

int main(int argc, char **argv) {
  const char *ref = argc > 1 ? argv[1] : "test/reference-qmd.txt";
  const char *cubin_path = argc > 2 ? argv[2] : "../spike/vecadd.sm120.cubin";

  char device[MAXLINE], derived[MAXLINE], launch[MAXLINE], release[MAXLINE], hex[MAXLINE];
  if (!value_of(ref, "device", device, sizeof(device)) || !value_of(ref, "derived", derived, sizeof(derived)) ||
      !value_of(ref, "launch", launch, sizeof(launch)) || !value_of(ref, "release", release, sizeof(release))) {
    printf("  cannot read %s: run tools/nv_reference_qmd.py\n", ref);
    return 1;
  }

  // the kernel, read with this driver's own reader rather than taken from the reference
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
  CHECK(cb.nkernels == 1, "%s holds %d kernels, this test wants one", cubin_path, cb.nkernels);
  if (cb.nkernels != 1) return 1;
  const tinynv_kernel_desc_t *k = &cb.kernels[0];

  // and the numbers derived from it, which the reference states so a divergence in the reader is caught here
  // rather than showing up as an unexplained byte
  uint32_t shmem = (0x400 + k->static_smem + 127) & ~127u; // the oracle's rounding: a floor of 0x400, then up to 128
  CHECK(k->regs == field_from(derived, "regs "), "register count %u, the oracle read %llu",
        k->regs, (unsigned long long)field_from(derived, "regs "));
  CHECK(shmem == field_from(derived, "shmem "), "shared memory %#x, the oracle derived %#llx",
        shmem, (unsigned long long)field_from(derived, "shmem "));
  CHECK(k->text_size == field_from(derived, "prog_size "), "program size %#llx, the oracle read %#llx",
        (unsigned long long)k->text_size, (unsigned long long)field_from(derived, "prog_size "));
  CHECK(k->const0_size == field_from(derived, "const0_size "), "constant bank 0 is %#llx, the oracle read %#llx",
        (unsigned long long)k->const0_size, (unsigned long long)field_from(derived, "const0_size "));

  uint32_t slm = (uint32_t)field_from(device, "slm_per_thread ");
  uint32_t sm_version = (uint32_t)field_from(device, "sm_version ");

  tinynv_qmd_program_t prog = {.regs = k->regs, .shmem = shmem, .slm_per_thread = slm,
                               .prog_size = (uint32_t)k->text_size, .sass_version = tinynv_sass_version(sm_version)};
  prog.constbuf_used[0] = 1;
  prog.constbuf_size[0] = (uint32_t)k->const0_size;

  static uint8_t want[TINYNV_QMD_BYTES];
  tinynv_qmd_t q;
  memset(&q, 0, sizeof(q));
  CHECK(!tinynv_qmd_program(&q, &prog), "building the program's descriptor failed: %s", tinynv_last_error());
  if (value_of(ref, "template", hex, sizeof(hex)) && unhex(hex, want, sizeof(want)) == TINYNV_QMD_BYTES)
    compare("template", &q, want, TINYNV_QMD_BYTES);
  else { printf("  FAIL: no template in %s\n", ref); fails++; }

  tinynv_qmd_launch_t l = {0};
  { // "launch grid A B C block D E F program_addr X cbuf0_addr Y"
    const char *g = strstr(launch, "grid ") + 5;
    char *end;
    for (int i = 0; i < 3; i++) { l.grid[i] = (uint32_t)strtoul(g, &end, 0); g = end; }
    const char *b = strstr(launch, "block ") + 6;
    for (int i = 0; i < 3; i++) { l.block[i] = (uint32_t)strtoul(b, &end, 0); b = end; }
  }
  l.program_addr = field_from(launch, "program_addr ");
  l.constbuf_addr[0] = field_from(launch, "cbuf0_addr ");
  l.constbuf_set[0] = 1;
  CHECK(!tinynv_qmd_launch(&q, &l), "applying the launch failed: %s", tinynv_last_error());
  if (value_of(ref, "launched", hex, sizeof(hex)) && unhex(hex, want, sizeof(want)) == TINYNV_QMD_BYTES)
    compare("launched", &q, want, TINYNV_QMD_BYTES);

  int slot = tinynv_qmd_release(&q, field_from(release, "addr "), field_from(release, "payload "), 0);
  CHECK(slot == (int)field_from(release, "slot "), "the release took slot %d, the oracle took %llu",
        slot, (unsigned long long)field_from(release, "slot "));
  if (value_of(ref, "released", hex, sizeof(hex)) && unhex(hex, want, sizeof(want)) == TINYNV_QMD_BYTES)
    compare("released", &q, want, TINYNV_QMD_BYTES);

  CHECK(!tinynv_qmd_chain(&q, field_from(release, "next_qmd "), 1), "chaining failed: %s", tinynv_last_error());
  if (value_of(ref, "chained", hex, sizeof(hex)) && unhex(hex, want, sizeof(want)) == TINYNV_QMD_BYTES)
    compare("chained", &q, want, TINYNV_QMD_BYTES);

  // Nothing above can catch two fields written over each other: a reference built the same way makes the identical
  // mistake and the bytes still agree. The descriptor counts its own claimed bits, so that is checked here instead.
  CHECK(q.overlaps == 0, "%u bits of the descriptor were claimed by two fields", q.overlaps);

  // constant buffer 0's driver parameters, which the kernel reads its memory windows out of
  uint32_t dwords = (uint32_t)field_from(derived, "cbuf0_dwords ");
  uint32_t *cb0 = calloc(dwords, 4);
  CHECK(tinynv_qmd_cbuf0(cb0, dwords, field_from(device, "shared_window "), field_from(device, "local_window "), NULL, NULL) == dwords,
        "constant buffer 0 came back the wrong length");
  static uint8_t want_cb[4096];
  if (value_of(ref, "cbuf0", hex, sizeof(hex))) {
    int n = unhex(hex, want_cb, sizeof(want_cb));
    CHECK(n == (int)dwords * 4, "the oracle's constant buffer 0 is %d bytes, this one is %u", n, dwords * 4);
    if (n == (int)dwords * 4) {
      if (memcmp(cb0, want_cb, (size_t)n)) {
        for (int i = 0; i < n; i++)
          if (((uint8_t *)cb0)[i] != want_cb[i]) {
            printf("  FAIL: constant buffer 0 differs at dword %d: %#010x, the oracle builds %#010x\n", i / 4,
                   cb0[i / 4], *(uint32_t *)(want_cb + (i / 4) * 4));
            fails++;
            break;
          }
      } else printf("  %-10s %3u dwords, identical to the oracle\n", "cbuf0", dwords);
    }
  }

  // The launch geometry, at the two offsets measured on the card. Pinned here as well as commented, because these are
  // the one pair of numbers in the descriptor path that no recording could have supplied - the oracle never writes them
  // - and a comment is not something a build can check.
  {
    uint32_t geo[TINYNV_QMD_CBUF0_MIN_DWORDS];
    const uint32_t grid[3] = {0x111, 0x222, 0x333}, block[3] = {0x444, 0x555, 0x666};
    tinynv_qmd_cbuf0(geo, TINYNV_QMD_CBUF0_MIN_DWORDS, 0, 0, grid, block);
    for (int i = 0; i < 3; i++) {
      CHECK(geo[TINYNV_CBUF0_NCTAID + i] == grid[i], "gridDim[%d] is not at constant buffer word %d", i,
            TINYNV_CBUF0_NCTAID + i);
      CHECK(geo[TINYNV_CBUF0_NTID + i] == block[i], "blockDim[%d] is not at constant buffer word %d", i,
            TINYNV_CBUF0_NTID + i);
    }
    // and with none given it must match the oracle exactly, which is what every comparison above depends on
    tinynv_qmd_cbuf0(geo, TINYNV_QMD_CBUF0_MIN_DWORDS, 0, 0, NULL, NULL);
    CHECK(!geo[TINYNV_CBUF0_NCTAID] && !geo[TINYNV_CBUF0_NTID],
          "asking for no geometry still wrote some, which would break every comparison against the oracle");
  }

  // and a field that does not fit must be refused rather than quietly wrapped into its neighbour
  CHECK(tinynv_qmd_set(&q, TINYNV_QMD_F(QMD_MAJOR_VERSION), 0xffff), "an oversized value was written into a 4 bit field");

  // The chain check, against descriptors built wrong on purpose. A check that cannot fail reads exactly like one that
  // works, and the failure it exists to catch - a kernel running beside the one it depends on - is silent and
  // intermittent, so there is no later stage that would notice it instead.
  {
    tinynv_qmd_t a, b, none;
    // Realistic descriptor addresses: 256 aligned, and under the 1 TB the 32 bit chain pointer reaches once shifted by
    // eight. A test address above that would fail to chain for a reason that has nothing to do with what is being tested.
    const uint64_t A_VA = 0x1020018400ull, B_VA = 0x1020018600ull;
    const char *why;
#define SOUND_R(qq, rel, pay, next, what) do { why = NULL; \
      if (tinynv_qmd_link_check((qq), (rel), (pay), (next), &why)) { printf("  FAIL: %s was refused: %s\n", what, why); fails++; } \
      else printf("  %-46s accepted\n", what); } while (0)
#define ROTTEN_R(qq, rel, pay, next, what) do { why = NULL; \
      if (!tinynv_qmd_link_check((qq), (rel), (pay), (next), &why)) { printf("  FAIL: %s was accepted\n", what); fails++; } \
      else printf("  %-46s refused: %s\n", what, why); } while (0)
#define SOUND(qq, pay, next, what) SOUND_R(qq, 1, pay, next, what)
#define ROTTEN(qq, pay, next, what) ROTTEN_R(qq, 1, pay, next, what)

    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    memset(&none, 0, sizeof(none));
    CHECK(tinynv_qmd_release(&a, 0x1000, 7, 0) >= 0, "the descriptor had no free release slot");
    CHECK(tinynv_qmd_release(&b, 0x1000, 8, 0) >= 0, "the descriptor had no free release slot");
    CHECK(!tinynv_qmd_chain(&a, B_VA, 1), "chaining failed");

    SOUND(&a, 7, B_VA, "a link that chains");
    SOUND(&b, 8, 0, "the last link");
    ROTTEN(&a, 6, B_VA, "a link releasing the wrong value");
    ROTTEN(&a, 7, A_VA, "a link scheduling the wrong one");
    ROTTEN(&a, 7, 0, "a last link that still schedules");
    ROTTEN(&b, 8, B_VA, "a link that schedules nothing");
    ROTTEN(&none, 1, 0, "a link that releases nothing");

    // The other shape, which TINYNV_TAIL_RELEASE builds: every link but the last stays silent and the last carries the
    // whole chain's value. Checked in both directions, because a check that accepts either shape cannot tell a chain
    // built wrong from a chain built the other way - and the reason this mode exists is to measure what the releases it
    // removes were costing, which is worthless if it silently keeps making them.
    tinynv_qmd_t quiet, tail;
    memset(&quiet, 0, sizeof(quiet));
    memset(&tail, 0, sizeof(tail));
    CHECK(!tinynv_qmd_chain(&quiet, B_VA, 1), "chaining the silent link failed");
    CHECK(tinynv_qmd_release(&tail, 0x1000, 9, 0) >= 0, "the tail had no free release slot");
    SOUND_R(&quiet, 0, 9, B_VA, "a silent link in a tail-only chain");
    SOUND_R(&tail, 1, 9, 0, "the tail of a tail-only chain");
    ROTTEN_R(&a, 0, 7, B_VA, "a link that releases when asked not to");
    ROTTEN_R(&quiet, 1, 9, B_VA, "a silent link where one per launch was asked");
    ROTTEN_R(&tail, 1, 8, 0, "a tail releasing the wrong value");

    // And the third shape, which TINYNV_NO_CHAIN_DEPS builds for one measurement: descriptors that neither release nor
    // schedule anything, because the command stream launches each of them and a host-method release ends the batch.
    // Pinned here for the same reason as the other two - a check that accepted every shape would tell us nothing about
    // the one actually built, and this is the shape where a stray link would silently restore the ordering the
    // measurement exists to remove.
    SOUND_R(&none, 0, 0, 0, "an unlinked, silent descriptor");
    ROTTEN_R(&quiet, 0, 9, 0, "an unlinked descriptor that still schedules");
    ROTTEN_R(&tail, 0, 9, 0, "an unlinked descriptor that still releases");
  }

  // Two releases on one descriptor, which is what the profile's tail timestamp needs: the timeline in slot 0 and the
  // gpu clock in slot 1. The profiled instrument reported nothing physical on its first hardware run, and "the second
  // release never encoded" was one of the candidate causes - the one that can be settled without a card. This does not
  // say the hardware honours two releases; it says we asked for two and did not, for instance, overwrite the first.
  {
    tinynv_qmd_t q;
    memset(&q, 0, sizeof(q));
    int s0 = tinynv_qmd_release(&q, 0x1000, 42, 0);
    int s1 = tinynv_qmd_release(&q, 0x2000, 43, 1);
    CHECK(s0 == 0, "the first release took slot %d rather than 0", s0);
    CHECK(s1 == 1, "the second release took slot %d rather than 1 - a descriptor cannot carry the timeline and a "
                   "timestamp", s1);
    CHECK(tinynv_qmd_get(&q, TINYNV_QMD_F(RELEASE0_ENABLE)) != 0, "the first release was turned off again");
    CHECK(tinynv_qmd_get(&q, TINYNV_QMD_F(RELEASE1_ENABLE)) != 0, "the second release is not enabled");
    // And the sizes, which is the half that decides whether a clock is written at all: two words is payload only.
    CHECK(tinynv_qmd_get(&q, TINYNV_QMD_RELEASE_STRUCTURE_SIZE_0_LO, TINYNV_QMD_RELEASE_STRUCTURE_SIZE_0_HI) ==
          TINYNV_QMDV_RELEASE_STRUCTURE_SIZE_SEMAPHORE_TWO_WORDS, "the timeline release asks for a timestamp");
    CHECK(tinynv_qmd_get(&q, TINYNV_QMD_RELEASE_STRUCTURE_SIZE_1_LO, TINYNV_QMD_RELEASE_STRUCTURE_SIZE_1_HI) ==
          TINYNV_QMDV_RELEASE_STRUCTURE_SIZE_SEMAPHORE_FOUR_WORDS,
          "the timestamp release is two words, so no clock is ever written");
    int s2 = tinynv_qmd_release(&q, 0x3000, 44, 0);
    CHECK(s2 < 0, "a third release was accepted onto a descriptor with two");
  }

  free(cb0);
  free(blob);
  tinynv_cubin_free(&cb);
  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
