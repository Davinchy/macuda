// Checks the descriptor writer and the command-buffer encoder against bytes a 5090 was actually handed.
//
// Every other comparison in this suite is libtinynv against the python oracle: two derivations agreeing, which catches a
// great deal but cannot tell you that either one is what the hardware takes. These are the descriptors and command
// buffers from a recorded session, captured inside the driver at the moment they existed as bytes, with each command
// buffer's length independently confirmed by the ring entry the card was given in the wire trace of the same run.
//
// Addresses are excluded, by construction rather than by concession. In the driver they stay symbolic until link time,
// so the recorded bytes hold zeros where they go, and libtinynv is handed them by its caller rather than deriving them.
// The reference names those offsets and both sides are blanked there. Everything the driver computes is compared.
//
// The claimed-bit mask is the part that byte equality cannot do. A field written to the value the block already held -
// a zero over a zero - leaves no trace in the bytes, so a transcription that sets the wrong field, or misses one whose
// value happens to be zero, would pass a byte comparison. The mask says which bits each side's fields claimed.
#include "qmd.h"
#include "submit.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

typedef struct { const char *name; uint16_t lo, hi; } field_info_t;
#define X(n, lo, hi) {n, lo, hi},
static const field_info_t FIELDS[] = {TINYNV_QMD_FIELDS(X)};
#undef X

static const char *field_at_bit(uint32_t bit) {
  for (size_t i = 0; i < sizeof(FIELDS) / sizeof(*FIELDS); i++)
    if (FIELDS[i].lo <= bit && bit <= FIELDS[i].hi) return FIELDS[i].name;
  return "no field";
}

#define MAXHEX 65536
static char line[MAXHEX];

static long num_after(const char *s, const char *key) {
  const char *p = strstr(s, key);
  return p ? strtol(p + strlen(key), NULL, 0) : -1;
}
static int triple(const char *s, const char *key, uint32_t out[3]) {
  const char *p = strstr(s, key);
  if (!p) return -1;
  char *end;
  p += strlen(key);
  for (int i = 0; i < 3; i++) { out[i] = (uint32_t)strtoul(p, &end, 0); p = end + 1; }
  return 0;
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

// the two batches this driver builds, against the ones the card took. the semaphore address and its payload are patched
// in at link time on the recorded side, so both sides are blanked at the offsets the reference names.
static void blank(uint8_t *b, const char *patches, uint32_t len) {
  for (const char *p = patches; p && *p; ) {
    long off = strtol(p, (char **)&p, 0);
    if (off >= 0 && (uint32_t)off + 4 <= len) memset(b + off, 0, 4);
    if (*p == ',') p++;
    else break;
  }
}

static void compare_bytes(const char *what, const uint8_t *got, const uint8_t *want, uint32_t n, int as_bits) {
  checks++;
  for (uint32_t i = 0; i < n; i++)
    if (got[i] != want[i]) {
      uint32_t bit = i * 8;
      for (int b = 0; b < 8; b++) if (((got[i] ^ want[i]) >> b) & 1) { bit = i * 8 + (uint32_t)b; break; }
      if (as_bits) printf("  FAIL: %s differs at byte %u: 0x%02x, the card was handed 0x%02x (field %s)\n",
                          what, i, got[i], want[i], field_at_bit(bit));
      else printf("  FAIL: %s differs at dword %u: %#010x, the card was handed %#010x\n", what, i / 4,
                  ((const uint32_t *)got)[i / 4], ((const uint32_t *)want)[i / 4]);
      fails++;
      return;
    }
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "test/reference-hw.txt";
  FILE *f = fopen(path, "r");
  if (!f) { printf("  cannot open %s: run tools/nv_reference_hw.py against a recorded session\n", path); return 1; }

  uint32_t slm = 0, sass = 0;
  uint64_t shared_window = 0, local_window = 0, bytes_per_tpc = 0;
  tinynv_die_t die = {0};
  int descriptors = 0, buffers = 0, rebuilt = 0;

  // carried between the "launch=" line and the blocks that follow it
  tinynv_qmd_t q;
  char pending[512] = {0};
  uint32_t cbuf0_dwords = 0;
  static uint8_t want[MAXHEX / 2], mine[MAXHEX / 2];

  while (fgets(line, sizeof(line), f)) {
    line[strcspn(line, "\n")] = 0;
    if (line[0] == '#' || !line[0]) continue;

    if (!strncmp(line, "device ", 7)) {
      slm = (uint32_t)num_after(line, "slm_per_thread=");
      sass = (uint32_t)num_after(line, "sass_version=");
      shared_window = strtoull(strstr(line, "shared_window=") + 14, NULL, 0);
      local_window = strtoull(strstr(line, "local_window=") + 13, NULL, 0);
      die.num_gpcs = (uint32_t)num_after(line, "num_gpcs=");
      die.num_tpc_per_gpc = (uint32_t)num_after(line, "num_tpc_per_gpc=");
      die.num_sm_per_tpc = (uint32_t)num_after(line, "num_sm_per_tpc=");
      die.max_warps_per_sm = (uint32_t)num_after(line, "max_warps_per_sm=");
      CHECK(sass == tinynv_sass_version(0xa04), "sass version %u, the card reported %u", tinynv_sass_version(0xa04), sass);
      tinynv_local_memory_size(&die, slm, &bytes_per_tpc);
      continue;
    }

    if (!strncmp(line, "launch=", 7)) {
      snprintf(pending, sizeof(pending), "%s", line);
      cbuf0_dwords = (uint32_t)num_after(line, "cbuf0_dwords=");

      tinynv_qmd_program_t prog = {.regs = (uint32_t)num_after(line, "regs="),
                                   .shmem = (uint32_t)num_after(line, "shmem="),
                                   .slm_per_thread = slm, .prog_size = (uint32_t)num_after(line, "prog_size="),
                                   .sass_version = sass};
      prog.constbuf_used[0] = 1;
      prog.constbuf_size[0] = (uint32_t)num_after(line, "const0_size=");
      memset(&q, 0, sizeof(q));
      CHECK(!tinynv_qmd_program(&q, &prog), "building the program's descriptor failed: %s", tinynv_last_error());

      tinynv_qmd_launch_t l = {0};
      CHECK(!triple(line, "grid=", l.grid), "no grid in %s", line);
      CHECK(!triple(line, "block=", l.block), "no block in %s", line);
      l.constbuf_set[0] = 1; // the address is patched in later, so it is zero here on both sides
      CHECK(!tinynv_qmd_launch(&q, &l), "applying the launch failed: %s", tinynv_last_error());

      // the release slots, in the order the card's driver took them
      const char *r = strstr(line, "releases=");
      if (r && strncmp(r + 9, "-", 1)) {
        for (const char *p = r + 9; *p && *p != ' '; ) {
          int ts = !strncmp(p, "ts", 2);
          CHECK(tinynv_qmd_release(&q, 0, 0, ts) >= 0, "the card took a release slot this descriptor has none left for");
          while (*p && *p != ',' && *p != ' ') p++;
          if (*p == ',') p++; else break;
        }
      }
      if (num_after(line, "chain=") == 1) CHECK(!tinynv_qmd_chain(&q, 0, 1), "chaining failed: %s", tinynv_last_error());
      CHECK(q.overlaps == 0, "%u bits of the descriptor were claimed by two fields", q.overlaps);
      descriptors++;
      continue;
    }

    if (!strncmp(line, "qmd=", 4)) {
      CHECK(unhex(line + 4, want, sizeof(want)) == TINYNV_QMD_BYTES, "the recorded descriptor is not %d bytes", TINYNV_QMD_BYTES);
      compare_bytes("descriptor", q.b, want, TINYNV_QMD_BYTES, 1);
      continue;
    }
    if (!strncmp(line, "claimed=", 8)) {
      CHECK(unhex(line + 8, want, sizeof(want)) == TINYNV_QMD_BYTES, "the recorded claim mask is not %d bytes", TINYNV_QMD_BYTES);
      compare_bytes("claimed bits", q.claimed, want, TINYNV_QMD_BYTES, 1);
      continue;
    }
    if (!strncmp(line, "cbuf0=", 6)) {
      int n = unhex(line + 6, want, sizeof(want));
      uint32_t *cb0 = calloc(cbuf0_dwords, 4);
      CHECK(tinynv_qmd_cbuf0(cb0, cbuf0_dwords, shared_window, local_window, NULL, NULL) == cbuf0_dwords,
            "constant buffer 0 came back the wrong length");
      CHECK(n == (int)cbuf0_dwords * 4, "the card's constant buffer 0 is %d bytes, this one is %u", n, cbuf0_dwords * 4);
      if (n == (int)cbuf0_dwords * 4) compare_bytes("constant buffer 0", (const uint8_t *)cb0, want, (uint32_t)n, 0);
      free(cb0);
      continue;
    }

    if (!strncmp(line, "cmdbuf=", 7)) { snprintf(pending, sizeof(pending), "%s", line); continue; }
    if (!strncmp(line, "words=", 6)) {
      int n = unhex(line + 6, want, sizeof(want));
      if (n < 0) continue;
      buffers++;
      const char *patches = strstr(pending, "patches=");
      patches = patches ? patches + 8 : NULL;

      uint32_t storage[256];
      tinynv_cmdbuf_t c = {.words = storage, .n = 0, .cap = 256};
      const char *ops = strstr(pending, "ops=");
      int unbuilt = 0, bad = 0;
      char what[256] = "batch";
      if (ops) {
        snprintf(what, sizeof(what), "%s", ops + 4);
        for (const char *o = ops + 4; *o && *o != ' ' && !bad; ) {
          // this operation's argument, if it has one: the digits after a colon, before the comma that ends the op
          size_t len = strcspn(o, ", ");
          const char *colon = memchr(o, ':', len);
          long arg = colon ? strtol(colon + 1, NULL, 0) : 0;
          if (!strncmp(o, "wait:", 5)) bad |= tinynv_cmd_wait(&c, 0, (uint64_t)arg);
          else if (!strncmp(o, "release_ts:", 11)) bad |= tinynv_cmd_release_ts(&c, 0, (uint64_t)arg, 1);
          else if (!strncmp(o, "release:", 8)) bad |= tinynv_cmd_release(&c, 0, (uint64_t)arg);
          else if (!strncmp(o, "barrier", 7) && o[7] != '?') bad |= tinynv_cmd_memory_barrier(&c);
          else if (!strncmp(o, "set_object4:", 12)) bad |= tinynv_cmd_set_object(&c, 4, (uint32_t)arg);
          else if (!strncmp(o, "set_object:", 11)) bad |= tinynv_cmd_set_object(&c, 1, (uint32_t)arg);
          else if (!strncmp(o, "local_window", 12)) bad |= tinynv_cmd_shader_window(&c, 0, local_window);
          else if (!strncmp(o, "shared_window", 13)) bad |= tinynv_cmd_shader_window(&c, 1, shared_window);
          else if (!strncmp(o, "launch", 6)) bad |= tinynv_cmd_launch(&c, 0);
          else if (!strncmp(o, "copy:", 5)) bad |= tinynv_cmd_copy(&c, 0, 0, (uint64_t)arg);
          else if (!strncmp(o, "dma_timestamp", 13)) bad |= tinynv_cmd_dma_timestamp(&c, 0);
          else if (!strncmp(o, "dma_signal:", 11)) bad |= tinynv_cmd_dma_signal(&c, 0, (uint32_t)arg);
          else if (!strncmp(o, "local_memory:", 13)) {
            // "local_memory:<address>:<bytes per cluster>:<throttle>". The size is not taken from the reference: it is
            // derived here from the die's shape, and it having to come out the same is the point of comparing it.
            const char *second = colon ? memchr(colon + 1, ':', len - (size_t)(colon + 1 - o)) : NULL;
            uint64_t recorded = second ? strtoull(second + 1, NULL, 0) : 0;
            CHECK(bytes_per_tpc == recorded, "local memory works out at %#llx per cluster, the card was told %#llx",
                  (unsigned long long)bytes_per_tpc, (unsigned long long)recorded);
            bad |= tinynv_cmd_local_memory(&c, (uint64_t)arg, bytes_per_tpc);
          }
          else { unbuilt = 1; break; }  // a method nothing here builds yet: say so rather than pass over the batch
          while (*o && *o != ',' && *o != ' ') o++;
          if (*o == ',') o++; else break;
        }
      } else unbuilt = 1;

      if (unbuilt) { printf("  %-18s %2d dwords: holds a method this driver does not build yet, not compared\n",
                            "not rebuilt", n / 4); continue; }
      CHECK(!bad, "building the batch failed: %s", tinynv_last_error());
      CHECK((int)c.n * 4 == n, "this driver builds %u dwords where the card took %d (%s)", c.n, n / 4, what);
      if (!bad && (int)c.n * 4 == n) {
        int before = fails;
        memcpy(mine, c.words, c.n * 4);
        blank(mine, patches, (uint32_t)n);
        blank(want, patches, (uint32_t)n);
        compare_bytes(what, mine, want, (uint32_t)n, 0);
        if (fails == before) {
          rebuilt++;
          printf("  %2d dwords, identical to what the card took: %s\n", n / 4, strtok(what, " "));
        }
      }
      continue;
    }
  }
  fclose(f);

  printf("  %d descriptors rebuilt: bytes, claimed bits and constant buffer 0 all as recorded\n", descriptors);
  printf("  %d command buffers in the recording, %d rebuilt byte for byte\n", buffers, rebuilt);
  CHECK(descriptors >= 1, "%s held no descriptors", path);
  CHECK(rebuilt >= 1, "%s held no command buffer this driver could rebuild", path);
  printf(fails ? "%d of %d checks failed\n" : "all %d checks passed\n", fails ? fails : checks, checks);
  return fails ? 1 : 0;
}
