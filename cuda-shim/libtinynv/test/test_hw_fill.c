// Fill video memory to refusal and prove the top of what the manager hands out is usable memory, not the firmware's.
//
// fw_layout.h derives the reservation at the top of video memory from the sizes handed to the firmware, and the boot
// checks it against the chip's wpr2 registers before anything is allocated. This is the other half of the proof, from
// below: allocate until the manager refuses, write a pattern through the engine into the first and last page of every
// buffer, read them back, and ask the firmware whether it still answers. The buffers between them cover the whole
// managed range, so the highest one covers the manager's top. A buffer inside WPR2 reads back zeros - the FB MMU drops
// a non-secure write and blocks the read - and one inside the firmware's unprotected heap corrupts GSP-RM, which is
// what the sensor and free-heap reads afterwards would show. Both are the failure this probe exists to see, and
// neither is something the card reports on its own.
//
//   TINYNV_HW=1 TINYNV_SOCKET=<socket> test_hw_fill     (sh tools/nv_shim_step.sh A probe build/shim/nv/test_hw_fill)
//
// Expected: the total allocated lands within a few hundred MB of the manager's size (the exec arena, the staging and
// window pools and the page tables the mappings need take the rest), the refusal names the video memory region, every
// pattern reads back, and the firmware answers afterwards exactly as before.
#include "tinynv.h"
#include "fw_layout.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BUFS 512

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

int main(int argc, char **argv) {
  if (!getenv("TINYNV_HW") || !getenv("TINYNV_SOCKET")) {
    printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s\n", argv[0]);
    return 2;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  tinynv_device_t d = NULL;
  if (tinynv_device_get(&d, 0) != TINYNV_OK) { printf("  no device: %s\n", tinynv_last_error()); return 1; }
  tinynv_device_props_t props;
  if (tinynv_device_props(d, &props) != TINYNV_OK) { printf("  cannot open the card: %s\n", tinynv_last_error()); return 1; }

  uint64_t base = 0, limit = 0, vram = 0, managed_end = 0;
  if (tinynv_device_wpr2_range(d, &base, &limit, &vram, &managed_end)) {
    printf("  could not read the wpr2 registers: %s\n", tinynv_last_error());
    return 1;
  }
  const double MB = 1048576.0;
  printf("  video memory %.1f MB, the manager hands out %.1f MB and stops at %#llx; wpr2 %#llx..%#llx\n", vram / MB,
         (double)props.total_mem / MB, (unsigned long long)managed_end, (unsigned long long)base,
         (unsigned long long)limit);

  uint64_t heap_before = 0;
  int heap_known = !tinynv_device_gsp_free_heap(d, &heap_before);
  tinynv_sensors_t s0, s1;
  int sens_known = tinynv_sensors(d, &s0) == TINYNV_OK;

  tinynv_stream_t st;
  if (tinynv_stream_create(d, &st) != TINYNV_OK) { printf("  no stream: %s\n", tinynv_last_error()); return 1; }

  // Allocate to refusal, big pieces first so the count stays small, then finer ones to reach the last few megabytes.
  static tinynv_devptr_t bufs[MAX_BUFS];
  static uint64_t sizes[MAX_BUFS];
  int n = 0;
  uint64_t total = 0;
  const uint64_t steps[] = {1ull << 30, 64ull << 20, 4ull << 20, 1ull << 20};
  const char *refusal = "";
  for (size_t k = 0; k < sizeof steps / sizeof *steps; k++) {
    while (n < MAX_BUFS) {
      tinynv_devptr_t p = 0;
      if (tinynv_malloc(d, (size_t)steps[k], &p) != TINYNV_OK) { refusal = tinynv_last_error(); break; }
      bufs[n] = p; sizes[n] = steps[k]; n++; total += steps[k];
    }
    printf("  after %llu MB pieces: %d buffers, %.1f MB in total; refused with: %s\n",
           (unsigned long long)(steps[k] >> 20), n, (double)total / MB, refusal);
  }
  CHECK(n > 0 && n < MAX_BUFS, "%d buffers - either nothing could be allocated or the table filled before the card", n);
  CHECK(strstr(refusal, "video memory region") != NULL, "the refusal does not name the video memory region: %s", refusal);
  // What was allocated against what the manager says it has: the difference is what the driver itself holds (the
  // exec arena and its regions, the staging and window pools, the page tables these mappings needed).
  double held = ((double)props.total_mem - (double)total) / MB;
  printf("  the manager reports %.1f MB, %.1f MB was allocated, so the driver itself holds %.1f MB\n",
         (double)props.total_mem / MB, (double)total / MB, held);
  CHECK(held >= 0 && held < 1024, "%.1f MB unaccounted for between the manager's size and what could be allocated", held);

  // Every buffer's first and last page: written through the engine, read back, compared. A page inside the
  // write-protected region reads back zeros.
  static unsigned char pattern[4096], back[4096];
  int bad = 0;
  for (int i = 0; i < n && bad < 4; i++) {
    for (int e = 0; e < 2; e++) {
      tinynv_devptr_t at = e ? bufs[i] + sizes[i] - 4096 : bufs[i];
      for (int j = 0; j < 4096; j++) pattern[j] = (unsigned char)(j * 7 + i * 13 + e * 101 + 1);
      memset(back, 0, sizeof back);
      if (tinynv_memcpy_htod(st, at, pattern, 4096) != TINYNV_OK || tinynv_stream_sync(st) != TINYNV_OK ||
          tinynv_memcpy_dtoh(st, back, at, 4096) != TINYNV_OK || tinynv_stream_sync(st) != TINYNV_OK) {
        CHECK(0, "copy to or from buffer %d %s page refused: %s", i, e ? "last" : "first", tinynv_last_error());
        bad++;
        continue;
      }
      if (memcmp(pattern, back, 4096)) {
        int zeros = 1;
        for (int j = 0; j < 4096; j++) if (back[j]) { zeros = 0; break; }
        CHECK(0, "buffer %d %s page (%#llx) read back %s - %s", i, e ? "last" : "first", (unsigned long long)at,
              zeros ? "all zeros" : "different bytes",
              zeros ? "the write was dropped: this address is inside the write-protected region" : "corrupted");
        bad++;
      }
    }
  }
  if (!bad) printf("  every buffer's first and last page reads back what was written (%d buffers, %d pages)\n", n, 2 * n);

  // And the firmware afterwards: the same heap figure it gave before, and a thermometer that still answers.
  if (heap_known) {
    uint64_t heap_after = 0;
    if (tinynv_device_gsp_free_heap(d, &heap_after))
      CHECK(0, "gsp-rm answered the free-heap query before the fill and refuses it after: %s", tinynv_last_error());
    else
      CHECK(heap_after == heap_before, "gsp-rm's free heap moved from %llu to %llu bytes across a fill that allocated "
            "nothing from it", (unsigned long long)heap_before, (unsigned long long)heap_after);
  }
  if (sens_known) CHECK(tinynv_sensors(d, &s1) == TINYNV_OK, "the sensors answered before the fill and not after: %s",
                        tinynv_last_error());

  for (int i = n - 1; i >= 0; i--) tinynv_free(d, bufs[i]);
  tinynv_stream_destroy(st);
  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
