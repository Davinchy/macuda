// Replay a captured workload, so the interleaving is the one that broke rather than the one I would have invented.
//
// test_hw_chain submits thousands of dependent launches without syncing, and it passes. It also runs one kernel, one
// dtype, one stream, and allocates everything up front. Session A's exotic-kernel op set does none of those things, and
// on the asynchronous driver it took a 5090 off the bus after passing 450/450 synchronously. A test that passes on the
// shape that works is not evidence about the shape that does not.
//
// So this reads a capture - the same format test_mm replays off-device, plus the queue each operation went to - and
// drives the real driver through it: allocations and frees churning against the page tables while launches and copies
// alternate between the two engines. What is being reproduced is the *order*, not the arithmetic of the original ops.
// Every launch still adds one to a chain buffer, so the count at the end says whether any of it ran twice, early, or
// not at all, and the allocation traffic around it is what makes that question interesting.
//
// Capture format: three whitespace-separated columns, OP QUEUE DETAIL, a "-" where a column does not apply. Unknown
// operations are counted and skipped, so a newer capture still replays here rather than failing to parse.
//   MALLOC - <bytes> / FREE - <bytes>      the churn, against the driver's own allocator
//   LAUNCH COMPUTE <kernel>                the kernel name is recorded, not run: we have one cubin here, not A's 183
//   MEMCPY* / MEMSET* COPY -               the copy engine, which is the half chain-with-crossings barely reaches
//   UNLOAD - -                             a module going away, which drains before unmapping its code
//   streamSynchronize / SYNC               where the original workload actually waited
//
// What the first capture says, before a line of it is replayed: of 48 places where the two queues change hands, 30 have
// a streamSynchronize between them - so the acquire is already satisfied and orders nothing - and the other 18 are all
// the same direction, COPY then COMPUTE. Not one unsynchronised COMPUTE-then-COPY in the whole run. So the suspect is
// narrower than "the cross-queue acquire": it is a launch made to wait on a copy, and eighteen sites to check.
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + t.tv_nsec * 1e-9; }

#define N 256                  // the chain buffer, in floats
#define MAX_LIVE 512           // live allocations carried at once; A's capture peaked near 106

// The churn pool. Frees are recorded by size, the way the capture records them, so a FREE takes back the most recent
// allocation of that size - which is what the original allocator did and what test_mm already assumes.
typedef struct { tinynv_devptr_t p; unsigned long long bytes; } live_t;
static live_t live[MAX_LIVE];
static int nlive;

static int take(unsigned long long bytes, tinynv_devptr_t *out) {
  for (int i = nlive - 1; i >= 0; i--)
    if (live[i].bytes == bytes) { *out = live[i].p; live[i] = live[--nlive]; return 1; }
  return 0;
}

int main(int argc, char **argv) {
  const char *trace = argc > 1 ? argv[1] : "opseq.txt";
  const char *path = argc > 2 ? argv[2] : "../spike/vecadd.sm120.cubin";
  // Counting what a capture would do costs no card, and a capture that does not parse is worth finding out about
  // before a hardware slot is spent on it - especially one whose expected outcome is the card going off the bus.
  int dry = getenv("TINYNV_REPLAY_DRYRUN") != NULL;
  if (!dry && (!getenv("TINYNV_HW") || !getenv("TINYNV_SOCKET"))) {
    printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s <capture> [cubin]\n", argv[0]);
    printf("       TINYNV_REPLAY_DRYRUN=1 %s <capture>   parses and counts, touching no hardware\n", argv[0]);
    return 2;
  }
  // A run whose expected outcome is the card going off the bus must not lose its diagnostics to a stdio buffer that
  // never gets flushed. Everything below is written as it happens.
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("libtinynv build %s\n", tinynv_build_id());

  FILE *tf = fopen(trace, "r");
  if (!tf) { printf("  cannot open capture %s\n", trace); return 1; }

  FILE *f = fopen(path, "rb");
  if (!f) { printf("  cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  rewind(f);
  void *blob = malloc((size_t)len);
  if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) { printf("  cannot read %s\n", path); return 1; }
  fclose(f);

  tinynv_device_t dev; tinynv_module_t mod; tinynv_kernel_t k; tinynv_kernel_info_t info; tinynv_stream_t s;
  tinynv_devptr_t a, b, c, d; tinynv_status_t st;
#define T(call) do { if (!dry && (st = (call)) != TINYNV_OK) { printf("  %s: %s\n    %s\n", #call, tinynv_status_str(st), tinynv_last_error()); return 1; } } while (0)
  a = b = c = d = 0; mod = 0; k = 0; s = 0; dev = 0; memset(&info, 0, sizeof(info));
  if (dry) goto replay;
  T(tinynv_init());
  T(tinynv_device_get(&dev, 0));
  T(tinynv_module_load(dev, blob, (size_t)len, &mod));
  T(tinynv_get_kernel(mod, "vecadd", &k));
  T(tinynv_kernel_info(k, &info));
  T(tinynv_stream_create(dev, &s));
  T(tinynv_malloc(dev, N * sizeof(float), &a));
  T(tinynv_malloc(dev, N * sizeof(float), &b));
  T(tinynv_malloc(dev, N * sizeof(float), &c));
  T(tinynv_malloc(dev, N * sizeof(float), &d));

  static float ha[N], hb[N], hc[N];
  for (int i = 0; i < N; i++) { ha[i] = (float)i; hb[i] = 1.0f; hc[i] = -1.0f; }
  T(tinynv_memcpy_htod(s, a, ha, sizeof(ha)));
  T(tinynv_memcpy_htod(s, b, hb, sizeof(hb)));
  T(tinynv_memcpy_htod(s, c, hc, sizeof(hc)));

  unsigned char params[64];
  unsigned n = N;
  size_t plen = (size_t)info.params[3].offset + sizeof(int);
#define SET(x, y, z) do { memset(params, 0, sizeof(params)); \
  uint64_t p0 = (uint64_t)(x), p1 = (uint64_t)(y), p2 = (uint64_t)(z); \
  memcpy(params + info.params[0].offset, &p0, 8); memcpy(params + info.params[1].offset, &p1, 8); \
  memcpy(params + info.params[2].offset, &p2, 8); memcpy(params + info.params[3].offset, &n, 4); } while (0)

  // The first launch seeds the chain from a; every one after it adds to the running total in place.
  SET(a, b, c);
  T(tinynv_launch(s, k, N / 64, 1, 1, 64, 1, 1, 0, params, plen));
replay:;
  long launches = 1, copies = 0, mallocs = 0, frees = 0, syncs = 0, unloads = 0, skipped = 0, oom = 0, unmatched = 0;

  char line[512], prev_q[32] = "";
  long lineno = 0;
  int crossing = 0, synced_since = 1;
  double t0 = now();
  while (fgets(line, sizeof(line), tf)) {
    char op[64] = "", queue[32] = "", detail[256] = "";
    unsigned long long bytes = 0;
    lineno++;
    if (line[0] == '#' || line[0] == '\n') continue;
    if (sscanf(line, "%63s %31s %255s", op, queue, detail) < 1) continue;
    bytes = strtoull(detail, NULL, 10);

    // The eighteen places worth naming. A hand-off from the copy engine to the compute engine with no synchronisation
    // between them is where a launch is made to wait on a copy, and a completion signal that lands early is invisible
    // everywhere else. If this run ends with the card off the bus, the last number printed is where it happened.
    if (!strcmp(queue, "COMPUTE") || !strcmp(queue, "COPY")) {
      if (prev_q[0] && strcmp(prev_q, queue) && !synced_since)
        printf("  crossing %d: %s -> %s at capture line %ld\n", ++crossing, prev_q, queue, lineno);
      snprintf(prev_q, sizeof(prev_q), "%s", queue);
      synced_since = 0;
    }
    if (!strcmp(op, "MALLOC")) {
      tinynv_devptr_t p;
      // A capture taken on a 32 GB card replayed against one that is busy can legitimately run out. That is not the
      // bug being hunted, so it is counted and the replay carries on rather than reporting a failure it did not find.
      p = 0;
      if (nlive >= MAX_LIVE || (!dry && tinynv_malloc(dev, (size_t)bytes, &p) != TINYNV_OK)) { oom++; continue; }
      live[nlive++] = (live_t){p, bytes};
      mallocs++;
    } else if (!strcmp(op, "FREE")) {
      tinynv_devptr_t p;
      // A free whose allocation was skipped above has nothing to give back; it is not a divergence in the driver.
      if (!take(bytes, &p)) { unmatched++; continue; }
      T(tinynv_free(dev, p));
      frees++;
    } else if (!strcmp(op, "LAUNCH")) {
      // The captured kernel is not the one that runs - we have one cubin here, not A's 183. What is being replayed is
      // that a launch happened at this point in the sequence, on the compute queue, with allocations live around it.
      SET(c, b, c);
      T(tinynv_launch(s, k, N / 64, 1, 1, 64, 1, 1, 0, params, plen));
      launches++;
    } else if (!strncmp(op, "MEMCPY", 6) || !strncmp(op, "MEMSET", 6) || !strcmp(op, "COPY")) {
      // The copy engine, at the point in the sequence the capture says it was used. The size is the captured one only
      // as far as the chain buffer goes: what matters is that the queue changed here, because a queue change is what
      // puts a semaphore acquire into the command stream, and that acquire is the mechanism under suspicion.
      T(tinynv_memcpy_dtod(s, d, c, N * sizeof(float)));
      T(tinynv_memcpy_dtod(s, c, d, N * sizeof(float)));
      copies++;
    } else if (!strcmp(op, "UNLOAD")) {
      // Worth replaying rather than skipping: unloading drains the timeline and then unmaps the module's code, and it
      // is one of the few places where getting the order wrong means an engine fetching instructions from a page that
      // has gone. A second handle on the same cubin gives that path real traffic without needing A's 183.
      tinynv_module_t tmp;
      if (dry) { unloads++; continue; }
      if (tinynv_module_load(dev, blob, (size_t)len, &tmp) == TINYNV_OK) { T(tinynv_module_unload(tmp)); unloads++; }
      else oom++;
    } else if (!strcmp(op, "streamSynchronize") || !strcmp(op, "SYNC")) {
      T(tinynv_stream_sync(s));
      synced_since = 1;
      syncs++;
    } else {
      skipped++;
    }
  }
  fclose(tf);
  double issued = now() - t0;

  T(tinynv_stream_sync(s));
  double done = now() - t0;
  printf("replayed %s: %ld launches, %ld copy-engine crossings, %ld mallocs, %ld frees, %ld syncs, %ld module unloads\n",
         trace, launches, copies, mallocs, frees, syncs, unloads);
  if (skipped || oom || unmatched)
    printf("  (%ld lines not understood, %ld allocations refused, %ld frees with no matching allocation)\n",
           skipped, oom, unmatched);
  printf("  %d unsynchronised queue hand-offs, every one of them a launch made to wait on a copy\n", crossing);
  printf("  issuing took %.3f s; everything finished %.3f s in\n", issued, done);
  printf("  %d allocations still live at the end\n", nlive);

  if (dry) { printf("dry run: the capture parses; nothing was run\n"); return 0; }

  T(tinynv_memcpy_dtoh(s, hc, c, sizeof(hc)));
  int wrong = 0, first = -1;
  for (int i = 0; i < N; i++) {
    float want = (float)i + (float)launches;   // b is all ones, so every launch adds exactly one
    if (hc[i] != want) { if (first < 0) first = i; wrong++; }
  }
  if (wrong) {
    printf("  FAIL: %d of %d wrong; the first is [%d] = %g where %g belongs\n", wrong, N, first, (double)hc[first],
           (double)((float)first + (float)launches));
    printf("        too few is work that ran out of order or scratch reused while an engine was still reading it;\n"
           "        too many is a batch submitted twice.\n");
    return 1;
  }

  for (int i = 0; i < nlive; i++) T(tinynv_free(dev, live[i].p));
  T(tinynv_free(dev, a)); T(tinynv_free(dev, b)); T(tinynv_free(dev, c)); T(tinynv_free(dev, d));
  T(tinynv_stream_destroy(s));
  T(tinynv_module_unload(mod));
  printf("all %ld launches accounted for, through %ld allocations and %ld frees of churn\n", launches, mallocs, frees);
  return 0;
}
