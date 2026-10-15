// How much of a launch is the kernel, and how much is the space between kernels?
//
// Decode runs 1992 kernels a token at 28.7 us each where the same kernels take 5.5 us on the vendor's driver, and
// everything that could explain it has been eliminated by measurement: descriptors, host cost, constant buffers, chain
// seams, fault polling, clocks, and - once the trace showed a single stream - concurrency, because CUDA serialises a
// stream too. What is left is that our kernels are slower, or that the gap between them is longer. Those want different
// fixes and nothing measured so far tells them apart.
//
// This tells them apart without needing a timestamp the hardware may not give us, by making the kernel's own work the
// independent variable. A chain of launches costs N * (gap + execution). Execution scales with the bytes the kernel
// touches; the gap does not. So measure per-launch time at several sizes and read the two apart: the slope is what the
// memory system is actually delivering, and the intercept - where the work goes to nothing - is the gap.
//
// Both halves are worth having. A slope near the card's 1.79 TB/s says a big kernel saturates memory and execution is
// not the problem, which puts the whole 5x on the gap. A slope far below it says a single serialised kernel cannot fill
// the bus here, which is a different problem entirely and not one the driver can fix by launching faster.
//
// The first run answered that: the slope is full and the gap is 5.87 us. But 5.87 is not decode's ~25, and the
// difference this bench cannot see is that it launches ONE kernel over and over where decode launches nearly two
// thousand distinct ones. So TINYNV_GAP_MODULES loads the same cubin as several modules and launches round robin
// across them: identical work, identical geometry, but a different descriptor and a different program address every
// time, which is the one thing separating this bench from the workload it is trying to explain. If the gap climbs
// toward 25 us as the module count rises, that is the cost of a fresh descriptor and it says what moving descriptors
// into video memory is worth before anyone spends a refactor finding out. If it stays at 5.87, the difference is
// somewhere else and the refactor would have been aimed at the wrong thing.
//
// Worth saying: the eviction version of that story is already refuted by this bench's own largest point. If the gap
// were descriptors being pushed out of cache by the weights streaming past them, the 50 MB launch would show a bigger
// gap than the 3 KB one. It shows a smaller one - about 3 us against 5.87.
//
// That sweep answered its question - 5.72 us reused, 34.51 at 128 distinct modules, past real decode's ~25 - and moving
// the command arena into video memory did not flatten it, so the distinct-module penalty is something other than where
// the descriptor lives. What it cannot say is WHICH of the remaining candidates, because it changes two things at once:
// with N modules in round robin, every launch both switches module AND has N of them resident. Those are different
// mechanisms with different fixes. A cost paid on the switch is a cold fetch - the program image or its constant banks
// not being where the last launch left things - and the fix is to stop switching, which is what a recorded, replayed
// chain would do. A cost paid for having N resident is capacity, and no amount of ordering helps.
//
// TINYNV_GAP_RUN separates them for the price of one card run. It launches R times from one module before moving to the
// next, so the number resident is unchanged and the number of switches falls by R. Default 1, which is exactly the
// round robin this bench has always done, so every number already recorded still compares.
//
// The prediction, on the record before the run: if the penalty is paid per switch, the gap at 128 modules should fall
// as roughly 5.7 + (34.5 - 5.7)/R - about 20 us at R=2, 12 at R=4, under 8 by R=8. If it stays near 34.5 at R=8, it is
// residency, the switch is innocent, and record-and-replay would not have helped the thing it looks like it should.
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + t.tv_nsec * 1e-9; }

#define MAXN (4u << 20)          // 4M floats: 48 MB touched per launch, about what a real decode kernel costs
#define BLOCK 256

// Enough launches at each size to swamp the timing, but not so many that the big ones take all day.
static uint32_t launches_for(uint32_t n) { return n <= (64u << 10) ? 4000 : (n <= (256u << 10) ? 2000 : 500); }

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "../spike/vecadd.sm120.cubin";
  if (!getenv("TINYNV_HW") || !getenv("TINYNV_SOCKET")) {
    printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s [cubin]\n", argv[0]);
    return 2;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("libtinynv build %s\n", tinynv_build_id());

  FILE *f = fopen(path, "rb");
  if (!f) { printf("  cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  rewind(f);
  void *blob = malloc((size_t)len);
  if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) { printf("  cannot read %s\n", path); return 1; }
  fclose(f);

#define MAX_MODULES 128
  tinynv_device_t dev; tinynv_module_t mod[MAX_MODULES]; tinynv_kernel_t k[MAX_MODULES];
  tinynv_kernel_info_t info; tinynv_stream_t s;
  tinynv_devptr_t a, b, c; tinynv_status_t st;
  int nmod = getenv("TINYNV_GAP_MODULES") ? atoi(getenv("TINYNV_GAP_MODULES")) : 1;
  if (nmod < 1) nmod = 1;
  if (nmod > MAX_MODULES) nmod = MAX_MODULES;
  // How many launches from one module before moving to the next. 1 is round robin, which is what this bench did before
  // this knob existed and is still the default, so the numbers on record are still the numbers this produces.
  int run = getenv("TINYNV_GAP_RUN") ? atoi(getenv("TINYNV_GAP_RUN")) : 1;
  if (run < 1) run = 1;
  // How many independent passes of the whole sweep to do in one process. One is what this bench always did.
  //
  // It is here because a single pass turned out not to be a measurement. The shipped one-module floor read 7.60 us and
  // then 6.87 us on a repeat of the same binary on the same card, and a knob being compared against it was worth 0.76 -
  // which is to say the difference under test was smaller than the drift between runs, and neither number could be
  // believed. Passes make that drift visible inside one process, where it cannot be confused with the thing being
  // measured: if pass 1 and pass 3 differ by more than the effect, the effect has not been measured yet.
  int passes = getenv("TINYNV_GAP_PASSES") ? atoi(getenv("TINYNV_GAP_PASSES")) : 1;
  if (passes < 1) passes = 1;
#define T(call) do { if ((st = (call)) != TINYNV_OK) { printf("  %s: %s\n    %s\n", #call, tinynv_status_str(st), tinynv_last_error()); return 1; } } while (0)
  T(tinynv_init());
  T(tinynv_device_get(&dev, 0));
  // Each module is its own copy of the same cubin, so each carries its own program address and its own descriptor.
  for (int i = 0; i < nmod; i++) {
    T(tinynv_module_load(dev, blob, (size_t)len, &mod[i]));
    T(tinynv_get_kernel(mod[i], "vecadd", &k[i]));
  }
  T(tinynv_kernel_info(k[0], &info));
  // Named in the output rather than remembered: the whole value of this knob is comparing two runs, and a pair of
  // numbers whose difference is which knob was set is worthless if either number cannot say which it was.
  if (run == 1)
    printf("launching round robin across %d module%s\n", nmod, nmod == 1 ? " (one kernel, reused)" : "s");
  else
    printf("launching across %d module%s, %d launches from each before moving on (TINYNV_GAP_RUN): same number "
           "resident, %d times fewer switches\n", nmod, nmod == 1 ? "" : "s", run, run);
  T(tinynv_stream_create(dev, &s));
  T(tinynv_malloc(dev, (size_t)MAXN * sizeof(float), &a));
  T(tinynv_malloc(dev, (size_t)MAXN * sizeof(float), &b));
  T(tinynv_malloc(dev, (size_t)MAXN * sizeof(float), &c));
  T(tinynv_memset(s, a, 0, (size_t)MAXN * sizeof(float)));
  T(tinynv_memset(s, b, 0, (size_t)MAXN * sizeof(float)));
  T(tinynv_stream_sync(s));

  unsigned char params[64];
  size_t plen = (size_t)info.params[3].offset + sizeof(int);

  const uint32_t sizes[] = {256, 4096, 65536, 262144, 1048576, MAXN};
  const int nsizes = (int)(sizeof(sizes) / sizeof(*sizes));
  double per[16];

  // --- blocks mode: what does a launch cost per BLOCK, with the work held down? ----------------------------------
  //
  // Diffusion pays ~30-40 us for elementwise kernels that move a few megabytes and should cost ~3 us at bandwidth,
  // while LLM decode pays 6.3 us of non-memory cost per launch and compute-heavy kernels reach native throughput on
  // the same card. The structural difference Session A found is the grid: decode launches a few to a few hundred
  // blocks, a 1280x1280 elementwise convert launches about 6,400.
  //
  // The size sweep below cannot answer this, because there n sets the grid AND the work at once, so a bigger grid is
  // also more bytes. This mode holds the work at one block's worth and raises the grid anyway: vecadd tests i < n, so
  // every block past the first returns without touching memory. What is left scales with blocks and nothing else.
  //
  // Read it as: if us/launch climbs with the block count, the cost is per block - dispatch, or occupancy throttled by
  // something the descriptor declares - and it is in the driver's lane. If it stays flat, blocks are not the story and
  // the elementwise kernels are simply slow on our path, which is a codegen question and not a submission one.
  if (getenv("TINYNV_GAP_BLOCKS")) {
    const uint32_t counts[] = {1, 17, 170, 1700, 6800, 16384};
    uint32_t n1 = BLOCK;                       // one block's worth of real work, whatever the grid
    printf("\n  blocks mode: the same kernel, one block of work, the grid raised anyway\n");
    printf("     blocks    launches   us/launch   us/issued     ns/block\n");
    memset(params, 0, sizeof(params));
    { uint64_t p0 = (uint64_t)a, p1 = (uint64_t)b, p2 = (uint64_t)c;
      memcpy(params + info.params[0].offset, &p0, 8);
      memcpy(params + info.params[1].offset, &p1, 8);
      memcpy(params + info.params[2].offset, &p2, 8);
      memcpy(params + info.params[3].offset, &n1, 4); }
    for (int i = 0; i < (int)(sizeof(counts) / sizeof(*counts)); i++) {
      uint32_t g = counts[i], reps = g >= 6800 ? 2000 : 4000;
      T(tinynv_launch(s, k[0], g, 1, 1, BLOCK, 1, 1, 0, params, plen));
      T(tinynv_stream_sync(s));
      double t0 = now();
      for (uint32_t r = 0; r < reps; r++)
        T(tinynv_launch(s, k[r % (uint32_t)nmod], g, 1, 1, BLOCK, 1, 1, 0, params, plen));
      double t_issued = now();
      T(tinynv_stream_sync(s));
      double us = (now() - t0) * 1e6 / reps, us_issue = (t_issued - t0) * 1e6 / reps;
      printf("  %9u    %8u    %8.2f    %8.2f    %9.1f\n", g, reps, us, us_issue, us * 1000.0 / (double)g);
    }
    printf("\n  climbing with the grid means the cost is per block and lives in the descriptor or the distributor;\n"
           "  flat means blocks are innocent and the elementwise kernels themselves are what is slow.\n");
  }

  // --- idle mode: what does a launch cost when the engine has been doing nothing? -------------------------------
  //
  // The reason this exists. A decode leaves the engine dry for ~1.7 ms at every token boundary, and the budget for
  // that gap is now split three ways: ~0.7 ms is llama.cpp turning the token around, ~0.13 ms is libtinynv, and
  // ~0.8 ms is the engine taking that long to start work that is ALREADY ON ITS RING. 0.8 ms is six hundred times the
  // 1.28 us back-to-back launch floor, so either something wakes slowly or a clock is in the wrong place.
  //
  // Session A named the candidate that fits best: the engine powers down during an idle of that order, and waking it
  // costs hundreds of microseconds. It explains the one thing the other candidates do not, which is that the number is
  // nearly the same for two models with very different work - 828 us on the 27B and 726 on the mixture. A cost that
  // depends on the idle rather than on the work should behave exactly like that.
  //
  // So make the idle the independent variable, which is the same trick the size sweep above uses on the work. Hold the
  // launch fixed and tiny, vary how long the engine has been idle before it, and time one launch end to end.
  //
  // Read it as: flat across the sweep means the engine does not care how long it has been idle, the ~800 us is in the
  // refill batch itself - its acquire, its descriptor copy, its stamp - and we look there. A curve that climbs with
  // idle and reaches the hundreds of microseconds is a cold start, and then the knee says at what idle it begins,
  // which is the number a keep-warm would be sized against: a trivial kernel every so often during a stall costs a few
  // microseconds and would be worth it only if it lands the far side of that knee.
  //
  // The idle is a busy-wait rather than a sleep, deliberately. nanosleep's own overhead is tens of microseconds here,
  // which is the same order as the smallest intervals being asked about, and a sleeping host is a second variable -
  // the question is what the ENGINE does while nothing is submitted, not what the scheduler does to the host.
  if (getenv("TINYNV_GAP_IDLE")) {
    const double idles_us[] = {0, 50, 100, 250, 500, 1000, 2000, 5000, 10000};
    uint32_t n1 = BLOCK;                       // one block of work: the kernel is not what is being measured
    int reps = getenv("TINYNV_GAP_IDLE_REPS") ? atoi(getenv("TINYNV_GAP_IDLE_REPS")) : 200;
    printf("\n  idle mode: one small launch, timed end to end, after the engine has been idle for a known time\n");
    printf("    idle us    launches    mean us     min us     max us\n");
    memset(params, 0, sizeof(params));
    { uint64_t p0 = (uint64_t)a, p1 = (uint64_t)b, p2 = (uint64_t)c;
      memcpy(params + info.params[0].offset, &p0, 8);
      memcpy(params + info.params[1].offset, &p1, 8);
      memcpy(params + info.params[2].offset, &p2, 8);
      memcpy(params + info.params[3].offset, &n1, 4); }
    for (int i = 0; i < (int)(sizeof(idles_us) / sizeof(*idles_us)); i++) {
      double idle_s = idles_us[i] * 1e-6, sum = 0, lo = 1e9, hi = 0;
      // One launch first, so the first measured one is not also the first ever - a cold module would otherwise be
      // counted as a cold engine, and those are different things.
      T(tinynv_launch(s, k[0], 1, 1, 1, BLOCK, 1, 1, 0, params, plen));
      T(tinynv_stream_sync(s));
      for (int r = 0; r < reps; r++) {
        double until = now() + idle_s;
        while (now() < until) { }                     // the engine has nothing to do for exactly this long
        double t0 = now();
        T(tinynv_launch(s, k[0], 1, 1, 1, BLOCK, 1, 1, 0, params, plen));
        T(tinynv_stream_sync(s));
        double us = (now() - t0) * 1e6;
        sum += us;
        if (us < lo) lo = us;
        if (us > hi) hi = us;
      }
      printf("  %9.0f    %8d   %8.2f   %8.2f   %8.2f\n", idles_us[i], reps, sum / reps, lo, hi);
    }
    printf("\n  flat means the engine does not care how long it sat, and the ~800 us at a token boundary is in the\n"
           "  refill batch rather than in waking up; climbing means a cold start, and the knee is what a keep-warm\n"
           "  would have to beat.\n");
  }

  double gaps[16];
  for (int pass = 0; pass < passes; pass++) {
  if (passes > 1) printf("\n  pass %d of %d\n", pass + 1, passes);
  printf("        n    launches   us/launch   us/issued        GB/s   bytes touched\n");
  for (int i = 0; i < nsizes; i++) {
    uint32_t n = sizes[i], reps = launches_for(n);
    memset(params, 0, sizeof(params));
    uint64_t p0 = (uint64_t)a, p1 = (uint64_t)b, p2 = (uint64_t)c;
    memcpy(params + info.params[0].offset, &p0, 8);
    memcpy(params + info.params[1].offset, &p1, 8);
    memcpy(params + info.params[2].offset, &p2, 8);
    memcpy(params + info.params[3].offset, &n, 4);

    T(tinynv_launch(s, k[0], (n + BLOCK - 1) / BLOCK, 1, 1, BLOCK, 1, 1, 0, params, plen));   // warm the path
    T(tinynv_stream_sync(s));

    double t0 = now();
    for (uint32_t r = 0; r < reps; r++)
      T(tinynv_launch(s, k[(r / (uint32_t)run) % (uint32_t)nmod], (n + BLOCK - 1) / BLOCK, 1, 1, BLOCK, 1, 1, 0, params,
                      plen));
    // Issue and completion are different questions and this bench used to answer only their sum. The processor spends
    // real time in tinynv_launch - Session A measured about 4.5 us of it per launch on a real workload - so a wall of
    // 7.7 us per launch could be a GPU that cannot finish faster OR a processor that cannot issue faster, and those
    // have nothing to do with each other. Splitting them costs one timestamp and decides which question the rest of
    // the number belongs to: if issued and completed are the same, the GPU was never the limit and no amount of
    // reordering, chaining or unchaining will move it.
    double t_issued = now();
    T(tinynv_stream_sync(s));
    double us = (now() - t0) * 1e6 / reps, us_issue = (t_issued - t0) * 1e6 / reps;
    per[i] = us;
    double bytes = 12.0 * n;   // two reads and a write per element
    printf("  %8u    %8u    %8.2f    %8.2f    %8.1f    %8.2f MB\n", n, reps, us, us_issue, bytes / us / 1000.0,
           bytes / 1e6);
  }

  // Two points far apart give the line. The largest size is where execution dominates and the smallest is where it is
  // nearly nothing, so the slope between them is the memory rate and the intercept is what a launch costs with no work
  // in it at all.
  double n_lo = sizes[0], n_hi = sizes[nsizes - 1];
  double slope = (per[nsizes - 1] - per[0]) / (n_hi - n_lo);     // us per element
  double gap = per[0] - slope * n_lo;
  gaps[pass < 16 ? pass : 15] = gap;
  printf("\n  us/issued is the processor alone - how fast tinynv_launch can hand work over, with nothing waited on.\n");
  printf("  where it equals us/launch the GPU was never the limit and the number below is the processor's, not the card's.\n");
  printf("\n  fitted from the smallest and largest: the gap between kernels is %.2f us,\n", gap);
  printf("  and the memory rate during one is %.0f GB/s of the card's ~1790.\n", 12.0 / slope / 1000.0);
  }

  // The spread across passes, which is the only thing that says whether a difference between two runs meant anything.
  if (passes > 1) {
    double lo = gaps[0], hi = gaps[0], sum = 0;
    int np = passes < 16 ? passes : 16;
    for (int i = 0; i < np; i++) { if (gaps[i] < lo) lo = gaps[i]; if (gaps[i] > hi) hi = gaps[i]; sum += gaps[i]; }
    printf("\n  gap across %d passes: mean %.2f us, spread %.2f (%.2f to %.2f).\n", np, sum / np, hi - lo, lo, hi);
    printf("  A difference between two configurations smaller than %.2f us has not been measured by this bench.\n",
           hi - lo);
  }
  printf("\n  a gap near the 28.7 us a real decode launch costs means the space between kernels is the whole problem;\n"
         "  a rate far below 1790 means one serialised kernel cannot fill the bus and launching faster will not help.\n");

  T(tinynv_free(dev, a)); T(tinynv_free(dev, b)); T(tinynv_free(dev, c));
  T(tinynv_stream_destroy(s));
  for (int i = 0; i < nmod; i++) T(tinynv_module_unload(mod[i]));
  return 0;
}
