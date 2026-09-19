// Which submission path the driver takes, checked without a GPU.
//
// This test exists because that decision was, until it was written, invisible off a card: tinynv_exec_init runs only
// after a boot, so the shipping default could be confirmed only by plugging in an eGPU and reading a startup line. A
// default that can flip between "every batch waited on" and "nothing waited on" is not a good thing to find out about
// that way, and the flip to async is exactly the change that needed a check somewhere cheaper.
//
// So the whole table is pinned here: what an unset environment gets, both spellings of each ask, the consistent pairs,
// and the two contradictions. The reason strings are checked too, not only the mode - the startup line is what every
// report of a hang is read against, and a line that names the wrong cause is worse than one that says nothing.
#include "exec.h"
#include <stdio.h>
#include <string.h>

static int fails;
// Every CHECK counts itself, and the run asserts it executed at least as many as there are CHECK sites in this file -
// a number the Makefile greps out at build time, so it cannot go stale the way a literal would.
//
// The reason, from Session C: their guarantee-5 test reported PASSED while testing nothing. The branch that asked the
// question was never reached, it printed "(inconclusive)" and counted nothing, and the suite called that success. The
// rule they drew is "any branch that cannot reach a verdict is a failure", and the same hole is open in this file
// wherever checks sit inside a loop over recorded data: a recording that failed to parse would run the loop zero
// times and still print "all checks passed". A floor cannot catch every skipped case, but it catches a whole block
// going missing, which is the shape that actually happens.
static int checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

// NULL is unset; "" is set-but-empty, which every knob in exec.c treats as no answer either way.
static const struct { const char *async, *sync; int want; const char *why; const char *what; } TABLE[] = {
  {NULL, NULL, 0, "the default",                "an unset environment is asynchronous - this is the shipping default"},
  {"",   "",   0, "the default",                "empty is not an answer, so it is still the default"},
  {"1",  NULL, 0, "TINYNV_ASYNC was asked for", "TINYNV_ASYNC=1, which every existing script and note still carries"},
  {"1",  "",   0, "TINYNV_ASYNC was asked for", "and it survives an empty TINYNV_SYNC beside it"},
  {NULL, "1",  1, "TINYNV_SYNC was asked for",  "TINYNV_SYNC=1 is the opt-out, and the whole point of keeping sync"},
  {NULL, "0",  0, "TINYNV_SYNC=0 was asked for","TINYNV_SYNC=0 is the other spelling of asking for async"},
  {"0",  NULL, 1, "TINYNV_ASYNC=0 was asked for","TINYNV_ASYNC=0 is the other spelling of asking for sync"},
  {"0",  "1",  1, "TINYNV_SYNC was asked for",  "both asking for sync is not a contradiction"},
  {"1",  "0",  0, "TINYNV_ASYNC was asked for", "both asking for async is not one either"},
  // A command line that already carries TINYNV_ASYNC=1 - and they all do - with TINYNV_SYNC=1 added to it is someone
  // reaching for the proven path because something is wrong. Async winning there would make that step do nothing.
  {"1",  "1",  1, "TINYNV_SYNC and TINYNV_ASYNC ask for opposite things; taking the synchronous path",
   "asking for both takes the synchronous path and says so"},
  {"0",  "0",  1, "TINYNV_SYNC and TINYNV_ASYNC ask for opposite things; taking the synchronous path",
   "refusing both is the same contradiction and is reported the same way"},
  // A value is on when it is anything non-empty that does not start with '0', which is the convention the arena and
  // prefetch knobs already use. Nobody should write these, but somebody will.
  {NULL, "yes", 1, "TINYNV_SYNC was asked for", "a non-numeric value is still an ask"},
  {NULL, "0x", 0, "TINYNV_SYNC=0 was asked for","and one starting with '0' is still off"},
};

int main(void) {
  for (size_t i = 0; i < sizeof(TABLE) / sizeof(*TABLE); i++) {
    const char *why = NULL;
    int got = tinynv_exec_sync_mode(TABLE[i].async, TABLE[i].sync, &why);
    CHECK(got == TABLE[i].want, "TINYNV_ASYNC=%s TINYNV_SYNC=%s submitted %s, expected %s (%s)",
          TABLE[i].async ? TABLE[i].async : "(unset)", TABLE[i].sync ? TABLE[i].sync : "(unset)",
          got ? "synchronously" : "asynchronously", TABLE[i].want ? "synchronously" : "asynchronously", TABLE[i].what);
    CHECK(why && !strcmp(why, TABLE[i].why), "TINYNV_ASYNC=%s TINYNV_SYNC=%s gave the reason \"%s\", expected \"%s\"",
          TABLE[i].async ? TABLE[i].async : "(unset)", TABLE[i].sync ? TABLE[i].sync : "(unset)",
          why ? why : "(none)", TABLE[i].why);
  }
  // The two defaults a run with NO environment at all now depends on. They were opt-in until 2026-09-15 and are the
  // path itself now, so "the defaults carry it" has to be something that can fail here rather than something read off
  // a startup line by whoever happened to run it.
  struct { const char *e; int want; const char *what; } depths[] = {
    {NULL,   TINYNV_EXEC_CHAIN_DEFAULT, "unset chains as deep as the default"},
    {"",     TINYNV_EXEC_CHAIN_DEFAULT, "empty is the same as unset"},
    {"32",   32,                        "a value below the maximum is taken as written"},
    {"1",    1,                         "one launch a batch is allowed, and is how chaining is turned off"},
    {"0",    1,                         "zero is not a depth; it becomes one rather than a batch that never goes"},
    {"-5",   1,                         "and neither is a negative"},
    {"9999", TINYNV_EXEC_CHAIN_MAX,     "above the maximum is clamped, not refused: the array is sized at compile time"},
  };
  for (size_t i = 0; i < sizeof(depths) / sizeof(*depths); i++) {
    int got = tinynv_exec_chain_depth(depths[i].e);
    CHECK(got == depths[i].want, "TINYNV_CHAIN_DEPTH=%s chained %d deep, expected %d (%s)",
          depths[i].e ? depths[i].e : "(unset)", got, depths[i].want, depths[i].what);
  }
  CHECK(TINYNV_EXEC_CHAIN_DEFAULT == 128, "the default chain depth is %d, and the fast path wants 128",
        TINYNV_EXEC_CHAIN_DEFAULT);

  struct { const char *e; int vram, want; const char *why; const char *what; } dma[] = {
    {NULL, 1, 1, "the default",            "unset delivers by copy engine, which is the whole of the fast path"},
    {"",   1, 1, "the default",            "empty is the same as unset"},
    {"1",  1, 1, "TINYNV_ARENA_DMA was asked for", "asking for it is still an ask, and says so"},
    {"0",  1, 0, "TINYNV_ARENA_DMA=0 was asked for", "zero is the way back to writing across the link"},
    {"0x", 1, 0, "TINYNV_ARENA_DMA=0 was asked for", "anything starting with zero is off, as the other knobs read it"},
    {NULL, 0, 0, "the arena is in host memory, so there is nothing to deliver",
     "host memory forces it off with a reason, rather than being corrected afterwards"},
    {"1",  0, 0, "the arena is in host memory, so there is nothing to deliver",
     "and asking for it there does not override that"},
  };
  for (size_t i = 0; i < sizeof(dma) / sizeof(*dma); i++) {
    const char *why = NULL;
    int got = tinynv_exec_arena_dma(dma[i].e, dma[i].vram, &why);
    CHECK(got == dma[i].want, "TINYNV_ARENA_DMA=%s with the arena in %s delivered %s, expected %s (%s)",
          dma[i].e ? dma[i].e : "(unset)", dma[i].vram ? "video memory" : "host memory",
          got ? "by copy engine" : "across the link", dma[i].want ? "by copy engine" : "across the link", dma[i].what);
    CHECK(why && !strcmp(why, dma[i].why), "TINYNV_ARENA_DMA=%s gave the reason \"%s\", expected \"%s\"",
          dma[i].e ? dma[i].e : "(unset)", why ? why : "(none)", dma[i].why);
  }

  // Where the timeline release lives, which decides which release carries the profile's clock. Pinned here because
  // getting it wrong is not visible in any output: both configurations run, both produce correct results, and the only
  // symptom is an instrument measuring a release that the run never executes. That happened three times on the card
  // before anyone checked which branch the default takes.
  {
    static const struct { const char *e; int want; const char *what; } tails[] = {
      {NULL, 1, "unset: only the chain's tail releases, the default since 2026-09-19 and the path a decode takes"},
      {"", 1, "empty: same as unset"},
      {"0", 0, "explicitly off: every descriptor releases, the setting that can locate a stall inside a chain"},
      {"1", 1, "asked for: the tail only"},
    };
    for (size_t i = 0; i < sizeof(tails) / sizeof(*tails); i++) {
      int got = tinynv_exec_tail_release(tails[i].e);
      CHECK(got == tails[i].want, "TINYNV_TAIL_RELEASE=%s attached the release to %s, expected %s (%s)",
            tails[i].e ? tails[i].e : "(unset)", got ? "the tail only" : "every descriptor",
            tails[i].want ? "the tail only" : "every descriptor", tails[i].what);
    }
    CHECK(tinynv_exec_tail_release(NULL) == 1,
          "the default changed: the flush-time tail release is no longer the one a decode executes, so whatever the "
          "profile stamps must move with it - and the delta path's per-token identity of descriptors goes with it");
  }

  // The delta path's two knobs. On by default since 2026-09-19 because the three together (with the tail release
  // above) are what measured as a win on both models; delivery without the rewind is the one combination measured
  // as a loss, and the rewind without delivery is meaningless, so it follows delivery off.
  {
    static const struct { const char *d, *r; int want_d, want_r; const char *what; } deltas[] = {
      {NULL, NULL, 1, 1, "an empty environment patches and rewinds - the shipping default"},
      {"", "", 1, 1, "empty is not an answer either way"},
      {"1", "1", 1, 1, "both asked for"},
      {"0", NULL, 0, 0, "delivery off takes the rewind with it"},
      {"0", "1", 0, 0, "the rewind cannot be asked for without delivery"},
      {"1", "0", 1, 0, "delivery without the rewind: allowed, and the startup line warns it is the measured loss"},
      {NULL, "0", 1, 0, "the rewind alone turned off"},
    };
    for (size_t i = 0; i < sizeof(deltas) / sizeof(*deltas); i++) {
      int d = tinynv_exec_delta_delivery(deltas[i].d), r = tinynv_exec_delta_rewind(d, deltas[i].r);
      CHECK(d == deltas[i].want_d && r == deltas[i].want_r,
            "TINYNV_DELTA_DELIVERY=%s TINYNV_DELTA_REWIND=%s resolved to delivery %d rewind %d, expected %d %d (%s)",
            deltas[i].d ? deltas[i].d : "(unset)", deltas[i].r ? deltas[i].r : "(unset)", d, r, deltas[i].want_d,
            deltas[i].want_r, deltas[i].what);
    }
    CHECK(tinynv_exec_delta_delivery(NULL) == 1 && tinynv_exec_delta_rewind(1, NULL) == 1 && tinynv_exec_tail_release(NULL) == 1,
          "the three knobs that measured as a win together are no longer all on by default");
  }

  // The descriptor-tail measurement knobs: the oracle's choice unless asked, and only the spellings that name a mode.
  {
    static const struct { const char *e; int want; } mb[] = {{NULL, TINYNV_QMD_MEMBAR_NONE}, {"", TINYNV_QMD_MEMBAR_NONE}, {"sys", TINYNV_QMD_MEMBAR_SYS},
                                                             {"gpu", TINYNV_QMD_MEMBAR_GPU}, {"none", TINYNV_QMD_MEMBAR_NONE}, {"1", TINYNV_QMD_MEMBAR_NONE}, {"GPU", TINYNV_QMD_MEMBAR_NONE}};
    for (size_t i = 0; i < sizeof(mb) / sizeof(*mb); i++)
      CHECK(tinynv_exec_qmd_membar(mb[i].e) == mb[i].want, "TINYNV_QMD_MEMBAR=%s gave %d, expected %d", mb[i].e ? mb[i].e : "(unset)",
            tinynv_exec_qmd_membar(mb[i].e), mb[i].want);
    static const struct { const char *e; int want; } inv[] = {{NULL, TINYNV_QMD_INVALIDATE_ALL}, {"", TINYNV_QMD_INVALIDATE_ALL}, {"all", TINYNV_QMD_INVALIDATE_ALL},
                                                              {"cb0", TINYNV_QMD_INVALIDATE_CB0}, {"none", TINYNV_QMD_INVALIDATE_NONE}, {"0", TINYNV_QMD_INVALIDATE_ALL}};
    for (size_t i = 0; i < sizeof(inv) / sizeof(*inv); i++)
      CHECK(tinynv_exec_qmd_invalidate(inv[i].e) == inv[i].want, "TINYNV_QMD_INVALIDATE=%s gave %d, expected %d",
            inv[i].e ? inv[i].e : "(unset)", tinynv_exec_qmd_invalidate(inv[i].e), inv[i].want);
    CHECK(tinynv_exec_qmd_membar(NULL) == TINYNV_QMD_MEMBAR_NONE && tinynv_exec_qmd_invalidate(NULL) == TINYNV_QMD_INVALIDATE_ALL,
          "the defaults moved: no barrier on non-releasing descriptors (2026-09-19, +3%% dense) and the oracle's invalidates");
  }

  // The inline-upload size cap: 4,096 unless asked, a dword multiple, never past the per-call ceiling.
  {
    static const struct { const char *e; uint32_t want; } im[] = {{NULL, TINYNV_INLINE_MAX}, {"", TINYNV_INLINE_MAX}, {"4096", 4096},
                                                                   {"8192", 8192}, {"8193", 8192}, {"3", TINYNV_INLINE_MAX},
                                                                   {"abc", TINYNV_INLINE_MAX}, {"0", TINYNV_INLINE_MAX},
                                                                   {"32764", 32764}, {"65536", TINYNV_INLINE_MAX}};
    for (size_t i = 0; i < sizeof(im) / sizeof(*im); i++)
      CHECK(tinynv_exec_inline_max(im[i].e) == im[i].want, "TINYNV_INLINE_MAX=%s gave %u, expected %u",
            im[i].e ? im[i].e : "(unset)", tinynv_exec_inline_max(im[i].e), im[i].want);
    CHECK(tinynv_exec_inline_max(NULL) == TINYNV_INLINE_MAX && TINYNV_INLINE_MAX == 32764,
          "the inline-upload default moved from the 32,764-byte per-call ceiling (2026-09-19, +2.3%% MoE)");
  }

  // The per-kernel profile: off unless asked, four windows not counted unless asked, and the name reader that turns
  // what nvcc emits into what a reader of the report wants. Real shapes from ggml-cuda, plus the edges.
  {
    static const struct { const char *e; int want; } kp[] = {{NULL, 0}, {"", 0}, {"0", 0}, {"1", 1}, {"yes", 1}};
    for (size_t i = 0; i < sizeof(kp) / sizeof(*kp); i++)
      CHECK(tinynv_exec_kernel_profile(kp[i].e) == kp[i].want, "TINYNV_KERNEL_PROFILE=%s resolved the wrong way",
            kp[i].e ? kp[i].e : "(unset)");
    CHECK(tinynv_exec_kernel_profile(NULL) == 0, "the kernel profile is a measurement and must be off unless asked");
    static const struct { const char *e; int want; } ks[] = {{NULL, 4}, {"", 4}, {"0", 0}, {"12", 12}, {"-3", 4}, {"x", 0}};
    for (size_t i = 0; i < sizeof(ks) / sizeof(*ks); i++)
      CHECK(tinynv_exec_kernel_profile_skip(ks[i].e) == ks[i].want, "TINYNV_KERNEL_PROFILE_SKIP=%s gave %d, expected %d",
            ks[i].e ? ks[i].e : "(unset)", tinynv_exec_kernel_profile_skip(ks[i].e), ks[i].want);
    static const struct { const char *m, *want; } nm[] = {
        {"_Z13mul_mat_vec_qIL9ggml_type12ELi1ELi1EEvPKvS2_PKiPfiiiiiiiiiiii", "mul_mat_vec_q<(ggml_type)12, 1, 1>"},
        {"_Z12rms_norm_f32ILi1024EEvPKfPfiiiiiiiii", "rms_norm_f32<1024>"},
        {"_Z11k_bin_bcastIXadL_Z6op_addffEEfffEvPKT0_PKT1_PT2_iiii", "k_bin_bcast<...>"},
        {"_Z22quantize_mmq_q8_1_cudaILb1EEvPKfPKiPvxxxxxx", "quantize_mmq_q8_1_cuda<true>"},
        {"_Z8soft_maxILb0ELi32EEvPKfS1_Pfiiffi", "soft_max<false, 32>"},
        {"_Z3fooIL9ggml_type2ELS_14EEv", "foo<(ggml_type)2, (ggml_type)14>"},
        {"_Z3barILin1EEv", "bar<-1>"},
        {"_ZN2ns6kernelEv", "ns::kernel"},
        {"_Z7k_scalePfS_fi", "k_scale"},
        {"copy1d", "copy1d"},
        {"_Z99x", "..."},
        {NULL, "?"}};
    for (size_t i = 0; i < sizeof(nm) / sizeof(*nm); i++) {
      char out[112];
      tinynv_kernel_short_name(nm[i].m, out, sizeof(out));
      CHECK(!strcmp(out, nm[i].want), "%s read as \"%s\", expected \"%s\"", nm[i].m ? nm[i].m : "(null)", out, nm[i].want);
    }
    { char small[8]; tinynv_kernel_short_name("_Z13mul_mat_vec_qILi1EEv", small, sizeof(small));
      CHECK(strlen(small) == 7 && !strncmp(small, "mul_mat", 7), "a short buffer is filled and terminated, got \"%s\"", small); }
  }

  // Whether a download drains the compute on the host before issuing its copy. Off by default: it is a test of a
  // mechanism, not yet a fix, and the two arms want measuring against each other rather than one being assumed.
  {
    static const struct { const char *e; int want; } dl[] = {{NULL, 0}, {"", 0}, {"0", 0}, {"1", 1}};
    for (size_t i = 0; i < sizeof(dl) / sizeof(*dl); i++)
      CHECK(tinynv_exec_download_sync_first(dl[i].e) == dl[i].want,
            "TINYNV_DOWNLOAD_SYNC_FIRST=%s resolved the wrong way", dl[i].e ? dl[i].e : "(unset)");
    CHECK(tinynv_exec_download_sync_first(NULL) == 0, "a download still issues its copy before draining, by default");
  }

  // The keep-alive: off unless asked, its spin ceiling 2,000 us unless asked and never absurd.
  {
    static const struct { const char *e; int want; } ka[] = {{NULL, 0}, {"", 0}, {"0", 0}, {"1", 1}};
    for (size_t i = 0; i < sizeof(ka) / sizeof(*ka); i++)
      CHECK(tinynv_exec_keepalive(ka[i].e) == ka[i].want, "TINYNV_KEEPALIVE=%s resolved the wrong way", ka[i].e ? ka[i].e : "(unset)");
    static const struct { const char *e; unsigned want; } kus[] = {{NULL, 2000}, {"", 2000}, {"500", 500}, {"10", 2000}, {"abc", 2000}, {"99999", 20000}};
    for (size_t i = 0; i < sizeof(kus) / sizeof(*kus); i++)
      CHECK(tinynv_exec_keepalive_us(kus[i].e) == kus[i].want, "TINYNV_KEEPALIVE_US=%s gave %u, expected %u",
            kus[i].e ? kus[i].e : "(unset)", tinynv_exec_keepalive_us(kus[i].e), kus[i].want);
    static const struct { const char *e; unsigned want; } kmin[] = {{NULL, 64}, {"", 64}, {"0", 0}, {"1024", 1024}, {"-5", 64}, {"x", 0}};
    for (size_t i = 0; i < sizeof(kmin) / sizeof(*kmin); i++)
      CHECK(tinynv_exec_keepalive_min_kb(kmin[i].e) == kmin[i].want, "TINYNV_KEEPALIVE_MIN_KB=%s gave %u, expected %u",
            kmin[i].e ? kmin[i].e : "(unset)", tinynv_exec_keepalive_min_kb(kmin[i].e), kmin[i].want);
    CHECK(tinynv_exec_keepalive(NULL) == 0, "the keep-alive is unmeasured and must be off unless asked");
  }

  // Whether small host-to-device copies ride in the pushbuffer. ON by default since it measured clean twice and +3%
  // on both models; it shipped off for a day first, because it changes the data path rather than an instrument and a
  // wrong one reads stale memory somewhere else entirely. The row for the default is the one that matters here - a
  // silent flip back is a 3% regression on every model with nothing to attribute it to.
  {
    static const struct { const char *e; int want; } inl[] = {{NULL, 1}, {"", 1}, {"0", 0}, {"1", 1}};
    for (size_t i = 0; i < sizeof(inl) / sizeof(*inl); i++)
      CHECK(tinynv_exec_inline_upload_mode(inl[i].e) == inl[i].want,
            "TINYNV_INLINE_UPLOAD=%s resolved the wrong way", inl[i].e ? inl[i].e : "(unset)");
    CHECK(tinynv_exec_inline_upload_mode(NULL) == 1,
          "small copies no longer ride in the pushbuffer by default. That is a single-stream regression, and if it is "
          "a response to corruption then c48a35e's ordering fix has been undone rather than this knob being wrong");
    CHECK(tinynv_exec_inline_upload_mode("0") == 0, "there is no way back to the copy engine");
  }

  // The short first chain after a standstill. Off by default and it must stay off until a card says otherwise: it
  // makes the batch that restarts the engine a two-launch chain, which costs ~22 us of chain overhead a token, and
  // that is only worth paying if it removes a handoff that costs more.
  {
    static const struct { const char *e; unsigned want; } sc[] = {{NULL, 0}, {"", 0}, {"0", 0}, {"2", 2}, {"-3", 0}};
    for (size_t i = 0; i < sizeof(sc) / sizeof(*sc); i++)
      CHECK(tinynv_exec_short_chain(sc[i].e) == sc[i].want, "TINYNV_SHORT_FIRST_CHAIN=%s gave %u, expected %u",
            sc[i].e ? sc[i].e : "(unset)", tinynv_exec_short_chain(sc[i].e), sc[i].want);
    CHECK(tinynv_exec_short_chain(NULL) == 0, "the first chain after a standstill is shortened by default, which is a "
                                              "chain's overhead a token for a benefit no card has confirmed");
  }

  // How many small uploads are held before one must be pushed out. A knob rather than a constant so the number in the
  // default comes off a measured curve: 16 is what a single-slot decode needs and is probably wrong for eight slots,
  // and picking the replacement by arithmetic is what nearly had a list sized against the wrong field this morning.
  {
    static const struct { const char *e; unsigned want; } pe[] = {
      {NULL, 16}, {"", 16}, {"1", 1}, {"64", 64}, {"0", 1}, {"-4", 1}, {"100000", TINYNV_INLINE_PEND_CAP_N}};
    for (size_t i = 0; i < sizeof(pe) / sizeof(*pe); i++)
      CHECK(tinynv_exec_pend_entries(pe[i].e) == pe[i].want, "TINYNV_INLINE_PEND=%s gave %u, expected %u",
            pe[i].e ? pe[i].e : "(unset)", tinynv_exec_pend_entries(pe[i].e), pe[i].want);
    CHECK(tinynv_exec_pend_entries(NULL) == 16, "the held-list default moved without a measurement saying so");
  }

  if (checks < TINYNV_MODE_CHECKS) {
    printf("  FAIL: %d checks ran but this file has %d CHECK sites - a block was skipped\n", checks,
           TINYNV_MODE_CHECKS);
    fails++;
  }
  printf("submission mode: %zu environments resolved, the unset one asynchronously; TINYNV_SYNC=1 is the way back\n",
         sizeof(TABLE) / sizeof(*TABLE));
  printf("defaults with no environment at all: chaining %d deep, descriptors delivered by the copy engine; "
         "%zu depths and %zu delivery cases resolved\n", tinynv_exec_chain_depth(NULL),
         sizeof(depths) / sizeof(*depths), sizeof(dma) / sizeof(*dma));

  // Last statement in main, for the reason spelled out at the end of test_rm_free.c: that file's guard sat
  // partway through main, the file grew past it, and two hundred lines of checks stopped counting without
  // any sign. This one was still correct when that was found - it is written this way so it stays correct
  // no matter what gets appended below.
  printf("test_exec_mode: %d checks, %d failed\n", checks, fails);
  return fails ? 1 : 0;
}
