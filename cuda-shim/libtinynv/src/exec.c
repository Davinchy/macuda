// See exec.h. This is the thinnest layer that turns encoded bytes into work a GPU actually does.
//
// It was deliberately synchronous - every batch submitted and then waited on - because the first question about a driver
// that has never run a kernel is whether the kernel ran, not how fast. That question is answered: an 8B model runs and
// produces correct text. The answer cost 600 waits per decoded token, each one a round trip to the card for work that
// nothing was waiting on, and that was most of the decode time.
//
// So work is submitted without the caller waiting, and that is now the default. It was not always: the first async
// submission took the card off the bus on Session A's exotic-kernel op set, and a driver that drops a card is worse
// than a slow one, so it spent a day being asked for by name while the three bugs behind that drop were found - a copy
// engine releasing a semaphore the engine had not reached (9db55a5), one semaphore slot written by two engines
// (de14f96), and whole chains overlapping because nothing ordered one against the next (588ba25). What flipped this
// default is not the speed, it is that the async path has since carried both target models through op-verify, a
// five-hundred token decode, byte-identical greedy runs and every hardware session, while the synchronous path was
// only ever the one that answered the first question. Synchronous remains, whole and tested, as TINYNV_SYNC=1: a
// driver whose fast path is new should keep its slow one reachable in one word. Three things make async safe, and all
// three are load-bearing:
//
//   The scratch an engine reads from is not reused until it has finished with it. The arena is only reset when it runs
//   out of room, and resetting waits for everything outstanding first - so a wrap costs one wait per few thousand
//   batches instead of one per batch.
//
//   Work on one queue is ordered against work on the other by a semaphore acquire in the command stream, not by the
//   host. Compute and copy run independently, so a copy submitted after a kernel could otherwise read what the kernel
//   had not yet written. The recorded oracle orders its queues exactly this way.
//
//   Anything that reads a result, or takes away memory the engine might still be reading, waits first. There are few
//   such places and they are named at each call.
#include "exec.h"
#include "internal.h"
#include "nv_structs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SEM_BYTES   0x1000
// Where each queue's slot lives in that allocation. Far enough apart that two engines writing at once are not writing
// the same cache line, which is not required for correctness here but costs nothing to avoid.
#define SEM_SLOT(i) ((uint64_t)(i) * 128)
// Two scratch slots for the engine's own clock, used only under the profile. A timestamped release writes the payload
// and then the clock eight bytes after it, so each needs sixteen bytes; they sit well clear of the queue slots.
//
// They exist because the residual this driver has left against the vendor is invisible to host-side profiling. The host
// spends about a microsecond a launch and the wall clock allows four and a half, so three quarters of a launch is the
// engine either working or waiting, and only the engine can say which. TS_DONE is written when a batch finishes;
// TS_START at the head of the first batch after the host stood still. The difference is the card idle across a
// synchronisation - the refill - which is the one bucket large enough to hold what is missing.
#define SEM_TS_START 384
// A second head stamp, taken BEFORE the batch's cross-queue acquire. The pair brackets the acquire exactly: one clock
// when the engine reaches this batch, one when it is allowed to proceed. See batch_begin.
#define SEM_TS_PRE   512
// When the copy engine last finished something. Read at the moment the host's wait returns, which is before the next
// batch's descriptor delivery, so what it holds then is the copy the token ENDED with - the logits coming back.
#define SEM_TS_COPY  640
// When the compute engine went idle at the end of a chained batch, written by a host release that waits for idle, into
// an address nothing else touches. Not read from the timeline's own report: see tinynv_exec_flush.
#define SEM_TS_TAIL  768
// When the copy engine BEGAN a transfer, as against when it finished. Stamped as the batch's first method after its
// acquire, so the pair brackets the transfer itself and separates "late to start" from "slow to run".
#define SEM_TS_CSTART 896
// Command buffers and descriptors both. Large on purpose: the arena wrapping is the only thing that stalls now, so
// its size is how many batches run between stalls - a few thousand here, against roughly six hundred launches a token.
#define CMD_BYTES   0x400000
// What the command buffer region gets of it; the rest is descriptors and their constant buffers.
#define CMD_REGION_BYTES 0x80000
// Megabytes of descriptor region when the copy engine delivers it and the window no longer bounds its size.
#define DESC_MB_DEFAULT 64
#define STAGE_BYTES (16u << 20)  // host memory copies pass through, a chunk at a time

// Phases, named so a reader of the summary does not have to hold an order in their head.
// WAIT and BLOCKED were added because the profile could not answer the question it was built for.
//
// On the 35B mixture the host spends 1.05 us a launch against a wall-clock budget of 4.6, so it is idle three quarters
// of the time - which means submission cost cannot be what separates us from the vendor driver on that model, and every
// lever aimed at submission would have missed. The time is somewhere the profile did not look, and the only bucket big
// enough is the host standing still at a synchronisation: llama.cpp asks for one about nineteen times a token.
//
// WAIT is every call; BLOCKED is only the ones that actually had to wait for the engine. The difference matters,
// because a wait that was already satisfied costs nothing and a wait that was not is the GPU and the host taking turns
// instead of overlapping.
enum { PROF_LAUNCH, PROF_FLUSH, PROF_PUSH, PROF_SUBMIT, PROF_POLL, PROF_WAIT, PROF_BLOCKED, PROF_N };
static const char *const PROF_NAME[7] = {"a launch, all of it", "a flush, all of it",
                                         "  of which pushing the shadow across",
                                         "  of which submitting (ring, pointer, doorbell)",
                                         "  of which asking about faults",
                                         "waiting at a synchronisation, every call",
                                         "  of which actually stood still for the engine"};
#define PROF_START(ex) double prof_t0_ = (ex)->profile ? now() : 0.0
#define PROF_END(ex, which) do { if ((ex)->profile) { (ex)->prof_ns[which] += (uint64_t)((now() - prof_t0_) * 1e9); \
                                                      (ex)->prof_n[which]++; } } while (0)

// Which region an allocation comes from. Descriptors and their constant buffers in one, command buffers in the other,
// exactly one writer each - which is what turns the wrap guard below from a partial check into a sufficient one.
enum { AR_DESC, AR_CMD };

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

// The timeline semaphore, read back through the same view the recorder sees, so this path is replayable like the rest.
static uint64_t sem_read_slot(tinynv_exec_t *ex, int q) {
  uint64_t v = 0;
  nv_rd_block(&ex->sem.dma.view, SEM_SLOT(q), &v, sizeof(v));
  // A slot is written by one channel, which retires in order, so it must never go backwards. If it does, the whole
  // scheme this file rests on is void - an acquire can be parked on a value that has already been and gone - and that
  // is worth saying loudly rather than discovering again three probes later.
  if (v > ex->sem_high[q]) ex->sem_high[q] = v;
  else if (v < ex->sem_high[q]) {
    // Rate limited, and it has to be: a wait spins on this, so one report per read turned a stalled run into a 4.4 GB
    // log. The first few say what is happening and the count at the stall says how much of it there was; nothing after
    // that is information.
    if (ex->nbackwards++ < 8)
      fprintf(stderr, "tinynv: the %s timeline went backwards, %llu after %llu - a slot is being written by more than "
              "one thing\n", q ? "copy" : "compute", (unsigned long long)v, (unsigned long long)ex->sem_high[q]);
    else if (ex->nbackwards == 8)
      fprintf(stderr, "tinynv: (further backwards reads counted, not printed)\n");
  }
  return v;
}

// What both queues have finished, which is the smaller promise: a caller waiting for "everything" waits for each queue
// to have reached what it was last asked for, and those are different numbers on the same counter.
static int sem_reached(tinynv_exec_t *ex, uint64_t value) {
  for (int q = 0; q < 2; q++) {
    uint64_t want = ex->q_last[q];
    if (want > value) want = value;          // only as far as the caller asked
    if (want && sem_read_slot(ex, q) < want) return 0;
  }
  return 1;
}

// A cheap checksum over a descriptor, so that "is it still what we wrote" can be asked at a stall. It does not need to
// be a good hash; it needs to notice a descriptor whose bytes have been handed to something else.
static uint64_t qmd_sum(const uint8_t *b) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < TINYNV_QMD_BYTES; i++) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}

int tinynv_exec_wait(tinynv_exec_t *ex, uint64_t value, double seconds) {
  // Anything staged and not announced is work the engine has not been told about, so waiting for it would sit out the
  // whole timeout. This is normally empty; it is not when a flush failed between staging the delivery and submitting
  // the chain that would have announced it, and that is exactly the path where a hang would be hardest to read.
  if (tinynv_submit_ring(ex->g)) return -1;
  PROF_START(ex);
  uint64_t was = ex->prof_ns[PROF_WAIT];
  int spun = 0;
  double deadline = now() + seconds;
  for (;;) {
    // `done` decides; `v` is only for saying where things stand, and with a slot per queue there is no single number
    // for that any more. The compute slot is the one worth naming, because that is where the launches report.
    int done = sem_reached(ex, value);
    uint64_t v = sem_read_slot(ex, 0), vcopy = sem_read_slot(ex, 1);
    if (done) {
      // Everything the engine has passed is no longer interesting, and dropping it here is what keeps room for the
      // oldest thing that is still outstanding - which is the one a stall needs to show.
      uint64_t reached = v < vcopy ? vcopy : v;
      int keep = 0;
      for (int i = 0; i < ex->ntrail; i++)
        if (ex->trail[i].upto > reached) ex->trail[keep++] = ex->trail[i];
      ex->ntrail = keep;
      // Counted on the way out rather than at the top, because whether this call stood still is only known here.
      PROF_END(ex, PROF_WAIT);
      // Harvest first, arm second, and the order is the whole of it: the slot the "from" lives in is overwritten by
      // every batch, so a pair has to be closed before the next one is opened.
      if (ex->profile && ex->refill_armed) {
        uint64_t started = 0, reached = 0;
        nv_rd_block(&ex->sem.dma.view, SEM_TS_START + 8, &started, sizeof(started));
        // When the engine REACHED this batch, as against when it was allowed to proceed. The difference is the
        // acquire - on the launch path, waiting for the copy engine to deliver this batch's descriptors.
        nv_rd_block(&ex->sem.dma.view, SEM_TS_PRE + 8, &reached, sizeof(reached));
        ex->refill_reach_raw = reached;
        // A refill can only be forward in time. Anything else means the pair did not belong together - the head
        // timestamp had not landed when this was read - and a negative interval is worth discarding loudly rather than
        // folding into an average that then looks plausible.
        // Bounded, and the bound is the point. A refill gap is microseconds; a decode token is milliseconds. Anything
        // near a second is not a gap, it is two numbers that do not belong to the same clock - which is precisely what
        // this instrument reported for a day, at 3.35 SECONDS, while looking like an answer. The two halves come from
        // different mechanisms now (a host semaphore method for the head, a descriptor's second release slot for the
        // tail) and both write the gpu's clock at +8 of a 16-byte report - which test_headers holds against NVIDIA's
        // own NvGpuSemaphore, so it is a checked fact rather than the belief it was for an hour. The bound stays
        // anyway: the offset was only ever one of the ways two timestamps can fail to be on the same clock.
        // The raw pair, for the first few. Three wrong answers have come out of this instrument today and each time I
        // reasoned from the derived number to a cause and was wrong. The two inputs say immediately which kind of
        // wrong it is: a zero tail means the descriptor never wrote, a small integer means we are reading the payload
        // instead of the clock, and two plausible clocks far apart mean they are not the same clock.
        // Two windows, because the first eight refills are all prefill - A spotted that the printed pairs were ~2 ms
        // apart at the start of a run, which is pp256, so they said nothing about a decode. The second window opens
        // partway in, where a batch is a token.
        //
        // And the report's PAYLOAD beside its timestamp, which is the decisive one. Each descriptor releases its own
        // value to this address and the host waits for the last. If the payload read back is the LAST descriptor's
        // value while the timestamp beside it is the FIRST descriptor's clock, then the two halves of one 16 byte
        // report came from different releases - which means the later releases wrote two words where they were told
        // four, and per-descriptor timestamps are not a mechanism we can use. If both halves agree, they are not.
        unsigned w = ex->refill_raw_n < TINYNV_REFILL_RAW / 2                              ? ex->refill_raw_n
                   : (ex->refill_n + ex->refill_bad >= ex->refill_raw_from) ? ex->refill_raw_n : TINYNV_REFILL_RAW;
        if (w < TINYNV_REFILL_RAW) {
          uint64_t payload = 0;
          nv_rd_block(&ex->sem.dma.view, SEM_SLOT(0), &payload, sizeof(payload));
          ex->refill_raw[w][0] = ex->refill_from;
          ex->refill_raw[w][1] = started;
          ex->refill_raw[w][2] = payload;
          // q_last[0], not ex->timeline. ex->timeline is the highest value handed to EITHER queue, and the descriptor
          // delivery copy takes numbers after the chain's, so comparing against it reported "payload is NOT the last
          // value" on every healthy refill. The label was wrong, not the data, and it is why the two-release lag read
          // as a defect in the report rather than as the tail being stamped before the lm_head.
          ex->refill_raw[w][3] = ex->q_last[0];
          // The copy engine's clock as captured at the stall, so the 833 us it reports can be read against the tail
          // rather than inferred from a mean. A's sweeps say the logits transfer costs <=245 us end to end after any
          // idle this decode produces, so either this interval contains something other than the transfer or the
          // capture is not where I think it is - and a mean cannot tell those apart.
          ex->refill_raw[w][4] = ex->refill_copy_from;
          // `reached` and the copy's start, raw. A's column: the bimodal ~490/~1,150 interval lies after the tiny
          // copies (the host times those and they are steady at ~370 us on 23 of 24 tokens), so it is between the
          // refill batch's doorbell and its first method executing. reached splits that in two - if reached moves with
          // the population it is the compute engine's pickup after the doorbell; if reached is steady and head-reached
          // moves, it is the acquire on the descriptor delivery, whose 37 us MEAN would be hiding a bimodal 37/740.
          ex->refill_raw[w][5] = ex->refill_reach_raw;
          ex->refill_raw[w][6] = ex->refill_cstart;
          ex->refill_raw_n = w + 1;
        }
        // A tail that has not moved since the last refill is a specific failure with a specific cause - the report
        // being read is not the one the completed work wrote - and it is worth naming rather than counting as "not
        // physical". It cost three card runs to find by hand, twice because I had stamped a release that the running
        // configuration does not execute, and the symptom was always this: one value, stuck, while the head advanced.
        if (ex->refill_from && ex->refill_from == ex->refill_last_tail) ex->refill_stuck++;
        ex->refill_last_tail = ex->refill_from;
        // HEAD TO HEAD, which is the token period as the card sees it and cannot be fooled by where the tail sits.
        //
        // A found the sixth instrument of mine today measuring an interval that was not the one its label named: with
        // the download and the tiny uploads both served on the compute queue, the last thing to stamp a tail before a
        // refill is one of THOSE batches, a few hundred microseconds into what used to be the boundary. tail->head
        // duly collapsed to ~200 us and the bimodality "vanished" - while the token period barely moved, because the
        // interval had shrunk around the thing it was measuring rather than the thing getting cheaper.
        //
        // The head stamp is emitted only on a refill batch, so the gap between consecutive heads IS one token. It does
        // not care which batches stamp tails, how many queues are involved, or what the driver reroutes next.
        if (ex->refill_prev_head && started > ex->refill_prev_head) {
          uint64_t period = started - ex->refill_prev_head;
          if (period < 100000000ull) {
            ex->refill_period_ns += period;
            ex->refill_period_n++;
            // And again over the decode window alone. A caught the mean reading 18.3 ms on a run whose tokens are
            // 15.7 ms: the heads include prefill and warm-up batches, whose spacing is not a token, and they drag it
            // up by ~2.5 ms. That is a mean over a mixed population - the same shape that hid the answer twice today
            // - and the fix is to say which population, not to pick a cleverer average.
            if (ex->refill_n + ex->refill_bad >= ex->refill_raw_from) {
              ex->refill_period_dec_ns += period;
              ex->refill_period_dec_n++;
            }
          }
        }
        ex->refill_prev_head = started;
        uint64_t gap = started > ex->refill_from ? started - ex->refill_from : 0;
        if (gap && gap < 100000000ull) {           // 100 ms
          ex->refill_ns += gap;
          ex->refill_n++;
          if (gap > ex->refill_max_ns) ex->refill_max_ns = gap;
          // Inside the accepted branch, and that is the fix rather than a tidy-up. These had a bound of their own -
          // none - while the gap had one, so refill 0's four second load gap went into this mean and nowhere else. It
          // came out as "12,071 us before the engine reached the batch" inside a 1,775 us gap: one sample in 411
          // carrying 4.2 seconds. A saw the number was not physical; the cause was not the one either of us proposed,
          // and it was visible in the arithmetic without another card run - 12,071 x 411 is 4.96 s, which is refill 0
          // plus 410 ordinary gaps.
          //
          // The rule this should have followed from the start: a derived number must be accepted or rejected by the
          // SAME test as the number it decomposes, or the two disagree and the disagreement looks like physics.
          // How much of the gap the copy engine was still working through after the last kernel retired.
          if (ex->refill_copy_from > ex->refill_from && started >= ex->refill_copy_from) {
            ex->refill_copy_ns += ex->refill_copy_from - ex->refill_from;
            ex->refill_copy_n++;
            // And the split: how long the copy engine took to BEGIN, against how long the transfer then ran.
            if (ex->refill_cstart >= ex->refill_from && ex->refill_copy_from >= ex->refill_cstart) {
              ex->refill_cwait_ns += ex->refill_cstart - ex->refill_from;
              ex->refill_crun_ns += ex->refill_copy_from - ex->refill_cstart;
              ex->refill_cstart_n++;
            }
          }
          if (reached > ex->refill_from && started >= reached) {
            ex->refill_reach_ns += reached - ex->refill_from;
            ex->refill_acq_ns += started - reached;
            ex->refill_reach_n++;
          }
        } else {
          ex->refill_bad++;
        }
        ex->refill_armed = 0;
      }
      // Not profile-gated, unlike refill_pending beside it: the short first chain is a MODE, and a mode that only
      // works while you are measuring is not a mode. That mistake cost three card runs this morning in the other
      // direction - instrumenting a branch the running configuration never takes.
      if (spun) ex->after_stall = 1;
      if (spun && ex->profile) {
        ex->prof_ns[PROF_BLOCKED] += ex->prof_ns[PROF_WAIT] - was;
        ex->prof_n[PROF_BLOCKED]++;
        // The finish time of the batch that just satisfied this wait, taken now because the refill will overwrite it.
        // +8 of the compute timeline's own report: the clock at the moment the value the host just waited for was
        // released. See tinynv_exec_flush for why this is not a slot of its own.
        nv_rd_block(&ex->sem.dma.view, SEM_TS_TAIL + 8, &ex->refill_from, sizeof(ex->refill_from));
        // And what the copy engine had last finished at this same instant. A's point, and it is a real hole in every
        // bracket so far: the tail above is a COMPUTE clock - the token's last kernel - but between that kernel and
        // the next batch the copy engine reads the logits back to the host, ~600 KB to 1 MB over thunderbolt. That
        // work is real, necessary, and invisible to anything stamped on the compute queue, so it has been sitting
        // inside "the engine takes 1.7 ms to reach a batch already on its ring" as though it were latency.
        //
        // Read HERE rather than at the harvest, and that is the whole trick: by harvest time the refill batch's own
        // descriptor delivery has overwritten this slot. At this instant the last copy to have finished is the one
        // the host was waiting for.
        nv_rd_block(&ex->sem.dma.view, SEM_TS_COPY + 8, &ex->refill_copy_from, sizeof(ex->refill_copy_from));
        nv_rd_block(&ex->sem.dma.view, SEM_TS_CSTART + 8, &ex->refill_cstart, sizeof(ex->refill_cstart));
        // The engine has just gone idle and the host is about to start building again. Whatever goes out next is the
        // refill, and it is the only batch whose start time is worth knowing.
        ex->refill_pending = 1;
        ex->refill_asked++;
        // The host's own clock at the moment the engine went idle, and how much launch-building this driver had done
        // by then. Paired at the arm below, these say whether the gap the gpu reports is the host taking that long to
        // come back - and if it is, how much of it was spent inside libtinynv rather than above it.
        //
        // This is the question that decides who fixes it. The gpu says the engine is dry for ~1.8 ms a token. If the
        // host turnaround over the same interval is ~1.8 ms, the engine is dry because nobody gave it work; if it is
        // much less, the work was submitted and the engine took its time starting. Those have nothing in common.
        ex->refill_host_t0 = ex->refill_submit_from = now();
        ex->refill_build_t0 = ex->prof_ns[PROF_LAUNCH];
      }
      return 0;
    }
    spun = 1;

    if (now() > deadline) {
      // Before giving up, ask the firmware. An engine that faults says so on the status queue, and the message names
      // what it faulted on; without this the driver reports "it did not finish" and knows nothing else.
      int bad = tinynv_gsp_poll(ex->g);
      char what[160];
      // An address beats a pointer to the scrollback. The report above carries the kind and the access as well, but the
      // one number that decides where to look next belongs in the sentence that says the work did not finish.
      if (bad && ex->g->gsp.fault_have)
        snprintf(what, sizeof(what), "an engine faulted on address %#llx - the kind and the access are printed above",
                 (unsigned long long)ex->g->gsp.fault_va);
      else if (bad && ex->g->gsp.fault_sm_have)
        snprintf(what, sizeof(what), "no mmu fault, but an SM reports error %#x, so a kernel did something illegal "
                 "rather than touched something unmapped", ex->g->gsp.fault_sm_esr);
      else if (bad && ex->g->gsp.err_fn)
        snprintf(what, sizeof(what), "gsp-rm reported event %u (%s), payload %u bytes, printed above",
                 ex->g->gsp.err_fn,
                 ex->g->gsp.err_fn == TINYNV_MSG_EVENT_MMU_FAULT_QUEUED ? "mmu fault" : "error log",
                 ex->g->gsp.err_len);
      else if (bad)
        snprintf(what, sizeof(what), "gsp-rm refused something, which is recorded above but is not a fault report");
      else
        snprintf(what, sizeof(what), "gsp-rm reports nothing wrong, so the work was never picked up rather than "
                                     "attempted - look at what was submitted, not at what ran");
      // Where the engine stopped relative to the last chain. This is the difference between "a chain was never
      // dispatched" and "a chain was dispatched and stopped partway", which are different bugs with different fixes,
      // and no amount of staring at the two bare numbers separates them.
      char where[240] = "";
      if (ex->flush_n && ex->tail_release) {
        // Say it plainly rather than reporting a position derived from values that are no longer published. A chain
        // that releases only at its tail cannot be located from the timeline: the number either reached the end or it
        // did not, and everything in between is invisible by construction.
        snprintf(where, sizeof(where), " [TINYNV_TAIL_RELEASE is set, so only the last launch of each chain reports: "
                 "the chain covering %llu..%llu %s, and WHERE inside it the engine stopped cannot be known - re-run "
                 "without that knob to find out]", (unsigned long long)ex->flush_lo,
                 (unsigned long long)(ex->flush_lo + (uint64_t)ex->flush_n - 1),
                 v >= ex->flush_lo + (uint64_t)ex->flush_n - 1 ? "completed" : "did not complete");
      } else if (ex->flush_n) {
        uint64_t hi = ex->flush_lo + (uint64_t)ex->flush_n - 1;
        if (v >= ex->flush_lo && v <= hi)
          snprintf(where, sizeof(where), " [it stopped at link %llu of the %d in the chain covering %llu..%llu, so the "
                   "chain was dispatched and died partway]", (unsigned long long)(v - ex->flush_lo + 1), ex->flush_n,
                   (unsigned long long)ex->flush_lo, (unsigned long long)hi);
        else if (v < ex->flush_lo)
          snprintf(where, sizeof(where), " [it is short of the chain covering %llu..%llu, which was handed over and "
                   "never started]", (unsigned long long)ex->flush_lo, (unsigned long long)hi);
        else
          snprintf(where, sizeof(where), " [it is past the last chain (%llu..%llu), so what is missing was reserved "
                   "and never handed over]", (unsigned long long)ex->flush_lo, (unsigned long long)hi);
      }
      // What the driver published, which is worth knowing: if this has not moved, the submission never happened. The
      // engine's own read pointer is deliberately NOT reported. GPGet in the control block read as zero on both rings
      // while both engines were demonstrably running - the copy ring had published 1588 transfers that had plainly
      // happened - so it is not maintained the way the header's "read only" suggests, and the oracle never reads it
      // either. A number that is always zero is not evidence of anything, and printing it invites the reader to
      // conclude "the engine has taken none of it", which is exactly the wrong conclusion drawn from a broken gauge.
      char ring[200] = "";
      for (int qi = 0; qi < 2; qi++) {
        tinynv_queue_t *q = qi ? &ex->g->gsp.copy_q : &ex->g->gsp.compute_q;
        uint32_t get = 0, put = 0;
        uint64_t published = 0;
        tinynv_queue_state(ex->g, q, &get, &put, &published);
        char one[96];
        snprintf(one, sizeof(one), "%s%s ring: %u published, %llu written", qi ? "; " : " [",
                 qi ? "copy" : "compute", put, (unsigned long long)published);
        strncat(ring, one, sizeof(ring) - strlen(ring) - 1);
      }
      strncat(ring, "]", sizeof(ring) - strlen(ring) - 1);
      // And the batches the engine has not finished, oldest first, with what each is waiting for. This is the question
      // the other numbers cannot answer: an acquire for a value nothing will release parks a channel in silence, and it
      // shows up here as a batch whose "waits for" is a number no later batch ever reaches.
      if (ex->nbackwards)
        fprintf(stderr, "tinynv: a timeline slot read backwards %llu times - that is the bug, and everything below is "
                "downstream of it\n", (unsigned long long)ex->nbackwards);
      fprintf(stderr, "tinynv: outstanding batches, oldest first (compute is at %llu):\n", (unsigned long long)v);
      if (ex->ndropped)
        fprintf(stderr, "tinynv:   (%llu later submissions not recorded - the trail holds the oldest outstanding, which "
                "is the one that matters)\n", (unsigned long long)ex->ndropped);
      int shown = 0;
      for (int i = 0; i < ex->ntrail; i++) {
        size_t at = (size_t)i;
        if (ex->trail[at].upto <= v) continue;   // already done
        fprintf(stderr, "tinynv:   #%llu %s cmdbuf %#llx %u dwords, %u chained, reaches %llu",
                (unsigned long long)ex->trail[at].seq, ex->trail[at].copy ? "copy   " : "compute", (unsigned long long)ex->trail[at].va,
                ex->trail[at].dwords, ex->trail[at].links, (unsigned long long)ex->trail[at].upto);
        if (ex->trail[at].acquire)
          fprintf(stderr, ", waits for %llu%s", (unsigned long long)ex->trail[at].acquire,
                  ex->trail[at].acquire > v ? "  <-- NOT YET RELEASED, this channel is parked here" : " (already released)");
        if (ex->trail[at].acquire_other)
          fprintf(stderr, ", and on the other queue for %llu%s", (unsigned long long)ex->trail[at].acquire_other,
                  ex->trail[at].acquire_other > (ex->trail[at].copy ? v : vcopy)
                      ? "  <-- NOT YET RELEASED, this channel is parked here" : " (already released)");
        fprintf(stderr, "\n");

        // The two things that make a ready batch not run, asked of memory rather than guessed at. If the ring entry is
        // not what the driver wrote, the engine was handed something that was never valid work. If the chain head's
        // descriptor no longer checksums to what was written, its bytes were given to something else while it waited -
        // which is the arena reused underneath submitted work, and would be my bug rather than the hardware's.
        tinynv_queue_t *tq = ex->trail[at].copy ? &ex->g->gsp.copy_q : &ex->g->gsp.compute_q;
        uint64_t want = tinynv_ring_entry(ex->trail[at].va, ex->trail[at].dwords);
        uint64_t got = tinynv_ring_entry_read(ex->g, tq, ex->trail[at].slot);
        if (got != want)
          fprintf(stderr, "tinynv:       ring slot %llu holds %#llx, the driver wrote %#llx  <-- THE ENTRY IS WRONG\n",
                  (unsigned long long)ex->trail[at].slot, (unsigned long long)got, (unsigned long long)want);
        else
          fprintf(stderr, "tinynv:       ring slot %llu holds exactly what was written\n",
                  (unsigned long long)ex->trail[at].slot);
        if (ex->trail[at].head_va) {
          uint64_t off = ex->trail[at].head_va - ex->region[AR_DESC].mem.va;
          if (off + TINYNV_QMD_BYTES <= ex->region[AR_DESC].size) {
            // Read it from where the engine reads it. Checking the shadow was checking the copy that is never wrong
            // against itself, which is why every stall reported an intact descriptor while the engine was looking at
            // an arena those bytes had never reached.
            uint8_t seen[TINYNV_QMD_BYTES];
            if (ex->arena_vram && !ex->arena_dma)
              nv_rd_block(&ex->g->dev.vram, ex->region[AR_DESC].mem.ranges[0].paddr + off, seen, sizeof(seen));
            else if (ex->arena_vram)
              memcpy(seen, (const uint8_t *)ex->mirror.dma.va + off, sizeof(seen));  // what the engine was handed
            else memcpy(seen, (const uint8_t *)ex->region[AR_DESC].mem.dma.va + off, sizeof(seen));
            uint64_t now_sum = qmd_sum(seen);
            fprintf(stderr, "tinynv:       head descriptor at %#llx %s\n", (unsigned long long)ex->trail[at].head_va,
                    now_sum == ex->trail[at].head_sum ? "is still exactly as written"
                                                      : "HAS CHANGED SINCE IT WAS WRITTEN  <-- the arena was reused under it");
          }
        }
        shown++;
      }
      if (!shown) fprintf(stderr, "tinynv:   none - everything submitted has finished, so what is missing was never submitted\n");
      return tinynv_fail("the engines did not reach %llu within %.1fs (compute is at %llu, copy at %llu): %s%s%s",
                         (unsigned long long)value, seconds, (unsigned long long)v, (unsigned long long)vcopy,
                         what, where, ring);
    }
  }
}

// Scratch the GPU reads: command buffers, and the kernel descriptors that go with them. Two regions, one writer each,
// bump allocated, recycled a piece at a time when the bump reaches the top. Video memory by default now rather than
// host memory - the copy engine delivers the descriptors, so nothing but the engine reads them where they live.
//
// It is an arena rather than an allocation per use, and that is not an optimisation. Host memory the GPU can read is
// mapped by the dext, which allows 128 mappings per connection and **never gives one back** - freeing releases the
// pages and leaves the slot used. A descriptor allocated per launch would therefore work for 128 launches and then stop,
// which is a fault a test that launches a few times cannot find. Session A found the same cliff from the shim's side.
//
// Coming round is safe because it waits, and what it waits for is in arena.c: the piece being reused, and nothing else.
// It used to be safe because every batch outstanding was waited on, which is no longer true, and the difference
// between those two sentences is this whole file's change.

// Whether an allocation fits without the region having to start again from the bottom. Callers that hold a chain use
// this to flush before they allocate, because coming round with a built-but-unsubmitted descriptor in the region would
// hand its bytes to something else - arena() refuses that outright, so this is how a caller avoids reaching it.
//
// It asks the allocator rather than working it out, and that is the point of the line. This used to be a second copy
// of the placement arithmetic, and a second copy is the shape that drifts: the copy in arena.c is driven by a test and
// this one was not, so had they diverged the test would have gone on passing while the callers of THIS one decided
// wrongly whether they needed to flush.
static int arena_fits(tinynv_exec_t *ex, int r, uint64_t bytes, uint64_t align) {
  int wrap = 0;
  tinynv_arena_place(&ex->region[r], bytes, align, &wrap);
  return !wrap;
}

// Push what has been built since the last push into video memory. The engine reads descriptors from there, so bytes
// that are only in the shadow are bytes the engine will not see - it reads whatever the arena held before.
static void push_shadow(tinynv_exec_t *ex) {
  if (!ex->arena_vram) return;
  // Two spans rather than one, and that is required rather than tidy: allocations are consecutive within a region and
  // are not consecutive across them, so a single span covering both would push everything between - which, with the
  // regions megabytes apart, is most of the arena.
  for (int r = 0; r < 2; r++) {
    // In DMA mode the descriptor span is delivered by the copy engine at the flush and its span is cleared there, so
    // there is nothing here to write. Command buffers still come this way and must: the engine fetches a command
    // buffer to find the acquire inside it, so the acquire cannot be what protects it, and it has to be present before
    // the engine is told about it at all.
    if (ex->arena_dma && r == AR_DESC) continue;
    if (ex->region[r].dirty_hi <= ex->region[r].dirty_lo) continue;
    nv_wr_block(&ex->g->dev.vram, ex->region[r].mem.ranges[0].paddr + ex->region[r].dirty_lo,
                ex->region[r].shadow + ex->region[r].dirty_lo,
                (size_t)(ex->region[r].dirty_hi - ex->region[r].dirty_lo));
    ex->region[r].dirty_lo = ex->region[r].dirty_hi = 0;
  }
}

static void *arena(tinynv_exec_t *ex, int r, uint64_t bytes, uint64_t align, uint64_t *va) {
  tinynv_exec_region_t *rg = &ex->region[r];
  int wrap = 0;
  uint64_t off = tinynv_arena_place(rg, bytes, align, &wrap);
  if (wrap) {
    // Coming round hands back scratch an engine may still be reading. It must not happen while a chain is pending:
    // those descriptors live in this region and nothing has been told to run them yet, so no wait would cover them and
    // the rewind would give their bytes away.
    //
    // That guard used to be a partial check, and the gap cost six hardware runs: flush() zeroes the chain count and
    // then allocates its own command buffer, so when THAT allocation wrapped, the count was already zero and the reset
    // went ahead under descriptors built moments earlier. With a region each, flush's command buffer can no longer
    // reach the descriptors, so the check now covers every wrap that could reach them. It still has to exist - the
    // descriptors are live and nothing has been told to run them - but it is no longer the only thing standing between
    // us and that bug.
    if (r == AR_DESC && ex->nchain) {
      tinynv_fail("scratch wrapped with %d launches still unsubmitted", ex->nchain);
      return NULL;
    }
    // Push before rewinding, not after: whatever was built in the shadow and not yet sent still has to reach video
    // memory, and the rewind is about to hand those bytes to something else.
    push_shadow(ex);
    if (tinynv_exec_flush(ex)) return NULL;

    // The timeline restarts here, and this is the only place it safely can - but now only when it has to.
    //
    // It is a 32-bit counter because the copy engine's release is one word, and it advances once per launch - about a
    // hundred and thirty thousand a second while generating - so four billion is nine hours of continuous work rather
    // than the "nothing will reach it" the comment beside the limit used to claim. A server left running overnight
    // reaches it and every wait afterwards refuses.
    //
    // Restarting means saying that no value anybody still holds matters, and the only way to be sure of that is to
    // stand still: every outstanding batch finished, the slots zeroed with nothing left to write them. That used to be
    // free, because coming round drained the region anyway. It is not free now - the whole point of recycling in
    // pieces is that coming round no longer waits for everything - so it is taken on the counter's terms instead of
    // the region's, near the ceiling rather than every lap. In between the counter simply keeps climbing, which is
    // what it is for.
    if (r == AR_DESC && ex->reserved > TINYNV_EXEC_REBASE_AT) {
      double t0 = now();
      if (tinynv_exec_idle(ex)) return NULL;
      ex->wrap_wait_ns += (uint64_t)((now() - t0) * 1e9);
      ex->reserved = ex->timeline = 0;
      ex->q_last[0] = ex->q_last[1] = 0;
      ex->sem_high[0] = ex->sem_high[1] = 0;
      ex->last_q = NULL;                       // nothing to order the next batch against; it is all finished
      ex->ntrail = 0;                          // the trail's values are absolute and would now read as the future
      ex->flush_lo = ex->flush_n = 0;
      // And so would these. They are the whole reason the restart has to be taken from a standstill: a piece still
      // holding last lap's value would, after the counter moved under it, be asked to wait for a number the engine
      // will not reach for hours.
      for (int i = 0; i < 2; i++) memset(ex->region[i].seg_value, 0, sizeof ex->region[i].seg_value);
      // The lap position is deliberately left alone. Setting it to "everything entered" was here, on the grounds that
      // an idle leaves nothing to wait for - which is true, and is not the reason it was safe. The rewind below runs
      // in the same breath and zeroes it, so the line did nothing in the path anyone would trace; it survived only in
      // the case where flush had already come round, and there it was doing work whose justification was somewhere
      // else. With the records zeroed, entering a piece costs a comparison and returns nothing. That is cheap enough
      // that buying it with an argument was the wrong trade.
      memset(ex->sem.dma.va, 0, (size_t)ex->sem.size);
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    // Ask again rather than assuming. Flush allocates its own command buffer, from this very region when this is the
    // command-buffer region, and so may have come round already; rewinding on top of a buffer the engine has just been
    // handed would give its bytes away while it reads them.
    off = tinynv_arena_place(rg, bytes, align, &wrap);
    if (wrap) {
      // Refuse rather than carry on. Coming round with a span outstanding means bytes are about to be handed out that
      // something still owns, and that is the one failure here the card would not report - it would launch whatever is
      // in the descriptor now and return a wrong answer thousands of launches later.
      if (tinynv_arena_rewind(rg)) {
        tinynv_fail("the %s region came round with work neither submitted nor pushed",
                    r == AR_DESC ? "descriptor" : "command buffer");
        return NULL;
      }
      off = tinynv_arena_place(rg, bytes, align, &wrap);
      ex->wraps[r]++;
    }
  }
  if (wrap) {
    tinynv_fail("scratch: %llu bytes wanted, the %s region holds %llu", (unsigned long long)bytes,
                r == AR_DESC ? "descriptor" : "command buffer", (unsigned long long)rg->size);
    return NULL;
  }
  // And the same question for a boundary crossed without coming round at all: the piece being entered was last used a
  // lap ago, and its work has to have finished before its bytes are handed out again. This is where the cost of
  // recycling actually lands, and on a region sized for the work it is almost never taken - the piece being reused is
  // several pieces old by the time it comes up.
  //
  // No flush first, deliberately. A piece's value is recorded when its batch is submitted, never before, so what this
  // waits for is already on its way and nothing here has to push it. Flushing would submit the chain being built early
  // - a chain handover costs about fifteen microseconds - to no purpose, and from the command-buffer region it would
  // allocate out from under this very call.
  uint64_t want = tinynv_arena_enter(rg, off, bytes);
  if (want && !sem_reached(ex, want)) {
    double t0 = now();
    if (tinynv_exec_wait(ex, want, 30.0)) return NULL;
    ex->wrap_wait_ns += (uint64_t)((now() - t0) * 1e9);
    ex->seg_waits++;
  }
  *va = rg->mem.va + off;
  tinynv_arena_take(rg, off, bytes);
  // The shadow when the arena is in video memory, the mapping itself when it is in host memory. Either way the caller
  // writes here and the bytes reach the engine when the batch goes.
  return rg->shadow ? rg->shadow + off : (uint8_t *)rg->mem.dma.va + off;
}

static uint32_t *cmd_space(tinynv_exec_t *ex, uint32_t dwords, uint64_t *va) {
  return arena(ex, AR_CMD, (uint64_t)dwords * 4, 4, va);
}

// Begin a batch on a queue.
//
// If the last work went somewhere else, the engine is told to wait for it first. The two queues are independent - a copy
// submitted after a kernel will otherwise start before the kernel has written what it is copying - and while everything
// was waited on by the host that could not happen. It can now, so the ordering has to be in the command stream, which
// is where the oracle puts it too.
// Declared here because batch_begin needs it: a copy batch has to push any held inline uploads out as a compute batch
// of their own before it can proceed, and run() is defined below.
static int run(tinynv_exec_t *ex, tinynv_queue_t *q, tinynv_cmdbuf_t *c, uint64_t cmdbuf_va);

// Hold bytes for the next compute batch to carry. The caller has already checked they fit.
static void inline_pend_push(tinynv_exec_t *ex, uint64_t dst, const void *src, uint32_t len) {
  memcpy(ex->inline_pend_buf + ex->inline_pend_bytes, src, len);
  ex->inline_pend[ex->inline_pend_n].dst = dst;
  ex->inline_pend[ex->inline_pend_n].off = ex->inline_pend_bytes;
  ex->inline_pend[ex->inline_pend_n].len = len;
  ex->inline_pend_n++;
  ex->inline_pend_bytes += len;
}

// Room for one more held span of `len` bytes?
static int inline_pend_fits(const tinynv_exec_t *ex, uint32_t len) {
  return ex->inline_pend_n < ex->inline_pend_max && ex->inline_pend_bytes + len <= ex->inline_pend_bytes_max;
}

static int batch_begin(tinynv_exec_t *ex, tinynv_queue_t *q, uint32_t dwords, tinynv_cmdbuf_t *c, uint64_t *va) {
  // Everything that is not a launch is ordered after the launches already built: a copy that reads what a kernel wrote,
  // a wait, the setup batches. flush() clears the count before it builds anything, so the call it makes back into here
  // finds nothing pending and this does not recurse.
  if (tinynv_exec_flush(ex)) return -1;
  // A copy batch cannot carry compute-class methods, and a download that read memory whose bytes were still sitting in
  // our buffer would read stale device memory. So they go out first, in a batch of their own - the old behaviour, and
  // it happens only when a copy follows an upload with no launch between them. One level of recursion at most: the
  // list is empty when that call returns.
  // ...but not during a chain flush. The only copy that runs there is the descriptor delivery, which reads our own
  // mirror and never a caller's memory, so there is nothing for held bytes to be ordered against - and pushing them
  // out here as a compute batch is what broke the timeline. They ride out with the chain batch instead, which is
  // where they belonged all along.
  if (q == &ex->g->gsp.copy_q && ex->inline_pend_n && !ex->in_chain_flush) {
    uint64_t iva;
    tinynv_cmdbuf_t ic;
    if (batch_begin(ex, &ex->g->gsp.compute_q, 40, &ic, &iva)) return -1;
    ex->submitting_inline = 1;
    int rc = run(ex, &ex->g->gsp.compute_q, &ic, iva);
    ex->submitting_inline = 0;
    if (rc) return -1;
  }
  // The refill's head timestamp, and it has to be built here rather than in run(): run() is handed a command buffer
  // that already holds the batch's launches, so anything it appends times the end. One semaphore method is six dwords
  // and the space for it is reserved here, by the only code that knows the batch has not been sized for it.
  int head_ts = ex->profile && ex->refill_pending && q != &ex->g->gsp.copy_q;
  // Room for any small uploads waiting to ride along. They are compute-class methods, so only a compute batch can
  // carry them; a copy batch has already pushed them out above.
  uint32_t inl = 0;
  if (q != &ex->g->gsp.copy_q)
    for (unsigned i = 0; i < ex->inline_pend_n; i++) inl += TINYNV_INLINE_DWORDS(ex->inline_pend[i].len);
  uint32_t *w = cmd_space(ex, dwords + inl + (head_ts ? 12u : 0u), va);
  if (!w) return -1;
  *c = (tinynv_cmdbuf_t){.words = w, .n = 0, .cap = dwords + inl + (head_ts ? 12u : 0u)};
  // Synchronously there is nothing to order against: the previous batch was waited on before this one was built, so
  // the acquire would always be already satisfied. Leaving it out is what makes TINYNV_SYNC=1 the pre-d29cb32 path
  // rather than the new path with a wait bolted on, which matters when the question is whether the acquire is the bug.
  //
  // The flush above is load-bearing for this line, and the order of the two is the whole of it. A semaphore acquire
  // breaks the descriptor chain - the oracle says so and works around it by hand, waiting for the previous launch
  // whenever it emits a cross-queue wait on the compute queue. Here the flush has already handed the chain over and
  // moved the timeline to its last value, so waiting for the timeline *is* waiting for the previous launch, and the
  // workaround is the ordering of these two statements rather than a special case. Reverse them and dependent kernels
  // start running underneath a copy that was meant to follow them.
  // Before the acquire, deliberately, and it is the whole point of this one. The stamp below sits AFTER the acquire,
  // so it reports when the batch was allowed to PROCEED - which on the launch path means when the descriptor copy this
  // batch waits on had finished. The pair brackets that wait: this clock is when the engine reached the batch, that
  // one is when it could start, and the difference is time spent waiting on the copy engine rather than on anything
  // the compute engine was doing.
  //
  // Needed because the idle sweep came back flat: a launch after 2 ms of idle costs ~57 us end to end, not 800, so the
  // ~800 us a token boundary spends "starting work already on its ring" is not the engine waking up. A suggested the
  // acquire, and the acquire is the first thing this batch executes.
  if (head_ts && tinynv_cmd_timestamp(c, ex->sem.va + SEM_TS_PRE, ex->reserved + 1)) return -1;
  ex->pending_acquire = ex->pending_acquire_other = 0;
  int qi = q == &ex->g->gsp.copy_q;
  if (!ex->sync && ex->last_q && ex->last_q != q) {
    // Wait on the queue we are switching away from, in its own slot, for the value it was last asked for. Waiting on a
    // shared location for "the last value anyone asked for" was the bug: that number may have been written and then
    // overwritten by the other engine finishing something older, and the acquire then waits for it forever.
    int other = !qi;
    if (ex->q_last[other]) {
      if (tinynv_cmd_wait(c, ex->sem.va + SEM_SLOT(other), ex->q_last[other])) return -1;
      ex->pending_acquire_other = ex->q_last[other];
    }
  }
  // Emitted here: after the acquire that orders this batch, before anything the caller appends, and before the memory
  // barrier a launch batch emits next - so the bytes are written and flushed before any kernel in this batch can read
  // them, and ordered against the previous chain by the same acquire that orders everything else.
  if (q != &ex->g->gsp.copy_q && ex->inline_pend_n) {
    for (unsigned i = 0; i < ex->inline_pend_n; i++)
      if (tinynv_cmd_inline_upload(c, ex->inline_pend[i].dst, ex->inline_pend_buf + ex->inline_pend[i].off,
                                   ex->inline_pend[i].len))
        return -1;
    ex->inline_batches_saved += ex->inline_pend_n;
    ex->inline_pend_n = 0;
    ex->inline_pend_bytes = 0;
  }
  if (q != &ex->g->gsp.copy_q) ex->after_stall = 0;
  ex->last_q = q;
  // After the acquire, deliberately. Before it, this would record the moment the engine reached a wait it may then sit
  // on for as long as the other queue takes - which is not when this batch started working, and would report an idle
  // gap shorter than the truth by exactly the part that is somebody else's fault.
  if (head_ts) {
    if (tinynv_cmd_timestamp(c, ex->sem.va + SEM_TS_START, ex->reserved + 1)) return -1;
    ex->refill_pending = 0;
    ex->refill_armed = 1;
    ex->refill_armed_n++;
    if (ex->refill_host_t0 > 0.0) {
      ex->refill_host_ns += (uint64_t)((now() - ex->refill_host_t0) * 1e9);
      ex->refill_build_ns += ex->prof_ns[PROF_LAUNCH] - ex->refill_build_t0;
      ex->refill_host_n++;
      ex->refill_host_t0 = 0.0;
    }
    // The head timestamp is written into this batch's command buffer HERE, at its beginning, but the engine does not
    // execute it until the batch is submitted and picked up. So the gpu-measured gap runs from the engine going idle
    // all the way to the engine starting again, and the host-measured gap above stops at the moment the host began
    // BUILDING - the launches, the push and the submit all come after it. That is why the two differ by ~850 us and
    // neither half is wrong: they are measuring different ends of the same interval.
    //
    // This closes it. The clock is taken again when the batch is actually handed over, so the interval splits into
    // three: the caller turning the token around, this driver building and submitting, and whatever the engine takes
    // to start work that is already on the ring. The last of those is the only one with nowhere else to hide.
    ex->refill_submit_open = 1;
  }
  return 0;
}

// Hand a finished batch to a queue and wait for it. The release at the end of every batch is what is waited on.
// The half that is the same however the batch releases: make the bytes visible, hand them over, and move the timeline
// to what the batch will have reached. Synchronously, wait for it here.
static int submit_batch(tinynv_exec_t *ex, tinynv_queue_t *q, tinynv_cmdbuf_t *c, uint64_t cmdbuf_va, uint64_t upto,
                       int links) {
  uint64_t seq = ex->nsubmit++;
  if (ex->ntrail < TINYNV_EXEC_TRAIL) {
    size_t at = (size_t)ex->ntrail++;
    ex->trail[at].seq = seq;
    ex->trail[at].slot = q->entries ? q->put % q->entries : 0;
    ex->trail[at].head_va = ex->pending_head_va;
    ex->trail[at].head_sum = ex->pending_head_sum;
    ex->trail[at].va = cmdbuf_va;
    ex->trail[at].acquire = ex->pending_acquire;
    ex->trail[at].acquire_other = ex->pending_acquire_other;
    ex->trail[at].upto = upto;
    ex->trail[at].dwords = c->n;
    ex->trail[at].copy = q == &ex->g->gsp.copy_q;
    ex->trail[at].links = (uint8_t)(links > 255 ? 255 : links);
  } else {
    ex->ndropped++;
  }
  ex->pending_head_va = ex->pending_head_sum = 0;
  ex->pending_acquire = ex->pending_acquire_other = 0;

  // The one crossing. Everything this batch needs the engine to read - descriptors, constant buffers, the command
  // buffer itself - was built in the shadow and goes over in a single block write, so the engine then reads all of it
  // out of video memory locally. That is the whole change: it replaces roughly twenty small non-posted fetches coming
  // back across the link per launch with one posted write going out per batch.
  // Nothing is read back after this. These bytes and the ring's write pointer go to the same place by the same path,
  // and tinynv_submit reads that pointer back before it rings, which cannot pass these writes either.
  for (int r = 0; r < 2; r++) tinynv_arena_submitted(&ex->region[r], upto);
  { PROF_START(ex); push_shadow(ex); PROF_END(ex, PROF_PUSH); }
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  { PROF_START(ex);
    // Deferred, this writes the ring entry and the pointer and stops. The next batch out calls tinynv_submit, whose
    // ring half announces everything staged - so the pair costs one round trip instead of two. Never deferred
    // synchronously: submit_batch waits at the end of this function, and waiting on a batch nobody has rung is a hang.
    int rc = (ex->defer_ring && !ex->sync) ? tinynv_submit_stage(ex->g, q, cmdbuf_va, c->n)
                                           : tinynv_submit(ex->g, q, cmdbuf_va, c->n);
    PROF_END(ex, PROF_SUBMIT);
    if (rc) return -1; }
  // The highest value handed over, not the most recent one. With the descriptor delivery submitting a copy that takes
  // a number after the chain's, the compute batch is submitted with a LOWER value than the copy that preceded it, and
  // a timeline that took the latest would step backwards - under-waiting at the next idle, which is a wrap handing out
  // bytes an engine is still reading.
  // See the arm in batch_begin: this is the far end of the host's half of a refill.
  if (ex->refill_submit_open && ex->refill_submit_from > 0.0) {
    ex->refill_submit_ns += (uint64_t)((now() - ex->refill_submit_from) * 1e9);
    ex->refill_submit_n++;
    ex->refill_submit_open = 0;
    ex->refill_submit_from = 0.0;
  }
  // Anything that gets here other than an inline upload is work a caller may have to wait for. Set at the point of
  // submission rather than inferred later: a flag that is only correct if every future path remembers to set it is the
  // shape that has cost this project three card runs today, so the default is "needs waiting" and the single
  // exception is stated where it is taken.
  // Loud, because the silent version of this cost a day. A compute submission inside a chain flush takes a LATER
  // timeline value than the chain that is handed over after it, so the slot goes backwards by the length of the
  // chain - and the only symptom is a warning thousands of launches later, or a fault, or nothing at all for 370
  // requests. Anything reaching here in that window is the same bug wearing a different hat.
  if (ex->in_chain_flush && q == &ex->g->gsp.compute_q)
    return tinynv_fail("a compute batch was submitted inside a chain flush: it takes timeline value %llu and the "
                       "chain handed over after it releases a lower one, so the timeline would go backwards by the "
                       "length of the chain", (unsigned long long)upto);
  if (!ex->submitting_inline) ex->needs_wait = 1;
  if (upto > ex->timeline) ex->timeline = upto;
  ex->q_last[q == &ex->g->gsp.copy_q] = upto;
  return ex->sync ? tinynv_exec_wait(ex, upto, 30.0) : 0;
}

// Diagnostic only, built for the 2026-09-18 MoE per-token descriptor-delta measurement. Dumps every submitted
// batch's raw dwords, exactly as built, so two decode tokens' worth of batches can be diffed offline. TINYNV_
// DUMP_CMD=<path> turns it on; unset, this is one getenv call, cached.
//
// NEEDS A REAL DEVICE - checked and it does not work on the null device, corrected here after finding out the
// hard way: tinynv.c's launch path only reaches tinynv_exec_run (and so this hook) when s->dev->has_pci; without
// a card it takes an early-return "would launch" print and never touches exec.c's batch machinery at all. There
// is no card-free way to capture this.
//
// MEASURED, real hardware, 16-token MoE decode (Qwen3.5-35B-A3B), two consecutive steady-state tokens (same
// 26-record size sequence, so aligned position-for-position): of 2,435 dwords submitted per token across those
// 26 batches, only 65 (2.7%, 260 bytes) actually differ from the previous token - one dword in the big compute
// chain, three per small copy-engine record (mostly KV-cache write offset/address pairs). The changed values are
// not just small, they are PREDICTABLE FIXED-STRIDE COUNTERS: the offset field advances by exactly the same
// delta every layer within a token (e.g. +2635), and the paired address dwords advance by exactly the same
// delta every layer too (+4276224 bytes) - consistent with the ggml-cuda code-level finding that MoE expert
// selection is expressed as small on-device buffer *content* (mmid.cu's ids/ids_src1), never as changed launch
// addresses. This is well inside TINYNV_INLINE_MAX (a single call) and the default inline-pend budget with
// large margin - confirms extending the existing, already-hardware-proven TINYNV_INLINE_UPLOAD path to patch
// this delta in place is sufficient; no second GPFIFO channel is needed for this specific problem.
// Record format: [u32 queue: 0=compute 1=copy][u32 n dwords][n*4 bytes], appended, never truncated.
static void dump_cmd_maybe(const tinynv_queue_t *q, const tinynv_exec_t *ex, const tinynv_cmdbuf_t *c) {
  static int checked = 0;
  static FILE *f = NULL;
  if (!checked) {
    checked = 1;
    const char *path = getenv("TINYNV_DUMP_CMD");
    if (path && *path) f = fopen(path, "ab");
  }
  if (!f) return;
  uint32_t hdr[2] = {(uint32_t)(q == &ex->g->gsp.copy_q), c->n};
  fwrite(hdr, sizeof(hdr), 1, f);
  fwrite(c->words, 4, c->n, f);
  fflush(f);
}

static int run(tinynv_exec_t *ex, tinynv_queue_t *q, tinynv_cmdbuf_t *c, uint64_t cmdbuf_va) {
  dump_cmd_maybe(q, ex, c);
  // The timeline only moves once the work is actually on the ring. Advancing first and then failing to submit would
  // leave a value nothing will ever release, and every wait after it - including the arena's - would sit out its whole
  // timeout before reporting a fault that happened somewhere else entirely.
  uint64_t at = ex->reserved + 1;
  // The copy engine releases the timeline itself; the compute engine is released by the host methods. See
  // tinynv_cmd_copy_release: a host-method release on a copy channel can retire before the engine has drained, which
  // asynchronously means a kernel starts reading a buffer the copy is still writing. The copy engine's own release is
  // one word, so the timeline has to stay inside 32 bits for the two to agree on the same eight bytes. That is 4
  // billion batches, which nothing will reach, and refusing is still better than wrapping where nobody is looking.
  if (at > 0xffffffffull) return tinynv_fail("the timeline has run past 32 bits (%llu batches)", (unsigned long long)at);
  int is_copy = q == &ex->g->gsp.copy_q;
  uint64_t slot = ex->sem.va + SEM_SLOT(is_copy);
  // The head timestamp used to be emitted here and was wrong twice over: appended after the caller's launches, and
  // carrying RELEASE_WFI, so it recorded when the batch FINISHED. batch_begin emits it now, at the head, without the
  // wait. See tinynv_cmd_timestamp.
  // Under the profile the compute release carries a clock, for the same reason the chain tail's does: everything that
  // advances this timeline has to leave a timestamp beside it, or a batch that ends here rather than in a chain leaves
  // the previous chain's clock sitting at +8 and the next refill measures against it. Free - this release already
  // waits for idle, and a four word report costs eight bytes of a 128 byte stride.
  if (ex->profile && is_copy && tinynv_cmd_copy_timestamp(c, ex->sem.va + SEM_TS_COPY, at)) return -1;
  if (is_copy ? tinynv_cmd_copy_release(c, slot, at)
              : (ex->profile ? tinynv_cmd_release_clocked(c, slot, at) : tinynv_cmd_release(c, slot, at)))
    return -1;
  ex->reserved = at;
  return submit_batch(ex, q, c, cmdbuf_va, at, 0);
}

// Hand over the chain built so far: one launch method for the head, and the descriptors carry the rest. Each descriptor
// releases its own timeline value when its kernel finishes, so there is no release in the command stream and no
// wait-for-idle between kernels - which is the whole point, since that wait is what stopped one kernel overlapping the
// next. The last value in the chain is what the timeline reaches.
// What the chain has to be true of before it is handed over. This is not defensive tidiness: the way this goes wrong is
// a kernel running beside the one it depends on, which produces a wrong number rather than a fault, intermittently and
// only under load. A chain that does not link, or a timeline value released twice or never, is exactly that bug one
// step earlier, where it can still be turned into a refusal. Cheap - a few reads per flush, against a kernel launch.
static int chain_is_sound(tinynv_exec_t *ex, int n) {
  uint64_t base = ex->reserved - (uint64_t)n;   // the chain releases base+1 .. base+n, in order
  for (int i = 0; i < n; i++) {
    const char *why = NULL;
    // Tail-only: every link but the last must be silent, and the last carries the whole chain's value. Checking the
    // shape that was asked for rather than either shape is the point - a check that accepts both cannot tell a chain
    // built wrong from a chain built the other way.
    int releases = ex->no_chain_deps ? 0 : (!ex->tail_release || i == n - 1);
    uint64_t want = ex->tail_release ? ex->reserved : base + 1 + (uint64_t)i;
    uint64_t next = ex->no_chain_deps ? 0 : (i + 1 < n ? ex->chain[i + 1].va : 0);
    if (tinynv_qmd_link_check(&ex->chain[i].qmd, releases, want, next, &why))
      return tinynv_fail("launch %d of %d in the chain is wrong: %s", i, n, why ? why : "unspecified");
  }
  return 0;
}

// TINYNV_DELTA_DELIVERY's per-launch diffs, once they clear the per-launch TINYNV_INLINE_MAX check, are also
// capped as a whole flush against this. Deliberately not TINYNV_INLINE_MAX itself: that number prices ONE small
// copy against the copy engine moving it instead; the real alternative on a miss here is the FULL envelope
// fallback, ~800us of copy-engine time for the whole flush's dirty span (libtinynv-design.md SS4f, "writing a
// full chain's ~192 KB across the link instead is ~800 us"). Processor writes cost ~4.2us/KB (measured: 1.5KB at
// a launch's descriptor is 6.3us of ~7 - libtinynv-design.md SS4d), so this many KB of inline patches costs a few
// tens of us against an ~800us fallback either way - the aggregate can be generous without the trade flipping.
// 16 KB is a first estimate, not a swept default: unlike TINYNV_INLINE_PEND's 16, it has not been measured
// against smaller or larger values yet. Treat it as provisional until it is.
#define TINYNV_DELTA_AGGREGATE_MAX (64u * 1024u)

int tinynv_exec_flush(tinynv_exec_t *ex) {
  if (!ex->nchain) return 0;
  PROF_START(ex);
  int n = ex->nchain;
  ex->nchain = 0;   // before anything that could come back through batch_begin
  // The tail could not be known while the chain was growing, so its release is attached here, once it is. It carries
  // the last value the chain reserved, which is the only value anything waits for: every wait in this file is for what
  // submit_batch was told, and that is this number.
  // Under the profile the tail's release carries the gpu clock as well as the value: a four word report instead of
  // two, payload at +0 where the timeline has always been read and the clock at +8. Nothing else changes - same slot,
  // same address, same moment, eight more bytes into a 128 byte stride.
  if (ex->tail_release && !ex->no_chain_deps &&
      tinynv_qmd_release(&ex->chain[n - 1].qmd, ex->sem.va + SEM_SLOT(0), ex->reserved, ex->profile ? 1 : 0) < 0)
    return tinynv_fail("the last descriptor has no free release slot");
  // Why the clock rides on the timeline's own release rather than a second one of its own.
  //
  // RETRACTED, and the retraction matters more than the arrangement. This comment used to say "the second release slot
  // simply does not fire for us after the first descriptor, and I do not know why" - stated as an established property
  // of the hardware, on the evidence that changing the writer changed nothing in A's raw pairs. That evidence was
  // real and the conclusion was false. Neither writer was running: the flush-time release this stamped is behind
  // TINYNV_TAIL_RELEASE, which is OFF unless asked for, so both "fixes" edited code that a decode never executes. The
  // one value that appeared came from a setup batch on another path, which is why the stuck tail was always the FIRST
  // batch's end.
  //
  // I left that false mechanism in the code for an hour after correcting the behaviour, which is the worse half: a
  // wrong fix gets found, a wrong explanation sitting in a comment gets believed. There is no evidence either way
  // about whether a descriptor's second release slot works, and anyone who needs it should find out rather than read
  // it here.
  //
  // What is true is the arrangement below: the clock rides on the release that advances the timeline, which
  // demonstrably fires on every chain because the host is already waiting on it. Same slot, same moment, and it
  // cannot be stale relative to the value that satisfied the wait, because it IS that value's report.
  if (chain_is_sound(ex, n)) return -1;

  // The descriptors were edited in place while the chain grew - each one's pointer to the next is written when the next
  // arrives - so they are copied out only now that no more editing can happen.
  for (int i = 0; i < n; i++) memcpy(ex->chain[i].host, ex->chain[i].qmd.b, TINYNV_QMD_BYTES);

  // What this chain's own descriptors release, captured before anything else can take a number. The delivery below
  // submits a batch of its own, and a submitted batch takes the next timeline value - so after it, ex->reserved is the
  // COPY's value and no longer the chain's. Submitting the chain with that number is what stalled every delivered
  // chain on hardware: the compute slot was declared to reach a value its descriptors never release, so the first wait
  // on it sat out its whole timeout while the engine, correctly, had nothing left to do.
  const uint64_t chain_upto = ex->reserved;
  // From here until the chain is submitted, NOTHING may take a timeline value on the compute queue. The chain's
  // descriptors release base+1..chain_upto, and the chain batch is handed over LAST - so a compute batch submitted in
  // this window releases a HIGHER value first and the timeline goes backwards by the length of the chain.
  //
  // That is the bug that made inline uploads corrupt the dense 27B: held bytes were pushed out here, inside the
  // delivery copy's batch_begin, as a compute batch of their own. A's logs name it exactly - "12952 after 13070" is a
  // drop of 118, which is a chain - and it was deterministic per build because the arena and the load sequence are.
  ex->in_chain_flush = 1;

  // Hand the descriptors over by copy engine, before the batch that will read them is begun - so that batch_begin
  // below sees the copy as the last queue used and emits the acquire that orders the two. The copy in turn acquires on
  // the previous compute chain, which is what stops it overwriting descriptors a running chain is still reading.
  // A SMALL descriptor span rides in the compute batch itself, so there is no copy-engine batch and therefore no
  // cross-queue acquire and no channel switch at all.
  //
  // This is the thing the whole boundary investigation arrives at. The copy engine delivering descriptors is the last
  // copy-engine work at a token boundary, and the switch between the two channels costs ~700 us on about half the
  // tokens. Writing a full chain's ~192 KB across the link instead is ~800 us, worse; inlining a full chain does not
  // fit in a pushbuffer. What fits is a SHORT first chain - and the gap-depth curve prices one at ~22 us of chain
  // overhead (gap(d) = 1.0 + 22/d across five depths), so two launches cost the token ~20 us to take the handoff off
  // its critical path.
  //
  // Correctness is the same argument as any other held span: emitted after the acquire that orders the batch, before
  // the launch method that reads them, flushed by the same FLUSH_ONLY the caller's tensors get.
  //
  // TINYNV_DELTA_DELIVERY: the span about to be delivered, captured before any branch below clears it, so
  // last_delivered can be brought up to date at the end regardless of which path actually did the delivering.
  uint64_t delta_lo = ex->region[AR_DESC].dirty_lo, delta_hi = ex->region[AR_DESC].dirty_hi;
  int delta_handled = 0;
  // Per-launch diff bounds and how much of it there is, hoisted out of the block below because the actual write
  // happens after batch_begin() further down - see the comment there for why it moved.
  uint64_t plo[TINYNV_EXEC_CHAIN_MAX], phi[TINYNV_EXEC_CHAIN_MAX];
  unsigned delta_pend_n = 0;
  uint32_t delta_pend_bytes = 0;
  if (ex->delta_delivery && ex->region[AR_DESC].last_delivered && ex->arena_dma &&
      ex->wraps[AR_DESC] && delta_hi > delta_lo) {
    // Trusted only past the region's first lap (wraps[AR_DESC] set): before that, last_delivered has never been
    // compared against a real prior delivery at this address, so there is nothing yet to diff against honestly.
    //
    // Per launch, not one envelope over the whole flush. A flush holds up to chain_max launches before delivering
    // (n here), and one min-max diff over that much content reads as "almost 100% different" the instant even one
    // byte differs near each end - measured on hardware 2026-09-18 (TINYNV_DELTA_VERBOSE), which is why the single-
    // envelope version of this that shipped in a4f9010 was a correctness-safe no-op in practice. Each launch's own
    // span (its QMD slot plus its own cbuf0, chain[i].len) is where the real per-token delta actually lives, so
    // diffing there finds it at the granularity it occurs at.
    //
    // All-or-nothing per flush, checked before anything is pushed: either every launch's own diff fits inline, or
    // none of them are pushed here and the flush falls through to the existing full-span paths below untouched.
    // Committing some launches' diffs here and then letting a fallback path resend the same bytes as part of a
    // wider copy would deliver a byte twice under two different values of "what's there now" - a correctness bug,
    // not a slower correct answer - which is why the check pass below commits nothing until it has looked at
    // every launch in the chain.
    uint8_t *sh = ex->region[AR_DESC].shadow, *ld = ex->region[AR_DESC].last_delivered;
    unsigned need_n = 0;
    uint32_t need_bytes = 0;
    int all_fit = 1;
    for (int i = 0; i < n; i++) {
      uint64_t lo = ex->chain[i].va - ex->region[AR_DESC].mem.va, hi = lo + ex->chain[i].len;
      uint64_t dlo = hi, dhi = lo;
      uint64_t genuine = 0;   // DIAGNOSTIC ONLY, TINYNV_DELTA_VERBOSE: bytes that actually differ within [dlo,dhi)
      for (uint64_t j = lo; j < hi; j++)
        if (sh[j] != ld[j]) { if (j < dlo) dlo = j; dhi = j + 1; genuine++; }
      if (dhi > dlo && getenv("TINYNV_DELTA_VERBOSE") && ex->delta_full_n < 3)
        fprintf(stderr, "libtinynv: delta envelope check: launch %d span-width %llu genuine-diff-bytes %llu (%.0f%% of span)\n",
                i, (unsigned long long)(dhi - dlo), (unsigned long long)genuine, 100.0 * (double)genuine / (double)(dhi - dlo));
      if (dhi > dlo) {
        // tinynv_cmd_inline_upload requires a whole number of dwords at a 4-byte-aligned destination (submit.c) -
        // the byte-precise diff found above has neither guarantee, so it is widened to the nearest dword on each
        // side before anything downstream sees it. Safe to widen: each launch's own [lo,hi) is itself allocated on
        // a 256-byte boundary with a 256-byte-rounded length (tinynv_exec_run's cbuf0_bytes rounding), so rounding
        // dlo down / dhi up can never cross into a neighbouring launch's span. This crashed on hardware once
        // (2026-09-19, "an inline upload is a whole number of dwords, not N bytes") before this rounding existed -
        // the single-envelope version before it had the identical latent bug, just never taken often enough to hit it.
        dlo &= ~(uint64_t)3;
        dhi = (dhi + 3) & ~(uint64_t)3;
      }
      plo[i] = dlo;
      phi[i] = dhi;
      if (dhi <= dlo) continue;   // this launch is byte-identical to its last delivery - nothing to send for it
      uint32_t dlen = (uint32_t)(dhi - dlo);
      // Capped against its own budget (TINYNV_DELTA_AGGREGATE_MAX, defined above tinynv_exec_flush), not the shared
      // inline_pend one: these patches are written straight into the chain's own delivery batch after batch_begin()
      // (see below), not queued through inline_pend. Measured on hardware 2026-09-19: sharing inline_pend's budget
      // with TINYNV_INLINE_UPLOAD was why this fired on only 5 of 1223 flushes in a 96-token decode, even with that
      // shared budget maxed at its ceiling (TINYNV_INLINE_PEND=128) - see the constant's own comment for why the
      // aggregate cap here is deliberately not TINYNV_INLINE_MAX itself.
      if (dlen > TINYNV_INLINE_MAX || need_bytes + dlen > TINYNV_DELTA_AGGREGATE_MAX) {
        all_fit = 0;
        if (getenv("TINYNV_DELTA_VERBOSE") && ex->delta_full_n <= 20)
          fprintf(stderr, "libtinynv: per-launch delta miss #%llu: launch %d of %d, span [%llu,%llu) len %llu\n",
                  (unsigned long long)ex->delta_full_n + 1, i, n, (unsigned long long)dlo, (unsigned long long)dhi,
                  (unsigned long long)dlen);
        break;
      }
      need_n++;
      need_bytes += dlen;
    }
    if (all_fit) {
      // Nothing above touched arena state - safe to commit now that every launch has been checked. The actual
      // tinynv_cmd_inline_upload calls happen after batch_begin() below, once the chain's own command buffer
      // exists to write them into; plo[]/phi[]/delta_pend_n (set here) are what that later code reads.
      delta_pend_n = need_n;
      delta_pend_bytes = need_bytes;
      // Ownership moves over the whole originally-dirty span regardless of how many launches actually had bytes to
      // send, same as the full-span paths below: every launch in the chain read consistently as of this delivery,
      // not just the ones that changed.
      tinynv_arena_mark(&ex->region[AR_DESC], delta_lo, delta_hi, chain_upto);
      ex->region[AR_DESC].dirty_lo = ex->region[AR_DESC].dirty_hi = 0;
      if (need_n) {
        ex->hybrid_n++;
        ex->delta_n++;
        ex->delta_bytes += need_bytes;
      } else {
        // Every launch in the chain matched its last delivery - a decode step can repeat a cache-hit prefix
        // exactly. Ownership still has to move; no bytes need to.
        ex->delta_skip_n++;
      }
      delta_handled = 1;
    } else {
      // At least one launch's own diff did not fit inline (an unexpectedly large change, or a chain shape that
      // does not match what was here last lap) - fall through to the existing full-span paths below, unchanged.
      // Conservative on purpose: this is never worse than today's behaviour, only sometimes not as good as it
      // could be.
      ex->delta_full_n++;
    }
  }
  if (!delta_handled && ex->hybrid_delivery && ex->arena_dma &&
      ex->region[AR_DESC].dirty_hi > ex->region[AR_DESC].dirty_lo &&
      ex->region[AR_DESC].dirty_hi - ex->region[AR_DESC].dirty_lo <= TINYNV_INLINE_MAX &&
      inline_pend_fits(ex, (uint32_t)(ex->region[AR_DESC].dirty_hi - ex->region[AR_DESC].dirty_lo))) {
    uint64_t lo = ex->region[AR_DESC].dirty_lo, len = ex->region[AR_DESC].dirty_hi - lo;
    inline_pend_push(ex, ex->region[AR_DESC].mem.va + lo, ex->region[AR_DESC].shadow + lo, (uint32_t)len);
    tinynv_arena_mark(&ex->region[AR_DESC], lo, lo + len, chain_upto);
    ex->region[AR_DESC].dirty_lo = ex->region[AR_DESC].dirty_hi = 0;
    ex->hybrid_n++;
  }
  if (!delta_handled && ex->arena_dma && ex->region[AR_DESC].dirty_hi > ex->region[AR_DESC].dirty_lo) {
    uint64_t lo = ex->region[AR_DESC].dirty_lo, len = ex->region[AR_DESC].dirty_hi - lo;
    // One sequential write into memory the engine can read, in place of the same bytes going out as scattered
    // four-byte register writes. Same volume, and this way the processor writes it once, in order.
    memcpy((uint8_t *)ex->mirror.dma.va + lo, ex->region[AR_DESC].shadow + lo, (size_t)len);
    // Against the chain that reads these descriptors rather than the copy that delivers them: the chain finishing
    // implies the copy did, since it acquires on it, and the bytes are in use until the chain is done with them.
    tinynv_arena_mark(&ex->region[AR_DESC], lo, lo + len, chain_upto);
    ex->region[AR_DESC].dirty_lo = ex->region[AR_DESC].dirty_hi = 0;
    // Written but not announced. The compute batch built just below is the next thing out, and its submit announces
    // both - one read before two doorbells instead of one read each. That read is a message to another process here, so
    // it costs a full round trip and drains whatever writes are queued ahead of it, which is why halving the count is
    // worth more than the microseconds suggest. Every flush in delivery mode was paying for two.
    ex->defer_ring = 1;
    int rc = tinynv_exec_copy(ex, ex->region[AR_DESC].mem.va + lo, ex->mirror.va + lo, len);
    ex->defer_ring = 0;
    if (rc) return -1;
  }
  // Whichever path just ran (delta patch, full inline, full copy, or none because there was nothing dirty) has
  // now made video memory match shadow across the whole originally-dirty span: the delta path patched only what
  // differed because the rest was already correct there, and the two full-span paths always send the whole span.
  // Bringing last_delivered up to date here, once, regardless of which branch fired, is what makes the NEXT lap's
  // diff honest - a bug in this line, and only this line, would be the one way this feature could ever go from
  // "sometimes not as good as it could be" to "wrong": if the recorded mirror ever drifts from the bytes actually
  // resident, a future skip could omit an update that had to reach video memory.
  if (ex->region[AR_DESC].last_delivered && delta_hi > delta_lo)
    memcpy(ex->region[AR_DESC].last_delivered + delta_lo, ex->region[AR_DESC].shadow + delta_lo, delta_hi - delta_lo);

  uint64_t va;
  tinynv_cmdbuf_t c;
  // Four dwords per launch method, and in the unlinked mode every descriptor needs its own rather than the head
  // alone carrying the rest.
  // Eight more under the profile for the tail stamp appended below.
  // Room for this flush's own delta patches, on top of batch_begin()'s own reservation for whatever TINYNV_INLINE_UPLOAD
  // has pending: TINYNV_INLINE_DWORDS(len) per call, 8 + len/4, summed - exact since every plo[]/phi[] span was
  // widened to a dword multiple above, so delta_pend_bytes/4 has no remainder to lose.
  uint32_t want_dwords = (ex->no_chain_deps ? 48 + 4 * (uint32_t)n : 48) + (ex->profile ? 8u : 0u) +
                          (delta_pend_n ? 8u * delta_pend_n + delta_pend_bytes / 4u : 0u);
  if (batch_begin(ex, &ex->g->gsp.compute_q, want_dwords, &c, &va)) return -1;
  // Written here, not queued through inline_pend: batch_begin() has already reserved the room want_dwords asked
  // for and drained whatever TINYNV_INLINE_UPLOAD had pending into this same c, in that order, ahead of this -
  // same "after the acquire, before anything the caller appends" placement inline_pend's own drain relies on
  // (see batch_begin), just written directly since these were never held anywhere else to begin with.
  if (delta_pend_n) {
    uint8_t *sh = ex->region[AR_DESC].shadow;
    for (int i = 0; i < n; i++)
      if (phi[i] > plo[i] &&
          tinynv_cmd_inline_upload(&c, ex->region[AR_DESC].mem.va + plo[i], sh + plo[i], (uint32_t)(phi[i] - plo[i])))
        return -1;
  }

  // Wait for the chain before this one. A chain orders the kernels inside it - each descriptor schedules the next - but
  // nothing ordered one chain against the next, because the thing that used to do that was the wait-for-idle release
  // this whole change removed. So two chains ran at once: their descriptors wrote the same slot, a later chain's tail
  // landed before an earlier chain's, and the timeline went 2041 then 1885. That is not a reporting artefact, it is
  // dependent kernels from different chains running side by side, which is the bug the chaining was meant to avoid,
  // one level up from where I fixed it.
  //
  // One acquire per chain, not per launch: thirty-two kernels still pipeline inside it, and only the seam between
  // chains is serialised. That is the cost of correctness here and it is a thirty-second of what the old release cost.
  if (!ex->sync && ex->q_last[0]) {
    if (tinynv_cmd_wait(&c, ex->sem.va + SEM_SLOT(0), ex->q_last[0])) return -1;
    ex->pending_acquire = ex->q_last[0];
  }
  // the caches have to forget what the last kernel left in them, or this one reads its predecessor's constants
  if (tinynv_cmd_memory_barrier(&c)) return -1;
  if (ex->no_chain_deps) {
    // Every descriptor launched from the command stream, nothing scheduling anything, and the batch ending in a
    // host-method release that waits for the engine to go idle - which is what makes the wait afterwards mean
    // "all of them finished" when no descriptor can be trusted to be last.
    for (int i = 0; i < n; i++) if (tinynv_cmd_launch(&c, ex->chain[i].va)) return -1;
    if (tinynv_cmd_release(&c, ex->sem.va + SEM_SLOT(0), chain_upto)) return -1;
  } else if (tinynv_cmd_launch(&c, ex->chain[0].va)) return -1;
  // When this batch's work actually ends, and it has to be a host release that waits for idle rather than the
  // timeline's own report.
  //
  // The report was wrong and the way it was wrong is the lesson. Every descriptor releases its value to one address
  // and each overwrites the last, so the report should hold the final descriptor's payload and clock. A saw
  // "payload N-2" on every refill and I dismissed the two-release lag as microseconds against a 1.8 ms gap. That
  // assumed descriptors are interchangeable. They are not: the LAST descriptor of a decode token is the lm_head -
  // 248,320 rows of Q4_K against the hidden state, ~700 MB of weights, ~400 us - which is the biggest kernel in the
  // token. Stamping two releases early put the lm_head's whole duration inside every "dry engine" figure of the day,
  // and it fits both models: the mixture's lm_head is ~170 us and its number is ~150-200 us lower.
  //
  // So the tail comes from a release that cannot be ambiguous about which work it followed. WFI, its own address,
  // nothing else writing there. The ordering it imposes already exists - every chain opens with an acquire on the
  // previous chain's timeline value - and it is profile-only regardless.
  // Not on the driver's own work. A download served by a caller's kernel is still a launch, and it would otherwise
  // stamp "the token's last kernel finished" partway through the boundary it is part of.
  if (ex->profile && !ex->internal_work && tinynv_cmd_release_clocked(&c, ex->sem.va + SEM_TS_TAIL, chain_upto))
    return -1;
  ex->flush_lo = chain_upto - (uint64_t)n + 1;
  ex->flush_n = n;
  ex->pending_head_va = ex->chain[0].va;
  ex->pending_head_sum = qmd_sum(ex->chain[0].host);
  ex->in_chain_flush = 0;
  if (submit_batch(ex, &ex->g->gsp.compute_q, &c, va, chain_upto, n)) return -1;
  // Once per chain rather than once per launch: see tinynv_exec_run. This reports on work already handed over, which
  // is what it reported on before too - a launch that has not been submitted cannot have faulted.
  { PROF_START(ex); int bad = tinynv_gsp_poll(ex->g); PROF_END(ex, PROF_POLL);
    if (bad) return tinynv_fail("gsp-rm reported a fault, logged above"); }
  PROF_END(ex, PROF_FLUSH);
  return 0;
}

// Waiting for everything means everything, including launches that are built and not yet handed over: those have
// timeline values nothing has been told to release, so waiting without flushing first waits out the whole timeout.
int tinynv_exec_idle(tinynv_exec_t *ex) {
  if (tinynv_exec_flush(ex)) return -1;
  if (tinynv_exec_wait(ex, ex->timeline, 30.0)) return -1;
  ex->needs_wait = 0;
  return 0;
}

// Whether a synchronise has to wait for the engine, or only hand over what has been built.
//
// The driver answers this rather than the caller counting alongside it. A caller mirroring what the driver does can
// drift, and the failure mode of drift is a synchronise that returns while a kernel is still running - a wrong answer
// with no symptom at the place it was caused. Everything that requires completion sets the flag at the point it is
// submitted; only an inline upload does not, because its bytes travel in the command stream.
int tinynv_exec_needs_wait(const tinynv_exec_t *ex) { return ex->needs_wait; }

int tinynv_exec_copy(tinynv_exec_t *ex, uint64_t dst_va, uint64_t src_va, uint64_t bytes) {
  if (!bytes) return 0;
  uint64_t va;
  tinynv_cmdbuf_t c;
  if (batch_begin(ex, &ex->g->gsp.copy_q, ex->profile ? 78 : 72, &c, &va)) return -1;
  // Before the transfer methods and after batch_begin's acquire, so it marks the moment this batch's own work starts.
  // A's question, and it is the one split nothing so far can make: the decode's logits copy takes ~600 us from the end
  // of compute to its completion stamp, against <=245 us for the same 993 KB transfer issued cold in d2h_bench. If
  // start-tail is most of the difference the copy engine is late to begin; if end-start is, the transfer itself runs
  // slower here than on the bench, which would point at where the staging buffer lives rather than at scheduling.
  if (ex->profile && tinynv_cmd_copy_timestamp(&c, ex->sem.va + SEM_TS_CSTART, ex->reserved + 1)) return -1;
  if (tinynv_cmd_copy(&c, dst_va, src_va, bytes)) return -1;
  return run(ex, &ex->g->gsp.copy_q, &c, va);
}

// Whether a small host-to-device copy rides in the pushbuffer instead of going to the copy engine.
//
// Session A timed the host side of a decode's token boundary call by call: seven copies - 20480, 4, 16, 8, 8, 512 and
// 4 bytes - cost ~370 us together, about 50 us each including the two synchronisations each one carries. That is 2.3%
// of a dense token and 4.8% of a mixture's, spent moving 21 KB, and six of the seven are under 512 bytes.
//
// ON by default since 2026-09-15, after it measured clean twice and +3% on both models - the largest validated win
// since copy-engine descriptor delivery. TINYNV_INLINE_UPLOAD=0 turns it off.
//
// It shipped off for a day because it changes the DATA path rather than an instrument: a write that did not land
// would have kernels reading stale memory, which op-verify and a byte-identical text catch - but "would catch" is a
// reason to measure behind a knob, not a reason to ship without one. Measured: op-verify 450/450 at three chain
// depths with 2,538 copies riding per run, text byte-identical, 27B 62.49 -> 64.49 twice and MoE 127.4 -> 131.4.
//
// The win is not the routing, which changed nothing on its own. It is that the upload stopped taking a batch: the
// bytes are held and emitted into the next compute batch, so six submits a token became zero. Session A established
// that the cost was getting a batch onto the ring rather than waiting for it, by removing the wait and seeing nothing
// move.
int tinynv_exec_inline_upload_mode(const char *e) { return !e || !*e || *e != '0'; }

// How many small uploads the driver will hold before one has to be pushed out. 16 is what a single-slot decode needs -
// seven a token with room to spare - and it is almost certainly too small for eight slots, where a step issues about
// seven per slot. TINYNV_INLINE_PEND=<n> sweeps it up to the fixed ceiling, so the default comes off a measured curve
// rather than out of the arithmetic, which is the same arithmetic that would have had me sizing a list against a
// number that turned out to be the wrong field.
unsigned tinynv_exec_pend_entries(const char *e) {
  if (!e || !*e) return 16;
  int n = atoi(e);
  if (n < 1) n = 1;
  if (n > TINYNV_INLINE_PEND_CAP_N) n = TINYNV_INLINE_PEND_CAP_N;
  return (unsigned)n;
}

// How many launches the first chain after a standstill carries, so its descriptors fit in a pushbuffer and the batch
// that restarts the engine needs no copy engine. 0 means the mode is off and every chain is full depth.
//
// Off by default. It is the fix the whole token-boundary investigation arrives at, and it is also the least directly
// evidenced: the isolating experiment was confounded (writing a full chain's descriptors across the link costs about
// as much as the switch it removes, so the two arms differ by more than the copy engine), and this is the test that
// separates them rather than a conclusion drawn from one.
unsigned tinynv_exec_short_chain(const char *e) {
  if (!e || !*e) return 0;
  int n = atoi(e);
  if (n < 0) n = 0;
  if (n > TINYNV_EXEC_CHAIN_MAX) n = TINYNV_EXEC_CHAIN_MAX;
  return (unsigned)n;
}

int tinynv_exec_upload(tinynv_exec_t *ex, uint64_t dst_va, const void *src, size_t n) {
  const uint8_t *p = src;
  // Small, whole dwords, aligned: everything else goes to the engine built for moving bytes. The 20480-byte copy of
  // the seven stays there too - past a few KB the pushbuffer itself has to reach the card, so inlining stops paying.
  if (ex->inline_upload && n && n <= TINYNV_INLINE_MAX && !(n & 3u) && !(dst_va & 3u)) {
    // BUFFERED, not submitted, and that is the whole of the fix. This took a batch, a doorbell and an idle of its own
    // - and worse, batch_begin FLUSHES the pending chain, so every tiny copy forced whatever launches were queued out
    // early. Six a token. Session A measured the round trip at ~50 us a copy and then showed, by removing the wait
    // after it and seeing nothing move, that the cost is getting the batch onto the ring rather than waiting for it.
    //
    // The bytes are copied out of the caller's buffer here, so the caller may reuse it the moment this returns - which
    // is the only guarantee a synchronous upload owes. Nothing has to reach the card until something else goes, and
    // batch_begin emits these into the next compute batch before anything in it can observe them.
    if (ex->inline_pend_n == ex->inline_pend_max || ex->inline_pend_bytes + n > ex->inline_pend_bytes_max) {
      // Full. Prefer to let the held bytes ride out with work that is ALREADY pending rather than building a batch
      // for them - which is the whole point of holding them, and the version of this branch that went to the card
      // first got it wrong: it called batch_begin, whose flush hands the pending chain over, and the chain's own
      // batch_begin drained the held list into THAT batch. The list was then empty and the batch built here carried
      // nothing but an acquire and a release. A wasted submit, on the path whose entire purpose is to remove submits,
      // and it only shows up when a chain is pending - which at one slot it never is and under batching it always is.
      if (tinynv_exec_flush(ex)) return -1;
      if (ex->inline_pend_n) {
        // Nothing was pending to carry them, so they do need a batch. This is the original case: a caller uploading
        // far more than a step's worth without ever launching anything.
        uint64_t va;
        tinynv_cmdbuf_t c;
        if (batch_begin(ex, &ex->g->gsp.compute_q, 40, &c, &va)) return -1;
        ex->submitting_inline = 1;
        int rc = run(ex, &ex->g->gsp.compute_q, &c, va);
        ex->submitting_inline = 0;
        if (rc) return -1;
        ex->inline_forced_n++;
      } else {
        ex->inline_rode_flush_n++;
      }
    }
    inline_pend_push(ex, dst_va, src, (uint32_t)n);
    ex->inline_n++;
    ex->inline_bytes += n;
    return 0;   // no batch, no doorbell, no wait; and needs_wait is deliberately untouched
  }
  ex->upload_ce_n++;
  ex->needs_wait = 1;
  for (size_t off = 0; off < n; off += STAGE_BYTES) {
    size_t chunk = n - off < STAGE_BYTES ? n - off : STAGE_BYTES;
    memcpy(ex->stage.dma.va, p + off, chunk);
    if (tinynv_exec_copy(ex, dst_va + off, ex->stage.va, chunk)) return -1;
    if (tinynv_exec_idle(ex)) return -1;   // the next chunk overwrites what this copy is reading
  }
  return 0;
}

// The staging buffer, for the one caller that has to drive the device->stage step itself: a download served by a
// caller's kernel rather than by the copy engine. Kernels live in tinynv.c and the stage lives here; this is the
// smaller thing to expose.
// Bracket the driver's own work, so the profile does not mistake it for the caller's. See the tail stamp in
// tinynv_exec_flush.
void tinynv_exec_set_internal(tinynv_exec_t *ex, int on) { ex->internal_work = on; }

uint64_t tinynv_exec_stage_va(const tinynv_exec_t *ex) { return ex->stage.va; }
void *tinynv_exec_stage_host(const tinynv_exec_t *ex) { return ex->stage.dma.va; }
size_t tinynv_exec_stage_bytes(void) { return STAGE_BYTES; }

int tinynv_exec_download(tinynv_exec_t *ex, void *dst, uint64_t src_va, size_t n) {
  uint8_t *p = dst;
  for (size_t off = 0; off < n; off += STAGE_BYTES) {
    size_t chunk = n - off < STAGE_BYTES ? n - off : STAGE_BYTES;
    // Let the compute drain before the copy is issued, so the copy's acquire is already satisfied when the channel is
    // first looked at and it is never switched out waiting. See tinynv_exec_download_sync_first.
    if (ex->download_sync_first && tinynv_exec_idle(ex)) return -1;
    if (tinynv_exec_copy(ex, ex->stage.va, src_va + off, chunk)) return -1;
    if (tinynv_exec_idle(ex)) return -1;   // reading a result is one of the few places that has to wait
    // Recorded before the write, not after, so a fault inside this very memcpy still leaves the span behind.
    tinynv_note_host_write(p + off, chunk, "download: stage -> caller");
    memcpy(p + off, ex->stage.dma.va, chunk);
  }
  return 0;
}

// Zero a range of device memory, through the copy engine because that is the only thing that can reach it.
//
// `tinynv_mm_alloc_buffer`'s zero flag cannot do this: it writes through the window onto video memory and skips, in
// silence, anything past the end of it. On a card behind thunderbolt the window is 256 MB of 32 GB, so for almost every
// allocation that flag is a no-op that looks like a guarantee.
int tinynv_exec_zero(tinynv_exec_t *ex, uint64_t va, uint64_t bytes) {
  if (!bytes) return 0;
  uint64_t chunk = ex->stage.size < bytes ? ex->stage.size : bytes;
  memset(ex->stage.dma.va, 0, (size_t)chunk);
  for (uint64_t off = 0; off < bytes; off += chunk) {
    uint64_t n = bytes - off < chunk ? bytes - off : chunk;
    if (tinynv_exec_copy(ex, va + off, ex->stage.va, n)) return -1;
  }
  return tinynv_exec_idle(ex);   // the pattern lives in the staging buffer; nobody may refill it until this is done
}

int tinynv_exec_launch(tinynv_exec_t *ex, uint64_t qmd_va) {
  uint64_t va;
  tinynv_cmdbuf_t c;
  if (batch_begin(ex, &ex->g->gsp.compute_q, 40, &c, &va)) return -1;
  // the caches have to forget what the last kernel left in them, or this one reads its predecessor's constants
  if (tinynv_cmd_memory_barrier(&c)) return -1;
  if (tinynv_cmd_launch(&c, qmd_va)) return -1;
  return run(ex, &ex->g->gsp.compute_q, &c, va);
}

// Local memory is a property of the die, not of a kernel: every thread that could be resident at once needs its own, so
// it is sized for the whole chip and only grown when a kernel wants more per thread than the last one did.
int tinynv_exec_ensure_local_memory(tinynv_exec_t *ex, uint32_t per_thread) {
  if (per_thread <= ex->slm_per_thread) return 0;
  ex->slm_per_thread = (per_thread + 31) & ~31u;

  tinynv_gsp_t *gsp = &ex->g->gsp;
  tinynv_die_t die = {.num_gpcs = gsp->max_gpcs, .num_tpc_per_gpc = gsp->max_tpc_per_gpc,
                      .num_sm_per_tpc = gsp->max_sm_per_tpc, .max_warps_per_sm = gsp->max_warps_per_sm};
  uint64_t per_tpc = 0, total = tinynv_local_memory_size(&die, ex->slm_per_thread, &per_tpc);

  tinynv_vmap_t slm;
  if (tinynv_mm_alloc_buffer(&ex->g->mm, total, 0, 0, 0, 1, 0, &slm)) return -1;
  // the old range is about to stop being mapped, and a kernel already submitted is still using it as its stack
  if (tinynv_exec_idle(ex)) { tinynv_vmap_free(&ex->g->mm, &slm); return -1; }
  tinynv_vmap_free(&ex->g->mm, &ex->slm);
  ex->slm = slm;

  uint64_t va;
  tinynv_cmdbuf_t c;
  if (batch_begin(ex, &ex->g->gsp.compute_q, 40, &c, &va)) return -1;
  if (tinynv_cmd_local_memory(&c, ex->slm.va, per_tpc)) return -1;
  return run(ex, &ex->g->gsp.compute_q, &c, va);
}

// Putting a cubin where the GPU can run it: lay the sections out into one block, fix up the addresses inside it now
// that the block has one, and copy it across. The guard page after it is not decoration - the instruction prefetcher
// reads past the end of a kernel and faults on memory that is not mapped.
#define IMAGE_GUARD 0x1000

int tinynv_exec_load(tinynv_exec_t *ex, const tinynv_cubin_t *cb, const char *kernel, tinynv_exec_module_t *out) {
  memset(out, 0, sizeof(*out));
  if (tinynv_cubin_image(cb, kernel, 1, &out->layout)) return -1;
  uint64_t bytes = ((out->layout.len + 0xfff) & ~0xfffull) + IMAGE_GUARD;
  if (tinynv_mm_alloc_buffer(&ex->g->mm, bytes, 0, 0, 0, 1, 0, &out->mem)) { tinynv_image_free(&out->layout); return -1; }
  if (tinynv_image_relocate(&out->layout, out->mem.va) ||
      tinynv_exec_upload(ex, out->mem.va, out->layout.bytes, out->layout.len) ||
      // the tail, including the guard page: the prefetcher reads past the end of a kernel, and what it finds should be
      // zeros this driver put there rather than whatever the last tenant of that memory left
      tinynv_exec_zero(ex, out->mem.va + out->layout.len, bytes - out->layout.len)) {
    tinynv_exec_unload(&ex->g->mm, out);
    return -1;
  }
  return 0;
}

void tinynv_exec_unload(tinynv_mm_t *mm, tinynv_exec_module_t *m) {
  // no idle here: the caller owns the ordering, because it is the one that knows no kernel from this module is queued
  tinynv_vmap_free(mm, &m->mem);
  tinynv_image_free(&m->layout);
  memset(m, 0, sizeof(*m));
}

// The whole of a launch as the caller sees it, timed as one thing. Everything else the summary reports is subtracted
// from this, so if the parts do not add up the difference is real work nobody has named yet rather than a rounding.
int tinynv_exec_run(tinynv_exec_t *ex, tinynv_exec_module_t *m, const tinynv_kernel_desc_t *k, const uint32_t grid[3],
                    const uint32_t block[3], uint32_t dyn_smem, const void *params, size_t params_len) {
  int tinynv_exec_run_inner(tinynv_exec_t *, tinynv_exec_module_t *, const tinynv_kernel_desc_t *, const uint32_t[3],
                            const uint32_t[3], uint32_t, const void *, size_t);
  PROF_START(ex);
  int rc = tinynv_exec_run_inner(ex, m, k, grid, block, dyn_smem, params, params_len);
  PROF_END(ex, PROF_LAUNCH);
  return rc;
}

int tinynv_exec_run_inner(tinynv_exec_t *ex, tinynv_exec_module_t *m, const tinynv_kernel_desc_t *k,
                          const uint32_t grid[3], const uint32_t block[3], uint32_t dyn_smem, const void *params,
                          size_t params_len) {
  tinynv_gsp_t *gsp = &ex->g->gsp;

  // A kernel's stack, plus what the driver reserves below it. This is per thread; the die multiplies it out.
  if (tinynv_exec_ensure_local_memory(ex, k->min_stack + 0x240)) return -1;

  // The descriptor and constant buffer 0 share one allocation, the way the oracle lays it out: the descriptor in the
  // first slot and the buffer right after it, with the driver's parameters first and the kernel's at the offset the
  // cubin names. On Blackwell that offset is 0x380, which is exactly the 224 dwords of driver parameters, so the two
  // meet with no gap - a property of this architecture rather than a rule, since sm_86 puts it at 0x160 and the driver
  // parameters somewhere else entirely. Nothing here should be carried to another architecture without checking that.
  uint32_t cbuf0_bytes = (uint32_t)(k->param_base + params_len);
  if (cbuf0_bytes < TINYNV_QMD_CBUF0_MIN_DWORDS * 4) cbuf0_bytes = TINYNV_QMD_CBUF0_MIN_DWORDS * 4;
  cbuf0_bytes = (cbuf0_bytes + 0xff) & ~0xffu;                 // constant buffers start and end 256 aligned
  uint64_t slot = TINYNV_QMD_SLOT_BYTES;

  // Two reasons to hand the chain over before building into it: it is full, or the arena has no room left and is about
  // to start again from the bottom, which would give these descriptors' bytes to something else. Both are checked here
  // rather than inside the allocator, which cannot flush without re-entering itself.
  // The first chain after the engine has stood still is deliberately short, so its descriptors fit in a pushbuffer
  // and the batch that restarts the engine needs no copy engine. Every chain after it is full depth; the cost is one
  // chain's overhead (~22 us) once a token, and what it buys is the delivery handoff moving off the critical path.
  unsigned cap = (ex->hybrid_delivery && ex->after_stall) ? ex->short_chain : (unsigned)ex->chain_max;
  if (ex->nchain >= (int)cap && tinynv_exec_flush(ex)) return -1;
  if (!arena_fits(ex, AR_DESC, slot + cbuf0_bytes, 256) && tinynv_exec_flush(ex)) return -1;

  uint64_t qmd_va;
  uint8_t *host = arena(ex, AR_DESC, slot + cbuf0_bytes, 256, &qmd_va);  // the launch method carries the address shifted by eight
  if (!host) return -1;

  tinynv_qmd_t q;
  memset(&q, 0, sizeof(q));
  tinynv_qmd_program_t prog = {.regs = k->regs,
                               // What the kernel declared plus what the caller asked for at the call site. The descriptor
                               // holds one number, so the two have to be added here: a kernel given only its static
                               // share does not fail, it indexes past the end of what it was given.
                               .shmem = (uint32_t)((0x400 + k->static_smem + dyn_smem + 127) & ~127u),
                               .slm_per_thread = ex->slm_per_thread,
                               .prog_size = (uint32_t)k->text_size,
                               .sass_version = tinynv_sass_version(gsp->sm_version)};
  tinynv_qmd_launch_t l = {.program_addr = m->mem.va + m->layout.text_off};
  for (int i = 0; i < 3; i++) { l.grid[i] = grid[i]; l.block[i] = block[i]; }

  // Bank 0 is the one being built here, in this allocation; the rest are sections of the image, and a kernel that reads
  // a __device__ table finds its address in one of them.
  prog.constbuf_used[0] = 1;
  // The size bound is the cubin's section, not the allocation: nvcc sizes .nv.constant0 to exactly the driver's
  // parameters plus the kernel's, so it always covers the arguments written below, and the allocation is only larger
  // because constant buffers are placed on 256 byte boundaries. Binding the allocation's size instead would tell the
  // hardware a buffer is bigger than the compiler said it was.
  prog.constbuf_size[0] = m->layout.constbuf[0].used ? (uint32_t)m->layout.constbuf[0].size : cbuf0_bytes;
  l.constbuf_addr[0] = qmd_va + slot;
  l.constbuf_set[0] = 1;
  for (int i = 1; i < TINYNV_QMD_CONSTBUFS; i++) {
    if (!m->layout.constbuf[i].used) continue;
    prog.constbuf_used[i] = 1;
    prog.constbuf_size[i] = (uint32_t)m->layout.constbuf[i].size;
    l.constbuf_addr[i] = m->mem.va + m->layout.constbuf[i].off;
    l.constbuf_set[i] = 1;
  }

  int rc = tinynv_qmd_program(&q, &prog) || tinynv_qmd_launch(&q, &l);
  if (!rc && q.overlaps) rc = tinynv_fail("%u bits of the descriptor were claimed twice", q.overlaps);
  if (rc) return -1;

  // The kernel reports its own completion rather than the command stream reporting it. That is what removes the
  // wait-for-idle between launches: a release in the command stream has to wait for the engine to go quiet before it
  // can be trusted, and waiting for the engine to go quiet is exactly what stops this kernel overlapping the next.
  uint64_t at = ex->reserved + 1;
  if (at > 0xffffffffull) return tinynv_fail("the timeline has run past 32 bits (%llu batches)", (unsigned long long)at);
  // Under TINYNV_TAIL_RELEASE the release is attached at the flush instead, to the last descriptor only. The value is
  // still reserved here, so nothing about how work is numbered changes - only how much of that numbering the engine is
  // asked to report. What is being measured is whether that report is what a launch costs.
  // THIS is the release the default configuration uses, and it is the one the profile has to stamp. TINYNV_TAIL_RELEASE
  // is OFF unless asked for, so the flush-time release in tinynv_exec_flush - which I stamped first, and then stamped
  // again a different way - is not executed on any ordinary run. Both of those "fixes" changed code that does not run,
  // which is why the number they were meant to fix did not move either time. Every descriptor releasing a four word
  // report is not a cost: they already all release, they retire in order, and each overwrites the last, so +8 ends up
  // holding the clock of whichever kernel finished most recently. Which is the definition of the thing being measured.
  if (!ex->tail_release && !ex->no_chain_deps &&
      tinynv_qmd_release(&q, ex->sem.va + SEM_SLOT(0), at, ex->profile) < 0)
    return tinynv_fail("the descriptor has no free release slot");

  // Ordering. Nothing in the command stream separates one kernel from the next any more, so the order has to be in the
  // descriptors: the one before this points at it and schedules it when it finishes. Without this a launch and the
  // launch after it are both resident at once, which is right for independent kernels and silently wrong for the ones
  // that read what their predecessor wrote - which is almost all of them.
  if (!ex->no_chain_deps && ex->nchain &&
      tinynv_qmd_chain(&ex->chain[ex->nchain - 1].qmd, qmd_va, ex->chain_prefetch)) return -1;

  memset(host + TINYNV_QMD_BYTES, 0, slot - TINYNV_QMD_BYTES);
  tinynv_qmd_cbuf0((uint32_t *)(host + slot), cbuf0_bytes / 4, TINYNV_SHARED_WINDOW, TINYNV_LOCAL_WINDOW,
                   l.grid, l.block);
  if (params_len) memcpy(host + slot + k->param_base, params, params_len);

  // Held, not handed over: the next launch still has to be able to write its address into this descriptor. The bytes
  // are copied into the arena by flush(), once no more editing can happen.
  ex->chain[ex->nchain].qmd = q;
  ex->chain[ex->nchain].host = host;
  ex->chain[ex->nchain].va = qmd_va;
  ex->chain[ex->nchain].len = (uint32_t)(slot + cbuf0_bytes);
  ex->nchain++;
  ex->reserved = at;

  // Asking the firmware whether anything faulted happens at the flush, not here. It is two reads across the link every
  // time it is asked - the status queue's read and write pointers - and across a link where a round trip is a
  // microsecond or two, doing that once per launch was a third of the entire per-launch cost to ask a question whose
  // answer almost always is "nothing". Once per chain asks it thirty-two times less often and answers the same
  // question: a kernel that faults on some of its blocks still releases its semaphore, so the only place a fault is
  // ever mentioned is the status queue, and reading it a little later still reads it.
  return ex->sync ? tinynv_exec_flush(ex) : 0;
}

// Chain depth, asked for rather than compiled in. Sweeping it is how the two failures at 32 and at 2 were told apart,
// and doing that by editing a header means every measurement comes from a build whose id says "dirty" - which is
// exactly the provenance you want when a result is surprising, and exactly what you cannot have if the knob is a
// constant. The array is still sized by the compile-time maximum; this only lowers it.
// Taken as an argument rather than read here, for the same reason the submission mode is: the resolved value is what
// a run depends on, and with the environment unset that value is now the ONLY thing carrying the fast path. A default
// nobody can check without a card is a claim that can only be agreed with.
// Where the chain's timeline release is attached: one per descriptor (0, the default) or one on the tail only (1).
//
// A resolver like the others, because being resolved inline is part of how this cost three card runs. Every other mode
// in this file can be asked about without a card - which is what makes "the default is X" a checkable claim instead of
// a line of code somebody has to find. This one was a getenv buried in the middle of exec_init, and I twice
// instrumented the release it selects AWAY from while believing I had instrumented the one that runs.
int tinynv_exec_tail_release(const char *e) { return e && *e && *e != '0'; }

// Whether a download waits for the compute to finish on the HOST before issuing its copy, instead of issuing the copy
// straight away and letting it wait on the card.
//
// Session A's measurement is what this exists to test. In a decode the logits copy ends 520-880 us after the token's
// last kernel retires, while the same transfer issued cold costs <=245 us end to end. The copy is not slow, it starts
// late - and the spread is the tell. llama.cpp issues the copy while compute is still running, so it lands on the copy
// channel behind an acquire on the compute timeline; a channel whose acquire is unsatisfied is switched out and
// revisited on the runlist's timeslice, which is a delay of up to a timeslice rather than a constant. The compute
// batch's own acquire on the descriptor delivery never stalls - 37 us - because that copy has already finished by the
// time the batch reaches it, so it never switches out. Prefill agrees: there the copy is issued into a compute that
// ends sooner relative to it and the same interval is 48-280 us.
//
// Waiting on the host first costs a poll (7 us granularity) and a submit (~30 us) and buys not being switched out. It
// is a knob rather than a change because it could be worse if the mechanism is something else, and because the two
// arms want measuring in the same session rather than across two builds.
int tinynv_exec_download_sync_first(const char *e) { return e && *e && *e != '0'; }

int tinynv_exec_chain_depth(const char *e) {
  if (!e || !*e) return TINYNV_EXEC_CHAIN_DEFAULT;
  int n = atoi(e);
  if (n < 1) n = 1;
  if (n > TINYNV_EXEC_CHAIN_MAX) n = TINYNV_EXEC_CHAIN_MAX;
  return n;
}

// And the delivery path, resolved the same way. On unless TINYNV_ARENA_DMA starts with '0', and forced off with a
// reason when the arena is in host memory, where there is nothing to deliver across.
int tinynv_exec_arena_dma(const char *e, int arena_vram, const char **why) {
  int on = !(e && *e && *e == '0');
  if (!on) { if (why) *why = "TINYNV_ARENA_DMA=0 was asked for"; return 0; }
  if (!arena_vram) { if (why) *why = "the arena is in host memory, so there is nothing to deliver"; return 0; }
  if (why) *why = (e && *e) ? "TINYNV_ARENA_DMA was asked for" : "the default";
  return 1;
}

// Which submission path to take, and the words for why - the startup line has to name what decided it, and "the
// default" is a different report from "you asked for this", especially now that the default is the faster path.
//
// The two environment values are arguments rather than read here so that the whole table can be checked without a GPU
// and without touching the environment (test_exec_mode). Offline there is otherwise no way to see this decision at all:
// tinynv_exec_init is reached only after a card has booted, so before this test the shipping default could only be
// confirmed by running on hardware, which is the one thing a default flip should not require.
//
// A value is off when it starts with '0' and on when it is anything else non-empty, which is the convention the other
// knobs in this file already use. Empty or unset is not an answer either way.
//
// When the two contradict, the synchronous path wins and the line says so. That is deliberate: TINYNV_SYNC=1 is what
// someone adds to a command line that already carries TINYNV_ASYNC=1 - every script and note in this project still has
// it - and they are adding it because something is wrong and they want the proven path. Silently keeping async there
// would make a debugging step do nothing while appearing to work.
int tinynv_exec_sync_mode(const char *async, const char *sync, const char **why) {
  int on_async = async && *async, on_sync = sync && *sync;
  int ask_sync  = (on_sync && *sync != '0') || (on_async && *async == '0');
  int ask_async = (on_async && *async != '0') || (on_sync && *sync == '0');
  if (ask_sync && ask_async) {
    *why = "TINYNV_SYNC and TINYNV_ASYNC ask for opposite things; taking the synchronous path";
    return 1;
  }
  if (ask_sync)  { *why = on_sync ? "TINYNV_SYNC was asked for" : "TINYNV_ASYNC=0 was asked for"; return 1; }
  if (ask_async) { *why = on_async ? "TINYNV_ASYNC was asked for" : "TINYNV_SYNC=0 was asked for"; return 0; }
  *why = "the default";
  return 0;
}

int tinynv_exec_init(tinynv_gpu_t *g, tinynv_exec_t *ex) {
  memset(ex, 0, sizeof(*ex));
  ex->g = g;
  const char *why_sync = NULL;
  ex->sync = tinynv_exec_sync_mode(getenv("TINYNV_ASYNC"), getenv("TINYNV_SYNC"), &why_sync);
  ex->chain_max = tinynv_exec_chain_depth(getenv("TINYNV_CHAIN_DEPTH"));
  { const char *a = getenv("TINYNV_ARENA_VRAM"); ex->arena_vram = !(a && *a == '0'); }
  // On unless asked otherwise, which is what the oracle does. Off is a one bit experiment against the theory that a
  // prefetched descriptor's resources are reserved before its predecessor has released them.
  { const char *e2 = getenv("TINYNV_CHAIN_PREFETCH"); ex->chain_prefetch = !(e2 && *e2 == '0'); }
  ex->tail_release = tinynv_exec_tail_release(getenv("TINYNV_TAIL_RELEASE"));
  ex->download_sync_first = tinynv_exec_download_sync_first(getenv("TINYNV_DOWNLOAD_SYNC_FIRST"));
  { const char *e = getenv("TINYNV_SHORT_FIRST_CHAIN");
    ex->short_chain = tinynv_exec_short_chain(e);
    ex->hybrid_delivery = ex->short_chain != 0;
    if (ex->hybrid_delivery)
      fprintf(stderr, "libtinynv: the first chain after a standstill carries %u launches and its descriptors ride in "
                      "the batch (was asked for)\n", ex->short_chain); }

  // Measured 2026-09-18 (TINYNV_DUMP_CMD, see the diagnostic near tinynv_exec's run()): a steady-state MoE decode
  // token's descriptor content is 2.7% different from the token before it - a handful of KV-cache offset/address
  // dwords advancing by a fixed stride, never the launch addresses themselves (ggml-cuda's MoE dispatch expresses
  // expert routing as small on-device buffer content, not as changed kernel addresses). The ring already reuses
  // the same physical address a lap later; this diffs the fresh shadow content against what was last actually
  // sent there and patches only the difference, instead of resending a structurally-fresh full chain every time.
  { const char *e = getenv("TINYNV_DELTA_DELIVERY");
    ex->delta_delivery = e && *e && *e != '0';
    if (ex->delta_delivery)
      fprintf(stderr, "libtinynv: AR_DESC delivery patches only what changed since the last lap (was asked for)\n"); }

  { const char *e = getenv("TINYNV_INLINE_PEND");
    ex->inline_pend_max = tinynv_exec_pend_entries(e);
    // Bytes scale with entries rather than being a second knob: the two would otherwise have to be swept together and
    // the interesting variable is how many uploads a step makes, not how big they are.
    ex->inline_pend_bytes_max = ex->inline_pend_max * 512u;
    if (ex->inline_pend_bytes_max > TINYNV_INLINE_PEND_CAP_BYTES)
      ex->inline_pend_bytes_max = TINYNV_INLINE_PEND_CAP_BYTES; }

  { const char *e = getenv("TINYNV_INLINE_UPLOAD");
    ex->inline_upload = tinynv_exec_inline_upload_mode(e);
    // Named like every other mode, with what decided it. A ran the arm and could not show from the log that the knob
    // had taken effect - and a routing that silently never routed looks exactly like a routing that changed nothing:
    // clean, correct and unmoved. That is the same shape as the sensor knob that gated only half of what it claimed,
    // and as TINYNV_TAIL_RELEASE being off while I instrumented the branch it selects away from. Three times in one
    // day is a convention, not a coincidence.
    fprintf(stderr, "libtinynv: small host-to-device copies %s (%s)%s\n",
            ex->inline_upload ? "ride in the pushbuffer" : "go to the copy engine",
            e && *e ? "was asked for" : "the default",
            ex->inline_upload ? ", held until the next batch goes out" : ""); }
  // Where the second window of raw refill pairs opens. The first eight are always the start of a run, which is prefill;
  // 200 is comfortably inside a decode for any run long enough to matter, and a short run simply prints fewer.
  { const char *e = getenv("TINYNV_REFILL_RAW_FROM"); ex->refill_raw_from = e && *e ? strtoull(e, NULL, 10) : 200; }
  { const char *e4 = getenv("TINYNV_NO_CHAIN_DEPS"); ex->no_chain_deps = e4 && *e4 && *e4 != '0'; }
  { const char *e5 = getenv("TINYNV_LAUNCH_PROFILE"); ex->profile = e5 && *e5 && *e5 != '0'; }
  // Delivered by the copy engine by default since 2026-09-15, having been the opt-in path for a day and a 62-minute
  // soak: 1,876 requests, 343,160 tokens, 188 greedy probes all identical, no refusals. TINYNV_ARENA_DMA=0 goes back to
  // writing descriptors across the link, for a card or a link where this turns out to be wrong rather than because the
  // default is in doubt.
  const char *why_dma = NULL;
  ex->arena_dma = tinynv_exec_arena_dma(getenv("TINYNV_ARENA_DMA"), ex->arena_vram, &why_dma);
  // The host-memory case is decided inside that call rather than corrected after it, so there is one place that knows
  // when delivery is off and why. It used to be decided twice - set from the environment here, then cleared again
  // below with a message - and two places deciding one thing is how the reason and the value come apart.
  if (!ex->arena_dma && ex->arena_vram)
    fprintf(stderr, "libtinynv: descriptors are written across the link, not delivered by the copy engine (%s)\n",
            why_dma);
  // One line, once, naming the path. Which of the two is running is the first thing to establish about a report of a
  // hang or a dropped card, and asking the reporter to remember what they exported is how that gets established wrong.
  fprintf(stderr, "libtinynv: submitting %s (%s), chaining up to %d launches (%s)\n",
          ex->sync ? "synchronously, every batch waited on" : "asynchronously",
          why_sync, ex->chain_max,
          getenv("TINYNV_CHAIN_DEPTH") && *getenv("TINYNV_CHAIN_DEPTH") ? "TINYNV_CHAIN_DEPTH was asked for"
                                                                       : "the default");
  fprintf(stderr, "libtinynv: the command arena is in %s\n",
          ex->arena_vram ? "video memory" : "host memory (TINYNV_ARENA_VRAM=0 was asked for)");
  if (!ex->chain_prefetch) fprintf(stderr, "libtinynv: chain prefetch is off (TINYNV_CHAIN_PREFETCH=0)\n");
  // Loudly, because it is a measurement mode and not a setting to leave on: it costs the two diagnostics that locate a
  // stall inside a chain, and a run that hangs with this set will say less about why than the same run without it.
  if (ex->tail_release)
    fprintf(stderr, "libtinynv: releasing ONCE PER CHAIN, not per launch (TINYNV_TAIL_RELEASE) - the timeline no longer "
                    "has a value per launch, so a stall cannot be located inside a chain\n");
  if (ex->no_chain_deps)
    fprintf(stderr, "libtinynv: *** NOTHING ORDERS ONE KERNEL AGAINST THE NEXT (TINYNV_NO_CHAIN_DEPS) *** every\n"
                    "libtinynv: descriptor is launched independently, so dependent kernels run side by side and any\n"
                    "libtinynv: result from this run is meaningless. It exists to time submission with nothing\n"
                    "libtinynv: waiting on anything. Do not use it to produce numbers anyone will act on.\n");
  // All three are host memory the GPU reads across the bus: the processor writes every one of them, and on a card behind
  // thunderbolt the window onto video memory is far too small to be where the two sides meet.
  if (tinynv_mm_alloc_buffer(&g->mm, SEM_BYTES, 1, 1, 1, 0, 1, &ex->sem)) return -1;
  // The allocator's zero flag does nothing for host memory - it zeroes through the window onto video memory, which this
  // is not - so the timeline starts as whatever the dext handed over. Every wait in this file is "has the engine reached
  // N yet", so a single non-zero byte here means the first wait returns immediately and nothing is ever really waited
  // for: work would appear to complete before it ran, and the first symptom would be a result read back too early.
  memset(ex->sem.dma.va, 0, (size_t)ex->sem.size);
  // In video memory, where the engine reads a descriptor without crossing the link. That is the difference between a
  // launch costing twenty-nine microseconds and eleven: on the 27B, decode went from 17.9 tokens a second to 45.5, and
  // on the 8B, which is smaller and so more launch-bound still, from about thirty to a hundred and thirty.
  //
  // Host memory remains available as TINYNV_ARENA_VRAM=0. It is not there because this is doubted - it has been through
  // both models, a five-hundred token decode across many wraps, three byte-identical greedy runs, and the synchronous
  // path - but because a card whose window does not reach, or a backend whose block write behaves differently, would
  // otherwise have nowhere to go.
  // How big the descriptor region is, and it is a different question in each mode. When the processor writes it, it has
  // to live in the window onto video memory, and that window is 256 MB with the boot's own reserve already inside it -
  // so a few megabytes is all there is. When the copy engine delivers it, the processor never touches it, so it comes
  // from plain video memory and the only thing bounding it is the host memory its shadow and mirror cost beside it.
  //
  // That is worth paying. The wrap is the last thing on this path that stops to wait, and an hour of real serving spent
  // 14.5% of itself draining, at 150,297 wraps of a 3.5 MB region. Sixty-four megabytes is eighteen times fewer.
  uint64_t desc_bytes = CMD_BYTES - CMD_REGION_BYTES;
  if (ex->arena_dma) {
    desc_bytes = (uint64_t)DESC_MB_DEFAULT << 20;
    const char *e = getenv("TINYNV_DESC_MB");
    if (e && *e) { long mb = atol(e); if (mb >= 1 && mb <= 1024) desc_bytes = (uint64_t)mb << 20; }
  }
  ex->region[AR_DESC].size = desc_bytes;
  ex->region[AR_CMD].size = CMD_REGION_BYTES;

  if (ex->arena_vram) {
    // Only what the processor still writes has to be reachable through the window: command buffers always, descriptors
    // only when the copy engine is not delivering them.
    uint64_t visible = CMD_REGION_BYTES + (ex->arena_dma ? 0 : desc_bytes);
    if (tinynv_mm_reserve_cpu_region(&g->mm, visible + (1u << 20))) return -1;
    if (tinynv_mm_alloc_buffer(&g->mm, CMD_REGION_BYTES, 0, 1, 1, 1, 0, &ex->region[AR_CMD].mem)) return -1;
    if (tinynv_mm_alloc_buffer(&g->mm, desc_bytes, 0, ex->arena_dma ? 0 : 1, 1, 1, 0, &ex->region[AR_DESC].mem))
      return -1;
    for (int r = 0; r < 2; r++)
      if (!(ex->region[r].shadow = calloc(1, (size_t)ex->region[r].size)))
        return tinynv_fail("out of memory for a %llu byte scratch shadow", (unsigned long long)ex->region[r].size);
    if (ex->delta_delivery &&
        !(ex->region[AR_DESC].last_delivered = calloc(1, (size_t)ex->region[AR_DESC].size)))
      return tinynv_fail("out of memory for a %llu byte last-delivered mirror",
                         (unsigned long long)ex->region[AR_DESC].size);
  } else {
    if (tinynv_mm_alloc_buffer(&g->mm, CMD_REGION_BYTES, 1, 1, 1, 0, 1, &ex->region[AR_CMD].mem)) return -1;
    if (tinynv_mm_alloc_buffer(&g->mm, desc_bytes, 1, 1, 1, 0, 1, &ex->region[AR_DESC].mem)) return -1;
  }
  ex->region[AR_DESC].next = ex->region[AR_CMD].next = 0;
  // Tell the crash witness about the buffers this driver holds for the life of the device. Recording them once here
  // rather than on each write keeps the launch path free of a store, and it answers the question a per-write list
  // cannot: whether a faulting address is inside something of ours at all.
  for (int r = 0; r < 2; r++) {
    tinynv_note_host_region(ex->region[r].shadow, (size_t)ex->region[r].size,
                            r == AR_DESC ? "descriptor shadow" : "command shadow");
    tinynv_note_host_region(ex->region[r].mem.dma.va, (size_t)ex->region[r].mem.dma.size,
                            r == AR_DESC ? "descriptor region (host memory)" : "command region (host memory)");
  }

  if (tinynv_mm_alloc_buffer(&g->mm, STAGE_BYTES, 1, 1, 1, 0, 0, &ex->stage)) return -1;
  tinynv_note_host_region(ex->stage.dma.va, (size_t)ex->stage.dma.size, "copy staging buffer");
  // Laid out at the descriptor region's own offsets, so a dirty span copies to the place it came from and no mapping
  // arithmetic exists anywhere to get wrong. One allocation whose base does not move for the life of the device, which
  // is also what lets phase 2 bind an aperture to it once rather than renegotiating whenever it grows.
  if (ex->arena_dma && tinynv_mm_alloc_buffer(&g->mm, desc_bytes, 1, 1, 1, 0, 0, &ex->mirror)) return -1;
  tinynv_note_host_region(ex->mirror.dma.va, (size_t)ex->mirror.dma.size, "descriptor mirror");
  if (ex->arena_dma)
    // Naming what decided it, not just the knob. "(TINYNV_ARENA_DMA)" read as "you asked for this" on every line it
    // ever printed, which was true while it was opt-in and stopped being true the moment it became the default - and a
    // run started with no environment at all is exactly the run where that distinction is the thing being checked.
    fprintf(stderr, "libtinynv: descriptors are delivered by the copy engine (%s), not written across the link by the "
                    "processor; TINYNV_ARENA_DMA=0 goes back\n", why_dma);

  // Each engine is told which class it runs and, for the compute engine, where local and shared memory appear. Until
  // this batch has been through, the engines will not run anything: an unbound subchannel faults on the first method.
  uint64_t va;
  tinynv_cmdbuf_t c;
  if (batch_begin(ex, &g->gsp.compute_q, 64, &c, &va)) return -1;
  if (tinynv_cmd_set_object(&c, 1, TINYNV_CLASS_COMPUTE)) return -1;
  if (tinynv_cmd_shader_window(&c, 0, TINYNV_LOCAL_WINDOW)) return -1;
  if (tinynv_cmd_shader_window(&c, 1, TINYNV_SHARED_WINDOW)) return -1;
  if (run(ex, &g->gsp.compute_q, &c, va)) return -1;

  if (batch_begin(ex, &g->gsp.copy_q, 40, &c, &va)) return -1;
  if (tinynv_cmd_set_object(&c, 4, TINYNV_CLASS_DMA_COPY)) return -1;
  if (run(ex, &g->gsp.copy_q, &c, va)) return -1;
  // the engines must be bound before anything is built on them, and this happens once
  if (tinynv_exec_idle(ex)) return -1;

  ex->ready = 1;
  return 0;
}

void tinynv_exec_fini(tinynv_exec_t *ex) {
  // One line, and only when it happened, because a wrap is a stall and a run that paid several wants to know without
  // having to have asked in advance.
  if (ex->wraps[AR_DESC] || ex->wraps[AR_CMD] || ex->seg_waits)
    fprintf(stderr, "\ntinynv: scratch came round %llu times in the descriptor region and %llu in the command buffer; "
                    "%llu of the pieces reused were still in use, costing %.2f ms of waiting in total\n",
            (unsigned long long)ex->wraps[AR_DESC], (unsigned long long)ex->wraps[AR_CMD],
            (unsigned long long)ex->seg_waits, (double)ex->wrap_wait_ns / 1e6);
  // OUTSIDE the profile, deliberately. It was inside it, which would have been useless for exactly the run that
  // needed it: A's inline arm is a CORRECTNESS experiment - op-verify and a byte-identical text - and nobody runs
  // those with a launch profile on. A count that only appears when you were already measuring cannot answer "was the
  // knob in effect", which is the one question it exists for.
  if (ex->hybrid_n)
    fprintf(stderr, "libtinynv: %llu descriptor deliveries rode in the compute batch instead of crossing on the copy "
                    "engine.\n", (unsigned long long)ex->hybrid_n);
  if (ex->delta_n || ex->delta_skip_n || ex->delta_full_n)
    fprintf(stderr, "libtinynv: delta delivery patched %llu spans (%llu bytes total, %.1f avg), skipped %llu with "
                    "nothing changed, and %llu fell back full because the genuine diff did not fit inline either.\n",
            (unsigned long long)ex->delta_n, (unsigned long long)ex->delta_bytes,
            ex->delta_n ? (double)ex->delta_bytes / (double)ex->delta_n : 0.0,
            (unsigned long long)ex->delta_skip_n, (unsigned long long)ex->delta_full_n);
  if (ex->inline_n || ex->upload_ce_n)
    fprintf(stderr, "libtinynv: %llu host-to-device copies rode in the pushbuffer (%llu bytes) and %llu went to the "
                    "copy engine.\n           Held list (%u entries, %u bytes): %llu rode in a batch already going "
                    "out, %llu overflowed onto work\n           already pending, %llu overflowed with nothing pending "
                    "and cost a batch of their own.\n           Only the LAST number is the win being given back; a "
                    "large one means the list wants to be bigger.\n",
            (unsigned long long)ex->inline_n, (unsigned long long)ex->inline_bytes,
            (unsigned long long)ex->upload_ce_n, ex->inline_pend_max, ex->inline_pend_bytes_max, (unsigned long long)ex->inline_batches_saved,
            (unsigned long long)ex->inline_rode_flush_n, (unsigned long long)ex->inline_forced_n);

  if (ex->profile && ex->prof_n[PROF_LAUNCH]) {
    uint64_t L = ex->prof_n[PROF_LAUNCH];
    fprintf(stderr, "libtinynv: where the processor's time went, per launch, over %llu launches\n",
            (unsigned long long)L);
    for (int i = 0; i < 5; i++)
      fprintf(stderr, "libtinynv:   %-46s %7.2f us  (%llu calls)\n", PROF_NAME[i],
              (double)ex->prof_ns[i] / 1e3 / (double)L, (unsigned long long)ex->prof_n[i]);
    // The two wait lines are per launch like everything above, so they can be added to the others and compared against
    // what a launch is actually allowed to cost. That comparison is the point: if building a launch is a fraction of
    // the budget and standing still is most of it, then no amount of faster submission changes the answer.
    for (int i = PROF_WAIT; i <= PROF_BLOCKED; i++)
      fprintf(stderr, "libtinynv:   %-46s %7.2f us  (%llu calls)\n", PROF_NAME[i],
              (double)ex->prof_ns[i] / 1e3 / (double)L, (unsigned long long)ex->prof_n[i]);
    double named = (double)(ex->prof_ns[PROF_FLUSH]);
    double rest_of_flush = named - (double)(ex->prof_ns[PROF_PUSH] + ex->prof_ns[PROF_SUBMIT] + ex->prof_ns[PROF_POLL]);
    fprintf(stderr, "libtinynv:   %-46s %7.2f us\n", "  of which everything else in a flush",
            rest_of_flush / 1e3 / (double)L);
    fprintf(stderr, "libtinynv:   %-46s %7.2f us\n", "building a launch, outside any flush",
            (double)(ex->prof_ns[PROF_LAUNCH] - ex->prof_ns[PROF_FLUSH]) / 1e3 / (double)L);
    // And the summary that answers the question, stated as a share rather than left to be worked out. The host is
    // either the thing holding this up or it is not, and the two numbers that decide it are how much of a launch it
    // spends working and how much it spends stopped.
    {
      double work = (double)ex->prof_ns[PROF_LAUNCH] / 1e3 / (double)L;
      double stopped = (double)ex->prof_ns[PROF_BLOCKED] / 1e3 / (double)L;
      fprintf(stderr, "libtinynv:   so per launch the processor spends %.2f us building and %.2f us stopped waiting "
                      "for the engine;\n", work, stopped);
      fprintf(stderr, "libtinynv:   %llu of %llu waits had to stand still, and everything above is host time - the rest "
                      "of a launch is the engine working.\n",
              (unsigned long long)ex->prof_n[PROF_BLOCKED], (unsigned long long)ex->prof_n[PROF_WAIT]);
    }
    // The engine's own account of the refill, which is the only part of a synchronisation neither side was measuring.
    for (unsigned i = 0; i < ex->refill_raw_n; i++)
      fprintf(stderr, "libtinynv:   refill %u raw: tail %llu  head %llu  (head-tail %lld ns)  report payload %llu, "
                      "timeline %llu, copy-end %llu, reached %llu, copy-start %llu%s\n", i,
              (unsigned long long)ex->refill_raw[i][0], (unsigned long long)ex->refill_raw[i][1],
              (long long)(ex->refill_raw[i][1] - ex->refill_raw[i][0]),
              (unsigned long long)ex->refill_raw[i][2], (unsigned long long)ex->refill_raw[i][3],
              (unsigned long long)ex->refill_raw[i][4], (unsigned long long)ex->refill_raw[i][5],
              (unsigned long long)ex->refill_raw[i][6],
              ex->refill_raw[i][2] == ex->refill_raw[i][3]
                  ? "  (payload is the compute queue's last value)"
                  : "  (payload is NOT the compute queue's last value)");
    if (ex->refill_host_n)
      fprintf(stderr, "libtinynv:   and on the host's own clock the same gap took %.1f us, of which %.1f us was inside "
                      "this driver\n            building launches - the rest is the caller between one token and the "
                      "next (%llu measured).\n",
              (double)ex->refill_host_ns / 1e3 / (double)ex->refill_host_n,
              (double)ex->refill_build_ns / 1e3 / (double)ex->refill_host_n,
              (unsigned long long)ex->refill_host_n);
    if (ex->refill_period_n)
      fprintf(stderr, "libtinynv:   head to head, every refill: %.1f us across %llu - includes prefill and warm-up, "
                      "whose spacing is not a token.\n",
              (double)ex->refill_period_ns / 1e3 / (double)ex->refill_period_n,
              (unsigned long long)ex->refill_period_n);
    if (ex->refill_period_dec_n)
      fprintf(stderr, "libtinynv:   head to head, DECODE ONLY: %.1f us across %llu - one token as the card times it, "
                      "and the number\n            to hold everything else against: it cannot be shrunk by moving a "
                      "stamp, which the intervals above can.\n",
              (double)ex->refill_period_dec_ns / 1e3 / (double)ex->refill_period_dec_n,
              (unsigned long long)ex->refill_period_dec_n);
    else if (ex->refill_period_n)
      fprintf(stderr, "libtinynv:   (no decode-window periods: the run ended before refill %llu, so the line above is "
                      "all there is)\n", (unsigned long long)ex->refill_raw_from);
    if (ex->refill_copy_n)
      fprintf(stderr, "libtinynv:   %.1f us of that was the copy engine still working after the last kernel retired - "
                      "the logits\n            going back to the host, which is work rather than latency (%llu "
                      "measured).\n",
              (double)ex->refill_copy_ns / 1e3 / (double)ex->refill_copy_n,
              (unsigned long long)ex->refill_copy_n);
    // NOT "transferring", which is what this said until A measured it at 2 us on every single refill. The copy's
    // LAUNCH_DMA moves its data asynchronously to the methods that follow it, and neither of these stamps waits for
    // it - so the pair brackets the two stamps' own spacing and nothing else. The second number is kept because it is
    // evidence about the copy engine's method throughput, and renamed because calling it the transfer made a 2 us
    // reading look like a measurement of a 993 KB copy. Timing the transfer needs a release that waits for it.
    if (ex->refill_cstart_n)
      fprintf(stderr, "libtinynv:   of that, %.1f us before the copy engine reached the copy batch, then %.1f us to "
                      "the next stamp\n            (NOT the transfer: nothing here waits for it) (%llu measured).\n",
              (double)ex->refill_cwait_ns / 1e3 / (double)ex->refill_cstart_n,
              (double)ex->refill_crun_ns / 1e3 / (double)ex->refill_cstart_n,
              (unsigned long long)ex->refill_cstart_n);
    if (ex->refill_reach_n)
      fprintf(stderr, "libtinynv:   of which %.1f us before the engine reached the batch and %.1f us with the batch "
                      "reached but\n            held at its acquire - the copy engine delivering this batch's "
                      "descriptors (%llu measured).\n",
              (double)ex->refill_reach_ns / 1e3 / (double)ex->refill_reach_n,
              (double)ex->refill_acq_ns / 1e3 / (double)ex->refill_reach_n,
              (unsigned long long)ex->refill_reach_n);
    if (ex->refill_submit_n)
      fprintf(stderr, "libtinynv:   and %.1f us had passed when that batch was actually handed over, so the engine took "
                      "the\n            remaining %.1f us to start work already on its ring (%llu measured).\n",
              (double)ex->refill_submit_ns / 1e3 / (double)ex->refill_submit_n,
              (double)ex->refill_ns / 1e3 / (double)(ex->refill_n ? ex->refill_n : 1) -
                  (double)ex->refill_submit_ns / 1e3 / (double)ex->refill_submit_n,
              (unsigned long long)ex->refill_submit_n);
    if (ex->refill_asked)
      fprintf(stderr, "libtinynv:   %llu stalls asked to be measured, %llu got a head timestamp, %llu were paired - so "
                      "the average below\n            is over that last group, not over every stall.\n",
              (unsigned long long)ex->refill_asked, (unsigned long long)ex->refill_armed_n,
              (unsigned long long)(ex->refill_n + ex->refill_bad));
    if (ex->refill_stuck)
      fprintf(stderr, "libtinynv:   the tail timestamp did not advance on %llu refills - the report being read is not "
                      "the one\n            the finished work wrote, so nothing below is a measurement of idleness.\n",
              (unsigned long long)ex->refill_stuck);
    if (ex->refill_bad)
      fprintf(stderr, "libtinynv:   %llu refill measurements were discarded as not physical (zero, or over 100 ms). If "
                      "that is most\n            of them the two timestamps are not on the same clock and the average "
                      "below means nothing.\n", (unsigned long long)ex->refill_bad);
    if (ex->refill_n)
      fprintf(stderr, "libtinynv:   the engine sat idle %.1f us on average across %llu refills (worst %.1f us), which "
                      "is its own clock\n            between finishing a batch and starting the one the host built "
                      "after standing still.\n",
              (double)ex->refill_ns / 1e3 / (double)ex->refill_n, (unsigned long long)ex->refill_n,
              (double)ex->refill_max_ns / 1e3);
    fprintf(stderr, "libtinynv: the profile costs two clock reads per phase and about 1%% of a launch; it is measuring\n"
                    "libtinynv: itself as well as the driver, so read the shares rather than the absolute total.\n");
  }
  tinynv_vmap_free(&ex->g->mm, &ex->slm);
  tinynv_vmap_free(&ex->g->mm, &ex->stage);
  if (ex->mirror.size) tinynv_vmap_free(&ex->g->mm, &ex->mirror);
  for (int r = 0; r < 2; r++) {
    if (ex->region[r].mem.size) tinynv_vmap_free(&ex->g->mm, &ex->region[r].mem);
    free(ex->region[r].shadow);
    free(ex->region[r].last_delivered);
  }
  tinynv_vmap_free(&ex->g->mm, &ex->sem);
  memset(ex, 0, sizeof(*ex));
}
