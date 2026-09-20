// Running work on a GPU that is already up: memory the caller can use, copies, and kernels.
//
// Everything here sits on top of the encoders in submit.c and qmd.c, which build bytes, and adds the part that has to
// exist for those bytes to mean anything: somewhere for them to live, a semaphore to tell when the engine is done, and
// the once-per-channel setup the engines need before they will run anything at all.
#ifndef TINYNV_EXEC_H
#define TINYNV_EXEC_H
#include "cubin.h"
#include "gpu.h"
#include "qmd.h"
#include "submit.h"

// Where a kernel's local and shared memory appear in the address space. Fixed addresses the hardware recognises, not
// allocations: the oracle hardcodes the same two.
// How many launches may be chained before the chain is handed over. The whole chain is dispatched by one launch method,
// so this is also how much of the per-submission cost a kernel pays: at 32 a decoded token's six hundred launches cost
// nineteen submissions instead of six hundred. Larger is not better without limit - a launch does nothing until the
// chain it is in is flushed, so an over-long chain leaves the engine idle while the host is still building.
// How deep a chain may be asked to go, and how deep it goes when nobody asks. They are separate numbers now because
// the submit cost is paid once per flush and divided by the depth: the profile puts it at ~222 us per flush against
// 2.9 for pushing the descriptors, so depth is a direct divider on the largest term in a launch - 3.5 us per launch at
// 64, 1.7 at 128, against 6.9 at 32. Raising the ceiling lets that be measured; the default stays where it was
// measured, because a default that moves without a hardware run is the thing this project has twice refused to ship.
// How many pieces a region is recycled in. One piece is a bump arena that has to drain everything before it can start
// again; several means coming round only ever waits for the work that used the piece being reused, which by then is
// several pieces old and almost always finished. Eight is enough that the wait is essentially never taken and few
// enough that a chain still fits comfortably inside one piece.
// The region is recycled a piece at a time rather than all at once, and this is how many pieces. Each one records the
// last batch that used it, so coming round to a piece waits for that batch alone. More pieces means a shorter wait and
// a coarser-grained record; eight is small enough that the per-allocation check stays a couple of comparisons.
#ifndef TINYNV_EXEC_SEGMENTS
#define TINYNV_EXEC_SEGMENTS 8
#endif

// Where the timeline counter is considered close enough to its 32-bit ceiling to be worth restarting. The counter
// advances once per launch, so the ceiling is about nine hours of continuous generating; this leaves a quarter of a
// billion launches of headroom, which is half an hour at the rate decode reaches, and a region wrap comes round every
// few thousand launches. So the restart is always taken in time and is almost never taken at all.
#define TINYNV_EXEC_REBASE_AT 0xF0000000ull
// The deepest chain the array below can hold. 128 until 2026-09-19; raised to 1024 for the depth sweep that measures
// what a chain handover (~15-22 us of engine idleness between chains, exec.c) still costs a token at the delta
// delivery defaults, and to enter the long-dependent-chain regime deliberately, with op-verify at each depth, before
// any launch-chain replay (docs/driver/chain-replay-plan.md SS7) would put ~2,000 links in one chain. The DEFAULT
// depth is unchanged at TINYNV_EXEC_CHAIN_DEFAULT; TINYNV_CHAIN_DEPTH asks for more. A chain's descriptors must fit
// the descriptor region in one piece (1024 x ~1.5 KB = 1.5 MB of 64 MB), and the delta span list holds one slot per
// launch (asserted below). The sweep's answer (interleaved tg128, both models, 2026-09-19 12:30-12:38): 128 IS the
// optimum - 1024 costs the MoE 14% and the dense 17%, 512/256 in between, 32 costs the MoE 10%, 64 is within noise
// below - because the engine cannot start a chain until the host has built all of it, so seams were already hidden
// under the next chain's build and a deeper chain only makes the engine wait longer for its first descriptor.
// op-verify passed 450/450 at 256, 512 and 1024: a thousand-link dependent chain completes correctly on this card.
#define TINYNV_EXEC_CHAIN_MAX 1024

// TINYNV_DELTA_DELIVERY patches a launch's descriptor span as the runs of dwords that actually differ from what was
// last delivered there, not as one envelope from its first differing byte to its last (tinynv_delta_runs, and the check
// pass in tinynv_exec_flush). A flush's runs are kept in one flat list in launch order: launches sit consecutively in
// AR_DESC and the list is read straight into the chain's own batch, so a per-launch table would only add a second cap
// to reason about. Measured 2026-09-19, a launch's ~1.5 KB span carries 40-240 genuinely changed bytes a token - a
// handful of runs - so 4096 across a 128-launch flush is generous; a launch that wants more than
// TINYNV_DELTA_LAUNCH_RUNS_MAX collapses back to its one envelope rather than failing the flush, which is exactly what
// shipped before runs existed and so never worse than the day before. Offsets are 32-bit: the region is at most 1 GB
// (TINYNV_DESC_MB), checked where it is sized.
#define TINYNV_DELTA_SPANS_MAX       4096
#define TINYNV_DELTA_LAUNCH_RUNS_MAX 64
typedef struct { uint32_t lo, hi; } tinynv_delta_span_t;   // [lo,hi) as offsets into AR_DESC, both dword multiples

// TINYNV_KERNEL_PROFILE: the engine's clock at every kernel's completion, by kernel. See the fields in tinynv_exec_t.
#define TINYNV_KPROF_SLOTS 16384                 // a token is ~1,000-2,000 launches; the ring holds several
#define TINYNV_KPROF_ROWS 1024                   // distinct kernel instantiations a run may name
#define TINYNV_KPROF_INDEX (TINYNV_KPROF_ROWS * 4)
typedef struct {
  uint64_t at;        // the timeline value this launch reserved, which is also the payload its stamp carries
  int32_t row;        // which kernel, or -1 once the rows ran out
  uint8_t first;      // the first launch of its chain: the interval ending here also carries the chain seam
  uint8_t boundary;   // the first launch after a standstill: the interval ending here is the token boundary
  uint8_t tail;       // the last launch of its chain: stamped by the command stream, after the timeline release
  uint8_t broken;     // a launch before this one went unstamped, so the interval ending here spans two kernels
} tinynv_kprof_slot_t;
// TINYNV_GRAPH_RESIDENT: a recorded token. See tinynv_exec_graph_begin in exec.c.
// A recorded token is its chains, each of at most chain_max descriptors: one long chain measured slower than
// thirteen of 128 (2026-09-19 16:42, -6.5%), so a replay issues the chains as consecutive batches under one
// announcement, each acquiring on the previous one's tail release.
#define TINYNV_GRAPH_CHAINS 128
typedef struct { uint64_t head_va, tail_va; uint8_t *head_host; tinynv_qmd_t tail_qmd; uint32_t n; } tinynv_graph_chain_t;
typedef struct tinynv_graph_rec {
  tinynv_vmap_t mem;       // video memory: what the engine reads on every replay
  tinynv_vmap_t mirror;    // host memory the engine can read: built here, copied across once when sealed
  uint64_t size, used;
  uint8_t *prev_host;      // the previous descriptor of the chain being recorded, to write its link into
  tinynv_qmd_t prev_qmd;
  tinynv_graph_chain_t chains[TINYNV_GRAPH_CHAINS];
  int nchains;
  uint32_t n;              // launches in total
  int sealed, failed;
  uint64_t launches;
} tinynv_graph_rec_t;
typedef struct {
  const void *key;    // the kernel descriptor's address: one row per instantiation
  char name[112];     // readable (tinynv_kernel_short_name), copied at the launch while the module is certainly loaded
  uint64_t n, ns, max_ns;
} tinynv_kprof_row_t;
// The check pass holds one slot back for every launch still to come, so every launch always has room for at least its
// envelope - which needs the list to be at least as long as a chain.
_Static_assert(TINYNV_DELTA_SPANS_MAX >= TINYNV_EXEC_CHAIN_MAX, "every launch in a chain needs a delta slot");

// A region lives in video memory, where the engine reads a descriptor without crossing the link. The processor cannot
// memcpy into that, so each region has a shadow of ordinary malloc'd memory: building in it is free, and what crosses
// is one block write per batch, or - for descriptors now - one copy-engine transfer.
//
// Each region is its own allocation. They want different memory: command buffers must stay where the processor can
// write them, which is inside a 256 MB window, while descriptors are no longer written by the processor at all and
// can come from plain video memory and be as large as the wrap rate wants.
//
// There are two of them, and the reason is a bug rather than tidiness. flush() zeroes the chain count and THEN
// allocates its own command buffer, so a wrap during that allocation saw a count of zero, passed the guard meant to
// refuse exactly that, and reset the arena under descriptors built moments earlier - which then existed only in the
// host shadow while the engine read whatever video memory held a wrap ago. Six hardware runs. With one writer per
// region, flush's command buffer cannot reach the descriptors at all: the guard stops being a partial check.
//
// It is also what makes delivering descriptors by DMA possible. A command buffer has to be present before the engine
// reads it, so it cannot arrive in a transfer that the engine has not been told about yet; descriptors can. Separate
// regions let the small thing go by register write and the large one by copy engine.
typedef struct {
  tinynv_vmap_t mem;   // where the engine reads this region
  uint8_t *shadow;     // where the processor builds it, or NULL when it writes the mapping directly
  // TINYNV_DELTA_DELIVERY only: a copy of shadow as of the last time each byte was actually sent to video memory.
  // The ring wraps and reuses the same physical addresses, so diffing fresh shadow content against this - rather
  // than against nothing - finds the genuine per-token delta (measured 2026-09-18: 2.7% of a steady-state MoE
  // decode chain) instead of re-sending a structurally-fresh span in full every time. NULL when the feature is
  // off; allocated the same size as shadow when on. Trusted only after this region has completed a full lap
  // (exec.c's wraps[] counter), so every byte has a real prior delivery to diff against - see the delivery code.
  uint8_t *last_delivered;
  uint64_t size, next, dirty_lo, dirty_hi;
  // What has been handed out since the last submission, which is a different question from what has yet to be
  // written to video memory. The dirty span is cleared whenever the bytes are pushed, and pushing can happen without
  // a submission; if ownership were recorded from that span, bytes pushed early would be recorded against nothing
  // and handed out again a lap later while the engine was still reading them. This span is cleared only by the
  // submission that gives it a value.
  uint64_t used_lo, used_hi;
  // The highest timeline value of any batch that has used each segment of this region. Coming round to a segment
  // means waiting for its value and nothing else - not for everything outstanding, which is what a bump arena
  // resetting to its base had to do and what cost an hour of serving 14.5% of itself.
  uint64_t seg_value[TINYNV_EXEC_SEGMENTS];
  // How many pieces this lap has entered. The bump pointer only moves forward within a lap, so a piece already entered
  // holds nothing but this lap's own work - work that is being built or has just been sent, never work whose bytes are
  // about to be handed out again. Asking the record inside a piece already entered would therefore wait for the batch
  // just submitted, on every allocation, which is a stall per launch rather than a stall per lap. So the record is read
  // once, when the piece is entered, and cleared there.
  uint64_t seg_entered;
} tinynv_exec_region_t;

// The recycling decisions, kept apart from the machinery that acts on them so that they can be checked without a card.
// Between them they answer the three questions a region is asked: where would this land, what has to finish first, and
// who owns it now. test/test_arena.c drives them against a byte-by-byte record of who wrote what.
uint64_t tinynv_arena_place(const tinynv_exec_region_t *rg, uint64_t bytes, uint64_t align, int *wrap);
uint64_t tinynv_arena_wait_for(const tinynv_exec_region_t *rg, uint64_t off, uint64_t bytes);
uint64_t tinynv_arena_enter(tinynv_exec_region_t *rg, uint64_t off, uint64_t bytes);
void tinynv_arena_take(tinynv_exec_region_t *rg, uint64_t off, uint64_t bytes);
void tinynv_arena_mark(tinynv_exec_region_t *rg, uint64_t lo, uint64_t hi, uint64_t value);
void tinynv_arena_submitted(tinynv_exec_region_t *rg, uint64_t value);
int tinynv_arena_rewind(tinynv_exec_region_t *rg);   // 0 if the caller had submitted and pushed first

// 128 rather than 32 since 2026-09-15. The sweep that found it also found why a default this deep was not obviously
// right: a chain handover costs about fifteen microseconds, so depth buys less the deeper it goes, and the failures at
// 32 and at 2 were different failures that looked alike. It is a default rather than a constant because that sweep has
// to stay possible without a build whose id says "dirty".
#define TINYNV_EXEC_CHAIN_DEFAULT 128

#define TINYNV_SHARED_WINDOW 0x729400000000ull
#define TINYNV_LOCAL_WINDOW  0x729300000000ull

typedef struct {
  tinynv_gpu_t *g;
  // The timeline. One allocation, but a slot per queue, and that is not tidiness: a single location written by two
  // engines is not monotonic. Each engine writes whatever value it has just finished, so a batch that completes later
  // with a lower number overwrites a higher one, and an acquire waiting on the higher number is then parked forever on
  // a value that has already been and gone. A slot per queue is monotonic because a channel retires in order, and it is
  // what the oracle does - one signal per queue - which is a thing I noticed early and did not act on.
  tinynv_vmap_t sem;
  uint64_t q_last[2];    // the last value each queue was asked to release; index 0 compute, 1 copy
  uint64_t sem_high[2];  // the highest value ever read back from each slot, so going backwards can be noticed at all
  uint64_t nbackwards;   // how many times one did, since a wait spins and printing every one buries the run

  tinynv_exec_region_t region[2];
  // Which of the two the arena is. Video memory by default: the engine reads a descriptor without crossing the link,
  // which is most of what a launch used to cost. TINYNV_ARENA_VRAM=0 goes back to host memory, for a card whose window
  // does not reach rather than because the default is in doubt.
  int arena_vram;
  // Set while a batch should be written but not announced, so that the batch after it can announce both with one
  // fence. See tinynv_submit_stage: the read before the doorbell is a round trip on this backend, and one read orders
  // every write issued before it, so two batches going out together need one read rather than two.
  int defer_ring;
  tinynv_vmap_t cmd;     // command buffers, written here and read by the engine over the bus
  tinynv_vmap_t stage;   // where host memory passes through on its way to the card and back
  tinynv_vmap_t slm;     // shader local memory, which has to exist before any kernel that uses a stack
  uint64_t timeline;     // the last value asked for on any queue; what a caller means by "everything so far"

  // Launches built but not yet handed to the ring. A kernel is ordered against the one before it by the descriptor
  // chain, not by draining the engine between them, and a chain has to be complete before its head is dispatched -
  // so a launch is held until something needs it to have happened. See flush() in exec.c.
  struct {
    tinynv_qmd_t qmd;    // the descriptor, still being edited: the next launch writes its chain pointer into this one
    uint8_t *host;       // where it will be copied, in the arena the GPU reads
    uint64_t va;         // and that slot's address, which is what the chain pointer and the launch method carry
    // How many bytes this launch occupies in AR_DESC (the QMD slot plus its own cbuf0, which varies per kernel's
    // parameter count) - TINYNV_DELTA_DELIVERY diffs this span alone against last_delivered, rather than one
    // envelope over every launch in the flush. See tinynv_exec_flush.
    uint32_t len;
    uint32_t kslot;      // TINYNV_KERNEL_PROFILE: the ring slot this launch stamps, or TINYNV_KPROF_SLOTS for none
  } chain[TINYNV_EXEC_CHAIN_MAX];
  int nchain;
  int chain_max;   // how deep a chain may get before it is handed over; TINYNV_EXEC_CHAIN_MAX unless asked otherwise
  int chain_prefetch;   // whether a chained descriptor is fetched early; on unless TINYNV_CHAIN_PREFETCH=0
  // Timeline values handed out, including to launches not yet submitted. `timeline` is what the engine has been asked
  // for and may be waited on; `reserved` runs ahead of it while a chain is being built. Waiting on the difference is
  // waiting for work nothing has been told about, so everything that waits flushes first.
  uint64_t reserved;
  // What the last chain handed over covered, kept only so a stall can say where the engine stopped relative to it.
  // "The engine is at 2036" answers nothing on its own; "it is at link 20 of the 32 in the chain covering 2017..2048"
  // says the chain was dispatched and died partway, which is a different bug from one never dispatched at all.
  uint64_t flush_lo;
  int flush_n;

  // The last few things handed to a ring, kept so a stall can show what is outstanding rather than only that something
  // is. The question a stall cannot currently answer is *which* batch the engine stopped at and what that batch was
  // waiting for - and an acquire for a value nothing will release parks a channel silently, which is what this is for.
#define TINYNV_EXEC_TRAIL 128
  struct {
    uint64_t va;        // where the command buffer was
    uint64_t acquire;   // the value its semaphore acquire waits for on its OWN queue's slot, or 0 if it has none
    uint64_t acquire_other;  // and on the other queue's slot, since a batch can wait on both and once did
    uint64_t upto;      // the timeline value it will have reached when it is done
    uint64_t seq;       // which submission this was, as a label; the slot it sits in here moves as finished ones go
    uint64_t slot;      // the ring slot it went into, so the entry can be read back and compared
    uint64_t head_va;   // the chain head's descriptor, 0 if this is not a launch batch
    uint64_t head_sum;  // and a checksum of that descriptor as it was written, to catch it being overwritten since
    uint32_t dwords;
    uint8_t copy;       // which ring it went to
    uint8_t links;      // how many launches were chained in it, 0 if it is not a launch batch
  } trail[TINYNV_EXEC_TRAIL];
  // The trail keeps the OLDEST outstanding batches, not the most recent ones, and the difference is the whole point.
  // A ring of the last N holds whatever was submitted most recently, which - when the host has run a thousand values
  // ahead of a stalled engine - is precisely the work that is fine. The batch worth looking at is the one the engine
  // stopped at, which is the oldest thing still unfinished. So this fills in order and stops when full, and finished
  // entries are dropped as the timeline passes them rather than by being overwritten.
  int ntrail;           // how many slots are in use, oldest first
  uint64_t ndropped;    // submissions not recorded because every slot held something still outstanding
  uint64_t nsubmit;     // total submissions ever, which is only a label
  // What the batch being built acquires on, handed from batch_begin to submit_batch. Two of them, because a compute
  // batch that follows a copy waits on BOTH - the copy's slot, emitted by batch_begin when the queue changes, and its
  // own slot, emitted by flush to order one chain against the last. One field meant the second overwrote the first and
  // the stall report named only the compute wait, which sent Session A looking for a missing cross-queue acquire that
  // was in the command buffer all along. A report that names one of two waits is worse than one that names neither.
  uint64_t pending_acquire, pending_acquire_other;
  uint64_t pending_head_va, pending_head_sum;   // and the same for the chain head, handed from flush
  tinynv_queue_t *last_q; // which queue the last batch went to, so a switch can be ordered against it
  int ready;
  // Wait for every batch before returning, the way this file worked before d29cb32. Slow - it is a round trip to the
  // card per batch, and on the 27B it is nine tokens a second against forty-five - but it is whole, it is the path the
  // first model ran a token through, and it is what a hang should be re-run on before anything else is suspected.
  // Off unless asked for; set from TINYNV_SYNC / TINYNV_ASYNC at init by tinynv_exec_sync_mode below.
  int sync;
  // Release once per chain instead of once per launch, which is what the oracle does. The default since 2026-09-19
  // (see tinynv_exec_tail_release): it trades the two diagnostics that depend on per-link timeline values - the
  // "stopped at link N of M" message and the first failure mode of tinynv_qmd_link_check - for descriptors that stay
  // identical token to token, which is what the delta path lives on. Both diagnostics say loudly that they are
  // degraded rather than quietly saying less, and TINYNV_TAIL_RELEASE=0 brings them back.
  int tail_release;
  // TINYNV_INLINE_MAX: the largest caller upload that rides in the pushbuffer instead of going to the copy engine,
  // bytes, a dword multiple, at most submit.h's TINYNV_INLINE_MAX (the per-call ceiling). 4,096 by default.
  uint32_t inline_max;
  // TINYNV_QMD_MEMBAR=sys|gpu|none and TINYNV_QMD_INVALIDATE=all|cb0|none: what every NON-releasing descriptor makes
  // the engine do at its tail (qmd.c). Measurement knobs, the oracle's choice by default; a descriptor that releases
  // the timeline keeps the system-scope barrier whatever these say (tinynv_exec_flush, tinynv_exec_run).
  int qmd_membar, qmd_invalidate;
  int download_sync_first;
  int inline_upload;
  // How many uploads took each path. Without these, "the knob did nothing" and "the knob was never in effect" are the
  // same observation.
  uint64_t inline_n, inline_bytes, upload_ce_n, inline_batches_saved, inline_forced_n, inline_rode_flush_n;
  // Small uploads waiting to ride in the next batch rather than each taking a batch, a submit and an idle of their
  // own. Session A measured the round trip at ~50 us a copy and six copies a token; the bytes are copied out of the
  // caller's buffer here, so the caller may reuse it the moment the call returns and nothing has to reach the card
  // until something else does.
// The room the list CAN use, and the room it is allowed to use. They are separate so the limit is a runtime knob
// against a fixed allocation: sweeping it costs a card run rather than a rebuild, and the number that ends up in the
// default comes off a curve instead of out of my arithmetic. 128 entries and 64 KB is 66 KB per device, once.
#define TINYNV_INLINE_PEND_CAP_N     128
#define TINYNV_INLINE_PEND_CAP_BYTES 65536
  struct { uint64_t dst; uint32_t off, len; } inline_pend[TINYNV_INLINE_PEND_CAP_N];
  unsigned inline_pend_n, inline_pend_max;
  uint32_t inline_pend_bytes, inline_pend_bytes_max;
  uint8_t inline_pend_buf[TINYNV_INLINE_PEND_CAP_BYTES];
  // Whether anything outstanding requires the engine to finish. Default "yes" for everything submitted; an inline
  // upload is the one exception and says so at the point it takes it.
  int needs_wait, submitting_inline;
  // "The engine stood still and has not been given work since." Set by any blocked wait, cleared when the next compute
  // batch is begun. Unlike refill_pending this is not profile-gated: a mode that only works while measuring is not a
  // mode. See the short first chain in tinynv_exec_run.
  int after_stall;
  // Set between a chain flush capturing the value its descriptors will release and handing the chain over. Nothing may
  // take a compute timeline value in that window. See tinynv_exec_flush.
  int in_chain_flush;
  int hybrid_delivery;
  int delta_delivery;   // TINYNV_DELTA_DELIVERY: diff AR_DESC against last_delivered, patch only what changed (default on)
  uint32_t delta_gap;   // TINYNV_DELTA_GAP: unchanged bytes two differing runs may straddle and still go as one patch
  uint32_t delta_aggregate_max;   // TINYNV_DELTA_AGGREGATE_KB, in bytes: a flush's patches' pushbuffer footprint,
                                  // headers included, past which the flush takes the full delivery instead
  tinynv_delta_span_t delta_span[TINYNV_DELTA_SPANS_MAX];   // the flush being built: its patches, in launch order
  unsigned short_chain;
  uint64_t hybrid_n;
  uint64_t delta_n, delta_bytes, delta_skip_n, delta_full_n;
  // Sub-launch accounting: patches (inline-upload calls) made, their pushbuffer footprint headers included, and
  // launches whose runs cost more than their one envelope - or overflowed their room - and went out as that instead.
  uint64_t delta_span_n, delta_pb_bytes, delta_collapse_n;
  // TINYNV_DELTA_REWIND=1 (with delta delivery): start the descriptor region from the bottom at the first launch after
  // a standstill, so each token's launches land where the previous token's did and the diff is a launch against its
  // own previous-token self. See tinynv_exec_run.
  int delta_rewind;
  uint64_t delta_rewind_n;
  // Launch every descriptor from the command stream and link none of them, so nothing orders one kernel against the
  // next. THIS PRODUCES WRONG ANSWERS for any real workload - dependent kernels run side by side - and exists for one
  // measurement: whether the submission path can sustain a launch a microsecond when nothing waits on anything. The
  // vendor dispatches in about one on this card and this driver's floor is near six, so the question is whether the
  // chain link's turnaround is the difference. Completion comes from a host-method release with wait-for-idle at the
  // end of the batch, the way it did before chaining, because with no links there is no last descriptor to trust.
  int no_chain_deps;
  // Where the processor's time goes per launch, accumulated when TINYNV_LAUNCH_PROFILE is set. The card is not the
  // limit on this path - a bench that timed issuing separately from finishing found them equal - so the whole
  // remaining per-launch cost is here, and only about a fifth of it is accounted for by anything measured offline.
  // Five probes chosen so the arithmetic closes: a launch minus a flush is the per-launch build, and a flush minus
  // the push, the submit and the poll is everything else a flush does.
  // How many times each region has come round, and how long the engine took to drain when it did. A wrap is the only
  // thing on this path that stalls: it waits for everything outstanding before handing the bytes back. It was silent,
  // which meant "no wrap line in the log" read as "no wraps" when it only ever meant "no wrap FAILURE" - the failures
  // are loud and the ordinary case said nothing at all. Counted always; reported at teardown only if it happened.
  uint64_t wraps[2], wrap_wait_ns, seg_waits;
  uint64_t nsubmit_kind[3];   // launch chains, other compute batches, copy-engine batches (always counted)
  // The last 128 submissions in order, never trimmed (the trail is): a token's worth, printed under the profile.
  struct { uint32_t dwords; uint8_t copy, links; } hist[128];
  uint32_t nhist;
  // Deliver the descriptor region by copy engine instead of by processor stores. The bytes are the whole per-launch
  // cost - 1.5 KB a launch at about 235 MB/s of register-write bandwidth, which is 6.3 us of the 7 - and the engine
  // reads host memory far faster than the processor writes video memory through the window. The mirror is a host
  // allocation the engine can read, laid out at the same offsets as the arena so a span copies to itself; its reuse is
  // governed by the arena's own wrap discipline, which already waits for everything outstanding, so no separate
  // lifetime tracking is needed and none is done.
  int arena_dma;
  tinynv_vmap_t mirror;
  int profile;
  uint64_t prof_ns[8], prof_n[8];
  // TINYNV_KERNEL_PROFILE: the engine's clock at every kernel's completion, by kernel. Each descriptor releases a
  // four-word report - its timeline value, then the clock - into a slot of its own in `kring`, host memory the engine
  // writes and the processor reads. The chain's tail is stamped by a command-stream release with wait-for-idle into
  // its slot instead, so nothing here depends on a descriptor's second release slot, which has never been shown to
  // fire (tinynv_exec_flush). Slots are read in order once the timeline has passed them, and the interval from one
  // stamp to the next is attributed to the later kernel: its own execution plus the dispatch gap before it, which no
  // completion stamp can separate. Off unless asked: it is a measurement, and the report at teardown is what it is
  // for. See kprof_harvest and kprof_report in exec.c.
  int kprof;                    // asked for
  int kprof_on;                 // asked for AND the chain shape allows it: tail release, chained descriptors
  int kprof_skip;               // TINYNV_KERNEL_PROFILE_SKIP: windows not counted at the start (load, prompt, warm-up)
  tinynv_vmap_t kring;          // TINYNV_KPROF_SLOTS x 16 bytes
  tinynv_kprof_slot_t *kslot;   // what each ring slot was given to
  uint32_t kwrite, kharvest;    // ring cursors: the next slot to give out, the oldest not yet read
  uint64_t kprev_clock;         // the previous stamp read, which the next interval starts from
  int kprev_valid, kgap_break;  // whether it is one; whether a launch since went unstamped
  int kboundary_pending;        // a wait returned with the compute queue fully retired: the next launch is a boundary
  uint64_t kwindows;            // host waits that retired stamped work; the first kprof_skip are not counted
  uint64_t kstamped, kmissing, klost, kskipped, kbackwards;
  uint64_t kboundary_ns, kboundary_n, kfirst_n, kfirst_ns, kattr_ns, kattr_n, kunnamed_ns, kunnamed_n;
  // Intervals of 50 us and more that are not the boundary: a stall inside a token. Split by whether the launch was
  // its chain's first (the engine waited for the host to hand the chain over) or not (something inside a chain).
#define TINYNV_KPROF_LONG_NS 50000ull
#define TINYNV_KPROF_TOP 8
  uint64_t klong_n[2], klong_ns[2];
  struct { uint64_t ns; int32_t row; uint8_t first; } ktop[TINYNV_KPROF_TOP];   // the longest, by length
  tinynv_kprof_row_t *krow;
  int nkrow;
  int32_t *kindex;              // open addressing over the descriptor's address -> row
  // TINYNV_KEEPALIVE: a lent kernel keeps the engine scheduled across a synchronisation. The flag it polls lives in
  // ka_flag (host memory); armed (flag 0, pending) before the launch, released (flag 1) at the next chain's hand-over or
  // at any wait, whichever comes first. See tinynv_exec_keepalive_arm/release and tinynv_stream_sync in tinynv.c.
  // TINYNV_RING_ASYNC: an announcement's fence read is sent with the batch and its reply taken - and the doorbells rung
  // - sixteen launches into the next chain, or at the next wait or announcement, whichever comes first. The round
  // trip (~25 us to the server process) then overlaps descriptor building instead of blocking it.
  int ring_async;
  // TINYNV_DTOD_VIA_COMPUTE: serve device-to-device copies with the lent copy kernel so they join the launch chain
  // instead of handing it over. See dtod_by_kernel in tinynv.c for the measurement that motivated it.
  int dtod_via_compute;
  int graph_resident;           // TINYNV_GRAPH_RESIDENT: record a caller's token and replay it as one chain
  int graph_pace;               // TINYNV_GRAPH_PACE: a measurement knob, hand chains over one ahead instead of all at once
  // Under the launch profile, every resident chain's batch ends with a clocked release into a slot of its own, and the
  // next wait reads them back: chain k's stamp minus chain k-1's is that chain plus the hand-over before it.
  uint64_t graph_last_at0;
  int graph_last_n;
  uint64_t graph_chain_ns[TINYNV_GRAPH_CHAINS], graph_chain_n[TINYNV_GRAPH_CHAINS];
  tinynv_graph_rec_t *rec;      // the token being recorded, while one is
  // Regions of recordings that were let go, kept for the next recording: mapping 16 MB of host memory for the engine
  // and the same of video memory costs tens of milliseconds, which a bench that re-captures every run paid five times.
  struct { tinynv_vmap_t mem, mirror; } graph_pool[4];
  int graph_pool_n;
  uint64_t graph_records, graph_replays;
  int torn_down;                // set by the owner before the card is put down: recordings freed after that touch nothing
  int keepalive;                // asked for (TINYNV_KEEPALIVE)
  unsigned keepalive_us;        // TINYNV_KEEPALIVE_US: the spin's ceiling, the watchdog
  unsigned keepalive_min_kb;    // TINYNV_KEEPALIVE_MIN_KB: only a download at least this large is a token boundary
  tinynv_vmap_t ka_flag;
  int ka_pending;
  uint64_t ka_value;            // the pending keep-alive's own timeline value: a wait that needs no more than it ignores it
  // Caller launches since the last keep-alive. A synchronisation after fewer than TINYNV_KEEPALIVE_MIN_LAUNCHES of them
  // is one of the ~24 small read-backs a token, not the boundary; a keep-alive after each of those cost 20%.
#define TINYNV_KEEPALIVE_MIN_LAUNCHES 64
  uint64_t launches_since_ka;
  uint64_t ka_launched, ka_released_chain, ka_released_wait;
  // The engine's own clock across a synchronisation: how long it sat idle between finishing one batch and starting the
  // next. Host profiling cannot see this - the host is busy building at the time - and it is where the residual against
  // the vendor driver has to be hiding.
  uint64_t refill_ns, refill_n, refill_max_ns, refill_bad;
#define TINYNV_REFILL_RAW 16
  // tail, head, the report's payload beside the tail, and the timeline value the host waited for. Two windows: the
  // first half from the start of the run, the second from refill_raw_from on, which is a decode rather than a prefill.
  uint64_t refill_raw[TINYNV_REFILL_RAW][7];
  unsigned refill_raw_n;
  uint64_t refill_raw_from;
  uint64_t refill_last_tail, refill_stuck;
  // How many stalls asked to be measured, how many got a head timestamp, and (refill_n + refill_bad) how many were
  // paired. A is right that ~410 refills against ~4,700 blocked waits means this measures a SUBSET of the stalls, and
  // a mean over a subset chosen by the mechanism is not the mean of the thing. These say how big the subset is.
  uint64_t refill_asked, refill_armed_n;
  // The same interval on the host's clock: from the engine going idle to the next batch being begun, and how much of
  // that this driver spent building launches. Answers whether the dry engine is the host being slow to come back.
  double refill_host_t0;
  uint64_t refill_build_t0, refill_host_ns, refill_build_ns, refill_host_n;
  // The far end of the host's half: when the refill batch was actually handed to the engine. The gpu-measured gap
  // minus this is what the engine took to start work that was already on its ring.
  double refill_submit_from;
  int refill_submit_open;
  uint64_t refill_submit_ns, refill_submit_n;
  // The gpu-side split of the gap: engine-idle to engine-reached-the-batch, and reached to allowed-to-proceed. The
  // second is the cross-queue acquire, which on the launch path is the descriptor copy.
  uint64_t refill_reach_ns, refill_acq_ns, refill_reach_n;
  // The copy engine's clock at the instant the host's wait returned: the token's last copy, which on a decode is the
  // logits coming back. Sits inside the compute-side gap and is not latency.
  uint64_t refill_copy_from, refill_copy_ns, refill_copy_n;
  // The same interval split at the moment the copy engine actually began: late to start, or slow to run.
  uint64_t refill_cstart, refill_cwait_ns, refill_crun_ns, refill_cstart_n;
  uint64_t refill_reach_raw;   // the pre-acquire stamp as read, kept so the raw table can show it
  // The previous refill's head, and the token period it gives. Immune to where tails are stamped.
  uint64_t refill_prev_head, refill_period_ns, refill_period_n;
  // The same, over the decode window alone. A mean over prefill and decode together is a mean over a mixed population.
  uint64_t refill_period_dec_ns, refill_period_dec_n;
  // Set while the driver runs work of its own - the download kernel - so its batch does not stamp a tail as though it
  // were the caller's last kernel.
  int internal_work;
  int refill_pending;   // the next batch out is the first after a standstill and should timestamp its head
  int refill_armed;     // a head timestamp is in flight and its pair can be read once that batch retires
  uint64_t refill_from; // when the batch before the standstill finished, snapshotted before the refill overwrites it
  uint32_t slm_per_thread;
} tinynv_exec_t;

int tinynv_exec_init(tinynv_gpu_t *g, tinynv_exec_t *ex);
void tinynv_exec_fini(tinynv_exec_t *ex);

// Resolve the two submission knobs into a mode, given their values (getenv results, or anything for a test) and
// returning through `why` the words the startup line reports. 1 is synchronous, 0 is asynchronous, and asynchronous is
// what an unset environment gets. Exposed rather than static because the decision is otherwise only observable on
// hardware: nothing reaches tinynv_exec_init without a booted card. See test_exec_mode.
int tinynv_exec_sync_mode(const char *async, const char *sync, const char **why);
int tinynv_exec_chain_depth(const char *e);
// 1 = only the chain's tail releases the timeline (the default since 2026-09-19); 0 = every descriptor does. Which
// one is selected decides which release the profile has to stamp, so it is resolved where it can be tested.
int tinynv_exec_tail_release(const char *e);
// TINYNV_DELTA_DELIVERY (1 = patch only what changed, the default since 2026-09-19; 0 = the full copy every flush)
// and TINYNV_DELTA_REWIND (1 = come round at every standstill, the default; 0 = only when the region runs out),
// the second following the first off. Resolved here so test_exec_mode pins what an empty environment gets.
int tinynv_exec_delta_delivery(const char *e);
int tinynv_exec_delta_rewind(int delivery, const char *e);
// TINYNV_QMD_MEMBAR: "sys" (the default) = TINYNV_QMD_MEMBAR_SYS, "gpu" = _GPU, "none" = _NONE; anything else is the
// default. TINYNV_QMD_INVALIDATE: "all" (the default), "cb0", "none". Resolved here so test_exec_mode pins them.
int tinynv_exec_qmd_membar(const char *e);
int tinynv_exec_qmd_invalidate(const char *e);
// TINYNV_INLINE_MAX=<bytes>: unset, empty, unreadable or under 4 is the 4,096-byte default; rounded down to a dword
// multiple and clamped to the per-call ceiling. Resolved here so test_exec_mode pins the default.
uint32_t tinynv_exec_inline_max(const char *e);
// 0 = a download issues its copy immediately and the card waits (the default); 1 = the host waits for compute first,
// so the copy's acquire is satisfied before the channel is ever looked at.
int tinynv_exec_download_sync_first(const char *e);
// TINYNV_KERNEL_PROFILE and TINYNV_KERNEL_PROFILE_SKIP, resolved where test_exec_mode can drive them: off unless
// asked, and four windows not counted unless asked.
int tinynv_exec_kernel_profile(const char *e);
// TINYNV_KEEPALIVE and TINYNV_KEEPALIVE_US: off unless asked; 2,000 us of spin at most unless asked.
int tinynv_exec_ring_async(const char *e);
int tinynv_exec_dtod_via_compute(const char *e);
int tinynv_exec_graph_resident(const char *e);
// Record a token (1 = the driver will not record: the knob is off or the chain shape is wrong), seal and run it
// once, run it again, let it go.
int tinynv_exec_graph_begin(tinynv_exec_t *ex);
int tinynv_exec_graph_end(tinynv_exec_t *ex, tinynv_graph_rec_t **out);
int tinynv_exec_graph_launch(tinynv_exec_t *ex, tinynv_graph_rec_t *r);
void tinynv_exec_graph_free(tinynv_exec_t *ex, tinynv_graph_rec_t *r);
int tinynv_exec_keepalive(const char *e);
unsigned tinynv_exec_keepalive_us(const char *e);
// TINYNV_KEEPALIVE_MIN_KB: the download size from which the keep-alive is launched; 64 unless asked. A decode reads
// back ~24 small tensors a token and the logits once (~1 MB); a keep-alive after each small one cost 20%.
unsigned tinynv_exec_keepalive_min_kb(const char *e);
// Arm the keep-alive (flag to 0, pending) and say where the flag is; release it (flag to 1). Release is called at
// every wait and at every caller chain's hand-over, and is free when nothing is pending.
int tinynv_exec_keepalive_arm(tinynv_exec_t *ex, uint64_t *flag_va);
void tinynv_exec_keepalive_release(tinynv_exec_t *ex, int by_wait);
int tinynv_exec_kernel_profile_skip(const char *e);
// A kernel's mangled name reduced to what a reader wants: the function and its literal template arguments,
// "mul_mat_vec_q<(ggml_type)12, 1, 1>" for what nvcc emits. Anything it cannot read is "...". Returns out.
const char *tinynv_kernel_short_name(const char *mangled, char *out, size_t cap);
// 0 = every host-to-device copy goes to the copy engine (the default); 1 = small aligned ones ride in the pushbuffer.
int tinynv_exec_inline_upload_mode(const char *e);
// Launches in the first chain after a standstill; 0 = off, every chain full depth. See TINYNV_SHORT_FIRST_CHAIN.
unsigned tinynv_exec_short_chain(const char *e);
// How many small uploads are held before one must be pushed out. Default 16; TINYNV_INLINE_PEND sweeps it.
unsigned tinynv_exec_pend_entries(const char *e);
// TINYNV_DELTA_GAP=<bytes>: how much unchanged content two differing runs in one launch's span may straddle and still
// be sent as one patch. Default 32, rounded down to a dword multiple; 0 is a value (only contiguous dwords merge).
uint32_t tinynv_exec_delta_gap(const char *e);
// TINYNV_DELTA_AGGREGATE_KB=<kb>: the pushbuffer footprint, headers included, a flush's delta patches may reach before
// the flush falls back to a full delivery. Default 128 KB, at most 256. Returns bytes.
uint32_t tinynv_exec_delta_aggregate(const char *e);
// The runs of dwords in [lo,hi) where `fresh` differs from `last`, two runs merged when at most `gap` bytes of unchanged
// content separate them, written to out[]. Returns how many, or -1 when more than `max` would be needed; either way
// *envelope is the span from the first differing dword to the last (empty, hi <= lo, when nothing differs). lo and hi
// are dword multiples, and so is every run and the envelope. Exposed for test_delta.
int tinynv_delta_runs(const uint8_t *fresh, const uint8_t *last, uint32_t lo, uint32_t hi, uint32_t gap,
                      tinynv_delta_span_t *out, unsigned max, tinynv_delta_span_t *envelope);
int tinynv_exec_arena_dma(const char *e, int arena_vram, const char **why);

// Wait until the engine has released `value`, or give up. Returns -1 on timeout, which is a fault rather than slowness:
// the engine either ran the batch or faulted, and a batch that never completes means the second.
int tinynv_exec_wait(tinynv_exec_t *ex, uint64_t value, double seconds);

// Wait for everything submitted so far. The caller needs this before it reads a result, before it takes away memory the
// engine might still be reading, and nowhere else - which is the whole point of submitting without waiting.
int tinynv_exec_idle(tinynv_exec_t *ex);
int tinynv_exec_needs_wait(const tinynv_exec_t *ex);
void tinynv_exec_set_internal(tinynv_exec_t *ex, int on);
uint64_t tinynv_exec_stage_va(const tinynv_exec_t *ex);
void *tinynv_exec_stage_host(const tinynv_exec_t *ex);
size_t tinynv_exec_stage_bytes(void);
// Hand over any launches that are built but not yet submitted, without waiting for them. Anything that reads the
// timeline, or records a value from it, has to do this first: until a chain is handed over its values are promises
// nothing has been told to keep.
int tinynv_exec_flush(tinynv_exec_t *ex);

// A copy, device address to device address. Host memory reaches it by being staged first; see tinynv_exec_upload.
int tinynv_exec_copy(tinynv_exec_t *ex, uint64_t dst_va, uint64_t src_va, uint64_t bytes);
int tinynv_exec_zero(tinynv_exec_t *ex, uint64_t va, uint64_t bytes);
int tinynv_exec_upload(tinynv_exec_t *ex, uint64_t dst_va, const void *src, size_t n);
int tinynv_exec_download(tinynv_exec_t *ex, void *dst, uint64_t src_va, size_t n);

// A cubin's loadable image, in device memory, plus where everything landed inside it.
typedef struct {
  tinynv_vmap_t mem;      // device memory holding the image
  tinynv_image_t layout;  // offsets within it: the code, each constant bank
} tinynv_exec_module_t;

int tinynv_exec_load(tinynv_exec_t *ex, const tinynv_cubin_t *cb, const char *kernel, tinynv_exec_module_t *out);
void tinynv_exec_unload(tinynv_mm_t *mm, tinynv_exec_module_t *m);

// Run one kernel: build its descriptor, put its parameters where it will look for them, and launch it.
int tinynv_exec_run(tinynv_exec_t *ex, tinynv_exec_module_t *m, const tinynv_kernel_desc_t *k, const uint32_t grid[3],
                    const uint32_t block[3], uint32_t dyn_smem, const void *params, size_t params_len);

// A kernel, its descriptor already written. `qmd_va` is where that descriptor lives; the engine is told and told to go.
int tinynv_exec_launch(tinynv_exec_t *ex, uint64_t qmd_va);

// Enough local memory for a kernel that needs this much per thread, grown when a hungrier kernel turns up.
int tinynv_exec_ensure_local_memory(tinynv_exec_t *ex, uint32_t per_thread);

#endif
