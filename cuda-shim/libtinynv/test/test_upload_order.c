// An upload never lands before a launch that precedes it (H4's address-0 fault, 2026-09-21), checked without a card.
//
// The REAL exec.c (its batching, its flush, its held-upload list) runs against recording stand-ins for the command
// encoders and the submit: each records what it would have put in the command stream, and each submit closes a batch.
// Everything else exec.c can reach stops the test with its own name (fake_exec_unreachable.c). The launches are placed
// in the pending chain directly, as tinynv_exec_run would place them: what is under test is what happens AFTER that.
//
// Arms, each a guest order and the order the command streams must preserve:
//   mid-chain upload   launch A, small upload U, launch B, flush -> U after A's batch, before B, and behind a
//                      same-queue wait for A (the H4 shape: U landed before A, and A overwrote it)
//   held, then a copy  launch A, flush, small upload U, a copy -> the batch carrying U waits for A before U
//   nothing pending    launch A, flush, small upload U, launch B, flush -> U rides in B's batch, ahead of B, behind
//                      the wait for A: no batch of its own (the inline fast path, which the fix must keep)
//   large upload       launch A, an upload of TINYNV_INLINE_MAX + 4 bytes, launch B -> A's batch, then the copy
//                      batch waiting for A, then B's batch waiting for the copy: unchanged by the fix, and checked
//                      so that the fix is seen not to have moved it
//
// PRE-REGISTERED (B, before the first run): against exec.c before the fix (e2d6c5b), "mid-chain upload" puts U before
// A in the one batch the chain goes out as, "held, then a copy" emits U with no wait for A, and "nothing pending" writes
// U ahead of the wait for A. All three fail; "large upload" passes on both builds.
#include "exec.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

// ---- the record ----
enum { EV_WAIT, EV_INLINE, EV_LAUNCH, EV_BARRIER, EV_RELEASE, EV_COPY, EV_SUBMIT };
static const char *EVN[] = {"wait", "inline", "launch", "barrier", "release", "copy", "submit"};
static struct { int kind, batch; uint64_t a, b; } ev[512];
static int nev, batch;
static uint64_t sem_base;
static void rec(int kind, uint64_t a, uint64_t b) { if (nev < 512) { ev[nev].kind = kind; ev[nev].batch = batch; ev[nev].a = a; ev[nev].b = b; nev++; } }
static int take(tinynv_cmdbuf_t *c, uint32_t dwords) {
  if (c->n + dwords > c->cap) return tinynv_fail("the batch was sized for %u dwords and needs %u", c->cap, c->n + dwords);
  c->n += dwords;
  return 0;
}

// ---- recording stand-ins, with the real dword counts (submit.c) so a batch sized too small is refused ----
// Each count is the encoder's own: a method is one header word plus its data words (tinynv_cmd_method). G's review
// (20260921-131013, finding 7) found two wrong here: a release is 8, not 6 (the semaphore's 6 plus a 2-word non-stall
// interrupt, tinynv_cmd_release_ts), and a copy is 9 per 2 GiB step, not 20 (5 + 2 + 2, tinynv_cmd_copy).
int tinynv_cmd_wait(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value) { rec(EV_WAIT, (addr - sem_base) / 128, value); return take(c, 6); }
int tinynv_cmd_inline_upload(tinynv_cmdbuf_t *c, uint64_t dst, const void *src, uint32_t n) { (void)src; rec(EV_INLINE, dst, n); return take(c, TINYNV_INLINE_DWORDS(n)); }
int tinynv_cmd_launch(tinynv_cmdbuf_t *c, uint64_t qmd_addr) { rec(EV_LAUNCH, qmd_addr, 0); return take(c, 4); }
int tinynv_cmd_memory_barrier(tinynv_cmdbuf_t *c) { rec(EV_BARRIER, 0, 0); return take(c, 2); }
int tinynv_cmd_release(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value) { rec(EV_RELEASE, (addr - sem_base) / 128, value); return take(c, 8); }
int tinynv_cmd_release_clocked(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value) { return tinynv_cmd_release(c, addr, value); }
int tinynv_cmd_copy_release(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value) { rec(EV_RELEASE, (addr - sem_base) / 128, value); return take(c, 6); }
int tinynv_cmd_copy(tinynv_cmdbuf_t *c, uint64_t dst, uint64_t src, uint64_t bytes) {
  (void)src; rec(EV_COPY, dst, bytes);
  return take(c, 9u * (uint32_t)((bytes + (1ull << 31) - 1) >> 31));
}
int tinynv_cmd_timestamp(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value) { (void)addr; (void)value; return take(c, 6); }
int tinynv_cmd_copy_timestamp(tinynv_cmdbuf_t *c, uint64_t addr, uint64_t value) { (void)addr; (void)value; return take(c, 6); }
static tinynv_exec_t *cur;
// A batch "retires" the moment it is submitted: its releases land in the semaphore memory, so a path that waits (the
// large upload idles after each chunk) completes. Order is judged from the record, never from these values.
static uint8_t sem_host[4096] __attribute__((aligned(128)));
static struct { uint64_t va, value; } placed[64];
static int nplaced;
int tinynv_submit(tinynv_gpu_t *g, tinynv_queue_t *q, uint64_t cmdbuf_va, uint32_t dwords) {
  (void)cmdbuf_va; (void)dwords;
  // A chain goes out as ONE launch method for its head; the rest ride the descriptors, each linked to the next, and run
  // after the head in chain order. They are recorded here, straight after the head, as links (b = 1), so the record
  // holds every kernel in the order the engine runs them and not only the ones a launch method names.
  for (int i = 0; i < nev && cur && cur->flush_n > 1; i++)
    if (ev[i].batch == batch && ev[i].kind == EV_LAUNCH && ev[i].b == 0 && ev[i].a == cur->chain[0].va) {
      int links = cur->flush_n - 1;
      if (nev + links > 512) break;
      memmove(&ev[i + 1 + links], &ev[i + 1], (size_t)(nev - i - 1) * sizeof ev[0]);
      for (int k = 0; k < links; k++) { ev[i + 1 + k].kind = EV_LAUNCH; ev[i + 1 + k].batch = batch; ev[i + 1 + k].a = cur->chain[1 + k].va; ev[i + 1 + k].b = 1; }
      nev += links;
      break;
    }
  for (int i = 0; i < nev; i++) {
    if (ev[i].batch != batch) continue;
    if (ev[i].kind == EV_RELEASE && ev[i].a < 2) memcpy(sem_host + ev[i].a * 128, &ev[i].b, 8);
    // a launched descriptor releases its own value on the compute slot (exec.c's chain); the test placed it, so it
    // knows the value
    if (ev[i].kind == EV_LAUNCH) for (int k = 0; k < nplaced; k++) if (placed[k].va == ev[i].a) memcpy(sem_host, &placed[k].value, 8);
  }
  rec(EV_SUBMIT, q == &g->gsp.copy_q, 0); batch++; return 0;
}
int tinynv_submit_ring(tinynv_gpu_t *g) { (void)g; return 0; }
int tinynv_submit_stage(tinynv_gpu_t *g, tinynv_queue_t *q, uint64_t cmdbuf_va, uint32_t dwords) { return tinynv_submit(g, q, cmdbuf_va, dwords); }
uint64_t tinynv_arena_place(const tinynv_exec_region_t *rg, uint64_t bytes, uint64_t align, int *wrap) {
  (void)align; *wrap = 0; uint64_t off = rg->next; ((tinynv_exec_region_t *)rg)->next += (bytes + 255) & ~255ull; return off;
}
uint64_t tinynv_arena_enter(tinynv_exec_region_t *rg, uint64_t off, uint64_t bytes) { (void)rg; (void)off; (void)bytes; return 0; }
void tinynv_arena_take(tinynv_exec_region_t *rg, uint64_t off, uint64_t bytes) { (void)rg; (void)off; (void)bytes; }
void tinynv_arena_submitted(tinynv_exec_region_t *rg, uint64_t value) { (void)rg; (void)value; }
void tinynv_arena_mark(tinynv_exec_region_t *rg, uint64_t lo, uint64_t hi, uint64_t value) { (void)rg; (void)lo; (void)hi; (void)value; }
int tinynv_arena_rewind(tinynv_exec_region_t *rg) { (void)rg; return 0; }
int tinynv_qmd_link_check(const tinynv_qmd_t *q, int releases, uint64_t want_payload, uint64_t next_va, const char **why) {
  (void)q; (void)releases; (void)want_payload; (void)next_va; (void)why; return 0;
}
// the membar knob patches a chain tail's descriptor at handover (TINYNV_QMD_MEMBAR), and a former tail's release is
// cleared when its chain grows; the record this test judges holds encoder calls and batches, never descriptor bytes,
// so both stand-ins only report success
int tinynv_qmd_membar(tinynv_qmd_t *q, int membar) { (void)q; (void)membar; return 0; }
int tinynv_qmd_release_clear(tinynv_qmd_t *q) { (void)q; return 0; }
int tinynv_gsp_poll(tinynv_gpu_t *g) { (void)g; return 0; }
void tinynv_note_host_write(const void *dst, size_t n, const char *what) { (void)dst; (void)n; (void)what; }
void tinynv_note_host_region(const void *base, size_t n, const char *what) { (void)base; (void)n; (void)what; }

// ---- an exec with no card behind it ----
static uint8_t cmd_shadow[1 << 20], stage_buf[1 << 16], qmd_host[TINYNV_EXEC_CHAIN_MAX][TINYNV_QMD_BYTES];
static tinynv_gpu_t *gpu;
static tinynv_exec_t *fresh(void) {
  static tinynv_exec_t ex;
  memset(&ex, 0, sizeof ex);
  if (!gpu) gpu = calloc(1, sizeof *gpu);
  ex.g = gpu;
  ex.sem.va = sem_base = 0x30000000ull;
  memset(sem_host, 0, sizeof sem_host); ex.sem.dma.view.ptr = (volatile uint32_t *)sem_host; ex.sem.dma.view.size = sizeof sem_host;
  ex.region[1].shadow = cmd_shadow; ex.region[1].mem.va = 0x10000000ull; ex.region[1].size = sizeof cmd_shadow;
  ex.stage.dma.va = stage_buf; ex.stage.va = 0x20000000ull;
  ex.inline_upload = 1; ex.inline_pend_max = 16; ex.inline_pend_bytes_max = 8192;
  // this driver's inline ceiling is a RUNTIME knob (TINYNV_INLINE_MAX env), not the submit.h per-call constant;
  // an exec built by hand gets the same default an unset environment gets, or every arm below rides the bulk path
  ex.inline_max = 4096;
  nev = 0; batch = 0; nplaced = 0; cur = &ex;
  return &ex;
}
// a launch as tinynv_exec_run leaves it: one more descriptor in the pending chain, one more timeline value reserved
static void launch(tinynv_exec_t *ex, uint64_t qmd_va) {
  memset(&ex->chain[ex->nchain], 0, sizeof ex->chain[ex->nchain]);
  ex->chain[ex->nchain].va = qmd_va;
  ex->chain[ex->nchain].host = qmd_host[ex->nchain];
  ex->nchain++;
  ex->reserved++;
  placed[nplaced].va = qmd_va; placed[nplaced++].value = ex->reserved;
}
static int find(int kind, uint64_t a) { for (int i = 0; i < nev; i++) if (ev[i].kind == kind && ev[i].a == a) return i; return -1; }
static void show(const char *arm) {
  printf("  %s:", arm);
  for (int i = 0; i < nev; i++) {
    if (ev[i].kind == EV_SUBMIT) printf(" | %s", ev[i].a ? "copy-batch" : "compute-batch");
    else if (ev[i].kind == EV_WAIT) printf(" wait(%s>=%llu)", ev[i].a ? "copy" : "compute", (unsigned long long)ev[i].b);
    else if (ev[i].kind == EV_RELEASE) printf(" release(%llu)", (unsigned long long)ev[i].b);
    else if (ev[i].kind == EV_BARRIER) continue;
    else if (ev[i].kind == EV_LAUNCH && ev[i].b) printf(" link(%#llx)", (unsigned long long)ev[i].a);
    else printf(" %s(%#llx)", EVN[ev[i].kind], (unsigned long long)ev[i].a);
  }
  printf("\n");
}
// the wait for `value` on the compute queue that orders event `at`: emitted earlier in the same batch
static int waited_before(int at, uint64_t value) {
  for (int i = at - 1; i >= 0 && ev[i].batch == ev[at].batch; i--) if (ev[i].kind == EV_WAIT && ev[i].a == 0 && ev[i].b >= value) return 1;
  return 0;
}

int main(void) {
  const uint64_t A = 0xA000, B = 0xB000, U = 0x5000;
  // sized for the largest arm below (TINYNV_INLINE_MAX + 4 bytes): the large-upload arm memcpys the whole length
  // from this buffer, and at this tree's 32,764-byte ceiling a 64-dword array put the read past the stack guard page
  static uint32_t data[(TINYNV_INLINE_MAX + 4) / 4];

  { // mid-chain upload: the H4 shape
    tinynv_exec_t *ex = fresh();
    launch(ex, A);
    CHECK(!tinynv_exec_upload(ex, U, data, 16), "mid-chain: upload: %s", tinynv_last_error());
    launch(ex, B);
    CHECK(!tinynv_exec_flush(ex), "mid-chain: flush: %s", tinynv_last_error());
    show("mid-chain upload");
    int a = find(EV_LAUNCH, A), u = find(EV_INLINE, U), b = find(EV_LAUNCH, B);
    CHECK(a >= 0 && u >= 0 && b >= 0, "mid-chain: a launch or the upload is missing from the record (%d %d %d)", a, u, b);
    CHECK(a >= 0 && u > a && ev[u].batch > ev[a].batch, "mid-chain: the upload lands before launch A, which the guest issued first");
    CHECK(u >= 0 && b > u, "mid-chain: the upload lands after launch B, which the guest issued after it");
    CHECK(u >= 0 && waited_before(u, 1), "mid-chain: the upload is not behind a same-queue wait for launch A (value 1)");
  }
  { // held, then pushed out by a copy
    tinynv_exec_t *ex = fresh();
    launch(ex, A);
    CHECK(!tinynv_exec_flush(ex), "held/copy: flush: %s", tinynv_last_error());
    CHECK(!tinynv_exec_upload(ex, U, data, 16), "held/copy: upload: %s", tinynv_last_error());
    CHECK(!tinynv_exec_copy(ex, 0x6000, 0x7000, 64), "held/copy: copy: %s", tinynv_last_error());
    show("held, then a copy");
    int u = find(EV_INLINE, U), cp = find(EV_COPY, 0x6000);
    CHECK(u >= 0 && cp > u, "held/copy: the held bytes do not go out before the copy");
    CHECK(u >= 0 && waited_before(u, 1), "held/copy: the batch carrying the held bytes does not wait for launch A (value 1) before writing them");
  }
  { // nothing pending: the upload still rides inline, in the batch of the launch that follows it (the fast path kept)
    tinynv_exec_t *ex = fresh();
    launch(ex, A);
    CHECK(!tinynv_exec_flush(ex), "inline kept: flush: %s", tinynv_last_error());
    int before = batch;
    CHECK(!tinynv_exec_upload(ex, U, data, 16), "inline kept: upload: %s", tinynv_last_error());
    CHECK(batch == before, "inline kept: an upload with no launch pending cost a batch of its own");
    launch(ex, B);
    CHECK(!tinynv_exec_flush(ex), "inline kept: flush: %s", tinynv_last_error());
    show("nothing pending");
    int u = find(EV_INLINE, U), b = find(EV_LAUNCH, B);
    CHECK(u >= 0 && b > u && ev[b].batch == ev[u].batch, "inline kept: the upload did not ride in launch B's batch, ahead of it");
    CHECK(u >= 0 && waited_before(u, 1), "inline kept: the upload is written before the same-queue wait for launch A (value 1)");
  }
  { // large upload: the copy-engine path, unchanged
    tinynv_exec_t *ex = fresh();
    launch(ex, A);
    CHECK(!tinynv_exec_upload(ex, U, data, TINYNV_INLINE_MAX + 4 > sizeof stage_buf ? 0 : TINYNV_INLINE_MAX + 4), "large: upload: %s", tinynv_last_error());
    launch(ex, B);
    CHECK(!tinynv_exec_flush(ex), "large: flush: %s", tinynv_last_error());
    show("large upload");
    int a = find(EV_LAUNCH, A), cp = find(EV_COPY, U), b = find(EV_LAUNCH, B);
    CHECK(a >= 0 && cp > a && b > cp, "large: the order is not A, the copy, B");
    int w = -1; for (int i = cp - 1; i >= 0 && ev[i].batch == ev[cp].batch; i--) if (ev[i].kind == EV_WAIT && ev[i].a == 0) w = i;
    CHECK(w >= 0, "large: the copy does not wait on the compute queue for launch A");
    int wb = -1; for (int i = b - 1; i >= 0 && ev[i].batch == ev[b].batch; i--) if (ev[i].kind == EV_WAIT && ev[i].a == 1) wb = i;
    CHECK(wb >= 0, "large: launch B does not wait on the copy queue for the copy");
  }
  printf(fails ? "%d of %d checks failed\n" : "upload order: all %d checks passed\n", fails ? fails : checks, checks);
  return fails ? 1 : 0;
}
