// Replay backend: the driver runs against a recorded boot instead of a GPU. Reads are answered with the bytes hardware gave
// the python driver; writes are compared against what it wrote. That makes the whole bring-up developable and regression
// testable with no card attached, which matters because GSP bring-up is the one part that can wedge real hardware.
//
// Matching is deliberately semantic, not positional (design doc 4b). Three things differ run to run even when the logic is
// identical: how many times a poll spins before hardware answers, RPC sequence numbers, and any address an allocator picks.
// So a read looks ahead for the same (resource, offset) rather than demanding the very next line, a differing repeat count
// is not a divergence, and the recorded addresses are handed back to the driver so its embedded pointers match by
// construction. What remains is logic, which is what we want the first divergence to be.
#include "internal.h"
#include "sha256.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPLAY_DEFAULT_WINDOW 256 // how far ahead a read may look for its match before calling it a divergence

typedef enum { OP_CFGR, OP_CFGW, OP_BARI, OP_MMIOR, OP_MMIOW, OP_MEMR, OP_MEMW, OP_DMA, OP_RESET, OP_OTHER } opkind_t;

typedef struct {
  opkind_t kind;
  int res;          // bar index, or dma chunk id, or -1
  uint64_t off, len, blob_off, hash, val, arg;
  int count;        // a coalesced read serves its bytes this many times
  int line;
} rop_t;

// one place the driver has read, and the last value it saw there
// One place the driver has read, the last value it saw there, and whether anything has been written to it since. The
// last is for a question the value test cannot answer on its own: a read the oracle does AFTER writing a place is a
// read-back that would catch a failed write on real hardware, and skipping one of those is not the same as skipping a
// surplus re-read of something already held. Under replay both look identical; on hardware they do not.
typedef struct { int res, used, written_since_read; uint64_t off, hash; size_t forgiven; } held_t;

typedef struct {
  rop_t *ops;
  size_t nops, cursor;
  int served;       // how many times the op at the cursor has already been served
  uint8_t *blob;
  size_t blob_len;
  size_t window;
  int submission_recorded; // the trace's own header says whether work submission was visible to the recorder
  int echo;         // print every matched operation, for finding where a driver and a recording part company
  // what happened
  size_t consumed, skipped, skipped_writes, divergences;
  size_t ops_done;  // distinct recorded operations finished, as against `consumed`, which counts each serving
  size_t ngaps, gap_ops;
  size_t forgiven_readbacks; // of the forgiven reads, how many were the only read-back of a write
  int forgiven[64], forgiven_res[64];
  uint64_t forgiven_off[64];
  unsigned char forgiven_is_bar_query[64];
  size_t nforgiven;
  // The last value the driver read at each place, with no bound on how many places are remembered. A recorded read it
  // stepped over is forgiven only when this says it already holds exactly what that read would have returned.
  //
  // Unbounded on purpose, which is session A's correction to an eight entry ring. The value equality is the whole of the
  // safety and is sufficient by itself: the recording attests that this place held this value at that moment, so if
  // anything had changed it, the recorded value would differ from what is held and the read would be a divergence. A
  // bound adds nothing except false divergences when a place falls off the end, which is exactly how the ring failed.
  held_t *held;
  size_t nheld, held_cap;
  int books_broken; // the identity below has already been reported once; do not report it on every call after that
  char first_divergence[256];
} replay_t;

static const char *kind_name(opkind_t k) {
  static const char *names[] = {"a config read", "a config write", "a window query", "a register read", "a register write",
                                "a memory read", "a memory write", "an allocation", "a reset", "something else"};
  return names[k];
}

static opkind_t kind_of(const char *s, size_t n) {
  if (!strncmp(s, "cfgr", n)) return OP_CFGR;
  if (!strncmp(s, "cfgw", n)) return OP_CFGW;
  if (!strncmp(s, "bari", n)) return OP_BARI;
  if (!strncmp(s, "mmior", n)) return OP_MMIOR;
  if (!strncmp(s, "mmiow", n)) return OP_MMIOW;
  if (!strncmp(s, "memr", n)) return OP_MEMR;
  if (!strncmp(s, "memw", n)) return OP_MEMW;
  if (!strncmp(s, "dma", n)) return OP_DMA;
  if (!strncmp(s, "reset", n)) return OP_RESET;
  return OP_OTHER;
}

// "bar:3" / "mem:12" -> 3 / 12
static int res_of(const char *tok) {
  const char *colon = strchr(tok, ':');
  return colon ? (int)strtol(colon + 1, NULL, 10) : -1;
}

static int parse_line(char *line, int lineno, rop_t *op) {
  char *save = NULL, *tok = strtok_r(line, " \t", &save);
  if (!tok || *tok == '#') return 0;
  memset(op, 0, sizeof(*op));
  op->line = lineno;
  op->count = 1;
  op->res = -1;
  op->kind = kind_of(tok, strlen(tok));
  if (op->kind == OP_OTHER) return 0;

  char *f[8];
  int n = 0;
  while (n < 8 && (f[n] = strtok_r(NULL, " \t", &save)) != NULL) {
    if (f[n][0] == 'x' && f[n][1] >= '1' && f[n][1] <= '9') { op->count = (int)strtol(f[n] + 1, NULL, 10); break; }
    n++;
  }
  switch (op->kind) {
    case OP_CFGR: // cfgr <off> <size> <val>
    case OP_CFGW: // cfgw <off> <size> <val>
      if (n < 3) return 0;
      op->off = strtoull(f[0], NULL, 0);
      op->len = strtoull(f[1], NULL, 0);
      op->val = strtoull(f[2], NULL, 0);
      return 1;
    case OP_BARI: // bari <bar> <base> <size>
      if (n < 3) return 0;
      op->res = (int)strtol(f[0], NULL, 0);
      op->val = strtoull(f[1], NULL, 0);
      op->len = strtoull(f[2], NULL, 0);
      return 1;
    case OP_MMIOR: // mmior bar:<n> <off> <len> <blob_off> <hash>
    case OP_MEMR:  // memr  mem:<n> <off> <len> <blob_off> <hash>
      if (n < 5) return 0;
      op->res = res_of(f[0]);
      op->off = strtoull(f[1], NULL, 0);
      op->len = strtoull(f[2], NULL, 0);
      op->blob_off = strtoull(f[3], NULL, 0);
      op->hash = strtoull(f[4], NULL, 0);
      return 1;
    case OP_MMIOW: // mmiow bar:<n> <off> <len> <hash> [hex]
    case OP_MEMW:
      if (n < 4) return 0;
      op->res = res_of(f[0]);
      op->off = strtoull(f[1], NULL, 0);
      op->len = strtoull(f[2], NULL, 0);
      op->hash = strtoull(f[3], NULL, 0);
      return 1;
    case OP_DMA: // dma mem:<n> <size> <npages> <p0>...
      if (n < 3) return 0;
      op->res = res_of(f[0]);
      op->len = strtoull(f[1], NULL, 0);
      op->arg = strtoull(f[2], NULL, 0);
      op->val = n > 3 ? strtoull(f[3], NULL, 0) : 0;
      return 1;
    case OP_RESET: return 1;
    default: return 0;
  }
}

// FNV-1a, the same the recorder used, so a payload can be checked without keeping it
static uint64_t fnv1a(const void *data, size_t n) {
  const uint8_t *p = data;
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}

static void diverge(replay_t *r, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  if (!r->divergences) vsnprintf(r->first_divergence, sizeof(r->first_divergence), fmt, ap);
  va_end(ap);
  r->divergences++;
}

static int is_write(opkind_t k) { return k == OP_MMIOW || k == OP_MEMW || k == OP_CFGW; }

// The cursor moves for exactly three reasons: an operation was finished, an operation was stepped over, or a declared gap
// jumped it. So the three counters must always add up to where the cursor is, and any code path that advances it without
// going through the accounting breaks that sum. This is session A's suggestion, and it exists because a path that did
// exactly that -- dma allocation matching by order, outside the matcher -- hid a missing-write hole until the totals
// happened to be compared by hand. Now the books are checked on every move instead of when someone notices.
static void audit(replay_t *r, const char *where) {
  if (r->books_broken) return;
  size_t want = r->ops_done + r->skipped + r->gap_ops;
  if (r->cursor == want) return;
  r->books_broken = 1;
  diverge(r, "the replay's books do not balance after %s: cursor %zu, but %zu finished + %zu stepped over + %zu gapped = %zu."
             " some path moved the cursor without accounting for it",
          where, r->cursor, r->ops_done, r->skipped, r->gap_ops, want);
}

// Look ahead for the next op of this kind on this resource and offset.
//
// Two things may be stepped over for free, and only two. A read of the very place we are about to read, which is a poll
// that span a different number of times before the value it was waiting for arrived. And a read whose place AND value
// the driver already holds, because it took a read of exactly that a moment ago: consulting something twice where the
// oracle consulted it three times is not a difference in what the driver knew.
//
// The value has to match for the second one, and that is the whole of its safety. Forgiving a skipped read by place
// alone would forgive skipping the read that saw a change, and a driver that stopped one read early would decide on
// stale state with nothing to catch it. Requiring the value means what was skipped is provably what is already held.
//
// Everything else is a divergence: a recorded write the driver never issued, and a recorded read at a place it never
// looked at all, which is a decision made on information it did not gather. The strictness earns its keep -- it is what
// turned "19 tolerated" into a specific finding, that the oracle reads the queue position once more than this driver.
// What an operation told the driver. Reads carry a hash of the bytes; asking where a window is carries its base and
// size. Both are answers the driver can already hold, which is what makes repeating them free.
static int is_query(opkind_t k) { return k == OP_MMIOR || k == OP_MEMR || k == OP_BARI; }
static uint64_t answer_of(const rop_t *op) { return op->kind == OP_BARI ? op->val ^ (op->len * 0x9e3779b97f4a7c15ull) : op->hash; }

static size_t held_slot(replay_t *r, int res, uint64_t off) {
  uint64_t h = ((uint64_t)res * 0x9e3779b97f4a7c15ull) ^ (off * 0xff51afd7ed558ccdull);
  size_t mask = r->held_cap - 1, i = (size_t)(h >> 32) & mask;
  while (r->held[i].used && (r->held[i].res != res || r->held[i].off != off)) i = (i + 1) & mask;
  return i;
}

static void held_grow(replay_t *r) {
  size_t old_cap = r->held_cap;
  held_t *old = r->held;
  r->held_cap = old_cap ? old_cap * 2 : 1024;
  r->held = calloc(r->held_cap, sizeof(*r->held));
  for (size_t i = 0; i < old_cap; i++)
    if (old[i].used) r->held[held_slot(r, old[i].res, old[i].off)] = old[i]; // rehashing moves places, it adds none
  free(old);
}

// does the driver already hold exactly this, place and value both?
static int already_held(replay_t *r, const rop_t *sk) {
  if (!r->held_cap) return 0;
  if (!is_query(sk->kind)) return 0;
  size_t i = held_slot(r, sk->res, sk->off);
  return r->held[i].used && r->held[i].hash == answer_of(sk);
}

static void remember(replay_t *r, const rop_t *op) {
  if (!is_query(op->kind) && !is_write(op->kind)) return;
  if (!r->held_cap || (r->nheld + 1) * 4 >= r->held_cap * 3) held_grow(r);
  size_t i = held_slot(r, op->res, op->off);
  if (!r->held[i].used) { r->held[i].used = 1; r->held[i].res = op->res; r->held[i].off = op->off; r->nheld++; }
  if (is_write(op->kind)) { r->held[i].written_since_read = 1; return; }
  r->held[i].written_since_read = 0;
  r->held[i].hash = answer_of(op);
}

static rop_t *take_matching(replay_t *r, opkind_t kind, int res, uint64_t off, int any_place) {
  audit(r, "arriving at an operation");
  size_t start_cursor = r->cursor;
  for (size_t i = r->cursor; i < r->nops && i < r->cursor + r->window; i++) {
    rop_t *op = &r->ops[i];
    if (op->kind != kind || (!any_place && (op->res != res || op->off != off))) continue;
    for (size_t j = r->cursor; j < i; j++) {
      rop_t *sk = &r->ops[j];
      if (is_write(sk->kind)) {
        r->skipped_writes++;
        diverge(r, "line %d: the recording writes %llu bytes to %s:%d at %#llx and we never did",
                sk->line, (unsigned long long)sk->len, sk->kind == OP_MEMW ? "mem" : sk->kind == OP_CFGW ? "cfg" : "bar",
                sk->res, (unsigned long long)sk->off);
      } else if (already_held(r, sk)) {
        // was this the driver's only chance to notice the write it just made had not taken?
        size_t slot = held_slot(r, sk->res, sk->off);
        if (r->held[slot].written_since_read) r->forgiven_readbacks++;
        r->held[slot].forgiven++;
        if (r->nforgiven < sizeof(r->forgiven) / sizeof(*r->forgiven)) {
          r->forgiven[r->nforgiven] = sk->line;
          r->forgiven_res[r->nforgiven] = sk->res;
          r->forgiven_off[r->nforgiven] = sk->off;
          r->forgiven_is_bar_query[r->nforgiven] = sk->kind == OP_BARI;
        }
        r->nforgiven++;
        continue;
      } else if (sk->kind != op->kind || sk->res != op->res || sk->off != op->off) {
        diverge(r, "line %d: the recording asks %s:%d at %#llx (%llu bytes) and we never did; it is not something the "
                   "driver already holds, so it is a difference in what it looked at",
                sk->line, sk->kind == OP_MEMR ? "mem" : "bar", sk->res, (unsigned long long)sk->off,
                (unsigned long long)sk->len);
      }
    }
    if (i != r->cursor) { r->skipped += i - r->cursor; r->served = 0; }
    r->cursor = i;
    r->consumed++;
    if (++r->served >= op->count) { r->cursor = i + 1; r->served = 0; r->ops_done++; } // a coalesced read is spent
    if (r->echo) fprintf(stderr, "    replay: %s %s:%d at %#llx matched line %d%s\n", kind_name(op->kind),
                         op->kind == OP_MEMR || op->kind == OP_MEMW ? "mem" : "bar", op->res, (unsigned long long)op->off,
                         op->line, i == start_cursor ? "" : " (after stepping over some)");
    remember(r, op);
    audit(r, "taking an operation");
    return op;
  }
  return NULL;
}

static rop_t *take(replay_t *r, opkind_t kind, int res, uint64_t off) { return take_matching(r, kind, res, off, 0); }

// Some operations are identified by their order rather than by an address: the driver asks for "another host buffer", not
// for a particular one. They still have to go through the same accounting, or a recorded write sitting just before one
// would be stepped over for free -- which is the hole this replaced.
static rop_t *take_next(replay_t *r, opkind_t kind) { return take_matching(r, kind, -1, 0, 1); }

static uint32_t rp_cfg_read(tinynv_pci_t *p, uint32_t off, uint32_t size) {
  replay_t *r = p->ctx;
  rop_t *op = take(r, OP_CFGR, -1, off);
  if (!op) { diverge(r, "no recorded config read of %#x (size %u) near line %d", off, size, (int)r->cursor); return 0xffffffff; }
  return (uint32_t)op->val;
}

static void rp_cfg_write(tinynv_pci_t *p, uint32_t off, uint32_t size, uint32_t val) {
  replay_t *r = p->ctx;
  rop_t *op = take(r, OP_CFGW, -1, off);
  if (!op) { diverge(r, "config write of %#x to %#x is not in the trace", val, off); return; }
  if (op->val != val) diverge(r, "line %d: config %#x was written %#llx, we wrote %#x", op->line, off, (unsigned long long)op->val, val);
}

static int rp_bar_info(tinynv_pci_t *p, int bar, uint64_t *base, uint64_t *size) {
  replay_t *r = p->ctx;
  rop_t *op = take(r, OP_BARI, bar, 0);
  if (!op) return tinynv_fail("replay: bar %d was never sized in the trace", bar);
  if (base) *base = op->val; // the recorded address, so the driver embeds the same one the trace did
  if (size) *size = op->len;
  return 0;
}

static void rp_read(tinynv_mmio_t *m, uint64_t off, void *dst, size_t n, opkind_t kind) {
  replay_t *r = m->ctx;
  rop_t *op = take(r, kind, m->bar, m->off + off);
  if (!op || op->blob_off + op->len > r->blob_len) {
    diverge(r, "no recorded read of %s:%d at %#llx (%zu bytes) near line %d, where the recording is doing %s",
            kind == OP_MEMR ? "mem" : "bar", m->bar, (unsigned long long)(m->off + off), n,
            r->cursor < r->nops ? r->ops[r->cursor].line : -1,
            r->cursor < r->nops ? kind_name(r->ops[r->cursor].kind) : "nothing");
    memset(dst, 0xff, n);
    return;
  }
  if (op->len != n) diverge(r, "line %d: recorded read is %llu bytes, we asked for %zu", op->line, (unsigned long long)op->len, n);
  memcpy(dst, r->blob + op->blob_off, n < op->len ? n : op->len);
}

static void rp_write(tinynv_mmio_t *m, uint64_t off, const void *src, size_t n, opkind_t kind) {
  replay_t *r = m->ctx;
  rop_t *op = take(r, kind, m->bar, m->off + off);
  if (!op) { diverge(r, "write to %s:%d at %#llx is not in the trace", kind == OP_MEMW ? "mem" : "bar", m->bar,
                     (unsigned long long)(m->off + off)); return; }
  uint64_t h = fnv1a(src, n);
  if (op->len != n) diverge(r, "line %d: recorded write is %llu bytes, we wrote %zu", op->line, (unsigned long long)op->len, n);
  else if (op->hash != h) diverge(r, "line %d: %s:%d at %#llx differs (recorded %#llx, ours %#llx)", op->line,
                                  kind == OP_MEMW ? "mem" : "bar", m->bar, (unsigned long long)(m->off + off),
                                  (unsigned long long)op->hash, (unsigned long long)h);
}

static uint32_t rp_mem_rd32(tinynv_mmio_t *m, uint64_t off) {
  uint32_t v = 0xffffffff;
  rp_read(m, off, &v, 4, OP_MEMR);
  return v;
}
static void rp_mem_wr32(tinynv_mmio_t *m, uint64_t off, uint32_t v) { rp_write(m, off, &v, 4, OP_MEMW); }
static void rp_mem_rd_block(tinynv_mmio_t *m, uint64_t off, void *dst, size_t n) { rp_read(m, off, dst, n, OP_MEMR); }
static void rp_mem_wr_block(tinynv_mmio_t *m, uint64_t off, const void *src, size_t n) { rp_write(m, off, src, n, OP_MEMW); }

static uint32_t rp_rd32(tinynv_mmio_t *m, uint64_t off) {
  uint32_t v = 0xffffffff;
  rp_read(m, off, &v, 4, OP_MMIOR);
  return v;
}
static void rp_wr32(tinynv_mmio_t *m, uint64_t off, uint32_t v) { rp_write(m, off, &v, 4, OP_MMIOW); }
static void rp_rd_block(tinynv_mmio_t *m, uint64_t off, void *dst, size_t n) { rp_read(m, off, dst, n, OP_MMIOR); }
static void rp_wr_block(tinynv_mmio_t *m, uint64_t off, const void *src, size_t n) { rp_write(m, off, src, n, OP_MMIOW); }

static int rp_bar_map(tinynv_pci_t *p, int bar, uint64_t off, uint64_t size, tinynv_mmio_t *out) {
  replay_t *r = p->ctx;
  uint64_t bar_size = 0;
  if (rp_bar_info(p, bar, NULL, &bar_size)) return -1;
  if (!size) size = bar_size - off;
  *out = (tinynv_mmio_t){.ptr = NULL, .ctx = r, .bar = bar, .off = off, .size = size,
                         .rd32 = rp_rd32, .wr32 = rp_wr32, .rd_block = rp_rd_block, .wr_block = rp_wr_block};
  return 0;
}

static void rp_bar_unmap(tinynv_pci_t *p, tinynv_mmio_t *m) { memset(m, 0, sizeof(*m)); }

static int rp_dma_alloc(tinynv_pci_t *p, size_t size, tinynv_dma_t *out) {
  replay_t *r = p->ctx;
  // a dma line records its chunk id in res, so these are matched on kind alone and taken in the order they were recorded
  rop_t *op = take_next(r, OP_DMA);
  if (!op) return tinynv_fail("replay: no recorded dma allocation left for %zu bytes", size);
  if (op->len != size) diverge(r, "line %d: recorded dma is %llu bytes, we asked for %zu", op->line, (unsigned long long)op->len, size);

  size_t npages = (size_t)op->arg;
  void *va = calloc(npages ? npages : 1, 0x1000);
  uint64_t *pages = calloc(npages ? npages : 1, sizeof(uint64_t));
  // the recorded device addresses, so anything the driver embeds matches the trace by construction
  for (size_t i = 0; i < npages; i++) pages[i] = op->val + i * 0x1000;
  *out = (tinynv_dma_t){.va = va, .size = npages * 0x1000, .pages = pages, .npages = npages, .ctx = r};
  // the recording names each host buffer, so the view carries that name in place of a bar index and every write into it
  // is checked the same way a register write is
  out->view = (tinynv_mmio_t){.ptr = NULL, .ctx = r, .bar = op->res, .off = 0, .size = out->size,
                              .rd32 = rp_mem_rd32, .wr32 = rp_mem_wr32, .rd_block = rp_mem_rd_block, .wr_block = rp_mem_wr_block};
  return 0;
}

static void rp_dma_free(tinynv_pci_t *p, tinynv_dma_t *dma) {
  free(dma->pages);
  free(dma->va);
  memset(dma, 0, sizeof(*dma));
}

static int rp_reset(tinynv_pci_t *p) {
  replay_t *r = p->ctx;
  if (!take(r, OP_RESET, -1, 0)) diverge(r, "a reset that the trace does not contain");
  return 0;
}

static void rp_quiesce(tinynv_pci_t *p) { (void)p; }

static void rp_close(tinynv_pci_t *p) {
  replay_t *r = p->ctx;
  if (!r) return;
  free(r->ops);
  free(r->blob);
  free(r->held);
  free(r);
  p->ctx = NULL;
}

int tinynv_pci_open_replay(const char *prefix, tinynv_pci_t *out) {
  char path[1024];
  replay_t *r = calloc(1, sizeof(replay_t));
  r->window = (size_t)(getenv("TINYNV_REPLAY_WINDOW") ? atoi(getenv("TINYNV_REPLAY_WINDOW")) : REPLAY_DEFAULT_WINDOW);
  r->echo = getenv("TINYNV_REPLAY_ECHO") != NULL;

  snprintf(path, sizeof(path), "%s.trace", prefix);
  FILE *f = fopen(path, "r");
  if (!f) { free(r); return tinynv_fail("replay: %s: cannot open", path); }
  size_t cap = 1 << 14;
  r->ops = malloc(cap * sizeof(rop_t));
  char line[4096];
  int lineno = 0;
  while (fgets(line, sizeof(line), f)) {
    lineno++;
    if (strstr(line, "submission: recorded")) r->submission_recorded = 1;
    rop_t op;
    if (!parse_line(line, lineno, &op)) continue;
    if (r->nops == cap) r->ops = realloc(r->ops, (cap *= 2) * sizeof(rop_t));
    r->ops[r->nops++] = op;
  }
  fclose(f);

  snprintf(path, sizeof(path), "%s.blob", prefix);
  if ((f = fopen(path, "rb"))) {
    fseek(f, 0, SEEK_END);
    r->blob_len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    r->blob = malloc(r->blob_len ? r->blob_len : 1);
    if (fread(r->blob, 1, r->blob_len, f) != r->blob_len) r->blob_len = 0;
    fclose(f);
  }
  if (!r->nops) { free(r->ops); free(r->blob); free(r); return tinynv_fail("replay: %s.trace has no operations", prefix); }

  *out = (tinynv_pci_t){.ctx = r, .name = "replay", .cfg_read = rp_cfg_read, .cfg_write = rp_cfg_write, .bar_info = rp_bar_info,
                        .bar_map = rp_bar_map, .bar_unmap = rp_bar_unmap, .dma_alloc = rp_dma_alloc, .dma_free = rp_dma_free,
                        .reset = rp_reset, .quiesce = rp_quiesce, .close = rp_close};
  return 0;
}

void tinynv_replay_stats(tinynv_pci_t *p, tinynv_replay_stats_t *s) {
  replay_t *r = p->ctx;
  audit(r, "reporting");
  *s = (tinynv_replay_stats_t){.ops_done = r->ops_done,.total = r->nops, .consumed = r->consumed, .skipped = r->skipped,
                               .skipped_writes = r->skipped_writes, .divergences = r->divergences,
                               .cursor = r->cursor, .ngaps = r->ngaps, .gap_ops = r->gap_ops, .nforgiven = r->nforgiven,
                               .forgiven_readbacks = r->forgiven_readbacks, .submission_recorded = r->submission_recorded,
                               .first = r->first_divergence};
  for (size_t i = 0; i < r->nforgiven && i < sizeof(s->forgiven) / sizeof(*s->forgiven); i++) {
    s->forgiven[i] = r->forgiven[i];
    s->forgiven_res[i] = r->forgiven_res[i];
    s->forgiven_off[i] = r->forgiven_off[i];
    s->forgiven_is_bar_query[i] = r->forgiven_is_bar_query[i];
  }
}

void tinynv_replay_break_books_for_test(tinynv_pci_t *p) {
  replay_t *r = p->ctx;
  if (r->cursor < r->nops) r->cursor++; // the exact shape of the bug: forward progress nobody wrote down
}

int tinynv_replay_gap(tinynv_pci_t *p, int first, int last, const char *why) {
  replay_t *r = p->ctx;
  if (r->cursor >= r->nops) { diverge(r, "gap %d..%d (%s): the trace is already spent", first, last, why); return -1; }
  if (r->ops[r->cursor].line != first) {
    diverge(r, "gap %d..%d (%s): the driver is at line %d, not %d", first, last, why, r->ops[r->cursor].line, first);
    return -1;
  }
  size_t end = r->cursor;
  while (end < r->nops && r->ops[end].line <= last) end++;
  if (!end || r->ops[end - 1].line != last) {
    diverge(r, "gap %d..%d (%s): line %d is not an operation in the trace", first, last, why, last);
    return -1;
  }
  r->gap_ops += end - r->cursor;
  r->ngaps++;
  r->cursor = end;
  r->served = 0;
  audit(r, "declaring a gap");
  return 0;
}

// Walks the trace on its own, checking that every recorded read's bytes are present in the blob and hash as recorded.
// This validates the recorder and the parser together, without a driver, and prints what the boot was made of.
static int by_place(const void *a, const void *b) {
  const held_t *x = a, *y = b;
  if (x->res != y->res) return x->res < y->res ? -1 : 1;
  if (x->off != y->off) return x->off < y->off ? -1 : 1;
  return 0;
}

void tinynv_replay_forgiven_table(tinynv_pci_t *p, FILE *out) {
  replay_t *r = p->ctx;
  held_t *rows = calloc(r->nheld ? r->nheld : 1, sizeof(*rows));
  size_t n = 0;
  for (size_t i = 0; i < r->held_cap; i++) if (r->held[i].used && r->held[i].forgiven) rows[n++] = r->held[i];
  qsort(rows, n, sizeof(*rows), by_place);
  for (size_t i = 0; i < n; i++) fprintf(out, "bar:%d %#llx %zu\n", rows[i].res, (unsigned long long)rows[i].off, rows[i].forgiven);
  free(rows);
}

int tinynv_replay_fingerprint(const char *prefix, char out[65]) {
  tinynv_sha256_t c;
  tinynv_sha256_init(&c);
  for (int i = 0; i < 2; i++) {
    char path[1024];
    snprintf(path, sizeof(path), "%s.%s", prefix, i ? "blob" : "trace");
    FILE *f = fopen(path, "rb");
    if (!f) return tinynv_fail("replay: %s: cannot open to fingerprint it", path);
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) tinynv_sha256_update(&c, buf, n);
    fclose(f);
  }
  uint8_t digest[32];
  tinynv_sha256_final(&c, digest);
  tinynv_sha256_hex(digest, out);
  return 0;
}

int tinynv_replay_selftest(const char *prefix, FILE *report) {
  tinynv_pci_t pci;
  if (tinynv_pci_open_replay(prefix, &pci)) return -1;
  replay_t *r = pci.ctx;
  size_t bad = 0, reads = 0, writes = 0, polls = 0, bytes = 0;
  const char *names[] = {"cfgr", "cfgw", "bari", "mmior", "mmiow", "memr", "memw", "dma", "reset", "other"};
  size_t by_kind[OP_OTHER + 1] = {0};

  for (size_t i = 0; i < r->nops; i++) {
    rop_t *op = &r->ops[i];
    by_kind[op->kind]++;
    if (op->count > 1) polls++;
    if (op->kind == OP_MMIOR || op->kind == OP_MEMR) {
      reads++;
      bytes += op->len;
      if (op->blob_off + op->len > r->blob_len) {
        if (report && bad < 5) fprintf(report, "  line %d: blob range %#llx+%llu is past the %zu byte blob\n",
                                       op->line, (unsigned long long)op->blob_off, (unsigned long long)op->len, r->blob_len);
        bad++;
      } else if (fnv1a(r->blob + op->blob_off, op->len) != op->hash) {
        if (report && bad < 5) fprintf(report, "  line %d: blob bytes do not hash to the recorded value\n", op->line);
        bad++;
      }
    } else if (op->kind == OP_MMIOW || op->kind == OP_MEMW) writes++;
  }
  if (report) {
    fprintf(report, "  %zu operations", r->nops);
    for (int k = 0; k <= OP_OTHER; k++) if (by_kind[k]) fprintf(report, ", %s %zu", names[k], by_kind[k]);
    fprintf(report, "\n  %zu reads (%zu bytes of recorded data), %zu writes, %zu coalesced polls, %zu bad\n",
            reads, bytes, writes, polls, bad);
  }
  pci.close(&pci);
  return bad ? -1 : 0;
}
