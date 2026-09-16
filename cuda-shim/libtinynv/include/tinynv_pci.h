// The PCI boundary libtinynv sits on. Everything above this header is portable C11: chip bring-up, GSP, MMU, channels.
// Three backends implement it: Linux sysfs (the GPU is ours), the TinyGPU socket (a signed macOS server owns the device),
// and replay (a recorded trace, for developing the boot sequence with no GPU). Shaped like tinygrad's PCIDevice on purpose,
// so a C run and a Python run can be diffed operation for operation.
#ifndef TINYNV_PCI_H
#define TINYNV_PCI_H
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define TINYNV_MAX_BARS 6

typedef struct tinynv_mmio tinynv_mmio_t;

// A window onto a BAR. ptr is non-NULL when the backend could map it into this process; otherwise every access is a call.
struct tinynv_mmio {
  volatile uint32_t *ptr;
  void *ctx;
  int bar;
  uint64_t off, size;
  uint32_t (*rd32)(tinynv_mmio_t *m, uint64_t off);
  void (*wr32)(tinynv_mmio_t *m, uint64_t off, uint32_t v);
  void (*rd_block)(tinynv_mmio_t *m, uint64_t off, void *dst, size_t n);
  void (*wr_block)(tinynv_mmio_t *m, uint64_t off, const void *src, size_t n);
};

// Host memory the GPU can read: mapped here, and described to the GPU by its 4 KB page addresses (IOVAs behind an IOMMU).
//
// `view` is how the driver touches these bytes. It exists because the recorded boot the replay backend checks against
// includes every write into host memory the GPU later reads, so those writes have to go through something the backend
// can see rather than through the raw pointer. On a real device the view is just that pointer.
typedef struct {
  void *va;
  size_t size;
  uint64_t *pages;
  size_t npages;
  void *ctx;
  tinynv_mmio_t view;
} tinynv_dma_t;

typedef struct tinynv_pci tinynv_pci_t;
struct tinynv_pci {
  void *ctx;
  const char *name; // how the device is addressed: a bdf on linux, "usb4" over tinygpu

  uint32_t (*cfg_read)(tinynv_pci_t *p, uint32_t off, uint32_t size);
  void (*cfg_write)(tinynv_pci_t *p, uint32_t off, uint32_t size, uint32_t val);
  int (*bar_info)(tinynv_pci_t *p, int bar, uint64_t *base, uint64_t *size);
  int (*bar_map)(tinynv_pci_t *p, int bar, uint64_t off, uint64_t size, tinynv_mmio_t *out);
  void (*bar_unmap)(tinynv_pci_t *p, tinynv_mmio_t *m);
  int (*dma_alloc)(tinynv_pci_t *p, size_t size, tinynv_dma_t *out);
  void (*dma_free)(tinynv_pci_t *p, tinynv_dma_t *dma);
  int (*reset)(tinynv_pci_t *p);   // function level reset
  void (*quiesce)(tinynv_pci_t *p); // clear bus master, read back. safe to call at any time, including twice
  void (*close)(tinynv_pci_t *p);
};

// register access. the compiler must not reorder or merge these, and a bar that is not mapped turns each one into a call
static inline uint32_t nv_rd32(tinynv_mmio_t *m, uint64_t off) {
  return m->ptr ? m->ptr[off / 4] : m->rd32(m, off);
}
static inline void nv_wr32(tinynv_mmio_t *m, uint64_t off, uint32_t v) {
  if (m->ptr) m->ptr[off / 4] = v; else m->wr32(m, off, v);
}
static inline void nv_rd_block(tinynv_mmio_t *m, uint64_t off, void *dst, size_t n) {
  if (!m->ptr) { m->rd_block(m, off, dst, n); return; }
  volatile uint32_t *s = m->ptr + off / 4;
  uint32_t *d = (uint32_t *)dst;
  size_t i = 0;
  for (; i < n / 4; i++) d[i] = s[i];
  for (size_t b = i * 4; b < n; b++) ((uint8_t *)dst)[b] = ((const volatile uint8_t *)m->ptr)[off + b];
}
static inline void nv_wr_block(tinynv_mmio_t *m, uint64_t off, const void *src, size_t n) {
  if (!m->ptr) { m->wr_block(m, off, src, n); return; }
  volatile uint32_t *d = m->ptr + off / 4;
  const uint32_t *s = (const uint32_t *)src;
  size_t i = 0;
  for (; i < n / 4; i++) d[i] = s[i];
  for (size_t b = i * 4; b < n; b++) ((volatile uint8_t *)m->ptr)[off + b] = ((const uint8_t *)src)[b];
}

// Point a dma buffer's view straight at the mapping, which is what every backend that owns real memory wants. The replay
// backend is the exception and binds its own, so that writes into host memory are checked against the recording too.
static inline void tinynv_dma_bind_host(tinynv_dma_t *d) {
  d->view = (tinynv_mmio_t){.ptr = (volatile uint32_t *)d->va, .ctx = d, .bar = -1, .off = 0, .size = d->size};
}

// backends. each returns 0 on success and fills *out; on failure it reports through tinynv_last_error().
int tinynv_pci_open_tinygpu(const char *sock_path, tinynv_pci_t *out); // any host with the TinyGPU.app server (macOS)
#ifdef __linux__
int tinynv_pci_open_sysfs(const char *bdf, tinynv_pci_t *out);         // the device unbound from its driver, this process privileged
#endif
const char *tinynv_last_error(void);

// replay: the driver runs against a recorded boot instead of a gpu, so bring-up is developed and regression tested offline
int tinynv_pci_open_replay(const char *prefix, tinynv_pci_t *out); // <prefix>.trace and <prefix>.blob from tools/nv_trace.py
typedef struct {
  size_t total;       // operations in the trace
  size_t consumed;    // how many this run matched
  size_t skipped;       // stepped over: a poll that span a different number of times, tolerated by design
  size_t skipped_writes; // recorded writes the driver never issued. always a divergence too; counted for the report
  size_t divergences; // real differences: a write we made that the recording did not, or made differently
  size_t cursor;
  size_t ops_done;       // distinct recorded operations finished; `consumed` counts each serving of a coalesced poll
  size_t ngaps, gap_ops; // stages declared unported, and how many recorded operations they cover
  // Of the forgiven reads, how many were the only read-back of a write to that place. Under replay a surplus re-read and
  // an unmade read-back are indistinguishable, because both return a value already held. On hardware they are not: the
  // second is a check that would have caught a write that did not take. This must be zero.
  size_t forgiven_readbacks;
  // Whether the recording can be evidence about work submission at all. On a remote device the driver lowers the ring
  // entry, the write pointer and the doorbell into a write(2) on the socket, so they never pass the recorder. Those
  // three writes ARE the submission protocol, so a trace without them proves nothing about launching, however complete
  // the rest of it looks. A trace whose header does not say says no.
  int submission_recorded;
  // which recorded lines were forgiven, so a test can pin the exact sites rather than only how many. a count says six
  // reads were tolerated; the lines say which six, and moving one shows up as loudly as adding one.
  int forgiven[64], forgiven_res[64];
  uint64_t forgiven_off[64];
  unsigned char forgiven_is_bar_query[64]; // 1 when the forgiven operation was "where is this window", not a read
  size_t nforgiven;
  const char *first;  // the first divergence, which is the one worth reading
} tinynv_replay_stats_t;
// The books always balance: cursor == ops_done + skipped + gap_ops. The backend checks that itself on every move, and a
// failure is a divergence, so a future path that advances the cursor outside the accounting is caught where it happens.
void tinynv_replay_stats(tinynv_pci_t *p, tinynv_replay_stats_t *s);

// Declare that recorded lines first..last belong to a stage this driver has not ported yet, and step over them.
//
// This is the only way to skip a recorded write without it counting as a divergence, and it is deliberately awkward: the
// caller names the exact first and last line, so the gap cannot quietly grow to swallow a neighbouring operation, and
// every gap is counted and printed in the report. An unported stage should be visible in the numbers, not absorbed by a
// tolerance. The gap is an error unless the driver has arrived exactly at `first`.
int tinynv_replay_gap(tinynv_pci_t *p, int first, int last, const char *why);
int tinynv_replay_selftest(const char *prefix, FILE *report); // checks a trace against its own blob, no driver needed

// The identity of a recording: one hash over the trace and its blob together. Every expectation a test states is really
// a statement about a particular recorded boot, so the recording has to be pinned like any other dependency. Re-record
// the trace and this changes, which is what stops a new reference sliding in underneath assertions written for the old.
int tinynv_replay_fingerprint(const char *prefix, char out[65]);

// Every place a read was forgiven, with how many times, sorted. Written out as a table a test can compare against a
// committed one: a count alone keeps growth visible but not composition, since two places can trade forgiven reads and
// still total the same. The table is regenerated only when someone asks for it, so a change to it arrives as a diff in
// a commit where it can be looked at.
void tinynv_replay_forgiven_table(tinynv_pci_t *p, FILE *out);

// Test hook: step the cursor without accounting for it, which is precisely the bug the identity above exists to catch.
// Nothing in the driver calls this; it is here so the auditor's refusal can be demonstrated rather than asserted.
void tinynv_replay_break_books_for_test(tinynv_pci_t *p);

#endif
