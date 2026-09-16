// Shared plumbing for the driver's own translation units. Not installed; the public headers are include/tinynv*.h.
#ifndef TINYNV_INTERNAL_H
#define TINYNV_INTERNAL_H
#include "tinynv.h"
#include "tinynv_pci.h"

// records the message and returns -1, so a failing path reads `return tinynv_fail("bar %d is too small", bar);`
int tinynv_fail(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

// monotonic seconds, for the wait loops. one definition, because three copies is three chances to get a deadline wrong
double tinynv_now_s(void);
// Remember a host span this driver is about to write, so a later crash can say whether we wrote the faulting address.
void tinynv_note_host_write(const void *dst, size_t n, const char *what);
// A buffer this driver writes into for the life of the device, recorded once instead of on every write.
void tinynv_note_host_region(const void *base, size_t n, const char *what);
// Write every recorded span to a file and name it on stderr. Called from the crash handler; exposed for its test.
void tinynv_dump_host_writes(int sig);

// Mirror every read the oracle makes, including the ones whose answers this driver already holds.
//
// Those reads are skipped by default because each is a bus round trip on a path every mapping takes, and the replay
// proves the values are already held. But a skipped read is a debt rather than a pass: it puts this driver one operation
// ahead of the recording, which is harmless until something downstream depends on position. So the debt is switchable.
// Faithful mode reproduces the recording operation for operation with nothing forgiven, and is what the first hardware
// runs use, so the first time the card sees this driver it sees a sequence it has already accepted. (Session A's
// suggestion, 2026-09-14.)
int tinynv_is_faithful(void);
void tinynv_set_faithful(int on);

// What the last fault was, for a caller holding only a device handle. Declared here rather than in the public header
// because it is not part of the shim's contract: it exists so a test can assert that a fault report said the right
// thing, rather than that it said something. Returns 1 for an mmu fault (and fills va, type and access), 2 for an SM
// error with no mmu fault (and fills sm_esr), 0 for neither. Any out parameter may be NULL.
int tinynv_device_last_fault(tinynv_device_t d, uint64_t *va, uint32_t *type, uint32_t *access, uint32_t *sm_esr);

#endif
