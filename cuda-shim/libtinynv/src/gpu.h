// A GPU, as the driver holds it: the chip, its memory, and the two firmware blocks that have to come up in order.
#ifndef TINYNV_GPU_H
#define TINYNV_GPU_H
#include "dev.h"
#include "flcn.h"
#include "gsp.h"
#include "mmu.h"

struct tinynv_gpu {
  tinynv_dev_t dev;
  tinynv_mm_t mm;
  // Whether the read before the doorbell has been confirmed to reach the device; see tinynv_submit. On a backend where
  // that read is a message rather than a load from a mapped bar, it is worth knowing rather than assuming.
  int submit_checked;
  // Batches whose ring entry and write pointer are written but whose doorbell has not been rung. The read that orders
  // those writes ahead of the doorbell is one round trip, and one read fences every write issued before it on that
  // path - so several batches can be staged and then announced together, paying one round trip instead of one each.
  // Two is what the flush path needs (the descriptor delivery and the chain it feeds); four is room to spare.
  tinynv_queue_t *staged[64];
  int nstaged;
  // An announcement in flight: the fence read has been sent, the doorbells wait for its reply. At most one, and the
  // next send or any wait completes it. See tinynv_submit_ring_send / _complete.
  int ring_pending, ring_n;
  uint32_t ring_want;
  tinynv_queue_t *ring_q[64];
  tinynv_flcn_t flcn;
  tinynv_gsp_t gsp;
};

// Identify the chip, put it in a known state, and carve up its memory. Reads and writes registers; runs no firmware.
int tinynv_gpu_open(tinynv_gpu_t *g, tinynv_pci_t *pci);

// Prepare everything both firmware blocks need in memory. No hardware is touched, so this is safe to run anywhere.
int tinynv_gpu_init_sw(tinynv_gpu_t *g);

// Start the firmware: the chain of trust, then waiting for GSP-RM to say it is up. This talks to the GPU.
int tinynv_gpu_init_hw(tinynv_gpu_t *g);

void tinynv_gpu_close(tinynv_gpu_t *g);

#endif
