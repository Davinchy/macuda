// GSP-RM: the firmware that actually runs the GPU, and the structures the driver builds for it before it starts.
//
// Almost everything a modern NVIDIA GPU does happens inside GSP-RM, a RISC-V processor on the die running NVIDIA's own
// resource manager. The host driver's job is to place its firmware in memory the GPU can reach, describe that placement,
// and then talk to it over a message queue. This header covers the placement; the queue is the next stage.
#ifndef TINYNV_GSP_H
#define TINYNV_GSP_H
#include "fw.h"
#include "mmu.h"
#include "nv_structs.h"

typedef struct tinynv_gpu tinynv_gpu_t;

// One direction of the shared ring. `tx` describes our side of it and is read back from memory rather than assumed;
// `rx_off` is where the reader's position is kept, which by the queue's convention lives in the *other* queue's header.
typedef struct {
  uint64_t base;      // where this queue starts inside the shared allocation
  uint64_t entries;   // where its message slots start
  uint64_t rx_off;    // where the read pointer lives, or 0 until the other end exists
  tinynv_msgq_tx_header_t tx;
  uint32_t seq;
} tinynv_rpcq_t;

// A channel the driver submits work to: where its ring is, how the hardware is told the ring has grown, and the token
// that names it at the doorbell.
typedef struct tinynv_queue {
  uint32_t channel, object, token, entries;
  uint64_t ring_va, gpput_off;
  uint64_t put;           // how many entries have been published, which is what names the next slot
  tinynv_vmap_t notifier;
  tinynv_vmap_t grctx[3]; // this channel's own copy of the graphics context, on a compute channel
} tinynv_queue_t;

typedef struct {
  tinynv_gpu_t *gpu;

  // the message queue shared with GSP-RM, its page table, and the arguments that describe both
  tinynv_bootmem_t queues, rm_args;

  // Sensors. Host memory we allocate and the firmware writes into on a timer; see tinynv_gsp_sensors_init.
  // Zero when sensors were not asked for, which is the default.
  tinynv_vmap_t rusd_mem;
  uint32_t rusd_obj;   // zero when the card refused the client-facing class, which is not fatal
  int rusd_armed;      // the two controls went through, which is what actually decides whether this works
  uint64_t rm_args_sysmem, cmd_q_off, queue_size;
  tinynv_rpcq_t cmd_q, stat_q;
  int err_state;      // gsp-rm reported an error or a fault; the boot is no longer trustworthy
  int up;             // gsp-rm sent its start-up notice and has not been told to go: there is something to unload
  int init_timed_out; // the wait for the start-up notice ran out, which is the one failure a fresh boot may cure
  // What it actually said, kept so a wait that times out can name it. "gsp-rm reported a fault" with nothing else is
  // the least useful true sentence a driver can print: an MMU fault means a bad address and an error log means the
  // firmware is complaining about something it was asked to do, and those lead opposite ways.
  uint32_t err_fn;    // the event function number, 0 if none
  uint32_t err_len;   // how long its payload was
  uint8_t err_head[64];   // and the start of it, which is where anything addressable lives
  // Whether the debugger has already been asked where the fault was. One attempt only, and the flag is set before the
  // asking rather than after: the questions are RPCs to firmware that has just reported something wrong, so if one of
  // them does not come back, a second attempt would not come back either, and it would be made from a path that is
  // already reporting a failure.
  int fault_reported;
  // And what it said, kept rather than only printed. A printed line is for a person reading a log; these are for the
  // failure message one layer up, which can then name the address instead of pointing at something further up the
  // screen, and for a test to assert on - the difference between a report that runs and a report that is right.
  int fault_have;         // an mmu fault with at least one address
  uint64_t fault_va;      // the first of them
  uint32_t fault_type, fault_access;
  int fault_sm_have;      // no mmu fault, but an SM reported an error: a kernel did something illegal
  uint32_t fault_sm_esr;  // that SM's warp error status, which says what
  uint32_t cpu_seq_requests; // register sequences gsp-rm asked us to run. zero in the recording, so watch it on hardware
  uint32_t cpu_seq_saved[8]; // the sequencer's save-register op: eight slots, matching rpc_run_cpu_sequencer's own area
  // An rpc gave up waiting, so its reply may still be coming and nothing has claimed it. The next request must not be
  // allowed to match it: replies carry a function number and no request identity, so a late reply to one control looks
  // exactly like the answer to the next control, and the parameters would be copied back from the wrong one. Set on a
  // timeout, cleared once the queue has been emptied before a send.
  int rpc_desync;

  // the object tree inside GSP-RM that everything later hangs off
  uint32_t next_handle, priv_root, device, subdevice, vaspace;
  uint32_t nengines, engines[TINYNV_FIFO_DEVICE_ENTRIES], runlists[TINYNV_FIFO_DEVICE_ENTRIES];
  // A second channel group on a COPY engine of its own (TINYNV_CE_GROUP). The copy channel in the graphics group
  // shares CE0's PBDMA with GR0 and cannot overlap a kernel; CE1 reports its own runlist and its own PBDMA pair.
  // Allocated and scheduled but not yet routed to - see gsp.c.
  uint32_t ce_group, ce_ctxshare, ce_runlist, ce_engine;
  // RM's own status from the last control call. rm_control_as reports but does not enforce it, so a caller that
  // actually depends on a control having worked has to look here.
  uint32_t last_ctrl_status;
  tinynv_queue_t ce_copy_q;
  uint64_t gr_size, patch_size; // how big the graphics context and its patch buffer must be, per the firmware
  uint64_t reserved_va;   // the range whose page tables gsp-rm was given
  int reserved_levels;

  // the channel: the ring of command buffer pointers, the areas gsp-rm keeps its state in, and its handle
  tinynv_vmap_t gpfifo, ramfc;
  tinynv_bootmem_t mthdbuf;
  uint32_t channel, channel_runlist;

  // the graphics engine's own state, and the classes the channel runs
  tinynv_vmap_t grctx[12];
  int ngrctx;
  // Handles for the objects allocated on the channel, NOT the class numbers those objects are instances of. The
  // distinction matters at exactly one place - SET_OBJECT in a command buffer wants the class - and putting a handle
  // there produced "Graphics Exception: Class 0x5 Subchannel 0x1 Mismatch", 0x5 being the low byte of 0xcf000005.
  uint32_t compute_obj, dma_copy_obj;

  // the ordinary client everything after the graphics context belongs to
  uint32_t user_root, user_device, user_subdevice, user_virtmem;
  uint32_t user_vaspace, user_group, user_ctxshare;
  // A second subcontext for the copy channel, when TINYNV_SPLIT_CTXSHARE asks for one. Zero means both channels share
  // user_ctxshare, which is what they have always done.
  uint32_t copy_ctxshare;
  // The chip's topology, which is the FULL die and not what is enabled. These multiply to the physical maximum: on a
  // GB202 that is 192 streaming multiprocessors, where an RTX 5090 ships with 170 and the rest fused off. Whatever fills
  // the device properties must NOT report the product of these as the processor count - ggml divides work by
  // multiProcessorCount, so 192 would be thirteen percent optimistic in exactly the direction that hurts. The enabled
  // count comes from the SM-count control or from the floorsweeping masks, not from here. (Session A, 2026-09-14.)
  uint32_t max_gpcs, max_tpc_per_gpc, max_sm_per_tpc, max_warps_per_sm, sm_version;

  tinynv_vmap_t scratch;  // one page of device memory this client takes before anything else
  tinynv_vmap_t staging;  // host memory every copy in either direction passes through

  // the rings work is submitted through, both inside one processor-visible allocation
  tinynv_vmap_t fifo_mem;
  tinynv_queue_t compute_q, copy_q;
  uint32_t user_debugger;

  // the regions GSP-RM's operating system is handed at start-up
  tinynv_bootmem_t logbuf, libos_args;
  uint64_t libos_args_sysmem;

  // GSP-RM's firmware: the image under a three level page table, its signature, and the bootloader that starts it
  tinynv_blob_t gsp_fw, bl_fw;
  tinynv_bootmem_t radix3, signature, bootloader;
  size_t gsp_image_size, bootloader_size;
  tinynv_riscv_ucode_desc_t bl_desc;

  // where all of that is, in one structure the boot firmware reads
  tinynv_bootmem_t wpr_meta;
  uint64_t wpr_meta_sysmem;
} tinynv_gsp_t;

// Everything GSP-RM needs to exist in memory before the chain of trust can authorise it. No hardware is touched.
int tinynv_gsp_init_sw(tinynv_gpu_t *g);

// Wait for GSP-RM to come up and say so, then point the two memory windows at it. Rings the doorbell, so this is the
// first thing in the driver that expects the firmware to be alive.
int tinynv_gsp_init_hw(tinynv_gpu_t *g);
// Tell GSP-RM the driver is going and wait for it to halt. After this the falcon may be reset and reused (flcn.c).
int tinynv_gsp_unload(tinynv_gpu_t *g);

// Create the privileged client and the device, chip and address space objects under it. Every later call needs these.
int tinynv_gsp_init_objects(tinynv_gpu_t *g);

// Create the channel work is submitted through. Needs the object tree.
int tinynv_gsp_init_channel(tinynv_gpu_t *g);

// Collect anything gsp-rm has reported and not been asked for - an mmu fault, an error log - and say whether it was
// bad news. Cheap, and the only way a stalled engine ever explains itself.
int tinynv_gsp_poll(tinynv_gpu_t *g);

// Allocate the graphics engine's context buffers, tell GSP-RM where they are, and attach the compute and copy classes.
int tinynv_gsp_init_gr_context(tinynv_gpu_t *g);

// Open the ordinary client the driver submits work through, with its own device and address space.
int tinynv_gsp_open_client(tinynv_gpu_t *g);

// Create the compute and copy channels and start the group scheduling.
int tinynv_gsp_init_queues(tinynv_gpu_t *g);

// Ask the firmware to keep the sensor block fresh: one host buffer handed over, one object, two controls. `freq_ms` is
// how often the card refreshes it. Costs nothing on the submission path - after this the readings are a load, not a
// message - but it is off unless asked for, because the estimate of "nothing" is mine until it is measured.
// How much video memory GSP-RM believes it can still allocate. Zero is a meaningful answer and the one this driver
// expects, because nothing has ever given it any.
int tinynv_gsp_free_heap(tinynv_gpu_t *g, uint64_t *out);

int tinynv_gsp_sensors_init(tinynv_gpu_t *g, uint64_t poll_mask, uint32_t freq_ms);
// A consistent copy of one field, or zero if the firmware was mid-write every time we looked. `ts` of zero in the
// result means the card does not report that reading on this platform.
int tinynv_gsp_sensors_read(tinynv_gpu_t *g, tinynv_rusd_t *out);

void tinynv_gsp_fini(tinynv_gpu_t *g);

#endif
