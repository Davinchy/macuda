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
// What has been allocated, in the order it was allocated.
//
// The driver used to hand out handles from a counter and record nothing, which meant a free had nowhere to look: not a
// missing call but a missing record. This is that record, and it is deliberately more than a list of live handles,
// because the thing asking for a free may be a guest and a guest's free order is not a safe order. It can free a parent
// before its children, free the same object twice, free what it never allocated, or free another client's object. If
// this is the only thing that knows the tree, it is the only thing that can refuse - so refusing is its job rather than
// a check bolted on afterwards.
//
// Append-only and in creation order, which buys the hard part for nothing: a child is always created after its parent,
// so walking backwards frees children first without anyone having to sort anything.
// Grows rather than capping. A fixed ceiling was the first shape and it was wrong: bring-up uses about a dozen, but a
// guest running vLLM creates a context per worker - channel group, channels, address space, memory - and destroys them
// as workers cycle, so a fixed record would refuse at exactly object 257 and read as a driver limit rather than as
// pressure building. The warning below is what a cap was actually for: noticing a leak. It says so once and keeps going.
#define TINYNV_RM_OBJECTS_FIRST 64
#define TINYNV_RM_OBJECTS_LOUD 4096
// `size` is the allocation's size in bytes, or 0 for "not known here". Guarantee 8 bounds a guest's mapping offset
// against it - offset + length <= size - so an object whose size we never learned cannot be mapped at all. Refusing
// is the only safe reading of a missing bound: the alternative is a guest naming bytes past the end of its own
// allocation and us having nothing to say about it.
typedef struct { uint32_t client, parent, handle, cls, alloc_flags; uint64_t size; } tinynv_rm_obj_t;


// Is this allocation writable? Guarantee 7 compares a requested mapping's permission against it. Only meaningful when
// the size was captured too - an allocation whose params never reached us has no known permission either, and the
// safe reading of an unknown permission is the same as an unknown size: refuse.
int tinynv_rm_obj_writable(const tinynv_rm_obj_t *o);

// A live mapping, so that a free can find what it has to tear down first.
//
// Guarantee 5 of docs/driver/libtinynv-design.md §4g, and the one that matters most: our free CASCADES over children
// and dependants, so an allocation can go away underneath a live mapping and leave page tables pointing at memory the
// allocator can hand to somebody else. The record therefore lives beside the OBJECT record rather than beside the page
// tables - a free walks objects, and what it needs to consult has to be where it already is.
//
// `root` is the page-table tree the mapping was programmed into, because a guest's tree is not the driver's; see the
// separate-root decision in §4g.
typedef struct { uint32_t client, handle; uint64_t root, va, length; } tinynv_rm_map_t;

// What an unmap is allowed to do. Exact-match only, which is guarantee 12: the walk may program a page LARGER than
// asked for when the addresses permit, so a 2 MiB mapping can be a single entry, and unmapping 4 KiB of it would clear
// the whole thing while the record still believed the rest was live.
typedef enum {
  TINYNV_RM_UNMAP_OK = 0,
  TINYNV_RM_UNMAP_UNKNOWN,   // no mapping at that base for this client and tree
  TINYNV_RM_UNMAP_PARTIAL,   // a mapping is there and the length does not match it
} tinynv_rm_unmap_check_t;

// Pure, so the refusals can be driven with no card and no device the way test_rm_free.c does for the free path.
tinynv_rm_unmap_check_t tinynv_rm_unmap_check(const tinynv_rm_map_t *maps, int n, uint32_t client, uint64_t root,
                                              uint64_t va, uint64_t length, int *at);
// Does anything still map this object? Guarantee 5's question, asked by the free path before it frees.
int tinynv_rm_object_is_mapped(const tinynv_rm_map_t *maps, int n, uint32_t client, uint32_t handle);

// A guest's request to map memory, as UVM_MAP_EXTERNAL_ALLOCATION_PARAMS presents it. `root` is the page-table tree
// the request would be programmed into; `mapping_type` is gpuMappingType out of perGpuAttributes.
typedef struct {
  uint32_t client, handle, mapping_type;
  uint64_t root, va, length, offset;
} tinynv_rm_map_req_t;

// Why a mapping was refused. One value per guarantee that can refuse, so the refusal names the rule rather than
// saying no - a guest developer reading "REFUSED" learns nothing and a future reader of this code learns less.
typedef enum {
  TINYNV_RM_MAP_OK = 0,
  TINYNV_RM_MAP_NO_OBJECT,      // 1: never allocated here, or under a different client. Not "denied": not found.
  TINYNV_RM_MAP_NOT_GUEST_TREE, // 2: the root is the driver's own, so this would program a guest mapping into our tree
  TINYNV_RM_MAP_BAD_GEOMETRY,   // 3: zero length, or a wrap in va+length or offset+length
  TINYNV_RM_MAP_ALREADY_MAPPED, // 4: something already occupies that range in that tree
  TINYNV_RM_MAP_WOULD_WRITE,    // 7: read-write asked of an allocation whose flags say read-only
  TINYNV_RM_MAP_OUT_OF_BOUNDS,  // 8: offset+length past the end of the allocation
  TINYNV_RM_MAP_SIZE_UNKNOWN,   // 8: the allocation's params never reached us, so there is no bound to check against
  TINYNV_RM_MAP_MISALIGNED,     // 3: va, length or offset is not a multiple of the page size WE program
  // The last two are NOT the guest's fault, and are kept distinct for that reason: a guest handed "your request was
  // refused" for something its own request did not cause will go looking in the wrong place, and so will we.
  TINYNV_RM_MAP_BAD_RANGES,     // the physical ranges we were given do not describe the request: wrong total, or not
                                // page-aligned. GSP-RM supplied these, so this is a disagreement between us and it
  TINYNV_RM_MAP_NO_ROOM,        // the mapping record is full. Refused rather than programmed-and-unrecorded, because
                                // an unrecorded mapping is exactly what guarantee 5 exists to make impossible
} tinynv_rm_map_check_t;

// Pure: no device, no page tables, no card. Every refusal in the C4b guarantee set that can be decided from the
// records is decided here, so the decisions can be driven offline the way the free path's are.
tinynv_rm_map_check_t tinynv_rm_map_check(const tinynv_rm_obj_t *objs, int nobjs, const tinynv_rm_map_t *maps,
                                          int nmaps, uint64_t driver_root, const tinynv_rm_map_req_t *req);
// What a refusal means, for the message the guest is handed.
const char *tinynv_rm_map_why(tinynv_rm_map_check_t c);

// Does the firmware answer 0x410103 at all? Allocates its own class-0x0040 object, asks, frees. Diagnostic only -
// see test/test_hw_physattr.c. Returns 0 with the answer filled in, non-zero with tinynv_last_error() saying which
// half failed: the allocation or the control.
int tinynv_gsp_probe_phys_attr(tinynv_gpu_t *g, uint64_t size, uint64_t offset, uint64_t *paddr, uint64_t *contig,
                               uint32_t *aperture);

// Asks where one offset of a guest's allocation physically lives: 0 on success, non-zero to abandon the walk.
// A function pointer because the real one issues NV0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR to GSP-RM and a test issues
// nothing - so the WALK, which is where the arithmetic and the refusals are, is exercised offline while only the
// transport needs a card.
typedef int (*tinynv_phys_query_fn)(void *ctx, uint64_t offset, uint64_t *paddr, uint64_t *contig);

// Walk [offset, offset+length) of an allocation into physical ranges, largest contiguous piece at a time.
//
// Returns 0 on success with *nout set. Every failure is a REFUSAL rather than a partial answer: a short range list
// would pass through tinynv_c4b_map's total check as a mismatch and be reported as a firmware disagreement, which
// sends the next reader to the wrong place. The named ones matter - a contigSegmentSize of ZERO makes no progress,
// and a loop that asks again at the same offset does not terminate.
int tinynv_c4b_ranges(tinynv_phys_query_fn ask, void *ctx, uint64_t offset, uint64_t length,
                      tinynv_paddr_range_t *out, int maxout, int *nout);

// The C4b map itself: decide, then program, then record - in that order, and nothing happens unless all three can.
//
// THIS IS THE FUNCTION gsp.c USED TO CITE AND DID NOT HAVE. A comment listed guarantee 3's alignment as living in
// "tinynv_c4b_map" while no such function existed, which is how that guarantee went unenforced without anyone
// noticing; the alignment is now in tinynv_rm_map_check where it belongs, and this is the real thing.
//
// `ranges` are the PHYSICAL pages behind the guest's allocation. This driver cannot know them: the guest's class
// 0x0040 allocation went through our forwarding path, so GSP-RM holds the memory descriptor. NVIDIA's own UVM path
// reads pMemory->pMemDesc directly, which is in-kernel RM state and does not exist for us. They have to come from
// NV0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR (0x410103), walked with contigSegmentSize - a CARD question, since GSP is
// closed firmware and no recording contains that call. They are a parameter here so that everything except obtaining
// them is decided and tested offline.
//
// On OK the entries are programmed into req->root and the mapping is recorded. On anything else NOTHING has changed,
// including a failure partway through programming - the range is torn back down rather than left half-mapped.
tinynv_rm_map_check_t tinynv_c4b_map(tinynv_mm_t *mm, const tinynv_rm_obj_t *objs, int nobjs, tinynv_rm_map_t *maps,
                                     int *nmaps, int maxmaps, uint64_t driver_root, const tinynv_rm_map_req_t *req,
                                     const tinynv_paddr_range_t *ranges, int nranges);

// Whether a free is allowed, and why not. Separated from the sending of it so that the refusals - which are the whole
// point of the record - can be driven with no card and no device, the way test/test_rm_free.c does.
typedef enum {
  TINYNV_RM_FREE_OK = 0,
  TINYNV_RM_FREE_UNKNOWN,       // never allocated here, or allocated under a different client
  TINYNV_RM_FREE_HAS_CHILDREN,  // objects live under it, so freeing it takes them too
} tinynv_rm_free_check_t;

// Whether one object is below another in the same client, following the parent chain rather than one step of it.
int tinynv_rm_is_descendant(const tinynv_rm_obj_t *objs, int n, uint32_t client, uint32_t handle, uint32_t ancestor);
// `at` receives the object's index when the answer is OK, and is untouched otherwise.
tinynv_rm_free_check_t tinynv_rm_free_check(const tinynv_rm_obj_t *objs, int n, uint32_t client, uint32_t handle,
                                            int *at);

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

  // Every rm object this driver has created, oldest first. See tinynv_rm_obj_t above.
  tinynv_rm_obj_t *objs;
  int nobjs, objs_cap;
  tinynv_rm_map_t *maps;
  int nmaps, maps_cap;
  int objs_warned;

  // Sensors. Host memory we allocate and the firmware writes into on a timer; see tinynv_gsp_sensors_init.
  // Zero when sensors were not asked for, which is the default.
  tinynv_vmap_t rusd_mem;
  uint32_t rusd_obj;   // zero when the card refused the client-facing class, which is not fatal
  int rusd_armed;      // the two controls went through, which is what actually decides whether this works
  uint64_t rm_args_sysmem, cmd_q_off, queue_size;
  tinynv_rpcq_t cmd_q, stat_q;
  int err_state;      // gsp-rm reported an error or a fault; the boot is no longer trustworthy
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
  // An rpc gave up waiting, so its reply may still be coming and nothing has claimed it. The next request must not be
  // allowed to match it: replies carry a function number and no request identity, so a late reply to one control looks
  // exactly like the answer to the next control, and the parameters would be copied back from the wrong one. Set on a
  // timeout, cleared once the queue has been emptied before a send.
  int rpc_desync;

  // the object tree inside GSP-RM that everything later hangs off
  uint32_t next_handle, priv_root, device, subdevice, vaspace;
  uint32_t nengines, engines[TINYNV_FIFO_DEVICE_ENTRIES], runlists[TINYNV_FIFO_DEVICE_ENTRIES];
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

// Free one rm object, validated against the record: refused if it was never allocated, belongs to another client, or
// still has objects under it. Freeing is not a call this driver had at all until now - handles came from a counter and
// nothing was written down, so the missing piece was the record and not the message.
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

// The RM surface a caller outside this file uses, including a forwarder acting for a guest.
//
// ALL THREE, OR NONE. Allocation is the only place the record is written, so an object created any other way is one no
// free can find and no teardown can reach. `handle` of zero lets this driver choose; a forwarder passes the handle its
// guest chose, which GSP-RM accepts.
int tinynv_gsp_rm_alloc(tinynv_gpu_t *g, uint32_t client, uint32_t parent, uint32_t handle, uint32_t cls, void *params,
                        size_t params_len, uint32_t *out);
int tinynv_gsp_rm_control(tinynv_gpu_t *g, uint32_t client, uint32_t object, uint32_t cmd, void *params,
                          size_t params_len);
int tinynv_gsp_rm_free(tinynv_gpu_t *g, uint32_t client, uint32_t handle);
// Everything belonging to any client in [lo, hi), children first. This is the eviction case: a guest that dies
// mid-kernel frees nothing and tells nobody, and by then the only thing that knows what it made is this record. A range
// rather than a list of clients because nobody holds that list - the component that chose the handles died with the
// guest - while the range the guest's clients come from is fixed by construction. THE UPPER BOUND IS EXCLUSIVE and this
// driver's own clients sit just outside it, so a caller passing an inclusive bound would tear down the session it is
// running in; that is refused here rather than trusted to the caller. Returns the number refused.
int tinynv_gsp_rm_free_clients(tinynv_gpu_t *g, uint32_t lo, uint32_t hi);
// Whether [lo, hi) can be swept without taking one of this driver's own clients with it. Separated so the boundary can
// be driven with no card: the interesting inputs are the correct range and the off-by-one beside it, and the off-by-one
// destroys the session rather than returning an error.
int tinynv_rm_range_safe(uint32_t lo, uint32_t hi);
// Everything, newest first, which is children before parents. Keeps going past a refusal and reports the count.
int tinynv_gsp_rm_free_all(tinynv_gpu_t *g);

void tinynv_gsp_fini(tinynv_gpu_t *g);

#endif
