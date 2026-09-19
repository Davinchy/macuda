// Physical and virtual memory on the GPU: the regions video memory is carved into, and the page tables over them.
#ifndef TINYNV_MMU_H
#define TINYNV_MMU_H
#include "dev.h"
#include "tlsf.h"
#include "fw_layout.h"

#define TINYNV_BAD_ADDR ((uint64_t)-1)

// The page size this driver programs, and the ONE place it is written down for anyone outside pt.c. Guarantee 3
// refuses a guest mapping whose address, length or offset is not a multiple of it - the granularity is ours and never
// the guest's, because a guest that could choose it could choose one spanning outside its window. pt.c and mmu.c keep
// their own local PAGE for brevity and each assert it equals this, so the two cannot drift apart silently; that drift
// is exactly how a constant gets two values in one tree, which has already cost this project a day.
#define TINYNV_MAP_PAGE 0x1000ull

typedef enum { TINYNV_REGION_BOOT, TINYNV_REGION_PTABLE, TINYNV_REGION_VRAM } tinynv_region_kind_t;

// A region of physical memory with its own allocator. Segregated fit rather than a bump pointer for the same reason the
// address space uses one: an aligned allocation leaves a gap below it, the oracle puts a later small allocation in that
// gap, and a bump pointer puts it after. Every address after that point then differs.
typedef struct { const char *name; uint64_t base, size; struct tinynv_tlsf *alloc; } tinynv_region_t;

#define TINYNV_MAX_PT_LEVELS 8

typedef struct {
  tinynv_dev_t *dev;
  tinynv_region_t boot;    // what the firmware needs before the gpu can map anything itself
  tinynv_region_t ptable;  // reserved so page tables stay inside the window the processor can reach
  tinynv_region_t pa;      // everything else
  struct tinynv_tlsf *va;  // the gpu's virtual addresses, which must match the oracle's allocator exactly
  int reserve_ptable, va_bits, levels;
  uint64_t va_base;
  // Set when a mapping could not be torn down - the MMU did not acknowledge an invalidate - and never cleared. Every
  // allocation afterwards is refused: memory cannot be handed out safely once there may be a translation to it that
  // nobody confirmed was gone. See tinynv_vmap_free.
  int wedged;
  // How many page tables this driver has allocated to satisfy mappings. See the counter's comment in pt.c: this is the
  // resource a guest can exhaust by choosing a mapping SHAPE, and it is counted before it is capped.
  uint64_t tables_made;
  // How much memory has been leaked this way, and how many times. A leak nobody can see is a leak nobody will notice:
  // Session C could only demonstrate that the pages were genuinely leaked rather than merely fenced off by reaching
  // into this struct and re-opening the gate, which says the property was unobservable from outside. It is now.
  uint64_t leaked_bytes, leaked_n;
  // per level, from the root down: how much address space one entry covers, and how many entries a table holds
  uint64_t pte_covers[TINYNV_MAX_PT_LEVELS];
  uint32_t pte_cnt[TINYNV_MAX_PT_LEVELS];
  uint64_t root_page_table;
  // Memory reserved inside the window the processor can see, before anything else allocates. On a small bar only the
  // first bytes of video memory are visible, and the physical allocator hands out whatever is free, so a ring allocated
  // after a model's weights lands outside the window and its writes go nowhere at all.
  uint64_t bar_pool_base, bar_pool_size, bar_pool_next;
  // A second such run, taken after the boot for this driver's own use; see tinynv_mm_reserve_cpu_region.
  uint64_t exec_pool_base, exec_pool_size, exec_pool_next;
} tinynv_mm_t;

// Does the manager stop below what the firmware owns? Pure arithmetic on the two wpr2 registers (base, exclusive
// limit, both bytes), so test_mm drives it with synthetic values; tinynv.c reads the registers once the firmware has
// booted and calls this. Returns 0 when the layout is sound OR when the chip exposes no region (both registers zero:
// the vendor notes they can be hidden, so an unverified reservation is reported as such, not refused), -1 through
// tinynv_fail otherwise - nonsense registers included. `line` gets one sentence saying what was found.
int tinynv_mm_check_fw_carveout(const tinynv_mm_t *mm, uint64_t wpr2_lo, uint64_t wpr2_hi, char *line, size_t n);

// One table in the tree, named by where it is and how far down it sits.
typedef struct { uint64_t paddr; int lv; } tinynv_pt_t;

typedef struct { tinynv_pt_t pt; uint32_t idx; uint64_t covers; } tinynv_pt_frame_t;

typedef struct {
  tinynv_mm_t *mm;
  uint64_t vaddr;  // relative to the base of the address space, which is how the tables are indexed
  int create, depth;
  tinynv_pt_frame_t stack[TINYNV_MAX_PT_LEVELS];
} tinynv_pt_walk_t;

// A run of neighbouring entries at one level that together cover part of a mapping.
typedef struct { tinynv_pt_t pt; uint32_t idx, entries; uint64_t covers, off; } tinynv_pt_run_t;

void tinynv_pt_walk_begin(tinynv_pt_walk_t *c, tinynv_mm_t *mm, uint64_t root, uint64_t vaddr, int create);
int tinynv_pt_walk_next(tinynv_pt_walk_t *c, uint64_t *size, uint64_t *off, uint64_t paddr, tinynv_pt_run_t *out);
// What a leaf entry says about the memory under it, as a flags word rather than a run of adjacent ints. It was
// `int sysmem, int uncached` and read-only would have made three in a row at every call site - which is the shape
// that produces a silent swap, in the one path where a swap means a guest gets permissions it did not ask for.
#define TINYNV_PTE_SYSMEM   0x1u
#define TINYNV_PTE_UNCACHED 0x2u
// GUARANTEE 7's OTHER HALF. Until 2026-09-15 this could not be expressed at all: every PTE this driver wrote was
// PCF REGULAR_RW_ATOMIC, so a guest granted a READ-ONLY mapping - which tinynv_rm_map_check correctly allows, even
// against read-only memory - was handed a WRITABLE one. The refusal path was tested and the grant path programmed
// more permission than was asked for, which is the same defect the mapping-type constant had, in the other direction.
#define TINYNV_PTE_READONLY 0x4u

void tinynv_pt_set_entry(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint32_t idx, uint64_t paddr, int table,
                         uint32_t flags, int valid);
int tinynv_pt_is_page(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint32_t idx);
int tinynv_pt_valid(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint32_t idx);
// 1 writable, 0 read-only, -1 not a valid leaf. Guarantee 7's grant half, read back from the entry itself.
int tinynv_pt_writable(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint32_t idx);

// The chain of tables a range lives under, root first, creating any that are missing.
int tinynv_mm_page_tables(tinynv_mm_t *mm, uint64_t root, uint64_t vaddr, uint64_t size, tinynv_pt_t *out, int *n);

typedef struct { uint64_t paddr, size; } tinynv_paddr_range_t;

// Memory with an address the GPU can use: what was allocated, and where it was mapped. The physical side may be many
// pieces: a large allocation is made of the biggest chunks that fit, so the page tables can cover it with big pages.
typedef struct {
  uint64_t va, size;
  int nranges;
  tinynv_paddr_range_t *ranges;
  int in_bar_pool; // came from the processor-visible reserve, whose pages are never handed back
  tinynv_dma_t dma; // set when the memory is the host's rather than the gpu's
  int in_sysmem;
} tinynv_vmap_t;

void tinynv_vmap_free(tinynv_mm_t *mm, tinynv_vmap_t *m);
int tinynv_mm_unmap_range(tinynv_mm_t *mm, uint64_t root, uint64_t vaddr, uint64_t size);

// A buffer, allocated the way the oracle's allocator does: host memory or video memory depending on what the caller needs
// to be able to reach, out of the processor-visible reserve when it must be both video memory and writable from here.
int tinynv_mm_alloc_buffer(tinynv_mm_t *mm, uint64_t size, int host, int cpu_access, int uncached, int force_devmem,
                           int zero, tinynv_vmap_t *out);

// Reserve the cpu-visible pool. Does nothing when the whole of video memory is visible anyway.
int tinynv_mm_reserve_bar_pool(tinynv_mm_t *mm);
// Reserve a further run of processor-writable video memory, after the boot, for allocations of this driver's own.
int tinynv_mm_reserve_cpu_region(tinynv_mm_t *mm, uint64_t size);

int tinynv_mm_map_range(tinynv_mm_t *mm, uint64_t root, uint64_t vaddr, uint64_t size, const tinynv_paddr_range_t *paddrs, int npaddrs,
                        uint32_t flags);
int tinynv_mm_valloc(tinynv_mm_t *mm, uint64_t size, uint64_t align, int contiguous, int uncached, tinynv_vmap_t *out);

// A block of memory the firmware reads while booting, wherever it had to be put.
//
// On a card whose whole memory is visible it lives in video memory; on a small-bar card, which is every card behind
// thunderbolt, it lives in host memory instead and the GPU reaches it over the bus. Either way the firmware is told the
// same thing: a list of addresses, one per 4 KB page, which is `addrs`.
typedef struct {
  tinynv_mmio_t view;  // how the processor reads and writes these bytes
  uint64_t paddr;      // where it sits in video memory, or TINYNV_BAD_ADDR when it is host memory
  uint64_t *addrs;     // what the gpu is told, page by page
  size_t naddrs, size;
  tinynv_dma_t dma;
  int in_sysmem;
} tinynv_bootmem_t;

int tinynv_mm_init(tinynv_mm_t *mm, tinynv_dev_t *dev);
uint64_t tinynv_mm_palloc(tinynv_mm_t *mm, uint64_t size, uint64_t align, int zero, tinynv_region_kind_t kind);
uint64_t tinynv_mm_alloc_va(tinynv_mm_t *mm, uint64_t size, uint64_t align);

// Allocate boot memory and optionally fill it. `sysmem` is 1 for host memory, 0 for video memory, and -1 to let the size
// of the window decide, which is what the boot path asks for.
int tinynv_alloc_boot_mem(tinynv_mm_t *mm, size_t size, const void *data, int sysmem, tinynv_bootmem_t *out);
void tinynv_free_boot_mem(tinynv_mm_t *mm, tinynv_bootmem_t *m);

#endif
