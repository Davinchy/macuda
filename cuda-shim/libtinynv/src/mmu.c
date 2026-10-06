// Physical memory on the GPU, and the page tables that map it.
//
// Video memory is carved into three regions, mirroring tinygrad's MemoryManager: a small boot region at the bottom for
// everything the firmware needs before the GPU can map anything itself, then a region reserved for page tables, then the
// rest. The order matters on a small-BAR card: the processor can only reach the first 256 MB through the window, so the
// page tables and the boot structures must live down there, and the reservation is what keeps them there.
//
// The allocators bump upward. tinygrad uses a two-level segregated fit allocator, which returns the same addresses as a
// bump allocator while nothing has been freed — which is the whole of the boot sequence — so replay against a recorded
// boot verifies them as equivalent. When freeing starts to matter this has to grow a free list, and the recording will
// say so by diverging on an address.
#include <time.h>
#include "mmu.h"
#include "internal.h"
#include <stdlib.h>
#include <string.h>

#define MB (1ull << 20)
#define PAGE 0x1000ull
_Static_assert(PAGE == TINYNV_MAP_PAGE, "this file's page size and the one guarantee 3 refuses against "
                                        "have diverged, so a mapping could be aligned for one and not the other");

static uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

static uint64_t region_alloc(tinynv_region_t *r, uint64_t size, uint64_t align, const char *what) {
  if (!r->alloc) {
    tinynv_fail("%s: the %s region was never sized", what, r->name);
    return TINYNV_BAD_ADDR;
  }
  uint64_t at = tinynv_tlsf_alloc(r->alloc, size, align ? align : PAGE);
  if (at == TINYNV_TLSF_FAIL) {
    tinynv_fail("%s: %llu bytes do not fit the %llu MB %s region", what, (unsigned long long)size,
                (unsigned long long)(r->size >> 20), r->name);
    return TINYNV_BAD_ADDR;
  }
  return at;
}

// zero physical memory through the window onto it. only what the processor can reach can be zeroed this way, and on a
// small bar that is the low part of memory: anything above the window is left to the firmware, which initialises its own.
static void zero_vram(tinynv_mm_t *mm, uint64_t paddr, uint64_t size) {
  if (paddr + size > mm->dev->vram.size) return;
  // one write for the whole range, not a page at a time. the difference is visible from outside: the backend turns each
  // call into its own transfer, so chunking would put five messages on the wire where the oracle puts one.
  void *zeros = calloc(1, (size_t)size);
  if (!zeros) { tinynv_fail("out of memory to zero %llu bytes of video memory", (unsigned long long)size); return; }
  nv_wr_block(&mm->dev->vram, paddr, zeros, (size_t)size);
  free(zeros);
}

// Once a mapping has failed to tear down there may be a translation to recycled memory that nobody confirmed was
// gone, so nothing more is handed out. Both doors, because refusing one and not the other would be a gate with a
// hole in it. See tinynv_vmap_free.
// int, not uint64_t, and the convention is 0 for ok and non-zero for refused. Written as uint64_t first, where the
// refusal value is tinynv_fail's -1 widened to a very large positive number - true, so it worked, and exactly the
// shape that has bitten this project twice today: a helper whose failure value is truthy by accident rather than by
// contract. It works either way; only one of them still works when someone writes `!refuse_if_wedged(...)`.
static int refuse_if_wedged(tinynv_mm_t *mm) {
  return mm->wedged ? tinynv_fail("the mmu did not acknowledge an invalidate earlier, so this driver will not hand out "
                                  "memory again - something may still be able to reach what was freed")
                    : 0;
}

uint64_t tinynv_mm_palloc(tinynv_mm_t *mm, uint64_t size, uint64_t align, int zero, tinynv_region_kind_t kind) {
  if (refuse_if_wedged(mm)) return TINYNV_BAD_ADDR;
  tinynv_region_t *r = kind == TINYNV_REGION_BOOT ? &mm->boot : (kind == TINYNV_REGION_PTABLE && mm->reserve_ptable ? &mm->ptable : &mm->pa);
  uint64_t paddr = region_alloc(r, round_up(size, PAGE), align, "physical allocation");
  if (paddr == TINYNV_BAD_ADDR) return paddr;
  if (zero) zero_vram(mm, paddr, round_up(size, PAGE));
  return paddr;
}

// VA LEDGER. A fault names an address and nothing about what it was. This keeps the last VAL_N ranges the address
// allocator handed out (with a tag for who asked: a kernel image, a cudaMalloc, the driver's own arenas) and marks each
// one when it is given back, so the post-mortem can say "inside live range X", "inside range Y, FREED n allocations ago"
// or "in no range ever handed out", which are three different bugs.
#define VAL_N 16384
typedef struct { uint64_t va, size, seq, freed_seq; double t_alloc, t_free; char tag[56]; } val_entry_t;
static val_entry_t val_ring[VAL_N];
static uint64_t val_total, val_frees;
static char val_cur_tag[56] = "driver";
static double val_t0;

static double val_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double t = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
  if (!val_t0) val_t0 = t;
  return t - val_t0;
}

void tinynv_va_tag_set(const char *tag) { snprintf(val_cur_tag, sizeof(val_cur_tag), "%s", tag && *tag ? tag : "driver"); }

static void val_record(uint64_t va, uint64_t size) {
  val_entry_t *e = &val_ring[val_total % VAL_N];
  memset(e, 0, sizeof(*e));
  e->va = va; e->size = size; e->seq = ++val_total; e->t_alloc = val_now();
  snprintf(e->tag, sizeof(e->tag), "%s", val_cur_tag);
}

void tinynv_va_ledger_free(uint64_t va, uint64_t size) {
  uint64_t n = val_total < VAL_N ? val_total : VAL_N;
  for (uint64_t k = 0; k < n; k++) {
    val_entry_t *e = &val_ring[(val_total - 1 - k) % VAL_N];
    if (e->va == va && !e->freed_seq) {
      e->freed_seq = ++val_frees; e->t_free = val_now();
      (void)size;
      return;
    }
  }
}

int tinynv_va_ledger_tag(uint64_t va, char *out, size_t n) {
  uint64_t cnt = val_total < VAL_N ? val_total : VAL_N;
  for (uint64_t k = 0; k < cnt; k++) {
    const val_entry_t *e = &val_ring[(val_total - 1 - k) % VAL_N];
    if (va >= e->va && va < e->va + e->size && !e->freed_seq) { snprintf(out, n, "%s", e->tag); return 1; }
  }
  return 0;
}

void tinynv_va_ledger_describe(uint64_t fva) {
  uint64_t n = val_total < VAL_N ? val_total : VAL_N;
  fprintf(stderr, "tinynv: VA LEDGER: fault address %#llx; %llu allocations recorded (%llu kept), %llu freed, now t=%.2f s\n",
          (unsigned long long)fva, (unsigned long long)val_total, (unsigned long long)n, (unsigned long long)val_frees,
          val_now());
  int inside = 0;
  const val_entry_t *below = NULL, *above = NULL;
  for (uint64_t k = 0; k < n; k++) {
    const val_entry_t *e = &val_ring[(val_total - 1 - k) % VAL_N];
    if (fva >= e->va && fva < e->va + e->size) {
      inside++;
      fprintf(stderr, "tinynv:   INSIDE #%llu [%#llx, %#llx) %llu bytes, \"%s\", allocated t=%.2f s, %s (fault is +%#llx into it)\n",
              (unsigned long long)e->seq, (unsigned long long)e->va, (unsigned long long)(e->va + e->size),
              (unsigned long long)e->size, e->tag, e->t_alloc,
              e->freed_seq ? "FREED" : "still live",
              (unsigned long long)(fva - e->va));
      if (e->freed_seq)
        fprintf(stderr, "tinynv:     freed t=%.2f s (%.2f s after it was allocated), %llu allocations ago\n", e->t_free,
                e->t_free - e->t_alloc, (unsigned long long)(val_total - e->seq));
    } else {
      if (e->va + e->size <= fva && (!below || e->va + e->size > below->va + below->size)) below = e;
      if (e->va > fva && (!above || e->va < above->va)) above = e;
    }
  }
  if (!inside) fprintf(stderr, "tinynv:   in NO recorded range - nothing was ever handed out here (within the %llu kept)\n",
                       (unsigned long long)n);
  if (below)
    fprintf(stderr, "tinynv:   nearest range below: #%llu [%#llx, %#llx) \"%s\" %s, fault is %#llx bytes past its end\n",
            (unsigned long long)below->seq, (unsigned long long)below->va, (unsigned long long)(below->va + below->size),
            below->tag, below->freed_seq ? "FREED" : "live", (unsigned long long)(fva - below->va - below->size));
  if (above)
    fprintf(stderr, "tinynv:   nearest range above: #%llu [%#llx, %#llx) \"%s\" %s, starts %#llx bytes after the fault\n",
            (unsigned long long)above->seq, (unsigned long long)above->va, (unsigned long long)(above->va + above->size),
            above->tag, above->freed_seq ? "FREED" : "live", (unsigned long long)(above->va - fva));
}

uint64_t tinynv_mm_alloc_va(tinynv_mm_t *mm, uint64_t size, uint64_t align) {
  if (refuse_if_wedged(mm)) return TINYNV_BAD_ADDR;
  // the oracle aligns to the largest power of two that fits inside the size, not the smallest that contains it, so a
  // range can start on a boundary the page tables can cover with big pages
  uint64_t sz = round_up(size, PAGE), want = 1;
  while ((want << 1) <= sz) want <<= 1;
  uint64_t va = tinynv_tlsf_alloc(mm->va, sz, align > want ? align : want);
  if (va == TINYNV_TLSF_FAIL) return tinynv_fail("no virtual address range of %llu bytes is free", (unsigned long long)sz)
                                     ? TINYNV_BAD_ADDR : TINYNV_BAD_ADDR;
  val_record(va, sz);
  return va;
}

int tinynv_mm_init(tinynv_mm_t *mm, tinynv_dev_t *dev) {
  memset(mm, 0, sizeof(*mm));
  mm->dev = dev;

  // The top of video memory is the firmware's - its write-protected region and the unprotected heap just below it -
  // so the manager never sees it. Derived from the sizes this driver hands the firmware (fw_layout.h) since
  // 2026-09-19; the flat 64 MB before that was related to none of them and left the top ~169 MB of this region inside
  // what the firmware owns, unnoticed only because no run had ever come within that of the top. The boot reads the
  // wpr2 registers afterwards and refuses the open if this arithmetic turns out short (tinynv_mm_check_fw_carveout).
  uint64_t managed = dev->vram_size - TINYNV_FW_RESERVE_TOP;
  mm->reserve_ptable = !dev->large_bar;

  uint64_t ptable_size = mm->reserve_ptable ? round_up(managed / 512, MB) : 0;
  uint64_t rest = 2 * MB + ptable_size;
  mm->boot = (tinynv_region_t){.name = "boot", .base = 0, .size = 2 * MB};
  mm->ptable = (tinynv_region_t){.name = "page table", .base = 2 * MB, .size = ptable_size};
  mm->pa = (tinynv_region_t){.name = "video memory", .base = rest, .size = managed - rest};
  mm->boot.alloc = tinynv_tlsf_new(mm->boot.size, mm->boot.base);
  mm->ptable.alloc = tinynv_tlsf_new(mm->ptable.size, mm->ptable.base);
  mm->pa.alloc = tinynv_tlsf_new(mm->pa.size, mm->pa.base);
  if (!mm->boot.alloc || !mm->ptable.alloc || !mm->pa.alloc) return tinynv_fail("out of memory for the physical allocators");

  // How the address space is cut up. Each shift is where one level's index starts, smallest page first; the third
  // generation reaches 57 bits through six levels and the second 48 through five. A level's entry count follows from the
  // gap to the next shift, and both tables are stored root first, which is the reverse of the order they are written in.
  static const int V3[] = {12, 21, 29, 38, 47, 56}, V2[] = {12, 21, 29, 38, 47};
  const int *shifts = dev->mmu_ver == 3 ? V3 : V2;
  mm->va_bits = dev->mmu_ver == 3 ? 56 : 48;
  mm->levels = dev->mmu_ver == 3 ? 6 : 5;
  for (int i = 0; i < mm->levels; i++) {
    int from_root = mm->levels - 1 - i;
    int next = i + 1 < mm->levels ? shifts[i + 1] : mm->va_bits + 1;
    mm->pte_covers[from_root] = 1ull << shifts[i];
    mm->pte_cnt[from_root] = 1u << (next - shifts[i]);
  }
  mm->va_base = 0;
  if (!(mm->va = tinynv_tlsf_new(1ull << 44, 0x1000000000ull))) return tinynv_fail("out of memory for the address space");

  // the root of the page tables is the first thing allocated, and it must start zeroed or the firmware walks garbage
  mm->root_page_table = tinynv_mm_palloc(mm, PAGE, PAGE, 1, TINYNV_REGION_BOOT);
  if (mm->root_page_table == TINYNV_BAD_ADDR) return -1;
  return 0;
}

// Allocate what the firmware reads while it boots.
//
// This mirrors the oracle's _alloc_boot_mem exactly, including the choice it makes when nobody says where the memory
// should go: with a window too small to see all of video memory, host memory is the only place the processor and the GPU
// can both reach, so that is where everything on the boot path ends up on a thunderbolt card.
int tinynv_alloc_boot_mem(tinynv_mm_t *mm, size_t size, const void *data, int sysmem, tinynv_bootmem_t *out) {
  tinynv_dev_t *d = mm->dev;
  uint64_t sz = round_up(size, PAGE);
  memset(out, 0, sizeof(*out));
  out->paddr = TINYNV_BAD_ADDR;
  out->size = sz;

  if (sysmem < 0) sysmem = !d->large_bar;
  if (sysmem) {
    // the unrounded size goes to the backend, because that is the length the oracle asks for and the recording checks
    if (d->pci->dma_alloc(d->pci, size, &out->dma)) return -1;
    out->in_sysmem = 1;
    out->view = out->dma.view;
    out->addrs = out->dma.pages;
    out->naddrs = out->dma.npages;
    out->size = out->dma.size;
  } else {
    // zeroed, like every other physical allocation: the oracle's palloc zeroes by default and this is the one caller
    // that reaches it, so leaving it out meant handing the firmware a method buffer full of whatever was there.
    // The allocation comes before the window is asked about, which is the oracle's order and therefore the recording's.
    if ((out->paddr = tinynv_mm_palloc(mm, sz, PAGE, 1, TINYNV_REGION_VRAM)) == TINYNV_BAD_ADDR) return -1;
    uint64_t base = 0;
    if (d->pci->bar_info(d->pci, 1, &base, NULL)) return -1;
    out->view = d->vram;
    out->view.off += out->paddr;
    out->view.size = sz;
    out->naddrs = sz / PAGE;
    if (!(out->addrs = calloc(out->naddrs, sizeof(uint64_t)))) return tinynv_fail("out of memory for %zu page addresses", out->naddrs);
    // the oracle asks where the window is once per page, building the list in a comprehension; in faithful mode so does
    // this, because a question not asked is still a difference in the stream
    for (size_t i = 0; i < out->naddrs; i++) {
      if (tinynv_is_faithful() && i && d->pci->bar_info(d->pci, 1, NULL, &base)) return -1;
      out->addrs[i] = base + out->paddr + i * PAGE;
    }
  }

  if (data) nv_wr_block(&out->view, 0, data, size);
  return 0;
}

void tinynv_free_boot_mem(tinynv_mm_t *mm, tinynv_bootmem_t *m) {
  if (m->in_sysmem) mm->dev->pci->dma_free(mm->dev->pci, &m->dma);
  else free(m->addrs);
  memset(m, 0, sizeof(*m));
  m->paddr = TINYNV_BAD_ADDR;
}

int tinynv_mm_check_fw_carveout(const tinynv_mm_t *mm, uint64_t wpr2_lo, uint64_t wpr2_hi, char *line, size_t n) {
  const uint64_t vram = mm->dev->vram_size, top = mm->pa.base + mm->pa.size;
  if (!wpr2_lo && !wpr2_hi) {
    snprintf(line, n, "chip did not expose wpr2 (registers read 0); reservation unverified this run");
    return 0;
  }
  // Nonsense is refused rather than trusted: a region that is not inside video memory, or nowhere near the size of
  // the firmware image plus its heap, is a misread register, and a check that quietly passed on garbage would be no
  // check. The bounds are the image's and the reservation's own.
  const uint64_t span = wpr2_hi > wpr2_lo ? wpr2_hi - wpr2_lo : 0;
  if (wpr2_lo >= wpr2_hi || wpr2_hi > vram || span < TINYNV_FW_HEAP_SIZE + TINYNV_FW_IMAGE_BOUND / 2 ||
      span > TINYNV_FW_RESERVE_TOP)
    return tinynv_fail("the wpr2 registers read %#llx..%#llx, which is not a %llu-%llu MB region inside %llu MB of "
                       "video memory - a misread, not a layout",
                       (unsigned long long)wpr2_lo, (unsigned long long)wpr2_hi,
                       (unsigned long long)((TINYNV_FW_HEAP_SIZE + TINYNV_FW_IMAGE_BOUND / 2) >> 20),
                       (unsigned long long)(TINYNV_FW_RESERVE_TOP >> 20), (unsigned long long)(vram >> 20));
  // The unprotected heap sits just below the region (gsp_fw_wpr_meta.h), and it is the first thing an allocator that
  // bumps upward would reach: the bound is its bottom, not wpr2's.
  const uint64_t fw_low = wpr2_lo - TINYNV_FW_NONWPR_HEAP;
  if (top > fw_low)
    return tinynv_fail("the manager stops at %#llx but the firmware owns from %#llx (wpr2 %#llx..%#llx with %llu KB "
                       "of non-wpr heap below it): raise TINYNV_FW_RESERVE_TOP in fw_layout.h",
                       (unsigned long long)top, (unsigned long long)fw_low, (unsigned long long)wpr2_lo,
                       (unsigned long long)wpr2_hi, (unsigned long long)(TINYNV_FW_NONWPR_HEAP >> 10));
  snprintf(line, n, "chip says wpr2 %#llx..%#llx (%.1f MB) with %.1f MB of non-wpr heap below it; the manager stops "
                    "%.1f MB under that",
           (unsigned long long)wpr2_lo, (unsigned long long)wpr2_hi, (double)span / (double)MB,
           (double)TINYNV_FW_NONWPR_HEAP / (double)MB, (double)(fw_low - top) / (double)MB);
  return 0;
}

int tinynv_mm_reserve_bar_pool(tinynv_mm_t *mm) {
  tinynv_dev_t *d = mm->dev;
  uint64_t bar_size = 0;
  if (d->pci->bar_info(d->pci, 1, NULL, &bar_size)) return -1;
  if (bar_size >= d->vram_size) return 0; // the whole of video memory is visible; nothing needs reserving

  // Four megabytes, because that is what the oracle reserved and the recorded boot is reproduced write for write. The
  // command arena needs its own room in video memory the processor can write, and it takes it after the boot is over -
  // see tinynv_mm_reserve_cpu_region - rather than by widening this and diverging from the recording.
  uint64_t size = 4ull << 20;
  uint64_t paddr = tinynv_mm_palloc(mm, size, PAGE, 1, TINYNV_REGION_VRAM);
  if (paddr == TINYNV_BAD_ADDR) return -1;
  if (d->pci->bar_info(d->pci, 1, NULL, &bar_size)) return -1;
  if (paddr + size > bar_size)
    return tinynv_fail("the cpu visible pool [%#llx, %#llx) does not fit the %#llx byte window",
                       (unsigned long long)paddr, (unsigned long long)(paddr + size), (unsigned long long)bar_size);
  mm->bar_pool_base = mm->bar_pool_next = paddr;
  mm->bar_pool_size = size;
  return 0;
}

// Another run of video memory the processor can write, taken after the boot rather than during it. The boot is
// reproduced against a recording write for write, so the reserve it makes has to stay the size the oracle made it;
// anything this driver wants for its own reasons - the command arena, which lives in video memory now so that the
// engine reads descriptors without crossing the link - is taken here, where there is no recording to diverge from.
int tinynv_mm_reserve_cpu_region(tinynv_mm_t *mm, uint64_t size) {
  tinynv_dev_t *d = mm->dev;
  uint64_t bar_size = 0;
  if (d->pci->bar_info(d->pci, 1, NULL, &bar_size)) return -1;
  if (bar_size >= d->vram_size) return 0;   // all of video memory is writable; nothing needs reserving

  size = round_up(size, PAGE);
  uint64_t paddr = tinynv_mm_palloc(mm, size, PAGE, 1, TINYNV_REGION_VRAM);
  if (paddr == TINYNV_BAD_ADDR) return -1;
  if (paddr + size > bar_size)
    return tinynv_fail("a %#llx byte processor visible region at %#llx does not fit the %#llx byte window - the window "
                       "is where the processor and the engine meet, and past its end writes go nowhere",
                       (unsigned long long)size, (unsigned long long)paddr, (unsigned long long)bar_size);
  mm->exec_pool_base = mm->exec_pool_next = paddr;
  mm->exec_pool_size = size;
  return 0;
}
