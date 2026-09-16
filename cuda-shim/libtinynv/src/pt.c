// The page tables the GPU walks to turn a virtual address into a physical one.
//
// Five or six levels of table, each level covering a fixed span of the address space, and the walk stops as soon as one
// entry can cover what is being mapped. That is what makes a 512 MB mapping cost one entry rather than 131072 of them:
// at a level whose entries each span 512 MB, an aligned 512 MB range fits in one. The traversal below is the oracle's,
// which descends only as far as it must and creates tables on the way down.
//
// Two wrinkles are NVIDIA's rather than ours. One level holds a pair of entries per slot, for big and small pages, so
// its entries are 128 bits and some of its fields start past bit 63. And the levels are numbered from the root, while
// the spans are listed from the smallest page up, so the two are reversed against each other.
#include "internal.h"
#include "mmu.h"
#include "nv_regs.h"
#include "nv_structs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PAGE 0x1000ull
_Static_assert(PAGE == TINYNV_MAP_PAGE, "this file's page size and the one guarantee 3 refuses against "
                                        "have diverged, so a mapping could be aligned for one and not the other");

// how many entries a table at this level holds, and how much address space each of them covers
static uint64_t pte_covers(tinynv_mm_t *mm, int lv) { return mm->pte_covers[lv]; }
static uint32_t pte_count(tinynv_mm_t *mm, int lv) { return mm->pte_cnt[lv]; }

// the level whose entries are 128 bits, holding a big-page and a small-page half
static int is_dual(tinynv_mm_t *mm, int lv) { return lv == mm->levels - 2; }

static uint32_t pte_index(tinynv_mm_t *mm, int lv, uint64_t va) {
  return (uint32_t)((va / pte_covers(mm, lv)) % pte_count(mm, lv));
}

static uint64_t entry_addr(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint32_t idx) {
  return pt->paddr + (uint64_t)idx * (is_dual(mm, pt->lv) ? 16 : 8);
}

// A 128 bit entry is two 8 byte accesses, not one 16 byte one, and the order is load bearing rather than incidental.
// The half carrying the valid bit is written last, so an entry never exists in a state where the GPU could follow it to
// an address that has not been stored yet; reads take the same two halves in the opposite order.
//
// This matters most on the path it is hardest to see. Over the socket backend each write is its own message, delivered
// in order, so two writes really do put the valid half last from the GPU's point of view. One 16 byte write would be a
// single memcpy into the window on the far side with no ordering promise inside it at all - which looks obviously
// correct and would leave a moment where a half written entry is followable. (Session A's point, and the better half of
// the reason to mirror the recording here rather than tidy it.)
static void entry_read(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint32_t idx, uint64_t w[2]) {
  uint64_t at = entry_addr(mm, pt, idx);
  w[0] = w[1] = 0;
  if (!is_dual(mm, pt->lv)) { nv_rd_block(&mm->dev->vram, at, &w[0], 8); return; }
  nv_rd_block(&mm->dev->vram, at + 8, &w[1], 8);
  nv_rd_block(&mm->dev->vram, at, &w[0], 8);
}

static void entry_write(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint32_t idx, const uint64_t w[2]) {
  uint64_t at = entry_addr(mm, pt, idx);
  nv_wr_block(&mm->dev->vram, at, &w[0], 8);
  if (is_dual(mm, pt->lv)) nv_wr_block(&mm->dev->vram, at + 8, &w[1], 8);
}

// the lowest level holds pages by definition; above it, an entry says whether it is a page or a table
int tinynv_pt_is_page(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint32_t idx) {
  if (pt->lv >= mm->levels - 1) return 1;
  uint64_t w[2];
  entry_read(mm, pt, idx, w);
  return (int)(w[0] & 1);
}

// Does this leaf entry allow writes? Guarantee 7's grant half is only checkable by reading back what was actually
// programmed - the flag being passed in correctly is a different claim from the entry saying so, and until
// 2026-09-15 there was no encoding that could say it at all. Not test scaffolding: this is the question an audit of
// the page tables asks, and the answer has to come from the bits rather than from what the caller intended.
//
// Returns 1 writable, 0 read-only, -1 if this is not a valid leaf entry - three states, because "not a page" and
// "read-only" must not look the same to a caller deciding whether a guest can write somewhere.
int tinynv_pt_writable(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint32_t idx) {
  uint64_t w[2];
  if (!tinynv_pt_is_page(mm, pt, idx)) return -1;
  entry_read(mm, pt, idx, w);
  if (!NV_GET_E(w, NV_MMU_VER3_PTE, VALID)) return -1;
  uint32_t pcf = (uint32_t)NV_GET_E(w, NV_MMU_VER3_PTE, PCF);
  return !(pcf == TINYNV_PCF_RO_CACHED || pcf == TINYNV_PCF_RO_UNCACHED);
}

int tinynv_pt_valid(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint32_t idx) {
  uint64_t w[2];
  entry_read(mm, pt, idx, w);
  // The oracle reads this entry three times to answer the same question, because its accessor re-reads inside the test
  // that decides how to decode. In faithful mode so does this, so the streams match operation for operation. Not at the
  // lowest level: there the oracle's test answers "a page" without reading anything, so there is nothing to mirror.
  if (tinynv_is_faithful() && pt->lv < mm->levels - 1) entry_read(mm, pt, idx, w);
  if (tinynv_pt_is_page(mm, pt, idx)) return (int)NV_GET_E(w, NV_MMU_VER3_PTE, VALID);
  return is_dual(mm, pt->lv) ? NV_GET_E(w, NV_MMU_VER3_DUAL_PDE, APERTURE_SMALL) != 0
                             : NV_GET_E(w, NV_MMU_VER3_PDE, APERTURE) != 0;
}

static uint64_t entry_target(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint32_t idx) {
  uint64_t w[2];
  if (tinynv_is_faithful()) entry_read(mm, pt, idx, w); // the oracle's accessor reads once more than it needs to
  entry_read(mm, pt, idx, w);
  if (is_dual(mm, pt->lv)) return NV_GET_E(w, NV_MMU_VER3_DUAL_PDE, ADDRESS_SMALL) << 12;
  return NV_GET_E(w, NV_MMU_VER3_PDE, ADDRESS) << 12;
}

// an entry at this level can cover a whole span only if the level is near the bottom and the physical address lines up
static int supports_huge_page(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint64_t paddr) {
  return pt->lv >= mm->levels - 3 && paddr % pte_covers(mm, pt->lv) == 0;
}

void tinynv_pt_set_entry(tinynv_mm_t *mm, const tinynv_pt_t *pt, uint32_t idx, uint64_t paddr, int table,
                         uint32_t flags, int valid) {
  uint64_t w[2] = {0, 0};
  if (!table) {
    uint32_t pcf = (flags & TINYNV_PTE_READONLY) ? ((flags & TINYNV_PTE_UNCACHED) ? TINYNV_PCF_RO_UNCACHED : TINYNV_PCF_RO_CACHED)
                                                 : ((flags & TINYNV_PTE_UNCACHED) ? TINYNV_PCF_RW_UNCACHED : TINYNV_PCF_RW_CACHED);
    NV_PUT(w, NV_MMU_VER3_PTE, VALID, valid);
    NV_PUT(w, NV_MMU_VER3_PTE, ADDRESS_SYS, paddr >> 12);
    NV_PUT(w, NV_MMU_VER3_PTE, APERTURE, (flags & TINYNV_PTE_SYSMEM) ? 2 : 0);
    NV_PUT(w, NV_MMU_VER3_PTE, KIND, 6);
    NV_PUT(w, NV_MMU_VER3_PTE, PCF, pcf);
  } else if (is_dual(mm, pt->lv)) {
    NV_PUT(w, NV_MMU_VER3_DUAL_PDE, IS_PTE, 0);
    NV_PUT(w, NV_MMU_VER3_DUAL_PDE, APERTURE_SMALL, valid ? 1 : 0);
    NV_PUT(w, NV_MMU_VER3_DUAL_PDE, ADDRESS_SMALL, paddr >> 12);
    NV_PUT(w, NV_MMU_VER3_DUAL_PDE, PCF_SMALL, 0b10);
  } else {
    NV_PUT(w, NV_MMU_VER3_PDE, IS_PTE, 0);
    NV_PUT(w, NV_MMU_VER3_PDE, APERTURE, valid ? 1 : 0);
    NV_PUT(w, NV_MMU_VER3_PDE, ADDRESS, paddr >> 12);
    NV_PUT(w, NV_MMU_VER3_PDE, PCF, 0b10);
  }
  entry_write(mm, pt, idx, w);
}

// `root` rather than mm->root_page_table, because a guest's tree is not the driver's. C4b gives a guest its own
// page-table ROOT rather than a window in ours - see docs/driver/libtinynv-design.md §4g - so that a guest VA cannot
// name our memory by construction instead of by comparison. Every caller passes mm->root_page_table today and nothing
// changes; the parameter exists so the second tree is a caller's choice rather than a second copy of this code.
void tinynv_pt_walk_begin(tinynv_pt_walk_t *c, tinynv_mm_t *mm, uint64_t root, uint64_t vaddr, int create) {
  memset(c, 0, sizeof(*c));
  c->mm = mm;
  c->vaddr = vaddr - mm->va_base;
  c->create = create;
  c->depth = 1;
  c->stack[0].pt = (tinynv_pt_t){.paddr = root, .lv = 0};
  c->stack[0].idx = pte_index(mm, 0, c->vaddr);
  c->stack[0].covers = pte_covers(mm, 0);
}

// Step into the table this entry points at, creating and zeroing one if there is nothing there yet.
static int level_down(tinynv_pt_walk_t *c) {
  tinynv_mm_t *mm = c->mm;
  tinynv_pt_frame_t *f = &c->stack[c->depth - 1];
  if (c->depth >= (int)(sizeof(c->stack) / sizeof(*c->stack)))
    return tinynv_fail("the page table walk is deeper than %d levels", c->depth);

  if (!tinynv_pt_valid(mm, &f->pt, f->idx)) {
    if (!c->create) return tinynv_fail("virtual address %#llx has no page table at level %d",
                                       (unsigned long long)(c->vaddr + mm->va_base), f->pt.lv);
    // Counted, because a guest can choose the shape that makes this expensive and cannot be billed for it otherwise.
    //
    // Guarantee 10 named the dext's 128 mappings as the resource a guest could exhaust. Session C measured that the
    // whole VM spends ONE of those, for the slab, and the guest subdivides it - so a guest cannot reach that budget by
    // asking. The resource it CAN exhaust is this one: every mapping programs entries, entries live in tables this
    // driver allocates, and a table appears at each level a range does not already have. 47 MiB mapped as one range
    // costs a handful; the same 47 MiB as ten thousand 4 KiB ranges far apart costs thousands - and scattered is the
    // cheap direction for the guest and the expensive one for us.
    //
    // A count before a cap, deliberately. The cap wants a number and the number should come off a measurement of real
    // mapping shapes rather than out of my arithmetic, which is how the held-list size nearly got sized against the
    // wrong field this morning.
    mm->tables_made++;
    uint64_t child = tinynv_mm_palloc(mm, PAGE, PAGE, 1, TINYNV_REGION_PTABLE);
    if (child == TINYNV_BAD_ADDR) return -1;
    tinynv_pt_set_entry(mm, &f->pt, f->idx, child, 1, 0, 1);
  }
  if (tinynv_pt_is_page(mm, &f->pt, f->idx))
    return tinynv_fail("level %d entry %u is a page where a table was needed", f->pt.lv, f->idx);

  int lv = f->pt.lv + 1;
  tinynv_pt_frame_t *n = &c->stack[c->depth++];
  n->pt = (tinynv_pt_t){.paddr = entry_target(mm, &f->pt, f->idx), .lv = lv};
  n->idx = pte_index(mm, lv, c->vaddr);
  n->covers = pte_covers(mm, lv);
  return 0;
}

// Come back up past any table whose entries this walk has used up, stepping the parent on as it goes.
static void level_up(tinynv_pt_walk_t *c) {
  while (c->depth > 1 && c->stack[c->depth - 1].idx == pte_count(c->mm, c->stack[c->depth - 1].pt.lv)) {
    c->depth--;
    c->stack[c->depth - 1].idx++;
  }
}

// The next run of entries at one level that can cover part of this mapping. Returns 1 when it produced a run, 0 when the
// mapping is fully described, and -1 on failure.
int tinynv_pt_walk_next(tinynv_pt_walk_t *c, uint64_t *size, uint64_t *off, uint64_t paddr, tinynv_pt_run_t *out) {
  tinynv_mm_t *mm = c->mm;
  if (!*size) return 0;

  tinynv_pt_frame_t *f = &c->stack[c->depth - 1];
  if (c->create) {
    // descend until one entry covers no more than what is left, is allowed to be a page, and lines up with the address
    while (f->covers > *size || !supports_huge_page(mm, &f->pt, paddr + *off) || (c->vaddr & (f->covers - 1)) != 0) {
      if (level_down(c)) return -1;
      f = &c->stack[c->depth - 1];
    }
  } else {
    // looking rather than building: follow tables that already exist, and stop at the first thing that is not one
    while (!tinynv_pt_is_page(mm, &f->pt, f->idx) && tinynv_pt_valid(mm, &f->pt, f->idx)) {
      if (level_down(c)) return -1;
      f = &c->stack[c->depth - 1];
    }
  }

  uint64_t want = *size / f->covers, room = pte_count(mm, f->pt.lv) - f->idx;
  uint32_t entries = (uint32_t)(want < room ? want : room);
  // a walk that is only looking always reports at least one entry, so an unmapped range still yields somewhere to check
  if (!entries && !c->create) entries = 1;
  if (!entries) return tinynv_fail("a page table level covering %#llx cannot describe %#llx bytes",
                                   (unsigned long long)f->covers, (unsigned long long)*size);

  *out = (tinynv_pt_run_t){.pt = f->pt, .idx = f->idx, .entries = entries, .covers = f->covers, .off = *off};
  // A run can cover more than was asked for: a walk that is only looking reports one entry even when that entry spans
  // far more than the range. The oracle lets its remaining size go negative and stops; unsigned arithmetic here would
  // wrap instead and walk the rest of the address space, so it saturates at zero.
  uint64_t step = (uint64_t)entries * f->covers;
  *size = step >= *size ? 0 : *size - step;
  *off += step;
  c->vaddr += step;
  f->idx += entries;
  level_up(c);
  return 1;
}

// The chain of tables a range lives under, root first. Creating them is the point: GSP-RM is handed these addresses so
// that it and the driver agree on the tables from the start, rather than discovering each other's later.
int tinynv_mm_page_tables(tinynv_mm_t *mm, uint64_t root, uint64_t vaddr, uint64_t size, tinynv_pt_t *out, int *n) {
  tinynv_pt_walk_t c;
  tinynv_pt_walk_begin(&c, mm, root, vaddr, 1);
  tinynv_pt_run_t run;
  uint64_t left = size, off = 0;
  int rc = tinynv_pt_walk_next(&c, &left, &off, 0, &run);
  if (rc < 0) return -1;
  if (!rc) return tinynv_fail("a zero sized range has no page tables");
  for (int i = 0; i < c.depth; i++) out[i] = c.stack[i].pt;
  *n = c.depth;
  return 0;
}


// Map physical memory at a virtual address.
//
// Two passes, as the oracle does them. The first only looks, and refuses if anything in the range is already mapped -
// silently mapping over a live range is the kind of fault that surfaces a long way from its cause. The second builds the
// tables and writes the entries.
// Is any part of this range already mapped?
//
// Asking that needs one walk per entry, not one walk per run. A walk reports a run of neighbouring entries, but it only
// descended for the first of them: the rest are at that level whatever they are, and at a level above the leaves a
// valid entry is a page table, not a page. Page tables are never freed - an allocation that is unmapped leaves its
// empty tables behind for the next one - so treating "valid" as "mapped" across a run reports a collision with a table
// that describes nothing at all.
//
// That is not theoretical: it refused thirteen allocations in a real ggml run, on non-round sizes that happened to span
// a 2 MB region whose table had outlived its pages. The address it named was genuinely free, which is what made it
// confusing - the driver was right that something was there and wrong about what.
//
// Walking per entry is not slow: an invalid entry high in the tree skips everything beneath it in one step.
static int range_is_mapped(tinynv_mm_t *mm, uint64_t root, uint64_t vaddr, uint64_t size, uint64_t *where) {
  tinynv_pt_walk_t look;
  tinynv_pt_walk_begin(&look, mm, root, vaddr, 0);
  uint64_t left = size, off = 0;
  tinynv_pt_run_t run;
  int rc;
  while ((rc = tinynv_pt_walk_next(&look, &left, &off, 0, &run)) > 0)
    for (uint32_t e = 0; e < run.entries; e++) {
      if (!tinynv_pt_valid(mm, &run.pt, run.idx + e)) continue;
      uint64_t at = vaddr + run.off + (uint64_t)e * run.covers;
      if (tinynv_pt_is_page(mm, &run.pt, run.idx + e)) {
        if (where) *where = at;
        return 1;
      }
      // Valid and not a page: a table an earlier mapping left behind. Whether this range is free depends on what is
      // under it, so look there - and only here, which is why the reads this makes are the reads it always made
      // except in the case that used to be answered wrongly.
      uint64_t sub = run.covers < size ? run.covers : size;
      int deeper = range_is_mapped(mm, root, at, sub, where);
      if (deeper) return deeper;
    }
  return rc < 0 ? -1 : 0;
}

// Tell the GPU to forget the translations it has cached, and WAIT for it to have done so.
//
// Two things were missing here and both are the same hazard this driver already found once, in tinynv_submit.
//
// The entries go to video memory and this register is in the other window. Ordering holds along a path and not between
// two of them, so without a read on the entry path first, the invalidate can execute AHEAD of the writes it is meant to
// make visible - and the GPU then re-caches exactly the stale translations this call exists to remove. Reading one of
// the entries back is the fence, and it cannot pass the writes before it.
//
// And bit 31 is TRIGGER, not a doorbell: the hardware clears it when the invalidate has completed, which is why an ACK
// field exists at 8:7. Returning without waiting means the caller is free to hand the underlying page back to the
// allocator while the GPU is still able to reach it. On a host page that is not a stale mapping, it is a DMA into
// memory that now belongs to something else - and what lands there is whatever the engine was writing, which is why
// this was found as tensor floats inside another library's hash table rather than as a fault.
static int mmu_invalidate(tinynv_mm_t *mm, uint64_t root) {
  // The fence. Any read on the video-memory path serves; the page directory's own root is one this driver always has.
  (void)nv_rd32(&mm->dev->vram, root);
  tinynv_wr32(mm->dev, NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE,
              (1u << 0) | (1u << 1) | (1u << 6) | (1u << 31));
  // Then wait for it. A generous deadline, because the cost of being wrong in the other direction is a page handed back
  // while the engine can still write to it, and this is not on the launch path - it happens when a mapping changes.
  double deadline = tinynv_now_s() + 1.0;
  for (;;) {
    if (!(tinynv_rd32(mm->dev, NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE) & (1u << 31))) return 0;
    if (tinynv_now_s() > deadline)
      return tinynv_fail("the mmu did not finish invalidating within a second; a page freed now could still be written "
                         "by the engine");
  }
}

int tinynv_mm_map_range(tinynv_mm_t *mm, uint64_t root, uint64_t vaddr, uint64_t size, const tinynv_paddr_range_t *paddrs, int npaddrs,
                        uint32_t flags) {
  uint64_t total = 0;
  for (int i = 0; i < npaddrs; i++) total += paddrs[i].size;
  if (total != size) return tinynv_fail("mapping %#llx bytes from ranges totalling %#llx", (unsigned long long)size,
                                        (unsigned long long)total);

  uint64_t clash = 0;
  int busy = range_is_mapped(mm, root, vaddr, size, &clash);
  if (busy < 0) return -1;
  if (busy)
    return tinynv_fail("mapping %#llx+%#llx: %#llx is already mapped", (unsigned long long)vaddr,
                       (unsigned long long)size, (unsigned long long)clash);
  int rc;
  tinynv_pt_run_t run;
  uint64_t left, off;

  tinynv_pt_walk_t build;
  tinynv_pt_walk_begin(&build, mm, root, vaddr, 1);
  for (int i = 0; i < npaddrs; i++) {
    left = paddrs[i].size;
    off = 0;
    while ((rc = tinynv_pt_walk_next(&build, &left, &off, paddrs[i].paddr, &run)) > 0)
      for (uint32_t e = 0; e < run.entries; e++)
        tinynv_pt_set_entry(mm, &run.pt, run.idx + e, paddrs[i].paddr + run.off + (uint64_t)e * run.covers, 0,
                            flags, 1);
    if (rc < 0) return -1;
  }

  // Mapping stays fire-and-forget on purpose, and the asymmetry is the point. If this invalidate runs ahead of the
  // entries it publishes, the gpu re-caches an absent translation and the next access FAULTS - loud, attributable, and
  // something this driver reports. Unmapping has the opposite failure and it is silent, so that is where the cost of
  // fencing and waiting is worth paying. Keeping this one as it was also keeps the recorded boot valid as an oracle:
  // the boot maps constantly and unmaps nothing, so a read added here would diverge from a recording of a driver that
  // never took it.
  tinynv_wr32(mm->dev, NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE, (1u << 0) | (1u << 1) | (1u << 6) | (1u << 31));
  return 0;
}

// Take a range out of the page tables. The entries are cleared before the memory behind them is handed back, in that
// order and not the other way round: an entry left pointing at memory that now belongs to something else turns a kernel
// reading past the end of its buffer into corruption of live data instead of a fault.
int tinynv_mm_unmap_range(tinynv_mm_t *mm, uint64_t root, uint64_t vaddr, uint64_t size) {
  tinynv_pt_walk_t w;
  tinynv_pt_walk_begin(&w, mm, root, vaddr, 0);
  uint64_t left = size, off = 0;
  tinynv_pt_run_t run;
  int rc;
  while ((rc = tinynv_pt_walk_next(&w, &left, &off, 0, &run)) > 0)
    for (uint32_t e = 0; e < run.entries; e++) tinynv_pt_set_entry(mm, &run.pt, run.idx + e, 0, 0, 0, 0);
  if (rc < 0) return -1;
  return mmu_invalidate(mm, root);
}

// Which region a physical address came out of, so it goes back to the allocator that handed it out.
static tinynv_region_t *region_of(tinynv_mm_t *mm, uint64_t paddr) {
  tinynv_region_t *r[3] = {&mm->boot, &mm->ptable, &mm->pa};
  for (int i = 0; i < 3; i++)
    if (r[i]->alloc && paddr >= r[i]->base && paddr < r[i]->base + r[i]->size) return r[i];
  return NULL;
}

// Give a buffer back: the mapping, the addresses, and the memory itself.
//
// Until the driver started serving a caller that allocates and frees, nothing was ever freed - the boot allocates once
// and keeps everything - and this released nothing at all. That is survivable in a boot and not in a driver under
// llama.cpp, which allocates and frees continuously: it would run for a while and then stop, out of address space
// rather than out of memory, which is a confusing way to fail.
void tinynv_vmap_free(tinynv_mm_t *mm, tinynv_vmap_t *m) {
  if (!m->size) { free(m->ranges); memset(m, 0, sizeof(*m)); return; }
  // The one failure this call can report is the one that matters here, and it used to be discarded. mmu_invalidate
  // waits a second for the MMU to acknowledge and, on timeout, says "a page freed now could still be written by the
  // engine" - and then this function released the VA and handed the physical pages back to the allocator anyway.
  //
  // The page tables are correct in that case: the entries ARE cleared before the invalidate. What is stale is a TLB,
  // over memory that has been recycled. Session C demonstrated it rather than arguing it - stall the modelled MMU,
  // free 2 MiB, un-stall, allocate the same size, and the same physical address comes straight back out. Today the
  // blast radius is internal because this driver only frees its own allocations; under a guest it is guarantee 5
  // exactly, pages handed to one tenant while another's GPU can still reach them.
  //
  // So the pages are NOT returned. Leaking them is the safe direction: a leak is visible, bounded and recoverable,
  // and the alternative is memory that something may still write. The device is also marked wedged, because an MMU
  // that has not answered in a second is not going to be fine for the next allocation, and pretending otherwise turns
  // one loud failure into a quiet series of them.
  //
  // Not fatal, deliberately. This is a library inside somebody's process, and aborting it is a decision for the
  // program that owns the work, not for a free path. Refusing everything afterwards says the same thing without
  // taking the choice away.
  if (tinynv_mm_unmap_range(mm, mm->root_page_table, m->va, m->size)) {
    mm->wedged = 1;
    mm->leaked_bytes += m->size;
    mm->leaked_n++;
    tinynv_fail("the mapping at %#llx could not be torn down, so its %#llx bytes are being LEAKED rather than reused - "
                "returning them would hand out memory the gpu may still reach through a translation nobody confirmed "
                "was gone. every allocation after this is refused (%llu bytes leaked in %llu mappings so far)",
                (unsigned long long)m->va, (unsigned long long)m->size, (unsigned long long)mm->leaked_bytes,
                (unsigned long long)mm->leaked_n);
    free(m->ranges);
    memset(m, 0, sizeof(*m));
    return;
  }
  tinynv_tlsf_release(mm->va, m->va);

  if (m->in_sysmem) mm->dev->pci->dma_free(mm->dev->pci, &m->dma);
  else if (!m->in_bar_pool)
    // The processor-visible reserve is the exception: it is handed out by a bump pointer with nowhere to put anything
    // back, so memory from it is kept for the life of the device by design.
    for (int i = 0; i < m->nranges; i++) {
      tinynv_region_t *r = region_of(mm, m->ranges[i].paddr);
      if (r) tinynv_tlsf_release(r->alloc, m->ranges[i].paddr);
    }

  free(m->ranges);
  memset(m, 0, sizeof(*m));
}

// The chunk sizes a non-contiguous allocation is built from, largest first. Taking the biggest piece that fits keeps the
// page tables shallow: a 2 MB chunk on a 2 MB boundary is one entry rather than five hundred and twelve.
static const uint64_t PALLOC_RANGES[] = {512ull << 20, 2ull << 20, 4ull << 10};
#define NPALLOC_RANGES (sizeof(PALLOC_RANGES) / sizeof(*PALLOC_RANGES))

// Physical memory, a virtual address for it, and the mapping between them.
int tinynv_mm_valloc(tinynv_mm_t *mm, uint64_t size, uint64_t align, int contiguous, int uncached, tinynv_vmap_t *out) {
  memset(out, 0, sizeof(*out));
  size = (size + PAGE - 1) / PAGE * PAGE;
  uint64_t va = tinynv_mm_alloc_va(mm, size, align);
  if (va == TINYNV_BAD_ADDR) return -1;
  out->va = va;
  out->size = size;

  if (contiguous) {
    uint64_t paddr = tinynv_mm_palloc(mm, size, PAGE, 1, TINYNV_REGION_VRAM);
    if (paddr == TINYNV_BAD_ADDR) return -1;
    if (!(out->ranges = calloc(1, sizeof(*out->ranges)))) return tinynv_fail("out of memory for an allocation's ranges");
    out->nranges = 1;
    out->ranges[0] = (tinynv_paddr_range_t){.paddr = paddr, .size = size};
    return tinynv_mm_map_range(mm, mm->root_page_table, va, size, out->ranges, 1, uncached ? TINYNV_PTE_UNCACHED : 0);
  }

  size_t cap = 16;
  if (!(out->ranges = calloc(cap, sizeof(*out->ranges)))) return tinynv_fail("out of memory for an allocation's ranges");
  uint64_t left = size;
  size_t which = 0;
  while (left) {
    while (which < NPALLOC_RANGES && PALLOC_RANGES[which] > left) which++;
    if (which == NPALLOC_RANGES) return tinynv_fail("%#llx bytes cannot be built from the chunk sizes available",
                                                    (unsigned long long)left);
    uint64_t chunk = PALLOC_RANGES[which];
    uint64_t paddr = tinynv_mm_palloc(mm, chunk, chunk, 0, TINYNV_REGION_VRAM);
    if (paddr == TINYNV_BAD_ADDR) { which++; continue; } // no room at this size; fall back to a smaller piece
    if ((size_t)out->nranges == cap) {
      tinynv_paddr_range_t *bigger = realloc(out->ranges, (cap *= 2) * sizeof(*out->ranges));
      if (!bigger) return tinynv_fail("out of memory for an allocation's ranges");
      out->ranges = bigger;
    }
    out->ranges[out->nranges++] = (tinynv_paddr_range_t){.paddr = paddr, .size = chunk};
    left -= chunk;
  }
  return tinynv_mm_map_range(mm, mm->root_page_table, va, size, out->ranges, out->nranges, uncached ? TINYNV_PTE_UNCACHED : 0);
}

static uint64_t round_up_u(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

int tinynv_mm_alloc_buffer(tinynv_mm_t *mm, uint64_t size, int host, int cpu_access, int uncached, int force_devmem,
                           int zero, tinynv_vmap_t *out) {
  memset(out, 0, sizeof(*out));

  // Whether the window is small decides where this goes, and asking is an operation at the boundary like any other. The
  // question is skipped entirely when the caller has already said it wants host memory, which is what the oracle's
  // short-circuit does and therefore what the recording contains.
  int small_window = 0;
  if (!host) {
    uint64_t bar_size = 0;
    if (mm->dev->pci->bar_info(mm->dev->pci, 1, NULL, &bar_size)) return -1;
    small_window = bar_size < mm->dev->vram_size;
  }
  if (!host && !force_devmem && (small_window ? cpu_access : (uncached && cpu_access))) host = 1;

  // Host memory the GPU reads across the bus. Everything the processor must fill and the GPU must read goes here on a
  // small bar card, because the window onto video memory is too small to be the meeting point.
  if (host) {
    size = round_up_u(size, PAGE);
    // The backend decides the real size, and it is not always the one asked for: the dext allocates in the host's page
    // size, which on Apple silicon is 16 KB, and describes it as four 4 KB pages. So the memory is taken first and the
    // address range sized from what came back - asking for 4 KB and mapping 4 KB over a 16 KB allocation fails the
    // arithmetic in map_range, which is where this was found. Every host allocation in the recorded boot is megabytes
    // and a whole number of granules, so nothing before now could have noticed.
    if (mm->dev->pci->dma_alloc(mm->dev->pci, size, &out->dma)) return -1;
    if (out->dma.npages * PAGE > size) size = out->dma.npages * PAGE;
    uint64_t va = tinynv_mm_alloc_va(mm, size, PAGE);
    if (va == TINYNV_BAD_ADDR) return -1;
    out->in_sysmem = 1;
    out->va = va;
    out->size = size;
    out->nranges = (int)out->dma.npages;
    if (!(out->ranges = calloc(out->dma.npages ? out->dma.npages : 1, sizeof(*out->ranges))))
      return tinynv_fail("out of memory for an allocation's ranges");
    // described a page at a time, which is what the oracle passes and what the page tables then have to write out
    for (size_t i = 0; i < out->dma.npages; i++)
      out->ranges[i] = (tinynv_paddr_range_t){.paddr = out->dma.pages[i], .size = PAGE};
    return tinynv_mm_map_range(mm, mm->root_page_table, va, size, out->ranges, out->nranges, TINYNV_PTE_SYSMEM | TINYNV_PTE_UNCACHED);
  }

  // a large allocation is rounded to a big-page boundary, or its unaligned tail falls back to 4 KB pages and costs a
  // page table walk on every access to it
  size = round_up_u(size, size >= (8ull << 20) ? (2ull << 20) : PAGE);

  // memory that must be both video memory and writable from here can only come from a reserved window. There are two:
  // the one the boot makes, whose size is fixed by the recording it is reproduced against, and the one taken afterwards
  // for this driver's own use. The later one is tried first so that a large allocation does not eat the boot's reserve
  // and leave a channel with nowhere to put a ring.
  if (cpu_access && force_devmem && (mm->exec_pool_size || mm->bar_pool_size)) {
    size = round_up_u(size, PAGE);
    uint64_t *next = NULL;
    if (mm->exec_pool_size && round_up_u(mm->exec_pool_next, PAGE) + size <= mm->exec_pool_base + mm->exec_pool_size)
      next = &mm->exec_pool_next;
    else if (mm->bar_pool_size && round_up_u(mm->bar_pool_next, PAGE) + size <= mm->bar_pool_base + mm->bar_pool_size)
      next = &mm->bar_pool_next;
    if (!next)
      return tinynv_fail("neither processor visible reserve has room for %#llx bytes (boot reserve %#llx, later "
                         "reserve %#llx)", (unsigned long long)size, (unsigned long long)mm->bar_pool_size,
                         (unsigned long long)mm->exec_pool_size);
    uint64_t paddr = round_up_u(*next, PAGE);
    *next = paddr + size;

    uint64_t va = tinynv_mm_alloc_va(mm, size, PAGE);
    if (va == TINYNV_BAD_ADDR) return -1;
    if (!(out->ranges = calloc(1, sizeof(*out->ranges)))) return tinynv_fail("out of memory for an allocation's ranges");
    out->va = va;
    out->size = size;
    out->nranges = 1;
    out->in_bar_pool = 1;
    out->ranges[0] = (tinynv_paddr_range_t){.paddr = paddr, .size = size};
    return tinynv_mm_map_range(mm, mm->root_page_table, va, size, out->ranges, 1, uncached ? TINYNV_PTE_UNCACHED : 0);
  }
  return tinynv_mm_valloc(mm, round_up_u(size, PAGE), PAGE, cpu_access, uncached, out);
}
