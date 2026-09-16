// The memory manager, on its own, with no GPU.
//
// Everything the driver hands a caller comes through here: a virtual address, physical memory behind it, and page table
// entries joining the two. It has only ever been exercised by the recorded boot, which allocates a fixed sequence of
// round sizes once and never frees - so the first caller that allocated awkward sizes and freed them found a bug that
// 454,812 replayed operations could not.
//
// It needs no hardware. Page tables are written through the window onto video memory, so a buffer standing in for that
// window is enough to run the whole thing: allocate, map, free, unmap, and ask whether the result is self-consistent.
#include "mmu.h"
#include "nv_regs.h"
#include "internal.h"
#include "gsp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

// A window onto "video memory" that is really a buffer. The page tables live inside the first megabytes of it, which is
// all that is ever written: an allocation past the end is invisible here exactly as it is on a small bar card.
#define WINDOW (128u << 20)
#define VRAM (32ull << 30)
#define REGS (16u << 20)   // the invalidate this path rings sits at 0xb830b0, so the window has to reach it

static tinynv_dev_t dev;
static tinynv_mm_t mm;

// The allocator asks how big the window onto video memory is, to decide whether a buffer can live there at all. That is
// the only thing it asks the backend on this path, so the backend can be this.
static int stub_bar_info(tinynv_pci_t *p, int bar, uint64_t *base, uint64_t *size) {
  if (base) *base = 0;
  if (size) *size = WINDOW;
  return 0;
}
static tinynv_pci_t stub_pci = {.name = "test", .bar_info = stub_bar_info};

// A register window that models the one register the driver reads back. Writes are kept; a read of the invalidate
// returns it with the trigger clear, which is a device whose mmu completes the invalidate immediately. Anything else
// reads back what was written.
static uint32_t stub_regs[REGS / 4];
static uint32_t stub_rd32(tinynv_mmio_t *m, uint64_t off) {
  (void)m;
  uint32_t v = stub_regs[off / 4];
  if (off == NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE) v &= ~(1u << 31);
  return v;
}
static void stub_wr32(tinynv_mmio_t *m, uint64_t off, uint32_t v) { (void)m; stub_regs[off / 4] = v; }

static int start_up(void) {
  memset(&dev, 0, sizeof(dev));
  void *window = calloc(1, WINDOW);
  if (!window) { printf("  cannot allocate a %u MB window\n", WINDOW >> 20); return -1; }
  dev.vram.ptr = window;
  dev.vram.size = WINDOW;
  // Registers, through callbacks rather than plain memory, because the driver now READS one back.
  //
  // It used to say here that nothing reads these, and that was true until the mmu invalidate learned to wait for its
  // trigger bit to clear - which is what the hardware does and what the vendor driver polls for. Backed by memory, a
  // write of the trigger stores it and a read returns it forever, so the model was of a device that never finishes
  // anything and the wait would never end. That is not a reason to weaken the wait; it is a reason for the model to
  // model the thing the driver now depends on.
  dev.mmio.ptr = NULL;
  dev.mmio.rd32 = stub_rd32;
  dev.mmio.wr32 = stub_wr32;
  dev.mmio.size = REGS;
  dev.vram_size = VRAM;
  dev.large_bar = 0;        // the card this driver runs on, so the page tables are reserved low
  dev.mmu_ver = 3;
  dev.pci = &stub_pci;
  return tinynv_mm_init(&mm, &dev);
}

// Every allocation live at once, so an overlap has something to overlap with.
#define MAX_LIVE 64
static tinynv_vmap_t live[MAX_LIVE];
static uint64_t asked[MAX_LIVE];
static int nlive;

static int still_mapped(uint64_t vaddr, uint64_t size);
// The same question asked of a named tree, for the two-tree isolation check at the end.
static int still_mapped_in(uint64_t root, uint64_t vaddr, uint64_t size);

static int take(uint64_t bytes) {
  if (nlive == MAX_LIVE) return -1;
  if (tinynv_mm_alloc_buffer(&mm, bytes, 0, 0, 0, 1, 0, &live[nlive])) {
    printf("  FAIL: allocating %#llx bytes: %s\n", (unsigned long long)bytes, tinynv_last_error());
    // what the page tables actually hold, which is the question: a live range explains itself, a ghost does not
    printf("  mapped spans in the low 2 GB of the address space:\n");
    uint64_t base = 0x1000000000ull, span_from = 0;
    for (uint64_t at = base; at <= base + (2ull << 30); at += 4096) {
      int m = at < base + (2ull << 30) ? still_mapped(at, 4096) : 0;
      if (m && !span_from) span_from = at;
      else if (!m && span_from) {
        printf("    %#14llx .. %#14llx  (%#llx)\n", (unsigned long long)span_from, (unsigned long long)at,
               (unsigned long long)(at - span_from));
        span_from = 0;
      }
    }
    printf("  live at that moment (%d):\n", nlive);
    for (int i = 0; i < nlive; i++)
      printf("    %#14llx + %#9llx (asked %#llx)\n", (unsigned long long)live[i].va,
             (unsigned long long)live[i].size, (unsigned long long)asked[i]);
    fails++;
    return -1;
  }
  asked[nlive] = bytes;
  // what was handed back must cover what was asked for, and must not overlap anything already handed out
  CHECK(live[nlive].size >= bytes, "asked for %#llx and got a %#llx byte range", (unsigned long long)bytes,
        (unsigned long long)live[nlive].size);
  for (int i = 0; i < nlive; i++) {
    uint64_t a0 = live[i].va, a1 = a0 + live[i].size, b0 = live[nlive].va, b1 = b0 + live[nlive].size;
    CHECK(b1 <= a0 || a1 <= b0, "%#llx+%#llx overlaps %#llx+%#llx", (unsigned long long)b0,
          (unsigned long long)live[nlive].size, (unsigned long long)a0, (unsigned long long)live[i].size);
  }
  return nlive++;
}

// Is anything still mapped over this range? Freeing has to take the page table entries with it: an address that comes
// back to the allocator while its mapping does not is handed out again and collides, which is how this presents - and
// worse, until it collides the GPU can still reach memory that now belongs to something else.
// The tree is a parameter now, so the two-tree check can ask the same question of each. still_mapped keeps its
// signature and asks it of the driver's own.
// A stand-in for GSP-RM answering 0x410103, including the answers we would rather it never gave. `zero_at` and
// `unaligned_at` name the call number at which it misbehaves, so a defect can be placed anywhere in a walk rather
// than only at the start - the first answer is the one an implementation is most likely to get right by accident.
struct fake { uint64_t base, contig; int zero_at, unaligned_at, calls; };
static int fake_phys(void *ctx, uint64_t offset, uint64_t *paddr, uint64_t *contig) {
  struct fake *f = (struct fake *)ctx;
  int call = f->calls++;
  *paddr = f->base + offset + (call == f->unaligned_at ? 0x800 : 0);
  *contig = (call == f->zero_at) ? 0 : f->contig;
  return 0;
}

static int still_mapped_in(uint64_t root, uint64_t vaddr, uint64_t size) {
  int found = 0;
  for (uint64_t at = vaddr; at < vaddr + size; at += 4096) {
    tinynv_pt_walk_t w;
    tinynv_pt_walk_begin(&w, &mm, root, at, 0);
    uint64_t left = 4096, off = 0;
    tinynv_pt_run_t run;
    if (tinynv_pt_walk_next(&w, &left, &off, 0, &run) > 0 && tinynv_pt_valid(&mm, &run.pt, run.idx)) found++;
  }
  return found;
}

static int still_mapped(uint64_t vaddr, uint64_t size) {
  int found = 0;
  for (uint64_t at = vaddr; at < vaddr + size; at += 4096) {
    tinynv_pt_walk_t w;
    tinynv_pt_walk_begin(&w, &mm, mm.root_page_table, at, 0);
    uint64_t left = 4096, off = 0;
    tinynv_pt_run_t run;
    if (tinynv_pt_walk_next(&w, &left, &off, 0, &run) > 0 && tinynv_pt_valid(&mm, &run.pt, run.idx))
      found++;
  }
  return found;
}

static void give_back(int i) {
  if (i < 0 || i >= nlive) return;
  uint64_t va = live[i].va, size = live[i].size;
  tinynv_vmap_free(&mm, &live[i]);
  int left = still_mapped(va, size);
  CHECK(!left, "after freeing %#llx+%#llx, %d of its %llu pages are still mapped", (unsigned long long)va,
        (unsigned long long)size, left, (unsigned long long)(size / 4096));
  live[i] = live[--nlive];
  asked[i] = asked[nlive];
}

// Replay a sequence a real caller actually issued. The harness above guesses at orders; this one does not have to.
//
// A FREE names the size of the allocation that returned the pointer, not an address, so it is matched against the most
// recent live allocation of that size - which is what a caller reusing buffers of a repeated shape does.
static int replay(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) { printf("  cannot open %s\n", path); return 1; }
  char line[512];
  long mallocs = 0, frees = 0, launches = 0, unmatched = 0;
  // the driver's local memory grows when a kernel wants more stack than any before it, and never shrinks: a few large
  // frees early in a run, each leaving a hole the size of the last one
  static const uint64_t slm[] = {162ull << 20, 200ull << 20, 244ull << 20};
  size_t grown = 0;
  tinynv_vmap_t local = {0};

  while (fgets(line, sizeof(line), f)) {
    if (line[0] == '#') continue;
    unsigned long long bytes = 0;
    if (sscanf(line, "MALLOC %llu", &bytes) == 1) {
      if (take(bytes) < 0) { fclose(f); return 1; }
      mallocs++;
    } else if (sscanf(line, "FREE %llu", &bytes) == 1) {
      int at = -1;
      for (int i = nlive - 1; i >= 0; i--) if (asked[i] == bytes) { at = i; break; }
      if (at < 0) unmatched++; else give_back(at);
      frees++;
    } else if (!strncmp(line, "LAUNCH", 6)) {
      launches++;
      // model the local-memory growth a launch can trigger: free the old range, take a bigger one
      if (grown < sizeof(slm) / sizeof(*slm) && launches % 640 == 1) {
        tinynv_vmap_t next;
        if (tinynv_mm_alloc_buffer(&mm, slm[grown], 0, 0, 0, 1, 0, &next)) {
          printf("  FAIL: growing local memory to %llu MB: %s\n", (unsigned long long)(slm[grown] >> 20),
                 tinynv_last_error());
          fails++;
          fclose(f);
          return 1;
        }
        uint64_t lva = local.va, lsz = local.size;
        tinynv_vmap_free(&mm, &local);
        if (lsz) {
          int leftover = still_mapped(lva, lsz);
          CHECK(!leftover, "after freeing local memory %#llx+%#llx, %d of its %llu pages are still mapped",
                (unsigned long long)lva, (unsigned long long)lsz, leftover, (unsigned long long)(lsz / 4096));
        }
        local = next;
        grown++;
      }
    }
    if (fails) { printf("  stopped after %ld mallocs, %ld frees, %ld launches\n", mallocs, frees, launches); break; }
  }
  fclose(f);
  tinynv_vmap_free(&mm, &local);
  printf("  replayed %ld mallocs, %ld frees, %ld launches, %zu local-memory growths; %ld frees matched nothing live\n",
         mallocs, frees, launches, grown, unmatched);
  while (nlive) give_back(0);
  return fails ? 1 : 0;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);   // so a crash does not take the progress with it
  if (start_up()) { printf("  %s\n", tinynv_last_error()); return 1; }
  if (argc > 1) {   // a recorded sequence beats a guessed one
    printf("replaying %s\n", argv[1]);
    int rc = replay(argv[1]);
    printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
    return rc;
  }
  printf("video memory %llu GB, page tables reserved at %#llx, root table at %#llx\n",
         (unsigned long long)(VRAM >> 30), (unsigned long long)mm.ptable.base, (unsigned long long)mm.root_page_table);

  // Round sizes first, which is all the recorded boot ever asked for, so a failure here would mean something much worse.
  const uint64_t round[] = {4096, 64 << 10, 1 << 20, 2 << 20, 8 << 20, 32 << 20};
  for (size_t i = 0; i < sizeof(round) / sizeof(*round); i++) take(round[i]);
  printf("  %d round allocations placed without overlap\n", nlive);
  while (nlive) give_back(0);

  // And then the sizes a real caller asks for. ggml's test cases use dimensions like k=129 and n=509, which become
  // byte counts that are not a multiple of anything convenient - and those are what found the bug this test is for.
  const uint64_t awkward[] = {129 * 4, 509 * 4, 2051 * 4, 0x402000, 0x201000, 0x1001, 2 * 1024 * 1024 + 4096,
                              4 * 1024 * 1024 + 8192, 129 * 129 * 4, 509 * 4 * 33, 1, 0x300000 + 17};
  for (size_t i = 0; i < sizeof(awkward) / sizeof(*awkward); i++) take(awkward[i]);
  printf("  %d awkward allocations placed without overlap\n", nlive);

  // Then free every other one and allocate again into the gaps, which is what a caller that reuses buffers does and
  // what the boot never did. A range whose page table entries outlive its virtual address shows up here.
  for (int i = nlive - 1; i >= 0; i -= 2) give_back(i);
  int after_free = nlive;
  for (size_t i = 0; i < sizeof(awkward) / sizeof(*awkward); i++) take(awkward[i] + 4096);
  printf("  %d freed, %d more placed into the gaps\n", after_free, nlive - after_free);

  while (nlive) give_back(0);
  CHECK(nlive == 0, "%d allocations were still live at the end", nlive);

  // One more round after everything has been returned: if a free leaves anything behind, the address space it occupied
  // comes back and the mapping under it does not, which is exactly how this presents.
  for (size_t i = 0; i < sizeof(awkward) / sizeof(*awkward); i++) take(awkward[i]);
  printf("  %d placed again after everything was returned\n", nlive);
  while (nlive) give_back(0);

  // --- what a mapping SHAPE costs in page tables -----------------------------------------------------------------
  //
  // Guarantee 10 named the wrong resource. Session C measured that the whole VM spends ONE of the dext's 128 mappings
  // - for the slab, which the guest subdivides - so a guest cannot exhaust that by asking. What it CAN exhaust is
  // page-table memory: every mapping programs entries, entries live in tables this driver allocates, and a table
  // appears at each level a range does not already have.
  //
  // Measured rather than argued, and measured BEFORE a cap is chosen, so the number in the cap comes off real shapes
  // instead of my arithmetic. The same total bytes, two shapes: one contiguous range, then the same bytes as small
  // ranges spread far enough apart that each needs its own tables.
  {
    const uint64_t SPAN = 64u << 20;          // 64 MiB of mappings either way
    const uint64_t CHUNK = 4096;
    uint64_t base = mm.va_base + (1ull << 40); // somewhere nothing else has mapped
    uint64_t before = mm.tables_made;

    tinynv_paddr_range_t one = {.paddr = 0x10000000, .size = SPAN};
    CHECK(!tinynv_mm_map_range(&mm, mm.root_page_table, base, SPAN, &one, 1, 0),
          "the contiguous mapping failed: %s", tinynv_last_error());
    uint64_t contiguous = mm.tables_made - before;
    CHECK(!tinynv_mm_unmap_range(&mm, mm.root_page_table, base, SPAN), "unmapping the contiguous range failed");

    // The same bytes, scattered: one small range per 1 GiB, so every one lands under a different table at the level
    // above it. This is the shape a hostile guest would choose and it is CHEAPER for the guest than contiguous.
    before = mm.tables_made;
    const int N = 64;
    uint64_t scattered_base = mm.va_base + (2ull << 40);
    int placed = 0;
    for (int i = 0; i < N; i++) {
      tinynv_paddr_range_t bit = {.paddr = 0x20000000 + (uint64_t)i * CHUNK, .size = CHUNK};
      if (tinynv_mm_map_range(&mm, mm.root_page_table, scattered_base + (uint64_t)i * (1ull << 30), CHUNK, &bit, 1, 0))
        break;
      placed++;
    }
    uint64_t scattered = mm.tables_made - before;
    for (int i = 0; i < placed; i++)
      tinynv_mm_unmap_range(&mm, mm.root_page_table, scattered_base + (uint64_t)i * (1ull << 30), CHUNK);

    CHECK(placed == N, "only %d of %d scattered ranges could be mapped", placed, N);
    // The finding, asserted rather than only printed: scattered costs more tables PER BYTE, and by a wide margin. If
    // this ever stops being true the premise of the budget has changed and the cap built on it is wrong.
    CHECK(scattered > contiguous,
          "scattered mapping cost %llu tables and contiguous cost %llu for %llu times fewer bytes - the shape is not "
          "the expensive thing after all, and guarantee 10's replacement needs rethinking",
          (unsigned long long)scattered, (unsigned long long)contiguous, (unsigned long long)(SPAN / (CHUNK * N)));
    // WHERE THE COST SATURATES, and therefore what the worst case actually is. The line above says scattered is
    // dearer; it does not say how dear, and a cap needs that. Same 4 KiB range repeated at five VA spacings:
    //
    //        4 KiB spacing   0.012 tables/range      (ranges share every level)
    //       64 KiB spacing   0.039
    //        2 MiB spacing   1.008                   <- one table each
    //      512 MiB spacing   2.004                   <- saturated
    //        1 GiB spacing   2.004
    //
    // TWO CONSTRAINTS BOUND A HOSTILE GUEST, and which one binds decides the answer. The slab caps the RANGE COUNT
    // at one 4 KiB range per 4 KiB of slab; the virtual address space caps how far apart they can be. Wide spacing
    // costs 2 tables each but 16 TiB only holds 32,768 ranges at 512 MiB apart - 65,536 tables, 256 MiB. Narrow
    // spacing costs ~1 table each and the VA space stops binding, so the SLAB binds: at Session C's measured
    // 2,292,144 KiB peak that is 573,036 ranges at 2 MiB apart (1,146 GiB of VA, comfortably inside 16 TiB) for
    // ~577,600 tables - 2.20 GiB of page tables against a 2.19 GiB slab.
    //
    // SO THE WORST CASE IS ROUGHLY ONE BYTE OF PAGE TABLE PER BYTE OF SLAB, and it scales with the slab rather than
    // with anything this driver chooses. C's demand measurement raising the slab ~9.3x raises this exposure by the
    // same 9.3x. That is the number a cap has to be set against, and it is why the cap must be a BUDGET rather than
    // a count: a fixed table count that suits a 256 MiB slab is wrong by an order of magnitude at 2.8 GiB.
    {
      const uint64_t spacings[] = {4096, 64u << 10, 2u << 20, 512u << 20};
      double per_range[4] = {0};
      for (size_t si = 0; si < sizeof(spacings) / sizeof(*spacings); si++) {
        uint64_t sp = spacings[si], pb = mm.tables_made;
        uint64_t pbase = mm.va_base + (4ull << 40) + (uint64_t)si * (8ull << 40);
        int np = 0;
        for (int i = 0; i < 256; i++) {
          tinynv_paddr_range_t bit = {.paddr = 0x40000000 + (uint64_t)i * CHUNK, .size = CHUNK};
          if (tinynv_mm_map_range(&mm, mm.root_page_table, pbase + (uint64_t)i * sp, CHUNK, &bit, 1, 0)) break;
          np++;
        }
        per_range[si] = np ? (double)(mm.tables_made - pb) / (double)np : 0.0;
        for (int i = 0; i < np; i++)
          tinynv_mm_unmap_range(&mm, mm.root_page_table, pbase + (uint64_t)i * sp, CHUNK);
        CHECK(np == 256, "only %d of 256 ranges placed at %llu B spacing", np, (unsigned long long)sp);
      }
      // Asserted because the whole bound rests on it: cost per range RISES with spacing and stops at about 2. If it
      // ever exceeds 2 the arithmetic above understates the worst case and the budget derived from it is too small.
      CHECK(per_range[0] < per_range[2] && per_range[2] < per_range[3],
            "tables per range did not rise with VA spacing (%.3f, %.3f, %.3f) - the guest's cheap direction is not "
            "the one the budget assumes", per_range[0], per_range[2], per_range[3]);
      CHECK(per_range[3] <= 2.5,
            "a widely spaced range now costs %.3f tables, not ~2 - the saturation the worst case is computed from "
            "has moved and the page-table budget is understated", per_range[3]);
      // Stated against C's measured slab so it moves when that does, rather than being a number frozen in a comment.
      const double SLAB_KIB = 2292144.0;
      double ranges = SLAB_KIB * 1024.0 / 4096.0;
      printf("  page-table budget: %.3f tables per 4 KiB range at 2 MiB spacing, %.3f saturated at 512 MiB; a %.2f "
             "GiB slab cut into %.0f ranges costs %.2f GiB of page tables - about one byte per byte of slab\n",
             per_range[2], per_range[3], SLAB_KIB / 1048576.0, ranges, ranges * per_range[2] * 4096.0 / 1073741824.0);
    }
    printf("  page tables per shape: %llu for %llu MiB contiguous, %llu for %llu x %llu KiB scattered "
           "(%llu times the bytes, %.0fx the tables)\n",
           (unsigned long long)contiguous, (unsigned long long)(SPAN >> 20), (unsigned long long)scattered,
           (unsigned long long)N, (unsigned long long)(CHUNK >> 10),
           (unsigned long long)(SPAN / (CHUNK * N)), contiguous ? (double)scattered / (double)contiguous : 0.0);
  }

  // --- two trees, which is what makes guest isolation a fact rather than a check --------------------------------
  //
  // gsp.c hands the externally-owned user VASPACE `physAddress = g->mm.root_page_table` - OUR root - so the address
  // space a guest would use and the driver's own regions have always been ONE TREE. The only thing between a guest
  // and the command ring would have been a range check in software, and §4g says why that is not enough: a window
  // wide enough for libcuda's scattered addresses is a list of exclusions in disguise.
  //
  // With the root threaded (359ebad) a second tree is a page of memory and a parameter. This asserts the property the
  // whole design rests on: the SAME address mapped in one tree is absent from the other, in both directions. That is
  // guarantee 2 as a fact about translation rather than a comparison somebody has to remember to write.
  {
    uint64_t guest_root = tinynv_mm_palloc(&mm, 4096, 4096, 1, TINYNV_REGION_PTABLE);
    CHECK(guest_root != TINYNV_BAD_ADDR, "a second page-table root could not be allocated");
    CHECK(guest_root != mm.root_page_table, "the second root is the first one");

    if (guest_root != TINYNV_BAD_ADDR) {
      const uint64_t VA = mm.va_base + (3ull << 40), LEN = 0x1000;
      tinynv_paddr_range_t ours = {.paddr = 0x30000000, .size = LEN};
      tinynv_paddr_range_t theirs = {.paddr = 0x40000000, .size = LEN};

      CHECK(!tinynv_mm_map_range(&mm, mm.root_page_table, VA, LEN, &ours, 1, 0),
            "mapping into the driver's tree failed: %s", tinynv_last_error());
      // The same address in the guest's tree is FREE - that is the isolation, stated as a measurement.
      uint64_t clash = 0;
      CHECK(!still_mapped_in(guest_root, VA, LEN),
            "an address mapped in the driver's tree is also mapped in the guest's: the trees are not separate and a "
            "guest VA can name our memory");
      (void)clash;

      // And the guest can map the same address to its OWN memory without disturbing ours.
      CHECK(!tinynv_mm_map_range(&mm, guest_root, VA, LEN, &theirs, 1, 0),
            "the guest's tree refused an address the driver's tree already uses - the trees are sharing state");
      CHECK(still_mapped_in(mm.root_page_table, VA, LEN), "mapping in the guest's tree unmapped ours");
      CHECK(still_mapped_in(guest_root, VA, LEN), "the guest's mapping did not take");

      // GUARANTEE 7'S GRANT HALF, read back from the entry rather than from what was passed in. Until 2026-09-15
      // this could not even be expressed: pt.c wrote PCF = uncached ? 1 : 0, which is REGULAR_RW_ATOMIC either way,
      // so a guest allowed a READ-ONLY mapping - which tinynv_rm_map_check permits, including against read-only
      // memory - was handed a WRITABLE one. The decision was right and the thing acting on it gave away more than
      // was asked for, which is the same shape as the alignment that was cited to a function that did not exist.
      //
      // Asserted on the BITS, because "the flag was threaded correctly" and "the entry says read-only" are two
      // different claims and only the second is what the MMU acts on.
      {
        const uint64_t RO_VA = mm.va_base + (3ull << 40) + (1ull << 20);
        tinynv_paddr_range_t ro = {.paddr = 0x50000000, .size = LEN};
        CHECK(!tinynv_mm_map_range(&mm, guest_root, RO_VA, LEN, &ro, 1, TINYNV_PTE_READONLY),
              "a read-only mapping was refused outright: %s", tinynv_last_error());

        tinynv_pt_walk_t w;
        tinynv_pt_walk_begin(&w, &mm, guest_root, RO_VA, 0);
        uint64_t left = LEN, off = 0;
        tinynv_pt_run_t run;
        CHECK(tinynv_pt_walk_next(&w, &left, &off, 0, &run) > 0, "the read-only mapping is not there to inspect");
        CHECK(tinynv_pt_writable(&mm, &run.pt, run.idx) == 0,
              "a mapping programmed READ-ONLY came back writable - the guest asked for read access to memory it may "
              "only read, and this driver gave it write access to the same bytes");

        // The control: the same call without the flag must still be writable, or the assertion above passes for the
        // trivial reason that nothing is ever writable.
        tinynv_pt_walk_t w2;
        tinynv_pt_walk_begin(&w2, &mm, guest_root, VA, 0);
        left = LEN; off = 0;
        tinynv_pt_run_t run2;
        CHECK(tinynv_pt_walk_next(&w2, &left, &off, 0, &run2) > 0, "the read-write mapping is not there to inspect");
        CHECK(tinynv_pt_writable(&mm, &run2.pt, run2.idx) == 1,
              "a mapping programmed WITHOUT the read-only flag came back read-only, so the row above proves nothing "
              "and every ordinary mapping this driver makes is now unwritable");

        CHECK(!tinynv_mm_unmap_range(&mm, guest_root, RO_VA, LEN), "unmapping the read-only range failed");
      }

      // --- tinynv_c4b_map: decide, program, record, and the refusals leaving NOTHING behind ------------------
      //
      // The function gsp.c used to cite and did not have. What is worth testing here is not that a good request
      // works - it is that every refusal leaves the tree exactly as it found it, because a refusal that half-programs
      // is worse than a grant: nothing records it, so nothing will ever tear it down.
      {
        const uint64_t CVA = mm.va_base + (3ull << 40) + (2ull << 20);
        const uint64_t CLEN = 0x2000;
        tinynv_rm_obj_t objs[] = {
            {.client = 0xc1d0u, .handle = 0x5c01u, .cls = 0x0040u, .alloc_flags = 0x1c101u, .size = 0x200000u},
            {.client = 0xc1d0u, .handle = 0x5c02u, .cls = 0x0040u, .alloc_flags = 0x1c101u | 0x08000000u,
             .size = 0x200000u},
        };
        tinynv_rm_map_t recs[2];
        int nrecs = 0;
        tinynv_paddr_range_t ok2[2] = {{.paddr = 0x60000000, .size = 0x1000}, {.paddr = 0x61000000, .size = 0x1000}};
        tinynv_rm_map_req_t q = {.client = 0xc1d0u, .handle = 0x5c01u, .mapping_type = 1, .root = guest_root,
                                 .va = CVA, .length = CLEN, .offset = 0};

        // Refusals first, so that every later assertion about the tree starts from a tree nothing has touched.
        tinynv_rm_map_req_t ours = q; ours.root = mm.root_page_table;
        CHECK(tinynv_c4b_map(&mm, objs, 2, recs, &nrecs, 2, mm.root_page_table, &ours, ok2, 2) ==
                  TINYNV_RM_MAP_NOT_GUEST_TREE,
              "a mapping into the DRIVER's tree was programmed - a guest VA in our address space, beside the ring");
        CHECK(nrecs == 0, "a refused mapping was recorded");
        CHECK(!still_mapped_in(mm.root_page_table, CVA, CLEN), "a refusal left entries in the driver's tree");

        // This one is refused by tinynv_mm_map_range's own total check as well as by c4b's, so it does NOT
        // discriminate between them - a negative control removing c4b's check leaves it passing. Kept because the
        // OUTCOME is worth pinning, and labelled so nobody reads it as evidence for the check below it.
        tinynv_paddr_range_t shortfall[1] = {{.paddr = 0x60000000, .size = 0x1000}};
        CHECK(tinynv_c4b_map(&mm, objs, 2, recs, &nrecs, 2, mm.root_page_table, &q, shortfall, 1) ==
                  TINYNV_RM_MAP_BAD_RANGES,
              "ranges totalling less than the request were mapped, so the tail of the range is whatever was there");

        // AND THE CASE WHERE c4b's CHECK IS THE ONLY ONE THAT WORKS. map_range tests `total != size` on a total it
        // accumulates without watching for overflow. Two ranges whose sizes WRAP to exactly the requested length
        // pass that test, and it then walks a range of 0xfffffffffffff000 bytes. c4b refuses on the wrap itself,
        // before the addition can produce a number that looks right.
        //
        // No negative control on this one, stated rather than skipped quietly: removing c4b's check does not make
        // this FAIL, it makes the suite walk sixteen exabytes. That the control cannot be run is the finding - it is
        // what "map_range will catch it anyway" costs.
        tinynv_paddr_range_t wrap2[2] = {{.paddr = 0x60000000, .size = 0xfffffffffffff000ull},
                                         {.paddr = 0x61000000, .size = 0x3000}};
        CHECK(tinynv_c4b_map(&mm, objs, 2, recs, &nrecs, 2, mm.root_page_table, &q, wrap2, 2) ==
                  TINYNV_RM_MAP_BAD_RANGES,
              "two range sizes that WRAP to the requested length were accepted - the total matches, every entry "
              "after the first is nonsense, and map_range's own check cannot see it because it does the same sum");
        tinynv_paddr_range_t unaligned[2] = {{.paddr = 0x60000800, .size = 0x1000}, {.paddr = 0x61000000, .size = 0x1000}};
        CHECK(tinynv_c4b_map(&mm, objs, 2, recs, &nrecs, 2, mm.root_page_table, &q, unaligned, 2) ==
                  TINYNV_RM_MAP_BAD_RANGES,
              "an unaligned PHYSICAL base was accepted - set_entry writes paddr >> 12, so the low bits vanish and "
              "every entry points somewhere the guest was never granted");
        CHECK(nrecs == 0 && !still_mapped_in(guest_root, CVA, CLEN),
              "a BAD_RANGES refusal left the guest's tree changed");

        // Guarantee 7, both directions, through the real entry point rather than the decision alone.
        tinynv_rm_map_req_t rw_on_ro = q; rw_on_ro.handle = 0x5c02u;
        CHECK(tinynv_c4b_map(&mm, objs, 2, recs, &nrecs, 2, mm.root_page_table, &rw_on_ro, ok2, 2) ==
                  TINYNV_RM_MAP_WOULD_WRITE,
              "a write mapping was programmed over read-only memory");
        CHECK(nrecs == 0 && !still_mapped_in(guest_root, CVA, CLEN), "a WOULD_WRITE refusal programmed entries");

        // The grant, and the entry actually saying read-only - the half that was unimplementable until today.
        tinynv_rm_map_req_t ro_on_ro = rw_on_ro; ro_on_ro.mapping_type = TINYNV_RM_MAP_TYPE_READ_ONLY;
        CHECK(tinynv_c4b_map(&mm, objs, 2, recs, &nrecs, 2, mm.root_page_table, &ro_on_ro, ok2, 2) ==
                  TINYNV_RM_MAP_OK,
              "a read-only mapping of read-only memory was refused: %s", tinynv_last_error());
        CHECK(nrecs == 1, "a granted mapping was not recorded, so nothing can ever tear it down");
        CHECK(recs[0].root == guest_root && recs[0].va == CVA && recs[0].length == CLEN,
              "the record does not describe what was programmed");
        {
          tinynv_pt_walk_t cw;
          tinynv_pt_walk_begin(&cw, &mm, guest_root, CVA, 0);
          uint64_t cl = CLEN, co = 0;
          tinynv_pt_run_t cr;
          CHECK(tinynv_pt_walk_next(&cw, &cl, &co, 0, &cr) > 0, "the granted mapping is not in the guest's tree");
          CHECK(tinynv_pt_writable(&mm, &cr.pt, cr.idx) == 0,
                "a read-only mapping granted through tinynv_c4b_map was programmed WRITABLE - the decision and the "
                "entry disagree, which is the whole defect guarantee 7 had in both directions today");
        }

        // Overlap is refused once something is there, and the record is what notices.
        tinynv_rm_map_req_t again = ro_on_ro;
        CHECK(tinynv_c4b_map(&mm, objs, 2, recs, &nrecs, 2, mm.root_page_table, &again, ok2, 2) ==
                  TINYNV_RM_MAP_ALREADY_MAPPED,
              "the same range was mapped twice in the same tree");
        CHECK(nrecs == 1, "a refused second mapping was still recorded");

        // And a full record refuses rather than programming something nothing will remember.
        tinynv_rm_map_req_t elsewhere = q;
        elsewhere.va = CVA + (1ull << 30);
        recs[1] = (tinynv_rm_map_t){.client = 0xdead, .handle = 1, .root = 0, .va = 0, .length = 0};
        nrecs = 2;
        CHECK(tinynv_c4b_map(&mm, objs, 2, recs, &nrecs, 2, mm.root_page_table, &elsewhere, ok2, 2) ==
                  TINYNV_RM_MAP_NO_ROOM,
              "a mapping was programmed with no room to record it - guarantee 5 cannot tear down what it cannot find");
        CHECK(!still_mapped_in(guest_root, elsewhere.va, CLEN), "a NO_ROOM refusal programmed entries anyway");

        nrecs = 1;
        CHECK(!tinynv_mm_unmap_range(&mm, guest_root, CVA, CLEN), "unmapping the c4b range failed");
      }
      printf("  the c4b map: decide, program, record - and every refusal leaves the tree untouched\n");

      // --- the physical-range walk, with no card ---------------------------------------------------------------
      //
      // Only the TRANSPORT needs a card; the walk is arithmetic and refusals, and that is all here. The answers a
      // real GSP-RM would give are simulated by the query below, including the ones we would rather it never gave.
      {
        struct fake f;
        // Every offset lives at base+offset, and `contig` bytes are contiguous from wherever we are asked.
        tinynv_paddr_range_t got[8];
        int n = 0;

        f = (struct fake){.base = 0x70000000, .contig = 0x100000, .zero_at = -1, .unaligned_at = -1};
        CHECK(!tinynv_c4b_ranges(fake_phys, &f, 0, 0x200000, got, 8, &n),
              "a straightforward walk failed: %s", tinynv_last_error());
        CHECK(n == 2, "a 2 MiB allocation in 1 MiB contiguous pieces came back as %d ranges, not 2", n);
        CHECK(got[0].paddr == 0x70000000 && got[0].size == 0x100000, "the first range is wrong");
        CHECK(got[1].paddr == 0x70100000 && got[1].size == 0x100000, "the second range is wrong");

        // Contiguous to the end: one range, and the walk must CLIP to what was asked rather than take all of it.
        f = (struct fake){.base = 0x70000000, .contig = 0x800000, .zero_at = -1, .unaligned_at = -1};
        CHECK(!tinynv_c4b_ranges(fake_phys, &f, 0, 0x200000, got, 8, &n), "the contiguous walk failed");
        CHECK(n == 1 && got[0].size == 0x200000,
              "a segment longer than the request was taken whole - that maps %llu bytes the guest did not ask for",
              (unsigned long long)(n == 1 ? got[0].size : 0));

        // ZERO contiguous bytes. The one answer that must be refused rather than retried: it consumes nothing, so a
        // loop that asks again at the same offset never ends. This test exists because the obvious implementation
        // hangs here rather than failing, and a hang in a driver reads as hardware.
        f = (struct fake){.base = 0x70000000, .contig = 0x100000, .zero_at = 1, .unaligned_at = -1};
        CHECK(tinynv_c4b_ranges(fake_phys, &f, 0, 0x200000, got, 8, &n),
              "a contiguous length of ZERO was walked past - if this ever passes, check whether it returned or hung");

        // An unaligned physical address, refused where it can still be named rather than at the map.
        f = (struct fake){.base = 0x70000000, .contig = 0x100000, .zero_at = -1, .unaligned_at = 1};
        CHECK(tinynv_c4b_ranges(fake_phys, &f, 0, 0x200000, got, 8, &n),
              "an unaligned physical address was accepted into a range list");

        // Too many pieces: refused, NOT truncated. A short list reaches tinynv_c4b_map as a total mismatch and gets
        // reported as a firmware disagreement, which sends the next reader somewhere the problem is not.
        //
        // HOW THIS ONE AND THE UNALIGNED ONE ABOVE FAIL WHEN THE CHECK IS REMOVED: exit 139, not a FAIL line. Both
        // guard against writing past `got`, so deleting either corrupts the stack and the binary dies before it can
        // report anything. They DO discriminate - a segfault is not a pass, and make test stops on the non-zero exit
        // - but only if the control is measured by EXIT STATUS. Measured by counting FAIL lines, as these controls
        // first were, a crash and a clean run are the same number: zero. That is how the instrument validating every
        // other check in this file was itself wrong, which is worth more than the two checks it was validating.
        f = (struct fake){.base = 0x70000000, .contig = 0x1000, .zero_at = -1, .unaligned_at = -1};
        CHECK(tinynv_c4b_ranges(fake_phys, &f, 0, 0x200000, got, 8, &n),
              "an allocation needing more than 8 ranges was truncated to 8 instead of refused");

        // And the walk feeds the map: real ranges, real decision, programmed and recorded.
        f = (struct fake){.base = 0x70000000, .contig = 0x1000, .zero_at = -1, .unaligned_at = -1};
        CHECK(!tinynv_c4b_ranges(fake_phys, &f, 0, 0x2000, got, 8, &n), "the two-page walk failed");
        CHECK(n == 2, "two pages came back as %d ranges", n);
      }
      printf("  the physical-range walk: refuses a zero-length segment rather than looping on it\n");

      // Tearing one down leaves the other alone, which is the half a shared tree would break silently.
      CHECK(!tinynv_mm_unmap_range(&mm, guest_root, VA, LEN), "unmapping the guest's range failed");
      CHECK(!still_mapped_in(guest_root, VA, LEN), "the guest's mapping survived its own unmap");
      CHECK(still_mapped_in(mm.root_page_table, VA, LEN), "unmapping the guest's range took ours with it");
      CHECK(!tinynv_mm_unmap_range(&mm, mm.root_page_table, VA, LEN), "unmapping the driver's range failed");
    }
  }
  printf("  two trees: the same address is two different mappings, and neither unmap reaches the other\n");

  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
