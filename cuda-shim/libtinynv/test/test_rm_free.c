// Whether the record can refuse a free that would corrupt something.
//
// The driver had no free path at all until now, and the reason it was not a one-line addition is that handles came from
// a counter and nothing was written down: the missing piece was the record, not the message. Once the thing asking for
// a free may be a guest, the record stops being bookkeeping and becomes a validator, because a guest's free order is
// not a safe order. It can free a parent before its children, free the same object twice, free what it never
// allocated, or free an object belonging to another client. Those are all just handles arriving at an escape, and this
// is the only thing that knows enough to tell them apart.
//
// Driven here with no card, because every one of these refusals is about the shape of the record rather than about the
// hardware, and because the failure they prevent - GSP-RM tearing down an object this driver does not own, or children
// left pointing at a freed parent - is not something a card reports as an error.
#include "gsp.h"
#include <stdio.h>
#include <string.h>

static int fails, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

// A session shaped like the one bring-up actually builds: a privileged client with a device, subdevice, address space
// and a channel under it, and a second, user client that chose its own handles - which is what a guest does.
#define PRIV 0xcf000000u
#define USER 0xc1000000u
static tinynv_rm_obj_t objs[] = {
  {PRIV, 0,          PRIV,       0x0000, 0, 0},   // the root names itself
  {PRIV, PRIV,       0xcf000001, 0x0080, 0, 0},   // device
  {PRIV, 0xcf000001, 0xcf000002, 0x2080, 0, 0},   // subdevice
  {PRIV, 0xcf000001, 0xcf000003, 0x90f1, 0, 0},   // address space
  {PRIV, 0xcf000001, 0xcf000004, 0xc96f, 0, 0},   // channel
  {PRIV, 0xcf000004, 0xcf000005, 0xcec0, 0, 0},   // compute object, under the channel
  {PRIV, 0xcf000004, 0xcf000006, 0xcab5, 0, 0},   // copy object, under the channel
  {USER, 0,          USER,       0x0000, 0, 0},   // a second client, handles of its own choosing
  {USER, USER,       0xcf000001, 0x0080, 0, 0},   // DELIBERATE: the same handle the privileged device has
};
#define N ((int)(sizeof objs / sizeof objs[0]))

int main(void) {
  int at = -1;

  // A leaf goes.
  CHECK(tinynv_rm_free_check(objs, N, PRIV, 0xcf000005, &at) == TINYNV_RM_FREE_OK, "a leaf object cannot be freed");
  CHECK(at == 5, "the leaf was found at %d, expected 5", at);

  // Its parent does not, while it is there.
  CHECK(tinynv_rm_free_check(objs, N, PRIV, 0xcf000004, NULL) == TINYNV_RM_FREE_HAS_CHILDREN,
        "a channel with live objects under it was allowed to be freed");
  CHECK(tinynv_rm_free_check(objs, N, PRIV, 0xcf000001, NULL) == TINYNV_RM_FREE_HAS_CHILDREN,
        "a device with a subdevice under it was allowed to be freed");
  CHECK(tinynv_rm_free_check(objs, N, PRIV, PRIV, NULL) == TINYNV_RM_FREE_HAS_CHILDREN,
        "a root with a device under it was allowed to be freed");

  // Things that were never allocated, and things belonging to someone else.
  CHECK(tinynv_rm_free_check(objs, N, PRIV, 0xdeadbeef, NULL) == TINYNV_RM_FREE_UNKNOWN,
        "a handle that was never allocated was accepted");
  CHECK(tinynv_rm_free_check(objs, N, 0x12345678, 0xcf000005, NULL) == TINYNV_RM_FREE_UNKNOWN,
        "a free from a client that allocated nothing was accepted");

  // The one that a handle table alone would get wrong. Both clients hold 0xcf000001. The privileged one has a
  // subdevice under it and must be refused; the user one is a leaf and must be allowed. Scoping the parent search to
  // the client is the difference, and without it the user's device inherits the privileged device's children.
  CHECK(tinynv_rm_free_check(objs, N, USER, 0xcf000001, &at) == TINYNV_RM_FREE_OK,
        "a leaf was refused because another client happens to hold the same handle");
  CHECK(at == 8, "the user device was found at %d, expected 8", at);
  CHECK(tinynv_rm_free_check(objs, N, PRIV, 0xcf000001, NULL) == TINYNV_RM_FREE_HAS_CHILDREN,
        "the privileged device with the same handle was allowed through");

  // And the property teardown rests on: walking the record backwards frees children before parents, for free, because
  // a child cannot be created before its parent. Simulated by removing from the end and asking each time.
  int n = N, freed = 0;
  while (n) {
    if (tinynv_rm_free_check(objs, n, objs[n - 1].client, objs[n - 1].handle, NULL) != TINYNV_RM_FREE_OK) {
      printf("  FAIL: tearing down backwards hit an object with live children at index %d\n", n - 1);
      fails++;
      break;
    }
    n--; freed++;
  }
  checks++;
  CHECK(freed == N, "backwards teardown freed %d of %d objects", freed, N);

  // An empty record refuses everything rather than accepting anything.
  CHECK(tinynv_rm_free_check(objs, 0, PRIV, PRIV, NULL) == TINYNV_RM_FREE_UNKNOWN, "an empty record accepted a free");

  // Descendancy, which the cascade rests on. RM frees a resource's whole subtree - children and dependants, children
  // first - for ANY resource and not only a client, which is read out of the vendored resource server rather than
  // assumed. So a free has to find everything below the target, and the tree is deeper than two: a compute object sits
  // under a channel under a device under a client.
  CHECK(tinynv_rm_is_descendant(objs, N, PRIV, 0xcf000005, 0xcf000004) == 1, "compute is not below its channel");
  CHECK(tinynv_rm_is_descendant(objs, N, PRIV, 0xcf000005, 0xcf000001) == 1, "compute is not below the device two up");
  CHECK(tinynv_rm_is_descendant(objs, N, PRIV, 0xcf000005, PRIV) == 1, "compute is not below the client three up");
  CHECK(tinynv_rm_is_descendant(objs, N, PRIV, 0xcf000001, 0xcf000005) == 0, "the device was called a child of a leaf");
  CHECK(tinynv_rm_is_descendant(objs, N, PRIV, 0xcf000005, 0xcf000005) == 0, "an object was called its own descendant");
  CHECK(tinynv_rm_is_descendant(objs, N, PRIV, 0xcf000005, 0xcf000003) == 0,
        "compute was called a descendant of the address space, which is its uncle and not its ancestor");
  CHECK(tinynv_rm_is_descendant(objs, N, PRIV, PRIV, PRIV) == 0, "the client was called its own descendant");
  // And the collision case again, one level up: the user client holds a device with the SAME handle as the privileged
  // one. Walking a parent chain without scoping to the client would climb out of one tree and into the other.
  CHECK(tinynv_rm_is_descendant(objs, N, USER, 0xcf000001, USER) == 1, "the user device is not below the user client");
  CHECK(tinynv_rm_is_descendant(objs, N, USER, 0xcf000001, PRIV) == 0,
        "the user's device was called a descendant of the privileged client it shares a handle with");
  CHECK(tinynv_rm_is_descendant(objs, N, PRIV, 0xcf000001, USER) == 0,
        "the privileged device was called a descendant of the user client");

  // The eviction sweep's bounds. A guest that dies frees nothing, so the only way to clean up is to free every client
  // in the range guests come from - and this driver's own privileged client sits four bytes above the top of that
  // range. An inclusive upper bound there does not return an error, it tears down the client that owns the channels
  // this process is submitting through, which would present as the card dying rather than as a bad argument.
  CHECK(tinynv_rm_range_safe(0xc1d00000u, 0xc1e00000u) == 1,
        "the guest client range was refused, and it is the one the forwarder actually uses");
  CHECK(tinynv_rm_range_safe(0xc1d00000u, 0xc1e00005u) == 0,
        "a range reaching one past the privileged client was allowed; that sweep kills the running session");
  CHECK(tinynv_rm_range_safe(0xc1e00004u, 0xc1e00005u) == 0, "the privileged client alone was allowed");
  CHECK(tinynv_rm_range_safe(0xc1000000u, 0xc1000001u) == 0, "the driver's own user client was allowed");
  CHECK(tinynv_rm_range_safe(0x00000000u, 0xffffffffu) == 0, "sweeping everything was allowed");
  CHECK(tinynv_rm_range_safe(0xc1d00000u, 0xc1d00000u) == 1, "an empty range was refused");

  // The pass/fail decision is at the END of main, not here. It used to be here, when here WAS the end of main, and
  // then the file grew: the mapping record, writability, and the whole C4b decision were appended below a guard that
  // had already returned. Every check after this line accumulated into a variable nothing read again, and the file
  // exited 0 with FAIL printed on stdout - which is how the read-only rows added on 2026-09-15 were reported as
  // "suite green" while proving nothing. Keep running so every section reports; the exit status is the last thing.
  printf("  rm free: %d checks - a leaf goes, a parent takes its subtree with it as RM does, an unknown or "
         "other-client handle is refused, and the same handle under two clients is told apart\n", checks);
  printf("  backwards through the record is children before parents, for all %d objects\n", N);
  printf("  a free finds everything below its target, three levels up, and does not climb between two "
         "clients that share a handle\n");
  printf("  the eviction sweep takes the guest range and refuses the off-by-one that would take this "
         "driver's own client with it\n");

  // --- the mapping record, which guarantee 5 rests on ------------------------------------------------------------
  //
  // Same shape as the free record above and for the same reason: the refusals ARE the feature, and they can be driven
  // with no card. A free cascades over children and dependants, so an allocation can go away underneath a live
  // mapping and leave page tables pointing at memory the allocator can hand to somebody else.
  {
    const uint64_t R1 = 0x4400000, R2 = 0x8800000;   // two page-table roots: the driver's and a guest's
    static const tinynv_rm_map_t maps[] = {
      {.client = 0xc1d00000, .handle = 0x5c000001, .root = R1, .va = 0x200000000ull, .length = 0x200000},
      {.client = 0xc1d00000, .handle = 0x5c000002, .root = R1, .va = 0x300000000ull, .length = 0x1000},
      {.client = 0xc1d00001, .handle = 0x5c000003, .root = R2, .va = 0x200000000ull, .length = 0x200000},
    };
    const int nm = (int)(sizeof(maps) / sizeof(*maps));
    int at = -1;

    CHECK(tinynv_rm_unmap_check(maps, nm, 0xc1d00000, R1, 0x200000000ull, 0x200000, &at) == TINYNV_RM_UNMAP_OK,
          "an exact unmap of a live mapping was refused");
    CHECK(at == 0, "the exact unmap found index %d rather than 0", at);

    // Guarantee 12. The walk may program a 2 MiB page where 4 KiB was asked for, so a mapping can be ONE entry, and
    // unmapping part of it would clear all of it while the record still believed the rest was live.
    CHECK(tinynv_rm_unmap_check(maps, nm, 0xc1d00000, R1, 0x200000000ull, 0x1000, NULL) == TINYNV_RM_UNMAP_PARTIAL,
          "unmapping 4 KiB of a 2 MiB mapping was allowed - that clears the whole entry and leaves the record wrong");
    CHECK(tinynv_rm_unmap_check(maps, nm, 0xc1d00000, R1, 0x200001000ull, 0x1000, NULL) == TINYNV_RM_UNMAP_PARTIAL,
          "a base INSIDE a live mapping came back unknown rather than partial - those are different mistakes");

    // The same address in a different tree is a different mapping. This is what the separate root buys: after the
    // second tree exists, a guest's VA cannot name the driver's memory because the translation does not exist there.
    CHECK(tinynv_rm_unmap_check(maps, nm, 0xc1d00000, R2, 0x200000000ull, 0x200000, NULL) == TINYNV_RM_UNMAP_UNKNOWN,
          "an unmap crossed page-table roots: the same address in two trees is two mappings");
    // And the same address in the same tree under a different client.
    CHECK(tinynv_rm_unmap_check(maps, nm, 0xc1d00002, R1, 0x200000000ull, 0x200000, NULL) == TINYNV_RM_UNMAP_UNKNOWN,
          "an unmap crossed clients");
    CHECK(tinynv_rm_unmap_check(maps, nm, 0xc1d00000, R1, 0x900000000ull, 0x1000, NULL) == TINYNV_RM_UNMAP_UNKNOWN,
          "an unmap of a range nobody mapped was accepted");

    // Guarantee 5's question, asked by a free before it frees.
    CHECK(tinynv_rm_object_is_mapped(maps, nm, 0xc1d00000, 0x5c000001),
          "a live mapping of an object was not found, so a free would proceed and leave stale translations");
    CHECK(!tinynv_rm_object_is_mapped(maps, nm, 0xc1d00000, 0x5c000003),
          "an object mapped by ANOTHER client counted as this client's, which would refuse a legitimate free");
    CHECK(!tinynv_rm_object_is_mapped(maps, nm, 0xc1d00000, 0x5c00ffff), "an unmapped object was reported as mapped");
    CHECK(!tinynv_rm_object_is_mapped(maps, 0, 0xc1d00000, 0x5c000001), "an empty record found a mapping in it");
  }
  // Guarantee 7's input, now that there is one. The hostile case is the whole point and no recording will ever
  // contain it: torch sets neither read-only bit in any of the twelve recorded allocations, so every measurement we
  // have says "writable" - and a guest that sets DEVICE_READ_ONLY and then asks for a read-write mapping is exactly
  // what 7 exists for.
  {
    tinynv_rm_obj_t rw = {.client = USER, .handle = 0x5c000001, .cls = 0x0040, .size = 0x200000, .alloc_flags = 0x1c101};
    tinynv_rm_obj_t ro_dev = rw, ro_user = rw, no_size = rw;
    ro_dev.alloc_flags |= 0x08000000u;
    ro_user.alloc_flags |= 0x04000000u;
    no_size.size = 0;

    CHECK(tinynv_rm_obj_writable(&rw), "the flags torch actually sets (0x1c101) came back read-only");
    CHECK(!tinynv_rm_obj_writable(&ro_dev), "DEVICE_READ_ONLY was ignored, so a guest could map its own read-only "
                                            "allocation writable");
    CHECK(!tinynv_rm_obj_writable(&ro_user), "USER_READ_ONLY was ignored");
    CHECK(!tinynv_rm_obj_writable(&no_size), "an allocation whose params never reached us was reported writable - an "
                                             "unknown permission must refuse like an unknown bound, not assume");
    CHECK(!tinynv_rm_obj_writable(NULL), "a missing object was reported writable");

    // The aliasing, which is a policy choice and not an accident. 0x04000000 is USER_READ_ONLY and also SPARSE; on
    // class 0x0040 it cannot be sparse (RM refuses virtual-only flags on physical allocations - see nv_structs.h),
    // but on any other class it can, and this driver still refuses. That is the strict reading of an ambiguous bit,
    // deliberately: the alternative is letting a guest reach write access by picking a class where the bit is
    // deniable. If someone later "fixes" the puzzling refusal of a sparse allocation by making non-0x0040 classes
    // permissive, this is the row that should stop them - the refusal is correct, only its stated reason is
    // approximate, and the fix is the sentence in tinynv_rm_map_why rather than the outcome.
    tinynv_rm_obj_t sparse_virt = ro_user;
    sparse_virt.cls = 0x0070;  // NV01_MEMORY_VIRTUAL, where 0x04000000 really can mean SPARSE
    CHECK(!tinynv_rm_obj_writable(&sparse_virt),
          "bit 0x04000000 on a class where it may mean SPARSE rather than USER_READ_ONLY was treated as writable - an "
          "ambiguous permission bit must take the STRICT reading, or a guest picks the class that makes it deniable");

    // Both bits at once, which is the shape a guest would send if it were probing for a flags word we mishandle.
    tinynv_rm_obj_t both = rw;
    both.alloc_flags |= 0x04000000u | 0x08000000u;
    CHECK(!tinynv_rm_obj_writable(&both), "an allocation carrying both read-only bits was reported writable");

    // And the bits NEXT to them, to show the test is reading these two and not a neighbourhood. 0x02000000 is
    // KERNEL_MAPPING_MAP/MAXIMIZE_ADDRESS_SPACE and 0x10000000 is SKIP_RESOURCE_ALLOC; neither says anything about
    // permission, and an over-broad mask here would refuse legitimate allocations with a sentence about read-only.
    tinynv_rm_obj_t neighbours = rw;
    neighbours.alloc_flags |= 0x02000000u | 0x10000000u;
    CHECK(tinynv_rm_obj_writable(&neighbours),
          "a flag bit adjacent to the read-only pair was read as read-only, so the mask is wider than the two bits "
          "nv_structs.h documents and legitimate allocations will be refused with a misleading reason");
  }
  printf("  an allocation's writability, including the read-only case no recording contains and the aliased bits\n");

  // --- the C4b mapping decision, against the request torch actually stops on --------------------------------------
  //
  // Session C re-recorded it, so this is the real shape rather than an invented one: class 0x0040, 2 MiB, flags
  // 0x1c101 with neither read-only bit, and the request
  //
  //   UVM_MAP_EXTERNAL_ALLOCATION base 0x200000000 length 2097152 offset 0 memory 0x5c00000c attrs 1 type 1
  //
  // Every count in the offline replay matches the card run up to it, so the decision this function makes IS the
  // decision the card path will make.
  {
    const uint64_t DRIVER_ROOT = 0x4400000, GUEST_ROOT = 0x8800000;
    tinynv_rm_obj_t mem[] = {
      {USER, 0x5c000002, 0x5c00000c, 0x0040, 0x1c101, 0x200000},   // what C recorded
      {USER, 0x5c000002, 0x5c00000d, 0x0040, 0x1c101 | 0x08000000u, 0x200000},  // the same, marked read-only
      {USER, 0x5c000002, 0x5c00000e, 0x0040, 0, 0},                // params never reached us
    };
    const int nmem = (int)(sizeof(mem) / sizeof(*mem));
    tinynv_rm_map_t none[1];
    tinynv_rm_map_req_t r = {.client = USER, .handle = 0x5c00000c, .mapping_type = 1, .root = GUEST_ROOT,
                             .va = 0x200000000ull, .length = 0x200000, .offset = 0};

    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &r) == TINYNV_RM_MAP_OK,
          "the request torch stops on was refused: %s", tinynv_rm_map_why(tinynv_rm_map_check(mem, nmem, none, 0,
                                                                                             DRIVER_ROOT, &r)));

    // 2, and it is checked first because its failure is the silent catastrophic one: a guest mapping in the driver's
    // tree WORKS and shares an address space with the command ring.
    tinynv_rm_map_req_t ours = r; ours.root = DRIVER_ROOT;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &ours) == TINYNV_RM_MAP_NOT_GUEST_TREE,
          "a guest mapping into the DRIVER'S tree was allowed");
    tinynv_rm_map_req_t noroot = r; noroot.root = 0;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &noroot) == TINYNV_RM_MAP_NOT_GUEST_TREE,
          "a mapping with no tree at all was allowed");

    // 1: another client's handle is NOT FOUND, not denied - denied tells a guest the handle exists.
    tinynv_rm_map_req_t other = r; other.client = USER + 1;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &other) == TINYNV_RM_MAP_NO_OBJECT,
          "one client mapped another's allocation");

    // 8: the bound, and the wrap that would defeat it. C's `wrap` fixture is this row.
    tinynv_rm_map_req_t past = r; past.offset = 0x200000;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &past) == TINYNV_RM_MAP_OUT_OF_BOUNDS,
          "a mapping starting at the end of the allocation was allowed");
    tinynv_rm_map_req_t big = r; big.offset = 0x40000000;   // C's physical-side-supplied row, 1 GiB into 2 MiB
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &big) == TINYNV_RM_MAP_OUT_OF_BOUNDS,
          "a 1 GiB offset into a 2 MiB allocation was allowed");
    tinynv_rm_map_req_t wrap = r; wrap.offset = ~0ull - 0x1000;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &wrap) == TINYNV_RM_MAP_BAD_GEOMETRY,
          "an offset that wraps past the end of the address space was allowed - the bound check would have passed it");
    tinynv_rm_map_req_t vawrap = r; vawrap.va = ~0ull - 0x1000;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &vawrap) == TINYNV_RM_MAP_BAD_GEOMETRY,
          "a VA that wraps was allowed");
    tinynv_rm_map_req_t zero = r; zero.length = 0;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &zero) == TINYNV_RM_MAP_BAD_GEOMETRY,
          "a zero-length mapping was allowed");

    // 3's OTHER half. Until 2026-09-15 this was enforced NOWHERE: gsp.c said it was "arithmetic in tinynv_c4b_map",
    // a function that does not exist and never did, and tinynv_mm_map_range walks vaddr and size with no alignment
    // test on the way. So these three rows cover a guarantee that was documented, cited, and absent - which is worse
    // than an unimplemented one, because the citation is what stops anyone looking.
    tinynv_rm_map_req_t va_odd = r; va_odd.va += 1;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &va_odd) == TINYNV_RM_MAP_MISALIGNED,
          "a VA one byte off a page boundary was allowed - the entries this programs cover whole pages, so the "
          "mapping either misses the byte asked for or covers bytes that were not");
    tinynv_rm_map_req_t len_odd = r; len_odd.length -= 1;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &len_odd) == TINYNV_RM_MAP_MISALIGNED,
          "a length one byte short of a page multiple was allowed - it must be REFUSED rather than rounded up, "
          "because rounding up maps bytes past what the bound check above approved");
    tinynv_rm_map_req_t off_odd = r; off_odd.offset += 1;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &off_odd) == TINYNV_RM_MAP_MISALIGNED,
          "an offset one byte off a page boundary was allowed - the offset sets the physical base, so an unaligned "
          "one misaligns every entry even when the VA and the length are clean");
    // And the shape that must still pass, because a guarantee that refuses the real request is not a guarantee.
    // This is torch's own: base 0x200000000, offset 0, 2 MiB, all page-aligned.
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &r) == TINYNV_RM_MAP_OK,
          "the recorded request stopped being granted when the alignment check went in");
    CHECK((r.va % TINYNV_MAP_PAGE) == 0 && (r.length % TINYNV_MAP_PAGE) == 0,
          "the request this suite calls 'recorded' is not page aligned, so the row above proves nothing");

    // 8 again: no size means no bound, and no bound means refuse rather than assume.
    tinynv_rm_map_req_t nosize = r; nosize.handle = 0x5c00000e;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &nosize) == TINYNV_RM_MAP_SIZE_UNKNOWN,
          "an allocation of unknown size was mapped, so its offset was bounded against nothing");

    // 7: the case no recording will ever contain.
    tinynv_rm_map_req_t ro = r; ro.handle = 0x5c00000d;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &ro) == TINYNV_RM_MAP_WOULD_WRITE,
          "a read-write mapping of a read-only allocation was allowed");
    tinynv_rm_map_req_t ro_ok = ro; ro_ok.mapping_type = TINYNV_RM_MAP_TYPE_READ_ONLY;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &ro_ok) == TINYNV_RM_MAP_OK,
          "a READ-ONLY mapping (UvmRmGpuMappingTypeReadOnly = 3) of a read-only allocation was refused");
    // The row that caught the constant being off by one, and the only one that can. UvmRmGpuMappingTypeReadWrite is
    // 2; with READ_ONLY spelled 2 this request was GRANTED against read-only memory, which is the exact failure
    // guarantee 7 exists to prevent - and type 3 was refused instead, so the visible behaviour still looked
    // conservative while the real case was open.
    tinynv_rm_map_req_t rw2 = ro; rw2.mapping_type = 2;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &rw2) == TINYNV_RM_MAP_WOULD_WRITE,
          "mappingType 2 is UvmRmGpuMappingTypeReadWrite and was allowed against a read-only allocation - if this "
          "passes, TINYNV_RM_MAP_TYPE_READ_ONLY is 2 again and the check is inverted");
    tinynv_rm_map_req_t atomic = ro; atomic.mapping_type = 1;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &atomic) == TINYNV_RM_MAP_WOULD_WRITE,
          "ReadWriteAtomic (1) was allowed against a read-only allocation");
    tinynv_rm_map_req_t dflt = ro; dflt.mapping_type = 0;
    CHECK(tinynv_rm_map_check(mem, nmem, none, 0, DRIVER_ROOT, &dflt) == TINYNV_RM_MAP_WOULD_WRITE,
          "Default (0) was allowed against a read-only allocation - an unrecognised intent must take the stricter "
          "reading, not the permissive one");

    // 4: overlap in the same tree, and the same range in a different tree being fine.
    tinynv_rm_map_t live[] = {{USER, 0x5c00000c, GUEST_ROOT, 0x200000000ull, 0x200000}};
    CHECK(tinynv_rm_map_check(mem, nmem, live, 1, DRIVER_ROOT, &r) == TINYNV_RM_MAP_ALREADY_MAPPED,
          "a range was mapped twice with no unmap between");
    tinynv_rm_map_req_t partial = r; partial.va = 0x200100000ull; partial.length = 0x1000;
    CHECK(tinynv_rm_map_check(mem, nmem, live, 1, DRIVER_ROOT, &partial) == TINYNV_RM_MAP_ALREADY_MAPPED,
          "a range OVERLAPPING a live mapping was allowed - only an exact clash was being caught");
    tinynv_rm_map_t elsewhere[] = {{USER, 0x5c00000c, 0x9900000, 0x200000000ull, 0x200000}};
    CHECK(tinynv_rm_map_check(mem, nmem, elsewhere, 1, DRIVER_ROOT, &r) == TINYNV_RM_MAP_OK,
          "a mapping in ANOTHER tree blocked this one - the trees are not independent");
  }
  printf("  the C4b decision: the recorded request is granted, and each guarantee refuses with its own reason\n");

  printf("  the mapping record: exact unmaps only, partial refused as partial rather than unknown, and roots and "
         "clients do not blur\n");

  // THE EXIT STATUS IS THE LAST STATEMENT IN main, deliberately, and must stay that way. A guard placed anywhere
  // earlier stops covering whatever gets appended after it, and stops silently - there is no warning, the file just
  // starts passing. Making the final return itself the guard is the only shape a later section cannot grow past.
  printf("test_rm_free: %d checks, %d failed\n", checks, fails);
  return fails ? 1 : 0;
}
