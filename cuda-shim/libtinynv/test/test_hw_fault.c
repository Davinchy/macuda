// Does the fault report say what actually happened?
//
// The driver can now ask the debugger where a fault was instead of printing that there was one. That path cannot be
// tested without a card and a real fault, so this causes one on purpose, twice, in the two ways that lead opposite
// directions - and checks the report distinguishes them rather than merely producing output.
//
//   mode a: a kernel stores to a virtual address a known distance past the end of a real mapping. Pass is the report
//           naming THAT address, as a missing page table entry, written to. An address that is merely plausible is
//           what this exists to catch: the whole feature is worthless if it names the wrong one convincingly.
//   mode b: a kernel executes `trap`. Pass is the opposite shape - no mmu fault claimed, an SM error instead. Before
//           the report existed these two were the same sentence, and they mean different bugs with different fixes.
//
// THIS TEST DELIBERATELY FAULTS THE CARD. That is a different risk from every other test here: a fault can end in a
// reset, and over Thunderbolt it can end in a replug that only a person can do. So it runs last in a session, one mode
// per process, with the card quiesced between, and it needs its own go rather than inheriting a session's. Nothing
// here is safe to fold into a soak.
//
// One mode per process because the first fault leaves the channel in recovery and the report is asked once: a second
// fault in the same process would be asked about with the answer already spent, and would test nothing.
#include "tinynv.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)
#define OK(call) do { tinynv_status_t st_ = (call); \
  if (st_ != TINYNV_OK) { printf("  FAIL: %s: %s\n         %s\n", #call, tinynv_status_str(st_), tinynv_last_error()); \
                          return ++fails; } } while (0)

#define N 256

// The numbers the report should come back with. Spelled here rather than included from the driver's own header on
// purpose: this checks the value that reaches a caller, and a test that takes both sides from one definition cannot
// see a definition that is wrong. They are asserted against NVIDIA's header in test_headers, which is where that
// question belongs.
// A missing mapping reports the level the walk got to before it ran out, so the kind depends on HOW FAR OUT the
// address is, not on whether it is mapped: inside a directory that exists it is a missing page table entry, and far
// enough out that the directory does not exist either it is a missing page directory entry. Session A found this on
// the card - the first pass expected PTE and got PDE at a 256 GiB gap, which is the right answer for that gap. Both
// mean "nothing is mapped there", which is the claim this test is actually making, so both are accepted and the run
// prints which one came back.
#define WANT_TYPE_PDE   0x0u
#define WANT_TYPE_PTE   0x2u
#define WANT_ACCESS_WRITE 0x1u

static void *read_file(const char *path, size_t *len_out) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  rewind(f);
  void *blob = malloc((size_t)len);
  if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) { free(blob); fclose(f); return NULL; }
  fclose(f);
  *len_out = (size_t)len;
  return blob;
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "";
  // How far past the end of a real allocation to aim. A knob rather than a constant because the one thing that can make
  // mode a report nothing is this address landing inside something else that happens to be mapped, and finding that out
  // on the card should cost another run rather than another build.
  // Far enough out that nothing else has claimed it. The first default here was 64 MB and on the card that address
  // was inside something already mapped, so the kernel wrote to valid memory and nothing faulted - the confound this
  // knob exists for, hit on the first run. 256 GiB is what worked; it is still an argument because the right distance
  // is a property of the address space on the day, not of this file.
  unsigned long long gap = argc > 2 ? strtoull(argv[2], NULL, 0) : (256ull << 30);
  const char *cubin_path = argc > 3 ? argv[3] : NULL;
  if (!getenv("TINYNV_HW") || !getenv("TINYNV_SOCKET") || (strcmp(mode, "unmapped") && strcmp(mode, "trap"))) {
    printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s unmapped|trap [gap bytes] [cubin]\n", argv[0]);
    printf("  DELIBERATELY FAULTS THE GPU. Take the lock, run it last, quiesce after, and have a go for this step.\n");
    printf("  unmapped: a store past the end of a real mapping   -> expect an mmu fault at that address\n");
    printf("            (the gap defaults to 256 GiB; too small and the address is mapped and nothing faults)\n");
    printf("  trap:     a kernel that traps                      -> expect no mmu fault and an SM error\n");
    return 2;
  }
  int want_mmu = !strcmp(mode, "unmapped");
  if (!cubin_path) cubin_path = want_mmu ? "../spike/vecadd.sm120.cubin" : "../spike/trap_kernel.sm120.cubin";

  size_t len = 0;
  void *blob = read_file(cubin_path, &len);
  if (!blob) {
    printf("  cannot read %s\n", cubin_path);
    if (!want_mmu) printf("  the trap cubin is build output: python tools/build_spike_cubin.py ../spike/trap_kernel.cu\n"
                          "  (which needs the compile server; see env.sh)\n");
    return 1;
  }

  printf("libtinynv build %s\n", tinynv_build_id());
  printf("mode %s\n", mode);
  tinynv_device_t dev;
  OK(tinynv_init());
  OK(tinynv_device_get(&dev, 0));

  tinynv_module_t mod;
  tinynv_kernel_t k;
  OK(tinynv_module_load(dev, blob, len, &mod));
  OK(tinynv_get_kernel(mod, want_mmu ? "vecadd" : "trap_after_write", &k));

  tinynv_devptr_t good;
  OK(tinynv_malloc(dev, N * sizeof(float), &good));
  printf("a real mapping is at %#llx\n", (unsigned long long)good);

  tinynv_kernel_info_t info;
  OK(tinynv_kernel_info(k, &info));
  tinynv_stream_t s;
  OK(tinynv_stream_create(dev, &s));

  // The parameter block is laid out from the kernel's own table, the way the shim marshals it: the offsets are the
  // cubin's, not this file's guesses, and a kernel whose parameters are not the shape expected fails here with the
  // reason rather than on the card with a fault this test would then happily attribute to its own poison.
  unsigned char params[256];
  memset(params, 0, sizeof(params));
  size_t params_len = 0;
  uint64_t slots[4];
  int nslots;
  unsigned long long poison = 0;
  int n = N;
  if (want_mmu) {
    // The output pointer, and only it, is moved off the end of everything this process mapped. The inputs stay valid so
    // the kernel runs and faults on the store rather than never starting: a fault fetching a parameter would be a
    // different fault at a different address, and would not exercise the path the report is meant to explain.
    poison = (unsigned long long)good + gap;
    printf("storing to %#llx, %llu MB past it, which should be mapped by nothing\n", poison, gap >> 20);
    slots[0] = (uint64_t)good; slots[1] = (uint64_t)good; slots[2] = (uint64_t)poison; nslots = 3;
  } else {
    slots[0] = (uint64_t)good; nslots = 1;
  }
  CHECK(info.num_params == nslots + 1, "%s takes %d parameters, this mode passes %d", want_mmu ? "vecadd" : "trap_after_write",
        info.num_params, nslots + 1);
  if (fails) return fails;
  for (int i = 0; i < info.num_params; i++) {
    const void *src = i < nslots ? (const void *)&slots[i] : (const void *)&n;
    size_t want = i < nslots ? sizeof(uint64_t) : sizeof(int);
    CHECK((size_t)info.params[i].size == want, "parameter %d is %d bytes, this test passes %zu", i, info.params[i].size,
          want);
    memcpy(params + info.params[i].offset, src, want);
    if ((size_t)info.params[i].offset + want > params_len) params_len = (size_t)info.params[i].offset + want;
  }
  if (fails) return fails;

  tinynv_status_t launched = tinynv_launch(s, k, N / 64, 1, 1, 64, 1, 1, 0, params, params_len);
  if (launched != TINYNV_OK) printf("the launch itself was refused: %s\n", tinynv_status_str(launched));

  // The fault surfaces where something waits for work that will never finish, which is what this is. It takes the
  // driver's own timeout to get there, so this is the slow part of the test and not a hang.
  tinynv_status_t synced = tinynv_stream_sync(s);
  printf("sync returned %s\n", tinynv_status_str(synced));
  CHECK(synced != TINYNV_OK, "the work completed, so nothing faulted and there is nothing to report on%s",
        want_mmu ? " - try a larger gap" : "");

  uint64_t va = 0;
  uint32_t type = 0, access = 0, esr = 0;
  int kind = tinynv_device_last_fault(dev, &va, &type, &access, &esr);

  if (want_mmu) {
    CHECK(kind == 1, "the report says %s, not an mmu fault",
          kind == 2 ? "an SM error" : "nothing at all");
    if (kind == 1) {
      // Same page, not the same byte: the address the hardware records is the access that faulted, and a store the
      // compiler widened or an access the engine coalesced can name a neighbour within the page it touched.
      CHECK((va & ~0xfffull) == (poison & ~0xfffull),
            "the fault names %#llx, which is not in the page of %#llx - the report is confident and wrong, which is "
            "worse than silent", (unsigned long long)va, poison);
      CHECK(type == WANT_TYPE_PTE || type == WANT_TYPE_PDE,
            "the fault kind is %#x; a store to nothing should be a missing page table entry (%#x) or, far enough out, "
            "a missing page directory entry (%#x)", type, WANT_TYPE_PTE, WANT_TYPE_PDE);
      CHECK(access == WANT_ACCESS_WRITE, "the access is %#x, expected %#x (write)", access, WANT_ACCESS_WRITE);
      if (!fails) printf("the report named %#llx, %s, written to - the address this test chose\n",
                         (unsigned long long)va,
                         type == WANT_TYPE_PDE ? "no page directory entry that far out" : "no page table entry");
    }
  } else {
    CHECK(kind != 1, "an mmu fault at %#llx is claimed for a kernel that only executed trap, so the two cases are "
          "still not being told apart", (unsigned long long)va);
    CHECK(kind == 2, "no SM error was reported for a kernel that trapped");
    if (kind == 2 && !fails) printf("no mmu fault claimed, SM error %#x - the trap was read as a trap\n", esr);
  }

  free(blob);
  // Deliberately not tearing the device down by hand: the library's own exit path puts the card down, and this test is
  // exactly the situation that path exists for.
  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
