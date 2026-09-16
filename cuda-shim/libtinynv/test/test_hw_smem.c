// How much dynamic shared memory a launch actually gets, measured rather than assumed.
//
// A launch that returns without an error proves the front end accepted the numbers in the descriptor. It does not prove
// the memory is there. Above 100 KB the configuration fields are territory no recorded session has been through, and the
// difference between "the descriptor was accepted" and "the card gave me this much" is the difference between a kernel
// that works and one that reads a neighbour's data - which is why this makes the kernel write a pattern through the
// whole span it was promised and read it back. (Session A's design, 2026-09-14.)
//
// One size per run, on purpose. A sweep that walks upward until something breaks leaves the broken one last, and a
// faulting launch is the state this driver has least evidence about; stepping by hand means the operator sees each
// result before deciding whether there should be another.
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  const char *cubin_path = argc > 1 ? argv[1] : "../spike/smem_probe.sm120.cubin";
  unsigned bytes = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 0) : 64 * 1024;
  if (!getenv("TINYNV_HW") || !getenv("TINYNV_SOCKET")) {
    printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s [cubin] [bytes]\n", argv[0]);
    printf("  measures how much dynamic shared memory a launch really gets. take the gpu lock first.\n");
    return 2;
  }
  bytes &= ~3u; // the kernel works in words

  FILE *f = fopen(cubin_path, "rb");
  if (!f) { printf("  cannot open %s\n", cubin_path); return 1; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  rewind(f);
  void *blob = malloc((size_t)len);
  if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) { printf("  cannot read %s\n", cubin_path); return 1; }
  fclose(f);

  printf("libtinynv build %s\n", tinynv_build_id());
  tinynv_device_t dev;
  tinynv_module_t mod;
  tinynv_kernel_t k;
  tinynv_kernel_info_t info;
  tinynv_stream_t s;
  tinynv_devptr_t out;
  tinynv_status_t st;
#define TRY(call) do { if ((st = (call)) != TINYNV_OK) { \
    printf("  %s: %s\n    %s\n", #call, tinynv_status_str(st), tinynv_last_error()); return 1; } } while (0)
  TRY(tinynv_init());
  TRY(tinynv_device_get(&dev, 0));
  TRY(tinynv_module_load(dev, blob, (size_t)len, &mod));
  TRY(tinynv_get_kernel(mod, "smem_probe", &k));
  TRY(tinynv_kernel_info(k, &info));
  TRY(tinynv_stream_create(dev, &s));
  TRY(tinynv_malloc(dev, sizeof(unsigned), &out));

  unsigned none = 0xffffffffu, first_bad = 0;
  TRY(tinynv_memcpy_htod(s, out, &none, sizeof(none)));

  unsigned char params[64];
  memset(params, 0, sizeof(params));
  uint64_t p0 = (uint64_t)out;
  memcpy(params + info.params[0].offset, &p0, sizeof(p0));
  memcpy(params + info.params[1].offset, &bytes, sizeof(bytes));
  size_t params_len = (size_t)info.params[1].offset + sizeof(bytes);

  // Two different runs share this test, and they must not share a code path.
  //
  // Inside the configurations the card has been seen to accept, nothing is raised and nothing is set: the run goes
  // through exactly the path a real kernel takes, and it is the first proof on silicon that dynamic shared memory works
  // at all - ggml asks for tens of kilobytes on every matmul and none of it has been exercised yet.
  //
  // Only above that ceiling is the knob touched, because there the driver refuses by design and measuring is a
  // deliberate act. Setting it either way would mean the in-range run was not the production path.
  const unsigned PINNED = 100 * 1024;
  int raised = bytes + 0x400 > PINNED;
  if (raised) {
    // Permit the whole of the rest of the ladder and let the driver pick a real rung: a size between two of them is
    // accepted by the front end and then never scheduled, which is a stall rather than an error.
    setenv("TINYNV_SMEM_CEILING_KB", "228", 1);
    printf("asking for %u bytes (%.1f KB), above the %u KB this driver has evidence for; the remaining carveouts are "
           "permitted for the measurement\n", bytes, bytes / 1024.0, PINNED / 1024);
  } else {
    printf("asking for %u bytes (%.1f KB) of dynamic shared memory in one block of 256 threads, through the ordinary "
           "path with nothing raised\n", bytes, bytes / 1024.0);
  }
  st = tinynv_launch(s, k, 1, 1, 1, 256, 1, 1, bytes, params, params_len);
  if (st != TINYNV_OK) {
    // A refusal here is a result too, and the interesting one while the ceiling is where it is.
    printf("no result: %s\n", tinynv_last_error());
    printf("  a refusal before submitting is this driver declining; a wait that ran out is the card accepting the\n"
           "  descriptor and never scheduling the block, which is what an invalid carveout looks like.\n");
    return 1;
  }
  TRY(tinynv_stream_sync(s));
  TRY(tinynv_memcpy_dtoh(s, &first_bad, out, sizeof(first_bad)));

  if (first_bad == 0xffffffffu) {
    printf("every one of the %u words read back: the card really gave this launch %u bytes\n", bytes / 4, bytes);
    return 0;
  }
  printf("the first word that did not read back is %u, so the span is about %u bytes, not the %u asked for\n",
         first_bad, first_bad * 4, bytes);
  return 1;
}
