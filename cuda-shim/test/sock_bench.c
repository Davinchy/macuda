// sock_bench: prices the socket path a flush pays, on the same window the driver uses (BAR1 onto video memory, out
// of the processor-visible reserve). tinynv_submit writes the shadow without waiting, then reads the ring's write
// pointer back before ringing the doorbell. That read waits for a reply, and the server answers it only after it
// has applied everything queued ahead of it - so a read behind a write of N bytes prices the server's write path.
// Measured here: a read alone; a write of N bytes then a read (per-byte price); k small writes then a read
// (per-message price); the client's own cost of sending a write; and the flush's exact shape at depth 32/64/128.
// Build like a driver test (see the command in docs/SHARED-STATUS.md); run under the protocol:
//   sh tools/nv_shim_step.sh A probe cuda-shim/build/shim/nv/sock_bench
#include "tinynv.h"
#include "gpu.h"
#include "internal.h"
#include "mmu.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static tinynv_gpu_t g;
static volatile sig_atomic_t opened, quiesced;
static void on_signal(int sig) { (void)sig; if (opened && !quiesced) { quiesced = 1; tinynv_dev_quiesce(&g.dev); } _exit(130); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + (double)t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
  const char *sock = argc > 1 ? argv[1] : getenv("TINYNV_SOCKET");
  if (!sock || !getenv("TINYNV_HW")) { printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s\n", argv[0]); return 2; }
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("libtinynv build %s\n", tinynv_build_id());
  struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = on_signal; sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, NULL); sigaction(SIGTERM, &sa, NULL);
  tinynv_pci_t pci;
  if (tinynv_pci_open_tinygpu(sock, &pci)) { printf("cannot reach the gpu on %s: %s\n", sock, tinynv_last_error()); return 1; }
  int rc = tinynv_gpu_open(&g, &pci); opened = 1;
  if (rc) { printf("bring-up failed: %s\n", tinynv_last_error()); goto done; }
  if ((rc = tinynv_gpu_init_sw(&g)) || (rc = tinynv_gpu_init_hw(&g))) { printf("init failed: %s\n", tinynv_last_error()); goto done; }
  printf("GSP-RM is up; bar1 window %llu MB\n", (unsigned long long)(g.dev.vram.size >> 20));
  // the same kind of memory the command arena lives in (exec.c: host 0, cpu_access 1, uncached 1, force_devmem 1)
  tinynv_vmap_t buf;
  if ((rc = tinynv_mm_alloc_buffer(&g.mm, 2u << 20, 0, 1, 1, 1, 0, &buf))) { printf("alloc failed: %s\n", tinynv_last_error()); goto done; }
  {
  uint64_t pa = buf.ranges[0].paddr;
  tinynv_mmio_t *w = &g.dev.vram;
  printf("scratch at video memory %#llx (%d range%s)\n", (unsigned long long)pa, buf.nranges, buf.nranges == 1 ? "" : "s");
  uint8_t *src = malloc(1u << 20);
  for (size_t i = 0; i < (1u << 20); i++) src[i] = (uint8_t)(i * 7 + 3);
  nv_wr32(w, pa, 0x600df00d);
  uint32_t v = nv_rd32(w, pa);
  printf("write then read of one word: %s\n", v == 0x600df00d ? "reads back as written" : "DOES NOT READ BACK");
  for (int i = 0; i < 50; i++) nv_rd32(w, pa);
  double t0, t1; int reps;
  reps = 400; t0 = now(); for (int i = 0; i < reps; i++) nv_rd32(w, pa); t1 = now();
  double rd = (t1 - t0) / reps * 1e6;
  printf("\n  a read alone (the round trip):                %8.1f us\n", rd);
  reps = 400; t0 = now(); for (int i = 0; i < reps; i++) nv_wr32(w, pa + 4 * (uint64_t)(i & 255), (uint32_t)i); t1 = now();
  double wr = (t1 - t0) / reps * 1e6;
  t0 = now(); nv_rd32(w, pa); t1 = now();
  printf("  a 4-byte write, as the client sees it:        %8.2f us  (sent, not waited for; the read that drained the 400 took %.0f us)\n", wr, (t1 - t0) * 1e6);
  printf("\n  one write of N bytes, then one read. pair less the read alone = the server's price for the write\n");
  printf("  %10s %12s %12s %10s %10s\n", "bytes", "pair us", "write us", "us/KB", "MB/s");
  const size_t sizes[] = {8, 64, 512, 1536, 4096, 16384, 49152, 98304, 196608, 524288, 1048576};
  for (size_t si = 0; si < sizeof(sizes) / sizeof(*sizes); si++) {
    size_t n = sizes[si]; reps = n >= 196608 ? 20 : 60;
    nv_wr_block(w, pa, src, n); nv_rd32(w, pa);
    t0 = now(); for (int i = 0; i < reps; i++) { nv_wr_block(w, pa, src, n); nv_rd32(w, pa + n - 4); } t1 = now();
    double pair = (t1 - t0) / reps * 1e6, cost = pair - rd;
    printf("  %10zu %12.1f %12.1f %10.2f %10.1f\n", n, pair, cost, cost / ((double)n / 1024.0), (double)n / cost);
  }
  uint32_t last; memcpy(&last, src + 1048576 - 4, 4);
  printf("  last word of the 1 MiB write reads back %s\n", nv_rd32(w, pa + 1048576 - 4) == last ? "as written" : "WRONG");
  printf("\n  k 8-byte writes, then one read: the server's price per message\n");
  printf("  %10s %12s %12s %10s\n", "messages", "pair us", "writes us", "us/msg");
  const int ks[] = {1, 4, 16, 32, 64, 128};
  for (size_t ki = 0; ki < sizeof(ks) / sizeof(*ks); ki++) {
    int k = ks[ki]; reps = 40; uint64_t e = 0x1122334455667788ull;
    t0 = now(); for (int i = 0; i < reps; i++) { for (int j = 0; j < k; j++) nv_wr_block(w, pa + 8 * (uint64_t)j, &e, 8); nv_rd32(w, pa); } t1 = now();
    double pair = (t1 - t0) / reps * 1e6, cost = pair - rd;
    printf("  %10d %12.1f %12.1f %10.2f\n", k, pair, cost, cost / k);
  }
  printf("\n  the flush as tinynv_submit does it: shadow (1536 B a launch), ring entry, pointer, readback (doorbell not reproduced)\n");
  const int depths[] = {32, 64, 128};
  for (size_t di = 0; di < 3; di++) {
    int d = depths[di]; size_t n = (size_t)1536 * (size_t)d; reps = 30; uint64_t e = 1; uint32_t p = 1;
    t0 = now();
    for (int i = 0; i < reps; i++) { nv_wr_block(w, pa, src, n); nv_wr_block(w, pa + n, &e, 8); nv_wr32(w, pa + n + 8, p); nv_rd32(w, pa + n + 8); }
    t1 = now();
    double fl = (t1 - t0) / reps * 1e6;
    printf("  depth %3d: %7zu bytes  flush %8.1f us  per launch %6.2f us\n", d, n, fl, fl / d);
  }
  free(src);
  tinynv_vmap_free(&g.mm, &buf);
  }
done:
  if (opened && !quiesced) { quiesced = 1; tinynv_gpu_close(&g); }
  return rc ? 1 : 0;
}
