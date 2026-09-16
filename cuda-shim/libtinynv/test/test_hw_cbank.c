// A kernel that reads a __device__ table, on the card, through the public interface.
//
// This is the shape every IQ-quantised llama.cpp kernel has: the table's address is not known until the image is placed,
// so the compiler puts a pointer to it in a constant bank and emits a relocation, and the driver has to apply the
// relocation and bind that bank. Neither step announces itself if it is missing - the kernel launches, reads a pointer
// that is zero, and returns whatever lives at that address - so this checks the values it returned and nothing else.
//
// What the table should contain is read out of the cubin rather than written down here, so the source and the
// expectation cannot drift apart: the loadable image holds the table, and the relocation says where.
#include "tinynv.h"
#include "cubin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 64

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "../spike/cbank.sm120.cubin";
  const char *kernel = argc > 2 ? argv[2] : "cbank";
  if (!getenv("TINYNV_HW") || !getenv("TINYNV_SOCKET")) {
    printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s [cubin] [kernel]\n", argv[0]);
    return 2;
  }

  FILE *f = fopen(path, "rb");
  if (!f) { printf("  cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  rewind(f);
  void *blob = malloc((size_t)len);
  if (!blob || fread(blob, 1, (size_t)len, f) != (size_t)len) { printf("  cannot read %s\n", path); return 1; }
  fclose(f);

  // what the table holds, from the image the driver would build: the bank's pointer is relocated to the table, so the
  // relocation whose place is inside bank 4 names where the table sits
  tinynv_cubin_t cb;
  tinynv_image_t im;
  if (tinynv_cubin_parse(blob, (size_t)len, &cb) || tinynv_cubin_image(&cb, kernel, 1, &im)) {
    printf("  %s\n", tinynv_last_error());
    return 1;
  }
  uint64_t table_off = 0;
  for (int i = 0; i < im.nrelocs; i++)
    if (im.relocs[i].at >= im.constbuf[4].off && im.relocs[i].at < im.constbuf[4].off + im.constbuf[4].size)
      table_off = im.relocs[i].target;
  if (!im.constbuf[4].used || !table_off) {
    printf("  %s does not read a table through constant bank 4, so it cannot test one\n", path);
    return 1;
  }
  printf("bank 4 at image offset %#llx binds a pointer relocated to the table at %#llx\n",
         (unsigned long long)im.constbuf[4].off, (unsigned long long)table_off);
  static unsigned want[N];
  memcpy(want, im.bytes + table_off, sizeof(want));

  printf("libtinynv build %s\n", tinynv_build_id());
  tinynv_device_t dev;
  tinynv_module_t mod;
  tinynv_kernel_t k;
  tinynv_kernel_info_t info;
  tinynv_stream_t s;
  tinynv_devptr_t out;
  tinynv_status_t st;
#define T(c) do { if ((st = (c)) != TINYNV_OK) { printf("  %s: %s\n    %s\n", #c, tinynv_status_str(st), tinynv_last_error()); return 1; } } while (0)
  T(tinynv_init());
  T(tinynv_device_get(&dev, 0));
  T(tinynv_module_load(dev, blob, (size_t)len, &mod));
  T(tinynv_get_kernel(mod, kernel, &k));
  T(tinynv_kernel_info(k, &info));
  T(tinynv_stream_create(dev, &s));
  T(tinynv_malloc(dev, sizeof(want), &out));

  static unsigned got[N];
  for (int i = 0; i < N; i++) got[i] = 0xdeadbeefu;   // not a plausible table entry, so "never written" is visible
  T(tinynv_memcpy_htod(s, out, got, sizeof(got)));

  unsigned char params[64];
  memset(params, 0, sizeof(params));
  uint64_t p = (uint64_t)out;
  unsigned n = N;
  memcpy(params + info.params[0].offset, &p, sizeof(p));
  memcpy(params + info.params[1].offset, &n, sizeof(n));
  T(tinynv_launch(s, k, 1, 1, 1, 64, 1, 1, 0, params, (size_t)info.params[1].offset + sizeof(n)));
  T(tinynv_stream_sync(s));
  T(tinynv_memcpy_dtoh(s, got, out, sizeof(got)));

  int wrong = 0, first = -1;
  for (int i = 0; i < N; i++) if (got[i] != want[i]) { if (first < 0) first = i; wrong++; }
  if (wrong) {
    printf("  FAIL: %d of %d entries wrong; the first is [%d] = %#010x where %#010x belongs\n", wrong, N, first,
           got[first], want[first]);
    printf("        %s\n", got[first] == 0xdeadbeefu ? "the kernel never wrote it: the launch did not reach the table"
                                                     : "the kernel read something, but not the table: the bank's pointer "
                                                       "is wrong, which means the relocation or the binding");
    return 1;
  }
  printf("all %d entries match the table in the cubin: the relocation was applied and bank 4 was bound\n", N);
  tinynv_image_free(&im);
  tinynv_cubin_free(&cb);
  free(blob);
  return 0;
}
