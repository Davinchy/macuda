/* test_library_unload.c - cuLibraryUnload when the host refuses to free the module (G's review, finding 1; A).
 *
 * Usage: test_library_unload <path/to/libcuda.so.1> <fixture dir>
 * NEEDS A SHIM LINKED STRAIGHT AGAINST libtinynv (the null device), as built on the 3090 box for test_module_load_data:
 * the refusal is injected through libtinynv's tinynv_module_unload_drain, which this test finds in the shim with dlsym.
 * A guest shim talks to the daemon instead, and there the same refusal is covered by tinynvd --selftest ("module
 * unload"). Run with TINYCU_NULL_WANT_SM=120 so the loader extracts the fixture's sm_120 image.
 *
 * Arms: a library is loaded and its kernel resolved (uploaded); the drain is made to FAIL and cuLibraryUnload must
 * return CUDA_ERROR_NOT_READY (600) with the library, its module and its kernel still usable (the kernel launches);
 * the drain is made to succeed and the same cuLibraryUnload returns 0. (A second unload is not tried: it would read the
 * freed record.)
 *
 * PRE-REGISTERED (B, before the first run): with the shim before this change (it freed the function records first and
 * forgot the module whatever the host said) against the new libtinynv, the refused unload returns 0 and the kernel no
 * longer launches - the control fails both "refused" checks.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("   FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)
static int drain_fails(void *m) { (void)m; return -1; }
static int drain_ok(void *m) { (void)m; return 0; }

int main(int argc, char **argv)
{
	if (argc < 3) { printf("usage: %s <libcuda.so.1> <fixture dir>\n", argv[0]); return 2; }
	void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	if (!h) { printf("cannot dlopen %s: %s\n", argv[1], dlerror()); return 2; }
	CUresult (*Init)(unsigned); CUresult (*DevGet)(int *, int); CUresult (*Retain)(void **, int); CUresult (*SetCur)(void *);
	CUresult (*LibLoad)(void **, const void *, void *, void **, unsigned, const int *, void **, unsigned);
	CUresult (*LibKernel)(void **, void *, const char *); CUresult (*LibUnload)(void *);
	CUresult (*Launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **);
	int (**seam)(void *) = dlsym(h, "tinynv_module_unload_drain");
#define GET(v, n) do { *(void **)&v = dlsym(h, n); if (!v) { printf("%s has no %s\n", argv[1], n); return 2; } } while (0)
	GET(Init, "cuInit"); GET(DevGet, "cuDeviceGet"); GET(Retain, "cuDevicePrimaryCtxRetain"); GET(SetCur, "cuCtxSetCurrent");
	GET(LibLoad, "cuLibraryLoadData"); GET(LibKernel, "cuLibraryGetKernel"); GET(LibUnload, "cuLibraryUnload");
	GET(Launch, "cuLaunchKernel");
	if (!seam) { printf("%s has no tinynv_module_unload_drain: this test needs a shim linked against libtinynv\n", argv[1]); return 2; }

	char p[1024]; snprintf(p, sizeof p, "%s/kfat_sm120.fatbin", argv[2]);
	FILE *f = fopen(p, "rb"); if (!f) { printf("cannot open %s\n", p); return 2; }
	fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
	unsigned char *fat = malloc((size_t)n); if (fread(fat, 1, (size_t)n, f) != (size_t)n) return 2; fclose(f);

	int dev; void *ctx, *lib = 0, *k = 0;
	if (Init(0) || DevGet(&dev, 0) || Retain(&ctx, dev) || SetCur(ctx)) { printf("no context\n"); return 2; }
	printf("== cuLibraryUnload with the host's module refused, through %s\n", argv[1]);
	CHECK(LibLoad(&lib, fat, 0, 0, 0, 0, 0, 0) == 0 && LibKernel(&k, lib, "kfat") == 0 && k, "load and resolve kfat");
	uint64_t ptr = 0x1000; void *params[1] = { &ptr };

	int (*saved)(void *) = *seam;
	*seam = drain_fails;
	CUresult rc = LibUnload(lib);
	CHECK(rc == 600, "refused: cuLibraryUnload with a failed drain returned %d, want 600 (NOT_READY)", rc);
	CUresult rl = Launch(k, 1, 1, 1, 32, 1, 1, 0, 0, params, 0);
	CHECK(rl == 0, "refused: the kernel no longer launches after the refused unload (%d)", rl);
	printf("   refused: cuLibraryUnload -> %d, and the kernel still launches -> %d\n", rc, rl);

	*seam = drain_ok;
	rc = LibUnload(lib);
	CHECK(rc == 0, "freed: cuLibraryUnload after a successful drain returned %d", rc);
	printf("   freed: cuLibraryUnload -> %d\n", rc);
	*seam = saved;

	printf(fails ? "   %d check(s) FAILED\n" : "   all arms passed\n", fails);
	return fails ? 1 : 0;
}
