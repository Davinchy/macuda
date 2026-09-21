// A module whose device did not drain is NOT freed (G's review, finding 1; A): tinynv_module_unload refuses by name and
// the module, its code and its kernels stay usable; once the device drains, the unload frees it.
//
// The null device never uploads a module, so the real drain (tinynv_exec_idle) cannot be reached without a card. The
// decision goes through tinynv_module_unload_drain (tinynv.c), which this test replaces: one stand-in that FAILS the way
// the real drain does (a reason recorded, -1), one that succeeds. The default is restored for the last arm, which is
// the null device's own path (never uploaded -> nothing can be running -> freed).
//
// PRE-REGISTERED (B, before the first run): against the code before this change - which ignored the drain's result -
// arm "refused" gets TINYNV_OK and the module is freed. The CONTROL is tinynv.c with that one line put back (the drain
// asked, its answer ignored): it must fail the "refused" arm and pass the others.
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails, checks, drains;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static int drain_fails(tinynv_module_t m) { (void)m; drains++; return tinynv_fail("injected: the device did not go idle within 30 s"); }
static int drain_ok(tinynv_module_t m) { (void)m; drains++; return 0; }

static unsigned char *slurp(const char *path, long *len)
{
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END); *len = ftell(f); rewind(f);
	unsigned char *b = malloc((size_t)*len);
	if (b && fread(b, 1, (size_t)*len, f) != (size_t)*len) { free(b); b = NULL; }
	fclose(f);
	return b;
}

int main(int argc, char **argv)
{
	char path[512];
	long len = 0;
	snprintf(path, sizeof path, "%s/abi8-sm80.cubin", argc > 1 ? argv[1] : "test/fixtures");
	unsigned char *img = slurp(path, &len);
	tinynv_device_t d;
	tinynv_module_t m, m2;
	tinynv_kernel_t k;
	tinynv_kernel_info_t ki;
	int (*saved)(tinynv_module_t) = tinynv_module_unload_drain;

	CHECK(img != NULL, "cannot read %s", path);
	CHECK(tinynv_init() == TINYNV_OK && tinynv_device_get(&d, 0) == TINYNV_OK, "init");
	if (!img || fails) return 1;
	CHECK(tinynv_module_load(d, img, (size_t)len, &m) == TINYNV_OK, "load: %s", tinynv_last_error());
	CHECK(tinynv_get_kernel(m, "kfat", &k) == TINYNV_OK, "kfat: %s", tinynv_last_error());

	/* refused: the drain fails -> the unload is refused by name, and the module and its kernel still work */
	tinynv_module_unload_drain = drain_fails;
	drains = 0;
	tinynv_status_t st = tinynv_module_unload(m);
	const char *why = tinynv_last_error();
	CHECK(st != TINYNV_OK && strstr(why, "REFUSED") && strstr(why, "injected"),
	      "refused: an unload whose drain failed returned %d (\"%s\")", st, why);
	printf("  refused: drain failed -> %d, \"%.150s\"\n", st, why);
	CHECK(drains == 1, "refused: the drain was asked %d times, not once", drains);
	CHECK(st != TINYNV_OK && tinynv_get_kernel(m, "kfat", &k) == TINYNV_OK && tinynv_kernel_info(k, &ki) == TINYNV_OK &&
	      ki.max_threads > 0, "refused: the module or its kernel did not survive the refused unload");

	/* freed: the drain succeeds -> the same module unloads */
	tinynv_module_unload_drain = drain_ok;
	drains = 0;
	if (st != TINYNV_OK) {
		st = tinynv_module_unload(m);
		CHECK(st == TINYNV_OK && drains == 1, "freed: an unload after a successful drain returned %d (drain asked %d)", st, drains);
		printf("  freed: drain succeeded -> %d\n", st);
	}

	/* the default, on the null device: never uploaded, so nothing can be running, so it is freed */
	tinynv_module_unload_drain = saved;
	CHECK(tinynv_module_load(d, img, (size_t)len, &m2) == TINYNV_OK && tinynv_module_unload(m2) == TINYNV_OK,
	      "default: the null device's own unload failed: %s", tinynv_last_error());
	printf("  default: the null device's module (never uploaded) unloads -> OK\n");

	free(img);
	printf(fails ? "%d of %d checks failed\n" : "module unload: all %d checks passed\n", fails ? fails : checks, checks);
	return fails ? 1 : 0;
}
