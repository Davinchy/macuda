// The architecture is read from the right byte of e_flags for BOTH CUDA ELF ABIs.
//
// One PTX (fixtures/abi-kfat.ptx, from abi-kfat.cu) compiled by two ptxas: 12.8 (the cu128 site's Triton bundle, V12.8.93)
// and 13.0 (V13.0.88). ptxas 12.8 emits ABI 7 (OSABI 0x33) for sm_80 and sm_90 and ABI 8 for sm_120; ptxas 13.0 emits
// ABI 8 throughout. So abi7-sm80 and abi8-sm80 are the SAME kernel for the SAME architecture in the two layouts, and
// the only thing that can make them disagree here is which byte is read. README-abi.md has the hashes.
//
// Also checked per image, because the architecture decides it: the parameter base (0x160 on sm_8x, 0x210 on sm_90,
// 0x380 on sm_12x) and that the one kernel is found by name - the rest of the parse must not care which ABI it met.
#include "cubin.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails, checks;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

int main(int argc, char **argv)
{
	const char *dir = argc > 1 ? argv[1] : "test/fixtures";
	static const struct { const char *file; int osabi; unsigned arch, base; } F[] = {
		{"abi7-sm80.cubin", 0x33, 80, 0x160},
		{"abi8-sm80.cubin", 0x41, 80, 0x160},
		{"abi7-sm90.cubin", 0x33, 90, 0x210},
		{"abi8-sm120-ptxas12.cubin", 0x41, 120, 0x380},
	};
	for (size_t i = 0; i < sizeof F / sizeof F[0]; i++) {
		char path[512];
		snprintf(path, sizeof path, "%s/%s", dir, F[i].file);
		FILE *f = fopen(path, "rb");
		if (!f) { printf("  FAIL: cannot open %s\n", path); fails++; continue; }
		fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
		unsigned char *b = malloc((size_t)n);
		if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { printf("  FAIL: cannot read %s\n", path); fails++; fclose(f); free(b); continue; }
		fclose(f);
		CHECK(b[7] == F[i].osabi, "%s: OSABI %#x, the fixture was made as %#x", F[i].file, b[7], F[i].osabi);
		tinynv_cubin_t c;
		if (tinynv_cubin_parse(b, (size_t)n, &c)) { printf("  FAIL: %s does not parse: %s\n", F[i].file, tinynv_last_error()); fails++; free(b); continue; }
		const tinynv_kernel_desc_t *k = tinynv_cubin_kernel(&c, "kfat");
		CHECK(c.sm_arch == F[i].arch, "%s (ABI %d): sm_%u, want sm_%u", F[i].file, b[8], c.sm_arch, F[i].arch);
		CHECK(k && k->param_base == F[i].base, "%s: kfat %s, param_base %#x, want %#x", F[i].file, k ? "found" : "NOT found",
		      k ? k->param_base : 0, F[i].base);
		printf("  %-26s OSABI %#04x ABI %d e_flags %#010x -> sm_%u, kfat params at c[0][%#x]\n", F[i].file, b[7], b[8],
		       (unsigned)(b[0x30] | b[0x31] << 8 | b[0x32] << 16 | (unsigned)b[0x33] << 24), c.sm_arch, k ? k->param_base : 0);
		tinynv_cubin_free(&c);
		free(b);
	}
	printf(fails ? "%d of %d checks failed\n" : "ELF ABI 7 and 8: all %d checks passed\n", fails ? fails : checks, checks);
	return fails ? 1 : 0;
}
