/* the bench gate's reduction, v1 (bench_bf16.cu at 2dbc56f) against v2, on G's two mutants: an all-NaN result and one NaN */
#include <math.h>
#include <stdio.h>
static double v1(const float *x, const float *r, int n) { double mx = 0, mr = 0; for (int i = 0; i < n; i++) { mx = fmax(mx, fabs((double)x[i] - r[i])); mr = fmax(mr, fabs((double)r[i])); } return (isnan(mx) || mr == 0) ? INFINITY : mx / mr; }
static double v2(const float *x, const float *r, int n) { double mx = 0, mr = 0; for (int i = 0; i < n; i++) if (!isfinite(x[i]) || !isfinite(r[i])) return INFINITY; for (int i = 0; i < n; i++) { mx = fmax(mx, fabs((double)x[i] - r[i])); mr = fmax(mr, fabs((double)r[i])); } return mr == 0 ? INFINITY : mx / mr; }
int main(void) {
  float ref[8] = { 1, -2, 3, 0.5f, 4, -1, 2, 1 }, allnan[8], onenan[8];
  for (int i = 0; i < 8; i++) { allnan[i] = NAN; onenan[i] = ref[i]; }
  onenan[3] = NAN;
  printf("all-NaN: v1 %.3g (%s), v2 %.3g (%s)\n", v1(allnan, ref, 8), v1(allnan, ref, 8) > 2e-3 ? "refused" : "ACCEPTED", v2(allnan, ref, 8), v2(allnan, ref, 8) > 2e-3 ? "refused" : "ACCEPTED");
  printf("one NaN: v1 %.3g (%s), v2 %.3g (%s)\n", v1(onenan, ref, 8), v1(onenan, ref, 8) > 2e-3 ? "refused" : "ACCEPTED", v2(onenan, ref, 8), v2(onenan, ref, 8) > 2e-3 ? "refused" : "ACCEPTED");
  return 0;
}
