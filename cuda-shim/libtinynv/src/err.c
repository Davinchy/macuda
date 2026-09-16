// One error slot per thread. Every entry point returns a status and leaves the detail here, because a driver failure is
// usually one sentence ("bar 1 is 256 MB, the ring does not fit") that the caller should print verbatim.
//
// A success does not clear it. That is deliberate: a caller that notices a failure and then prints the reason has made
// several successful calls in between - looking up a status string, asking what device it had - and clearing on success
// would leave it printing "no error" for a call that plainly failed.
#include "internal.h"
#include <time.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local char g_err[512];

int tinynv_fail(const char *fmt, ...) {
  // formatted into a scratch buffer first: a caller that passes tinynv_last_error() as an argument would otherwise be
  // reading the destination while vsnprintf writes it, which is undefined and in practice truncates the message
  char tmp[sizeof(g_err)];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(tmp, sizeof(tmp), fmt, ap);
  va_end(ap);
  memcpy(g_err, tmp, sizeof(g_err));
  return -1;
}

const char *tinynv_last_error(void) { return g_err[0] ? g_err : "no error"; }

double tinynv_now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int g_faithful = -1;

int tinynv_is_faithful(void) {
  if (g_faithful < 0) g_faithful = getenv("TINYNV_FAITHFUL") != NULL;
  return g_faithful;
}

void tinynv_set_faithful(int on) { g_faithful = on ? 1 : 0; }
