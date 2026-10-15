// What the card is doing, while it is doing it.
//
// The awkward constraint this is shaped around: the TinyGPU broker accepts ONE client at a time, so this cannot open
// the card while a model is running - and that is exactly when the numbers are interesting. Worse, trying would sit in
// the broker's one-deep backlog and hold the slot a live run needs.
//
// So there are two ways in and it takes whichever is available. If a process holding the card is publishing (it does so
// from the synchronisation points it is already waiting at, four times a second), this reads that. If nothing holds the
// card, it opens the device itself. The first is the case that matters; the second is the convenience.
//
// Fan speed is not here, and not because it was skipped. NVIDIA publishes no way to ask: the two headers that would
// carry fan and thermal controls ship as empty stubs, and there is no fan field anywhere in the block the firmware
// keeps. Temperature, power, clocks, utilisation and the reason the clocks are being held back are the whole of what
// this card will tell us.
#include "tinynv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// Wall clock, matching what the publisher writes. Two processes cannot compare a per-process clock, and getting that
// wrong shows up as a reading from the future rather than as an error.
static double now_s(void) { struct timeval t; gettimeofday(&t, NULL); return (double)t.tv_sec + t.tv_usec / 1e6; }

static const char *published_path(void) {
  static char buf[512];
  const char *p = getenv("TINYNV_SENSOR_FILE");
  if (p && *p) return p;
  const char *dir = getenv("TMPDIR");
  snprintf(buf, sizeof(buf), "%stinynv-sensors", dir && *dir ? dir : "/tmp/");
  return buf;
}

// The published file is plain text on purpose: readable with cat, and this parser is a dozen lines rather than a
// format. Returns 0 if it was read and is fresh.
static int show_published(double stale_after) {
  FILE *f = fopen(published_path(), "r");
  if (!f) return -1;
  char line[256], out[4096];
  size_t n = 0;
  double stamp = -1;
  int pid = 0;
  while (fgets(line, sizeof(line), f)) {
    double v;
    if (sscanf(line, "time %lf", &v) == 1) { stamp = v; continue; }
    if (sscanf(line, "pid %d", &pid) == 1) continue;
    n += (size_t)snprintf(out + n, n < sizeof(out) ? sizeof(out) - n : 0, "  %s", line);
  }
  fclose(f);
  if (stamp < 0) return -1;
  double age = now_s() - stamp;
  // A stale file is worse than no file: it is a temperature from a run that ended, and nothing about it says so. A
  // reading from the future is worse again, because it would pass every freshness test - so it is refused rather than
  // trusted, and it means the clocks disagree rather than that the card is hot.
  if (age < -1.0) {
    printf("the published reading is timestamped %.0f seconds in the future, so its clock and this one disagree; "
           "not trusting it\n", -age);
    return -1;
  }
  if (age > stale_after) {
    printf("the last reading is %.0f seconds old, from process %d, so nothing is holding the card now\n", age, pid);
    return -1;
  }
  printf("from the process holding the card (pid %d, %.1fs ago)\n%s", pid, age, out);
  return 0;
}

// Whether somebody else is driving the card right now.
//
// This exists because of a real incident rather than a worry: the fallback below opened the device three times during
// a live decode, because nothing was publishing and it took that as "nobody is here". It is not. A card held by a
// process WITHOUT sensors looks exactly like a card nobody holds - the absence of a publication is the absence of
// evidence, and I built the fallback as though it were evidence of absence.
//
// The lock is the project's own protocol and the only thing that actually knows. Checked by reading the file rather
// than by running the script, because this must not depend on a shell being available or on a path that can move.
static int someone_holds_the_card(char *who, size_t n) {
  const char *root = getenv("EGPU_ROOT");
  char path[512];
  snprintf(path, sizeof(path), "%s/.gpu-lock", root && *root ? root : "."  /* EGPU_ROOT, else the working directory */);
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  if (!fgets(who, (int)n, f)) { who[0] = 0; }
  fclose(f);
  who[strcspn(who, "\n")] = 0;
  return 1;
}

static void show_direct(void) {
  // Refuse rather than open. Opening would take the broker's only client slot - it accepts one connection at a time -
  // and the run that is already there would be the one that suffered, which is precisely backwards.
  char holder[256] = {0};
  if (someone_holds_the_card(holder, sizeof(holder))) {
    printf("the card is held by another session and nothing is publishing, so there is nothing to show.\n"
           "  %s\n"
           "  not opening the device: the broker takes one client at a time and this would take the slot.\n", holder);
    return;
  }
  tinynv_device_t d = NULL;
  if (tinynv_device_get(&d, 0) != TINYNV_OK) {
    printf("nothing is publishing and no device could be opened (%s)\n", tinynv_last_error());
    return;
  }
  tinynv_sensors_t s;
  tinynv_status_t rc = tinynv_sensors(d, &s);
  // The null device is not a card with broken sensors, and saying so matters: one means check the hardware, the other
  // means TINYNV_SOCKET is unset.
  if (rc == TINYNV_ERR_NODEV) { printf("no card attached: TINYNV_SOCKET is unset, so this is the null device\n"); return; }
  if (rc != TINYNV_OK) { printf("the card is open but reports no sensors (%s)\n", tinynv_last_error()); return; }
  printf("read directly from an idle card\n");
  if (s.gpu_temp_ok) printf("  gpu_temp_c %.1f\n", s.gpu_temp_c);
  if (s.mem_temp_ok) printf("  mem_temp_c %.1f\n", s.mem_temp_c);
  if (s.power_ok) printf("  gpu_power_w %.1f\n", s.gpu_power_mw / 1000.0);
  if (s.limit_ok) printf("  power_limit_w %.1f\n", s.power_limit_mw / 1000.0);
  if (s.clock_ok) printf("  graphics_mhz %u\n  memory_mhz %u\n", s.graphics_mhz, s.memory_mhz);
  if (s.util_ok) printf("  gpu_busy_pct %u\n  mem_busy_pct %u\n", s.gpu_busy_pct, s.mem_busy_pct);
  if (s.throttle_ok) {
    const char *why[9];
    int n = tinynv_throttle_str(s.throttle, why, 9);
    printf("  throttle %s", n ? "" : "none");
    for (int i = 0; i < n; i++) printf("%s%s", i ? ", " : "", why[i]);
    printf("\n");
  }
  if (!s.gpu_temp_ok && !s.power_ok && !s.clock_ok)
    printf("  (the card reported none of these; over thunderbolt that may simply be what it does)\n");
}

int main(int argc, char **argv) {
  int watch = 0;
  double every = 1.0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-w") || !strcmp(argv[i], "--watch")) watch = 1;
    else if (!strcmp(argv[i], "-n") && i + 1 < argc) every = atof(argv[++i]);
    else { printf("usage: %s [-w|--watch] [-n seconds]\n", argv[0]); return 2; }
  }
  do {
    if (watch) printf("\033[H\033[2J");
    if (show_published(5.0)) show_direct();
    fflush(stdout);
    if (watch) { struct timespec t = {(time_t)every, (long)((every - (double)(time_t)every) * 1e9)}; nanosleep(&t, NULL); }
  } while (watch);
  return 0;
}
