// The TinyGPU socket's receive ceiling, poisoning and EINTR behavior , against a minimal mock.
//
// Mode "wedge" (test_socket_timeout.py): the mock answers the first config read and then wedges. With
// TINYNV_SOCKET_TIMEOUT_S=1:
//   read 1: the identity, promptly                       — the normal round trip still works
//   read 2: all ones after ~1 s                          — a wedged server is a bounded failure, not a forever recv
//   read 3: all ones immediately                         — the desynchronized connection is poisoned, never reused
//
// Mode "eintr": the mock delays its SECOND reply by ~1.2 s while a SIGALRM (handler installed WITHOUT SA_RESTART)
// fires mid-recv. A catchable signal must NOT poison a healthy connection: no byte was consumed, the transfer is
// retried, and the recovery unwind still has a working socket for its best-effort quiesce.
//   read 1: the identity, promptly
//   read 2: the identity after ~1.2 s despite the signal  — EINTR retried, connection intact
//   read 3: the identity, promptly                        — never poisoned
// No GPU, no TinyGPU.app, no tinygrad checkout needed.
#include "tinynv_pci.h"
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static volatile sig_atomic_t got_alarm;
static void on_alarm(int sig) { (void)sig; got_alarm = 1; } // flag only, like the probe's handlers

static int wedge_mode(tinynv_pci_t *pci) {
  double t0 = now_s();
  uint32_t v1 = pci->cfg_read(pci, 0, 4);
  double t1 = now_s();
  uint32_t v2 = pci->cfg_read(pci, 0, 4);
  double t2 = now_s();
  uint32_t v3 = pci->cfg_read(pci, 0, 4);
  double t3 = now_s();
  int fails = 0;
  if (v1 != 0x268410de || t1 - t0 > 0.5) {
    printf("  FAIL: first read %#x in %.3f s (want 268410de, prompt)\n", v1, t1 - t0);
    fails++;
  }
  if (v2 != 0xffffffff || t2 - t1 < 0.8 || t2 - t1 > 3.0) {
    printf("  FAIL: wedged read %#x in %.3f s (want all ones after ~1 s)\n", v2, t2 - t1);
    fails++;
  }
  if (v3 != 0xffffffff || t3 - t2 > 0.3) {
    printf("  FAIL: poisoned read %#x in %.3f s (want immediate all ones)\n", v3, t3 - t2);
    fails++;
  }
  if (!fails) printf("socket timeout: all checks passed (read2 %.3f s, read3 %.4f s)\n", t2 - t1, t3 - t2);
  return fails;
}

static int eintr_mode(tinynv_pci_t *pci) {
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = on_alarm; // no SA_RESTART: recv must return EINTR, exactly like the probe under SIGTERM
  sigaction(SIGALRM, &sa, NULL);
  struct itimerval it = {.it_value = {.tv_sec = 0, .tv_usec = 300000}};
  setitimer(ITIMER_REAL, &it, NULL);
  double t0 = now_s();
  uint32_t v1 = pci->cfg_read(pci, 0, 4);
  double t1 = now_s();
  uint32_t v2 = pci->cfg_read(pci, 0, 4); // the server stalls this reply; the alarm fires inside recv
  double t2 = now_s();
  uint32_t v3 = pci->cfg_read(pci, 0, 4);
  double t3 = now_s();
  int fails = 0;
  if (!got_alarm) { printf("  FAIL: the alarm never fired; the test proved nothing\n"); fails++; }
  if (v1 != 0x268410de || t1 - t0 > 0.5) {
    printf("  FAIL: first read %#x in %.3f s (want 268410de, prompt)\n", v1, t1 - t0);
    fails++;
  }
  if (v2 != 0x268410de || t2 - t1 < 0.8 || t2 - t1 > 3.0) {
    printf("  FAIL: interrupted read %#x in %.3f s (want the identity after ~1.2 s: EINTR retried, not poisoned)\n",
           v2, t2 - t1);
    fails++;
  }
  if (v3 != 0x268410de || t3 - t2 > 0.3) {
    printf("  FAIL: post-signal read %#x in %.3f s (want the identity, prompt: the connection must stay usable)\n",
           v3, t3 - t2);
    fails++;
  }
  if (!fails) printf("socket eintr: all checks passed (read2 %.3f s with a signal mid-recv, read3 %.4f s)\n",
                     t2 - t1, t3 - t2);
  return fails;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: test_socket_timeout <socket> [wedge|eintr]\n"); return 2; }
  const char *mode = argc > 2 ? argv[2] : "wedge";
  tinynv_pci_t pci;
  if (tinynv_pci_open_tinygpu(argv[1], &pci)) {
    fprintf(stderr, "  FAIL: open: %s\n", tinynv_last_error());
    return 1;
  }
  int fails = !strcmp(mode, "eintr") ? eintr_mode(&pci) : wedge_mode(&pci);
  pci.close(&pci);
  if (fails) printf("%d checks failed\n", fails);
  return fails ? 1 : 0;
}