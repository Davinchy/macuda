// Does the card come down when the process aborts?
//
// ggml's CUDA_CHECK calls ggml_abort, which calls abort(), which raises SIGABRT and never runs an atexit handler. Until
// the library caught that, every failed check inside llama.cpp left the GPU bus mastering into mappings about to be
// reclaimed, and macOS latched a fault flag that only re-enumerating the device clears: one person and one cable, per
// failure. Session A found it the first time a real op suite hit a real bug.
//
// A child boots the card and aborts the way ggml does. The parent then opens the device WITHOUT booting it and reads
// the command register, which must read bus mastering off.
//
// The state before proves nothing either way and asking about it was a mistake: a previous test that exited cleanly
// leaves the card quiesced, so "it was on before" is not a precondition this can rely on. What matters is that the
// child's boot turns bus mastering on - bringing the device up is what enables it - and that the handler turns it back
// off. So the child leaves a marker once it has booted, and the parent refuses to conclude anything without it.
#include "tinynv.h"
#include "tinynv_pci.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
  const char *sock = getenv("TINYNV_SOCKET");
  if (!sock || !getenv("TINYNV_HW")) {
    printf("usage: TINYNV_HW=1 TINYNV_SOCKET=<socket> %s\n", argv[0]);
    return 2;
  }

  printf("libtinynv build %s\n", tinynv_build_id());
  tinynv_pci_t pci;
  if (tinynv_pci_open_tinygpu(sock, &pci)) { printf("  cannot reach the gpu: %s\n", tinynv_last_error()); return 1; }
  uint32_t before = pci.cfg_read(&pci, 0x04, 2);
  pci.close(&pci);
  printf("command register before: %#06x (bus mastering %s)\n", before, before & 0x4 ? "on" : "off");

  const char *marker = "/tmp/tinynv-abort-booted";
  unlink(marker);
  pid_t child = fork();
  if (child == 0) {
    tinynv_device_t d;
    tinynv_device_props_t props;
    if (tinynv_init() != TINYNV_OK || tinynv_device_get(&d, 0) != TINYNV_OK) _exit(3);
    if (tinynv_device_props(d, &props) != TINYNV_OK) _exit(4);   // boots the card, which enables bus mastering
    FILE *m = fopen(marker, "w");                                // say so, so the parent knows what it is measuring
    if (m) { fputs("booted", m); fclose(m); }
    abort();                                                     // exactly how ggml_abort leaves
  }
  int status = 0;
  waitpid(child, &status, 0);
  printf("the child booted the card and aborted: %s\n",
         WIFSIGNALED(status) ? "killed by a signal, as an abort should be" : "exited without a signal, which is wrong");

  if (tinynv_pci_open_tinygpu(sock, &pci)) { printf("  cannot reach the gpu after the abort: %s\n", tinynv_last_error()); return 1; }
  uint32_t after = pci.cfg_read(&pci, 0x04, 2);
  pci.close(&pci);
  printf("command register after:  %#06x\n", after);

  FILE *m = fopen(marker, "r");
  int booted = m != NULL;
  if (m) fclose(m);
  unlink(marker);
  if (!booted) {
    printf("  the child never reported booting, so bus mastering was never turned on and this proves nothing\n");
    return 2;
  }
  int ok = WIFSIGNALED(status) && !(after & 0x4) && after != 0xffffffffu;
  printf(ok ? "the child booted the card (which turns bus mastering on) and aborted; it reads %#06x afterwards, so the\n"
              "handler put it down and the abort still happened\n"
            : "  FAIL: the card was left bus mastering after an abort\n", after);
  return ok ? 0 : 1;
}
