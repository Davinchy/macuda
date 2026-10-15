// The C driver against a real GPU: boot the chip, say how far it got, and put it down properly whatever happens.
//
// Nothing else in this suite touches hardware, and this does not either unless it is asked twice - a socket to drive and
// TINYNV_HW=1 - so `make test` can never reach a card by accident. That is deliberate: every other test can be run by
// anyone at any time, and this one cannot.
//
// PUTTING THE CARD DOWN IS THE POINT OF THIS FILE, more than booting it. A driver that leaves through an error path with
// bus mastering still on leaves the GPU writing into mappings that are being torn down underneath it, and macOS latches
// that on the device as a fault flag which no reset clears - only re-enumerating the device does, which means someone
// walking over and unplugging the box. It cost a session on 2026-09-14 and was not a driver bug: the python oracle raised
// past its own teardown. So there is exactly one way out of this program, every failure goes through it, and a signal
// goes through it too.
#include "tinynv.h"
#include "gpu.h"
#include "internal.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

// The card, reachable from the signal handler. There is one of it and one thread, which is what makes this safe enough.
static tinynv_gpu_t g;
static volatile sig_atomic_t opened, quiesced;

// Strictly, writing to a socket from a signal handler is not async-signal-safe. It is done anyway, and the reasoning is
// worth writing down rather than leaving as an oversight: the alternative is returning to the shell with the GPU still
// bus mastering into mappings the kernel is about to reclaim, which is the exact state this file exists to avoid, and
// which takes a human and a cable to undo. A rare deadlock in a handler is recoverable; that is not.
static void on_signal(int sig) {
  if (opened && !quiesced) {
    quiesced = 1;
    tinynv_dev_quiesce(&g.dev);
    const char msg[] = "\ninterrupted: bus mastering cleared before exit\n";
    ssize_t ignored = write(2, msg, sizeof(msg) - 1);
    (void)ignored;
  }
  _exit(130);
}

int main(int argc, char **argv) {
  const char *sock = argc > 1 ? argv[1] : getenv("TINYNV_SOCKET");
  const int mock = getenv("TINYNV_HW_MOCK") != NULL;
  if (!sock || !(getenv("TINYNV_HW") || mock)) {
    printf("usage: TINYNV_HW=1 %s <tinygpu socket>\n", argv[0]);
    printf("  this is the only test that drives a real GPU, so it needs both the socket and TINYNV_HW=1.\n");
    printf("  take the gpu lock first: tools/gpu-lock.sh acquire <A|B> \"what\"\n");
    return 2;
  }

  printf("libtinynv build %s\n", tinynv_build_id());
  struct sigaction sa = {.sa_handler = on_signal};
  sigemptyset(&sa.sa_mask);
  for (int s = 0; s < 2; s++) sigaction(s ? SIGTERM : SIGINT, &sa, NULL);

  tinynv_pci_t pci;
  if (tinynv_pci_open_tinygpu(sock, &pci)) {
    printf("cannot reach the gpu on %s: %s\n", sock, tinynv_last_error());
    return 1;
  }

  // What the command register says before anything is driven. The dext leaves bus mastering on at enumeration, so this
  // normally reads 0x0007, and recording it is what lets the check at the end mean something: an assertion that the bit
  // is clear proves nothing on a card that arrived with it clear already.
  uint32_t cmd_before = pci.cfg_read(&pci, 0x04, 2);
  printf("command register on arrival: %#06x (bus mastering %s)\n", cmd_before, cmd_before & 0x4 ? "on" : "already off");

  // From here every exit goes through `done`. There is no early return in what follows, and that is not a style choice.
  int rc = tinynv_gpu_open(&g, &pci);
  opened = 1;
  if (rc) { printf("bring-up failed: %s\n", tinynv_last_error()); goto done; }

  printf("chip %s (arch %#x impl %u, boot0 %#x), %s boot, mmu v%d%s\n", g.dev.chip_name, g.dev.architecture,
         g.dev.implementation, g.dev.chip_id, g.dev.fmc_boot ? "chain of trust" : "vbios", g.dev.mmu_ver,
         g.dev.wpr2_was_up ? ", chip was reset on arrival" : "");
  printf("vram %llu MB, bar1 window %llu MB (%s bar)\n", (unsigned long long)(g.dev.vram_size >> 20),
         (unsigned long long)(g.dev.vram.size >> 20), g.dev.large_bar ? "large" : "small");

  if ((rc = tinynv_gpu_init_sw(&g))) { printf("software init failed: %s\n", tinynv_last_error()); goto done; }
  printf("allocations placed and firmware loaded\n");

  if ((rc = tinynv_gpu_init_hw(&g))) { printf("hardware init failed: %s\n", tinynv_last_error()); goto done; }
  printf("GSP-RM is up\n");

done:
  // Against the mock server there is no chip to boot and the bring-up is expected to fail; what is being tested there is
  // everything below. That is how this file's one real check gets run by anyone, on any machine, with no GPU.
  if (!mock) CHECK(!rc, "the driver did not reach the end of the boot");

  // Twice is harmless - quiesce reads the command register, clears one bit and reads it back - and missing it is not.
  if (opened && !quiesced) { quiesced = 1; tinynv_gpu_close(&g); }

  // The check this file exists for, read back from the device rather than assumed from having issued the write.
  uint32_t cmd = pci.cfg_read(&pci, 0x04, 2);
  if (cmd == 0xffffffffu) CHECK(0, "the device is no longer answering on the bus; it cannot be confirmed quiesced");
  else CHECK(!(cmd & 0x4), "the device was left bus mastering: command register %#06x", cmd);
  if (!fails)
    printf(cmd_before & 0x4 ? "bus mastering %#06x -> %#06x: the bit was set on arrival and this run cleared it\n"
                            : "bus mastering off, %#06x -> %#06x - but it was already off, so this run proved nothing\n",
           cmd_before, cmd);
  CHECK(cmd_before & 0x4, "the card arrived with bus mastering already off, so the check at the end is not evidence");

  if (pci.close) pci.close(&pci);
  printf(fails ? "%d checks failed; the card was put down regardless\n" : "all checks passed; card quiesced\n", fails);
  return fails ? 1 : 0;
}
