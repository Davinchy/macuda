// runtime_unlock_harness.c — drives the runtime integration gate (tinynv_tensor_unlock_gate, the
// exact function device_boot calls) over the wire against mock_ad102_server, with the REAL pinned bundle.
//
// Proves CONTROL FLOW only (no GPU): the env gate, the unsupported-backend refusal, applied-and-verified,
// verified-existing, the contained wrong-device refusal, the recovery hold through the library path, and that the
// caller's signal dispositions are restored after the transaction window. Driven by test/test_runtime_unlock.py.
//
//   runtime_unlock_harness <socket>            open the tinygpu backend, call the gate, report, close
//   runtime_unlock_harness --wrong-backend     call the gate with a non-tinygpu backend (must refuse, no contact)
//
// Exit code = the gate's return code (0 proceed, 1 contained refusal, 3 confirmed recovery).
#include "tensor_unlock.h"
#include "tinynv.h"
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// The caller's own SIGUSR1 disposition, installed BEFORE the gate: after a successful transaction the gate must
// restore it (the normal boot's teardown handlers chain into whatever was here, never into the transaction's).
static volatile sig_atomic_t marker_usr1;
static void marker_handler(int sig) { (void)sig; marker_usr1 = 1; }

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc != 2) { fprintf(stderr, "usage: %s <socket>|--wrong-backend\n", argv[0]); return 2; }
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = marker_handler;
  sigaction(SIGUSR1, &sa, NULL);

  tinynv_pci_t pci;
  int wrong_backend = !strcmp(argv[1], "--wrong-backend");
  if (wrong_backend) {
    // A live backend that is NOT the validated TinyGPU transport: the gate must refuse before any device contact.
    memset(&pci, 0, sizeof(pci));
    pci.name = "replay";
    pci.live = 1;
  } else if (tinynv_pci_open_tinygpu(argv[1], &pci)) {
    fprintf(stderr, "harness: %s\n", tinynv_last_error());
    return 2;
  }
  int active = -1;
  int rc = tinynv_tensor_unlock_gate(&pci, &active);
  printf("gate rc=%d active=%d\n", rc, active);
  if (rc) printf("gate-error: %s\n", tinynv_last_error());
  if (rc == 0 && active == 1) {
    // The transaction window is over: the caller's SIGUSR1 handler must be back. If the gate left its own handler
    // installed, this raise would be swallowed (or pre-confirm a later hold) instead of setting the marker.
    raise(SIGUSR1);
    printf("sigusr1-restored %s\n", marker_usr1 ? "ok" : "FAIL");
  }
  if (!wrong_backend) pci.close(&pci);
  return rc;
}