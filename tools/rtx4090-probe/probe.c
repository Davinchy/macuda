// rtx4090-probe — the standalone, supervised AD102 tensor-unlock transaction.
//
// Experimental feature; see docs/driver/rtx4090-tensor-unlock/spec.md. This program is
// the ONLY live caller of the transaction in this card: nothing is wired into the normal macuda open path, and the
// installed TinyGPU app/server/dext are used unchanged.
//
// Protocol (README.md): read-only preflight -> exclusive lock -> ONE GPU process -> controlled release.
// Start from a power-cycled enclosure. Do NOT run this under an external timeout that kills it: if firmware may
// still DMA, this process must stay alive holding the socket and mappings until the enclosure is powered off.
//
//   rtx4090-probe preflight   read-only identity/cold-state snapshot (still takes the lock: the server has one slot)
//   rtx4090-probe unlock      the full transaction, with stage diagnostics and recovery hold on uncontained failure
//
// Exit codes: 0 success (applied-and-verified or verified-existing), 1 failure (contained), 2 usage/lock/setup,
// 3 failure that required recovery (returned only AFTER the owner confirms the enclosure power-off).
#include "tensor_unlock.h"
#include "tinynv.h"
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// Signal handling and the recovery hold live in libtinynv (tensor_unlock.c): the same tested code the
// runtime integration uses. Flag-only handlers; SIGUSR1 honored only inside the hold; no socket work in a handler.
static tensor_unlock_sigstate_t probe_sigs;

// ---- the project's gpu-lock.sh file protocol (one card, one lock; a dead holder's lock is stale) ----------------------

static char lock_path[1024];
static const char *lock_session = "rtx4090-probe";
static int lock_held;

// O_CREAT|O_EXCL, never truncate: two probes starting together must not both "acquire" by overwriting each other's
// line. A lock WITHOUT a pid is a hand-held session window (gpu-lock.sh protocol) and is never taken over; only a
// lock whose recorded pid is dead is stale.
static int lock_acquire(const char *what) {
  for (int attempt = 0; attempt < 2; attempt++) {
    int fd = open(lock_path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd >= 0) {
      FILE *f = fdopen(fd, "w");
      time_t now = time(NULL);
      char ts[64];
      strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S%z", localtime(&now));
      fprintf(f, "session=%s pid=%ld since=%s what=%s\n", lock_session, (long)getpid(), ts, what);
      fclose(f);
      lock_held = 1;
      fprintf(stderr, "rtx4090-probe: acquired %s (session=%s pid=%ld)\n", lock_path, lock_session, (long)getpid());
      return 0;
    }
    if (errno != EEXIST) {
      fprintf(stderr, "rtx4090-probe: cannot create lock %s: %s\n", lock_path, strerror(errno));
      return -1;
    }
    char line[512] = "";
    FILE *f = fopen(lock_path, "r");
    if (f) {
      if (!fgets(line, sizeof(line), f)) line[0] = 0;
      fclose(f);
    }
    long pid = 0;
    const char *p = strstr(line, "pid=");
    if (p) pid = strtol(p + 4, NULL, 10);
    if (pid <= 0) {
      fprintf(stderr, "rtx4090-probe: LOCKED (held by hand, no pid to check): %s", line);
      return -1;
    }
    if (kill((pid_t)pid, 0) == 0) {
      fprintf(stderr, "rtx4090-probe: LOCKED (holder pid %ld is alive): %s", pid, line);
      return -1;
    }
    fprintf(stderr, "rtx4090-probe: stale lock (holder pid %ld is dead), taking it over: %s", pid, line);
    unlink(lock_path); // then retry the exclusive create; a concurrent winner makes us re-read and refuse
  }
  fprintf(stderr, "rtx4090-probe: lock %s is contended; not acquiring\n", lock_path);
  return -1;
}

static void lock_release(void) {
  if (!lock_held) return;
  FILE *f = fopen(lock_path, "r");
  char line[512] = "";
  if (f) {
    if (!fgets(line, sizeof(line), f)) line[0] = 0;
    fclose(f);
  }
  char want[256];
  snprintf(want, sizeof(want), "session=%s ", lock_session);
  if (strncmp(line, want, strlen(want))) {
    fprintf(stderr, "rtx4090-probe: not releasing a lock that is not ours: %s", line);
    return;
  }
  unlink(lock_path);
  lock_held = 0;
  fprintf(stderr, "rtx4090-probe: released %s\n", lock_path);
}

// ---- recovery hold -----------------------------------------------------------------------------------------------------

// The containment cannot be proven: firmware may still DMA into the transaction buffers. The installed transport
// force-unmaps everything on disconnect (transport-audit O14), so this process keeps the socket, the mappings and
// the lock alive, issues no further GPU work, and waits for the owner to power the enclosure off. SIGKILL/crash/
// server death defeat this — that is the documented crash limitation, not a guarantee this program can make.
// The hold itself is tensor_unlock_recovery_hold (shared with the runtime integration; tested end-to-end by
// test_probe_recovery.py through this binary).

// ---- reporting -----------------------------------------------------------------------------------------------------------

static void report(const tensor_unlock_result_t *r, int preflight) {
  // A preflight applied NOTHING: saying "applied-and-verified" for a read-only snapshot would be a false claim.
  printf("result: %s\n", r->error ? "FAILED" : preflight ? "preflight-ok (read-only)"
                       : r->verified_existing ? "verified-existing" : "applied-and-verified");
  printf("classification: %s\n", r->classification ? r->classification : "-");
  printf("identity: %04x:%04x  chip: %#x\n", r->identity & 0xffff, r->identity >> 16, r->chip);
  printf("last-ok-stage: %s  failed-at: %s  elapsed: %.2f s\n", tensor_unlock_stage_name(r->last_ok),
         tensor_unlock_stage_name(r->failed_at), r->elapsed_s);
  if (r->first_error[0]) printf("first-error: %s\n", r->first_error);
  printf("cold: feat=%#x ss0=%#x ss1=%#x\n", r->feat_entry, r->ss0_entry, r->ss1_entry);
  static const char *phases[TU_READOUT_PHASES] = {"before-firmware", "after-payload", "selectors-applied", "pre-FLR", "post-FLR"};
  for (int i = 0; i < TU_READOUT_PHASES; i++)
    if (r->readout_valid[i] || r->readout[i])
      printf("readout[%s]: %#x (masked %#x)%s\n", phases[i], r->readout[i], r->readout[i] & 0x07ffffffu,
             r->readout_valid[i] ? "" : " INVALID");
  if (r->feat_final || r->ss0_final || r->ss1_final)
    printf("final: feat=%#x ss0=%#x ss1=%#x\n", r->feat_final, r->ss0_final, r->ss1_final);
  for (int i = 0; i < r->nbuffer; i++)
    printf("buffer[%s]: iova=%#llx size=%llu\n", r->buffers[i].name, (unsigned long long)r->buffers[i].iova,
           (unsigned long long)r->buffers[i].size);
  printf("dma: allocated=%d freed=%d  recovery-required=%d\n", r->ndma_alloc, r->ndma_freed, r->recovery_required);
  fflush(stdout);
}

static void usage(void) {
  fprintf(stderr,
          "usage: rtx4090-probe <preflight|unlock> [options]\n"
          "  --socket PATH     TinyGPU socket (default $TINYNV_SOCKET or ${TMPDIR}tinygpu.sock)\n"
          "  --bundle DIR      prepared unlock bundle (default $TINYNV_UNLOCK_BUNDLE or the build-time path)\n"
          "  --session NAME    gpu-lock session name (default rtx4090-probe)\n"
          "  --lock-file PATH  lock file (default $GPU_LOCK, $MACUDA_SHARED_ROOT/.gpu-lock or ./.gpu-lock)\n"
          "Run from the repository root, after tools/preflight.sh, on a power-cycled enclosure.\n"
          "Never run under a killing timeout; see README.md.\n");
}

int main(int argc, char **argv) {
  if (argc < 2) { usage(); return 2; }
  const char *cmd = argv[1];
  if (strcmp(cmd, "preflight") && strcmp(cmd, "unlock")) { usage(); return 2; }
  const char *sock = getenv("TINYNV_SOCKET");
  char default_sock[512];
  if (!sock || !*sock) {
    const char *tmp = getenv("TMPDIR");
    snprintf(default_sock, sizeof(default_sock), "%stinygpu.sock", tmp && *tmp ? tmp : "/tmp/");
    sock = default_sock;
  }
  const char *bundle = tensor_unlock_bundle_dir();
  const char *lf = getenv("GPU_LOCK");
  char shared_lock[1024];
  if (!lf || !*lf) {
    const char *shared = getenv("MACUDA_SHARED_ROOT");
    if (shared && *shared) {
      snprintf(shared_lock, sizeof(shared_lock), "%s/.gpu-lock", shared);
      lf = shared_lock;
    } else lf = ".gpu-lock";
  }
  snprintf(lock_path, sizeof(lock_path), "%s", lf);
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--socket") && i + 1 < argc) sock = argv[++i];
    else if (!strcmp(argv[i], "--bundle") && i + 1 < argc) bundle = argv[++i];
    else if (!strcmp(argv[i], "--session") && i + 1 < argc) lock_session = argv[++i];
    else if (!strcmp(argv[i], "--lock-file") && i + 1 < argc) snprintf(lock_path, sizeof(lock_path), "%s", argv[++i]);
    else { usage(); return 2; }
  }
  fprintf(stderr, "rtx4090-probe: build %s, command %s, socket %s\n", tinynv_build_id(), cmd, sock);
  if (!strcmp(cmd, "unlock")) fprintf(stderr, "rtx4090-probe: bundle %s\n", bundle);

  tensor_unlock_signals_install(&probe_sigs);
  if (lock_acquire(cmd)) return 2;

  tinynv_pci_t pci;
  if (tinynv_pci_open_tinygpu(sock, &pci)) {
    fprintf(stderr, "rtx4090-probe: %s\n", tinynv_last_error());
    lock_release();
    return 2;
  }
  tensor_unlock_result_t r;
  int rc;
  if (!strcmp(cmd, "preflight")) {
    rc = tensor_unlock_preflight(&pci, &r);
    report(&r, 1);
    pci.close(&pci);
    lock_release();
    return rc ? 1 : 0;
  }
  rc = tensor_unlock_run(&pci, bundle, &r);
  report(&r, 0);
  if (r.recovery_required) {
    // Keep everything alive: no pci.close, no lock release, no exit until the owner confirms the power-off.
    int code = tensor_unlock_recovery_hold(&r, "rtx4090-probe");
    pci.close(&pci);
    lock_release();
    return code;
  }
  pci.close(&pci);
  lock_release();
  return rc ? 1 : 0;
}
