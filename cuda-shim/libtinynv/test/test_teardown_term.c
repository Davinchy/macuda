// test_teardown_term.c - the teardown-under-signal witness, offline (2026-09-25, the day it cost a hard kill).
//
// THE SHAPE IT GUARDS: a card run whose exit-path drain never completes (the field case: the ring read-back stopped
// acking after all work finished), and then one SIGTERM. Before the teardown series (EGPU d49105d..b896eb7) that
// shape was unkillable: the booted flag cleared only AFTER the drain, so the handler re-entered the same wait and
// nested there, and each further caught signal stacked another spin. After the series, ONE SIGTERM must kill it.
//
// D's three conditions (2026-09-25 14:53, relayed by A), each asserted here because each one closes a hole the field
// shape actually exercised:
//   1. FIRST SIGTERM, COUNTED - this parent sends exactly one, by construction, and says so.
//   2. DEATH BY THE SIGNAL - waitpid must read WIFSIGNALED && WTERMSIG == SIGTERM (shell 143), never a clean exit,
//      because with a real drain an it-exited check could pass on the drain simply finishing.
//   3. THE HANDLER SHOWN TO RETURN - the driver prints a line between put_the_card_down returning and the re-raise;
//      that line must appear AFTER the drain-entered marker and BEFORE death, or the witness proved killability, not
//      the guard.
//
// SEEN TO FAIL: the mutant arm (TEARDOWN_MUTANT=1 in the environment) recreates the PRE-FIX ordering - the stub
// drain re-arms the booted flag, exactly what clear-after-drain meant - and the same one SIGTERM must then NOT kill
// the child within the window. A witness that cannot reproduce the wedge would pass on any tree.
//
// MECHANICS: this file compiles ../src/tinynv.c directly, because put_the_card_down, quiesce_and_continue and
// catch_the_ways_out are static by design and the witness needs the REAL functions, not a re-derivation. Every
// driver layer they can reach is stubbed below: the drain stub prints its marker and spins the way the real one
// spun in the field, and everything the teardown must never touch stops the test by name. No card, no socket, no
// trace. The child fakes exactly the two flags device_boot would have set, then leaves through exit(0) the way
// splitk_card did.
#include "../src/tinynv.c"

#include <sys/wait.h>
#include <poll.h>
#include <unistd.h>

static int checks, tfails;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); tfails++; } } while (0)

// ---- stubs: the teardown's reachable world -------------------------------------------------------------------
static int g_mutant;
int tinynv_exec_idle(tinynv_exec_t *ex) {
  (void)ex;
  fprintf(stderr, "WITNESS: drain entered, and it will never complete\n");   // stderr: unbuffered, the parent's cue
  if (g_mutant) g_dev.booted = 1;   // the PRE-FIX ordering, restated as state: the flag is still up while draining
  for (;;) pause();                 // the field shape: a wait that nothing will ever satisfy
}
void tinynv_exec_fini(tinynv_exec_t *ex) { (void)ex; fprintf(stderr, "WITNESS: exec_fini\n"); }
void tinynv_gpu_close(tinynv_gpu_t *g) { (void)g; fprintf(stderr, "WITNESS: gpu_close (mastering would drop here)\n"); }
void tinynv_dev_quiesce(tinynv_dev_t *d) { (void)d; fprintf(stderr, "WITNESS: dev_quiesce\n"); }
static void __attribute__((noreturn)) unreachable(const char *f) { fprintf(stderr, "WITNESS FAIL: teardown reached %s\n", f); _exit(97); }
#define STOP(name, ret, ...) ret name(__VA_ARGS__) { unreachable(#name); }
STOP(tinynv_exec_init, int, tinynv_gpu_t *g, tinynv_exec_t *e)
STOP(tinynv_gpu_open, int, tinynv_gpu_t *g, tinynv_pci_t *p)
STOP(tinynv_gpu_init_sw, int, tinynv_gpu_t *g)
STOP(tinynv_gpu_init_hw, int, tinynv_gpu_t *g)
// every other undefined symbol tinynv.c carries belongs to paths the child never takes (launch, upload, cubin,
// sensors, gsp setup); a stop for each keeps the link honest and the first wrong turn loud
STOP(tinynv_exec_copy, int, tinynv_exec_t *e, uint64_t d, uint64_t s, uint64_t n)
STOP(tinynv_exec_download, int, tinynv_exec_t *e, void *d, uint64_t s, size_t n)
STOP(tinynv_exec_upload, int, tinynv_exec_t *e, uint64_t d, const void *s, size_t n)
STOP(tinynv_exec_zero, int, tinynv_exec_t *e, uint64_t va, uint64_t n)
STOP(tinynv_exec_flush, int, tinynv_exec_t *e)
STOP(tinynv_exec_wait, int, tinynv_exec_t *e, uint64_t v, double s)
STOP(tinynv_exec_reached, int, tinynv_exec_t *e, uint64_t v)
STOP(tinynv_exec_needs_wait, int, const tinynv_exec_t *e)
STOP(tinynv_exec_run, int, tinynv_exec_t *e, tinynv_exec_module_t *m, const tinynv_kernel_desc_t *k, const uint32_t grid[3], const uint32_t block[3], uint32_t sm, const void *p, size_t n)
STOP(tinynv_exec_load, int, tinynv_exec_t *e, const tinynv_cubin_t *c, const char *kn, tinynv_exec_module_t *m)
STOP(tinynv_exec_unload, void, tinynv_mm_t *mm, tinynv_exec_module_t *m)
STOP(tinynv_exec_set_internal, void, tinynv_exec_t *e, int on)
STOP(tinynv_exec_stage_va, uint64_t, const tinynv_exec_t *e)
STOP(tinynv_exec_stage_host, void *, const tinynv_exec_t *e)
STOP(tinynv_exec_stage_bytes, size_t, void)
STOP(tinynv_exec_keepalive_arm, int, tinynv_exec_t *e, uint64_t *flag_va)
STOP(tinynv_exec_keepalive_release, void, tinynv_exec_t *e, int by_wait)
STOP(tinynv_exec_graph_begin, int, tinynv_exec_t *e)
STOP(tinynv_exec_graph_end, int, tinynv_exec_t *e, tinynv_graph_rec_t **out)
STOP(tinynv_exec_graph_launch, int, tinynv_exec_t *e, tinynv_graph_rec_t *r)
STOP(tinynv_exec_graph_free, void, tinynv_exec_t *e, tinynv_graph_rec_t *r)
STOP(tinynv_cubin_parse, int, const void *b, size_t n, tinynv_cubin_t *c)
STOP(tinynv_cubin_free, void, tinynv_cubin_t *c)
STOP(tinynv_cubin_kernel, const tinynv_kernel_desc_t *, const tinynv_cubin_t *c, const char *n)
STOP(tinynv_cubin_global, int, const tinynv_cubin_t *c, const char *n, uint64_t *off, uint64_t *size)
STOP(tinynv_cubin_image, int, const tinynv_cubin_t *c, const char *kernel, int with_bytes, tinynv_image_t *im)
STOP(tinynv_image_free, void, tinynv_image_t *im)
STOP(tinynv_kernel_thread_limit, unsigned, const tinynv_kernel_desc_t *d)
STOP(tinynv_mm_alloc_buffer, int, tinynv_mm_t *mm, uint64_t size, int host, int cpu_access, int uncached, int force_devmem, int zero, tinynv_vmap_t *out)
STOP(tinynv_mm_check_fw_carveout, int, const tinynv_mm_t *mm, uint64_t lo, uint64_t hi, char *l, size_t n)
STOP(tinynv_vmap_free, void, tinynv_mm_t *mm, tinynv_vmap_t *v)
STOP(tinynv_pci_open_tinygpu, int, const char *s, tinynv_pci_t *out)
STOP(tinynv_rd32, uint32_t, tinynv_dev_t *d, uint64_t off)
STOP(tinynv_gsp_init_objects, int, tinynv_gpu_t *g)
STOP(tinynv_gsp_init_channel, int, tinynv_gpu_t *g)
STOP(tinynv_gsp_init_gr_context, int, tinynv_gpu_t *g)
STOP(tinynv_gsp_init_queues, int, tinynv_gpu_t *g)
STOP(tinynv_gsp_open_client, int, tinynv_gpu_t *g)
STOP(tinynv_gsp_free_heap, int, tinynv_gpu_t *g, uint64_t *out)
STOP(tinynv_gsp_sensors_init, int, tinynv_gpu_t *g, uint64_t poll_mask, uint32_t freq_ms)
STOP(tinynv_gsp_sensors_read, int, tinynv_gpu_t *g, tinynv_rusd_t *out)
STOP(tinynv_gsp_probe_phys_attr, int, tinynv_gpu_t *g, uint64_t size, uint64_t offset, uint64_t *paddr, uint64_t *contig, uint32_t *aperture)
STOP(tinynv_tlsf_used, uint64_t, const tinynv_tlsf_t *a)
STOP(tinynv_tlsf_size, uint64_t, const tinynv_tlsf_t *a)

// ---- the witness ----------------------------------------------------------------------------------------------
static pid_t child_leaves_through_exit(int errfd) {
  pid_t pid = fork();
  if (pid) return pid;
  dup2(errfd, 2);
  // exactly what device_boot leaves behind, and nothing else: the two flags and the two registrations
  g_dev.booted = 1;
  g_dev.exec_ready = 1;
  atexit(put_the_card_down);
  catch_the_ways_out();
  exit(0);   // the clean way out; the drain hangs inside it, as it did in the field
}

// read the child's stderr until `want` appears or `ms` passes; every line read is echoed for the log
static int await_line(int fd, const char *want, int ms, int *saw_order_line) {
  static char buf[4096]; static size_t have;
  for (;;) {
    char *nl;
    while ((nl = memchr(buf, '\n', have))) {
      *nl = 0;
      printf("    child: %s\n", buf);
      int hit = strstr(buf, want) != NULL;
      if (saw_order_line && strstr(buf, "restoring the default disposition and re-raising")) *saw_order_line = 1;
      size_t rest = have - (size_t)(nl + 1 - buf);
      memmove(buf, nl + 1, rest); have = rest;
      if (hit) return 0;
    }
    struct pollfd p = { .fd = fd, .events = POLLIN };
    if (poll(&p, 1, ms) <= 0) return -1;
    ssize_t n = read(fd, buf + have, sizeof buf - have - 1);
    if (n <= 0) return -1;
    have += (size_t)n;
  }
}

int main(void) {
  int pipefd[2];
  g_mutant = getenv("TEARDOWN_MUTANT") && *getenv("TEARDOWN_MUTANT") == '1';
  printf("teardown-term witness, %s arm\n", g_mutant ? "MUTANT (pre-fix ordering, must NOT die)" : "fixed (must die of the first SIGTERM)");
  if (pipe(pipefd)) { printf("  FAIL: pipe\n"); return 1; }
  pid_t pid = child_leaves_through_exit(pipefd[1]);
  close(pipefd[1]);

  CHECK(await_line(pipefd[0], "drain entered", 5000, NULL) == 0, "the child never reached the drain");
  usleep(100000);   // let it settle INTO the spin, so the signal lands mid-wait as it did in the field
  printf("  one SIGTERM sent - the first and only (D's condition 1)\n");
  kill(pid, SIGTERM);

  if (!g_mutant) {
    int saw_reraise = 0;
    await_line(pipefd[0], "\xff never matches: drain the pipe until it closes", 3000, &saw_reraise);
    int st = 0;
    CHECK(waitpid(pid, &st, 0) == pid, "waitpid");
    CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGTERM, "death must be BY SIGTERM (D's condition 2); status %#x", st);
    CHECK(saw_reraise, "the handler's completion line must precede death (D's condition 3)");
  } else {
    int st = 0;
    struct timespec t = {2, 0}; nanosleep(&t, NULL);
    CHECK(waitpid(pid, &st, WNOHANG) == 0, "the mutant must survive the SIGTERM (the wedge reproduced); status %#x", st);
    kill(pid, SIGKILL);   // a plain test process with no card behind it; the arm's point has been made
    waitpid(pid, &st, 0);
  }
  close(pipefd[0]);
  printf("teardown-term witness: %d of %d checks passed%s\n", checks - tfails, checks, tfails ? "" : " - all");
  return tfails;
}
