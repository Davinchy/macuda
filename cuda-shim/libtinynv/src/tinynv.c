// The driver interface libtinycudart calls (cuda-shim/include/tinynv.h).
//
// Two devices live behind this interface. A real one, opened through a PCI backend, which boots the card on first use and
// runs everything on it; and a null one, host memory and reported launches, which exists so the whole shim above can be
// exercised on any machine with no GPU. The null device names itself in tinynv_device_props and reports compute
// capability 0, and naming a device that cannot be opened is an error rather than a quiet fall back to it - a shim that
// thinks it has a GPU and does not gets right-looking answers out of whatever was in host memory.
//
// Every call here is synchronous on a real device: a batch is submitted and waited on before the call returns. That
// throws away the point of a command queue and it is where this starts, because the first question about a driver that
// has never run a kernel is whether the kernel ran.
#include "tinynv.h"
#include "cubin.h"
#include "exec.h"
#include "qmd.h"
#include "internal.h"
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdatomic.h>
#include <string.h>

// What a device pointer refers to on a real GPU. A virtual address on its own cannot be freed - the allocation behind it
// has a physical side and a mapping - so the allocations are kept. There are few of them and they are large: ggml asks
// for a handful of pools and suballocates inside them, so a linear scan here is not the cost it looks like.
typedef struct { uint64_t va; tinynv_vmap_t map; } tinynv_alloc_t;

struct tinynv_device {
  tinynv_pci_t pci;
  int has_pci;      // false: the null device, host memory and reported launches
  int announced;

  // the gpu itself, brought up the first time something needs it rather than at init: opening the device should not
  // boot a card, and a shim that only asks what device is present should not pay for one
  int booted;
  // Set once the card has been put down. Nothing may touch it after that, and the calls that would are not errors: a
  // cudart shim unregisters its fat binaries from its own atexit handler, and the order two atexit handlers run in is
  // not something either of them chooses. So freeing after teardown does the host-side half and stops.
  int torn_down;
  // A caller's kernel for serving downloads from the compute engine, and how to size its grid. See
  // tinynv_set_download_kernel.
  tinynv_kernel_t download_kernel;
  unsigned download_block, download_bpt;
  int download_via_compute;
  // A caller's keep-alive kernel, launched after every synchronisation under TINYNV_KEEPALIVE=1 (exec.c owns the knob).
  tinynv_kernel_t keepalive_kernel;
  tinynv_gpu_t gpu;
  tinynv_exec_t exec;

  tinynv_alloc_t *allocs;
  int nallocs, cap_allocs;
};

struct tinynv_module {
  tinynv_device_t dev;
  void *image;      // our copy: the caller's fatbin may not outlive the module
  size_t len;
  tinynv_cubin_t cubin;
  // The cubin's code and constants in device memory. One upload for the whole module: the layout is a property of the
  // cubin, not of a kernel, so every kernel in it shares these bytes and differs only in where its own code sits.
  tinynv_exec_module_t loaded;
  int is_loaded;
};

struct tinynv_kernel {
  tinynv_module_t mod;
  const tinynv_kernel_desc_t *desc;
  tinynv_image_t layout;   // where this kernel's code and its constant banks sit inside the module's image
  int has_layout;
};

struct tinynv_stream { tinynv_device_t dev; uint64_t submitted; };
struct tinynv_event { tinynv_device_t dev; uint64_t value; };

static struct tinynv_device g_dev;
static int g_inited;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void announce(tinynv_device_t d) {
  if (d->has_pci || d->announced) return;
  d->announced = 1;
  fprintf(stderr, "tinynv: no gpu opened, running as a null device: memory is host memory and launches are reported, not executed\n");
}

// Bring the GPU up. Every stage the recorded boot covers, in the order it covers them, and then the execution context:
// the semaphore, the command buffers and the staging memory that everything below needs.
//
// Called from whichever entry point first needs the card rather than from init, and it holds the same lock, so two
// threads racing to be first get one boot between them.
// Put the card down when the process ends, however it ends.
//
// This is not tidiness. A process that exits with the GPU still bus mastering leaves it writing into mappings the kernel
// is about to reclaim; macOS latches that on the device as a fault flag that no reset clears, only re-enumerating the
// device does, and that means someone walking over to the box with a cable. It cost a session on 2026-09-14. The shim
// above this line is a library inside somebody else's program - llama.cpp - and cannot be relied on to call a teardown,
// so the teardown registers itself the moment there is a card to put down.
static void put_the_card_down(void) {
  if (!g_dev.booted) return;
  // Said on the way out whether or not anyone was watching for it. This is the one condition where memory the driver
  // owns was deliberately not given back, and a run that hit it should not have to be told by a test.
  if (g_dev.gpu.mm.leaked_n)
    fprintf(stderr, "libtinynv: %llu bytes in %llu mappings were leaked rather than reused, because the mmu never "
                    "acknowledged their invalidate. The card was not asked for memory again after that.\n",
            (unsigned long long)g_dev.gpu.mm.leaked_bytes, (unsigned long long)g_dev.gpu.mm.leaked_n);
  tinynv_exec_idle(&g_dev.exec);   // let what was submitted finish before the mappings under it go away
  g_dev.exec.torn_down = 1;
  g_dev.booted = 0;
  g_dev.torn_down = 1;
  tinynv_exec_fini(&g_dev.exec);
  tinynv_gpu_close(&g_dev.gpu);
}

// The same teardown, for the ways out that are not an exit.
//
// atexit is not enough, and the gap is not theoretical: ggml's CUDA_CHECK calls ggml_abort, which calls abort(), which
// raises SIGABRT and never runs an atexit handler. So every failed check inside llama.cpp - the ordinary way a bug
// announces itself - left the GPU bus mastering into mappings about to be reclaimed, which macOS latches as a fault
// flag that only re-enumerating the device clears: one person and one cable, per failure. Found by session A the first
// time a real op suite hit a real bug.
//
// This chains rather than replaces. The card is quiesced, the previous handler is put back, and the signal is raised
// again, so llama.cpp still aborts exactly as it would and still writes its core. A library that swallowed an abort to
// tidy up after itself would be worse than one that left the card dirty.
//
// SIGBUS is in the list for a reason that is specific and was demonstrated rather than imagined: llama.cpp maps the
// model file rather than reading it, and on this machine the models sit on an external drive. A drive that disappears
// mid-run does not give the reader an error - reads from a mapping whose file is gone raise SIGBUS - and that signal
// killed the process without ever reaching the handler that puts the card down. So the one event most likely to end a
// run badly was also the one event this handler did not cover, which is the worst possible place for a gap. It came up
// on 2026-09-14 when the drive went away for a minute with the card attached and idle - deliberately, as it happens,
// which is the point rather than a mitigation: a drive somebody can unplug is a drive that can go away during a run.
static struct sigaction g_prev[5];
static const int g_caught[5] = {SIGABRT, SIGSEGV, SIGBUS, SIGINT, SIGTERM};

// Every host address this driver has written to, most recent first, and what wrote it.
//
// There is exactly one way tensor bytes can reach a caller's memory from here: the memcpy at the end of a download.
// Session A's crash is tensor floats sitting inside another library's hash table, and I have spent three rounds
// reasoning about how they got there and been wrong each time. So instead of another hypothesis, the crash is made to
// say it: if the faulting address falls inside one of these spans, this driver wrote it and the argument is over. If it
// falls inside none of them, this driver did not, and that is worth just as much.
//
// Deliberately not a debug build or a knob. It costs two stores per download on a path that already idles the engine
// and copies megabytes, and the thing it is for happens once in four runs on a good day.
// EVERY host span, not the last sixteen.
//
// Sixteen was two to four tests' worth, and A's corrupted address - the bucket array of a static hash table at
// 0x116d07710 - sat 12.9 MB above the nearest of them. That settled that the last sixteen downloads did not write it
// and nothing else, because the array is allocated at the map's first insert and reallocated at each rehash, which is
// early. An absence only means something if the list is complete.
//
// Fixed and large rather than grown, because this is appended to from whatever thread is copying and a realloc racing
// a reader is a second bug to debug inside the first. 64K entries is 1.5 MB and a few thousand is what a full
// op-verify run produces.
#define WROTE_N 65536
static struct { const void *dst; size_t n; const char *what; } g_wrote[WROTE_N];
static _Atomic unsigned g_wrote_at;

// And the buffers this driver writes into for the life of the device, recorded once rather than per write. The
// descriptor shadows are plain malloc and the stage and mirror are host memory the engine also touches, so an address
// landing in one of those is this driver's doing just as much as a download is - and recording them per write would
// put a store on the launch path.
#define REGION_N 8
static struct { const void *base; size_t n; const char *what; } g_regions[REGION_N];
static _Atomic unsigned g_regions_at;

void tinynv_note_host_region(const void *base, size_t n, const char *what) {
  if (!base || !n) return;
  unsigned i = atomic_fetch_add(&g_regions_at, 1);
  if (i < REGION_N) { g_regions[i].base = base; g_regions[i].n = n; g_regions[i].what = what; }
}

void tinynv_note_host_write(const void *dst, size_t n, const char *what) {
  unsigned i = atomic_fetch_add(&g_wrote_at, 1);
  if (i < WROTE_N) { g_wrote[i].dst = dst; g_wrote[i].n = n; g_wrote[i].what = what; }
}

// Not static, and not because anything else calls it: it is only ever reached from a crash on a live card, which makes
// a bug in it cost a whole run to find. test_witness calls it directly and then asks the file the same question A will
// ask - is this address inside any of these - so the instrument is known to work before the crash it is for.
void tinynv_dump_host_writes(int sig) {
  unsigned at = atomic_load(&g_wrote_at), nreg = atomic_load(&g_regions_at);
  if (!at && !nreg) return;
  // To a file, because the whole list is thousands of lines and the useful question - is this one address inside any
  // of them - is asked with grep afterwards rather than read off a terminal during a crash.
  const char *dir = getenv("TMPDIR");
  char path[512];
  snprintf(path, sizeof(path), "%stinynv-writes-%d.txt", dir && *dir ? dir : "/tmp/", (int)getpid());
  FILE *f = fopen(path, "w");
  fprintf(stderr, "tinynv: signal %d. every host span this driver wrote is in %s (%u writes, %u standing regions)\n",
          sig, f ? path : "(could not be written)", at, nreg);
  if (!f) return;
  fprintf(f, "# every host address libtinynv wrote, oldest first. signal %d, pid %d.\n", sig, (int)getpid());
  fprintf(f, "# an address inside one of these was written by this driver. inside none of them, it was not.\n");
  for (unsigned i = 0; i < nreg && i < REGION_N; i++)
    fprintf(f, "region %p %p %zu %s\n", g_regions[i].base,
            (const void *)((const char *)g_regions[i].base + g_regions[i].n), g_regions[i].n, g_regions[i].what);
  if (at > WROTE_N) fprintf(f, "# TRUNCATED: %u writes happened, %u recorded\n", at, WROTE_N);
  for (unsigned i = 0; i < at && i < WROTE_N; i++)
    fprintf(f, "write %p %p %zu %s\n", g_wrote[i].dst,
            (const void *)((const char *)g_wrote[i].dst + g_wrote[i].n), g_wrote[i].n, g_wrote[i].what);
  fclose(f);
}

static void quiesce_and_continue(int sig) {
  if (sig == SIGSEGV || sig == SIGBUS || sig == SIGABRT) tinynv_dump_host_writes(sig);
  put_the_card_down();
  for (size_t i = 0; i < sizeof(g_caught) / sizeof(*g_caught); i++)
    if (g_caught[i] == sig) { sigaction(sig, &g_prev[i], NULL); break; }
  raise(sig);
}

static void catch_the_ways_out(void) {
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = quiesce_and_continue;
  sigemptyset(&sa.sa_mask);
  for (size_t i = 0; i < sizeof(g_caught) / sizeof(*g_caught); i++) sigaction(g_caught[i], &sa, &g_prev[i]);
}

static int device_boot(tinynv_device_t d) {
  if (d->booted) return 0;
  if (d->torn_down) return tinynv_fail("the gpu has been put down for good; this process cannot use it again");
  if (!d->has_pci) return tinynv_fail("no gpu: this is the null device");
  if (tinynv_gpu_open(&d->gpu, &d->pci)) return -1;
  d->booted = 1; // from here the card is live and must be put down even if a later stage fails
  atexit(put_the_card_down);
  catch_the_ways_out();
  if (tinynv_gpu_init_sw(&d->gpu) || tinynv_gpu_init_hw(&d->gpu)) return -1;
  if (tinynv_gsp_init_objects(&d->gpu) || tinynv_gsp_init_channel(&d->gpu) || tinynv_gsp_init_gr_context(&d->gpu) ||
      tinynv_gsp_open_client(&d->gpu) || tinynv_gsp_init_queues(&d->gpu))
    return -1;

  // Does the memory manager stop below what the firmware owns? The firmware has placed its write-protected region by
  // now, so the two registers describe the firmware THIS open booted and not a resident one from an earlier run
  // (dev.c resets that on arrival, before anything else). The reservation is arithmetic on the sizes we handed over
  // (fw_layout.h); this is the chip's own word on where they landed, and the open is refused if the two disagree -
  // the alternative is an allocator that hands out the firmware's heap the first time a model fills the card, which
  // fails as a garbled RPC hours later with nothing pointing here. Placed outside the recorded boot on purpose:
  // test_dev drives the gpu and gsp stages directly and ends after the queues, so these two reads never meet the
  // replay, which would count them as divergences. TINYNV_FW_RESERVE_CHECK=0 downgrades a refusal to the line, for a
  // diagnostic session that wants to see the numbers and carry on; nothing above the manager's top is safe that run.
  {
    uint32_t lo = tinynv_rd32(&d->gpu.dev, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
    uint32_t hi = tinynv_rd32(&d->gpu.dev, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
    // VAL is bits 31:4 and the alignment is 12; HI names the region's last 4 KB page, so its exclusive limit is one
    // page past it. A hidden region reads zero in both.
    uint64_t wlo = (uint64_t)(lo >> 4) << 12, whi = hi ? (((uint64_t)(hi >> 4) << 12) + 0x1000) : 0;
    char line[256];
    int rc = tinynv_mm_check_fw_carveout(&d->gpu.mm, wlo, whi, line, sizeof line);
    fprintf(stderr, "libtinynv: firmware carveout: held back %llu MB of %llu MB (the sizes handed to the firmware sum "
                    "to %llu MB); %s\n",
            (unsigned long long)(TINYNV_FW_RESERVE_TOP >> 20), (unsigned long long)(d->gpu.dev.vram_size >> 20),
            (unsigned long long)(TINYNV_FW_CARVEOUT_ESTIMATE >> 20), rc ? tinynv_last_error() : line);
    if (rc) {
      const char *e = getenv("TINYNV_FW_RESERVE_CHECK");
      if (!(e && *e == '0')) return -1;
      fprintf(stderr, "libtinynv: carrying on anyway (TINYNV_FW_RESERVE_CHECK=0 was asked for): nothing above the "
                      "manager's top is safe to touch this run\n");
    }
  }
  if (tinynv_exec_init(&d->gpu, &d->exec)) return -1;

  // Sensors, unless refused. Arming costs one 464-byte host buffer, one object and two controls, all once; after that
  // the firmware refreshes the block on its own timer and reading it is a load. Nothing here touches the submission
  // path, which is why this is on by default where the publishing below is not.
  {
    // On by default since 2026-09-15. It was off for a day because test-backend-ops crashed on two of three runs at
    // 911e68e - the first build where the publish path actually ran - and 450/450 with it off, which read as this
    // feature corrupting ggml's heap. It was not. Session A found the writer: llama.cpp's own CPU backend strides a
    // per-thread rope cache by std::hardware_destructive_interference_size (256 here) from C++ while ggml-cpu.c pads
    // the work buffer by its own CACHE_LINE_SIZE of 64, so the last threads' caches run off the end of the allocation.
    // Every garbage value any of us saw was a unit vector, cos2+sin2 = 1.000 over six samples, and it reproduces with
    // no card and no driver at all. Upstream fixed it in 2f53959, one day after our checkout.
    //
    // What the sensors did was move the heap, which changed whether that stray write landed on something live. That is
    // worth naming rather than filing as exoneration: an innocent change can convert a latent bug into a crash, the
    // correlation is real, and "it correlates with our feature" is not the same claim as "our feature does it". A day
    // of suspicion went to the wrong place because I could not tell those apart from the outside.
    //
    // TINYNV_SENSORS=0 turns it off. It gates the publishing below as well as the arming here - it did not, once, and
    // that bug made a run with sensors "off" look like evidence the publish path was innocent when it was still running.
    { const char *e = getenv("TINYNV_DOWNLOAD_VIA_COMPUTE");
      d->download_via_compute = e && *e && *e != '0';
      // Named at start-up like every other mode, including when it is asked for and no kernel has been lent - which
      // is a run where the knob is on and nothing happens, and would otherwise look exactly like a knob that did not
      // help. That distinction has cost this project three card runs today.
      if (d->download_via_compute)
        fprintf(stderr, "libtinynv: downloads are served by a caller's kernel on the compute engine (was asked for)%s\n",
                d->download_kernel ? "" : " - but NO KERNEL HAS BEEN REGISTERED, so the copy engine still serves them"); }

    const char *e = getenv("TINYNV_SENSORS");
    if (!e || *e != '0') {
      if (tinynv_gsp_sensors_init(&d->gpu, TINYNV_RUSD_POLL_THERMAL | TINYNV_RUSD_POLL_POWER |
                                           TINYNV_RUSD_POLL_CLOCK | TINYNV_RUSD_POLL_PERF, 1000))
        // Not fatal. A card that will not report its temperature is still a card that runs models, and refusing to
        // start over a thermometer would be the wrong trade.
        fprintf(stderr, "libtinynv: sensors are not available on this card (%s); everything else is unaffected\n",
                tinynv_last_error());
      else
        // Naming what decided it, the way every other mode line does. A run started with no environment at all has to
        // be tellable from one whose harness is still exporting things - which is exactly the distinction that made a
        // knob-that-did-not-gate look like evidence about the feature it did not gate.
        fprintf(stderr, "libtinynv: sensors are on (%s); tinynv-smi reads them without touching the card\n",
                e ? "was asked for" : "the default");
    } else {
      fprintf(stderr, "libtinynv: sensors are off (was asked for); tinynv-smi will have nothing to read\n");
    }
  }
  return 0;
}

static int booted(tinynv_device_t d) {
  pthread_mutex_lock(&g_lock);
  int rc = device_boot(d);
  pthread_mutex_unlock(&g_lock);
  return rc;
}

static tinynv_alloc_t *find_alloc(tinynv_device_t d, uint64_t va) {
  for (int i = 0; i < d->nallocs; i++) if (d->allocs[i].va == va) return &d->allocs[i];
  return NULL;
}

tinynv_status_t tinynv_init(void) {
  pthread_mutex_lock(&g_lock);
  if (g_inited) { pthread_mutex_unlock(&g_lock); return TINYNV_OK; }
  memset(&g_dev, 0, sizeof(g_dev));
  // a real device when one can be opened. the socket is how macOS reaches the gpu; TINYNV_PCI names a linux device instead
  const char *sock = getenv("TINYNV_SOCKET");
  // Empty is not a device name, it is the absence of one - which is how every other knob in this library reads its
  // environment, and the only reading that lets a caller say "definitely no card" in a variable. Without this, an
  // empty value is a name that cannot be opened, and the refusal below fires on a request nobody made.
  if (sock && !*sock) sock = NULL;
#ifdef __linux__
  const char *bdf = getenv("TINYNV_PCI");
  if (bdf && !tinynv_pci_open_sysfs(bdf, &g_dev.pci)) g_dev.has_pci = 1;
#endif
  if (!g_dev.has_pci && sock && !tinynv_pci_open_tinygpu(sock, &g_dev.pci)) g_dev.has_pci = 1;

  // Naming a device and not getting one is an error, not a fallback. Without this, a wrong socket path or a server that
  // is not running turns the whole library into the null device: every call succeeds, every kernel is reported rather
  // than run, and the answers are whatever was already in host memory. That is a far worse outcome than refusing to
  // start, and it is indistinguishable from working until someone checks the numbers.
  if (!g_dev.has_pci && (sock
#ifdef __linux__
                         || bdf
#endif
                         )) {
    g_inited = 1;
    pthread_mutex_unlock(&g_lock);
    return tinynv_fail("a gpu was named but could not be opened: %s", tinynv_last_error()), TINYNV_ERR_NODEV;
  }
  g_inited = 1;
  pthread_mutex_unlock(&g_lock);
  return TINYNV_OK;
}

int tinynv_device_count(void) { return 1; } // one device, real or null

tinynv_status_t tinynv_device_get(tinynv_device_t *out, int ordinal) {
  if (ordinal != 0) return TINYNV_ERR_NODEV;
  if (!g_inited && tinynv_init() != TINYNV_OK) return TINYNV_ERR_DRIVER;
  *out = &g_dev;
  return TINYNV_OK;
}

tinynv_status_t tinynv_device_props(tinynv_device_t d, tinynv_device_props_t *p) {
  memset(p, 0, sizeof(*p));
  p->warp_size = 32;
  p->max_threads_per_block = 1024;
  p->max_shared_per_block = 48 * 1024;   // what a kernel gets without asking for more, as real cuda reports it
  // The opt-in ceiling, measured on this architecture rather than quoted from a table. A caller deriving its own
  // maximum from this gets the number the driver will actually refuse above, which is the point of reporting it.
  p->max_shared_per_sm = TINYNV_SMEM_MAX_PER_BLOCK;

  if (!d->has_pci) {
    snprintf(p->name, sizeof(p->name), "tinynv-null (no gpu)");
    return TINYNV_OK;   // cc_major stays 0, which marks a device that cannot run anything
  }
  // Answering this properly means booting, because every number below comes from the card. A caller that only wanted to
  // know what is present pays for a boot it was going to pay for anyway.
  if (booted(d)) return TINYNV_ERR_DRIVER;

  const tinynv_gsp_t *gsp = &d->gpu.gsp;
  // The same rule the oracle uses, special case and all: 0xa04 is sm_120 and does not follow from its own bits.
  unsigned sm = gsp->sm_version, lo = sm & 0xff;
  if (sm == 0xa04) { p->cc_major = 12; p->cc_minor = 0; }
  else { p->cc_major = (int)((sm >> 8) & 0xff); p->cc_minor = (int)(lo > 0xf ? lo >> 4 : lo); }

  snprintf(p->name, sizeof(p->name), "%s (sm_%d%d)", d->gpu.dev.chip_name, p->cc_major, p->cc_minor);
  // What the manager can actually hand out, not the size of the chip's memory: the firmware's reservation at the top
  // and the boot and page-table regions at the bottom are nobody's to allocate. Reporting the chip's figure (as this
  // did until 2026-09-19) overstated the budget by ~322 MB, which is exactly the kind of number a caller sizing a
  // model's residency plans against. Free space is still reported as this total (cudart_api.c); that is the next lie
  // to remove, and it needs a live-bytes counter this manager does not keep yet.
  p->total_mem = d->gpu.mm.pa.size;
  // The most cores the die could have, not the number this part has enabled: the count of enabled ones is a separate
  // query this driver does not make yet. A caller sizing a grid by it will overshoot, which costs occupancy and not
  // correctness - but it is an upper bound and says so here rather than pretending to be measured.
  p->sm_count = (int)(gsp->max_gpcs * gsp->max_tpc_per_gpc * gsp->max_sm_per_tpc);
  return TINYNV_OK;
}

tinynv_status_t tinynv_module_load(tinynv_device_t d, const void *cubin, size_t len, tinynv_module_t *out) {
  if (!cubin || !len) return TINYNV_ERR_INVALID;
  tinynv_module_t m = calloc(1, sizeof(*m));
  m->dev = d;
  m->image = malloc(len);
  memcpy(m->image, cubin, m->len = len);
  if (tinynv_cubin_parse(m->image, m->len, &m->cubin)) {
    free(m->image);
    free(m);
    return TINYNV_ERR_INVALID;
  }
  *out = m;
  return TINYNV_OK;
}

tinynv_status_t tinynv_module_unload(tinynv_module_t m) {
  if (!m) return TINYNV_ERR_INVALID;
  if (m->is_loaded && !m->dev->torn_down) {
    if (m->dev->booted) tinynv_exec_idle(&m->dev->exec);   // a kernel from this module may still be queued
    tinynv_exec_unload(&m->dev->gpu.mm, &m->loaded);
  }
  tinynv_cubin_free(&m->cubin);
  free(m->image);
  free(m);
  return TINYNV_OK;
}

tinynv_status_t tinynv_get_kernel(tinynv_module_t m, const char *name, tinynv_kernel_t *out) {
  if (!m || !name) return TINYNV_ERR_INVALID;
  const tinynv_kernel_desc_t *d = tinynv_cubin_kernel(&m->cubin, name);
  if (!d) return TINYNV_ERR_INVALID;
  tinynv_kernel_t k = calloc(1, sizeof(*k));
  k->mod = m;
  k->desc = d;
  *out = k;
  return TINYNV_OK;
}

tinynv_status_t tinynv_kernel_info(tinynv_kernel_t k, tinynv_kernel_info_t *i) {
  if (!k) return TINYNV_ERR_INVALID;
  memset(i, 0, sizeof(*i));
  const tinynv_kernel_desc_t *d = k->desc;
  // the parameter base is the cubin's own: 0x160 on sm_8x, 0x210 on sm_90, 0x380 on sm_12x
  i->param_base = (int)d->param_base;
  i->regs = (int)d->regs;
  i->static_smem = (int)d->static_smem;
  int n = d->nparams;
  if (n > (int)(sizeof(i->params) / sizeof(i->params[0]))) return tinynv_fail("kernel %s has %d parameters, the interface carries %zu",
                                                                              d->name, n, sizeof(i->params) / sizeof(i->params[0])),
                                                                   TINYNV_ERR_INVALID;
  i->num_params = n;
  for (int j = 0; j < n; j++) {
    i->params[j].offset = (int)d->params[j].offset;
    i->params[j].size = (int)d->params[j].size;
  }
  return TINYNV_OK;
}

// *** memory. the null device answers with host memory so the whole shim path can be exercised without a gpu.

tinynv_status_t tinynv_malloc(tinynv_device_t d, size_t n, tinynv_devptr_t *out) {
  announce(d);
  if (d->has_pci) {
    if (booted(d)) return TINYNV_ERR_DRIVER;
    pthread_mutex_lock(&g_lock);
    if (d->nallocs == d->cap_allocs) {
      int cap = d->cap_allocs ? d->cap_allocs * 2 : 16;
      tinynv_alloc_t *a = realloc(d->allocs, (size_t)cap * sizeof(*a));
      if (!a) { pthread_mutex_unlock(&g_lock); return TINYNV_ERR_OOM; }
      d->allocs = a;
      d->cap_allocs = cap;
    }
    tinynv_alloc_t *a = &d->allocs[d->nallocs];
    // Video memory, not host memory, and not visible from here: this is where a model's weights go. The allocator hands
    // back a virtual address whose alignment the page tables guarantee, which is far more than the 256 promised above.
    if (tinynv_mm_alloc_buffer(&d->gpu.mm, n ? n : 1, 0, 0, 0, 1, 0, &a->map)) {
      pthread_mutex_unlock(&g_lock);
      return TINYNV_ERR_OOM;
    }
    a->va = a->map.va;
    d->nallocs++;
    *out = (tinynv_devptr_t)a->va;
    pthread_mutex_unlock(&g_lock);
    return TINYNV_OK;
  }
  // Aligned, and it is a guarantee rather than a convenience. Callers above this line lay tensors out from a buffer's
  // base assuming device pointers are aligned, and lose the difference to padding when they are not: ggml's allocator
  // assumes at least 128 and real CUDA promises 256. Plain malloc gives 16 on macOS, which passed one run on the luck of
  // the heap and failed the next with "needed 128, available 112". Found by session A running llama.cpp's op suite.
  void *p = NULL;
  if (posix_memalign(&p, TINYNV_DEVPTR_ALIGN, n ? n : 1)) p = NULL;
  *out = (tinynv_devptr_t)(uintptr_t)p;
  return p ? TINYNV_OK : TINYNV_ERR_OOM;
}

int tinynv_device_last_fault(tinynv_device_t d, uint64_t *va, uint32_t *type, uint32_t *access, uint32_t *sm_esr) {
  if (!d || !d->has_pci || !d->booted) return 0;
  const tinynv_gsp_t *gsp = &d->gpu.gsp;
  if (gsp->fault_have) {
    if (va) *va = gsp->fault_va;
    if (type) *type = gsp->fault_type;
    if (access) *access = gsp->fault_access;
    return 1;
  }
  if (gsp->fault_sm_have) {
    if (sm_esr) *sm_esr = gsp->fault_sm_esr;
    return 2;
  }
  return 0;
}

tinynv_status_t tinynv_free(tinynv_device_t d, tinynv_devptr_t p) {
  if (d->has_pci) {
    // freeing after the card is down is not an error; there is simply nothing left to give back
    if (d->torn_down) return TINYNV_OK;
    pthread_mutex_lock(&g_lock);
    // Work is submitted without waiting now, so a buffer being freed may still be under a kernel that has not run. CUDA
    // makes cudaFree synchronising for exactly this reason; unmapping memory an engine is reading is not something to
    // leave to the caller's discipline.
    if (d->booted && tinynv_exec_idle(&d->exec)) { pthread_mutex_unlock(&g_lock); return TINYNV_ERR_DRIVER; }
    tinynv_alloc_t *a = find_alloc(d, (uint64_t)p);
    if (!a) {
      pthread_mutex_unlock(&g_lock);
      // A pointer into the middle of an allocation, or one already freed. Saying so beats freeing something else.
      return tinynv_fail("free: %#llx is not the base of any allocation", (unsigned long long)p), TINYNV_ERR_INVALID;
    }
    tinynv_vmap_free(&d->gpu.mm, &a->map);
    *a = d->allocs[--d->nallocs];
    pthread_mutex_unlock(&g_lock);
    return TINYNV_OK;
  }
  free((void *)(uintptr_t)p);
  return TINYNV_OK;
}

// Every copy on a real device goes through the copy engine, and host memory reaches it by being staged: the pages a
// caller hands us are its own, and the GPU can only read memory that has been mapped for it.
#define ON_GPU(s) ((s) && (s)->dev->has_pci)
#define NEED_GPU(s) do { if (booted((s)->dev)) return TINYNV_ERR_DRIVER; } while (0)

// Everything that touches the device holds the same lock, and that includes the calls that move memory.
//
// They did not, which was survivable only because ggml drives this from one thread: they share the scratch arena, the
// timeline and the command buffers with launches and allocations, so two threads would have handed the same command
// buffer to the engine twice. Nothing nests - the public entry points take this and the internals never do - so a plain
// mutex is enough.
#define WITH_LOCK(expr) ({ pthread_mutex_lock(&g_lock); int rc_ = (expr); pthread_mutex_unlock(&g_lock); rc_; })

tinynv_status_t tinynv_memcpy_htod(tinynv_stream_t s, tinynv_devptr_t dst, const void *src, size_t n) {
  if (ON_GPU(s)) {
    NEED_GPU(s);
    return WITH_LOCK(tinynv_exec_upload(&s->dev->exec, (uint64_t)dst, src, n)) ? TINYNV_ERR_DRIVER : TINYNV_OK;
  }
  memcpy((void *)(uintptr_t)dst, src, n);
  return TINYNV_OK;
}

// One chunk at a time through the staging buffer, exactly as the copy-engine path does - same buffer, same size, same
// wait-then-memcpy - with a kernel launch where the copy-engine batch used to be.
static tinynv_status_t download_by_kernel(tinynv_stream_t s, void *dst, uint64_t src_va, size_t n) {
  tinynv_device_t d = s->dev;
  tinynv_kernel_info_t info;
  if (tinynv_kernel_info(d->download_kernel, &info) != TINYNV_OK) return TINYNV_ERR_DRIVER;
  const size_t stage = tinynv_exec_stage_bytes();
  uint8_t *p = dst;
  for (size_t off = 0; off < n; off += stage) {
    uint64_t chunk = n - off < stage ? n - off : stage;
    uint8_t params[64];
    memset(params, 0, sizeof(params));
    uint64_t a = tinynv_exec_stage_va(&d->exec), b = src_va + off;
    // Placed at the offsets the CUBIN declares, not at 0/8/16. The registration already refused anything that is not
    // three 8-byte parameters, so the only thing left to get wrong is where they sit, and the kernel says where.
    memcpy(params + info.params[0].offset, &a, 8);
    memcpy(params + info.params[1].offset, &b, 8);
    memcpy(params + info.params[2].offset, &chunk, 8);
    uint64_t per_block = (uint64_t)d->download_block * d->download_bpt;
    uint64_t blocks = (chunk + per_block - 1) / per_block;
    if (blocks > 0xffffffffull) return tinynv_fail("a %llu byte download needs %llu blocks", (unsigned long long)chunk,
                                                   (unsigned long long)blocks), TINYNV_ERR_INVALID;
    // The driver's own launch, not the caller's: marked so the profile does not stamp it as the token's last kernel.
    tinynv_exec_set_internal(&d->exec, 1);
    tinynv_status_t rc = tinynv_launch(s, d->download_kernel, (unsigned)blocks, 1, 1, d->download_block, 1, 1, 0,
                                       params, (size_t)info.params[2].offset + 8);
    tinynv_exec_set_internal(&d->exec, 0);
    if (rc != TINYNV_OK) return rc;
    if (WITH_LOCK(tinynv_exec_idle(&d->exec))) return TINYNV_ERR_DRIVER;
    tinynv_note_host_write(p + off, (size_t)chunk, "download by kernel: stage -> caller");
    memcpy(p + off, tinynv_exec_stage_host(&d->exec), (size_t)chunk);
  }
  return TINYNV_OK;
}

tinynv_status_t tinynv_set_download_kernel(tinynv_device_t d, tinynv_kernel_t k, unsigned block_threads,
                                           unsigned bytes_per_thread) {
  if (!d || !k) return TINYNV_ERR_INVALID;
  // Held against the cubin's own declaration, not against a convention we agreed in a message. A kernel whose
  // parameters are not what this path will write is refused HERE, where the error names the kernel, rather than at
  // the first download, where it would be a wrong copy with nothing pointing at the registration.
  tinynv_kernel_info_t info;
  if (tinynv_kernel_info(k, &info) != TINYNV_OK) return TINYNV_ERR_INVALID;
  if (info.num_params != 3)
    return tinynv_fail("a download kernel takes exactly three parameters - destination, source, byte count - and this "
                       "one declares %d", info.num_params), TINYNV_ERR_INVALID;
  for (int i = 0; i < 3; i++)
    if (info.params[i].size != 8)
      return tinynv_fail("a download kernel's parameter %d is %d bytes; all three must be 8", i, info.params[i].size),
             TINYNV_ERR_INVALID;
  if (!block_threads || !bytes_per_thread)
    return tinynv_fail("a download kernel needs a non-zero block size and bytes per thread; got %u and %u",
                       block_threads, bytes_per_thread), TINYNV_ERR_INVALID;
  d->download_kernel = k;
  d->download_block = block_threads;
  d->download_bpt = bytes_per_thread;
  return TINYNV_OK;
}

tinynv_status_t tinynv_set_keepalive_kernel(tinynv_device_t d, tinynv_kernel_t k) {
  if (!d || !k) return TINYNV_ERR_INVALID;
  tinynv_kernel_info_t info;
  if (tinynv_kernel_info(k, &info) != TINYNV_OK) return TINYNV_ERR_INVALID;
  if (info.num_params != 2 || info.params[0].size != 8 || info.params[1].size != 8)
    return tinynv_fail("a keep-alive kernel takes exactly two 8-byte parameters - the flag's address and a cycle budget - "
                       "and this one declares %d", info.num_params), TINYNV_ERR_INVALID;
  d->keepalive_kernel = k;
  if (d->exec.keepalive)
    fprintf(stderr, "libtinynv: keep-alive on (was asked for): after every synchronisation a lent kernel keeps the compute "
                    "engine scheduled until the next chain is handed over, %u us at most\n", d->exec.keepalive_us);
  return TINYNV_OK;
}

// The keep-alive launch: the driver's own work, after the engine has been idled by a synchronisation. Outside the lock,
// because tinynv_launch takes it. Any failure is reported once and the knob is dropped rather than retried per token.
static void keepalive_launch(tinynv_stream_t s) {
  tinynv_device_t d = s->dev;
  // Only after a token's worth of the caller's kernels: the small read-backs inside a token synchronise too (the shim
  // syncs the stream after every synchronous host copy), and those must not each start a spin the next wait pays for.
  if (d->exec.launches_since_ka < TINYNV_KEEPALIVE_MIN_LAUNCHES) return;
  tinynv_kernel_info_t info;
  uint64_t flag_va = 0;
  if (tinynv_kernel_info(d->keepalive_kernel, &info) != TINYNV_OK || WITH_LOCK(tinynv_exec_keepalive_arm(&d->exec, &flag_va))) {
    fprintf(stderr, "libtinynv: the keep-alive could not be armed (%s); it is off from here\n", tinynv_last_error());
    d->exec.keepalive = 0;
    return;
  }
  uint8_t params[64];
  memset(params, 0, sizeof(params));
  uint64_t cycles = (uint64_t)d->exec.keepalive_us * 3000ull;   // ~3 cycles a nanosecond at the boost clock: a ceiling
  memcpy(params + info.params[0].offset, &flag_va, 8);
  memcpy(params + info.params[1].offset, &cycles, 8);
  tinynv_exec_set_internal(&d->exec, 1);
  tinynv_status_t rc = tinynv_launch(s, d->keepalive_kernel, 1, 1, 1, 32, 1, 1, 0, params, (size_t)info.params[1].offset + 8);
  if (rc == TINYNV_OK && WITH_LOCK(tinynv_exec_flush(&d->exec))) rc = TINYNV_ERR_DRIVER;
  d->exec.ka_value = d->exec.timeline;   // the keep-alive was the last thing submitted; waits up to here ignore it
  tinynv_exec_set_internal(&d->exec, 0);
  if (rc != TINYNV_OK) {
    fprintf(stderr, "libtinynv: the keep-alive launch failed (%s); it is off from here\n", tinynv_last_error());
    WITH_LOCK((tinynv_exec_keepalive_release(&d->exec, 1), 0));
    d->exec.keepalive = 0;
  }
}

tinynv_status_t tinynv_graph_begin(tinynv_stream_t s) {
  if (!ON_GPU(s)) return TINYNV_ERR_UNSUPPORTED;
  NEED_GPU(s);
  int rc = WITH_LOCK(tinynv_exec_graph_begin(&s->dev->exec));
  return rc == 1 ? TINYNV_ERR_UNSUPPORTED : rc ? TINYNV_ERR_DRIVER : TINYNV_OK;
}
tinynv_status_t tinynv_graph_end(tinynv_stream_t s, tinynv_graph_t *out) {
  if (out) *out = NULL;
  if (!ON_GPU(s) || !out) return TINYNV_ERR_UNSUPPORTED;
  tinynv_graph_rec_t *r = NULL;
  int rc = WITH_LOCK(tinynv_exec_graph_end(&s->dev->exec, &r));
  *out = (tinynv_graph_t)r;
  return rc ? TINYNV_ERR_DRIVER : TINYNV_OK;
}
tinynv_status_t tinynv_graph_launch(tinynv_stream_t s, tinynv_graph_t g) {
  if (!ON_GPU(s) || !g) return TINYNV_ERR_INVALID;
  NEED_GPU(s);
  s->submitted++;
  return WITH_LOCK(tinynv_exec_graph_launch(&s->dev->exec, (tinynv_graph_rec_t *)g)) ? TINYNV_ERR_DRIVER : TINYNV_OK;
}
void tinynv_graph_free(tinynv_stream_t s, tinynv_graph_t g) {
  if (!g || !s) return;
  s->dev->exec.torn_down = s->dev->torn_down;
  WITH_LOCK((tinynv_exec_graph_free(&s->dev->exec, (tinynv_graph_rec_t *)g), 0));
}

tinynv_status_t tinynv_memcpy_dtoh(tinynv_stream_t s, void *dst, tinynv_devptr_t src, size_t n) {
  if (ON_GPU(s)) {
    NEED_GPU(s);
    // Served by the caller's kernel on the COMPUTE engine when asked for, so the copy engine is not in the picture at
    // a token boundary and the handoff that follows is compute after compute - a queue, not a switch.
    //
    // The unknown Session A named and neither of us should assume away: an SM writing host memory over Thunderbolt
    // may well be slower per byte than the copy engine's bursts, so this can trade a ~500-1,100 us handoff for a
    // slower transfer. The raw columns say which in one run - tail->copy-start should collapse, and copy-start to
    // copy-end becomes a real interval because this path's release does wait for its kernel.
    tinynv_status_t rc = (s->dev->download_via_compute && s->dev->download_kernel && n)
                             ? download_by_kernel(s, dst, (uint64_t)src, n)
                             : WITH_LOCK(tinynv_exec_download(&s->dev->exec, dst, (uint64_t)src, n)) ? TINYNV_ERR_DRIVER : TINYNV_OK;
    // A download waits for itself, so the engine is idle here - on a decode this is the token boundary, and the stream
    // synchronisation the caller issues next is answered without a wait. TINYNV_KEEPALIVE: keep the engine scheduled.
    // Only a download the size of the logits: the ~24 small read-backs a token are not the boundary, and a keep-alive
    // after each of them cost the decode 20% (measured 15:28, 2,287 launches for 96 tokens, 2,192 released by a wait).
    if (rc == TINYNV_OK && s->dev->exec.keepalive && s->dev->keepalive_kernel && !s->dev->torn_down &&
        n >= (size_t)s->dev->exec.keepalive_min_kb * 1024u)
      keepalive_launch(s);
    return rc;
  }
  memcpy(dst, (const void *)(uintptr_t)src, n);
  return TINYNV_OK;
}

tinynv_status_t tinynv_memcpy_dtod(tinynv_stream_t s, tinynv_devptr_t dst, tinynv_devptr_t src, size_t n) {
  if (ON_GPU(s)) {
    NEED_GPU(s);
    // The engine copies forward, so overlapping ranges are not the memmove the null device gives. CUDA does not promise
    // it either, but a caller that relied on the null device's behaviour would find out on hardware and not before.
    if (dst != src && dst < src + n && src < dst + n)
      return tinynv_fail("device to device copy of %zu bytes overlaps: the copy engine has no defined order for that", n),
             TINYNV_ERR_INVALID;
    return WITH_LOCK(tinynv_exec_copy(&s->dev->exec, (uint64_t)dst, (uint64_t)src, n)) ? TINYNV_ERR_DRIVER : TINYNV_OK;
  }
  memmove((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, n);
  return TINYNV_OK;
}

tinynv_status_t tinynv_memset(tinynv_stream_t s, tinynv_devptr_t dst, int v, size_t n) {
  if (ON_GPU(s)) {
    NEED_GPU(s);
    // The copy engine has a fill, and this does not use it yet: a pattern is written into the staging buffer once and
    // copied across in chunks. Correct, and one transfer per chunk more than it needs to be.
    tinynv_exec_t *ex = &s->dev->exec;
    if (!v) return WITH_LOCK(tinynv_exec_zero(ex, (uint64_t)dst, n)) ? TINYNV_ERR_DRIVER : TINYNV_OK;
    // held across the whole fill: the staging buffer holds the pattern, so another caller copying through it between
    // chunks would write its own bytes into the rest of this one
    pthread_mutex_lock(&g_lock);
    size_t chunk = ex->stage.size < n ? (size_t)ex->stage.size : n;
    memset(ex->stage.dma.va, v, chunk);
    int bad = 0;
    for (size_t off = 0; off < n && !bad; off += chunk) {
      size_t m = n - off < chunk ? n - off : chunk;
      bad = tinynv_exec_copy(ex, (uint64_t)dst + off, ex->stage.va, m);
    }
    pthread_mutex_unlock(&g_lock);
    return bad ? TINYNV_ERR_DRIVER : TINYNV_OK;
  }
  memset((void *)(uintptr_t)dst, v, n);
  return TINYNV_OK;
}

tinynv_status_t tinynv_host_alloc(tinynv_device_t d, size_t n, void **out) {
  announce(d);
  // TODO(M3.5): dma memory, which the dext describes in at most 32 segments. above that this returns INVALID by contract.
  // aligned for the same reason device memory is: a caller that stages through this then copies has the same layout
  // assumptions on both sides
  void *p = NULL;
  if (posix_memalign(&p, TINYNV_DEVPTR_ALIGN, n ? n : 1)) p = NULL;
  return (*out = p) ? TINYNV_OK : TINYNV_ERR_OOM;
}

tinynv_status_t tinynv_host_free(tinynv_device_t d, void *p) {
  free(p);
  return TINYNV_OK;
}

// *** streams, events, launch

tinynv_status_t tinynv_stream_create(tinynv_device_t d, tinynv_stream_t *out) {
  tinynv_stream_t s = calloc(1, sizeof(*s));
  s->dev = d;
  *out = s;
  return TINYNV_OK;
}
tinynv_status_t tinynv_stream_destroy(tinynv_stream_t s) { free(s); return TINYNV_OK; }
// Every batch this driver submits is waited on before the call that made it returns, so by the time anything can ask,
// the work is done. Waiting on the timeline anyway costs one read and keeps this honest if that ever stops being true.
// Celsius from the card's fixed point: signed, eight fractional bits, so 256ths of a degree.
static double rusd_celsius(int32_t t) { return (double)t / 256.0; }

// A reading counts as present when its timestamp is not the "never written" zero. See tinynv_gsp_sensors_read, which
// leaves a field zeroed when the firmware was mid-write every time it looked - so a caller cannot tell those two apart,
// and should not: both mean do not trust this number.
#define SENSOR_OK(f) ((f).ts != 0)

int tinynv_device_gsp_free_heap(tinynv_device_t d, uint64_t *out) {
  if (!d || !out) return -1;
  if (!d->has_pci) return -1;
  if (booted(d)) return -1;
  return WITH_LOCK(tinynv_gsp_free_heap(&d->gpu, out));
}

// Where the firmware's write-protected region ACTUALLY sits, read off the chip rather than computed from what we
// asked for. Diagnostic; see test/test_hw_wpr.c for why that difference is the whole point.
int tinynv_device_wpr2_range(tinynv_device_t d, uint64_t *base, uint64_t *limit, uint64_t *vram, uint64_t *managed_end) {
  if (!d || !base || !limit || !vram || !managed_end) return -1;
  if (!d->has_pci) return -1;
  if (booted(d)) return -1;
  // VAL is bits 31:4 of each register and the alignment is 12, so the address is the field shifted up by 12.
  uint32_t lo = tinynv_rd32(&d->gpu.dev, NV_PFB_PRI_MMU_WPR2_ADDR_LO);
  uint32_t hi = tinynv_rd32(&d->gpu.dev, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
  *base = (uint64_t)(lo >> 4) << 12;
  *limit = (uint64_t)(hi >> 4) << 12;
  *vram = d->gpu.dev.vram_size;
  *managed_end = d->gpu.mm.pa.base + d->gpu.mm.pa.size;
  return 0;
}

int tinynv_device_probe_phys_attr(tinynv_device_t d, uint64_t size, uint64_t offset, uint64_t *paddr,
                                  uint64_t *contig, unsigned *aperture) {
  if (!d || !paddr || !contig || !aperture) return -1;
  if (!d->has_pci) return -1;
  if (booted(d)) return -1;
  uint32_t ap = 0;
  int rc = WITH_LOCK(tinynv_gsp_probe_phys_attr(&d->gpu, size, offset, paddr, contig, &ap));
  *aperture = ap;
  return rc;
}

tinynv_status_t tinynv_sensors(tinynv_device_t d, tinynv_sensors_t *out) {
  if (!d || !out) return TINYNV_ERR_INVALID;
  memset(out, 0, sizeof(*out));
  if (!d->has_pci) return TINYNV_ERR_NODEV;
  if (booted(d)) return TINYNV_ERR_DRIVER;

  tinynv_rusd_t r;
  if (WITH_LOCK(tinynv_gsp_sensors_read(&d->gpu, &r))) return TINYNV_ERR_DRIVER;

  if ((out->gpu_temp_ok = SENSOR_OK(r.temperatures[0]))) out->gpu_temp_c = rusd_celsius(r.temperatures[0].temperature);
  if ((out->mem_temp_ok = SENSOR_OK(r.temperatures[1]))) out->mem_temp_c = rusd_celsius(r.temperatures[1].temperature);
  if ((out->power_ok = SENSOR_OK(r.avgPowerUsage))) {
    out->gpu_power_mw = r.avgPowerUsage.averageGpuPower;
    out->mem_power_mw = r.avgPowerUsage.averageMemoryPower;
  }
  if ((out->limit_ok = SENSOR_OK(r.powerLimitGpu))) out->power_limit_mw = r.powerLimitGpu.enforcedmW;
  if ((out->clock_ok = SENSOR_OK(r.clkPublicDomainInfos))) {
    out->graphics_mhz = r.clkPublicDomainInfos.targetClkMHz[0];
    out->memory_mhz = r.clkPublicDomainInfos.targetClkMHz[1];
    out->video_mhz = r.clkPublicDomainInfos.targetClkMHz[2];
    out->sm_mhz = r.clkPublicDomainInfos.targetClkMHz[3];
  }
  if ((out->util_ok = SENSOR_OK(r.perfDevUtil))) {
    out->gpu_busy_pct = r.perfDevUtil.gpuPercentBusy;
    out->mem_busy_pct = r.perfDevUtil.memoryPercentBusy;
  }
  if ((out->pstate_ok = SENSOR_OK(r.perfCurrentPstate))) out->pstate = r.perfCurrentPstate.currentPstate;
  if ((out->throttle_ok = SENSOR_OK(r.clkThrottleReason))) out->throttle = r.clkThrottleReason.reasonMask;
  return TINYNV_OK;
}

int tinynv_throttle_str(unsigned mask, const char **out, int max) {
  // Bit order and meanings from RUSD_CLK_THROTTLE_REASON_* in the pinned tree. Worth having as words rather than a
  // number, because "why is it slow" is the question these readings exist to answer, and a bitmask does not answer it.
  static const char *const names[] = {
    "idle", "application clock setting", "power cap", "hardware slowdown", "sync boost",
    "thermal (software)", "thermal (hardware)", "power brake", "display clock setting",
  };
  int n = 0;
  for (unsigned i = 0; i < sizeof(names) / sizeof(*names) && n < max; i++)
    if (mask & (1u << i)) out[n++] = names[i];
  return n;
}

// Publishing, and why it exists at all.
//
// The broker accepts ONE client at a time - listen(fd, 1), one connection served to completion - so a separate tool
// cannot open the card while a model is running, which is exactly when you want to know the temperature. Worse, it
// would sit in the backlog holding the only slot. So the process that holds the card writes what it sees, and the tool
// reads that. Plain text, because then it is useful with cat and needs no tool at all.
//
// Written by rename, which is atomic, so a reader never sees half a file. Driven from the synchronisation points the
// host is already waiting at, rate limited, so the cost on a decode is one clock read per sync - about 25 ns against a
// sync that costs tens of microseconds. TINYNV_SENSOR_FILE moves it; TINYNV_SENSOR_FILE= (empty) turns it off.
// Whether the firmware ever agreed to keep the block fresh. Cheap, and asked before anything else, because this runs
// from every synchronisation and a card that refused sensors should pay nothing at all for them.
static int tinynv_device_sensors_armed(tinynv_device_t d) { return d->has_pci && d->booted && d->gpu.gsp.rusd_armed; }

static void publish_sensors(tinynv_device_t d) {
  // Every static here is reached from whatever thread happened to synchronise, and llama.cpp has several. The rate
  // limit was a plain double read and written without synchronisation, which is a data race and therefore undefined
  // behaviour, not merely a sloppy timer - and this path is under suspicion for corrupting another library's memory, so
  // guessing that a race here is harmless is exactly the move not to make. One thread publishes at a time and the
  // winner is decided by an atomic exchange rather than by luck.
  static _Atomic double last;
  static _Atomic int busy;
  static int off = -1;
  if (off < 0) { const char *e = getenv("TINYNV_SENSOR_FILE"); off = (e && !*e); }
  if (off) return;
  // TINYNV_SENSORS=0 has to turn this off too, and it did not.
  //
  // This is the whole explanation for the one result that looked anomalous in A's data: at 911e68e, op-verify crashed
  // once in three WITH TINYNV_SENSORS=0, which made no sense if the knob disabled the feature. It did not. It gated
  // ARMING, while this function ran from every synchronisation regardless, did its clock read and its statics, called
  // in, and only then found nothing armed. So the knob named a thing it did not control, and anybody switching it off
  // to test a hypothesis - which is exactly what it was for - was testing the wrong thing.
  if (!d || !tinynv_device_sensors_armed(d)) return;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double now = (double)tv.tv_sec + tv.tv_usec / 1e6;   // wall clock: a reader in another process compares it
  if (now - atomic_load(&last) < 0.25) return;
  int expected = 0;
  if (!atomic_compare_exchange_strong(&busy, &expected, 1)) return;   // somebody else is already writing it
  atomic_store(&last, now);

  tinynv_sensors_t s;
  if (tinynv_sensors(d, &s) != TINYNV_OK) { atomic_store(&busy, 0); return; }

  const char *path = getenv("TINYNV_SENSOR_FILE");
  char fixed[512], tmp[540];
  if (!path || !*path) {
    const char *dir = getenv("TMPDIR");
    snprintf(fixed, sizeof(fixed), "%stinynv-sensors", dir && *dir ? dir : "/tmp/");
    path = fixed;
  }
  snprintf(tmp, sizeof(tmp), "%s.tmp", path);
  FILE *f = fopen(tmp, "w");
  if (!f) { atomic_store(&busy, 0); return; }
  fprintf(f, "time %.3f\npid %d\n", now, (int)getpid());
  if (s.gpu_temp_ok) fprintf(f, "gpu_temp_c %.1f\n", s.gpu_temp_c);
  if (s.mem_temp_ok) fprintf(f, "mem_temp_c %.1f\n", s.mem_temp_c);
  if (s.power_ok) fprintf(f, "gpu_power_w %.1f\nmem_power_w %.1f\n", s.gpu_power_mw / 1000.0, s.mem_power_mw / 1000.0);
  if (s.limit_ok) fprintf(f, "power_limit_w %.1f\n", s.power_limit_mw / 1000.0);
  if (s.clock_ok) fprintf(f, "graphics_mhz %u\nmemory_mhz %u\nsm_mhz %u\n", s.graphics_mhz, s.memory_mhz, s.sm_mhz);
  if (s.util_ok) fprintf(f, "gpu_busy_pct %u\nmem_busy_pct %u\n", s.gpu_busy_pct, s.mem_busy_pct);
  if (s.pstate_ok) fprintf(f, "pstate %u\n", s.pstate);
  if (s.throttle_ok) {
    const char *why[9];
    int n = tinynv_throttle_str(s.throttle, why, 9);
    fprintf(f, "throttle %s", n ? "" : "none");
    for (int i = 0; i < n; i++) fprintf(f, "%s%s", i ? ", " : "", why[i]);
    fprintf(f, "\n");
  }
  fclose(f);
  if (rename(tmp, path)) unlink(tmp);
  atomic_store(&busy, 0);
}

// See tinynv.h. Zero means a caller that wants its own host buffer back need only flush; non-zero means something
// outstanding has to finish first.
int tinynv_stream_needs_wait(tinynv_stream_t s) {
  if (!ON_GPU(s)) return 0;
  if (booted(s->dev)) return 1;   // cannot answer, so answer the safe way
  return WITH_LOCK(tinynv_exec_needs_wait(&s->dev->exec));
}

tinynv_status_t tinynv_stream_flush(tinynv_stream_t s) {
  if (!ON_GPU(s)) return TINYNV_OK;
  NEED_GPU(s);
  // Published from here as well as from the waiting sync, since 2026-09-19: a decode's synchronisations are answered
  // without a wait (the shim flushes and returns when nothing outstanding needs the engine - 15,431 of 15,431 in a
  // tg256 run), so with the publisher only on the waiting path every reading of clocks, power and busy this driver
  // had ever produced was from an idle moment between steps. Antonio's ear caught it: the fans never spin up under a
  // model. The rate limit inside publish_sensors (four a second) bounds the cost to one read of the firmware's own
  // utilisation block; it is not a control call.
  publish_sensors(s->dev);
  return WITH_LOCK(tinynv_exec_flush(&s->dev->exec)) ? TINYNV_ERR_DRIVER : TINYNV_OK;
}

tinynv_status_t tinynv_stream_sync(tinynv_stream_t s) {
  if (!ON_GPU(s)) return TINYNV_OK;
  NEED_GPU(s);
  publish_sensors(s->dev);
  // exec_idle, not a bare wait on the timeline: launches may be built and not yet handed over, and those carry timeline
  // values nothing has been told to release. Waiting on one without flushing first waits out the whole timeout.
  if (WITH_LOCK(tinynv_exec_idle(&s->dev->exec))) return TINYNV_ERR_DRIVER;
  // The engine is idle now and the host is about to go away and think. TINYNV_KEEPALIVE: keep it scheduled meanwhile.
  if (s->dev->exec.keepalive && s->dev->keepalive_kernel && !s->dev->torn_down) keepalive_launch(s);
  return TINYNV_OK;
}

tinynv_status_t tinynv_event_create(tinynv_device_t d, tinynv_event_t *out) {
  tinynv_event_t e = calloc(1, sizeof(*e));
  e->dev = d;
  *out = e;
  return TINYNV_OK;
}
tinynv_status_t tinynv_event_record(tinynv_event_t e, tinynv_stream_t s) {
  if (!e || !s) return TINYNV_ERR_INVALID;
  // The value the engine will have released once everything submitted so far is done. On the null device there is no
  // engine and the submission count is all there is to remember.
  //
  // Recording has to hand over any chain still being built first. An event names a point in the stream, and a launch
  // that has been built but not submitted is already behind that point as far as the caller is concerned - but its
  // timeline value will not be released until something hands it over, so an event recorded without flushing names a
  // value that may never arrive.
  if (s->dev->has_pci && s->dev->booted) {
    if (WITH_LOCK(tinynv_exec_flush(&s->dev->exec))) return TINYNV_ERR_DRIVER;
    e->value = s->dev->exec.timeline;
  } else {
    e->value = s->submitted;
  }
  return TINYNV_OK;
}
tinynv_status_t tinynv_event_sync(tinynv_event_t e) {
  if (!e || !e->dev->has_pci || !e->dev->booted) return TINYNV_OK;
  return WITH_LOCK(tinynv_exec_wait(&e->dev->exec, e->value, 30.0)) ? TINYNV_ERR_DRIVER : TINYNV_OK;
}
// Wait until the work an event was recorded after has finished.
//
// ggml's scheduler calls this on every llama.cpp run, not only across several GPUs: token embeddings live on the CPU
// backend, so every graph has cross-backend copies with an event recorded on one side and waited on the other. Refusing
// here aborts the run before it starts, which is why this is a real wait rather than the error it used to be.
//
// One in-order queue, and everything submitted to it is already waited on, so waiting for the event's timeline value is
// both correct now and still correct if a second queue ever appears - at which point this should become a semaphore
// acquire in the queue rather than a spin here, and the value it waits on is already the right one.
tinynv_status_t tinynv_stream_wait_event(tinynv_stream_t s, tinynv_event_t e) {
  if (!ON_GPU(s) || !e) return TINYNV_OK;
  NEED_GPU(s);
  return WITH_LOCK(tinynv_exec_wait(&s->dev->exec, e->value, 30.0)) ? TINYNV_ERR_DRIVER : TINYNV_OK;
}

tinynv_status_t tinynv_launch(tinynv_stream_t s, tinynv_kernel_t k, unsigned gx, unsigned gy, unsigned gz,
                              unsigned bx, unsigned by, unsigned bz, unsigned dyn_smem, const void *params, size_t params_len) {
  if (!k || !s) return TINYNV_ERR_INVALID;
  const tinynv_kernel_desc_t *d = k->desc;
  if (params_len > d->param_size)
    return tinynv_fail("launch %s: %zu bytes of parameters, the cubin declares %u", d->name, params_len, d->param_size), TINYNV_ERR_INVALID;
  // the cubin's own __launch_bounds__ when it declares one, else the hardware ceiling: a 256-bound kernel run at 512 faults
  unsigned threads = bx * by * bz, bound = d->max_threads ? d->max_threads : 1024;
  if (threads > bound)
    return tinynv_fail("launch %s: %u threads per block, the kernel allows %u", d->name, threads, bound), TINYNV_ERR_LAUNCH;
  s->submitted++;
  if (s->dev->has_pci) {
    if (!s->dev->booted) NEED_GPU(s);   // the boot check takes the lock; once booted, a flag says so
    tinynv_device_t dev = s->dev;
    pthread_mutex_lock(&g_lock);
    // The module's image goes across once, on the first launch out of it rather than at load: a ggml cubin is megabytes
    // of code and a run touches a fraction of it, so loading a module should not be a transfer.
    if (!k->mod->is_loaded) {
      if (tinynv_exec_load(&dev->exec, &k->mod->cubin, d->name, &k->mod->loaded)) {
        pthread_mutex_unlock(&g_lock);
        return TINYNV_ERR_DRIVER;
      }
      k->mod->is_loaded = 1;
    }
    // and this kernel's own offsets into it, which the first kernel already has from the upload
    if (!k->has_layout) {
      if (tinynv_cubin_image(&k->mod->cubin, d->name, 0, &k->layout)) {
        pthread_mutex_unlock(&g_lock);
        return TINYNV_ERR_DRIVER;
      }
      k->has_layout = 1;
    }
    tinynv_exec_module_t m = k->mod->loaded;
    m.layout = k->layout;
    uint32_t grid[3] = {gx, gy, gz}, block[3] = {bx, by, bz};
    int rc = tinynv_exec_run(&dev->exec, &m, d, grid, block, dyn_smem, params, params_len);
    pthread_mutex_unlock(&g_lock);
    return rc ? TINYNV_ERR_LAUNCH : TINYNV_OK;
  }

  announce(s->dev);
  fprintf(stderr, "tinynv: would launch %s sm_%u grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u regs=%u, %zu bytes of parameters at c[0][%#x]\n",
          d->name, k->mod->cubin.sm_arch, gx, gy, gz, bx, by, bz, dyn_smem, d->regs, params_len, d->param_base);
  return TINYNV_OK;
}

const char *tinynv_status_str(tinynv_status_t s) {
  switch (s) {
    case TINYNV_OK: return "ok";
    case TINYNV_ERR_NODEV: return "no such device";
    case TINYNV_ERR_OOM: return "out of memory";
    case TINYNV_ERR_LAUNCH: return "bad launch";
    case TINYNV_ERR_INVALID: return "invalid argument";
    case TINYNV_ERR_DRIVER: return "the driver could not do it: tinynv_last_error() says what went wrong";
    case TINYNV_ERR_UNSUPPORTED: return "not supported here";
  }
  return "unknown";
}
