// Boots the chip against a recorded boot: no GPU, but every register read is answered with what the hardware really said
// and every write is checked against what the python driver really wrote.
//
// The stages run in the order the oracle runs them, and the report says how far the driver got. Where a stage is not
// ported yet the test declares the exact recorded lines it is skipping, and asserts that set does not change: an
// unported stage should be a line in the report, not something absorbed quietly by the matcher.
#include "gpu.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// No stage is unported any more: the two prefilled RPC records that used to be skipped are written by gsp.c, so the
// replay runs from the first register read to GSP-RM reporting itself up with nothing declared away.
#define GAP_OPS 0

// What the replay forgives, and how much of it.
//
// Every one is an answer the driver already holds: a page table entry the oracle consults three times where this driver
// consults it twice, because python's accessor re-reads the entry inside the test that decides how to decode it. None of
// them is the only read-back of a write - that is measured, not assumed, and asserted below, because under replay a
// surplus re-read and an unmade read-back look identical while on hardware the second is a lost check.
//
// Three things hold this down. The count, so growth is visible. The committed table of places in tolerated-reads.txt, so
// composition is reviewed and not just the total: two places can trade forgiven reads and still sum to the same number.
// And the categorical assertion that every one is on the video memory window, so the rule cannot quietly start covering
// something it was not designed for.
//
// This driver does not add a third read to make the count zero. It would be a redundant round trip across the bus on the
// path every mapping takes, written only to flatter a report.
#define TOLERATED_READS 1069

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

// A driver that skips a recorded write must not pass. The recording opens with a read of the write-protect register,
// then a config write enabling bus mastering; do the reads and omit the write.
static void checker_selftest(const char *prefix) {
  tinynv_pci_t neg;
  if (tinynv_pci_open_replay(prefix, &neg)) return;
  tinynv_mmio_t m;
  neg.bar_map(&neg, 0, 0, 0, &m);
  nv_rd32(&m, 0x1fa828);
  neg.cfg_read(&neg, 0x04, 2);
  nv_rd32(&m, 0x0); // jumps the recorded config write
  tinynv_replay_stats_t ns;
  tinynv_replay_stats(&neg, &ns);
  printf("checker self-test: omitting one recorded write gives %zu divergences, %zu missed writes -> %s\n",
         ns.divergences, ns.skipped_writes, ns.divergences && ns.skipped_writes ? "refused, good" : "ACCEPTED, the checker is blind");
  CHECK(ns.divergences && ns.skipped_writes, "the checker tolerated an omitted write");
  neg.close(&neg);

  // and the same again on the path that is matched by order rather than by address. the recording zeroes the root page
  // table and then asks for its first host buffer; bring the chip up, skip the zeroing, and ask for the buffer.
  if (tinynv_pci_open_replay(prefix, &neg)) return;
  tinynv_dev_t d2;
  if (!tinynv_dev_early_init(&d2, &neg) && !tinynv_dev_mmu_init(&d2)) {
    tinynv_dma_t d;
    neg.dma_alloc(&neg, 0x50, &d);
    tinynv_replay_stats(&neg, &ns);
    printf("checker self-test: allocating over an omitted write gives %zu divergences, %zu missed writes -> %s\n",
           ns.divergences, ns.skipped_writes, ns.divergences && ns.skipped_writes ? "refused, good" : "ACCEPTED, the checker is blind");
    CHECK(ns.divergences && ns.skipped_writes, "an allocation stepped over a recorded write for free");
  }
  neg.close(&neg);

  // and the auditor that would have caught both of those the moment they happened, rather than when a total was compared
  // by hand. move the cursor without accounting for it and the books must refuse to balance.
  if (tinynv_pci_open_replay(prefix, &neg)) return;
  tinynv_mmio_t m2;
  neg.bar_map(&neg, 0, 0, 0, &m2);
  nv_rd32(&m2, 0x1fa828);
  tinynv_replay_break_books_for_test(&neg);
  nv_rd32(&m2, 0x0);
  tinynv_replay_stats(&neg, &ns);
  printf("checker self-test: unaccounted progress gives %zu divergences -> %s\n", ns.divergences,
         ns.divergences ? "refused, good" : "ACCEPTED, the books do not have to balance");
  CHECK(ns.divergences, "the cursor moved without accounting and nothing noticed");
  neg.close(&neg);
}

// The recording this test's expectations are written against. Everything below - the operation count, the addresses, the
// byte-for-byte comparison of the two records built from structures we cannot assert against NVIDIA's headers - is a
// statement about this particular boot. Re-record the trace and the reference moves, so the recording is pinned like any
// other dependency, and replacing it has to be deliberate.
#define TRACE_SHA "f031b84cb4fa180cbec1315a9203cd5f46efacd5fc1ae22281d9defda7543caf"

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <trace prefix>\n", argv[0]); return 2; }

  char fingerprint[65] = "";
  if (tinynv_replay_fingerprint(argv[1], fingerprint)) { fprintf(stderr, "%s\n", tinynv_last_error()); return 1; }
  printf("recording: %s\n", fingerprint);
  if (strcmp(fingerprint, TRACE_SHA)) {
    printf("  this is not the recording these expectations were written against (%s).\n", TRACE_SHA);
    printf("  re-point the test at the old one, or set TINYNV_ALLOW_NEW_TRACE=1 and re-check every number below by hand.\n");
    CHECK(getenv("TINYNV_ALLOW_NEW_TRACE"), "the recording changed and the expectations were not re-checked");
  }

  tinynv_pci_t pci;
  if (tinynv_pci_open_replay(argv[1], &pci)) { fprintf(stderr, "replay: %s\n", tinynv_last_error()); return 1; }

  tinynv_gpu_t g;
  int rc = tinynv_gpu_open(&g, &pci);
  if (rc) printf("bring-up failed: %s\n", tinynv_last_error());
  else {
    printf("chip %s (arch %#x impl %u, boot0 %#x), %s boot, mmu v%d%s\n", g.dev.chip_name, g.dev.architecture,
           g.dev.implementation, g.dev.chip_id, g.dev.fmc_boot ? "chain of trust" : "vbios", g.dev.mmu_ver,
           g.dev.wpr2_was_up ? ", chip was reset on arrival" : "");
    printf("vram %llu MB, bar1 window %llu MB (%s bar)\n", (unsigned long long)(g.dev.vram_size >> 20),
           (unsigned long long)(g.dev.vram.size >> 20), g.dev.large_bar ? "large" : "small");
    printf("memory: boot %llu MB, page tables %llu MB at %#llx, the rest %llu MB at %#llx; root table at %#llx\n",
           (unsigned long long)(g.mm.boot.size >> 20), (unsigned long long)(g.mm.ptable.size >> 20),
           (unsigned long long)g.mm.ptable.base, (unsigned long long)(g.mm.pa.size >> 20),
           (unsigned long long)g.mm.pa.base, (unsigned long long)g.mm.root_page_table);
  }

  // Without firmware this test can only reach bring-up: 12 operations against the 456,618 a whole boot records,
  // and it must NOT call that a pass. The floor below is the number that enforces it; this comment used to name a
  // count from an early recording and went stale silently, which is the failure mode a floor in a comment always has.
  // Skipping is opt-in and loud, because a green run that tested nothing is worse than a red one that says why. Session A
  // ran this binary from the wrong directory, got 12 operations and "all checks passed", and was one careless read away
  // from reporting the chain of trust as verified.
  tinynv_blob_t probe;
  int have_fw = !rc && !tinynv_fw_load(g.dev.fw_name, "fmc-" TINYNV_FW_VER ".bin", NULL, &probe);
  int allow_no_fw = getenv("TINYNV_ALLOW_NO_FIRMWARE") != NULL;
  if (have_fw) tinynv_fw_free(&probe);
  else if (!rc) {
    printf("no firmware under %s: %s\n", *tinynv_fw_root() ? tinynv_fw_root() : "(nowhere configured)", tinynv_last_error());
    printf("  this run stops after bring-up and covers 12 of the operations the firmware boot needs.\n");
    printf("  run tools/fetch_firmware.sh, or set TINYNV_ALLOW_NO_FIRMWARE=1 to accept the reduced run deliberately.\n");
    CHECK(allow_no_fw, "firmware is missing and the reduced run was not asked for");
  }

  // the software init of both firmware blocks: allocations and firmware images, no hardware
  int sw = have_fw ? tinynv_gpu_init_sw(&g) : -1;
  if (sw && have_fw) printf("software init: %s\n", tinynv_last_error());
  else if (!sw) {
    printf("firmware: fmc %zu KB, gsp-rm %zu MB under a %zu page radix tree, riscv bootloader %zu KB (code %#x data %#x manifest %#x)\n",
           g.flcn.fmc_image.size >> 10, g.gsp.gsp_image_size >> 20, g.gsp.radix3.naddrs, g.gsp.bootloader_size >> 10,
           g.gsp.bl_desc.monitorCodeOffset, g.gsp.bl_desc.monitorDataOffset, g.gsp.bl_desc.manifestOffset);
    printf("placed: boot args at %#llx, fmc at %#llx, libos args at %#llx, wpr metadata at %#llx\n",
           (unsigned long long)g.flcn.boot_args_sysmem, (unsigned long long)g.flcn.fmc_sysmem,
           (unsigned long long)g.gsp.libos_args_sysmem, (unsigned long long)g.gsp.wpr_meta_sysmem);
  }

  // starting the firmware: this one does touch hardware, which under replay means the recorded hardware
  int cot = sw ? 1 : tinynv_flcn_init_hw(&g);
  if (!cot) printf("chain of trust accepted: the gsp falcon has left boot lockdown\n");
  else if (!sw) printf("chain of trust: %s\n", tinynv_last_error());

  int hw = cot ? 1 : tinynv_gsp_init_hw(&g);
  if (!hw) printf("gsp-rm is up: it answered on the status queue and the memory windows are retargeted\n");
  if (!hw) {
    printf("register sequences gsp-rm asked the driver to run: %u\n", g.gsp.cpu_seq_requests);
    // the recording contains none, so the sequencer is unexercised code; if that ever changes, the number says so
    CHECK(g.gsp.cpu_seq_requests == 0, "the recording requested %u register sequences, which it did not before",
          g.gsp.cpu_seq_requests);
  }
  else if (!cot) printf("gsp-rm: %s\n", tinynv_last_error());

  int obj = hw ? 1 : tinynv_gsp_init_objects(&g);
  if (!obj)
    printf("object tree: client %#x, device %#x, subdevice %#x, address space %#x; %u engines reported\n",
           g.gsp.priv_root, g.gsp.device, g.gsp.subdevice, g.gsp.vaspace, g.gsp.nengines);
  else if (!hw) printf("object tree: %s\n", tinynv_last_error());

  int chan = obj ? 1 : tinynv_gsp_init_channel(&g);
  if (!chan) printf("channel %#x on runlist %u: ring at virtual %#llx (physical %#llx), state at %#llx\n", g.gsp.channel,
                    g.gsp.channel_runlist, (unsigned long long)g.gsp.gpfifo.va,
                    (unsigned long long)g.gsp.gpfifo.ranges[0].paddr, (unsigned long long)g.gsp.ramfc.ranges[0].paddr);
  else if (!obj) printf("channel: %s\n", tinynv_last_error());

  int grc = chan ? 1 : tinynv_gsp_init_gr_context(&g);
  if (!grc) printf("graphics context: %d buffers promoted, compute object %#x, copy object %#x\n", g.gsp.ngrctx,
                   g.gsp.compute_obj, g.gsp.dma_copy_obj);
  else if (!chan) printf("graphics context: %s\n", tinynv_last_error());

  int cli = grc ? 1 : tinynv_gsp_open_client(&g);
  if (!cli) printf("client %#x: device %#x, subdevice %#x, virtual memory %#x, address space %#x, group %#x, share %#x\n",
                   g.gsp.user_root, g.gsp.user_device, g.gsp.user_subdevice, g.gsp.user_virtmem, g.gsp.user_vaspace,
                   g.gsp.user_group, g.gsp.user_ctxshare),
         printf("die: up to %u partitions of %u groups of %u cores (%u at most, not the number enabled), %u warps each,"
                " instruction set %#x\n", g.gsp.max_gpcs, g.gsp.max_tpc_per_gpc, g.gsp.max_sm_per_tpc,
                g.gsp.max_gpcs * g.gsp.max_tpc_per_gpc * g.gsp.max_sm_per_tpc, g.gsp.max_warps_per_sm, g.gsp.sm_version);
  else if (!grc) printf("client: %s\n", tinynv_last_error());

  int q = cli ? 1 : tinynv_gsp_init_queues(&g);
  if (!q) printf("queues: compute channel %#x token %#x, copy channel %#x token %#x, rings at %#llx\n",
                 g.gsp.compute_q.channel, g.gsp.compute_q.token, g.gsp.copy_q.channel, g.gsp.copy_q.token,
                 (unsigned long long)g.gsp.fifo_mem.va);
  else if (!cli) printf("queues: %s\n", tinynv_last_error());

  checker_selftest(argv[1]);

  tinynv_replay_stats_t s;
  tinynv_replay_stats(&pci, &s);
  // the headline is how far into the recording the boot reached, in recorded operations. the serving count is larger than
  // the recording because one coalesced poll is served many times, which reads as nonsense in a headline.
  printf("replay (%s): %zu of %zu recorded operations reproduced, %zu tolerated, %zu divergences\n",
         tinynv_is_faithful() ? "faithful" : "fast", s.ops_done, s.total, s.skipped, s.divergences);
  printf("  %zu reads and writes issued in total, the difference being polls the recording coalesced\n", s.consumed);
  if (s.skipped) {
    printf("  %zu reads forgiven, each a place and value the driver already held, never a place left unread\n", s.skipped);
    printf("  of those, %zu were the only read-back of a write (a check lost on hardware, not just a repeat)\n",
           s.forgiven_readbacks);
    // grouped by place as well as listed by line, so a new site shows up as a new place rather than as a bigger number
    for (size_t i = 0; i < s.nforgiven && i < 64; i++) {
      size_t first = i, n = 0;
      for (size_t j = 0; j < i; j++)
        if (s.forgiven_res[j] == s.forgiven_res[i] && s.forgiven_off[j] == s.forgiven_off[i] &&
            s.forgiven_is_bar_query[j] == s.forgiven_is_bar_query[i]) { first = j; break; }
      if (first != i) continue;
      if (s.forgiven_is_bar_query[i]) printf("    where bar %d is, asked again, lines", s.forgiven_res[i]);
      else printf("    bar:%d at %#llx, lines", s.forgiven_res[i], (unsigned long long)s.forgiven_off[i]);
      for (size_t j = i; j < s.nforgiven && j < 64; j++)
        if (s.forgiven_res[j] == s.forgiven_res[i] && s.forgiven_off[j] == s.forgiven_off[i] &&
            s.forgiven_is_bar_query[j] == s.forgiven_is_bar_query[i]) { printf(" %d", s.forgiven[j]); n++; }
      printf(" (%zu)\n", n);
    }
  }
  if (s.divergences) printf("  first divergence: %s\n", s.first);
  if (s.skipped_writes) printf("  %zu recorded writes were never issued\n", s.skipped_writes);
  printf("declared gaps: %zu covering %zu operations (unported stages, reported not hidden)\n", s.ngaps, s.gap_ops);
  printf("work submission: %s\n", s.submission_recorded
         ? "recorded, so this trace is evidence about launching as well as booting"
         : "NOT in this recording - the ring, the write pointer and the doorbell bypass the recorder on a remote device,"
           " so nothing here can verify a launch");
  printf("accounting: cursor %zu = %zu finished + %zu stepped over + %zu gapped\n", s.cursor, s.ops_done, s.skipped, s.gap_ops);
  CHECK(s.cursor == s.ops_done + s.skipped + s.gap_ops, "the replay's books do not balance");

  // The composition of what was forgiven, against a committed table. Regenerating is deliberate and explicit, so the
  // table changes only when someone means it to, and the change is then a diff someone can read.
  {
    const char *golden = "test/tolerated-reads.txt";
    if (tinynv_is_faithful()) { /* nothing is forgiven in faithful mode, so there is no composition to compare */ }
    else if (getenv("TINYNV_REGENERATE_TOLERATED")) {
      FILE *f = fopen(golden, "w");
      if (f) { tinynv_replay_forgiven_table(&pci, f); fclose(f); printf("regenerated %s\n", golden); }
      else CHECK(0, "cannot write %s", golden);
    } else {
      char *buf = NULL;
      size_t len = 0;
      FILE *mem = open_memstream(&buf, &len);
      if (mem) {
        tinynv_replay_forgiven_table(&pci, mem);
        fclose(mem);
        FILE *f = fopen(golden, "r");
        if (!f) CHECK(0, "%s is missing; run with TINYNV_REGENERATE_TOLERATED=1 once and read the diff", golden);
        else {
          char *want = calloc(1, len + 4096);
          size_t got = fread(want, 1, len + 4095, f);
          fclose(f);
          CHECK(got == len && !memcmp(want, buf, len),
                "the places where reads are forgiven have changed; run with TINYNV_REGENERATE_TOLERATED=1 and read the diff");
          free(want);
        }
        free(buf);
      }
    }
  }

  CHECK(!rc, "bring-up failed");
  CHECK(!s.divergences, "%zu divergences against the recording", s.divergences);
  CHECK(!s.skipped_writes, "%zu recorded writes were never issued", s.skipped_writes);
  int faithful = tinynv_is_faithful();
  if (have_fw && faithful) {
    // nothing forgiven at all: this is the mode the first hardware runs use, so the card sees a sequence it has already
    // accepted, with no operation missing from it
    CHECK(s.skipped == 0, "faithful mode forgave %zu reads; it is supposed to make every one the oracle makes", s.skipped);
    CHECK(s.forgiven_readbacks == 0, "faithful mode forgave %zu read-backs", s.forgiven_readbacks);
  }
  if (have_fw) {
    CHECK(!sw, "software init failed");
    CHECK(!cot, "the chain of trust did not complete");
    CHECK(!hw, "gsp-rm did not report itself up");
    CHECK(!obj, "the object tree was not created");
    CHECK(g.gsp.nengines > 0, "gsp-rm reported no engines");
    CHECK(!chan, "the channel was not created");
    CHECK(!grc, "the graphics context was not set up");
    CHECK(!cli, "the working client was not opened");
    CHECK(!q, "the work channels were not created");
    // nothing is declared away any more, and nothing may be added back without saying so here
    CHECK(s.ngaps == 0, "expected no declared gaps, got %zu", s.ngaps);
    if (!faithful)
      CHECK(s.skipped == (size_t)TOLERATED_READS, "%zu reads were forgiven, not the %d the page table walks account for",
            s.skipped, TOLERATED_READS);
    // all of them on the video memory window, which is what the rule was designed to cover and nothing else
    for (size_t i = 0; i < s.nforgiven && i < 64; i++)
      CHECK(s.forgiven_res[i] == 1, "an operation was forgiven on bar %d, not the video memory window", s.forgiven_res[i]);
    // and none of them the only read-back of a write: that would be a check this driver does not make on hardware
    CHECK(s.forgiven_readbacks == 0, "%zu forgiven reads were the only read-back of a write to that place",
          s.forgiven_readbacks);
    CHECK(s.gap_ops == GAP_OPS, "declared gaps cover %zu operations, not %d", s.gap_ops, GAP_OPS);
  } else {
    CHECK(s.ngaps == 0, "a run that never got to the gap declared %zu", s.ngaps);
  }
  // a run that matched almost nothing must not pass either
  size_t floor = have_fw ? 286000 : 12;
  CHECK(s.ops_done >= floor, "only %zu recorded operations were reproduced, expected at least %zu", s.ops_done, floor);

  if (!rc) tinynv_gpu_close(&g);
  pci.close(&pci);
  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
