// Chip bring-up: identify the GPU, put it in a known state, and work out which boot path it needs.
//
// This mirrors tinygrad's NVDev._early_ip_init and _early_mmu_init operation for operation, because the python driver is
// the oracle: run this against a recorded boot and any difference is a difference in logic. Two architectures matter:
// Ampere (GA10x) boots its falcon from the VBIOS, Blackwell (GB20x) through the FSP chain of trust, and they differ in
// MMU version too. Everything here is register access plus config space, so it works over any of the PCI backends.
#include "dev.h"
#include "internal.h"
#include <stdlib.h>
#include "nv_regs.h"
#include "nv_structs.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#define PCI_COMMAND 0x04
#define PCI_COMMAND_MEMORY 0x2
#define PCI_COMMAND_MASTER 0x4

static void sleep_ms(int ms) {
  struct timespec ts = {.tv_sec = ms / 1000, .tv_nsec = (long)(ms % 1000) * 1000000L};
  nanosleep(&ts, NULL);
}

// write a config value and read it back, so the write has landed before anything depends on it
static void cfg_write_flush(tinynv_dev_t *d, uint32_t off, uint32_t val, uint32_t size) {
  d->pci->cfg_write(d->pci, off, size, val);
  d->pci->cfg_read(d->pci, off, size);
}

uint32_t tinynv_rd32(tinynv_dev_t *d, uint64_t off) { return nv_rd32(&d->mmio, off); }
void tinynv_wr32(tinynv_dev_t *d, uint64_t off, uint32_t v) { nv_wr32(&d->mmio, off, v); }

// poll until a register satisfies a predicate. a boot that stalls here is a hardware state problem, not a logic one,
// so the message says which register and what it read rather than just timing out.
int tinynv_wait_reg(tinynv_dev_t *d, uint64_t off, uint32_t mask, uint32_t want, int timeout_ms, const char *what) {
  double deadline = tinynv_now_s() + timeout_ms / 1000.0;
  uint32_t v;
  do {
    if (((v = tinynv_rd32(d, off)) & mask) == want) return 0;
  } while (tinynv_now_s() < deadline);
  return tinynv_fail("%s: register %#llx read %#x (wanted %#x under %#x) after %d ms",
                     what, (unsigned long long)off, v, want, mask, timeout_ms);
}

// Ampere's power-on firmware signals it has finished through the always-on scratch group rather than the thermal one,
// and it takes TWO reads: the privilege mask on the group has to have dropped to level 0, and then the scratch's own low
// byte has to read 0xff. The ORDER is not incidental. The oracle tests the mask first and reads the scratch only once
// that has passed, so a recording taken from it contains no scratch read before the mask has dropped, and a driver that
// read them the other way round would diverge on the first poll of a replay while being perfectly correct on hardware.
static int wait_gfw_ampere(tinynv_dev_t *d, int timeout_ms) {
  double deadline = tinynv_now_s() + timeout_ms / 1000.0;
  uint32_t plm = 0, scratch = 0;
  do {
    plm = tinynv_rd32(d, NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK);
    if (!NV_GET(plm, NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK, READ_PROTECTION_LEVEL0)) continue;
    if (((scratch = tinynv_rd32(d, NV_PGC6_AON_SECURE_SCRATCH_GROUP_05(0))) & 0xff) == 0xff) return 0;
  } while (tinynv_now_s() < deadline);
  return tinynv_fail("the boot firmware did not finish in %d ms (privilege mask %#x, scratch %#x)", timeout_ms, plm, scratch);
}

static const char *arch_prefix(uint32_t arch) {
  switch (arch) {
    case 0x17: return "GA1";
    case 0x19: return "AD1";
    case 0x1b: return "GB2";
    default: return NULL;
  }
}

static const char *fw_dir(const char *chip_name) {
  if (!strncmp(chip_name, "GB2", 3)) return "gb202";
  if (!strncmp(chip_name, "AD1", 3)) return "ad102";
  if (!strncmp(chip_name, "GA1", 3)) return "ga102";
  return NULL;
}

// Reset a card that still has a firmware on it, and wait for it to be a card again.
//
// This used to be a reset and a 100 ms sleep, and on a 3060 over thunderbolt that was the beginning of every failed
// afternoon: the recorded boot never resets anything (the oracle opened a fresh card), so nothing on this path had
// ever been checked against anything. What tools/nv_e3_flr.py established by hand is done here: the config space is
// polled until the card answers to its own name rather than assumed back after a fixed sleep; the BARs are compared
// with what they were and put back if the reset cleared them; and the caller then checks that the region actually
// came down, because a reset that leaves it up is a reset that did nothing, and booting over it is what produced a
// firmware that came up, ran its register sequence and went silent - three times in one afternoon (2026-09-17).
static int warm_reset(tinynv_dev_t *d, tinynv_pci_t *pci, uint32_t wpr2_hi) {
  fprintf(stderr, "tinynv: a firmware is resident (write-protected region hi %#x) - the last run did not unload it. "
                  "Resetting the card\n", wpr2_hi);
  uint32_t vd = pci->cfg_read(pci, 0x00, 4), bars[TINYNV_MAX_BARS];
  for (int i = 0; i < TINYNV_MAX_BARS; i++) bars[i] = pci->cfg_read(pci, 0x10 + 4 * i, 4);

  uint32_t cmd = pci->cfg_read(pci, PCI_COMMAND, 2);
  cfg_write_flush(d, PCI_COMMAND, cmd & ~PCI_COMMAND_MASTER, 2);
  double t0 = tinynv_now_s();
  if (pci->reset(pci)) return -1;

  // back when it answers to its name. All ones is the function still in reset; anything else that is not the name is
  // a card that came back as something else, which is not a state to continue from
  uint32_t now = 0xffffffff;
  double deadline = tinynv_now_s() + 10.0;
  while ((now = pci->cfg_read(pci, 0x00, 4)) != vd && tinynv_now_s() < deadline) sleep_ms(10);
  if (now != vd) return tinynv_fail("the card did not come back from its reset in 10 s: vendor/device reads %#x, was %#x", now, vd);
  double back = tinynv_now_s() - t0;

  int restored = 0;
  for (int i = 0; i < TINYNV_MAX_BARS; i++) {
    if (pci->cfg_read(pci, 0x10 + 4 * i, 4) == bars[i]) continue;
    pci->cfg_write(pci, 0x10 + 4 * i, 4, bars[i]);
    restored++;
  }
  fprintf(stderr, "tinynv: the card answered %.2f s after the reset%s\n", back,
          restored ? " and its bars had been cleared; put back from the snapshot" : "");
  return 0;
}

int tinynv_dev_early_init(tinynv_dev_t *d, tinynv_pci_t *pci) {
  memset(d, 0, sizeof(*d));
  d->pci = pci;
  if (pci->bar_map(pci, 0, 0, 0, &d->mmio)) return -1;

  // a GPU whose write-protected region is still up has a live firmware on it from a previous run: reset it, but stop it
  // mastering the bus first, or it keeps writing into host memory that is about to be unmapped underneath it
  // ALL ONES IS NOT A FIRMWARE. This read is the first thing the driver does, and if the memory window is not decoding
  // it answers 0xffffffff - which is non-zero, so it used to read as "a firmware is resident", which asked for a reset,
  // which clears the window again. A loop that sustains itself: every open after the first one reset a healthy card and
  // then found it unreadable, and the only thing that ever broke it was a physical replug, because re-enumeration is
  // what reprograms the command register. Cost most of a day on a 3060 and was blamed on the enclosure's power supply.
  uint32_t wpr2 = tinynv_rd32(d, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
  d->wpr2_was_up = wpr2 != 0 && wpr2 != 0xffffffff;
  if (d->wpr2_was_up && warm_reset(d, pci, wpr2)) return -1;

  // Bus mastering AND the memory window. The window was the backend's business, settled when it opened the device - but
  // the reset above happens after that, and a reset clears the command register outright. So on the one path where this
  // matters nobody was restoring it, and the driver went on to read a card whose window was off.
  //
  // This cannot diverge from the recording, which is why it is safe to widen: the recorded boot reads 0x7 here and
  // writes 0x7, and 0x7 with these two bits set is still 0x7. It only does anything on a register the oracle never saw,
  // because the oracle never opened a card it had just reset.
  uint32_t cmd = pci->cfg_read(pci, PCI_COMMAND, 2);
  cfg_write_flush(d, PCI_COMMAND, cmd | PCI_COMMAND_MASTER | PCI_COMMAND_MEMORY, 2);

  // THE FIRST READING WAS MADE THROUGH A WINDOW THAT MAY HAVE BEEN OFF. If it answered all ones, it said nothing about
  // whether a firmware is resident - and a card left with its window switched off is exactly the state a previous run
  // that could not finish leaves behind (see gsp.c's init_wait_report). So ask again now that the window is on. This
  // cannot diverge from the recording: there, the first read answers zero and this never runs.
  if (wpr2 == 0xffffffff) {
    wpr2 = tinynv_rd32(d, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
    if (wpr2 && wpr2 != 0xffffffff) {
      fprintf(stderr, "tinynv: with the memory window back on, the write-protected region reads %#x: a firmware IS "
                      "resident and the first reading could not see it\n", wpr2);
      d->wpr2_was_up = 1;
      if (warm_reset(d, pci, wpr2)) return -1;
      uint32_t cmd2 = pci->cfg_read(pci, PCI_COMMAND, 2);
      cfg_write_flush(d, PCI_COMMAND, cmd2 | PCI_COMMAND_MASTER | PCI_COMMAND_MEMORY, 2);
    }
  }

  // Two impossible answers, not one. All ones is a window that is not decoding or a card that has gone; ZERO is a chip
  // that is present and not answering - halted, still in reset, or not yet back from having been put down at the end of
  // a previous run. Only the first was checked, so the second fell through to the architecture table and came out as
  // "architecture 0 is not one this driver knows", which reads like an unsupported GPU and sent a day's debugging after
  // the wrong thing. A real NV_PMC_BOOT_0 is never either value.
  d->chip_id = tinynv_rd32(d, NV_PMC_BOOT_0);
  if (d->chip_id == 0xffffffff)
    return tinynv_fail("the gpu reads all ones: it has fallen off the bus, or its memory window is not enabled");
  if (d->chip_id == 0)
    return tinynv_fail("the gpu reads zero: it is on the bus but not answering - halted, in reset, or not yet back from "
                       "being put down. A replug re-enumerates it; tools/nv_quiesce.sh does not wake it.");

  // WHAT GOOD LOOKS LIKE, remembered while it is still true. On this machine a card can lose its command register and
  // all six base address registers WHILE A FIRMWARE IS RUNNING ON IT (2026-09-21, every second boot in an enumeration):
  // config space stays readable, the card stays on the bus, macOS logs nothing, and the windows are simply gone. With
  // the windows gone every register reads all ones and every page table entry written through them is lost. macOS will
  // not reassign them - only re-enumeration does, which means a person and a cable - but the values it assigned are
  // right here in config space, so the driver can write them back itself. Live cards only: these six reads are
  // operations the recorded boot never made.
  if (pci->live) {
    d->cfg_cmd = pci->cfg_read(pci, PCI_COMMAND, 2);
    for (int i = 0; i < 6; i++) d->cfg_bars[i] = pci->cfg_read(pci, 0x10 + 4 * i, 4);
    d->cfg_saved = 1;
    // The PCI Express capability, for the link registers. Walked the standard way: the pointer at 0x34, then each
    // capability's id and next pointer, stopping at id 0x10 or at the end of the list.
    uint32_t cap = pci->cfg_read(pci, 0x34, 1) & 0xfc;
    for (int guard = 0; cap && cap != 0xfc && guard < 48; guard++) {
      uint32_t id = pci->cfg_read(pci, cap, 1);
      if (id == 0x10) { d->pcie_cap = cap; break; }
      cap = pci->cfg_read(pci, cap + 1, 1) & 0xfc;
    }
    unsigned gen = 0, width = 0;
    tinynv_dev_link(d, &gen, &width);
    // TINYNV_LINK_TARGET=<gen> writes the endpoint's target link speed (Link Control 2, bits 3:0), which bounds the
    // speed the next retrain settles at - the retrain itself is the upstream port's, or the firmware's, to start.
    const char *t = getenv("TINYNV_LINK_TARGET");
    if (t && *t && d->pcie_cap) {
      uint32_t lc2 = pci->cfg_read(pci, d->pcie_cap + 0x30, 2), want = (uint32_t)atoi(t) & 0xf;
      pci->cfg_write(pci, d->pcie_cap + 0x30, 2, (lc2 & ~0xfu) | want);
      fprintf(stderr, "tinynv: link control 2 target speed set to gen%u (was gen%u)\n", want, lc2 & 0xf);
    }
    fprintf(stderr, "tinynv: pcie link x%u at gen%u (%s), target gen%u\n", width, gen,
            gen == 1 ? "2.5 GT/s" : gen == 2 ? "5 GT/s" : gen == 3 ? "8 GT/s" : gen == 4 ? "16 GT/s" : gen == 5 ? "32 GT/s" : "?",
            d->pcie_cap ? pci->cfg_read(pci, d->pcie_cap + 0x30, 2) & 0xf : 0);
  }

  uint32_t boot42 = tinynv_rd32(d, NV_PMC_BOOT_42);
  d->architecture = NV_GET(boot42, NV_PMC_BOOT_42, ARCHITECTURE);
  d->implementation = NV_GET(boot42, NV_PMC_BOOT_42, IMPLEMENTATION);

  const char *prefix = arch_prefix(d->architecture);
  if (!prefix) return tinynv_fail("architecture %#x is not one this driver knows (boot42 %#x)", d->architecture, boot42);
  snprintf(d->chip_name, sizeof(d->chip_name), "%s%02u", prefix, d->implementation);
  const char *fw = fw_dir(d->chip_name);
  if (!fw) return tinynv_fail("no firmware directory for %s", d->chip_name);
  snprintf(d->fw_name, sizeof(d->fw_name), "%s", fw);

  // Blackwell and later boot the falcon through the FSP chain of trust and use the third MMU generation
  d->fmc_boot = d->architecture >= 0x1a;
  d->mmu_ver = d->fmc_boot ? 3 : 2;

  // The channel, compute and copy classes the engines answer to. Ampere's are the base set, because that is how the
  // oracle spells it: Ada keeps Ampere's channel and copy classes and moves only compute, Blackwell moves all three.
  d->class_gpfifo = TINYNV_CLASS_GPFIFO_AMPERE;
  d->class_compute = TINYNV_CLASS_COMPUTE_AMPERE;
  d->class_dma_copy = TINYNV_CLASS_DMA_COPY_AMPERE;
  if (d->architecture == 0x19) d->class_compute = TINYNV_CLASS_COMPUTE_ADA;
  else if (d->architecture >= 0x1b) {
    d->class_gpfifo = TINYNV_CLASS_GPFIFO_BLACKWELL;
    d->class_compute = TINYNV_CLASS_COMPUTE_BLACKWELL;
    d->class_dma_copy = TINYNV_CLASS_DMA_COPY_BLACKWELL;
  }

  // The firmware that runs at power-on signals it has finished; until then the chip is not ready to be driven. Where it
  // signals differs by architecture, which is the whole of the difference here.
  //
  // The refusal for an architecture whose falcon boot is not implemented used to be on this line. It is in flcn.c now,
  // because that is where the gap is: identifying the chip and setting up its page tables are the same work whatever
  // boots the falcon, so they should run for any chip this driver can name, and a replay should get as far as the thing
  // that is actually missing rather than stopping three reads in.
  if (d->fmc_boot ? tinynv_wait_reg(d, NV_THERM_I2CS_SCRATCH, 0xffffffff, 0xff, 10000, "waiting for the boot firmware")
                  : wait_gfw_ampere(d, 10000)) return -1;

  // What the falcon boot may assume about the region. On the recorded path it is the read made on arrival, so nothing
  // new is read; after a reset it is read again, because changing it was the whole point of the reset. NVIDIA's driver
  // refuses to boot GSP-RM over a region that is up ("unexpected WPR2 already up, cannot proceed") and so does this,
  // here, where the reason is known, rather than sixty seconds later as a start-up notice that never comes.
  d->wpr2_hi_now = wpr2;
  if (d->wpr2_was_up) {
    d->wpr2_hi_now = tinynv_rd32(d, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
    if (d->wpr2_hi_now == 0xffffffff) return tinynv_fail("the gpu reads all ones after its reset: the memory window did not come back");
    if (d->wpr2_hi_now)
      return tinynv_fail("the reset did not tear the write-protected region down (hi still %#x): gsp-rm cannot be booted "
                         "over a live one, and NVIDIA's own driver refuses to try. A replug re-enumerates the card. To need "
                         "no reset at all, let a run unload the firmware on its way out (TINYNV_UNLOAD=1, the default)",
                         d->wpr2_hi_now);
    fprintf(stderr, "tinynv: the write-protected region is down after the reset; booting the card cold\n");
  }
  return 0;
}

// Have the windows gone, and if so put them back. The card answers config space either way, so this is cheap and safe
// to ask; the answer is only ever acted on when a base address register has actually been zeroed.
int tinynv_dev_restore_decode(tinynv_dev_t *d) {
  tinynv_pci_t *pci = d->pci;
  if (!d->cfg_saved) return 0;
  uint32_t vd = pci->cfg_read(pci, 0x00, 4);
  if (vd == 0xffffffffu || vd == 0) return 0; // not a card that is answering: nothing to put back into
  uint32_t cmd = pci->cfg_read(pci, PCI_COMMAND, 2);
  int lost = (cmd & (PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER)) != (PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER);
  for (int i = 0; i < 6; i++) lost |= pci->cfg_read(pci, 0x10 + 4 * i, 4) != d->cfg_bars[i];
  if (!lost) return 0;

  for (int i = 0; i < 6; i++) pci->cfg_write(pci, 0x10 + 4 * i, 4, d->cfg_bars[i]);
  cfg_write_flush(d, PCI_COMMAND, d->cfg_cmd | PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER, 2);

  int back = 1;
  for (int i = 0; i < 6; i++) back &= pci->cfg_read(pci, 0x10 + 4 * i, 4) == d->cfg_bars[i];
  uint32_t now = pci->cfg_read(pci, PCI_COMMAND, 2);
  d->windows_restored++;
  d->windows_restored_at = tinynv_now_s();
  unsigned lgen = 0, lw = 0;
  tinynv_dev_link(d, &lgen, &lw);
  fprintf(stderr, "tinynv: the card had lost its windows under a running firmware (command %#06x, first base address "
                  "%#x, link now x%u gen%u); written back from what it reported when it was opened - command %#06x, %s\n",
          cmd, d->cfg_bars[0], lw, lgen, now, back ? "all six base address registers read back correctly" : "THEY DID NOT TAKE");
  return 1;
}

void tinynv_dev_link(tinynv_dev_t *d, unsigned *gen, unsigned *width) {
  *gen = 0; *width = 0;
  if (!d->pcie_cap) return;
  uint32_t ls = d->pci->cfg_read(d->pci, d->pcie_cap + 0x12, 2);   // Link Status: speed 3:0, width 9:4
  if (ls == 0xffff) return;
  *gen = ls & 0xf; *width = (ls >> 4) & 0x3f;
}

int tinynv_dev_reset_cold(tinynv_dev_t *d) {
  tinynv_pci_t *pci = d->pci;
  if (!d->cfg_saved) return tinynv_fail("no configuration snapshot to put back after a reset");
  uint32_t vd = pci->cfg_read(pci, 0x00, 4);
  if (vd == 0xffffffffu || vd == 0) return tinynv_fail("the card does not answer in config space (%#x): a reset cannot reach it", vd);
  uint32_t cmd = pci->cfg_read(pci, PCI_COMMAND, 2);
  cfg_write_flush(d, PCI_COMMAND, cmd & ~PCI_COMMAND_MASTER, 2);
  double t0 = tinynv_now_s();
  if (pci->reset(pci)) return -1;
  uint32_t now = 0xffffffff;
  double deadline = tinynv_now_s() + 10.0;
  while ((now = pci->cfg_read(pci, 0x00, 4)) != vd && tinynv_now_s() < deadline) sleep_ms(10);
  if (now != vd) return tinynv_fail("the card did not come back from its reset in 10 s: vendor/device reads %#x, was %#x", now, vd);
  // the snapshot from the open, not from now: a reset clears the base address registers and this is the one copy
  for (int i = 0; i < 6; i++) pci->cfg_write(pci, 0x10 + 4 * i, 4, d->cfg_bars[i]);
  cfg_write_flush(d, PCI_COMMAND, d->cfg_cmd | PCI_COMMAND_MASTER | PCI_COMMAND_MEMORY, 2);
  uint32_t id = tinynv_rd32(d, NV_PMC_BOOT_0);
  if (id == 0xffffffffu || id == 0) return tinynv_fail("after the reset the chip id reads %#x: the windows did not come back", id);
  if (d->fmc_boot ? tinynv_wait_reg(d, NV_THERM_I2CS_SCRATCH, 0xffffffff, 0xff, 10000, "waiting for the boot firmware after the reset")
                  : wait_gfw_ampere(d, 10000)) return -1;
  uint32_t wpr2 = tinynv_rd32(d, NV_PFB_PRI_MMU_WPR2_ADDR_HI);
  if (wpr2 && wpr2 != 0xffffffffu)
    return tinynv_fail("the reset left the write-protected region up (hi %#x): the card is not cold", wpr2);
  d->wpr2_hi_now = 0;
  d->wpr2_was_up = 1;
  fprintf(stderr, "tinynv: the card was reset and is cold again %.2f s later: configuration put back, boot firmware done, region down\n",
          tinynv_now_s() - t0);
  return 0;
}

int tinynv_dev_mmu_init(tinynv_dev_t *d) {
  // how much video memory there is, asked before the window onto it is mapped, in that order, because the oracle does
  d->vram_size = (uint64_t)tinynv_rd32(d, NV_PGC6_AON_SECURE_SCRATCH_GROUP_42) << 20;

  // the second BAR is the window onto video memory. over thunderbolt, and on boards without resizable BARs, it shows
  // only the first 256 MB of it, which is why page tables and anything the cpu must reach have to live down there.
  if (d->pci->bar_map(d->pci, 1, 0, 0, &d->vram)) return -1;
  d->pci->bar_unmap(d->pci, &d->mmio);                       // the early mapping goes before the oracle's second one
  if (d->pci->bar_map(d->pci, 0, 0, 0, &d->mmio)) return -1; // re-taken the way the oracle does, as a 32 bit view
  d->large_bar = d->vram.size >= d->vram_size;
  if (!d->vram_size) return tinynv_fail("the gpu reports no video memory");
  return 0;
}

void tinynv_dev_quiesce(tinynv_dev_t *d) {
  if (d->pci && d->pci->quiesce) d->pci->quiesce(d->pci);
}
