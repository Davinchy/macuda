// Chip bring-up: identify the GPU, put it in a known state, and work out which boot path it needs.
//
// This mirrors tinygrad's NVDev._early_ip_init and _early_mmu_init operation for operation, because the python driver is
// the oracle: run this against a recorded boot and any difference is a difference in logic. Two architectures matter:
// Ampere (GA10x) boots its falcon from the VBIOS, Blackwell (GB20x) through the FSP chain of trust, and they differ in
// MMU version too. Everything here is register access plus config space, so it works over any of the PCI backends.
#include "dev.h"
#include "internal.h"
#include "nv_regs.h"
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

int tinynv_dev_early_init(tinynv_dev_t *d, tinynv_pci_t *pci) {
  memset(d, 0, sizeof(*d));
  d->pci = pci;
  if (pci->bar_map(pci, 0, 0, 0, &d->mmio)) return -1;

  // a GPU whose write-protected region is still up has a live firmware on it from a previous run: reset it, but stop it
  // mastering the bus first, or it keeps writing into host memory that is about to be unmapped underneath it
  d->wpr2_was_up = tinynv_rd32(d, NV_PFB_PRI_MMU_WPR2_ADDR_HI) != 0;
  if (d->wpr2_was_up) {
    uint32_t cmd = pci->cfg_read(pci, PCI_COMMAND, 2);
    cfg_write_flush(d, PCI_COMMAND, cmd & ~PCI_COMMAND_MASTER, 2);
    if (pci->reset(pci)) return -1;
    sleep_ms(100); // the function is not answering until it has come out of reset
  }

  // only bus mastering: whether the memory window is decoding is the backend's business, settled when it opened the
  // device, and writing more than the oracle does would diverge on a board that comes up with it off
  uint32_t cmd = pci->cfg_read(pci, PCI_COMMAND, 2);
  cfg_write_flush(d, PCI_COMMAND, cmd | PCI_COMMAND_MASTER, 2);

  d->chip_id = tinynv_rd32(d, NV_PMC_BOOT_0);
  if (d->chip_id == 0xffffffff)
    return tinynv_fail("the gpu reads all ones: it has fallen off the bus, or its memory window is not enabled");

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

  // the firmware that runs at power-on signals it has finished; until then the chip is not ready to be driven
  if (!d->fmc_boot) return tinynv_fail("%s boots its falcon from the vbios, which this driver does not implement", d->chip_name);
  if (tinynv_wait_reg(d, NV_THERM_I2CS_SCRATCH, 0xffffffff, 0xff, 10000, "waiting for the boot firmware")) return -1;
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
