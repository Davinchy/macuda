// The GPU as the driver sees it before any firmware is running: registers, memory windows, and what kind of chip it is.
#ifndef TINYNV_DEV_H
#define TINYNV_DEV_H
#include "tinynv_pci.h"

typedef struct {
  tinynv_pci_t *pci;
  tinynv_mmio_t mmio;   // BAR0: registers
  tinynv_mmio_t vram;   // BAR1: the window onto video memory, 256 MB on a small bar card
  uint32_t chip_id, architecture, implementation;
  char chip_name[16];   // GA102, GB202 and so on
  char fw_name[16];     // the firmware directory the boot images come from
  int fmc_boot;         // Blackwell and later: boot through the FSP chain of trust rather than the VBIOS
  int mmu_ver;          // page table generation, 2 or 3
  int wpr2_was_up;      // firmware was already resident when we arrived, so the chip was reset
  int large_bar;        // the whole of video memory is cpu visible
  uint64_t vram_size;
} tinynv_dev_t;

int tinynv_dev_early_init(tinynv_dev_t *d, tinynv_pci_t *pci);
int tinynv_dev_mmu_init(tinynv_dev_t *d);
void tinynv_dev_quiesce(tinynv_dev_t *d);
uint32_t tinynv_rd32(tinynv_dev_t *d, uint64_t off);
void tinynv_wr32(tinynv_dev_t *d, uint64_t off, uint32_t v);
int tinynv_wait_reg(tinynv_dev_t *d, uint64_t off, uint32_t mask, uint32_t want, int timeout_ms, const char *what);

#endif
