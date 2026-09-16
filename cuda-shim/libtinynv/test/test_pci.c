// Drives the TinyGPU backend against the python mock server: config space, bar windows, posted writes, dma descriptors.
// Same server the python driver's tests use, so a difference here is a difference between the two clients.
#include "tinynv_pci.h"
#include <stdio.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

#define PCI_COMMAND 0x04

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <socket>\n", argv[0]); return 2; }
  tinynv_pci_t pci;
  if (tinynv_pci_open_tinygpu(argv[1], &pci)) { fprintf(stderr, "open: %s\n", tinynv_last_error()); return 1; }

  // config space
  CHECK(pci.cfg_read(&pci, 0, 4) == 0x2b8510de, "vendor/device = %#x", pci.cfg_read(&pci, 0, 4));
  CHECK(pci.cfg_read(&pci, 2, 2) == 0x2b85, "device id");
  pci.cfg_write(&pci, PCI_COMMAND, 2, 0x0407);
  CHECK(pci.cfg_read(&pci, PCI_COMMAND, 2) == 0x0407, "command register did not stick");

  // quiesce clears bus master and nothing else
  pci.quiesce(&pci);
  CHECK(pci.cfg_read(&pci, PCI_COMMAND, 2) == 0x0403, "quiesce left %#x", pci.cfg_read(&pci, PCI_COMMAND, 2));

  // bar windows
  uint64_t base = 0, size = 0;
  CHECK(pci.bar_info(&pci, 1, &base, &size) == 0 && size == (4 << 20), "bar1 size %#llx", (unsigned long long)size);

  tinynv_mmio_t regs;
  CHECK(pci.bar_map(&pci, 0, 0, 0, &regs) == 0, "map bar0: %s", tinynv_last_error());
  nv_wr32(&regs, 0x40, 0xdeadbeef);
  CHECK(nv_rd32(&regs, 0x40) == 0xdeadbeef, "readback %#x", nv_rd32(&regs, 0x40));

  uint32_t blk[4] = {1, 2, 3, 4}, out[4] = {0};
  nv_wr_block(&regs, 0x80, blk, sizeof(blk));
  nv_rd_block(&regs, 0x80, out, sizeof(out));
  CHECK(memcmp(blk, out, sizeof(blk)) == 0, "block readback mismatch");

  // a window inside a bar is offset by its own base, and one past the end is refused rather than dropped
  tinynv_mmio_t win;
  CHECK(pci.bar_map(&pci, 0, 0x1000, 0x100, &win) == 0, "map window");
  nv_wr32(&win, 0, 0xcafef00d);
  CHECK(nv_rd32(&regs, 0x1000) == 0xcafef00d, "window base is wrong");
  CHECK(pci.bar_map(&pci, 0, 0x1f000, 0x2000, &win) != 0, "a window past the end of a bar must be refused");

  // dma memory: mapped here, described to the gpu by page address
  tinynv_dma_t dma;
  CHECK(pci.dma_alloc(&pci, 0x3000, &dma) == 0, "dma_alloc: %s", tinynv_last_error());
  if (dma.va) {
    // the server rounds up to a page and never hands out less than 16 KB, the floor an IOMemoryDescriptor will describe
    CHECK(dma.size == 0x4000 && dma.npages == dma.size / 0x1000, "dma covers %zu pages of %zu bytes", dma.npages, dma.size);
    CHECK(dma.pages[1] == dma.pages[0] + 0x1000, "pages are not described in order");
    memcpy(dma.va, "tiny", 4); // the server shares this memory, so the write needs no message at all
    CHECK(memcmp(dma.va, "tiny", 4) == 0, "dma memory is not writable");
    pci.dma_free(&pci, &dma);
  }

  pci.close(&pci);
  printf(fails ? "%d checks failed\n" : "all checks passed\n", fails);
  return fails ? 1 : 0;
}
