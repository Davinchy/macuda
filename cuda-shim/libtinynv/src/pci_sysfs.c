// Linux backend: the GPU is ours, so config space and the BARs are files under /sys/bus/pci/devices/<bdf> and DMA memory is
// just locked anonymous memory whose physical pages we read out of /proc/self/pagemap. Mirrors tinygrad's PCIDevice.
// Needs the device unbound from nvidia.ko and the process privileged (root, or CAP_SYS_ADMIN + CAP_SYS_RAWIO for pagemap).
#define _GNU_SOURCE // glibc gates O_CLOEXEC, MAP_ANONYMOUS and MAP_POPULATE behind it; the bsd headers do not
#include "internal.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_LOCKED
#define MAP_LOCKED 0x2000
#endif
#ifndef MAP_POPULATE
#define MAP_POPULATE 0x008000
#endif

#define PCI_COMMAND 0x04
#define PCI_COMMAND_MASTER 0x4
#define SYSFS "/sys/bus/pci/devices/%s%s"

typedef struct {
  char bdf[16];
  int cfg_fd;
  int bar_fd[TINYNV_MAX_BARS];
  int pagemap_fd;
} sysfs_t;


static int sysfs_path(char *out, size_t n, const char *bdf, const char *leaf) { return snprintf(out, n, SYSFS, bdf, leaf); }

static int write_file(const char *path, const char *val) {
  int fd = open(path, O_WRONLY);
  if (fd < 0) return -1;
  ssize_t n = write(fd, val, strlen(val));
  close(fd);
  return n > 0 ? 0 : -1;
}

static uint32_t sf_cfg_read(tinynv_pci_t *p, uint32_t off, uint32_t size) {
  sysfs_t *s = p->ctx;
  uint32_t v = 0xffffffff;
  if (pread(s->cfg_fd, &v, size, off) != (ssize_t)size) return 0xffffffff;
  return v & (size >= 4 ? 0xffffffffu : (1u << (size * 8)) - 1);
}

static void sf_cfg_write(tinynv_pci_t *p, uint32_t off, uint32_t size, uint32_t val) {
  sysfs_t *s = p->ctx;
  // a dropped config write is how a device ends up mastering the bus after it was told not to, so it is worth a message
  if (pwrite(s->cfg_fd, &val, size, off) != (ssize_t)size)
    tinynv_fail("sysfs: config write of %#x to %#x failed: %s", val, off, strerror(errno));
}

// /sys/.../resource is one line per bar: start, end, flags
static int sf_bar_info(tinynv_pci_t *p, int bar, uint64_t *base, uint64_t *size) {
  sysfs_t *s = p->ctx;
  char path[256];
  sysfs_path(path, sizeof(path), s->bdf, "/resource");
  FILE *f = fopen(path, "r");
  if (!f) return tinynv_fail("sysfs: %s: %s", path, strerror(errno));
  unsigned long long st = 0, en = 0, fl = 0;
  for (int i = 0; i <= bar; i++)
    if (fscanf(f, "%llx %llx %llx", &st, &en, &fl) != 3) { fclose(f); return tinynv_fail("sysfs: bar %d missing", bar); }
  fclose(f);
  if (base) *base = st;
  if (size) *size = en > st ? en - st + 1 : 0;
  return 0;
}

static int sf_bar_map(tinynv_pci_t *p, int bar, uint64_t off, uint64_t size, tinynv_mmio_t *out) {
  sysfs_t *s = p->ctx;
  uint64_t bar_size = 0;
  if (bar < 0 || bar >= TINYNV_MAX_BARS) return tinynv_fail("sysfs: bar %d out of range", bar);
  if (sf_bar_info(p, bar, NULL, &bar_size)) return -1;
  if (!size) size = bar_size - off;
  if (off + size > bar_size) return tinynv_fail("sysfs: window [%#llx,%#llx) is outside bar %d of %#llx bytes",
                                         (unsigned long long)off, (unsigned long long)(off + size), bar, (unsigned long long)bar_size);
  if (s->bar_fd[bar] < 0) {
    char path[256], leaf[32];
    snprintf(leaf, sizeof(leaf), "/resource%d", bar);
    sysfs_path(path, sizeof(path), s->bdf, leaf);
    if ((s->bar_fd[bar] = open(path, O_RDWR | O_SYNC | O_CLOEXEC)) < 0) return tinynv_fail("sysfs: %s: %s", path, strerror(errno));
  }
  void *va = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, s->bar_fd[bar], off);
  if (va == MAP_FAILED) return tinynv_fail("sysfs: mmap bar %d: %s", bar, strerror(errno));
  *out = (tinynv_mmio_t){.ptr = va, .ctx = s, .bar = bar, .off = off, .size = size};
  return 0;
}

static void sf_bar_unmap(tinynv_pci_t *p, tinynv_mmio_t *m) {
  if (m->ptr) munmap((void *)m->ptr, m->size);
  memset(m, 0, sizeof(*m));
}

static int sf_dma_alloc(tinynv_pci_t *p, size_t size, tinynv_dma_t *out) {
  sysfs_t *s = p->ctx;
  size = (size + 0xfff) & ~(size_t)0xfff;
  void *va = mmap(NULL, size, PROT_READ | PROT_WRITE,
                  MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE | MAP_LOCKED, -1, 0);
  if (va == MAP_FAILED) return tinynv_fail("sysfs: dma mmap of %zu bytes: %s", size, strerror(errno));

  // the pages are locked, so their physical addresses are stable for as long as the mapping lives
  size_t npages = size / 0x1000;
  uint64_t *pages = calloc(npages, sizeof(uint64_t));
  for (size_t i = 0; i < npages; i++) {
    uint64_t ent = 0;
    off_t at = (off_t)(((uintptr_t)va / 0x1000 + i) * 8);
    if (pread(s->pagemap_fd, &ent, 8, at) != 8 || !(ent & (1ull << 63))) {
      free(pages);
      munmap(va, size);
      return tinynv_fail("sysfs: page %zu has no physical address (need root for /proc/self/pagemap)", i);
    }
    pages[i] = (ent & ((1ull << 55) - 1)) * 0x1000;
  }
  *out = (tinynv_dma_t){.va = va, .size = size, .pages = pages, .npages = npages, .ctx = s};
  tinynv_dma_bind_host(out);
  return 0;
}

static void sf_dma_free(tinynv_pci_t *p, tinynv_dma_t *dma) {
  (void)p;
  free(dma->pages);
  if (dma->va) munmap(dma->va, dma->size);
  memset(dma, 0, sizeof(*dma));
}

static int sf_reset(tinynv_pci_t *p) {
  sysfs_t *s = p->ctx;
  char path[256];
  sysfs_path(path, sizeof(path), s->bdf, "/reset");
  return write_file(path, "1") ? tinynv_fail("sysfs: reset: %s", strerror(errno)) : 0;
}

static void sf_quiesce(tinynv_pci_t *p) {
  uint32_t cmd = sf_cfg_read(p, PCI_COMMAND, 2);
  if (cmd == 0xffffffff) return;
  sf_cfg_write(p, PCI_COMMAND, 2, cmd & ~PCI_COMMAND_MASTER);
  sf_cfg_read(p, PCI_COMMAND, 2);
}

static void sf_close(tinynv_pci_t *p) {
  sysfs_t *s = p->ctx;
  if (!s) return;
  for (int i = 0; i < TINYNV_MAX_BARS; i++)
    if (s->bar_fd[i] >= 0) close(s->bar_fd[i]);
  if (s->cfg_fd >= 0) close(s->cfg_fd);
  if (s->pagemap_fd >= 0) close(s->pagemap_fd);
  free(s);
  p->ctx = NULL;
}

int tinynv_pci_open_sysfs(const char *bdf, tinynv_pci_t *out) {
  char path[256];
  sysfs_t *s = calloc(1, sizeof(sysfs_t));
  snprintf(s->bdf, sizeof(s->bdf), "%s", bdf);
  s->cfg_fd = s->pagemap_fd = -1;
  for (int i = 0; i < TINYNV_MAX_BARS; i++) s->bar_fd[i] = -1;

  // take the device away from whatever driver holds it, and drop its sibling functions (the hdmi audio device)
  sysfs_path(path, sizeof(path), bdf, "/driver/unbind");
  if (!access(path, W_OK)) write_file(path, bdf);
  sysfs_path(path, sizeof(path), bdf, "/driver");
  if (!access(path, F_OK)) { free(s); return tinynv_fail("sysfs: %s still has a driver bound", bdf); }
  for (int fn = 1; fn < 8; fn++) {
    char sib[32];
    snprintf(sib, sizeof(sib), "%.*s%d/remove", (int)(strlen(bdf) - 1), bdf, fn);
    sysfs_path(path, sizeof(path), sib, "");
    if (!access(path, W_OK)) write_file(path, "1");
  }

  sysfs_path(path, sizeof(path), bdf, "/enable");
  if (write_file(path, "1")) { free(s); return tinynv_fail("sysfs: cannot enable %s: %s", bdf, strerror(errno)); }

  sysfs_path(path, sizeof(path), bdf, "/config");
  if ((s->cfg_fd = open(path, O_RDWR | O_SYNC | O_CLOEXEC)) < 0) { free(s); return tinynv_fail("sysfs: %s: %s", path, strerror(errno)); }
  if ((s->pagemap_fd = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC)) < 0) {
    close(s->cfg_fd);
    free(s);
    return tinynv_fail("sysfs: /proc/self/pagemap: %s", strerror(errno));
  }

  *out = (tinynv_pci_t){.ctx = s, .name = s->bdf, .cfg_read = sf_cfg_read, .cfg_write = sf_cfg_write, .bar_info = sf_bar_info,
                        .bar_map = sf_bar_map, .bar_unmap = sf_bar_unmap, .dma_alloc = sf_dma_alloc, .dma_free = sf_dma_free,
                        .reset = sf_reset, .quiesce = sf_quiesce, .close = sf_close};
  return 0;
}
