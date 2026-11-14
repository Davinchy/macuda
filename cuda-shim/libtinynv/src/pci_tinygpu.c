// TinyGPU backend: on macOS a signed helper owns the DriverKit user client, so config space, BAR windows and DMA memory are
// reached over a unix socket. Wire format and semantics: docs/driver/architecture.md §2 (identical to the python client in
// tinygrad/runtime/support/system.py, so the two can be diffed). Writes are posted and carry no reply; reads are round trips.
#include "internal.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

enum {
  CMD_PROBE = 0, CMD_MAP_BAR, CMD_MAP_SYSMEM_FD, CMD_CFG_READ, CMD_CFG_WRITE, CMD_RESET,
  CMD_MMIO_READ, CMD_MMIO_WRITE, CMD_MAP_SYSMEM, CMD_SYSMEM_READ, CMD_SYSMEM_WRITE, CMD_RESIZE_BAR, CMD_PING,
};

#pragma pack(push, 1)
typedef struct { uint8_t cmd; uint32_t dev_id, bar; uint64_t arg0, arg1, arg2; } req_t;
typedef struct { uint8_t status; uint64_t resp0, resp1; } resp_t;
#pragma pack(pop)
_Static_assert(sizeof(req_t) == 33, "request is 33 packed bytes");
_Static_assert(sizeof(resp_t) == 17, "response is 17 packed bytes");

#define PCI_COMMAND 0x04
#define PCI_COMMAND_MASTER 0x4
#define MAX_DMA 128 // the server keeps at most this many mappings per session and frees none of them

typedef struct {
  int fd;
  // A read whose request has been sent and whose reply has not been taken. Any other request must first take that
  // reply (the socket carries one conversation), so rpc() drains it into pending_val and rd32_recv finds it there.
  int pending, pending_done;
  uint32_t pending_val;
  // A receive timeout (or any short transfer) leaves the conversation desynchronized: half a reply may still arrive
  // later. The connection is poisoned, every later operation fails fast, and the caller must treat the device state
  // as unknown (the unlock transaction enters its recovery path). Never silently retried on the same socket.
  int broken;
  int ndma;
  struct { void *va; size_t size; } dma[MAX_DMA];
  uint64_t bar_size[TINYNV_MAX_BARS];
} tg_t;


static int xfer_all(tg_t *tg, void *buf, size_t len, int sending) {
  if (tg->broken) return tinynv_fail("tinygpu: connection is broken (an earlier transfer failed); refusing further use");
  for (size_t off = 0; off < len;) {
    ssize_t n = sending ? send(tg->fd, (char *)buf + off, len - off, 0) : recv(tg->fd, (char *)buf + off, len - off, 0);
    // A catchable signal (SIGINT/SIGTERM -> the unlock's abort flag) interrupted the transfer before ANY byte moved.
    // Retrying is safe — nothing was consumed, the conversation stays in sync — and it matters: poisoning a healthy
    // socket here would make the recovery unwind's best-effort quiesce (BME clear) fail fast on a broken connection.
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) {
      tg->broken = 1;
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        return tinynv_fail("tinygpu: receive timed out after %zu of %zu bytes (server wedged?); connection poisoned", off, len);
      return tinynv_fail("tinygpu: connection closed after %zu of %zu bytes", off, len);
    }
    off += (size_t)n;
  }
  return 0;
}

// Take the reply of a split read that is still on the socket: the header, an error message if it carries one, and
// the four bytes read. Kept for rd32_recv, which may be asked for it after other requests have gone by.
static uint32_t tg_take_pending(tg_t *tg) {
  resp_t rp;
  uint32_t v = 0xffffffffu;
  tg->pending = 0;
  if (xfer_all(tg, &rp, sizeof(rp), 0)) return v;
  if (rp.status != 0) {
    char msg[256];
    size_t n = rp.resp0 < sizeof(msg) ? (size_t)rp.resp0 : sizeof(msg);
    if (n) xfer_all(tg, msg, n, 0);
    return v;
  }
  xfer_all(tg, &v, 4, 0);
  return v;
}
// one request, one reply. payload is appended to the request; readout is the bulk data that follows the reply.
static int rpc(tg_t *tg, resp_t *rp, uint8_t cmd, uint32_t bar, uint64_t a0, uint64_t a1, uint64_t a2,
               const void *payload, size_t payload_len, void *readout, size_t readout_len, int *fd_out) {
  if (tg->broken) return tinynv_fail("tinygpu: connection is broken (an earlier transfer failed); refusing further use");
  if (tg->pending) { tg->pending_val = tg_take_pending(tg); tg->pending_done = 1; }
  req_t rq = {.cmd = cmd, .dev_id = 0, .bar = bar, .arg0 = a0, .arg1 = a1, .arg2 = a2};
  if (xfer_all(tg, &rq, sizeof(rq), 1)) return -1;
  if (payload_len && xfer_all(tg, (void *)payload, payload_len, 1)) return -1;

  if (fd_out) { // the reply carries a descriptor for the shared memory the server just created
    *fd_out = -1;
    char cbuf[CMSG_SPACE(sizeof(int))];
    struct iovec iov = {.iov_base = rp, .iov_len = sizeof(*rp)};
    struct msghdr msg = {.msg_iov = &iov, .msg_iovlen = 1, .msg_control = cbuf, .msg_controllen = sizeof(cbuf)};
    ssize_t n;
    do { n = recvmsg(tg->fd, &msg, 0); } while (n < 0 && errno == EINTR); // same rule as xfer_all: no byte moved
    if (n != (ssize_t)sizeof(*rp)) {
      tg->broken = 1;
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        return tinynv_fail("tinygpu: receive timed out waiting for a dma reply; connection poisoned");
      return tinynv_fail("tinygpu: short reply (%zd bytes)", n);
    }
    for (struct cmsghdr *c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c))
      if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) memcpy(fd_out, CMSG_DATA(c), sizeof(int));
  } else if (xfer_all(tg, rp, sizeof(*rp), 0)) return -1;

  if (rp->status != 0) { // the message follows the reply, resp0 bytes of it
    char msg[256] = "unknown error";
    size_t n = rp->resp0 < sizeof(msg) - 1 ? (size_t)rp->resp0 : sizeof(msg) - 1;
    if (n && !xfer_all(tg, msg, n, 0)) msg[n] = 0;
    return tinynv_fail("tinygpu: command %u failed: %s", cmd, msg);
  }
  if (readout_len && xfer_all(tg, readout, readout_len, 0)) return -1;
  return 0;
}

static uint32_t tg_cfg_read(tinynv_pci_t *p, uint32_t off, uint32_t size) {
  resp_t rp;
  if (rpc(p->ctx, &rp, CMD_CFG_READ, 0, off, size, 0, NULL, 0, NULL, 0, NULL)) return 0xffffffff;
  return (uint32_t)rp.resp0;
}

static void tg_cfg_write(tinynv_pci_t *p, uint32_t off, uint32_t size, uint32_t val) {
  resp_t rp;
  rpc(p->ctx, &rp, CMD_CFG_WRITE, 0, off, size, val, NULL, 0, NULL, 0, NULL);
}

static int tg_bar_info(tinynv_pci_t *p, int bar, uint64_t *base, uint64_t *size) {
  tg_t *tg = p->ctx;
  resp_t rp;
  if (bar < 0 || bar >= TINYNV_MAX_BARS) return tinynv_fail("tinygpu: bar %d out of range", bar);
  if (rpc(tg, &rp, CMD_MAP_BAR, (uint32_t)bar, 0, 0, 0, NULL, 0, NULL, 0, NULL)) return -1;
  tg->bar_size[bar] = rp.resp1;
  if (base) *base = rp.resp0; // the server's own mapping address, not a bus address
  if (size) *size = rp.resp1;
  return 0;
}

// the window never lives in this address space: each access is one message, and writes are posted
static uint32_t tg_rd32(tinynv_mmio_t *m, uint64_t off) {
  uint32_t v = 0xffffffff;
  resp_t rp;
  rpc(m->ctx, &rp, CMD_MMIO_READ, (uint32_t)m->bar, m->off + off, 4, 0, NULL, 0, &v, 4, NULL);
  return v;
}
// The split read over the socket: the request goes now and the reply stays on the socket until asked for. Posted
// writes may go between; another request first drains it (rpc above).
static int tg_rd32_send(tinynv_mmio_t *m, uint64_t off) {
  tg_t *tg = m->ctx;
  if (tg->broken) return tinynv_fail("tinygpu: connection is broken; refusing further use");
  if (tg->pending) { tg->pending_val = tg_take_pending(tg); tg->pending_done = 1; }
  tg->pending_done = 0;
  req_t rq = {.cmd = CMD_MMIO_READ, .bar = (uint32_t)m->bar, .arg0 = m->off + off, .arg1 = 4};
  if (xfer_all(tg, &rq, sizeof(rq), 1)) return -1;
  tg->pending = 1;
  return 0;
}
static uint32_t tg_rd32_recv(tinynv_mmio_t *m) {
  tg_t *tg = m->ctx;
  if (tg->pending_done) { tg->pending_done = 0; return tg->pending_val; }
  if (tg->pending) return tg_take_pending(tg);
  return 0xffffffffu;
}
static void tg_wr32(tinynv_mmio_t *m, uint64_t off, uint32_t v) {
  tg_t *tg = m->ctx;
  req_t rq = {.cmd = CMD_MMIO_WRITE, .bar = (uint32_t)m->bar, .arg0 = m->off + off, .arg1 = 4};
  if (!xfer_all(tg, &rq, sizeof(rq), 1)) xfer_all(tg, &v, 4, 1);
}
static void tg_rd_block(tinynv_mmio_t *m, uint64_t off, void *dst, size_t n) {
  resp_t rp;
  if (rpc(m->ctx, &rp, CMD_MMIO_READ, (uint32_t)m->bar, m->off + off, n, 0, NULL, 0, dst, n, NULL)) memset(dst, 0xff, n);
}
static void tg_wr_block(tinynv_mmio_t *m, uint64_t off, const void *src, size_t n) {
  tg_t *tg = m->ctx;
  req_t rq = {.cmd = CMD_MMIO_WRITE, .bar = (uint32_t)m->bar, .arg0 = m->off + off, .arg1 = n};
  if (!xfer_all(tg, &rq, sizeof(rq), 1)) xfer_all(tg, (void *)src, n, 1);
}

static int tg_bar_map(tinynv_pci_t *p, int bar, uint64_t off, uint64_t size, tinynv_mmio_t *out) {
  tg_t *tg = p->ctx;
  uint64_t bar_size;
  if (tg_bar_info(p, bar, NULL, &bar_size)) return -1;
  if (!size) size = bar_size - off;
  if (off + size > bar_size) return tinynv_fail("tinygpu: window [%#llx,%#llx) is outside bar %d of %#llx bytes",
                                         (unsigned long long)off, (unsigned long long)(off + size), bar, (unsigned long long)bar_size);
  *out = (tinynv_mmio_t){.ptr = NULL, .ctx = tg, .bar = bar, .off = off, .size = size,
                         .rd32 = tg_rd32, .wr32 = tg_wr32, .rd_block = tg_rd_block, .wr_block = tg_wr_block,
                         .rd32_send = tg_rd32_send, .rd32_recv = tg_rd32_recv};
  return 0;
}

static void tg_bar_unmap(tinynv_pci_t *p, tinynv_mmio_t *m) { (void)p; memset(m, 0, sizeof(*m)); }

static int tg_dma_alloc(tinynv_pci_t *p, size_t size, tinynv_dma_t *out) {
  tg_t *tg = p->ctx;
  resp_t rp;
  int fd = -1;
  if (tg->ndma >= MAX_DMA) return tinynv_fail("tinygpu: the server allows %d dma mappings per session", MAX_DMA);
  if (rpc(tg, &rp, CMD_MAP_SYSMEM_FD, 0, size, 0, 0, NULL, 0, NULL, 0, &fd)) return -1;
  if (fd < 0) return tinynv_fail("tinygpu: dma reply carried no descriptor");

  size_t mapped = rp.resp0;
  void *va = mmap(NULL, mapped, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (va == MAP_FAILED) return tinynv_fail("tinygpu: mmap of %zu dma bytes failed: %s", mapped, strerror(errno));

  // the server leaves the mapping described at its own head: (address, length) pairs, terminated by (0, 0)
  size_t npages = mapped / 0x1000;
  uint64_t *pages = calloc(npages, sizeof(uint64_t)), *seg = va;
  size_t n = 0;
  for (size_t i = 0; seg[2 * i + 1] && n < npages; i++)
    for (uint64_t o = 0; o < seg[2 * i + 1] && n < npages; o += 0x1000) pages[n++] = seg[2 * i] + o;
  if (n < npages) {
    free(pages);
    munmap(va, mapped);
    return tinynv_fail("tinygpu: dma mapping describes %zu of %zu pages (the dext truncates past 32 segments)", n, npages);
  }

  tg->dma[tg->ndma].va = va;
  tg->dma[tg->ndma].size = mapped;
  tg->ndma++;
  // WHERE THE LARGE HOST MAPPINGS SIT ON THE BUS, every session. A DART fault is latched as an IOVA and nothing else, and
  // until this line no log could say which buffer an IOVA belonged to in the session that faulted: on 2026-10-14 two
  // latches at 0x84ac8000 (21:23, 23:32) could be placed only by a September boot trace, which put the GSP-RM image
  // there. One line per mapping of 1 MiB or more - a handful per boot - with the first and last page and the segments.
  if (mapped >= (1u << 20)) {
    size_t nseg = 0; while (seg[2 * nseg + 1]) nseg++;
    fprintf(stderr, "tinynv: dma mapping %d: %zu bytes, iova %#llx..%#llx in %zu segment(s)\n", tg->ndma - 1, mapped,
            (unsigned long long)pages[0], (unsigned long long)(pages[npages - 1] + 0xfff), nseg);
  }
  *out = (tinynv_dma_t){.va = va, .size = mapped, .pages = pages, .npages = npages, .ctx = tg};
  tinynv_dma_bind_host(out);
  return 0;
}

static void tg_dma_free(tinynv_pci_t *p, tinynv_dma_t *dma) {
  (void)p; // the server owns the pages until the connection drops: only the page list is ours to release
  free(dma->pages);
  dma->pages = NULL;
  dma->npages = 0;
}

static int tg_reset(tinynv_pci_t *p) {
  resp_t rp;
  return rpc(p->ctx, &rp, CMD_RESET, 0, 0, 0, 0, NULL, 0, NULL, 0, NULL);
}

static void tg_quiesce(tinynv_pci_t *p) {
  uint32_t cmd = tg_cfg_read(p, PCI_COMMAND, 2);
  if (cmd == 0xffffffff) return; // already gone from the bus
  tg_cfg_write(p, PCI_COMMAND, 2, cmd & ~PCI_COMMAND_MASTER);
  tg_cfg_read(p, PCI_COMMAND, 2); // read back, so the write has landed before any mapping is torn down
}

static void tg_close(tinynv_pci_t *p) {
  tg_t *tg = p->ctx;
  if (!tg) return;
  for (int i = 0; i < tg->ndma; i++) munmap(tg->dma[i].va, tg->dma[i].size);
  if (tg->fd >= 0) close(tg->fd);
  free(tg);
  p->ctx = NULL;
}

int tinynv_pci_open_tinygpu(const char *sock_path, tinynv_pci_t *out) {
  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  if (strlen(sock_path) >= sizeof(addr.sun_path)) return tinynv_fail("tinygpu: socket path is too long");
  strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return tinynv_fail("tinygpu: socket: %s", strerror(errno));
  if (connect(fd, (struct sockaddr *)&addr, sizeof(addr))) {
    close(fd);
    return tinynv_fail("tinygpu: no server on %s: %s", sock_path, strerror(errno));
  }
  int buf = 64 << 20;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
  // A wedged server must not block recv forever (transport-audit O9/D6): every read is a round trip the server
  // answers immediately, so a generous ceiling costs nothing on the normal path. A timeout poisons the connection
  // (xfer_all); it is a failure with containment, never a success and never a silent retry. 0 disables the ceiling.
  long timeout_s = 30;
  const char *t = getenv("TINYNV_SOCKET_TIMEOUT_S");
  if (t && *t) timeout_s = strtol(t, NULL, 10);
  if (timeout_s > 0) {
    struct timeval tv = {.tv_sec = timeout_s, .tv_usec = 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  }

  tg_t *tg = calloc(1, sizeof(tg_t));
  tg->fd = fd;
  *out = (tinynv_pci_t){.ctx = tg, .live = 1, .name = "usb4", .cfg_read = tg_cfg_read, .cfg_write = tg_cfg_write, .bar_info = tg_bar_info,
                        .bar_map = tg_bar_map, .bar_unmap = tg_bar_unmap, .dma_alloc = tg_dma_alloc, .dma_free = tg_dma_free,
                        .reset = tg_reset, .quiesce = tg_quiesce, .close = tg_close};
  return 0;
}
