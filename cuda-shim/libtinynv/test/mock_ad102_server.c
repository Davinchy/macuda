// A wire-protocol TinyGPU mock server backed by the same fake AD102 the unit suite uses.
//
// Purpose: drive the REAL probe binary (tools/rtx4090-probe) end-to-end over a unix socket — production asset pins,
// production loader, production transport code — so the recovery hold, signal handling and lock protocol are tested
// as they will run live. Control flow only: no GPU, no TinyGPU.app, no hardware behavior is proven here.
//
// Server semantics copied from the installed transport (transport-audit.md §1): one client; DMA buffers are shared
// memory whose fd is passed with SCM_RIGHTS, the segment table sits at the buffer head, and NOTHING is freed until
// the client disconnects (there is no per-buffer unmap command). On disconnect an evidence file is written: what the
// client did to PCI COMMAND, how many mappings were still live, resets, falcon starts.
//
//   mock_ad102_server <socket-path> <knob> <evidence-path>
//   knobs: none | no-perms | marker-only | wrong-device | stall-sec2 | retained
#include "fake_ad102.h"
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
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
_Static_assert(sizeof(req_t) == 33 && sizeof(resp_t) == 17, "wire layout");

static int cmd_writes, bme_clears, bme_clears_after_sec2;
static uint32_t last_command = 0xffff;

static int read_all(int fd, void *buf, size_t len) {
  for (size_t off = 0; off < len;) {
    ssize_t n = read(fd, (char *)buf + off, len - off);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return -1;
    off += (size_t)n;
  }
  return 0;
}
static int write_all(int fd, const void *buf, size_t len) {
  for (size_t off = 0; off < len;) {
    ssize_t n = write(fd, (const char *)buf + off, len - off);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return -1;
    off += (size_t)n;
  }
  return 0;
}
static int reply(int fd, uint8_t status, uint64_t r0, uint64_t r1) {
  resp_t rp = {status, r0, r1};
  return write_all(fd, &rp, sizeof(rp));
}

// CMD_MAP_SYSMEM_FD: shared memory + fd passing + segment table at the head, exactly like the installed server.
static int serve_dma(int fd, uint64_t size) {
  size_t mapped = (size + 0xfff) & ~(size_t)0xfff;
  if (mapped < 0x4000) mapped = 0x4000; // the server's 16 KiB minimum
  if (F.ndma >= 128) return reply(fd, 1, 0, 0);
  char name[64];
  snprintf(name, sizeof(name), "/tu-mock-%ld-%d", (long)getpid(), F.ndma);
  int shm = shm_open(name, O_CREAT | O_RDWR, 0600);
  if (shm < 0) return reply(fd, 1, 0, 0);
  shm_unlink(name); // the fd keeps the object alive, like the server's unlink-after-map lifetime
  if (ftruncate(shm, (off_t)mapped)) { close(shm); return reply(fd, 1, 0, 0); }
  void *va = mmap(NULL, mapped, PROT_READ | PROT_WRITE, MAP_SHARED, shm, 0);
  if (va == MAP_FAILED) { close(shm); return reply(fd, 1, 0, 0); }
  memset(va, 0, mapped);
  uint64_t iova = F.next_iova;
  F.next_iova += mapped;
  size_t npages = mapped / 0x1000;
  uint64_t *pages = malloc(npages * sizeof(uint64_t));
  for (size_t i = 0; i < npages; i++) pages[i] = iova + i * 0x1000;
  int i = F.ndma++;
  F.dma[i].va = va;
  F.dma[i].size = mapped;
  F.dma[i].iova = iova;
  F.dma[i].pages = pages;
  F.dma[i].npages = npages;
  F.dma[i].live = 1;
  F.ndma_alloc++;
  uint64_t *seg = va; // (iova, len) pairs, (0,0)-terminated; the client parses then overwrites
  seg[0] = iova; seg[1] = mapped; seg[2] = 0; seg[3] = 0;

  resp_t rp = {0, mapped, 0};
  struct iovec iov = {.iov_base = &rp, .iov_len = sizeof(rp)};
  char cbuf[CMSG_SPACE(sizeof(int))];
  struct msghdr msg = {.msg_iov = &iov, .msg_iovlen = 1, .msg_control = cbuf, .msg_controllen = sizeof(cbuf)};
  memset(cbuf, 0, sizeof(cbuf));
  struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
  c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(c), &shm, sizeof(int));
  ssize_t n;
  do { n = sendmsg(fd, &msg, 0); } while (n < 0 && errno == EINTR);
  close(shm);
  return n == (ssize_t)sizeof(rp) ? 0 : -1;
}

static void write_evidence(const char *path) {
  int live = 0;
  for (int i = 0; i < F.ndma; i++) live += F.dma[i].live;
  FILE *f = fopen(path, "w");
  if (!f) { perror(path); return; }
  fprintf(f, "cmd_writes=%d\nbme_clears=%d\nbme_clears_after_sec2=%d\nlast_command=%#x\nresets=%d\nstarts=%d\n"
             "dma_allocs=%d\ndma_live_at_disconnect=%d\nsec2_applied=%d\n",
          cmd_writes, bme_clears, bme_clears_after_sec2, last_command, F.nreset, F.nstarts, F.ndma_alloc, live,
          F.sec2_applied);
  fclose(f);
}

int main(int argc, char **argv) {
  if (argc != 4) { fprintf(stderr, "usage: mock_ad102_server <socket> <knob> <evidence>\n"); return 2; }
  const char *sockpath = argv[1], *knob = argv[2], *evidence = argv[3];
  int k = K_NONE;
  if (!strcmp(knob, "no-perms")) k = K_NO_PERMS;
  else if (!strcmp(knob, "marker-only")) k = K_MARKER_ONLY;
  else if (!strcmp(knob, "wrong-device")) k = K_WRONG_DEVID;
  else if (!strcmp(knob, "stall-sec2")) k = K_STALL_SEC2;
  else if (!strcmp(knob, "retained")) k = K_RETAINED;
  else if (strcmp(knob, "none")) { fprintf(stderr, "unknown knob %s\n", knob); return 2; }
  fake_init(k);
  F.wire_mode = 1;
  signal(SIGPIPE, SIG_IGN);

  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  if (strlen(sockpath) >= sizeof(addr.sun_path)) { fprintf(stderr, "socket path too long\n"); return 2; }
  strncpy(addr.sun_path, sockpath, sizeof(addr.sun_path) - 1);
  int srv = socket(AF_UNIX, SOCK_STREAM, 0);
  if (srv < 0 || bind(srv, (struct sockaddr *)&addr, sizeof(addr)) || listen(srv, 1)) {
    fprintf(stderr, "mock server: bind/listen %s: %s\n", sockpath, strerror(errno));
    return 2;
  }
  printf("ready\n");
  fflush(stdout);
  int fd = accept(srv, NULL, NULL);
  if (fd < 0) { fprintf(stderr, "mock server: accept: %s\n", strerror(errno)); return 2; }
  close(srv); // one client, like the installed server (a second connect is refused by the listener being gone)

  tinynv_mmio_t m = {.ptr = NULL, .ctx = NULL, .bar = 0, .off = 0, .size = BAR0_SIZE};
  int rc = 0;
  for (;;) {
    req_t rq;
    if (read_all(fd, &rq, sizeof(rq))) break; // the client disconnected: retention ends here, like the real server
    switch (rq.cmd) {
      case CMD_PROBE: case CMD_PING: rc = reply(fd, 0, 0, 0); break;
      case CMD_CFG_READ: rc = reply(fd, 0, cfg_rd(NULL, (uint32_t)rq.arg0, (uint32_t)rq.arg1), 0); break;
      case CMD_CFG_WRITE: {
        uint32_t off = (uint32_t)rq.arg0, val = (uint32_t)rq.arg2;
        if (off == 0x04) {
          cmd_writes++;
          last_command = val;
          if (!(val & 4)) {
            bme_clears++;
            if (F.nstarts >= 2) bme_clears_after_sec2++; // GSP-FWSEC start is first, SEC2 second
          }
        }
        cfg_wr(NULL, off, (uint32_t)rq.arg1, val);
        rc = reply(fd, 0, 0, 0);
        break;
      }
      case CMD_MAP_BAR:
        rc = rq.bar == 0 ? reply(fd, 0, 0, BAR0_SIZE) : reply(fd, 1, 0, 0);
        break;
      case CMD_RESET: fake_reset(NULL); rc = reply(fd, 0, 0, 0); break;
      case CMD_MMIO_READ: {
        if (F.stall_read_ms) { // the signal-delivery window: the client is provably blocked in recv meanwhile
          usleep((useconds_t)F.stall_read_ms * 1000);
          F.stall_read_ms = 0;
        }
        if (rq.arg1 == 4) {
          uint32_t v = fake_rd32(&m, rq.arg0);
          rc = reply(fd, 0, 0, 0) || write_all(fd, &v, 4);
        } else {
          uint8_t *buf = malloc(rq.arg1);
          fake_rd_block(&m, rq.arg0, buf, rq.arg1);
          rc = reply(fd, 0, 0, 0) || write_all(fd, buf, rq.arg1);
          free(buf);
        }
        break;
      }
      case CMD_MMIO_WRITE: { // posted: payload, no reply (the installed server's silent-drop hazard is not modeled)
        uint8_t *buf = malloc(rq.arg1);
        if (read_all(fd, buf, rq.arg1)) { free(buf); rc = -1; break; }
        for (uint64_t i = 0; i + 4 <= rq.arg1; i += 4) {
          uint32_t v;
          memcpy(&v, buf + i, 4);
          fake_wr32(&m, rq.arg0 + i, v);
        }
        free(buf);
        break;
      }
      case CMD_MAP_SYSMEM_FD: rc = serve_dma(fd, rq.arg0); break;
      default: rc = reply(fd, 1, 0, 0); break;
    }
    if (rc) break;
  }
  close(fd);
  write_evidence(evidence);
  fprintf(stderr, "mock server: client disconnected; evidence in %s\n", evidence);
  return 0;
}