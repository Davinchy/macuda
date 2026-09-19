// libtinycudart, part 3: the CUDA-graph subset ggml-cuda uses. A capture is a recording of what one stream was asked to
// do between cudaStreamBeginCapture and cudaStreamEndCapture - launches with their marshalled parameters, asynchronous
// copies and fills - and an executable graph is a copy of that recording; launching it replays the recording on the
// stream it is launched on. Events recorded or waited on the capturing stream are ordering marks between nodes that
// this single-queue driver already honours by order, so they record nothing. Anything that would need the capturing
// stream to finish (a synchronise) is refused the way CUDA refuses it, and the capture is marked invalid.
//
// Why replay at all, when a replayed launch still costs the driver the same as a fresh one: the caller's own per-launch
// work is gone (ggml's dispatch above the runtime layer, ~0.4 us a launch after the -O2 build), and - the reason this
// exists - a recording is a token's worth of launches known in advance, which is what the driver needs to keep a
// token's descriptors resident and hand the engine one chain instead of thirteen (the resident-chain replay, next).
// ggml decides the capture boundary and re-captures when a graph's node properties change, so a recording is only
// ever replayed for a token whose kernels and parameters are the ones recorded - the same contract CUDA graphs have.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "tinynv.h"
#undef cudaStreamBeginCapture
#undef cudaStreamEndCapture
#undef cudaStreamIsCapturing
#undef cudaGraphLaunch
#undef cudaMemcpyAsync
#undef cudaMemsetAsync

extern tinynv_stream_t tinycudart_default_stream(void);
extern void tinycudart_set_error(int e);
extern void tinycudart_count_launch(const char *name);
extern void tinycudart_time_launch(const char *name, double ns);
extern double tinycudart_now_ns(void);
extern int tinycudart_trace(void);

typedef struct {
  int kind;   // 0 a launch, 1 a copy, 2 a fill
  tinynv_kernel_t k;
  const char *name;   // the registered name, which lives as long as the process
  unsigned g[3], b[3], smem;
  size_t plen;
  unsigned char *params;   // the marshalled constant-bank bytes, exactly what tinynv_launch was going to be handed
  void *dst;
  const void *src;
  size_t n;
  int mkind, value;
} gnode_t;

struct CUgraph_st { gnode_t *nodes; size_t n, cap; };
struct CUgraphExec_st { gnode_t *nodes; size_t n; unsigned long launched; };

static struct { int active, invalid; tinynv_stream_t nvs; struct CUgraph_st *g; } cap;
static unsigned long g_captures, g_graph_launches, g_nodes_replayed, g_nodes_recorded;

static tinynv_stream_t nvs_of(cudaStream_t s) { return (uintptr_t)s > 2 ? (tinynv_stream_t)s : tinycudart_default_stream(); }
static cudaError_t E(cudaError_t e) { tinycudart_set_error(e); return e; }

// Is this (driver) stream the one being captured? Called on the launch and copy paths before they touch the driver.
int tinycudart_capturing_nv(tinynv_stream_t s) { return cap.active && s == cap.nvs; }

static gnode_t *node_new(void) {
  struct CUgraph_st *g = cap.g;
  if (g->n == g->cap) {
    size_t c = g->cap ? g->cap * 2 : 512;
    gnode_t *p = realloc(g->nodes, c * sizeof(*p));
    if (!p) return NULL;
    g->nodes = p;
    g->cap = c;
  }
  gnode_t *nd = &g->nodes[g->n++];
  memset(nd, 0, sizeof(*nd));
  g_nodes_recorded++;
  return nd;
}

cudaError_t tinycudart_capture_launch(tinynv_kernel_t k, const char *name, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                                      unsigned by, unsigned bz, unsigned smem, const void *params, size_t plen) {
  gnode_t *nd = node_new();
  if (!nd) { cap.invalid = 1; return E(cudaErrorMemoryAllocation); }
  nd->kind = 0; nd->k = k; nd->name = name;
  nd->g[0] = gx; nd->g[1] = gy; nd->g[2] = gz; nd->b[0] = bx; nd->b[1] = by; nd->b[2] = bz; nd->smem = smem;
  nd->plen = plen;
  if (plen) {
    nd->params = malloc(plen);
    if (!nd->params) { cap.invalid = 1; return E(cudaErrorMemoryAllocation); }
    memcpy(nd->params, params, plen);
  }
  return E(cudaSuccess);
}
cudaError_t tinycudart_capture_memcpy(void *dst, const void *src, size_t n, int kind) {
  gnode_t *nd = node_new();
  if (!nd) { cap.invalid = 1; return E(cudaErrorMemoryAllocation); }
  nd->kind = 1; nd->dst = dst; nd->src = src; nd->n = n; nd->mkind = kind;
  return E(cudaSuccess);
}
cudaError_t tinycudart_capture_memset(void *p, int v, size_t n) {
  gnode_t *nd = node_new();
  if (!nd) { cap.invalid = 1; return E(cudaErrorMemoryAllocation); }
  nd->kind = 2; nd->dst = p; nd->value = v; nd->n = n;
  return E(cudaSuccess);
}
// A synchronise on the capturing stream cannot be honoured: refused, and the capture is spoiled.
cudaError_t tinycudart_capture_refuse(void) { cap.invalid = 1; return E(cudaErrorStreamCaptureUnsupported); }

static void free_nodes(gnode_t *nodes, size_t n) {
  for (size_t i = 0; i < n; i++) free(nodes[i].params);
  free(nodes);
}
static gnode_t *clone_nodes(const gnode_t *src, size_t n) {
  gnode_t *out = calloc(n ? n : 1, sizeof(*out));
  if (!out) return NULL;
  for (size_t i = 0; i < n; i++) {
    out[i] = src[i];
    if (src[i].plen) {
      out[i].params = malloc(src[i].plen);
      if (!out[i].params) { free_nodes(out, i); return NULL; }
      memcpy(out[i].params, src[i].params, src[i].plen);
    }
  }
  return out;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t s, enum cudaStreamCaptureMode mode) {
  (void)mode;   // relaxed or not, only the captured stream's own work is recorded here
  if (cap.active) return E(cudaErrorStreamCaptureUnsupported);
  cap.g = calloc(1, sizeof(*cap.g));
  if (!cap.g) return E(cudaErrorMemoryAllocation);
  cap.nvs = nvs_of(s);
  cap.active = 1;
  cap.invalid = 0;
  g_captures++;
  if (tinycudart_trace()) fprintf(stderr, "[trace] graph capture begins on stream %p\n", (void *)s);
  return E(cudaSuccess);
}
cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t *pGraph) {
  if (!cap.active || nvs_of(s) != cap.nvs) return E(cudaErrorStreamCaptureUnmatched);
  struct CUgraph_st *g = cap.g;
  cap.active = 0;
  cap.g = NULL;
  if (cap.invalid) { free_nodes(g->nodes, g->n); free(g); if (pGraph) *pGraph = NULL; return E(cudaErrorStreamCaptureInvalidated); }
  if (tinycudart_trace()) fprintf(stderr, "[trace] graph capture ends: %zu nodes\n", g->n);
  if (pGraph) *pGraph = (cudaGraph_t)g;
  return E(cudaSuccess);
}
cudaError_t cudaStreamIsCapturing(cudaStream_t s, enum cudaStreamCaptureStatus *status) {
  if (status) *status = tinycudart_capturing_nv(nvs_of(s)) ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
  return E(cudaSuccess);
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pExec, cudaGraph_t graph, unsigned long long flags) {
  (void)flags;
  struct CUgraph_st *g = (struct CUgraph_st *)graph;
  if (!g || !pExec) return E(cudaErrorInvalidValue);
  struct CUgraphExec_st *e = calloc(1, sizeof(*e));
  if (!e) return E(cudaErrorMemoryAllocation);
  e->nodes = clone_nodes(g->nodes, g->n);
  if (!e->nodes) { free(e); return E(cudaErrorMemoryAllocation); }
  e->n = g->n;
  *pExec = (cudaGraphExec_t)e;
  return E(cudaSuccess);
}
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t exec, cudaGraph_t graph, cudaGraphExecUpdateResultInfo *info) {
  struct CUgraphExec_st *e = (struct CUgraphExec_st *)exec;
  struct CUgraph_st *g = (struct CUgraph_st *)graph;
  if (!e || !g) return E(cudaErrorInvalidValue);
  gnode_t *fresh = clone_nodes(g->nodes, g->n);
  if (!fresh) return E(cudaErrorMemoryAllocation);
  free_nodes(e->nodes, e->n);
  e->nodes = fresh;
  e->n = g->n;
  if (info) { memset(info, 0, sizeof(*info)); info->result = cudaGraphExecUpdateSuccess; }
  return E(cudaSuccess);
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t s) {
  struct CUgraphExec_st *e = (struct CUgraphExec_st *)exec;
  if (!e) return E(cudaErrorInvalidValue);
  tinynv_stream_t nvs = nvs_of(s);
  for (size_t i = 0; i < e->n; i++) {
    gnode_t *nd = &e->nodes[i];
    if (nd->kind == 0) {
      tinycudart_count_launch(nd->name);
      double t0 = tinycudart_now_ns();
      tinynv_status_t st = tinynv_launch(nvs, nd->k, nd->g[0], nd->g[1], nd->g[2], nd->b[0], nd->b[1], nd->b[2], nd->smem, nd->params, nd->plen);
      tinycudart_time_launch(nd->name, tinycudart_now_ns() - t0);
      if (st != TINYNV_OK) {
        fprintf(stderr, "[tinycudart] graph launch: node %zu (%s) failed: %s (%s)\n", i, nd->name, tinynv_status_str(st), tinynv_last_error());
        return E(cudaErrorUnknown);
      }
    } else if (nd->kind == 1) {
      cudaError_t r = cudaMemcpyAsync(nd->dst, nd->src, nd->n, (enum cudaMemcpyKind)nd->mkind, s);
      if (r != cudaSuccess) return r;
    } else {
      cudaError_t r = cudaMemsetAsync(nd->dst, nd->value, nd->n, s);
      if (r != cudaSuccess) return r;
    }
  }
  e->launched++;
  g_graph_launches++;
  g_nodes_replayed += e->n;
  return E(cudaSuccess);
}
cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  struct CUgraph_st *g = (struct CUgraph_st *)graph;
  if (g) { free_nodes(g->nodes, g->n); free(g); }
  return E(cudaSuccess);
}
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
  struct CUgraphExec_st *e = (struct CUgraphExec_st *)exec;
  if (e) { free_nodes(e->nodes, e->n); free(e); }
  return E(cudaSuccess);
}
// At exit, whenever a capture happened: how much of the run was replayed rather than issued by the caller.
void tinycudart_graph_stats(void) {
  if (!g_captures) return;
  fprintf(stderr, "[tinycudart] graphs: %lu captures recorded %lu nodes; %lu graph launches replayed %lu nodes (%.1f a launch)\n",
          g_captures, g_nodes_recorded, g_graph_launches, g_nodes_replayed,
          g_graph_launches ? (double)g_nodes_replayed / (double)g_graph_launches : 0.0);
}
