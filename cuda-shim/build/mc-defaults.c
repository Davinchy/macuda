// Linked into the Metal + CUDA (*-mc) binaries only, by link-null-metal.sh: CUDA graphs OFF unless MACUDA_SPLIT_GRAPHS=1.
// Why (2026-10-16, RTX 5090, llama.cpp ad6c668): on Qwen3.8-Flash-Next (MoE + Gated DeltaNet) split 27/21 with graphs on,
// llama-server faulted deterministically - an mmu fault at the same address in two separate processes, in a captured
// mul_mat_vec_q node - on the first request whose prompt was already fully cached (the recurrent-state checkpoint
// restore path; tools/decode_bench.py's `edit` prompts reach it on rep 1). Graphs off, the same run completed. The CUDA-only
// binaries on a dense hybrid model (Qwen3.8-27B) and the same model split 40/24 did NOT fault, and the split ran 16%
// faster with graphs (38.8 vs 33.3 tok/s): opt back in with MACUDA_SPLIT_GRAPHS=1 for models that are known to work.
// Upstream llama.cpp a30fb49 ran the same Flash-Next split graphs-on cleanly (token-identical, no speedup): a ggml bug in
// the pinned tree, fixed later. Revisit this default when the pin moves.
// ggml reads GGML_CUDA_DISABLE_GRAPHS when the CUDA backend initializes, after this constructor has run; a value already
// set in the environment is left alone.
#include <stdlib.h>
__attribute__((constructor)) static void macuda_split_defaults(void) {
  const char *g = getenv("MACUDA_SPLIT_GRAPHS");
  if (!(g && g[0] == '1')) setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);
}
