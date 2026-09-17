# Source from the repository root:   . ./env.sh
# Nothing here is needed to BUILD the shim (setup.sh / cuda-shim/Makefile find their own paths). It is for the operating
# tools under tools/, which use these:
export EGPU_ROOT=${EGPU_ROOT:-$(pwd)}
# Two tinygrad checkouts are referenced, both branches of https://github.com/Davinchy/tinygrad:
#   TINYGRAD_SRC    the driver's offline reference tests and mock TinyGPU server (cuda-shim/libtinynv/tools/nv_reference_*.py,
#                   libtinynv/test/mock_server.py, build_spike_cubin.py): branch egpu-hcq2-remote @ 0ff2c9299 — master-based, it
#                   has runtime/autogen/nv_570 (the structures the driver is asserted against) and test/unit/test_remote_pci.py.
#                   (2026-09-17: the pin here read 084d89582, which is not reachable on the remote. test/reference-hw.txt records
#                   that it was generated with 0ff2c9299 — the current branch head — so that, not 084d89582, is the operative revision.)
#   TINYGRAD_STABLE the card-recovery tools (tools/nv_e3_flr.py, nv_temp.py, nv_e2_regs.py, nvmini.py, nv_quiesce.sh), which need
#                   APLRemotePCIDevice — the macOS remote-PCI path upstream removed on 2026-09-05: branch egpu-5090-stable
#                   @ 5bf561309 (33cd373ad, the last CI-tested macOS TinyGPU recipe, plus fixes).
#                   (2026-09-17: 5bf561309 and the branch are gone with the old drive and were never pushed. What is checked out
#                   here is a worktree at the 33cd373ad base, branch egpu-5090-stable-rebuilt; the 'plus fixes' are NOT present.)
# Each with a Python 3.12 venv in $EGPU_ROOT/venv that has it installed editable (pip install -e "$TINYGRAD_SRC").
export TINYGRAD_SRC=${TINYGRAD_SRC:-$EGPU_ROOT/tinygrad}
export TINYGRAD_STABLE=${TINYGRAD_STABLE:-$EGPU_ROOT/tinygrad-stable}
export PYTHONPATH=${EGPU_TINYGRAD_FOR:-$TINYGRAD_STABLE}${PYTHONPATH:+:$PYTHONPATH}   # the recovery tools by default
[ -d "$EGPU_ROOT/venv/bin" ] && export PATH=$EGPU_ROOT/venv/bin:$PATH VIRTUAL_ENV=$EGPU_ROOT/venv
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$EGPU_ROOT/.cache}   # tinygrad's downloads (the TinyGPU zip, firmware) stay in the tree
# The device compile (ggml-cuda's .cu files to fatbins, the shim's own cubins) runs nvcc in a container by default — no GPU
# and no second machine needed, and on Apple silicon the image is native arm64:
#   export TINYCC_ARCH=sm_86            # sm_86 for the 3060 (Ampere), sm_120a for the 5090 (Blackwell)
#   export TINYCC_DOCKER_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04   # the default; any CUDA devel image with nvcc will do
# Setting TINYCC_HOST instead restores the original ssh path to a Linux box with nvcc:
#   export TINYCC_HOST=user@host        # required by cuda-shim/build/tinycc and build/fetch-cuda-headers.sh
#   export TINYCC_KEY=~/.ssh/id_ed25519 # optional identity file
#   export TINYCC_GGML_LINUX=ggml       # where llama.cpp/ggml is synced on that box (relative to $HOME there)
# ONE CARD, ONE LOCK. If the working tree at /Volumes/512SSD/EGPU exists on this Mac, share its lock and its DART-stale marker:
# two checkouts each holding their own lock can both believe they own the card, and two processes on the card wedge it.
if [ -d /Volumes/512SSD/EGPU ] && [ "$EGPU_ROOT" != /Volumes/512SSD/EGPU ]; then
  export GPU_LOCK=${GPU_LOCK:-/Volumes/512SSD/EGPU/.gpu-lock}
  export DART_STALE=${DART_STALE:-/Volumes/512SSD/EGPU/.dart-stale}
fi
