# Source from the repository root:   . ./env.sh
# Nothing here is needed to BUILD the shim (setup.sh / cuda-shim/Makefile find their own paths). It is for the operating
# tools under tools/, which use these:
export EGPU_ROOT=${EGPU_ROOT:-$(pwd)}
# Two tinygrad checkouts are referenced, both branches of https://github.com/Davinchy/tinygrad:
#   TINYGRAD_SRC    the driver's offline reference tests and mock TinyGPU server (cuda-shim/libtinynv/tools/nv_reference_*.py,
#                   libtinynv/test/mock_server.py, build_spike_cubin.py): branch egpu-hcq2-remote @ 084d89582 — master-based, it
#                   has runtime/autogen/nv_570 (the structures the driver is asserted against) and test/unit/test_remote_pci.py.
#   TINYGRAD_STABLE the card-recovery tools (tools/nv_e3_flr.py, nv_temp.py, nv_e2_regs.py, nvmini.py, nv_quiesce.sh), which need
#                   APLRemotePCIDevice — the macOS remote-PCI path upstream removed on 2026-09-05: branch egpu-5090-stable
#                   @ 5bf561309 (33cd373ad, the last CI-tested macOS TinyGPU recipe, plus fixes).
# Each with a Python 3.12 venv in $EGPU_ROOT/venv that has it installed editable (pip install -e "$TINYGRAD_SRC").
export TINYGRAD_SRC=${TINYGRAD_SRC:-$EGPU_ROOT/tinygrad}
export TINYGRAD_STABLE=${TINYGRAD_STABLE:-$EGPU_ROOT/tinygrad-stable}
export PYTHONPATH=${EGPU_TINYGRAD_FOR:-$TINYGRAD_STABLE}${PYTHONPATH:+:$PYTHONPATH}   # the recovery tools by default
[ -d "$EGPU_ROOT/venv/bin" ] && export PATH=$EGPU_ROOT/venv/bin:$PATH VIRTUAL_ENV=$EGPU_ROOT/venv
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$EGPU_ROOT/.cache}   # tinygrad's downloads (the TinyGPU zip, firmware) stay in the tree
# The device compile (ggml-cuda's .cu files to sm_120a fatbins, the shim's own cubins) runs nvcc on a Linux box over ssh:
#   export TINYCC_HOST=user@host        # required by cuda-shim/build/tinycc and build/fetch-cuda-headers.sh
#   export TINYCC_KEY=~/.ssh/id_ed25519 # optional identity file
#   export TINYCC_GGML_LINUX=ggml       # where llama.cpp/ggml is synced on that box (relative to $HOME there)
