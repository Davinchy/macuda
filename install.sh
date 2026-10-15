#!/bin/sh
# install.sh — take a fresh Apple Silicon Mac to a working build of this project, checking each prerequisite first and
# saying plainly what is missing and why it is needed.
#
#   sh install.sh           check, then install what is missing and build everything (asks before each install)
#   sh install.sh check     report only — installs nothing, builds nothing, touches nothing
#   sh install.sh build     skip the prerequisite checks and build (a second run, after check passed once)
#   YES=1 sh install.sh     do not ask; assume yes to every install prompt
#
# WHAT THIS DOES NOT DO, because it cannot: approve the DriverKit system extension (macOS asks you, in System Settings),
# plug the card in, or replug it when it wedges. Those are yours. Everything else is here.
#
# The one genuinely Linux-only piece is nvcc, which compiles ggml's ~190 CUDA translation units for the card. This
# installer runs it in a CUDA container instead of on a separate Linux machine: nvidia/cuda publishes arm64 images, so
# nvcc runs natively on Apple Silicon, and because it only ever compiles device code (--fatbin, never executed here) the
# container needs no GPU, no NVIDIA driver and no special runtime. Either runtime serves: Apple's own `container`
# (macOS 26+, a lightweight VM per container, no desktop app) when it is installed, Docker otherwise. The alternative —
# a Linux box over ssh — is still supported and faster if you have one: set TINYCC_HOST=user@box instead.
set -u
R=$(cd "$(dirname "$0")" && pwd); cd "$R"
mode=${1:-all}
CUDA_IMAGE=${TINYCC_CUDA_IMAGE:-nvidia/cuda:13.0.3-devel-ubuntu24.04}
TINYGPU_URL=https://github.com/tinygrad/tinygpu_releases/raw/c0d024f9ff0e1dc8fdf217f255da7101d91e8323/TinyGPU.zip
miss=0; warn=0
ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mMISSING\033[0m %s\n         -> %s\n' "$1" "$2"; miss=$((miss+1)); }
note() { printf '  \033[33mnote\033[0m  %s\n' "$1"; warn=$((warn+1)); }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }
ask() { # ask "what" ; 0 = go ahead
  [ "${YES:-0}" = 1 ] && return 0
  [ -t 0 ] || { echo "         (not a terminal and YES=1 not set — skipping: $1)"; return 1; }
  printf '         %s [y/N] ' "$1"; read a < /dev/tty; case "$a" in y|Y|yes) return 0;; *) return 1;; esac
}

# ---------------------------------------------------------------- the machine
head_ "The machine"
[ "$(uname -s)" = Darwin ] || { echo "  This is a macOS project; uname says $(uname -s). Stopping."; exit 1; }
[ "$(uname -m)" = arm64 ] || note "uname says $(uname -m), not arm64 — built and measured only on Apple Silicon"
ok "macOS $(sw_vers -productVersion) on $(uname -m)"
free_gb=$(df -g . | awk 'NR==2{print $4}')
[ "${free_gb:-0}" -ge 60 ] && ok "${free_gb} GB free here (the build needs ~25, models want far more)" \
                           || note "only ${free_gb} GB free here: the build alone needs ~25 GB, before any model"
if system_profiler SPPCIDataType 2>/dev/null | grep -q "0x2b85\|NVIDIA"; then
  ok "an NVIDIA card is on the PCI bus"
else
  note "no NVIDIA card found on the PCI bus — everything still BUILDS, but nothing can run until an RTX 5090 is"
  note "  plugged in over Thunderbolt (that hardware is the point of the project; see README §What you need)"
fi

# ---------------------------------------------------------------- tools
head_ "Build tools"
if xcode-select -p > /dev/null 2>&1; then ok "Xcode command line tools at $(xcode-select -p)"
else bad "Xcode command line tools (the C compiler, the linker and the macOS SDK)" "run: xcode-select --install, then re-run this script"
fi
if command -v brew > /dev/null 2>&1; then ok "Homebrew at $(command -v brew)"
else bad "Homebrew (used for llvm and cmake)" 'install it from https://brew.sh — then re-run this script'
fi
for f in cmake python3; do
  command -v $f > /dev/null 2>&1 && ok "$f at $(command -v $f)" || bad "$f" "run: brew install $f"
done
if [ -x /opt/homebrew/opt/llvm/bin/clang++ ]; then ok "Homebrew LLVM ($(/opt/homebrew/opt/llvm/bin/clang++ --version | head -1))"
else bad "Homebrew LLVM at /opt/homebrew/opt/llvm (its clang does the CUDA host compile; Apple's clang cannot)" "run: brew install llvm"
fi
if sh cuda-shim/build/sdk.sh > /dev/null 2>&1; then ok "a macOS SDK that links: $(sh cuda-shim/build/sdk.sh)"
else bad "a macOS SDK that links a one-line program" "install/select Xcode or the command line tools; see cuda-shim/build/sdk.sh"
fi

# ---------------------------------------------------------------- the linux half
head_ "The CUDA (Linux) half"
if [ -n "${TINYCC_HOST:-}" ]; then
  ok "TINYCC_HOST=$TINYCC_HOST — the device compile will use that box over ssh, and no container runtime is needed"
elif command -v container > /dev/null 2>&1; then
  # Apple's own runtime, preferred where it exists: a lightweight VM per container on Virtualization.framework, no
  # Docker Desktop to install or keep running. It takes the same run/-v/--network=none invocation Docker does.
  ok "Apple's container at $(command -v container)"
  if container system status > /dev/null 2>&1; then ok "its service is running"
  else bad "the container service (installed, but not started)" "run: container system kernel set --recommended && container system start"
  fi
elif command -v docker > /dev/null 2>&1; then
  ok "docker at $(command -v docker)"
  if docker info > /dev/null 2>&1; then ok "the Docker daemon is running"
  else bad "the Docker daemon (installed, but not running)" "open Docker Desktop (open -a Docker) and wait for it to say Running"
  fi
elif [ "$(sw_vers -productVersion | cut -d. -f1)" -ge 26 ] 2>/dev/null; then
  bad "a container runtime for nvcc, or a Linux box" "run: brew install container   (Apple's, macOS 26+), or brew install --cask docker, or set TINYCC_HOST=user@box"
else
  bad "Docker, or a Linux box with nvcc" "run: brew install --cask docker   (then open it once), or set TINYCC_HOST=user@box"
fi

# ---------------------------------------------------------------- the driver
head_ "The card's driver (tinygrad's, not NVIDIA's)"
if [ -d /Applications/TinyGPU.app ]; then ok "TinyGPU.app is installed"
  if pgrep -qf 'org\.tinygrad\.tinygpu\.driver2'; then ok "its DriverKit extension is loaded and running"
  else note "TinyGPU.app is there but its extension is not running: run"
       note "  /Applications/TinyGPU.app/Contents/MacOS/TinyGPU install   and approve it in System Settings"
  fi
else bad "TinyGPU.app and its DriverKit extension (this is what talks to the card at all)" "this script can download and install it for you — see below"
fi

if [ "$mode" = check ]; then
  printf '\n%s missing, %s notes.\n' "$miss" "$warn"
  [ "$miss" -eq 0 ] && echo "Nothing is missing: run  sh install.sh  to build." || echo "Fix the MISSING lines above, then run: sh install.sh check"
  exit $([ "$miss" -eq 0 ] && echo 0 || echo 1)
fi

# ---------------------------------------------------------------- install what is missing
if [ "$mode" != build ] && [ "$miss" -gt 0 ]; then
  head_ "Installing what is missing"
  command -v brew > /dev/null 2>&1 || { echo "  Homebrew has to be installed first (https://brew.sh). Stopping."; exit 1; }
  xcode-select -p > /dev/null 2>&1 || { echo "  Run xcode-select --install first (macOS puts up its own dialog). Stopping."; exit 1; }
  for f in cmake llvm; do
    case $f in cmake) command -v cmake > /dev/null 2>&1 && continue;; llvm) [ -x /opt/homebrew/opt/llvm/bin/clang++ ] && continue;; esac
    ask "brew install $f ?" && brew install $f
  done
  # Apple's runtime first on macOS 26+: a formula rather than a 1 GB desktop app, and nothing to leave running.
  if [ -z "${TINYCC_HOST:-}" ] && ! command -v container > /dev/null 2>&1 && ! command -v docker > /dev/null 2>&1 \
     && [ "$(sw_vers -productVersion | cut -d. -f1)" -ge 26 ] 2>/dev/null; then
    ask "brew install container ? (Apple's container runtime — it runs the nvcc container in a lightweight VM)" \
      && brew install container && container system kernel set --recommended && container system start
  fi
  if [ -z "${TINYCC_HOST:-}" ] && ! command -v container > /dev/null 2>&1 && ! command -v docker > /dev/null 2>&1; then
    ask "brew install --cask docker ? (Docker Desktop, ~1 GB — it runs the nvcc container)" && brew install --cask docker
  fi
  if command -v container > /dev/null 2>&1 && ! container system status > /dev/null 2>&1 && [ -z "${TINYCC_HOST:-}" ]; then
    ask "start Apple's container service now?" \
      && { container system kernel set --recommended > /dev/null 2>&1; container system start; }
  fi
  if command -v docker > /dev/null 2>&1 && ! command -v container > /dev/null 2>&1 \
     && ! docker info > /dev/null 2>&1 && [ -z "${TINYCC_HOST:-}" ]; then
    ask "start Docker Desktop now and wait for it?" && { open -a Docker; printf '         waiting for the Docker daemon'
      i=0; until docker info > /dev/null 2>&1 || [ $i -ge 90 ]; do printf '.'; sleep 2; i=$((i+1)); done; echo
      docker info > /dev/null 2>&1 && echo "         Docker is up" || { echo "         Docker did not come up; start it by hand and re-run"; exit 1; }; }
  fi
  if [ ! -d /Applications/TinyGPU.app ]; then
    if ask "download tinygrad's signed TinyGPU.app and unzip it to /Applications?"; then
      tmp=$(mktemp -d); curl -fL --progress-bar "$TINYGPU_URL" -o "$tmp/TinyGPU.zip" && unzip -q "$tmp/TinyGPU.zip" -d /Applications && rm -rf "$tmp"
      echo "         installed. NOW RUN, and approve the extension when macOS asks:"
      echo "           /Applications/TinyGPU.app/Contents/MacOS/TinyGPU install"
      echo "         (System Settings > General > Login Items & Extensions > Driver Extensions)"
    fi
  fi
fi

# ---------------------------------------------------------------- build
head_ "Building (llama.cpp ~5 min; the CUDA half ~3-5 min once the image is here, plus a 4 GB pull the first time)"
set -e
echo "== 1/6 headers and firmware (NVIDIA's open-gpu-kernel-modules headers, GSP firmware, CUDA headers)"
sh setup.sh deps
echo "== 2/6 llama.cpp at the pinned commit, CPU-only static tree"
sh setup.sh llama
echo "== 3/6 stable-diffusion.cpp"
sh setup.sh sd
echo "== 4/6 the shim itself (libtinynv + libtinycudart + libtinycublas; no GPU needed)"
sh setup.sh shim
echo "== 5/6 ggml's CUDA backend — device code through nvcc, host code through clang"
if [ -z "${TINYCC_HOST:-}" ]; then
  # Whichever runtime the shim's own scripts will use, asked once so the image is there before 190 compiles start
  # racing for it. `image inspect` and `image pull` are spelled the same by both.
  RT=$(sh cuda-shim/build/container-runtime.sh)
  "$RT" image inspect "$CUDA_IMAGE" > /dev/null 2>&1 || { echo "   pulling $CUDA_IMAGE with $RT (~4 GB, once)"; "$RT" image pull "$CUDA_IMAGE"; }
fi
sh cuda-shim/build/build-ggml-cuda.sh
echo "== 6/6 linking the binaries against the shim"
sh setup.sh link

# ---------------------------------------------------------------- what now
head_ "Built. What now"
echo "  1. Check the card is there and healthy, before anything touches it:"
echo "       sh tools/preflight.sh                # read-only; must say VERDICT: OK"
echo "  2. Get a model (any GGUF llama.cpp can read) into models/ — see models/README.md"
echo "  3. Run one:"
echo "       sh tools/nv_shim_step.sh A bench models/<your-model>.gguf"
echo "     which does the whole protocol for you: preflight, take the lock, start the server, run, release."
echo
echo "  Read README §Run before driving the card yourself: one process on it at a time, and a wedged card needs"
echo "  a physical replug that no script can do."
