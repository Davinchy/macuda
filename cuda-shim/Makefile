# cuda-shim — builds the three shim libraries and links them together (no GPU needed; the driver runs as a null device).
#   make        libtinynv.a (B's driver) + libtinycudart.a + libtinycublas.a + a link smoke test through the real CUDA headers
#   make test   libtinynv's suite, then the link smoke test
ROOT     := $(abspath .)
CC       := /opt/homebrew/opt/llvm/bin/clang
CXX      := /opt/homebrew/opt/llvm/bin/clang++
LLVM_INC := /opt/homebrew/opt/llvm/include
DEMANGLE := /opt/homebrew/opt/llvm/lib/libLLVMDemangle.a
# Which macOS SDK: build/sdk.sh picks the first candidate that actually links on this machine (the toolchain changed
# three times on 2026-09-14 and took the pinned SDK with it; see the script). Both compilers get it explicitly, since
# Homebrew clang hardcodes its own sysroot and ignores SDKROOT; libtinynv's Apple cc gets it on the sub-make's command
# line rather than through an exported SDKROOT, which xcrun would also apply to make, ar and strings.
SDK      := $(shell sh $(ROOT)/build/sdk.sh)
ifeq ($(SDK),)
$(error no usable macOS SDK on this machine; see build/sdk.sh)
endif
SYSROOT  := -isysroot $(SDK)
CUDA_INC ?= $(ROOT)/cuda-13/include
CFLAGS   := $(SYSROOT) -std=c11 -O2 -g -Wall -Wextra -Wno-unused-parameter -fPIC -I$(ROOT)/include -I$(ROOT)/libtinycudart
BUILD    := build/shim
# libtinynv is built into OUR build dir, never into libtinynv/build/, which Session B builds in concurrently
NV_BUILD := $(ROOT)/$(BUILD)/nv
NV_LIB   := $(NV_BUILD)/libtinynv.a
CUDART   := $(BUILD)/libtinycudart.a
CUBLAS   := $(BUILD)/libtinycublas.a

.PHONY: all test clean FORCE
all: $(NV_LIB) $(CUDART) $(CUBLAS) $(BUILD)/link_smoke

$(NV_LIB): FORCE
	@$(MAKE) --no-print-directory -C libtinynv BUILD=$(NV_BUILD) CC="cc $(SYSROOT)"

$(CUDART): $(BUILD)/cudart.o $(BUILD)/fatbin.o $(BUILD)/hostpool.o $(BUILD)/cudart_api.o $(BUILD)/graph.o $(BUILD)/demangle.o $(BUILD)/copy2d_cubin.o $(BUILD)/copy1d_cubin.o $(BUILD)/keepalive_cubin.o
	ar rcs $@ $^
# the strided-copy kernel travels inside the runtime the same way (libtinycudart/copy2d.cu, built by the remote nvcc)
$(BUILD)/copy2d_cubin.c: libtinycudart/copy2d.cubin
	@mkdir -p $(@D)
	cd libtinycudart && xxd -i copy2d.cubin > $(ROOT)/$@
$(BUILD)/copy2d_cubin.o: $(BUILD)/copy2d_cubin.c
	$(CC) $(CFLAGS) -c $< -o $@
# the 1-D copy kernel the driver launches for downloads when TINYNV_DOWNLOAD_VIA_COMPUTE=1 (libtinycudart/copy1d.cu)
$(BUILD)/copy1d_cubin.c: libtinycudart/copy1d.cubin
	@mkdir -p $(@D)
	cd libtinycudart && xxd -i copy1d.cubin > $(ROOT)/$@
$(BUILD)/copy1d_cubin.o: $(BUILD)/copy1d_cubin.c
	$(CC) $(CFLAGS) -c $< -o $@
# the keep-alive kernel the driver launches after a synchronisation when TINYNV_KEEPALIVE=1 (libtinycudart/keepalive.cu)
$(BUILD)/keepalive_cubin.c: libtinycudart/keepalive.cubin
	@mkdir -p $(@D)
	cd libtinycudart && xxd -i keepalive.cubin > $(ROOT)/$@
$(BUILD)/keepalive_cubin.o: $(BUILD)/keepalive_cubin.c
	$(CC) $(CFLAGS) -c $< -o $@
$(CUBLAS): $(BUILD)/cublas.o $(BUILD)/gemm_cubin.o
	ar rcs $@ $^

$(BUILD)/%.o: libtinycudart/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/demangle.o: libtinycudart/demangle.cpp
	@mkdir -p $(@D)
	$(CXX) $(SYSROOT) -std=c++17 -O2 -fPIC -I$(LLVM_INC) -c $< -o $@
$(BUILD)/cudart_api.o: libtinycudart/cudart_api.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(CUDA_INC) -c $< -o $@
$(BUILD)/graph.o: libtinycudart/graph.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(CUDA_INC) -c $< -o $@
$(BUILD)/cublas.o: libtinycublas/cublas.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(CUDA_INC) -c $< -o $@
# the GEMM cubin travels inside the library, so a linked binary needs no files at run time
$(BUILD)/gemm_cubin.c: libtinycublas/gemm.cubin
	@mkdir -p $(@D)
	cd libtinycublas && xxd -i gemm.cubin > $(ROOT)/$@
$(BUILD)/gemm_cubin.o: $(BUILD)/gemm_cubin.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/link_smoke: test/link_smoke.c $(CUDART) $(CUBLAS) $(NV_LIB)
	$(CC) $(CFLAGS) -I$(CUDA_INC) $< $(CUDART) $(CUBLAS) $(NV_LIB) $(DEMANGLE) -lc++ -o $@

test: all coverage
	@$(MAKE) --no-print-directory -C libtinynv test BUILD=$(NV_BUILD) CC="cc $(SYSROOT)"
	$(BUILD)/link_smoke

# every cuda*/cublas* symbol the compiled ggml-cuda archive needs must be defined by the shim (skipped if the archive is absent)
.PHONY: coverage
coverage: $(CUDART) $(CUBLAS)
	@if [ -f build/libggml-cuda.a ]; then \
	  nm -u build/libggml-cuda.a | awk '{print $$NF}' | grep -E '^_(cuda|cublas|__cuda)[A-Z_]' | sort -u > $(BUILD)/need.txt; \
	  nm $(CUDART) $(CUBLAS) | awk '$$2 ~ /^[TD]$$/ {print $$3}' | sort -u > $(BUILD)/have.txt; \
	  missing=$$(comm -13 $(BUILD)/have.txt $(BUILD)/need.txt); \
	  if [ -n "$$missing" ]; then echo "coverage: libggml-cuda.a needs symbols the shim does not define:"; echo "$$missing" | sed 's/^/  /'; exit 1; fi; \
	  echo "coverage: all $$(wc -l < $(BUILD)/need.txt | tr -d ' ') cuda*/cublas* symbols libggml-cuda.a needs are defined by the shim"; \
	else echo "coverage: build/libggml-cuda.a not present, skipped"; fi

clean:
	rm -rf $(BUILD)
