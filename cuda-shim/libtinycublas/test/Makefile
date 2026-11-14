# BF16 runtime-handoff and correctness test against shim libraries built for the target card.
# The shim itself is built by cuda-shim/Makefile (setup.sh shim); for Ada:
#   cd cuda-shim && make BUILD=build/shim-sm_89 CUBIN_ARCH=sm_86
# then here: make SHIM=../../build/shim-sm_89
ROOT := $(abspath ../..)
SHIM ?= $(ROOT)/build/shim-sm_89
CC := /opt/homebrew/opt/llvm/bin/clang
SDK := $(shell sh $(ROOT)/build/sdk.sh)
CUDA_INC ?= $(ROOT)/cuda-13/include
DEMANGLE := /opt/homebrew/opt/llvm/lib/libLLVMDemangle.a
CFLAGS := -isysroot $(SDK) -std=c11 -O2 -g -Wall -Wextra -Werror -Wno-unused-parameter \
          -I$(CUDA_INC) -I$(ROOT)/include -I$(ROOT)/libtinynv/include -I$(ROOT)/libtinynv/src

.PHONY: all clean
all: bf16_unlock_gemm

bf16_unlock_gemm: bf16_unlock_gemm.c $(SHIM)/libtinycudart.a $(SHIM)/libtinycublas.a $(SHIM)/nv/libtinynv.a
	$(CC) $(CFLAGS) $< $(SHIM)/libtinycudart.a $(SHIM)/libtinycublas.a $(SHIM)/nv/libtinynv.a $(DEMANGLE) -lc++ -o $@

clean:
	rm -f bf16_unlock_gemm
