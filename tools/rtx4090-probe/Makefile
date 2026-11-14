# rtx4090-probe — standalone supervised AD102 unlock transaction. No GPU is touched by building it.
CC      ?= cc
NV      := ../../cuda-shim/libtinynv
SDK     := $(shell sh ../../cuda-shim/build/sdk.sh 2>/dev/null)
CFLAGS  ?= -std=c11 -O2 -g -Wall -Wextra -Werror -Wno-unused-parameter
ifneq ($(SDK),)
CFLAGS  += -isysroot $(SDK)
endif
CFLAGS  += -I$(NV)/src -I$(NV)/include -I../../cuda-shim/include

.PHONY: all clean check
all: build/rtx4090-probe

$(NV)/build/libtinynv.a:
	$(MAKE) -C $(NV) build/libtinynv.a

build/rtx4090-probe: probe.c $(NV)/build/libtinynv.a
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ probe.c $(NV)/build/libtinynv.a

# the C asset pins must equal the bundle manifest pins; drift is a build failure
check: build/rtx4090-probe
	python3 check_pins.py

clean:
	rm -rf build
