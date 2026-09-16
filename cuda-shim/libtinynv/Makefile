# libtinynv — the C userspace NVIDIA driver (M3). See docs/driver/libtinynv-design.md.
# make        build the library
# make test   run every test that needs no GPU (today: the TinyGPU backend against the python mock server)
CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -g -Wall -Wextra -Werror -Wno-unused-parameter -fPIC -Iinclude -I../include
# Pick the same SDK the rest of the tree picks, even when this makefile is run on its own. The top level passes
# CC="cc -isysroot ..." and that still wins; this is for the direct sub-make, which otherwise takes Apple cc's default.
# It is not a style point: it produced driver test binaries stamped sdk 27.0 while every other binary on the box was
# 26.5, which is a difference nobody intended sitting underneath a measurement, and Session A found it rather than the
# build saying so. The machine has changed three times in a day; a build that chooses for itself is the only kind that
# stays right.
SDK     := $(shell sh $(dir $(lastword $(MAKEFILE_LIST)))../build/sdk.sh 2>/dev/null)
ifneq ($(SDK),)
CFLAGS  += -isysroot $(SDK)
endif
# absolute, from the source tree: a relative firmware path finds the images from one directory and not from another,
# which turns a missing fetch into a test that passes without testing anything
CFLAGS  += -DTINYNV_FW_DEFAULT_DIR=\"$(abspath third_party/firmware)\"
PYTHON  ?= python3
# NVIDIA's own headers, fetched by tools/fetch_nv_headers.sh and never committed (see .gitignore)
NVSRC   := third_party/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5
NVINC   := -I$(NVSRC)/src/nvidia/arch/nvalloc/common/inc -I$(NVSRC)/src/common/sdk/nvidia/inc -I$(NVSRC)/src/common/inc \
           -I$(NVSRC)/src/nvidia/inc -I$(NVSRC)/src/common/shared/msgq/inc -I$(NVSRC)/src/common/uproc/os/common/include
BUILD   := build
# track header dependencies. without this a change to a struct in a header rebuilds only the .c files that were edited,
# and the rest of the library keeps the old layout: the two halves then disagree about where a field is, which shows up
# as a value that was set a moment ago reading back as zero. cost me a debugging round; never again.
DEPFLAGS := -MMD -MP

# Which commit this build came from, and whether the tree was clean. Baked into one object so only that one rebuilds
# when it changes. Three runs in this project have been diagnosed against code they were not running; a binary that can
# say what it is turns "are you running the fix?" into a question with an answer.
# Scoped to this directory on purpose: the repository holds the shim as well, and a driver that called itself dirty
# because somebody else had an edit open would be reporting the wrong thing.
BUILD_ID := $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)$(shell git diff --quiet -- . 2>/dev/null || echo -dirty)
$(BUILD)/src/build_id.o: CFLAGS += -DTINYNV_BUILD_ID='"$(BUILD_ID)"'

# And the object has to be rebuilt every time, which is the only way make can be right about this: nothing in the source
# changes when the commit does, only a -D on the command line, so there is no timestamp for make to compare. The first
# attempt here went through a stamp file rewritten only when the value changed - and make compares the stamp's
# timestamp, which had not changed either, so the version marker went stale within minutes of being written, which is
# precisely the failure it exists to catch. Recompiling one function on every build is not a cost worth being clever
# about.
# `always` is a target with no recipe that is never up to date, so anything depending on it is rebuilt every time.
#
# It must not become the default goal. The first target a makefile defines is what a bare `make` builds, and this rule
# sits above `all` because it belongs with the value it exists for - so the goal is named explicitly below. Without that
# line, `make` silently did nothing at all, which is a worse failure than the stale version marker this was written to
# fix, and it presented the same way: a build that looked fine and changed nothing.
.DEFAULT_GOAL := all
.PHONY: always
always:
$(BUILD)/src/build_id.o: always

UNAME := $(shell uname -s)
SRCS  := src/err.c src/build_id.c src/qmd.c src/image.c src/arena.c src/exec.c src/cubin.c src/sha256.c src/fw.c src/dev.c src/mmu.c src/tlsf.c src/pt.c src/gpu.c src/gsp.c src/flcn.c src/submit.c \
         src/tinynv.c src/pci_replay.c src/pci_tinygpu.c
ifeq ($(UNAME),Linux)
SRCS  += src/pci_sysfs.c
endif
OBJS  := $(SRCS:%.c=$(BUILD)/%.o)

.PHONY: all test clean
# The smi tool is part of a build, not of a test run. It was reachable only through `make test`, which meant the one
# command anybody would run to get it did not produce it and it sat in a directory nobody had reason to look in.
all: $(BUILD)/libtinynv.a $(BUILD)/tinynv-smi

$(BUILD)/libtinynv.a: $(OBJS)
	@mkdir -p $(@D)
	ar rcs $@ $^

$(BUILD)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

# The counts in this test's summary line are grepped out of the test itself at build time. They were literals and had
# gone stale, reporting the same two numbers whatever the file held - so the line that says how much was checked was the
# one thing in the suite not checking anything.
$(BUILD)/tinynv-smi: tools/tinynv_smi.c $(BUILD)/libtinynv.a
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -Isrc -o $@ tools/tinynv_smi.c $(BUILD)/libtinynv.a

$(BUILD)/test_headers: test/test_headers.c src/nv_structs.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -Isrc $(NVINC) \
	  -DTINYNV_HDR_STRUCTS=$$(grep -c '^SAME_SIZE(' test/test_headers.c) \
	  -DTINYNV_HDR_FIELDS=$$(grep -c '^SAME_AT(' test/test_headers.c) \
	  -DTINYNV_HDR_CONSTS=$$(grep -c '^_Static_assert(' test/test_headers.c) $< -o $@

$(BUILD)/test_arena: test/test_arena.c src/arena.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -Isrc -o $@ test/test_arena.c src/arena.c

# The check floors are grepped out of each test at build time, never written as literals: a count stated as a constant
# stops counting the moment the file changes, and stops silently. Same reason test_headers' summary is grepped.
$(BUILD)/test_submit: test/test_submit.c $(BUILD)/libtinynv.a
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -Isrc -DTINYNV_SUBMIT_CHECKS=$$(grep -c 'CHECK(' test/test_submit.c) \
	  $< $(BUILD)/libtinynv.a -o $@

$(BUILD)/test_exec_mode: test/test_exec_mode.c $(BUILD)/libtinynv.a
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -Isrc -DTINYNV_MODE_CHECKS=$$(grep -c 'CHECK(' test/test_exec_mode.c) \
	  $< $(BUILD)/libtinynv.a -o $@

$(BUILD)/test_%: test/test_%.c $(BUILD)/libtinynv.a
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -Isrc $< $(BUILD)/libtinynv.a -o $@

# the mock server exits on its own when the socket disappears, so a failed test cannot leave one behind
TRACE ?= $(abspath ../../traces)/5090-boot-smoke
# the launch snapshot from the combined recording: the descriptors and command buffers the card was handed
SNAPSHOT ?= $(abspath ../../traces)/5090-launch2-launch.json

# Opening the TinyGPU socket is a card step even when no GPU work follows: the server's listen backlog is one, so a
# stray connection from an offline test takes the live server's only client slot and blocks there. It happened on
# 2026-09-14 - a probe that defaulted to $TMPDIR/tinygpu.sock sat unaccepted for five minutes behind llama-server.
#
# Nothing in this target wants a real card. The tests that need a socket are handed a mock one as an argument, which
# already wins over the environment, so this changes nothing about how they run - it removes the possibility that a
# future test forgets to pass one and silently inherits a live server instead. Structural rather than remembered.
test: export TINYNV_SOCKET :=
test: $(BUILD)/test_arena $(BUILD)/test_witness $(BUILD)/tinynv-smi $(BUILD)/test_rm_free $(BUILD)/test_image $(BUILD)/test_exec_mode $(BUILD)/test_qmd_cost $(BUILD)/test_pci $(BUILD)/test_cubin $(BUILD)/test_api $(BUILD)/test_fw $(BUILD)/test_submit $(BUILD)/test_reloc $(BUILD)/test_mm $(BUILD)/test_qmd $(BUILD)/test_qmd_hw $(BUILD)/test_hw $(BUILD)/test_hw_kernel $(BUILD)/test_hw_smem $(BUILD)/test_hw_cbank $(BUILD)/test_hw_dims $(BUILD)/test_hw_chain $(BUILD)/test_hw_replay $(BUILD)/test_hw_gap $(BUILD)/test_hw_printf $(BUILD)/test_hw_abort $(BUILD)/test_hw_fault $(BUILD)/test_dev
	@# FIRST, before any test runs: do the tests in this directory still report what they find? On 2026-09-15
	@# test_rm_free.c exited 0 with FAIL printed - it consulted its failure count at line 117 of a 318-line main, the
	@# file grew past that guard, and the whole C4b decision plus the rows proving the read-only fix landed below it.
	@# `make test` was green over all of it. Session C's lint derives each file's failure counter FROM THE CODE rather
	@# than from a list of names (a name list was its own blind spot, found here with a two-line negative control), and
	@# fails the build on LOSING - evidence being discarded now. --strict ALSO fails on UNGUARDED-ACCUMULATING: a file
	@# whose assertion macro increments a counter consulted before the end, where the natural way to add a test is to
	@# append a CHECK that then silently does not count. That is the shape that bit us, and it is worth failing on
	@# BEFORE it starts losing anything. UNGUARDED-but-FAIL-FAST never fails: every site there returns 1 itself, so a
	@# new check copied from its neighbours returns too - and failing on harmless findings is how a gate gets disabled,
	@# taking the real ones with it. All five of ours are fail-fast, so --strict costs nothing today and catches the
	@# next accumulating file the day it is written. It runs first because a suite whose verdicts are discarded makes
	@# everything after it unreadable.
	$(PYTHON) tools/exit_status_lint.py --strict
	@# AND whether what is about to run was built from what is in the tree. The make here is GNU Make 3.81 (Apple's
	@# 2006 build), which compares mtimes to the SECOND while APFS records nanoseconds - so a source edited inside the
	@# same second its object was compiled is "not newer" and is NOT rebuilt, silently. That defeats every negative
	@# control, which is the practice the rest of this suite's credibility rests on: a check "seen to fail" and then
	@# "seen to pass" proves nothing if both runs were the same binary. Invisible to a build id, too - build_id.c is
	@# recompiled every time, so the id looks current while every other object is stale.
	@$(PYTHON) tools/check_stale_objects.py
	@for cb in ../spike/vecadd.sm86.cubin ../spike/vecadd.sm90.cubin ../spike/vecadd.sm120.cubin; do $(BUILD)/test_cubin $$cb || exit 1; done
	@# `test -f $$cb && run || true` swallowed a real FAILURE, not just a missing file: in `A && B || C` the C runs
	@# whenever B fails, so these two cubins could have been failing since the line was written and the suite would
	@# have said nothing. Found by applying Session C's rule - grep for the reassuring construct, not for the
	@# condition - after their own test reported PASSED while testing nothing. A skip is also now announced rather
	@# than silent, because "it did not run" and "it passed" should never look the same.
	@for cb in ../spike/norm.cubin ../spike/mmvq.cubin; do \
	  if [ -f $$cb ]; then $(BUILD)/test_cubin $$cb - || exit 1; \
	  else echo "  (skipped: $$cb is not present, so nothing checked it)"; fi; done
	@# a kernel name longer than any in the local corpus. nothing here reaches 256 characters, which is why a fixed
	@# buffer survived until session A's 8012-kernel survey found two at 474.
	@$(PYTHON) tools/make_long_name_cubin.py ../spike/vecadd.sm120.cubin $(BUILD)/longname.cubin >/dev/null
	@$(BUILD)/test_cubin $(BUILD)/longname.cubin -
	@$(BUILD)/test_api ../spike/vecadd.sm86.cubin 0x160 && $(BUILD)/test_api ../spike/vecadd.sm120.cubin 0x380
	@$(BUILD)/test_fw
	@# which submission path an unset environment gets, and how the two knobs resolve. costs nothing and is the only
	@# place that decision is visible without a card.
	@# recycling the scratch region a piece at a time, against a byte-by-byte record of who wrote what. the failure
	@# it exists for - bytes handed back while an engine still reads them - produces a wrong answer and no fault, so
	@# there is nothing on the card that would report it.
	@$(BUILD)/test_arena
	@# the crash witness, which only ever runs during a crash on a live card - so a bug in it costs a whole run to
	@# find. checked here against the containment question its file exists to answer, in both directions.
	@$(BUILD)/test_witness
	@# the record that makes an rm object freeable, and the frees it has to refuse. no card: every refusal is about the
	@# shape of the record, and the corruption they prevent is not something the card reports.
	@$(BUILD)/test_rm_free
	@$(BUILD)/test_exec_mode
	@# what a launch costs the processor, with no gpu. sizes any change that removes host work per launch.
	@$(BUILD)/test_qmd_cost 20000
	@# the memory manager on its own: awkward sizes, freed and reallocated, which the recorded boot never does
	@$(BUILD)/test_mm
	@# and a sequence a real caller issued - ggml's k=129 f16 matmul cases, which is where the mapping check went wrong
	@$(BUILD)/test_mm test/allocseq-k129.txt
	@$(PYTHON) tools/nv_trace.py --selftest
	@# what a cubin references and does not contain, both halves: the device printf resolves, anything else is refused.
	@# the cubin is build output and not in the repository, so it is made if it is missing rather than assumed.
	@if [ ! -f ../spike/printf_kernel.sm120.cubin ]; then $(PYTHON) tools/build_spike_cubin.py ../spike/printf_kernel.cu \
	   || { echo "  cannot build the cubin test_reloc needs (a compiler server must be reachable; see env.sh)."; \
	        echo "  set TINYNV_ALLOW_NO_CUBINS=1 to skip this test deliberately."; [ -n "$$TINYNV_ALLOW_NO_CUBINS" ]; }; fi
	@if [ -f ../spike/printf_kernel.sm120.cubin ]; then $(BUILD)/test_reloc ../spike/printf_kernel.sm120.cubin; \
	 else echo "  (no printf_kernel cubin, test_reloc skipped by TINYNV_ALLOW_NO_CUBINS)"; fi
	@# the command buffers the card was handed, rebuilt from the oracle's own method numbers and compared word for word
	@$(PYTHON) tools/nv_reference_submit.py test/reference-submit.txt >/dev/null && $(BUILD)/test_submit test/reference-submit.txt
	@# and the kernel descriptor, against what tinygrad's own QMD class builds for the same cubin. the C side reads the
	@# cubin itself rather than being handed the numbers, so a divergence in the cubin reader is caught here too.
	@$(PYTHON) tools/nv_reference_qmd.py ../spike/vecadd.sm120.cubin test/reference-qmd.txt >/dev/null \
	  && $(BUILD)/test_qmd test/reference-qmd.txt ../spike/vecadd.sm120.cubin
	@# the loadable image: the stage between reading a cubin and launching from it, and the one where a mistake stays
	@# silent. A constant bank placed at the wrong offset loads fine, launches fine, and reads a table that is not there,
	@# so nothing downstream reports it - which is exactly why this has to run rather than merely exist. The corpus is a
	@# real ggml cubin (49 relocations, 4 kernels) and not the vector add, because the relocated __device__ tables every
	@# IQ-quantised kernel depends on are the thing at risk. norm.cubin is build output, so it is a guarded skip.
	@if [ -f ../spike/norm.cubin ]; then \
	   $(PYTHON) tools/nv_reference_image.py ../spike/norm.cubin test/reference-image.txt >/dev/null \
	     && $(BUILD)/test_image test/reference-image.txt ../spike/norm.cubin; \
	 elif [ -n "$$TINYNV_ALLOW_NO_CUBINS" ]; then echo "  (no norm.cubin, test_image skipped by TINYNV_ALLOW_NO_CUBINS)"; \
	 else echo "  no ../spike/norm.cubin: the image loader would not be checked against the oracle at all. Build it, or set TINYNV_ALLOW_NO_CUBINS=1 to skip it deliberately."; exit 1; fi
	@# and the same two things against bytes a 5090 actually took, from the recorded session of 2026-09-14. the reference
	@# is committed because the recording is not something to repeat casually; it is regenerated when the snapshot is there.
	@if [ -f $(SNAPSHOT) ]; then $(PYTHON) tools/nv_reference_hw.py $(SNAPSHOT) test/reference-hw.txt >/dev/null; fi
	@$(BUILD)/test_qmd_hw test/reference-hw.txt
	@# both modes: faithful reproduces the recording operation for operation with nothing forgiven, which is what the
	@# first hardware runs use; fast skips reads whose values are proven held, which is what everything else uses.
	@if [ -f $(TRACE).trace ]; then TINYNV_FAITHFUL=1 $(BUILD)/test_dev $(TRACE) && $(BUILD)/test_dev $(TRACE); \
	 elif [ -n "$$TINYNV_ALLOW_NO_TRACE" ]; then echo "  (no trace at $(TRACE), skipped by TINYNV_ALLOW_NO_TRACE)"; \
	 else echo "  no trace at $(TRACE): the whole boot replay would not run. Record one with tools/nv_trace.py, or set TINYNV_ALLOW_NO_TRACE=1 to skip it deliberately."; exit 1; fi
	@# and that no control this driver sends carries a pointer, at any nesting depth. A pointer in an rm control's
	@# parameters is an address in THIS process; the card reads whatever number is there and reports nothing. Session C
	@# found their own scan for these had two blind spots - nested structs, and raw pointers that are not spelled NvP64 -
	@# so this looks for both, and reads the control list out of nv_structs.h so it cannot drift from what is sent.
	@if [ -d $(NVSRC) ]; then $(PYTHON) tools/check_ctrl_pointers.py $(NVSRC)/src/common/sdk/nvidia/inc src/nv_structs.h; fi
	@if [ -d $(NVSRC) ]; then $(MAKE) -s $(BUILD)/test_headers && $(BUILD)/test_headers; \
	 elif [ -n "$$TINYNV_ALLOW_NO_HEADERS" ]; then echo "  (no vendored headers, skipped by TINYNV_ALLOW_NO_HEADERS)"; \
	 else echo "  no vendored headers: the struct cross-checks would not run. Run tools/fetch_nv_headers.sh, or set TINYNV_ALLOW_NO_HEADERS=1 to skip them deliberately."; exit 1; fi
	@# the way out of the driver, on a server of its own so nothing before it has already cleared bus mastering: the mock
	@# hands over a device with bus mastering on, as the dext does, and the driver has to give it back with that off.
	@sock=$$(mktemp -d)/tinygpu.sock; \
	$(PYTHON) test/mock_server.py $$sock 60 & srv=$$!; \
	for i in $$(seq 100); do [ -S $$sock ] && break; sleep 0.05; done; \
	TINYNV_HW_MOCK=1 $(BUILD)/test_hw $$sock; rc=$$?; \
	rm -f $$sock; wait $$srv 2>/dev/null; [ $$rc = 0 ] || exit $$rc
	@sock=$$(mktemp -d)/tinygpu.sock; \
	$(PYTHON) test/mock_server.py $$sock 60 & srv=$$!; \
	for i in $$(seq 100); do [ -S $$sock ] && break; sleep 0.05; done; \
	$(BUILD)/test_pci $$sock; rc=$$?; \
	rm -f $$sock; wait $$srv 2>/dev/null; [ $$rc = 0 ] || exit $$rc
	@$(PYTHON) test/test_wire_proxy.py $(BUILD)/test_pci

clean:
	rm -rf $(BUILD)

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
