# libtinynv — design for M3 (the C userspace NVIDIA driver)

Session B, 2026-09-13. Scope: milestone M3 of `docs/03-cuda-shim-plan.md` — a C library that boots and drives a GSP GPU from
userspace, implementing the `cuda-shim/include/tinynv.h` contract Session A's `libtinycudart` calls. Linux/3090 first, then the
5090, then the 5090 over TinyGPU on the Mac. The Python driver in `tinygrad/` stays the reference oracle and is not extended.

## 1. The two decisions that make this tractable

**(a) Do not transcribe any NVIDIA definition. Include NVIDIA's own headers.**
tinygrad does not hand-write its register and struct definitions: `tinygrad/runtime/autogen/__init__.py` pins
`open-gpu-kernel-modules` at commit `81fe4fb417c8ac3b9bdcc1d56827d116743892a5` (the 570 branch; `nv_580` and `nv_610` are pinned
too) and generates `nv_regs/*`, `nv.py` and `nv_570.py` from that tree. Those upstream files are C headers, dual MIT/GPL-2.0, so
the C driver includes them directly:

| What | Upstream path (in the pinned tree) | Python mirror |
|---|---|---|
| chip registers (`NV_PMC_BOOT_0`, falcon, FSP, therm, FB) | `src/common/inc/swref/published/<arch>/<chip>/dev_*.h` | `autogen/nv_regs/*` |
| MMU page table formats | `kernel-open/nvidia-uvm/hwref/<arch>/<chip>/dev_mmu.h` | `autogen/nv_regs/dev_mmu` |
| RM classes, alloc params, ctrl commands | `src/common/sdk/nvidia/inc/**` | `autogen/nv_570.py` |
| GSP boot + RPC structures (WPR meta, msgq, rpc headers, libos, FRTS/FWSEC) | `src/nvidia/inc/kernel/gpu/gsp/**`, `src/common/uproc/**` | `autogen/nv.py` |

**Sharpened 2026-09-14, after it caught a real one:** assert against **this architecture's** header, not the first one a
search across the tree offers. Writing the page-fault kinds from the older architectures gives `0xb` as
`COMPRESSION_FAILURE`, where GB202's own `dev_fault.h` has a confidential-computing violation, and omits the write-only
violation at `0x7` entirely; Blackwell also adds physical-address variants of every access kind, which is not a naming
detail — a fault on a physical address is not in this driver's page tables at all, so the mapping code would be the
wrong place to start looking. The static assertion in `test_headers` failed on the first compile and is the only reason
that did not ship as a confidently wrong fault report. Where a constant is architecture-specific, name the chip's header
in the include rather than the family's.

The C driver and the Python oracle are then provably the same definitions, because both derive from one pinned tree. The port
becomes a port of ~2,000 lines of *logic*, not of thousands of constants — which is where the 6–10 week estimate lives or dies.
A generator still earns its place, but only as a **verifier**: `tools/verify_defs.py` reads tinygrad's autogen and emits
`_Static_assert(sizeof/offsetof …)` for every struct and register the driver touches. If upstream's headers and tinygrad's
transcription ever disagree, the C build fails instead of the GPU.

**(b) Develop the boot sequence offline against a recorded trace, not against the GPU.**
GSP bring-up is the long pole and the only part that can wedge hardware. Both drivers get an operation tracer on the PCI
boundary (every config read/write, MMIO read/write, DMA allocation, reset), and the C side gets a `replay` PCI backend that
feeds back the reads a recorded run saw and asserts that the writes match. One captured boot on real hardware then turns into an
offline, deterministic, CI-able regression test for the entire boot path — developed on the Mac with no GPU attached.

That inverts the usual risk: by the time the C driver first touches a real GPU it has already reproduced a known-good boot
write-for-write. The trace format is one line per op (`seq dir kind bar offset size value`), identical on both sides.

## 2. Layering

```
 libtinycudart (Session A)                  cuda-shim/include/tinynv.h   ← the A↔B contract, already written
 ─────────────────────────────────────────────────────────────────────
 tinynv.c        device/module/kernel/memory/stream/event/launch        ~ ops_nv.py (NVDevice, NVAllocator)
 qmd.c  chan.c   QMD v3/v5, GPFIFO rings, USERD, doorbell, semaphores   ~ ops_nv.py (QMD, NVComputeQueue, NVCopyQueue)
 elf.c           cubin loader, .nv.info EIATTR_KPARAM_INFO/PARAM_CBANK  ~ ops_nv.py (NVProgramData)
 rm.c            RM object model over GSP RPC (alloc/control/free)      ~ nv/ip.py  (NV_GSP.rpc_rm_*)
 gsp.c           message queues, RPC framing, CPU sequencer, events     ~ nv/ip.py  (NVRpcQueue, NV_GSP)
 flcn.c          Ampere FRTS+booter / Blackwell FSP chain-of-trust      ~ nv/ip.py  (NV_FLCN, NV_FLCN_COT)
 mmu.c           page tables v2/v3, VA + physical allocators            ~ support/memory.py, nv/nvdev.py
 dev.c           chip id, early init, WPR2 reset, register access       ~ nv/nvdev.py (NVDev)
 ─────────────────────────────────────────────────────────────────────
 pci.h           backend interface: cfg rw, BAR map, DMA alloc, reset
   pci_sysfs.c     Linux: /sys/bus/pci/devices/<bdf>/{config,resourceN} ~ system.py (PCIDevice)
   pci_tinygpu.c   macOS: the TinyGPU unix socket protocol              ~ system.py (APLRemotePCIDevice)
   pci_replay.c    offline: replay a recorded trace, assert the writes  (new)
```

Everything above `pci.h` is portable C11 with no OS dependency beyond libc. The backend interface is deliberately the same
shape as tinygrad's `PCIDevice`, so the two drivers can be diffed op-for-op:

```c
typedef struct {
  uint32_t (*cfg_read)(void *ctx, uint32_t off, uint32_t size);
  void     (*cfg_write)(void *ctx, uint32_t off, uint32_t size, uint32_t val);
  int      (*bar_info)(void *ctx, int bar, uint64_t *base, uint64_t *size);
  int      (*bar_map)(void *ctx, int bar, uint64_t off, size_t len, tinynv_mmio_t *out);
  int      (*dma_alloc)(void *ctx, size_t len, tinynv_dma_t *out);   // host memory the GPU can read: va + per-page addresses
  void     (*dma_free)(void *ctx, tinynv_dma_t *dma);
  int      (*reset)(void *ctx);                                       // FLR
  void     (*quiesce)(void *ctx);                                     // clear bus master; always safe to call twice
} tinynv_pci_ops_t;
```

`tinynv_mmio_t` is a pointer when the BAR is mapped (Linux) and a handle plus read/write callbacks when it is not (TinyGPU),
which is the one place the macOS path costs a round trip. Register access goes through `nv_rd32/nv_wr32` so the socket backend
can batch and the replay backend can record.

## 3. Hardware differences to carry from day one

| | 3090 (GA102, dev target) | 5090 (GB202, the goal) |
|---|---|---|
| falcon boot | `NV_FLCN`: VBIOS FWSEC → FRTS, then booter over SEC2 | `NV_FLCN_COT`: FSP chain-of-trust message, GSP-FMC |
| MMU | v2, 4 levels | v3, 5 levels, different PTE/PDE encodings |
| classes | `AMPERE_*`, QMD v3 | `BLACKWELL_*`, QMD v5 |
| BAR1 | 256 MB (no ReBAR on that board) | 256 MB over Thunderbolt |
| recovery | FLR, host reachable, no reboot risk | FLR works (Session A, E3) but the link can latch a DART flag |

Both are small-BAR, which is the awkward case and the one the Mac needs: page tables live in the BAR window, GSP boot
structures in host DMA memory, and CPU-visible VRAM must be reserved low (the bug this session just fixed in the Python port).

## 4. Sub-milestones, and what each needs

| | Deliverable | Gate | Hardware |
|---|---|---|---|
| M3.0 | Repo layout, build, pinned OGK tree, `verify_defs.py` static asserts, `pci.h` + the TinyGPU backend | the C backend drives the Python mock server (the one in `test/unit/test_remote_pci.py`) | none |
| M3.1 | Tracer in both drivers, `pci_replay.c`, `nvtrace_diff.py` | a recorded Python boot replays into a byte-identical write stream | one capture |
| M3.2 | `dev.c` + `flcn.c` + `gsp.c` to `GSP_INIT_DONE` | replay of a real boot passes end to end | none after the capture |
| M3.3 | `mmu.c`, `rm.c`, `chan.c`, `qmd.c`, `elf.c`: submit a QMD, wait a semaphore | replay passes; the M2 vecadd cubin launches under replay | none |
| M3.4 | First real boot, Linux sysfs backend, 3090 | `vecadd` correct on the 3090 from C | **3090, exclusive** |
| M3.5 | CE copies, events, fault reporting, pinned host memory | the full `tinynv.h` surface passes a C test suite on the 3090 | 3090 |
| M3.6 | Blackwell: FSP/COT, MMU v3, QMD v5, over TinyGPU | `vecadd` then a ggml cubin on the 5090 from the Mac | 5090 |

M3.0–M3.3 are the bulk of the code and need no GPU at all. That is the point of the design.

## 4a. What `module_load` must actually do (M3.3), from Session A's review of our real cubins

Parsing metadata is not loading a module. Verified against `norm.fatbin` and `mmvq.fatbin`, which every IQ-quant GGUF needs:

1. **Load the whole image.** Concatenate every PROGBITS section with 128 byte alignment, giving `sh_addr == 0` sections a
   running offset, and upload it as one allocation of `round_up(size, 0x1000) + 0x1000` — the guard page is not optional,
   instruction prefetch faults without it. `.text.<name>` then lives at `image_va + sh_addr`.
2. **Bind every `.nv.constant<N>`, not just bank 0.** Bank 0 carries the parameters; our norm and mmvq cubins also use
   **bank 4**, a 26,792 byte table of pointers into a `.nv.global.init` section (`iq2xxs_grid`, `ksigns64`, `kvalues_iq4nl`
   and friends). Each bank needs its address and size written into the QMD.
3. **Apply relocations before upload.** Only three types occur: `R_CUDA_64` (type 2, write `image_va + target` as 8 bytes),
   `0x38` (low 32 bits at offset+4) and `0x39` (high 32). All eleven bank 4 relocations are type 2. Raise on anything else.
4. **Blackwell needs driver values injected into constant bank 0.** On QMD v5 the bank must hold at least 224 entries with
   the shared and local memory windows at entries 188 to 191 and `0xfffdc0` at 223. Pre-Blackwell those sit at 6 to 11.
   This is not the kernel's own data, and a parameter-only kernel still faults without it.

Skipping any of these produces wrong answers rather than errors, which is the worst failure mode to debug on hardware.

## 4b. What the replay forgives, and what it refuses

Session A's critique of §1(b) was right: a recorded trace pins one run's addresses and poll counts, and demanding a
positional match would fail on differences that are not differences. What was actually built is narrower than the address
-binding scheme sketched here originally, and narrower is better.

**Addresses are not translated, they are handed back.** The replay backend returns the recorded address for every
allocation and every window query, so anything the driver embeds matches the recording by construction. That removed the
need for a binding table, and it had a second effect worth stating: when the C allocator *would* have chosen differently,
nothing hides it — the next page-table entry it touches is the wrong one and the checker says so. That is how the
segregated-fit allocator was found to be necessary (finding #9, predicted 2026-09-14 00:10, landed 09:10).

**Exactly two things may be stepped over for free.** A read of the very place about to be read, which is a poll that span
a different number of times. And a read whose place *and value* the driver already holds, having just taken a read of
exactly it — the oracle consults a page table entry three times where the C consults it twice, because python's accessor
re-reads inside the test that decides how to decode. The value requirement is the whole of the safety: forgiving by place
alone would forgive skipping the read that saw a change, and a driver that stopped one read early would decide on stale
state with nothing to catch it.

That rests on a property of the recorder, so the recorder is in this repository and the property is tested: runs of
repeated reads are coalesced by **place and value**, so a run holds exactly one value and the read that first saw a change
always begins a new line. `tools/nv_trace.py --selftest` demonstrates the alternative being refused, not merely this one
being accepted.

**Everything else is a divergence**, including a recorded *read* the driver never made. A read it skipped is a decision it
made on information it did not gather.

**The books balance.** The cursor moves for exactly three reasons — an operation finished, an operation stepped over, a
declared gap — so `cursor == finished + stepped over + gapped`, always, checked on every move. This exists because a path
that advanced the cursor outside the accounting (allocation matching, which had its own loop) hid a missing-write hole
until the totals happened to be compared by hand. Session A's suggestion; it converts "when a total does not reconcile,
the gap is the finding" from a habit into a check.

**Bookkeeping is not a category.** Every operation at the boundary is checked, including the apparently uninteresting
question "where is this window". The processor-visible memory pool was once missing, and its only outward sign was two
window queries the driver never made. Had those been classed as bookkeeping, the pool would have stayed missing and the
symptom on hardware would have been writes that silently went nowhere. There is no operation at this boundary whose
absence is safe to ignore, because the boundary is the only place the driver is observable.

**A forgiven operation is a debt, not a pass.** Skipping a read the oracle makes is safe by the rule above and harmless
in isolation, but it puts the driver one operation ahead of the recording, and that is fatal the moment something
downstream depends on position. It happened: every allocation in the oracle begins by asking how large the memory window
is, this driver knew the answer and did not ask, the checker forgave it correctly — and when the channel allocations went
on top, the missing question landed the driver one operation early at a remote call that then never matched. Eighty-eight
million divergences from one question not asked.

So the debt is switchable rather than spread through the code, and the test runs both settings:

- **faithful** makes every read the oracle makes, including re-reads of values already held. It reproduces the recording
  operation for operation with nothing forgiven, and it is what the first hardware runs use, so the first time the card
  sees this driver it sees a sequence it has already accepted. It is also the first bisection step if the card ever
  misbehaves, and costs nothing but bus round trips.
- **fast** skips those reads. It must end at the same point in the recording with zero divergences and exactly the
  committed composition of forgiven reads, which is what makes switching between the two provably safe.

**Only faithful mode can catch the driver doing too much**, and that asymmetry is the reason it exists as a tested mode
rather than a debug flag. Fast mode can only ever under-read, so an extra read is invisible in it. On hardware an extra
read is a bus round trip at best and a side-effecting register access at worst — several status registers clear on read.
The first faithful run caught exactly that: an extra page table read at the lowest level, where the oracle answers
without reading at all. Every earlier catch had been an omission, which had quietly become the only shape of bug anyone
was looking for.

**Unported stages are declared, never implied.** A gap names its exact first and last recorded line, the test asserts the
gap set exactly, and the report prints it. At `GSP_INIT_DONE` the count went to zero and has stayed there.

## 4c. What a fault says, and how the addresses are got (2026-09-14, branch `fault-report`)

A fault arrives from GSP-RM as a bare notification on the status queue: an event number saying an engine touched an
address that is not mapped, and **nothing about which address, or what it was doing**. That sentence was printed through
several hardware sessions, and each time the next move was to guess which descriptor pointed somewhere wrong.

**The premise that made it look expensive to fix was wrong.** The driver's own comment — and `docs/HANDOFF.md` item 5 —
said the addresses live in a fault buffer this driver has not registered. **The oracle registers no fault buffer
either.** It asks the **debugger object** (`GT200_DEBUGGER`, class `0x83de`) for what was recorded against the channel,
*after the fact*, in `on_device_hang`. And that object already exists here: the recorded boot allocates one on the
compute channel, so `gsp->user_debugger` is a handle we hold rather than something to add.

That distinction is the whole cost of the feature. Registering a fault buffer is a **boot-time** act, and the boot is the
replay's to decide (§4b) — it would have diverged, and would have looked fine right up until hardware. Asking the
debugger afterwards adds **nothing to the boot**, so the replay still passes with 0 divergences.

Two questions, in the oracle's order:

1. `NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES` on the debugger, with the compute channel as the target. It returns
   the per-SM error registers **and a valid bit for an MMU fault on that channel** — so it also answers *"was this an MMU
   fault at all"*, which the notification does not.
2. Only if that bit is set, `NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO`, which returns up to **four** addresses with the
   kind and the access for each.

When the bit is clear, the SM registers are the answer instead: **a kernel did something illegal rather than touched
something unmapped**. Those are different bugs with different fixes, and before this they arrived here as the same
sentence.

**Rules this path follows, each for a reason:**

- **Asked once per process, with the flag set *before* the asking.** These are RPCs to firmware that has just reported
  something wrong. If one does not come back, a second attempt would not either, and it would be made from a path that is
  already reporting a failure. Worst case it adds two ten-second timeouts to a stall that has already waited thirty.
- **A debugger that does not answer is printed too.** That is a fact about how far gone the firmware is, and it is not
  the same as having nothing to report.
- **What it found is kept in `gsp` state, not only printed.** So the stall message in `exec.c` can name the address
  inline instead of pointing further up the screen, and so a test can assert the report is *right* rather than present.
- **Nothing runs until something has already gone wrong**, so the normal path is untouched.

**How it is exercised:** `test_hw_fault` causes both kinds deliberately — `unmapped` stores past the end of a real
mapping and expects that address back as a missing page table entry, written to (matched to the page, not the byte,
since a widened store can name a neighbour); `trap` runs a kernel with `BPT.TRAP` and expects the opposite shape, no MMU
fault claimed and an SM error instead. **It deliberately faults the card**, which over Thunderbolt can end in a replug,
so it runs last in a session with its own go, one mode per process, quiesced between, and never inside a soak.

## 4d. How a launch reaches the card, and why it is shaped this way (2026-09-15)

The per-launch cost was chased to the ground over one night and the answer was not any of the five things it looked
like. It was **1.5 KB of descriptors per launch, written across the link by the processor as individual 32-bit stores**
at about 19 ns each - roughly 235 MB/s of register-write bandwidth, which at ~2,000 launches a token is most of a
launch. It hid for so long because **a write on this backend is a copy into a 64 MB socket buffer and returns at once**:
the cost is not paid by the writer, it is paid by whoever next waits, which was the ordering read before the doorbell.
That made a read look seventy times more expensive than a write when they cost the same.

**What the path is now.**

1. A launch builds its descriptor and constant buffer into a **cached shadow** in ordinary host memory. Scattered small
   writes belong somewhere cheap.
2. At the flush the dirty span is copied **once, sequentially** into a **mirror** - a host allocation the engine can
   read, laid out at the descriptor region's own offsets, so a span copies to the place it came from.
3. The **copy engine** transfers it into the descriptor region in video memory. The compute batch that will read those
   descriptors acquires on that copy; the copy acquires on the previous compute chain. Both come from the cross-queue
   machinery that already existed - nothing new orders anything.
4. Command buffers do **not** take this path and cannot. **The engine fetches a command buffer in order to find the
   acquire inside it, so the acquire cannot be what protects that buffer**: it has to be present before the engine is
   told about it at all. They are still written by register, and they are small.

That asymmetry is why **the arena is two regions with one writer each**, which was also the fix for the wrap bug that
cost six hardware runs: `flush()` zeroes the chain count and then allocates its own command buffer, so a wrap during
*that* allocation passed a guard already told there was no chain. With a region each, that allocation cannot reach the
descriptors, and the guard stops being a partial check.

**Three properties that are load-bearing and easy to break.**

- **The mirror needs no lifetime tracking, and that is an argument rather than an oversight.** An offset comes round
  only when the arena wraps, and a wrap already waits for everything outstanding. It is the same argument that makes
  the arena itself safe. Phase 2 also binds a guest aperture to the mirror, so **it stays one allocation whose base
  does not move for the life of the device**.
- **The timeline restarts at a descriptor-region wrap.** It is 32 bits, because the copy engine's release is one word,
  and it advances **once per launch** - about nine hours of continuous generation before every wait refuses. A wrap is
  the one moment where the guard has refused a pending chain *and* the idle has waited for everything submitted, so
  nothing is outstanding and no held value matters. The **command-buffer region is excluded**: flush allocates from it
  holding a chain's value in flight, and moving the counter under that is exactly the bug that stalled every delivered
  chain the first time.
- **A chain's own timeline value must be captured before anything else can take a number.** The delivery submits a
  batch of its own, and a submitted batch takes the next value; submitting the chain with what the counter held
  afterwards declared the compute slot to reach a value its descriptors never release.

**What it bought, measured:** launch floor 7.26 → 1.28 µs, MoE decode 52.9 → 117.8 tok/s, 27B 43.4 → 61.8, served 27B
with its MTP head ~124, and a 62-minute soak at 111.9 average with every greedy probe byte-identical.

**What it bought after that, measured by Session A on the card:** the descriptor region was small because it used to
have to fit the processor-visible window, and an hour of serving spent **14.5% of itself draining at wraps**. It no
longer has to fit anything, so it grew to 64 MB: **18 wraps became 1** on the gap bench, drain 2.99 → 0.70 ms, and a
27B decode went 61.6 → 63.5 with its teardown line falling from 353 wraps / 1262 ms to 19 / 74 ms. With both defaults
flipped (below) a plain 27B decode measures **65.5 tok/s**, the highest yet.

**The defaults as of 2026-09-15 (`bfb1456`):** descriptors are **delivered by the copy engine unless
`TINYNV_ARENA_DMA=0`**, and the chain depth default is **128**. Both were opt-in for a day and through the soak before
the default moved. The startup line **names what decided each one** - "the default" or "was asked for" - because for a
day every run had stated them explicitly, so the line saying the knob's name meant "you asked for this", and that
stopped being true the moment nobody had to. A run with an empty environment is exactly the run where the distinction
is the thing being established. Both resolutions are driven offline by `test_exec_mode`, 7 depth cases and 7 delivery
cases, so the defaults are checkable without a card.

> **WITHDRAWN at 11:14 (`c0db26b`), FIXED at 11:22 (`c48a35e`), RESTORED at 11:47 (`bf56a75`).** The third default
> below was on for about an hour, corrupted memory, was taken off the same minute the evidence arrived, and is on
> again having passed two gates set before either ran. Read the section as the record of that, not as configuration —
> the configuration is: **on**.
>
> **The numbers that survived, measured on the commit that ships them** (`dac9aa7`, interleaved off/on within the same
> minute, two rounds, warning count zero on all eight runs): **+2.8% on the dense 27B** (63.36 → 65.12; rounds +1.8%
> and +3.8%) and **about +2.5% on the mixture**. Gated with an empty environment: **27B 65.25 ± 0.31, MoE
> 134.26 ± 0.23**, op-verify 450/450 at three chain depths, text byte-identical.
>
> **Not the +3.2% claimed below**, which came from a build that had the bug in it, and not the +1.6% a cross-run
> comparison suggested either — between them, which is where a properly interleaved measurement usually lands.
> Session A also declined a flattering number on the way: one mixture "off" run read 118.50 against 127–129 for every
> other off run of the day, and folding it in would have produced +7%. It is recorded as an outlier and excluded,
> which is the harder call when the number it inflates is your own result.
>
> **What it did.** `tinynv_exec_flush` captures `chain_upto = ex->reserved` *first* — deliberately, with a comment
> saying why — then delivers descriptors, then hands the chain over **last**. Held uploads were pushed out inside that
> delivery's `batch_begin` as a **compute** batch, which took a *higher* timeline value and reached the compute queue
> *before* the chain. The chain's descriptors then released `base+1..chain_upto`, all lower. **The timeline goes
> backwards by exactly the length of the chain** — Session A's logs read "12952 after 13070", a drop of 118 against a
> chain of 128, which is what closed the case.
>
> **What that cost.** ~25% at eight slots when it survived, and an MMU fault on the compute channel when it did not —
> one serve completed 370 requests *with the corruption present*, the next died in `argsort`. The warning naming it
> (`"the compute timeline went backwards … a slot is being written by more than one thing"`) printed at load in
> **every** inline-on dense 27B run since the first inline build, and in no MoE run and no inline-off run.
>
> **The lessons, and they are worth more than the 3%.** The invariant was **already written down in a comment four
> lines above where it was broken**, and read while the breaking code was being added. A single-stream measurement
> said +3.2% twice and correctness said 450/450 at three depths — neither touches a second stream, and the default was
> flipped on that evidence. And a **load-time driver warning is a bug report**: it was printing in every gate log for
> an hour before anyone grepped for it. Any compute submission inside a chain flush is now a hard failure with the
> reason attached (`c48a35e`).

**A third default since 2026-09-15 10:30 (`6a510cf`), WITHDRAWN at 11:14: small host-to-device copies ride in the
pushbuffer**
(`TINYNV_INLINE_UPLOAD=0` to go back). Same discipline — off for a day, correctness before speed, flipped only after
op-verify 450/450 at three chain depths with the counts confirming it engaged on every token. Worth **+3.2% on the
dense 27B and +3.1% on the mixture**, which took the empty-environment numbers to **65.3 and 131–132**.

The win is **not** the routing. Routing the bytes through the pushbuffer changed nothing measurable on its own: 2,506
copies demonstrably took the new path and throughput did not move. It is that **the upload stopped taking a batch** —
the bytes are held and emitted into the next compute batch, so six submits a token became zero, and the batch-begin
each one used to do was also flushing the pending chain early. Session A established the direction by removing the
*wait* after each copy and watching nothing happen, which said the cost was getting a batch onto the ring rather than
waiting for it.

**Watch the overflow count under batching.** The held list is 16 entries / 8 KB; past that an upload forces a batch of
its own *and* the chain flush that comes with it, which is the win handed back. A single-slot decode never reaches it.
The teardown line reports the count and says what a large one means — if serving at concurrency 8 shows it firing
often, the answer is a bigger list, not a revert.

### An instrument caveat, because this one has already misled me twice

**The teardown line's drain figure measures the HOST's idleness, not the card's, and on a GPU-bound run it is not
recoverable time.** Session A's image runs are the clean demonstration: Z-Image's drain accounting fell from 2775 ms
to 46 ms **and the wall time did not move**. On a run whose kernels dominate, the host waiting at a wrap is the host
waiting for work it needed anyway. So the teardown line is a good instrument for a decode, where the host is on the
critical path, and a **misleading one for a diffusion run**, where it will happily report a large saving that is not
there. I have twice reasoned as though drain time were time to be won back.

## 4f. Where the time actually goes, measured (2026-09-15 evening) — READ THIS BEFORE OPTIMISING THE LAUNCH PATH

**The host is not the bottleneck any more, and every lever in §4d and §4e aims at the quarter of a launch that it is.**
That is the single most useful thing learned today and it contradicts how this file has been written until now.

> **Correction, 2026-09-15 08:00.** I drew the wrong conclusion from that sentence twice, in writing, and it is the
> more important half of this section. "The host is idle three quarters of the time, therefore submission speed is not
> the lever" — the premise is true and the conclusion does not follow. *The host keeping up says nothing about what a
> launch costs the card once it arrives.* Those are two different quantities: how fast we can build and hand over
> work, and how long the engine takes to get going on it. The profile measures the first. The card's own utilisation
> counters measure the second, and they say the second is where the remaining time is. See the table immediately
> below, which was measured after this section was first written and which changes what it means.

### What the card's own counters say, and it is a different answer for each model (2026-09-15 08:00)

The RUSD block (§ sensors) carries `gpuPercentBusy` and `memoryPercentBusy` — the two numbers `nvidia-smi` reports as
`utilization.gpu` and `utilization.memory`. NVIDIA documents the first as *the percent of time during which one or more
kernels was executing* and the second as *the percent of time during which device memory was being read or written*.
Sampled at 0.5 s through a `tg256` decode on each model, steady state only:

| | dense 27B | mixture 35B-A3B |
|---|---|---|
| decode | 63.2 tok/s = 15.8 ms/token | 130.6 tok/s = 7.66 ms/token |
| **memory busy** | **55–59%** | **15–16%** |
| a kernel is executing | 85–92% | 61–64% |
| board power | 427–436 W | 214–218 W |
| sm clock | 2872 MHz | 2880 MHz |

**The dense number was predicted before it was measured.** The frame says 16.45 GB per token at ~1790 GB/s is 9.2 ms,
which is 57.9% of a 15.8 ms token. The card reports 55–59%. *The cost model this project sizes everything against is
correct for a dense decode*, and `memoryPercentBusy` measures what its name says — a counter over an unrelated window
would have no reason to land within two points of an independently computed ratio.

**The mixture was also predicted, as a falsifiable band.** 15–40% confirms, above 50% refutes. It came back 15–16%.

So the two models are limited by different things and the same optimisation does not serve both:

- **Dense is memory-bound**, at 57% of its token spent moving weights it cannot avoid moving. Its ceiling is ~109 tok/s
  and it runs at 63. A kernel is executing 88% of the time, so only ~12% of a token — **1.9 ms** — is a dry engine.
- **The mixture is not memory-bound at all.** It moves a sixth of the memory and draws half the power at the same
  clock. A kernel is executing only 61–64% of the time, so **~38% of every token, 2.9 ms, the engine has nothing to
  do.**

> **A methodological note, from auditing the day's retractions rather than its findings.** When the dry time was
> thought to be spread over synchronisations, the argument offered for it was: 2.9 ms ÷ 19–20 syncs = ~150 µs, *which
> falls in the 125–180 µs per-synchronisation band derived from the vendor residual*. That reads like two routes
> agreeing. It is not. The band was itself the residual divided by the same synchronisation count, and the residual
> and the dry time are close numbers — so the "agreement" was **arithmetic restating itself**, and it would have held
> just as well had the attribution been completely wrong. Which it was. **Dividing a total by a count and noting the
> result matches a per-unit figure derived from the same total is not independent confirmation.** The genuinely
> independent check is the one immediately below, where two different *measurements* — a vendor comparison and the
> card's own counters — land on the same pair.

**Why this is believable, and it is not the arithmetic above.** The residual over NVIDIA's own numbers was computed
days earlier from token times alone: ~2.5 ms/token dense, ~3.6 ms mixture. The measured dry-engine time is 1.9 ms and
2.9 ms. Two routes with nothing in common — vendor comparison and on-card counters — landing on the same pair, in the
same order, with the same ratio.

**What it means for work — and this is an attribution, which went wrong twice before it went right.** Dividing 2.9 ms
by the mixture's 1,634 launches gives 1.8 µs per launch; dividing by its 19–20 synchronisations gives ~150 µs each.
Both are arithmetic and both were wrong, because the dry time is not spread over launches *or* over synchronisations.

**It is one stall per token.** The refill instrument's own accounting settled it (Session A): of ~4,700 stalls in a
run, ~410 are measured — one per token — and the head spacing says which one. It is the **token boundary**, where the
host samples, rebuilds its graph and submits again.

| | dry per token (utilisation) | the token-boundary stall alone |
|---|---|---|
| dense 27B | 1.90 ms | **1.72–1.77 ms** |
| mixture | 2.91 ms | **1.58–1.72 ms** |

On dense the token boundary is **essentially the whole of it** — there is no dry time left for the other eleven stalls
to account for. The mixture carries a larger total deficit despite issuing **fewer** launches per token than dense
(1,634 against 1,992) because dense hides its gaps behind memory transfers it must perform anyway, and the mixture has
no such transfers to hide behind.

### What the token-boundary stall is made of — settled (2026-09-15 10:00)

Six instruments and eleven card runs got here, five of which measured an interval that was not the one their label
named. **Read the mechanism first; the numbers below are only worth what the labels are.**

Measured on the card's clock, dense 27B, decode-phase refills. Five stamps: the compute timeline's tail after the last
kernel, the copy channel's start and end, the compute batch reached, and the compute batch's first method.

| interval | fast tokens | slow tokens | what it is |
|---|---|---|---|
| tail → copy-start | ~500–600 µs | ~600–1,200 µs | the copy engine picking up the logits copy |
| copy-start → copy-end | **2 µs** | **2 µs** | *not the transfer* — nothing there waits for it |
| copy-end → reached | ~450 µs | ~1,070–1,330 µs | **the bimodal part** |
| reached → head | 39–43 µs | 39–43 µs | the acquire on descriptor delivery — steady |
| **tail → head** | **~1,000 µs** | **~1,700–2,300 µs** | and the token period follows it |

Host side, timed call by call from the shim's trace and steady to ±50 µs across 23 of 24 tokens: ~100 µs staging
memcpy, **~150 µs** llama.cpp sampling and rebuilding its graph, **~370 µs** for seven tiny host→device copies (each a
copy-engine batch with two synchronisations), ~130 µs building the next batch.

**The mechanism, established by a knob that then disqualified itself as a fix.** `TINYNV_TIMESLICE_US` at 100 µs
collapses copy-end → reached to ~450 µs on *every* refill and the bimodality goes with it — but the other handoff,
tail → copy-start, blew to 2.3–4.5 ms on six of eight refills in one run and produced the best boundary of the day at
the identical setting in another. 500 µs is inert. Non-monotone, unstable, opposite-signed on the two handoffs: **the
compute and copy channels contend at the boundary and the runlist's switching decides who waits.** On a bench with one
channel active it never happens (0 of 80 launches past 5 ms). Keep the knob, never as a default — it is the evidence.

**Ruled out, each by a measurement that could have gone the other way.** Engine cold-start (idle sweep flat: 44 µs at
idle 0, 57 at 2 ms, 80 at 10 ms). The descriptor copy (36–43 µs, steady, matching the h2d sweep's fixed cost). Host
launch-building throughput (136 µs inside a 1.7 ms gap). The acquire (steady, not bimodal). The lm_head (correcting
the tail moved the interval only ~50 µs). Draining compute before issuing the download (changed nothing). The seven
tiny copies (the host times them; steady on 23 of 24 tokens).

**Real but rare, and not this:** past ~1 ms of idle a minority of launches pay 385–835 µs where the median pays 45.
Session A's `ioreg` read gives it a mechanism — ASPM L1 on by Apple's default policy on the Thunderbolt bridges,
32–64 µs exit on the Apple ones. Right shape for a tail on a minority, wrong shape for a per-token cost.

**What is left, and who owns it.** ~370 µs of tiny copies — ours, and `TINYNV_INLINE_UPLOAD` carries six of the seven
in the pushbuffer instead. The channel contention — ours, and the fix is to not have two channels at the boundary:
the two are already in one TSG, so the open candidate is a separate context share (`TINYNV_SPLIT_CTXSHARE`, a guess),
and failing that a compute-channel download. ~150 µs of llama.cpp, which is not ours. Everything else is measured and
small.

### What the fixes were worth, and the instrument that flattered them (2026-09-15 10:30)

Two knobs came out of the above, both off by default, both correct on the card (op-verify 450/450 at three chain
depths, byte-identical text):

- **`TINYNV_INLINE_UPLOAD`** — a small host→device copy rides in the command stream instead of taking a copy-engine
  batch. Later (`53dbe89`) it stopped taking a batch at all: the bytes are held and emitted into the next compute
  batch, so six submits a token become zero.
- **`TINYNV_DOWNLOAD_VIA_COMPUTE`** — the logits readback is served by a kernel the caller lends, so the transfer
  leaves the copy engine.

**Together they were worth ~0.15 ms a token on the dense 27B and nothing measurable on the mixture** — measured
before the corruption was found, when the inline path was only *routing* and had not yet stopped taking a batch. What
made it worth +2.8% was the later change (`53dbe89`): the upload stopped submitting at all, so six submits a token
became zero. Routing alone changed nothing, twice measured.

**And the instrument said otherwise, convincingly.** With both on, `tail → head` collapsed to ~200 µs and the
bimodality vanished from it. It was an artefact: the tail stamp fires on a chained compute batch, and with the
download and the uploads both on the compute queue, the last batch to stamp one before a refill is *one of those* —
a few hundred µs into the boundary. **The interval shrank around the thing it was measuring.** Session A caught it by
holding it against the token period.

This was the sixth instrument of the day measuring an interval that was not the one its label named, and the only one
that produced a *plausible* number rather than an absurd one — 3.35 seconds, a 2 µs transfer of a megabyte, 12 ms
inside a 1.8 ms gap, all self-evidently wrong. A number that looks like success is the dangerous kind.

**So the profile reports `head to head`**, the gap between consecutive refill head stamps, which is one token as the
card times it. The head stamp fires only on a refill batch, so it cannot be moved by rerouting work between queues,
which is the property `tail → head` has now lost twice. **Where the two disagree, believe head-to-head, and treat the
disagreement as the finding.**

### The engine handoff at a token boundary: measured, mechanism known, unfixed on this enclosure (2026-09-15 10:45)

**Closed by a stopping condition set in advance, and met by a test that could have gone either way.**

**What it is.** A copy-engine ↔ compute-engine handoff at every token boundary costs **~0.35 ms per token on average
— ~0.7 ms on about half of them, ~0.45 ms on the rest**. The token period follows the split, so it is real time and
not an artefact.

**The mechanism is known.** `TINYNV_TIMESLICE_US` at 100 µs moves *exactly* these two intervals and nothing else: the
copy→compute handoff collapses to ~450 µs on every refill and the bimodality goes with it. It is the runlist's
switching between the two channels. On a bench with one channel active it never occurs — 0 of 80 launches past 5 ms.

**Four levers, all measured, none moved the token.** Each is a knob that is still in the tree, off, because it is the
evidence:

| lever | what happened |
|---|---|
| `TINYNV_TIMESLICE_US` | moves both handoffs, non-monotone and unstable — 100 µs was the best boundary of the day in one run and 2.3–4.5 ms of *the other* handoff in another. A scheduler being poked. |
| `TINYNV_SPLIT_CTXSHARE` | no effect. The two channels were already in one TSG; a separate context share changed neither handoff. |
| `TINYNV_DOWNLOAD_VIA_COMPUTE` | the logits transfer left the copy engine; the handoffs stayed, because the descriptor delivery is still copy-engine work the compute batch acquires on. |
| `TINYNV_SHORT_FIRST_CHAIN` | did exactly what it was designed to do — refill starts ~600 µs sooner, bimodality gone from `tail → head` — **and the token did not get shorter.** The switch moved rather than vanished: two launches are ~30 µs of work, the other 126 still arrive by copy engine, so the handoff now sits between chain one and chain two. "Behind work the engine is already doing" needs a first chain long enough to cover ~350–700 µs, and a chain that long puts the delivery back on the critical path. |

**Why there is no fifth lever.** Removing the handoff means not having the copy engine deliver descriptors, and both
alternatives are worse by arithmetic done before either was built: a full chain's ~192 KB written across the link is
**~800 µs** at 235 MB/s, and inlining a full chain does not fit a pushbuffer (4 KB carries ~2 launches). The short
first chain was the attempt to have both, and the measurement above is why it does not work.

**This note is about this enclosure, not about the hardware.** A Thunderbolt-tunnelled runlist is not a card in a
slot, and anyone reading this on desktop hardware should not inherit the caveat.

**And it was nearly recorded as a success.** `tail → head` under the short first chain reads 351–487 µs with no
bimodality — exactly what a fix working looks like. The token period, measured head-to-head, was unchanged. That is
the second time in two hours the token period caught an interval that had shrunk around the thing it was measuring,
and the first time it caught one of *mine* while I was inclined to believe it.

**A caveat that survives all of it.** `gpuPercentBusy` counts *a kernel executing*; its definition is NVIDIA's public
documentation rather than a header we can check — `cl00de.h` carries the field with no comment and no writer anywhere
in the open tree. `memoryPercentBusy` needs no such caveat: the dense prediction is its own verification.

## 4f-bis. What the offline suite cannot catch, with a worked example (2026-09-15)

The suite is good and it is trusted heavily — 17 sections, no card, seconds to run. **It is worth knowing precisely
what it does not reach, because the inline-upload corruption sat in exactly that gap and passed everything.**

**It reaches:** encoding (command streams held against recorded bytes, method kinds and field positions against
NVIDIA's headers), arithmetic (arena recycling against a byte-by-byte ownership record, containment both ways),
resolution (every mode's default, grepped rather than written down), and structure sizes and field offsets against the
vendored tree.

**It does not reach submission ordering, and cannot.** The null device refuses before `submit_batch` — *"no gpu: this
is the null device"* — and the recorded boot contains no submissions at all: *"the ring, the write pointer and the
doorbell bypass the recorder on a remote device, so nothing here can verify a launch."* So **no offline test can
observe the order in which batches reach a queue**, which is the whole of the bug below.

**The worked example.** Held uploads submitted a compute batch inside a chain flush, so the timeline went backwards by
the length of a chain (`c48a35e`). Against that change the suite passed 17/17; op-verify passed 450/450 at three chain
depths on the card; a 96-token text came back byte-identical; and a single-stream decode measured +3.2% twice. **Every
check we had said yes.** The failure needed a second stream to fault and a load-time warning to be read.

**What defends this class instead**, since a test cannot: the invariant is now a **hard failure at the point of
submission** — any compute batch submitted inside a chain flush fails with the reason attached. That is worse than a
test (it fires on a card, not in the suite) and better than a comment (it cannot be read and not noticed, which is
exactly how this one was broken: the rule was written four lines above the code that broke it).

**The general rule.** *A green suite is evidence about the things the suite reaches.* Where a change touches
submission order, engine interaction or concurrency, the suite's silence is not evidence, and the next instrument
should be a runtime invariant rather than another test.

### And a worse case than not reaching: a suite that stopped reaching, silently (2026-09-15, found the same day)

Everything above is about the suite's *edges*. This is about its middle. `test_rm_free.c` accumulated failures into
`fails` and consulted it at **line 117 of a 318-line `main`**. The guard was correct when written — line 117 *was* the
end of `main` then. The file then grew: the mapping record, writability, and **the whole C4b mapping decision** were
appended below a guard that had already returned. Two hundred lines of checks counted into a variable nothing read
again, and the binary exited `0` with `FAIL` printed on stdout. `make test` reported green.

**What this cost, concretely.** The four rows added earlier that day to prove the `TINYNV_RM_MAP_TYPE_READ_ONLY`
fix — the fix for a defect that inverted this very guarantee — sat below the guard. *"Suite green (17 sections)"* was
true and meant nothing. The fix is right, but it was right because it was read out of NVIDIA's header, **not because
anything tested it**; the test evidence cited for it was hollow. Found only by running a deliberate negative control
against a *new* check and noticing `make test` still exited 0.

**The fix is structural, not a correction.** Moving the guard to the end would work until the file grows again. The
exit status is now **the last statement of `main`** in both files that had this shape (`test_rm_free.c`,
`test_exec_mode.c`, the latter still correct but written the same way to stay that way) — `return fails ? 1 : 0;`
preceded by a counts line. A later section cannot grow past the final return. Both now print `N checks, M failed`, so
the count is visible rather than inferred from silence.

**The shape, for the catalogue.** This is the fifth *silent check* found across these sessions, and the worst: not a
check that is weak, or one measuring the wrong thing, but **a check whose result is computed correctly and then
discarded**. The others degrade evidence; this one manufactures it. Its tell is that nothing looks wrong at the call
site — every `CHECK` reads exactly right — and the defect is in the *distance* between where a result is produced and
where it is consumed. **Whenever a pass/fail decision is not the last thing a test does, it is load-bearing that
nothing gets appended after it, and nothing enforces that.**

**The standing practice this earns:** *a new check is not trusted until it has been seen to fail.* A negative control
costs one edit and one run, and it is the only thing that distinguishes a check that passes from a check that cannot
fail. Both defects on this page — the inverted constant and this guard — were invisible to every green run and visible
immediately to one deliberate break.

**Enforced since, and the reason to catch the shape rather than the symptom.** Session C's `exit_status_lint.py` is on
main and runs *first* in `make test` (`e236d8b`, `d8c292e`); it derives each file's failure counter from the code
rather than from a list of names, which was its own blind spot, found here with a two-line negative control. It grades
three ways:

| | what it means | fails the build |
|---|---|---|
| **LOSING** | assertion sites run after the last consult — printing FAIL and exiting 0 *now* | yes |
| **UNGUARDED-ACCUMULATING** | correct today, but an assertion macro increments a counter consulted before the end | yes (`--strict`, on) |
| **UNGUARDED, FAIL-FAST** | correct today; every site returns 1 itself | no |

**`--strict` is on although nothing here trips it**, and that is the point. `test_rm_free.c` was
UNGUARDED-ACCUMULATING for however long before it became LOSING — during that window nothing was wrong yet and
everything was already arranged to go wrong. The natural way to add a test to such a file is to append a `CHECK`,
which then silently does not count; the natural way to add one to a fail-fast file is to copy the neighbouring
`return 1`, which does. So the accumulating shape is worth failing on *before* it loses anything, and the fail-fast
shape is worth leaving alone — **failing the build on harmless findings is how a gate gets switched off, and it takes
the real findings with it.** The measured split: all five UNGUARDED files here have *zero* assertion-macro sites
between them; the two that broke had 66 and 18.

**A caveat the lint states itself, and the honest reading of it.** Six files are listed as unjudged — it finds no
failure counter and says so by name rather than counting them as coverage. All six are correct, each by a shape no
rule would be worth writing: `return ok ? 0 : 1;`, a direct return at each `FAIL`, macros that return 1, and
`test_headers.c`, which is `_Static_assert`s only — **a failure there is a build error, which is the strongest form of
this property and the one case such a lint should never have an opinion about.**

## 4f-ter. Where a guest's memory should come from, and why not from GSP-RM (2026-09-15 evening)

**Measured, end to end.** A guest is capped near **80 MiB of device memory**. Its `class 0x0040` allocations are
forwarded to GSP-RM and served from the firmware's own heap: Session C measured 40 × 2 MiB succeeding, the 41st
refused with `NV_ERR_NO_MEMORY`, and the firmware's free-heap counter going 85,487,616 → 1,601,536 bytes. My probe
gets the same ceiling from the privileged side (2–48 MiB allocate, 60 MiB and up refused), so **both clients draw on
one ~85 MB pool.** One training step of a 125M model — the smallest anyone would train — needs 2.29 GiB. **28×.**

**This is routing, not the card.** `mmu.c:88` holds back a flat 64 MB for the firmware and gives the driver's own
manager everything below it — **31.8 GB**, where the shim's 15 GB of weights already live. The guest is starved
beside a manager that is not half full.

**The obvious remedy is unavailable.** "Raise `gspFwHeapSize`" moves WPR2's base *lower*, and WPR2 already starts
**160.9 MB below where the manager stops** (measured; see §1 and `test_hw_wpr.c`). The reservation is a flat constant
that derives from nothing, so the heap cannot grow until it does.

> **Closed 2026-09-19 (`30751d1`; the plan is `wpr2-reservation-plan.md`).** The reservation is now derived: `fw_layout.h`
> names every size handed to the firmware, sums what they could occupy (~240 MB) and static-asserts that the 256 MB
> hold-back covers it; the boot reads `NV_PFB_PRI_MMU_WPR2_ADDR_LO/HI` after the firmware has placed itself and refuses
> the open if the manager's top is above WPR2's base minus the non-WPR heap. **Measured, twice (cold and warm open,
> identical):** WPR2 `0x7e7e20000..0x7f4901000`, 202.9 MB, starting 224.9 MB below the top of 32,607 MB; the manager now
> stops at `0x7e5f00000` (32,351 MB), 31.1 MB below WPR2 and 29.0 MB below the unprotected non-WPR heap under it. The
> old top, `0x7f1f00000`, was 161 MB inside the region - the number above, confirmed. The page-table reservation is the
> same 64 MB for any hold-back under 351 MB on this card, so `pa.base` stayed `0x4200000` and the recorded boot replays
> with zero divergences. `test_hw_fill` then allocated to refusal: 31 pieces of 1 GiB, 31,744 MB, the driver's own pools
> and tables holding the other 541 MB of the manager's 32,285; every piece's first and last page written through the
> engine and read back intact, GSP-RM's free-heap figure unchanged across the fill, sensors answering after. Raising
> `gspFwHeapSize` is now *possible* (the static assert says by how much the reservation must follow) and still not
> wanted: 135 MB is the vendor's formula for a 32 GB card, and this section's remedy for guest capacity stands.

**So: serve guest allocations from the driver's manager.** Beyond capacity, this is the version where **we know the
physical pages without asking.** `tinynv_c4b_ranges` needs `0x410103` today only because GSP-RM holds the memdesc; if
the driver allocated the memory it already has the ranges, the card-dependent transport disappears, and guarantee 5's
record consults our own allocator instead of reasoning about an object we did not create.

**The mechanism exists and is NVIDIA's own virtualization path.** `NV01_MEMORY_LIST_FBMEM` (class **0x82**,
`cl84a0.h`) creates a memory object from a **caller-supplied list of framebuffer pages**. Three details say it is
meant for exactly this:

- `hClient` / `hParent` — *"client to which object belongs (may differ from client creating the mapping)"*. The driver
  creates it; the **guest owns it**. Guarantee 1 stays true without a special case.
- `guestId` — *"ID of the guest VM, e.g. domain ID in case of Xen"*. This is the hypervisor path, not a repurposing.
- `RS_FLAGS_ALLOC_PRIVILEGED` (`resource_list.h:596`) — a guest **cannot** allocate it and we can, which is the right
  way round: the driver decides what memory a guest gets.

**The open question, named rather than assumed:** `pageNumberList` is an **`NvP64` — a pointer**, consumed by
`rmapiParamsCopyIn` (`mem_list.c:293`) out of the caller's address space. In a normal driver the CPU-side RM does that
copy; **here libtinynv *is* the CPU-side RM**, and whether the resolved page list crosses our RPC to GSP-RM in a form
it accepts is **not established**. That is the next thing to read, and it is the whole feasibility of this design.

> **A second pointer in allocation parameters, and the same blind spot.** `nv_structs.h` already records that
> `tinynv_nv0040_alloc_t.address` is an `NvP64` and that `check_ctrl_pointers.py` covers **controls** and has never
> looked at allocation params. This is the second instance, in the class we would be adding. *The tool's scope is a
> claim about the code it checks, and "no pointer reaches the card from a guest" is still verified only for controls.*

## 4f-quater. The page list crosses the RPC by value, and it is not the call we were looking at (2026-09-15 night)

**Answered: YES.** §4f-ter left one question standing between the guest-memory design and a plan — whether a resolved
page list can cross our RPC to GSP-RM. It can, NVIDIA's driver does it on **exactly this class**, and the pointer was
never the obstacle. But it does not travel in the call I was reading, and had we sent that call the failure would have
been a fault rather than a status.

**Why the original question had no answer.** `NV01_MEMORY_LIST_FBMEM` carries no RPC flag: its entry is
`RS_FLAGS_ALLOC_PRIVILEGED | RS_FLAGS_ALLOC_CPU_PLUGIN_FOR_SRIOV | RS_FLAGS_ACQUIRE_GPUS_LOCK`
(`resource_list.h:596`), and the allocation path forwards to physical RM **only** on
`RS_FLAGS_ALLOC_RPC_TO_VGPU_HOST | RS_FLAGS_ALLOC_RPC_TO_PHYS_RM` (`alloc_free.c:871`). So in NVIDIA's own driver an
0x82 allocation **never becomes a `GSP_RM_ALLOC`**; `memlistConstruct` runs in CPU-RM and reads `pageNumberList` out of
CPU-RM's own address space. There is no precedent for that message because nobody has ever sent it.

**And sending it would not have returned a status.** `rm_alloc_as` (`gsp.c:328`) copies the params verbatim into
`GSP_RM_ALLOC`, so the pointer would have arrived at the firmware as a number. On that side the params are kernel
params, `bUserModeArgs` is false, and `rmapiParamsCopyIn` degenerates to
`portMemCopy(pKernelParams, size, NvP64_VALUE(pUserParams), size)` (`param_copy.c:224`) — **a raw dereference of a host
pointer inside the firmware's address space.** Not a rejection; a wild read in GSP. *"Try it on the card and read the
status" was not available here, and the run that taught us that would have cost a replug.*

**What NVIDIA does instead, and it is one function.** `memdescSendMemDescToGSP` (`mem_desc.c:4703`) is the same
problem as ours in their own words — CPU-RM holds memory it allocated and wants GSP-RM to know about it — and it splits
in two at exactly the boundary we care about:

1. **Locally**: build the page-number array, allocate the `NV01_MEMORY_LIST_FBMEM` object through
   `rmapiGetInterface(RMAPI_GPU_LOCK_INTERNAL)` — no RPC — so the list is dereferenced by the same address space that
   wrote it, and is `portMemFree`d as soon as the call returns. **Nothing retains it and the firmware never sees it.**
2. **Then** `memRegisterWithGsp` (`mem.c:510`), which is the half that crosses: it picks
   `NV01_MEMORY_LIST_FBMEM` for `ADDR_FBMEM` and issues `NV_RM_RPC_ALLOC_MEMORY` (`rpc.h:113`) →
   `rpcAllocMemory_v13_01` (`rpc.c:3199`) → **`NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY`, function 4**, where
   `_issuePteDescRpc` (`rpc.c:2011`) writes the page numbers **inline into the message body**.

So the page list crosses **by value, in a different function, with a wire form that is entirely in the open source**:

    rpc_alloc_memory_v13_01 (g_rpc-structures.h:79)
      NvHandle hClient, hDevice, hMemory;  NvU32 hClass /* 0x82 */, flags /* os02 */;
      NvU32 pteAdjust, format /* pte kind */;  NvU64 length /* bytes */;  NvU32 pageCount;
      struct pte_desc pteDesc;               // sdk-structures.h:182
        NvU32 idr:2 /* IDR_NONE = 0 */, reserved1:14, length:16;   // then 8-aligned:
        NvU64 pte_pde[length];               // PAGE NUMBERS, physical address >> 12

and for a GSP client `rpcAllocMemory_v13_01` deliberately sends **only the PTEs the memdesc really has** — *"when the
memdesc is contiguous this only sends 1 PTE"*. A contiguous allocation out of our own manager is therefore **one page
number on the wire**, whatever its size.

**The transport is already built here.** `rpc_send` (`gsp.c:175`) splits any payload across
`CONTINUATION_RECORD`s, which is NVIDIA's large-RPC mechanism and the only thing a long page list needs. Function 4 is
not implemented in `nv_structs.h` yet; nothing else is missing.

**Three things can still refuse it, and each is a status rather than a fault.** Named now so a refusal is read
correctly the first time:

- **`pte_desc.length` is 16 bits** — at most 65535 entries per descriptor, and `_issuePteDescRpc` computes its record
  size from `pteCount` **without checking that bound**, so NVIDIA's own helper would truncate the field and send a
  correct-length record describing a short list. At 4 KiB pages that caps one message at 255 MiB. Contiguous
  allocations do not meet it; anything else must be chunked, and *we* must check what their helper does not.
- **The constructor checks two fields the wire form does not carry.** The FBMEM branch refuses unless
  `attr` says `LOCATION_VIDMEM` and `type < NVOS32_NUM_MEM_TYPES` (`mem_list.c:436`), and `rpc_alloc_memory_v13_01`
  carries neither. Either the firmware fills them for this class or the call comes back
  `NV_ERR_INVALID_ARGUMENT` — which would then mean *the wire form,* not *the design*.
- **Every page is bounded against GSP-RM's heap, and we do not know what that is in this boot.** The branch computes
  `trueLength = heapGetBase + heapGetSize` (`mem_list.c:476-478`) and refuses a page above it. Our manager hands out
  addresses up to ~31.8 GB; if GSP-RM believes its heap is the ~85 MB it has been serving guests from, every page we
  name is out of range. **This is now the load-bearing unknown**, and it is much cheaper to settle than the one it
  replaced.

**What the 85 MB number is, and is not.** `TINYNV_CTRL_GET_GSP_RM_FREE_HEAP` (`0x20800aeb`) is documented as *the free
heap size of GSP-RM* and CPU-RM subtracts it from FB-used as firmware-held memory (`mem_mgr.c:1226`). It is the
firmware's own heap — **not** necessarily `pHeap`, the vidmem heap the FBMEM branch bounds against. That guest
allocations tracked it 1:1 (40 × 2 MiB, 85,487,616 → 1,601,536) says the *payloads* came from the firmware heap; it
says nothing about what `heapGetBase`/`heapGetSize` would answer.

**The next two steps, in order, and both are small.**
1. **Ask the card what GSP-RM thinks its heap is.** `NV2080_CTRL_CMD_FB_GET_INFO_V2` (`0x20801303`) with
   `HEAP_START` (0x1E), `HEAP_SIZE` (0x09), `HEAP_FREE` (0x16) and `USABLE_RAM_SIZE` (0x20). Read-only, one step, and
   the V2 form takes an inline array rather than a pointer, so it adds nothing for `check_ctrl_pointers.py` to worry
   about. If the heap covers our manager's range, the bound is not in the way.
2. **One `ALLOC_MEMORY` for a contiguous range out of our own manager, then free it.** Accept or refuse, with a
   status, and no page table is programmed and nothing is written into WPR either way. That is the whole feasibility
   question, answered by a step no more dangerous than the `0x410103` probe at `1904bf1`.

**What is established and what is not.** Established: NVIDIA's CPU-side driver sends a page list by value to this
firmware, on this class, through function 4, and current GSP-client paths depend on it (`mem.c:485` under
`NVOS32_ATTR2_REGISTER_MEMDESC_TO_PHYS_RM`, the profiler's PMA buffers, display, deferred API). Not established: that
the firmware accepts **our** first one. The receiving code in `mem_list.c` is the same source compiled into GSP-RM, and
**I have not observed it run** — that is an inference from one tree, and the three refusals above are where it would
show.

**And the scope note stays.** This removes the *need* for a pointer in allocation params; it does not verify the
claim about them. `check_ctrl_pointers.py` still covers controls only, and `tinynv_nv0040_alloc_t.address` is still
the second `NvP64` nobody has audited. *A design that stops relying on a fact is not the same as a tool that checks it.*

## 4g. What the mapping path must guarantee before it programs a PTE for a guest (2026-09-15)

`UVM_MAP_EXTERNAL_ALLOCATION` is where a guest first hands us an address, and what we are willing to map is a policy
question that must not be answered implicitly by an implementation. Session C asked for these before writing any of
it, and was right to. Eleven invariants; **2, 5 and 8 are the ones that matter most.**

1. **A guest may only name memory under its own client.** The handle resolves through the same object record and
   descendancy logic the free path uses. Another client's handle is *not found*, never *refused* — "refused" tells a
   guest the handle exists.
2. **The VA must lie inside a window reserved for the guest, and the check is containment, not exclusion.**
   > **A zero handle is the same hazard one layer down, and NVIDIA's own code has it** (found 2026-09-15 reading
   > toward 5b). `SET_PAGE_DIRECTORY` resolves its target with
   > `vaspaceGetByHandleOrDeviceDefault(client, hDevice, pParams->hVASpace, &pVAS)` (`dma.c:450`) — a zero `hVASpace`
   > does **not** fail, it falls back to **the device's default VA space**. So a zero there silently retargets the
   > device's page directory rather than the space that was meant. This is exactly why `tinynv_rm_map_check` refuses
   > `!req->root` in the same breath as the driver's own root: *in this interface zero is not "unset", it is "the
   > default one", and the default one is ours.* Any future path that takes an `hVASpace` from anywhere but our own
   > hand needs the same check. Nothing does today. Our command
   arena, descriptor region, semaphores, mirror and page tables sit outside that window *by construction*. Never a list
   of regions to avoid: a list fails open the day someone adds a region, and the failure is a guest with write access
   to the command ring, which is arbitrary work submission. Check the `base + size` addition for wrap **before** the
   range test — an overflowed end address passes a naive containment check.
3. **Size and alignment.** Non-zero size; base and size aligned to the page size actually programmed. **The page size
   is ours, never the guest's** — a guest that could ask for big pages could ask for a granularity that spans outside
   its window.
   > **The alignment half was enforced NOWHERE until 2026-09-15, and a comment said it was.** `gsp.c` listed it under
   > "guarantees not decided here", as *"arithmetic in `tinynv_c4b_map`, which refuses what cannot be expressed as
   > entries at all"* — **a function that does not exist and never did.** Nothing else covered it: `tinynv_mm_map_range`
   > takes `vaddr` and `size` and walks, with no alignment test anywhere on the way, and the only rounding in `pt.c` is
   > inside `tinynv_mm_valloc`, which is the driver's own allocation path and never a guest's. So an unaligned guest
   > request would have been programmed rather than refused, and a length rounded up maps bytes past what the bound
   > check in guarantee 8 approved. It is now decided in `tinynv_rm_map_check` beside the rest of 3, against
   > `TINYNV_MAP_PAGE`, with the offset included because it sets the mapping's physical base — an unaligned offset
   > misaligns every entry even when the VA and length are clean.
   > **This is worse than an unimplemented guarantee, and that is the transferable part: the citation is what stops
   > anyone looking.** A guarantee listed as "handled over there" reads as done to every reader including the person
   > who wrote it. Found only by going to read `tinynv_c4b_map` before building on it — the same move that found the
   > mapping-type constant, which is to say: *go to the thing the comment names, not to the comment.*
   **The general rule this is an instance of** (2026-09-15, reasoning out `gpuCachingType` with Session C): *the guest
   may not choose anything that changes what this driver must do to tear the mapping down.* Page size is one such
   thing. **Caching is another, and it is the load-bearing case**: our unmap path clears the entries, invalidates the
   MMU and waits for the acknowledgement — it does **not** flush caches. A guest able to request a cached mapping
   could leave a cache line alive over memory we have recycled, which is guarantee 5's hazard reached by a different
   road. So `gpuCachingType` is ours, on the same footing as the page size, and the precedent already exists: nvstub
   forces `GPU_CACHEABLE_DEFAULT` to `_NO` on `NV50_MEMORY_VIRTUAL`.
4. **No silent overwrite.** Mapping over a live range refuses, or is an explicit remap with the old one torn down
   first. Silently rewriting PTEs leaves the previous allocation reachable by nothing and possibly still being read.
5. **Every mapping is recorded, and free tears down mappings before it frees.** Our free cascades over children and
   dependants, so an allocation can vanish underneath a live mapping and leave page tables pointing at pages the
   allocator can hand out again — a guest reading another tenant through stale PTEs. The record therefore belongs
   beside the **object** record, not beside the page tables.
6. **Unmap invalidates and waits** (already true in `pt.c`; stated so it is not optimised away). Map stays
   fire-and-forget: a stale map faults loudly, a stale unmap is silent, and adding a read to the map path once produced
   8,021,043 replay divergences.
7. **Permissions no broader than the allocation's.** Both sides are available. The request's is `gpuMappingType`, on
   the wire and guest-supplied. The allocation's is **`NV_MEMORY_ALLOCATION_PARAMS.flags` at offset 8**, in the struct
   we already capture: `NVOS32_ALLOC_FLAGS_USER_READ_ONLY` (0x04000000) and `..._DEVICE_READ_ONLY` (0x08000000).
   > **I claimed for an hour that this was not groundable and was wrong.** I grepped for read-only flags, found
   > `NVOS02_FLAGS_ALLOC_*`, saw they belonged to a different allocation interface, and stopped — without checking
   > whether `NVOS32` had its own. Session C found them four bytes into a struct I was already reading 128 bytes of.
   > The lesson is not "grep harder": it is that **a negative result from a search needs the search stated**, and mine
   > was recorded as "the params do not carry it" rather than "I looked for X and found Y".
   **Two limits on the evidence, both Session C's.** Twelve recorded allocations of one class establish an *instance*,
   not a rule — nothing says another class spells writability the same way. And every one of them reads `0x0001c101`
   with neither bit set, which says what *torch* does and not what a guest *could* do: a guest that sets
   `DEVICE_READ_ONLY` and then requests a read-write mapping is precisely the case 7 exists for, and **no recording
   will ever contain it**. It is covered by a unit test instead.
   **And a caveat on the source itself:** NVIDIA marks both flags *"TODO BUG 2488682: remove this after KMD
   transition"*. Guarantee 7's only input is a field its author intends to delete. If a future pin drops them, 7 loses
   its input again and the honest response is to refuse write mappings rather than to assume writable.
   > **The flags are also AMBIGUOUS TODAY, which is sharper than being deprecated** (Session C, building the constant
   > audit). NVIDIA spells two names on *the same bit*, not adjacent ones — `nvos.h:1498-1501`, and it is one of five
   > such pairs in that word:
   > ```
   > 0x04000000  NVOS32_ALLOC_FLAGS_SPARSE            and  NVOS32_ALLOC_FLAGS_USER_READ_ONLY
   > 0x08000000  NVOS32_ALLOC_FLAGS_DEVICE_READ_ONLY  and  NVOS32_ALLOC_FLAGS_ALLOCATE_KERNEL_PRIVILEGED
   > ```
   > So a bit test cannot recover which name the caller meant, and **"refuses write mappings on read-only
   > allocations" is not what this code can know.** What it can know, bit by bit:
   > - **0x04000000 is recoverable for the objects 7 actually sees, and only those.** `SPARSE` is in
   >   `NVOS32_ALLOC_FLAGS_VIRTUAL_ONLY` (`nvos.h:1517`), and all three physical allocation paths refuse anything in
   >   that mask with `NV_ERR_INVALID_ARGUMENT` — `video_mem.c:1188`, `system_mem.c:562`, `egm_mem.c:316`. Class
   >   `0x0040` is `VideoMemory` (`resource_list.h:460`), a physical class. A class-0x0040 object carrying this bit
   >   therefore **cannot** be sparse: RM would have failed the allocation before we recorded it. On any other class
   >   the bit is ambiguous and the refusal is a strict over-refusal, not a read of intent.
   > - **0x08000000 is not recoverable and does not need to be.** Both readings describe memory a guest must not get a
   >   write mapping on: `DEVICE_READ_ONLY` says so, and `ALLOCATE_KERNEL_PRIVILEGED` is kernel-only memory a
   >   non-kernel client cannot even allocate (`nvos.h:1447-1451`). **Benign by construction, not by luck** — worth
   >   writing down, because "benign by luck" stops being true quietly.
   >
   > The honest statement of 7 is therefore: *refuses write mappings on allocations carrying a bit that means
   > read-only, or — on classes other than 0x0040 — possibly sparse.* The day someone sees a legitimate allocation
   > refused, that sentence is what tells them whether it is a bug, so it is also the text of the refusal reason.
   > Neither bit is reachable from a recording (every recorded allocation is `0x1c101`), so **the first appearance of
   > either will come from a workload, not a test.** `test_headers.c` asserts *the aliasing itself*, so the day NVIDIA
   > fixes BUG 2488682 the build stops and this paragraph gets re-read instead of quietly outliving what it describes.
   *Separately,* 7 is **not** structural in the sense below: The field is
   `UvmGpuMappingAttributes.gpuMappingType`, guest-supplied, one per GPU in `perGpuAttributes[]`. Read-write requested
   on a read-only allocation must be refused against the record.
   **Two more things in that same array, found reviewing the wire form** (`UVM_MAP_EXTERNAL_ALLOCATION_PARAMS`):
   - `gpuAttributesCount` is an `NvU64` the guest supplies, indexing a **fixed** `perGpuAttributes[UVM_MAX_GPUS]`.
     **Clamp it to `UVM_MAX_GPUS` before indexing anything.** A count larger than the array is a read past the end of
     the request struct — on our side, from a number the guest chose, and it needs no invalid address to do it.
   - `gpuUuid` names *which* GPU the attributes apply to. We have one. A UUID that is not ours must be **refused, not
     skipped**: silently ignoring an entry for an unknown device means a guest can submit attributes that are never
     applied and never objected to, and then reason about a mapping that does not have them.
8. **Physical addresses are never taken from the guest** — only handles. A physical address or a page list on the wire
   is ignored; the physical side comes from our own record. Same shape as the privilege downgrade: the guest chooses
   the handle, we choose what it means.
   **Satisfied structurally** (confirmed 2026-09-15 by Session C against `uvm_ioctl.h`): the whole wire form is
   `base, length, offset, perGpuAttributes[], gpuAttributesCount, rmCtrlFd, hClient, hMemory`. There is no physical
   address, no page list, and no page-size field — **nothing on the wire to ignore**, so 8 is met by the message
   format itself rather than by a check. Recorded as such so it does not sit on the list looking unexamined.
   One exception worth naming: `rmCtrlFd` is a guest **file descriptor** — a number indexing a table in a process we
   are not. It cannot be dereferenced here and so is not dangerous the way a physical address would be, but it is a
   value from another machine's namespace and nothing on this side may read it.
   **An OFFSET is not the physical side** (clarified 2026-09-15 after Session C asked, because the original wording
   listed it alongside physical addresses and that was wrong). "Map bytes [offset, offset+length) of my allocation" is
   ordinary, legitimate addressing and must be honoured, not ignored. The distinction that matters:
   **the guest may say WHICH BYTES of its own allocation; it may never say WHERE THOSE BYTES ARE.** So an offset is
   bounds-checked against the record (`offset + length ≤ the allocation's size`, with the addition checked for wrap
   first) and then used as an index *into* the record — it never reaches the hardware as a value the guest supplied.
9. **No aliasing of our memory** — follows from 1 and 8 rather than being checked, which is the point. It should be
   inexpressible, not caught.
10. **A guest cannot exhaust the mapping budget.** Host-memory mappings take one of the dext's 128 never-returned
    slots; count per client and refuse past a cap, or a guest denies service by asking politely 128 times.
11. **The server re-checks everything the guest module already checked** — a check inside the untrusted domain is one a
    different guest declines to perform.
12. **An unmap must match a recorded map exactly — same base, same length — and a partial unmap is refused, not
    attempted.** Added 2026-09-15 after deciding that the page-table walk keeps its freedom to choose a *larger* page
    than the caller asked for (it descends `while (f->covers > *size || !supports_huge_page(…) || misaligned)`, so a
    big page is only used when that much length remains and both addresses are aligned — it cannot widen past
    `length`, and big pages are a straight win in entries and TLB). The cost of that freedom is that a 2 MiB mapping
    may be a *single entry*, so unmapping 4 KiB of it would clear the whole 2 MiB while the record still believed the
    rest was live. The cap belongs on the record, not on the walk.

**One more thing the geometry must refuse, and it is arithmetic rather than policy.** `pte_index` is
`(va / pte_covers) % pte_count` — a **modulo**. A VA past the end of the tree does not overrun, it **wraps**, so an
address near the top of the range silently aliases onto a low index in the same tree; and `tinynv_pt_walk_begin`
computes `vaddr - mm->va_base` with no floor, so a VA below the base wraps the other way. Both look like success. The
mapping path must therefore refuse `va < mm->va_base` and `(va - mm->va_base) + length > pte_covers(0) * pte_cnt(0)`
**in the geometry check, not in the window check** — the window would catch it today, and relying on that is the
list-of-exclusions failure §2 exists to avoid: widen the window one day and the aliasing returns with nothing to say
so.

## 4h. Delivering only what changed: the delta path, four laps of the same mistake (2026-09-18/19)

`TINYNV_DELTA_DELIVERY` (off by default) diffs the descriptor region's fresh shadow against a host mirror of what was
last actually delivered at each address (`last_delivered`) and patches only the difference through the inline-upload
path, instead of handing the whole dirty span to the copy engine every flush. It aims at the token-boundary handoff
in §4f: the copy engine delivering descriptors is the last copy-engine work at a boundary, and the switch between the
two channels is what costs ~700 µs on about half the tokens. A flush whose descriptors ride in the compute batch needs
no copy and so no switch.

**Why it should be worth doing, measured before anything was built (2026-09-18, `TINYNV_DUMP_CMD`).** On a 16-token
MoE decode, of 2,435 dwords submitted per steady-state token across 26 batches, only 65 (2.7%, 260 bytes) differ from
the token before. The changed values are predictable fixed-stride counters - a KV-cache offset field advancing by the
same delta every layer, paired address dwords advancing by the same larger delta - never the launch addresses
themselves: ggml-cuda's MoE routing is on-device buffer *content*, not a changed kernel address. So the shape of a
token's descriptor stream is nearly constant and the delta is small, which is the premise everything below rests on.

**What the four versions had in common is the finding.** Each one measured its diff at a granularity one level too
coarse, read the result as "the change is big", and was wrong the same way as the one before it.

1. **One envelope over the whole flush** (`a4f9010`). A flush accumulates up to 128 launches (~213 KB) before
   delivering. The first and last differing byte of that were 213,483 bytes apart - 100% "changed" - the moment even one
   byte differed near each end, which is near-certain at that scale. Correct, safe, a no-op in practice. Caught on
   hardware with `TINYNV_DELTA_VERBOSE`, not by reasoning.
2. **One envelope per launch** (`c728085`, `51a1011`). Each launch's own span (its QMD slot plus its own cbuf0,
   `chain[i].len`) is where the change lives. First hardware run aborted: the byte-precise bounds had no alignment
   guarantee and `tinynv_cmd_inline_upload` takes whole dwords at a dword address (`submit.c`) - "an inline upload is
   a whole number of dwords, not 1542 bytes". Widened to dwords; safe because a launch's span is 256-aligned with a
   256-rounded length, so widening cannot cross into a neighbour. The single-envelope version had the identical latent
   bug and never ran often enough to hit it.
3. **Its own budget** (`b76c16a`). Even fixed, 5 of 1,223 flushes in a 96-token decode took the path. The patches
   were queued through the same held list `TINYNV_INLINE_UPLOAD` uses (16 entries / 8 KB), which was mostly spent
   before the delta check ran. Now written straight into the chain's own delivery batch after `batch_begin()`,
   reserving their own room in `want_dwords`, under an aggregate cap of their own. Still 5 of 1,223 - and the
   diagnostic showed why: nearly every launch carries *some* change each token, but of a 1,400-1,800-byte envelope only
   40-240 bytes (3-15%, usually 10-13%) actually differ. The per-launch envelope overstated a flush's change 7-10x -
   the same shape as lap 1, one level down - which is why raising the cap (16 KB, then 64 KB) only moved the miss.
4. **Runs of dwords within a launch** (this lap, 2026-09-19, offline only so far). `tinynv_delta_runs` walks a
   launch's span a dword at a time and records each run of differing dwords, merging two runs whose gap is at most
   `TINYNV_DELTA_GAP` bytes (default 32: a second inline-upload call costs 8 header dwords, so carrying up to 8
   unchanged dwords is never dearer than splitting around them, and the tie merges - fewer LAUNCH_DMAs for the
   engine, which the dword count does not price). Dword-granular by construction, so there is nothing to widen. A
   flush's runs live in one flat list (`ex->delta_span`, 4096) with one slot held back per launch still to come, so
   every launch always has room for at least its envelope and the list can never cost a flush its fast path; a launch
   whose runs overflow their room, or cost more dwords than its envelope would (only possible with the gap swept
   below the default), goes out as the envelope - lap 3's behaviour, never worse. The aggregate cap now prices the
   pushbuffer footprint *with headers* (`TINYNV_DELTA_AGGREGATE_KB`, default 128): with runs the headers are a real
   fraction - a thousand small patches is 32 KB of headers alone - and a cap on data alone would let the batch outgrow
   what the check said. A run longer than one call carries is split at emit time, priced the same way.

**Offline evidence for lap 4.** `test_delta` (17,100 checks, most from a covering loop): runs are dword multiples
inside `[lo,hi)`, cover every differing dword and no other, never start or end on an unchanged dword, merge exactly at
the gap and not past it; an overflow still reports the exact envelope; at the default gap no pattern's runs cost more
than its envelope (400 random densities plus the adversarial 36-byte-gap and every-other-dword cases); both knob
parsers, both spellings of each ask. Two planted defects were seen to fail before the test was trusted: the tie
flipped from `<=` to `<` (356 failures, all on the 32-byte-gap property) and envelope tracking stopped at overflow
(3 failures, all on the overflow envelope).

**Properties that are load-bearing, and the one line that could make this wrong.**

- **All-or-nothing per flush.** The check pass commits nothing until every launch has been looked at. Patching some
  launches and then letting a fallback path resend the same bytes as part of a wider copy would deliver a byte twice
  under two different values of "what is there now" - a wrong answer, not a slow one.
- **`last_delivered` is brought up to date once, after whichever path delivered**, over the whole originally-dirty
  span. That single `memcpy` is the one place a bug turns "sometimes not as good as it could be" into "wrong": if the
  recorded mirror ever drifts from what is resident, a future skip omits an update that had to reach video memory.
- **Trusted only past the region's first lap** (`wraps[AR_DESC]`): before that, `last_delivered` has never been
  compared against a real prior delivery at that address.
- **Ownership moves over the whole span whether or not any bytes did** (`tinynv_arena_mark` as the full paths do), so
  every launch in the chain reads consistently as of this delivery.

**What is still an estimate, and what to read on hardware.** The gap (32) and the aggregate (128 KB) come from host
arithmetic - a pushbuffer dword is ~19 ns of register write (§4d) - and price nothing on the engine's side; what a
LAUNCH_DMA costs the front end is unmeasured. Both are knobs so the sweep is a card run each, not a rebuild. The
first run's `TINYNV_DELTA_VERBOSE` lines (`delta runs:` per launch, `delta flush:` per flush, first three flushes)
say whether the envelope-vs-genuine gap actually closed and what a flush's real footprint is; the summary line at exit
says how many flushes took the path. And the mechanism is per flush while the stall it aims at is per token boundary,
so the number that matters is not the flush count but the boundary instrument in §4f, read the same way as every
lever there: interleaved with the reference in the same minutes, host state on the record.

### Lap 4 on the card, and the lap after it (2026-09-19, 08:50-09:20)

**Runs engage everywhere and lose.** With sub-launch runs the fast path took 1223 of 1223 dense flushes and 938 of 938
MoE flushes, text byte-identical on both models, op-verify 450/450 at three depths - and interleaved tg128 pairs read
MoE 145-161 off against 89-90 on (-44%), dense 66-68 against 63-65 (-4%). Each flush's runs carried ~70 KB of its
~200 KB envelope, ~90 KB of pushbuffer with headers: ~380 µs of processor writes across the link at §4d's ~4.2 µs/KB,
on every flush, against the ~350 µs a token of boundary handoff the whole lever was meant to remove. A gap sweep
(0/8/32/128/512 bytes) moved patches a flush from 298 to 1504 at a near-constant ~100 KB and the speed did not move
(83-85 t/s): the cost is bytes written, not calls made. §4f's "why there is no fifth lever" arithmetic held as
written. **A mechanism that engages is not a mechanism that pays; the token period was the reading, as this file
already said it would be.**

**Why 35% of a launch changes when 2.7% was the premise.** The descriptor ring rewinds at a chain boundary, wherever
the region runs out, so what was last delivered at a launch's address is a *different* launch - the same kernel from
another layer or another point in the token, with other weight pointers and strides. The 2.7% measurement (§4h's
first paragraph, `26b0562`) compared consecutive batches within a token. Both numbers were right; the reading that
one implied the other was wrong, and it survived three laps because each lap measured its own granularity and
nothing measured the alignment.

**Lap 5: `TINYNV_DELTA_REWIND` (`74fcfe7`).** At the first launch after a standstill the descriptor region is
declared full, so the next placement comes round through `arena()`'s own wrap path - the only code that knows how to
come round safely (refuses with a chain pending, pushes, flushes, waits per piece, refuses on an unaccounted span).
Launch k of every token then lands where launch k of the last one did, and the diff is a launch against its own
previous-token self. Predicted before the run: under 15 KB a flush. Measured: 6.6 KB, of which ~4.5 KB was headers
on one patch per launch - the release value every descriptor carries, which advances each token. `TINYNV_TAIL_RELEASE=1`
(only the chain's tail releases) removes that patch: **2.1-2.8 KB a flush, ~24 patches, the genuine per-token
parameter changes.** Interleaved tg128, the three knobs together against the defaults: **dense 27B 67.8 -> 70.7
(+4.4%, ±0.2 both sides; 70.81 the highest this project has recorded), MoE 35B-A3B ~161 -> ~166 (+2-3%, of which tail
release alone is ~+2%).** op-verify 450/450 at three depths with all three on; both texts byte-identical.

What the win is made of, by the arithmetic: the copy engine leaves every flush (1561 of 1561 dense flushes rode in
the compute batch), so the per-flush cross-queue acquire and the boundary handoff go with it, and the 2.5 KB of
patches cost ~10 µs of writes a flush. The defaults were not flipped: `llama-server` with slots in flight,
speculative decode and image generation have not run with the three knobs, and tail release costs the ability to
locate a stalled chain from the timeline. That is the next card work, and it is Antonio's call.

### Four reads for the replay question, and where the MoE time actually is (2026-09-19 12:15-12:38)

Done on the new defaults at Antonio's go, written up in `chain-replay-plan.md`'s last section. In one breath: a
decode token's descriptors are byte-identical to the previous token's except the tail release once ggml's allocator
has settled (~4 tokens); a short first chain changes nothing; **chain depth 128 is the optimum** (1024 costs 14-17%,
32 costs the MoE 10%) because the engine cannot start a chain before the host has built all of it, so seams hide
under the next chain's build; and with the sensor publisher finally running during a decode (it only ran on waiting
syncs, which a decode never does), the card reads pstate 1 at 2,860 MHz, 154 W MoE / 306 W dense, no throttling,
**gpu busy 66% (MoE) / 78% (dense)**. So replay is not worth building, and the MoE gap is inside busy time: kernels
plus the per-QMD tail (§4h's next lever: the five invalidates and the L1_SYSMEMBAR membar every descriptor carries).

### The per-QMD tail, measured (2026-09-19 12:45-13:03)

Two knobs, `TINYNV_QMD_MEMBAR` and `TINYNV_QMD_INVALIDATE` (`f96ac28`, the oracle's choice by default). The five
per-launch cache invalidates are **load-bearing**: without them (or with only the constant-bank one) op-verify still
reads 450/450 and the 96-token greedy text **differs** - a chained decode reads what the previous kernel wrote through
the texture path, and only the byte-compare sees it. The system-scope barrier at the end of every non-releasing
descriptor costs the **dense decode 3%** (68.1/68.7 -> 70.7/70.3 tg128, interleaved) and the MoE nothing; gpu scope
buys nothing, so it is the barrier itself. `membar=none` passed op-verify at three depths, both texts and a
five-minute MTP soak; flipping it is Antonio's call. With that, every per-launch choice this driver makes at the
descriptor and the delivery has been measured; the MoE's remaining third against native is inside kernel-executing
time and the inter-kernel dispatch gap, not in anything measured here.

### Two more defaults, and the image models checked (2026-09-19 13:30-13:47)

Antonio: "no reason not to take free percentages." `f7a65d6`: the inline-upload cap defaults to the method's 32,764-byte
ceiling with a 16 x 2 KB held list (the 8 KB upload every token makes rides the pushbuffer: +2.3% MoE / +1.6% dense
when measured alone), and non-releasing descriptors end without a barrier (+3% dense alone). Together, interleaved
against the morning's defaults: MoE 140.3/140.5 -> 142.5/145.0, dense 70.9/70.6 -> 71.7/71.3; op-verify 450/450 at
three depths, both greedy texts byte-identical, SDXL Turbo / SD 1.5 / Z-Image PNGs byte-identical at equal sampling
times (image generation is indifferent to both), a five-minute MTP soak clean. The logits download served by the
compute engine instead of the copy engine measured nothing and stays a knob.

### 4i. The per-kernel profile, and where the MoE's time actually is (2026-09-19 14:11-14:22)

`TINYNV_KERNEL_PROFILE=1` (`f562d0b`..`8843e12`): every descriptor releases a four-word report - its timeline
value, then the engine's clock - into a ring slot of its own in host memory; the chain's tail is stamped by a
command-stream release with wait-for-idle into its slot instead, so nothing rests on a descriptor's second release
slot. Slots are read in order once the timeline has passed them and the interval between stamps goes to the later
kernel: its execution plus the dispatch gap before it, which a completion stamp cannot separate. The token boundary
(any wait that returned with the compute queue retired) is kept apart, the first four windows are not counted, and
the report lists kernels by instantiation and merged, the chain-seam total, the long intervals and the eight
longest. Both greedy texts byte-identical under it; zero stamps missing over 107 windows; cost ~6% MoE / ~3% dense
(the payload changes every token, ~4.6 KB a flush more for the delta path).

What it found (`docs/driver/moe-next-steps.md` section 8): the MoE's kernels run at native speed - 2.45 us a launch
inside a chain, the in-chain time of a token about native's whole token - and the loss is the engine idle at chain
seams (1.85 ms a window) and at the boundary (1.24 ms), because the host builds a launch in 4.1 us (2.38 in
ggml-cuda/llama.cpp, 1.75 in `tinynv_launch`) against the engine's 2.45. The dense is kernel-bound at 1.27 TB/s
effective. The chain-replay study's host figure of ~1.05 us a launch was the driver's share alone.

## 5. What this needs from the humans

- **The 3090's DMA is untranslated, so no kernel parameter change is needed** (Session A's finding #4, settled 2026-09-13):
  AMD-Vi is active on that box with `Default domain type: Translated`, but the card sits on a root complex the firmware's
  tables do not describe, so it is in no IOMMU group at all and physical addresses are bus addresses. Re-check this with
  `preflight-linux.sh` if the board or BIOS ever changes, because the failure mode is DMA faults during GSP boot.
- **M3.4 needs the 3090 exclusively**: tinygrad's PCI path unbinds `nvidia.ko` from `43:00.0`, so vLLM (currently 21.9 GB
  resident, serving `:8000`) and the `llama-server` on that box have to stop for the duration, and the box must be free to be
  rebooted if a bring-up wedges it. The NVRTC compile server on `:6481` is CPU-only and unaffected. Not urgent: it is weeks of
  work away, but the answer shapes M3.1's capture (a 3090 trace is the natural first oracle, and taking a 5090 trace instead
  means borrowing the Mac).
- **One trace capture** on whatever GPU is free first, with the recorder patch — a few minutes of GPU time, no risk beyond a
  normal boot.

## 6. Risks, and what each costs if it bites

| Risk | Why it might | Mitigation / fallback |
|---|---|---|
| The GSP RPC is version-locked to firmware 570.144 | struct layouts move between branches | the same pin as the Python oracle; `nv_580`/`nv_610` trees are pinned too if the firmware has to move |
| Replay diverges from reality (timing, polling loops) | the trace records values, not time | polls are recorded as repeated reads; the replay backend returns the recorded sequence, so a differing poll count is a real difference and should fail |
| Blackwell FSP path has no oracle until the 5090 is on the Mac | the 3090 uses a different boot | M3.6 is scheduled last and the Python driver already boots the 5090, so a capture is available whenever the box is on the Mac |
| `libtinycudart` needs an API the contract missed (graphs, IPC, VMM) | ggml uses more than the spike showed | the contract is A's to extend; the driver layers (rm/chan/qmd) are general enough to serve it |
| The C driver wedges the 5090 over Thunderbolt where Python did not | new code, same hardware path | inherit the whole stability recipe: keep GSP resident, clear bus master on every exit path including failures, standalone FLR to recover, never two clients |
