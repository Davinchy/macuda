# WPR2 reservation: the plan (desk research, 2026-09-19)

Written by a research agent spawned at Antonio's request to identify the best way forward on the README's second known limit (the flat 64 MB firmware hold-back against a ~203 MB WPR2). Desk work only: no card, no socket, no lock, nothing under the EGPU tree. Its file:line anchors were spot-checked afterwards (`mmu.c:89`, `gsp.c:1426-1430`, `flcn.c:124`, `tinynv.c:690-693`, `nv_regs.h:60`) and hold. Not implemented: the change needs a card run to validate and the decision is Antonio's.

---


All paths are under `/Volumes/512SSD/EGPU_MAC_Nvidia/`; `L` = `cuda-shim/libtinynv`, `N` = `L/third_party/open-gpu-kernel-modules-81fe4fb…/src/nvidia`.

## Recommendation

Do it, this way: replace the flat 64 MB at `L/src/mmu.c:89` with a 256 MB hold-back that is `_Static_assert`ed against the sum of the sizes the driver itself hands the firmware (`L/src/gsp.c:1426-1430`, `L/src/flcn.c:124-125`), and add one post-boot check in `L/src/tinynv.c:device_boot` that reads `NV_PFB_PRI_MMU_WPR2_ADDR_LO/HI` and refuses the open if the manager's top is above `WPR2_LO - nonWprHeapSize`. It is ~60 lines across mmu.c/gsp.h/tinynv.c plus ~40 lines of tests, it leaves the recorded boot byte-identical (shown below), and it costs 192 MB of a 32,607 MB card. Do not raise `gspFwHeapSize` afterwards either: 135 MB is exactly NVIDIA's own formula for this card, and the guest-capacity problem is already routed elsewhere by design.

## 1. The hazard, precisely

**What the allocator thinks it owns.** `tinynv_mm_init` (`L/src/mmu.c:84-99`) sets `managed = vram_size - 64 MB` and gives the "video memory" region `[2 MB + ptable, managed)`; it bumps upward (`mmu.c:8-11`, `tlsf.c:1-12`). `vram_size` is the usable-FB scratch register (`L/src/dev.c:108-110`, `NV_PGC6_AON_SECURE_SCRATCH_GROUP_42`, the same register the vendor reads at `N/src/kernel/gpu/mem_sys/arch/ampere/kern_mem_sys_ga102.c:32-48`). The recorded card returns `0x7f5f` = **32,607 MB** (`traces/5090-boot-smoke.trace:10`, blob offset 0x10), so today the manager stops at 32,543 MB.

**What the firmware actually owns.** The driver hands the firmware sizes only; on the chain-of-trust path the GSP-FMC/ACR places everything and patches the offsets (`N/src/kernel/gpu/gsp/arch/hopper/kernel_gsp_gh100.c:255-291,341-353`). The layout (top down, `kernel_gsp_gh100.c:265-286`, `N/arch/nvalloc/common/inc/gsp/gsp_fw_wpr_meta.h:33-58`):

| band (top down) | size, from the driver's own numbers | protected? |
|---|---|---|
| VGA workspace + PMU reservation + end-of-FB estimate, ending at FRTS end | 28 MB: `frtsVidmemOffset = 0x1c00000` (`flcn.c:124`) = `ALIGN_UP(0x220000 + 0x1000 + 0x1820000, 2 MB)` exactly as `N/src/kernel/gpu/fsp/arch/hopper/kern_fsp_gh100.c:1420-1437` computes it (`mem_mgr_gb100.c:53`, `kern_fsp_gh100.c:1621`, `pmuReservedSize` at `gsp.c:1427`) | no |
| **WPR2** = FRTS 1 MB + boot bin 0.19 MB + GSP-RM ELF 60.6 MB + WPR heap 135 MB + meta/LSB header | ≈197 MB before alignment; the heap absorbs the alignment padding (`kernel_gsp_gh100.c:341-344`), consistent with the ~203 MB the README quotes. Image size measured from `L/third_party/firmware/nvidia/ga102/gsp/gsp-570.144.bin` `.fwimage` = 63,541,248 B; bootloader `data_size` = 200,704 B | yes (WPR) |
| non-WPR heap, **below** WPR2 start | 2.125 MB (`nonWprHeapSize = 0x220000`, `gsp.c:1428`; GB20x HAL value 2,228,224 at `N/generated/g_kernel_gsp_nvoc.h:1321-1323`, selected at `g_kernel_gsp_nvoc.c:820-823`) | **no** |

Firmware-owned band from the top ≈ 28 + ~203 + 2.1 ≈ **233 MB**; the driver holds back 64. So roughly the top **169 MB of the "video memory" region is the firmware's**: the top ~161 MB is inside WPR2 (README) and the ~2 MB just below WPR2's base is the firmware's *unprotected* non-WPR heap. Because the allocator bumps upward, the first thing a full card reaches is that unprotected heap.

**Where the README's numbers came from.** `L/test/test_hw_wpr.c` reads the two registers through `tinynv_device_wpr2_range` (`L/src/tinynv.c:686-698`) and prints base/limit/gap. Its output was not saved anywhere in this checkout (the snapshot commit `2fe7091` carries only the README sentence; `docs/driver/libtinynv-design.md:651-653` restates "160.9 MB below where the manager stops"), so the exact register values are unknown here; the arithmetic above reproduces them to within alignment. The on-card step re-measures and records them.

**What happens if it bites.** Three outcomes, in the order the allocator would meet them: (a) a buffer in the non-WPR heap: plain writes land, GSP-RM's own heap is corrupted, and the failure is a firmware crash or a garbled RPC some time later, with no fault pointing at the buffer; (b) a buffer inside WPR2: the FB MMU drops non-secure writes and blocks reads, so the buffer "works" but reads back zeros, i.e. wrong tokens with no error, or an MMU fault (the vendored fault types include `REGION_VIOLATION`, `L/src/nv_structs.h:782`); (c) a buffer in the PMU/VGA band above WPR2: silent corruption of PMU state. Nothing in the checkout has observed any of these, because no run has allocated within 169 MB of the top.

## 2. What the reservation should be derived from

Two independent sources, and the fix uses both:

**The numbers the driver hands the firmware** (pre-boot, deterministic): `gspFwHeapSize 0x8700000`, `nonWprHeapSize 0x220000`, `pmuReservedSize 0x1820000`, `vgaWorkspaceSize 0x20000`, `frtsSize 0x100000` (`gsp.c:1426-1430`), `frtsVidmemOffset 0x1c00000` (`flcn.c:124`), plus the image and bootloader sizes (`gsp.c:1368`, `gsp.c:1400`). Every one of these is NVIDIA's own value:

- `gspFwHeapSize`: `_kgspCalculateFwHeapSize` (`N/src/kernel/gpu/gsp/kernel_gsp.c:5013-5058`) = OS carveout 22 MB (`N/inc/kernel/gpu/gsp/gsp_fw_heap.h:32`) + base RM 14 MB (`:41`) + `ALIGN_UP(96 KB × 32 GB)` = 3 MB (`:49`) + `48 KB × 2048` = 96 MB (`:69`) = **135 MB = 0x8700000**. Clamped to [88, 280] MB (`gsp_fw_heap.h:101-102`, `kernel_gsp.c:5104-5108`).
- `frtsSize` 1 MB: `kernel_gsp_frts_tu102.c:49-57`. `vgaWorkspaceSize` 128 KB: `kernel_gsp_gh100.c:316`. `frtsVidmemOffset`: `kern_fsp_gh100.c:1420-1452` (the same 2 MB-aligned sum; note the vendor also adds a *retry margin* of a whole WPR size per failed boot attempt, `kernel_gsp.c:5137-5170`, which is one reason a register check must back the estimate).

**The chip's registers** (post-boot, authoritative for WPR2 only): `NV_PFB_PRI_MMU_WPR2_ADDR_LO/HI` (`L/src/nv_regs.h:60-65`), `VAL` = bits 31:4, address = `VAL << 12` (`tinynv.c:690-693`); the vendor's own liveness test is `HI != 0` (`kernel_gsp_tu102.c:1143-1152`). They do not describe the non-WPR heap below or the PMU band above; those come from the handed sizes.

**What the vendor's own client allocator uses**, for the record: it does not read registers or the meta. GSP-RM reports the placed layout back in `GspStaticConfigInfo` (`fwWprLayoutOffset`, `N/inc/kernel/gpu/gsp/gsp_static_config.h:71-75,166`, copied in `kernel_gsp.c:3376-3393`) and an FB region table with `reserved` flags (`gsp_static_config.h:83`); CPU-RM's usable FB is the sum of the non-reserved regions (`N/src/kernel/gpu/mem_mgr/mem_mgr_gsp_client.c:60-97`). That RPC (`GET_GSP_STATIC_INFO`, id 65, `N/inc/kernel/vgpu/rpc_global_enums.h:75`) is not implemented in libtinynv (`grep` of `L/src/gsp.c` finds only the GR static info at `gsp.c:1086`), and its struct is large and version-pinned (`rpcGetGspStaticInfo_v14_00`). It is the "proper" derivation and it is not needed: the registers plus the handed sizes bound the same band from both sides.

## 3. The concrete change

**(a) `L/src/gsp.h` (or a new `fw_layout.h`): name the constants once.** Move the literals out of `gsp.c:1426-1430` and `flcn.c:124-125` into macros (`TINYNV_FW_HEAP_SIZE 0x8700000`, `TINYNV_FW_NONWPR_HEAP 0x220000`, `TINYNV_FW_PMU_RSVD 0x1820000`, `TINYNV_FW_VGA_WS 0x20000`, `TINYNV_FW_FRTS_SIZE 0x100000`, `TINYNV_FW_FRTS_FROM_END 0x1c00000`) and two bounds the images are checked against at load time (`TINYNV_FW_IMAGE_BOUND 64 MB`, checked in `init_gsp_image` after `gsp.c:1368`; `TINYNV_FW_BOOTBIN_BOUND 1 MB`, checked after `gsp.c:1400`). Define

```
TINYNV_FW_CARVEOUT_ESTIMATE = FRTS_FROM_END + FRTS_SIZE + BOOTBIN_BOUND + IMAGE_BOUND + HEAP_SIZE
                            + 1 MB (LSB header + meta) + NONWPR_HEAP + 8 MB (WPR alignment slack)   // ≈ 240 MB
TINYNV_FW_RESERVE_TOP       = 256 MB
_Static_assert(TINYNV_FW_CARVEOUT_ESTIMATE <= TINYNV_FW_RESERVE_TOP, "raise the reservation with the heap");
```

256 MB is also the vendor's own "pre-scrubbed top of FB" constant (`g_kernel_gsp_nvoc.h:1417-1419`; `gsp_fw_wpr_meta.h:30-31`: "no memory outside of this region may be used until the FW RM has scrubbed the remainder").

**(b) `L/src/mmu.c:89`:** `managed = dev->vram_size - TINYNV_FW_RESERVE_TOP;` and fix the comment. Nothing else in `tinynv_mm_init` changes. **Replay invariance:** `ptable_size = round_up(managed/512, MB)` (`mmu.c:92`) is 64 MB for any reservation below 351 MB on a 32,607 MB card (`(32607-64)/512 = 63.56` and `(32607-256)/512 = 63.19` both round to 64), so `pa.base` stays `0x4200000` and every recorded address is unchanged; `pa.size` only moves the refusal point. `test_dev` will prove it (nothing recorded depends on `pa.size`).

**(c) `L/src/mmu.c`: `int tinynv_mm_check_fw_carveout(tinynv_mm_t *mm, uint64_t wpr2_lo, uint64_t wpr2_hi, uint64_t nonwpr, char *line, size_t n)`**, pure arithmetic:
- if `wpr2_lo == 0 && wpr2_hi == 0`: not verifiable (the vendor notes the registers can be hidden, `kernel_gsp_gh100.c:225-231`); return 0 and say "unverified" in `line`;
- sanity: `wpr2_lo < wpr2_hi < vram_size`, span within `[HEAP_SIZE + IMAGE_BOUND/2, RESERVE_TOP]`; anything else is refused as "nonsense registers";
- `fw_low = wpr2_lo - nonwpr`; require `mm->pa.base + mm->pa.size <= fw_low`; else `tinynv_fail("the manager stops at %#llx but the firmware owns from %#llx (wpr2 %#llx..%#llx, non-wpr heap %llu MB below it): raise TINYNV_FW_RESERVE_TOP")`.

**(d) `L/src/tinynv.c:device_boot`, after `tinynv_gsp_init_queues` (line 231) and before `tinynv_exec_init` (233):** read the two registers with `tinynv_rd32` (the same reads `tinynv_device_wpr2_range` makes, `tinynv.c:694-695`, HI + 0x1000 for an exclusive limit), call (c), fail the open on refusal, and print one line. This placement is outside the recorded boot (`test_dev.c` drives `tinynv_gpu_*`/`tinynv_gsp_*` directly, `test_dev.c:104-193`; the recording ends after the queues, `traces/5090-boot-smoke.trace` tail), so the replay never sees the two extra reads; putting them in `gsp.c:tinynv_gsp_init_hw` instead would diverge the replay (`pci_replay.c:342-346` returns `0xffffffff` and counts a divergence for an unrecorded read).

**(e) The startup line** (stderr, alongside the sensors lines at `tinynv.c:260-278`):

```
libtinynv: firmware carveout: held back 256 MB of 32607 MB (sum of the sizes handed to the firmware: 240 MB);
           chip says wpr2 0x7e5e00000..0x7f2100000 (203 MB) with 2 MB of non-wpr heap below it; the manager stops 29 MB under that
```
or `… chip did not expose wpr2 (registers read 0); reservation unverified this run`. A run then states what it reserved, from which numbers, and what the chip said.

**(f) Honest capacity reporting** (small, optional but the residency work wants it): `tinynv_device_props` reports `total_mem = vram_size` (`tinynv.c:363`) and the shim returns `free = total` (`cuda-shim/libtinycudart/cudart_api.c:144`). Report `total_mem = mm.pa.size` and add a free-bytes counter to the TLSF so `cudaMemGetInfo` stops overstating by 322 MB plus whatever is live.

## 4. Risk class

Low-to-moderate, all failure modes loud:

- **A boot that no longer fits:** impossible from this change; the boot allocates from the bottom (`mmu.c:3-6`) and no boot-time allocation goes near the top. The FMC does not consult our reservation.
- **An allocator that refuses:** only if the ACR places WPR2 lower than the 240 MB estimate (a bigger firmware, a boot-retry margin). Then the open fails with both numbers and the fix is bumping `TINYNV_FW_RESERVE_TOP`; a bump past 351 MB moves `pa.base` and the replay will diverge, which is the right signal. For a diagnostic session only, `TINYNV_FW_RESERVE_CHECK=0` downgrades the refusal to the printed line.
- **Warm card, stale registers:** `tinynv_dev_early_init` reads `WPR2_ADDR_HI` first (`dev.c:65-72`); non-zero means a resident firmware, and the chip is FLR'd before anything else. The post-boot check runs only after `GSP_INIT_DONE` (`gsp.c:1512`), so the registers describe the firmware *we* booted. If a reset ever failed to clear WPR2, the chain of trust fails first (`flcn.c:130-134`); and even then the registers describe a real protected region, so the check is still right. Cold open (recorded: HI = 0, `trace:3`, blob offset 0) and warm open converge to the same numbers; the on-card step confirms by printing the line twice.
- **Misreading the field:** `(reg >> 4) << 12` matches the vendor's `VAL << ALIGNMENT(12)`; HI names the last 4 KiB page, so the exclusive limit is `+0x1000`.
- **Lost capacity:** 192 MB, 0.6% of the card.

## 5. Validation

**Offline first (no card).**
1. `L/test/test_mm.c`: set `VRAM` (`test_mm.c:24`) to the recorded card's 32,607 MB; assert `mm.pa.base == 0x4200000` (the replay-invariance guard) and `mm.pa.base + mm.pa.size == VRAM - TINYNV_FW_RESERVE_TOP`. Unit-test `tinynv_mm_check_fw_carveout` with synthetic registers: `(lo = VRAM - 225 MB, hi = VRAM - 22 MB)` passes with a 29 MB gap; `(lo = VRAM - 300 MB)` refuses; `(0, 0)` returns "unverified", not a refusal; a span of 5 MB or 900 MB is refused as nonsense. `make test` is not to be run in this session; the checks are written for the next one.
2. `L/test/test_dev.c`: after the queues (line 190) add `CHECK(g.mm.pa.base + g.mm.pa.size == g.dev.vram_size - TINYNV_FW_RESERVE_TOP)` and `CHECK(g.mm.pa.base == 0x4200000)`; the faithful replay (`Makefile:220`, `TINYNV_FAITHFUL=1`) must still report zero divergences and exactly `TOLERATED_READS` (`test_dev.c:31`). The trace cannot check the register-derived bound: its only WPR2 read is the cold-open zero.
3. `L/test/test_headers.c`: a static assert that `TINYNV_FW_HEAP_SIZE == (22 + 14 + 3 + 96) << 20` and `TINYNV_FW_NONWPR_HEAP == 2228224`, with the vendor lines cited, so the constants cannot drift from the headers they came from.

**Then on the card**, each step with Antonio's go, under the lock:
1. `BIN=… sh tools/nv_shim_step.sh A probe build/shim/nv/test_hw_wpr`: expect `NO OVERLAP … gap ≥ 20 MB`; **record the printed `wpr2 base..limit` and `manager ends` into `docs/driver/libtinynv-design.md` §4f-ter**, which is the measurement the checkout is missing. Run it twice in a row (second open is the warm case: `chip was reset on arrival`) and require the same two addresses.
2. New `L/test/test_hw_fill.c` (`probe` step): allocate 1 GiB buffers until refusal, then 64 MB, 4 MB, 1 MB until refusal; for each, `tinynv_device_probe_phys_attr` (`tinynv.c:702`) gives the paddr; assert the highest `paddr + size` equals `managed_end` and is `<= wpr2_lo - 0x220000`. The refusal reading to expect is `region_alloc`'s: `physical allocation: N bytes do not fit the M MB video memory region` (`mmu.c:31-33`). Then write a 4 KiB pattern to the topmost buffer with the copy engine, read it back (D2H), require it intact, and read the sensors (`tinynv_sensors`) to show GSP-RM still answers. Free everything; confirm `tinynv_device_gsp_free_heap` (`gsp.c:580`) is unchanged from before the fill.
3. The standard gate, `tools/wtgate.sh`: op-verify 450/450 at depths 32/64/128, the two tg128 numbers at the defaults, the 96-token greedy text byte-identical to `logs/shim-simple-20260914-225831.txt`, and the new startup line present in every step log.

## 6. Is raising `gspFwHeapSize` wanted afterwards, and what residency needs

No. The only pressure to raise it was the ~80 MiB guest cap (`libtinynv-design.md:641-645`: two clients on one ~85 MB free pool), and §4f-ter already decided the remedy is serving guest memory from the driver's own manager via `NV01_MEMORY_LIST_FBMEM`, not a bigger firmware heap. 135 MB is the vendor's computed value for a 32 GB card; the override ceiling is 280 MB (`gsp_fw_heap.h:102`), and going there would need a ~400 MB reservation, which crosses the 351 MB replay-invariance line. Leave it; the static assert makes any future raise fail the build until the reservation follows.

What the residency work (fitting ~31.8 GiB of experts) actually needs from this: an honest budget, not a bigger one. Usable VRAM is 31.84 GiB, the reserved top becomes 256 MB, the boot/page-table/CPU-visible pools take ~70 MB, and the exec arena and descriptor region take theirs, so the true ceiling is about **31.4 GiB**; 31.8 GiB of experts does not fit on this card under any reservation, and today's `cudaMemGetInfo` (free = total = 32,607 MB) hides that. Item 3(f) is what turns "OOM at the top" into a planning number llama.cpp can use, and item 5.2's fill probe is the proof that the ceiling is exactly where the driver says it is.
