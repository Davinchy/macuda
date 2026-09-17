#!/usr/bin/env python3
"""Emit the C register definitions libtinynv needs, from the same source the python driver uses.

tinygrad does not hand-write these either: `tinygrad/runtime/autogen/nv_regs` generates them from NVIDIA's
open-gpu-kernel-modules at a pinned commit. Generating the C header from that module keeps both drivers provably on the
same definitions, so a divergence under replay is a logic difference and never a mistyped offset.

    python tools/gen_nv_regs.py cuda-shim/libtinynv/src/nv_regs.h
"""
from __future__ import annotations
import sys, pathlib

# (module, architecture, register) for everything the driver touches. Add to this as the port grows; the generator
# fails loudly if a name is not in the upstream tables rather than emitting something plausible.
WANTED = [
  ("nv_ref", "", "NV_PMC_BOOT_0"),
  ("nv_ref", "", "NV_PMC_BOOT_42"),
  ("dev_fb", "tu102", "NV_PFB_PRI_MMU_WPR2_ADDR_LO"),
  ("dev_fb", "tu102", "NV_PFB_PRI_MMU_WPR2_ADDR_HI"),
  ("dev_therm", "gb202", "NV_THERM_I2CS_SCRATCH"),
  ("dev_gc6_island", "ga102", "NV_PGC6_AON_SECURE_SCRATCH_GROUP_42"),
  # the FSP chain of trust: the mailbox the chain-of-trust message goes through, and its two queues
  ("dev_fsp_pri", "gh100", "NV_PFSP_EMEMC"),
  ("dev_fsp_pri", "gh100", "NV_PFSP_EMEMD"),
  ("dev_fsp_pri", "gh100", "NV_PFSP_QUEUE_HEAD"),
  ("dev_fsp_pri", "gh100", "NV_PFSP_QUEUE_TAIL"),
  ("dev_fsp_pri", "gh100", "NV_PFSP_MSGQ_HEAD"),
  ("dev_fsp_pri", "gh100", "NV_PFSP_MSGQ_TAIL"),
  # the falcon's own configuration, read at an engine base rather than absolutely
  ("dev_falcon_v4", ("gh100", "ga102"), "NV_PFALCON_FALCON_HWCFG2"),
  # the doorbell that tells GSP-RM a message is waiting, and the two windows retargeted once it is up
  ("dev_gsp", "ga102", "NV_PGSP_QUEUE_HEAD"),
  ("dev_bus", "tu102", "NV_PBUS_BAR1_BLOCK"),
  ("dev_vm", "gh100", "NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR"),
  # the doorbell that makes the gpu forget what it cached about the page tables
  ("dev_vm", "tu102", "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE"),
  # the third generation page table entry formats. these have no address: they are the layout of a word in memory.
  ("dev_mmu", "gh100", "NV_MMU_VER3_PTE"),
  ("dev_mmu", "gh100", "NV_MMU_VER3_PDE"),
  ("dev_mmu", "gh100", "NV_MMU_VER3_DUAL_PDE"),
  # the SECOND generation of the same, which is what an Ampere card's page tables are made of
  ("dev_mmu", "tu102", "NV_MMU_VER2_PTE"),
  ("dev_mmu", "tu102", "NV_MMU_VER2_PDE"),
  ("dev_mmu", "tu102", "NV_MMU_VER2_DUAL_PDE"),

  # --- Ampere (GA10x): the VBIOS falcon boot ---------------------------------------------------------------------
  # Where a Blackwell card is released from lockdown by the FSP chain of trust above, an Ampere card runs FWSEC out of
  # its own VBIOS to place the write-protected region, then booter_load over SEC2 to unpack GSP-RM. These are the
  # registers that path touches, taken from tinygrad's NV_FLCN, which is the oracle for it.
  # whether the boot firmware has finished: Ampere signals it here rather than in the THERM scratch Blackwell uses
  ("dev_gc6_island", "ga102", "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05"),
  ("dev_gc6_island", "ga102", "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK"),
  # the GSP falcon, and the mailbox pair the libos arguments are handed through before it is started
  ("dev_gsp", "ga102", "NV_PGSP_FALCON_ENGINE"),
  ("dev_gsp", "ga102", "NV_PGSP_FALCON_MAILBOX0"),
  ("dev_gsp", "ga102", "NV_PGSP_FALCON_MAILBOX1"),
  # driving a falcon: reset it, move code and data in over its own DMA engine, point it at an entry and start it
  ("dev_falcon_v4", "ga102", "NV_PFALCON_FALCON_BOOTVEC"),
  ("dev_falcon_v4", "ga102", "NV_PFALCON_FALCON_CPUCTL"),
  ("dev_falcon_v4", "ga102", "NV_PFALCON_FALCON_CPUCTL_ALIAS"),
  ("dev_falcon_v4", "ga102", "NV_PFALCON_FALCON_DMACTL"),
  ("dev_falcon_v4", "ga102", "NV_PFALCON_FALCON_DMATRFBASE"),
  ("dev_falcon_v4", "ga102", "NV_PFALCON_FALCON_DMATRFBASE1"),
  ("dev_falcon_v4", "ga102", "NV_PFALCON_FALCON_DMATRFCMD"),
  ("dev_falcon_v4", "ga102", "NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B"),
  ("dev_falcon_v4", "ga102", "NV_PFALCON_FALCON_DMATRFFBOFFS"),
  ("dev_falcon_v4", "ga102", "NV_PFALCON_FALCON_DMATRFMOFFS"),
  ("dev_falcon_v4", ("gh100", "ga102"), "NV_PFALCON_FALCON_MAILBOX0"),
  ("dev_falcon_v4", ("gh100", "ga102"), "NV_PFALCON_FALCON_MAILBOX1"),
  ("dev_falcon_v4", ("gh100", "ga102"), "NV_PFALCON_FALCON_OS"),
  ("dev_falcon_v4", "ga102", "NV_PFALCON_FALCON_RM"),
  # the signature check the boot ROM performs on a heavy-secure image before it will run it
  ("dev_falcon_second_pri", "ga102", "NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID"),
  ("dev_falcon_second_pri", "ga102", "NV_PFALCON2_FALCON_BROM_ENGIDMASK"),
  ("dev_falcon_second_pri", "ga102", "NV_PFALCON2_FALCON_BROM_PARAADDR"),
  ("dev_falcon_second_pri", "ga102", "NV_PFALCON2_FALCON_MOD_SEL"),
  ("dev_falcon_second_pri", "ga102", "NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K"),
  # how a falcon's DMA engine addresses memory: which aperture, and physical rather than through a context
  ("dev_fbif_v4", "ga102", "NV_PFALCON_FBIF_CTL"),
  ("dev_fbif_v4", "ga102", "NV_PFALCON_FBIF_TRANSCFG"),
  ("dev_fbif_v4", "ga102", "NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL"),
  # the RISC-V core inside the GSP falcon: which core boots, and whether it came up
  ("dev_riscv_pri", "ga102", "NV_PRISCV_RISCV_BCR_CTRL"),
  ("dev_riscv_pri", "ga102", "NV_PRISCV_RISCV_CPUCTL"),
  # SEC2, the second falcon, which is where booter_load runs
  ("dev_sec_pri", "ga102", "NV_PSEC_FALCON_ENGINE"),
]

HEAD = """// Generated by tools/gen_nv_regs.py from tinygrad's autogen, which is itself generated from NVIDIA's
// open-gpu-kernel-modules at commit {commit}. Do not edit: regenerate instead, so the C driver and the
// python oracle can never drift apart on a register offset.
#ifndef TINYNV_NV_REGS_H
#define TINYNV_NV_REGS_H
#include <stdint.h>

// a register field is [lo, hi] inclusive, the way the upstream headers express them
static inline uint32_t nv_field(uint32_t v, int lo, int hi) {{ return (v >> lo) & (uint32_t)((1ull << (hi - lo + 1)) - 1); }}
#define NV_GET(val, reg, field) nv_field((val), reg##_##field##_LO, reg##_##field##_HI)
#define NV_SET(reg, field, v) (((uint32_t)(v) & (uint32_t)((1ull << (reg##_##field##_HI - reg##_##field##_LO + 1)) - 1)) << reg##_##field##_LO)

// Page table entries are wider than a register: 64 bits at most levels and 128 at the dual one, where a field can start
// past bit 63. So they are held as two words and addressed by absolute bit position, which keeps the generated field
// names usable unchanged.
static inline uint64_t nv_bits(int lo, int hi) {{ return hi - lo >= 63 ? ~0ull : ((1ull << (hi - lo + 1)) - 1); }}
static inline void nv_put128(uint64_t w[2], int lo, int hi, uint64_t v) {{
  for (int b = lo; b <= hi; b++) {{
    uint64_t bit = (v >> (b - lo)) & 1;
    w[b / 64] = (w[b / 64] & ~(1ull << (b % 64))) | (bit << (b % 64));
  }}
}}
static inline uint64_t nv_get128(const uint64_t w[2], int lo, int hi) {{
  uint64_t v = 0;
  for (int b = lo; b <= hi; b++) v |= ((w[b / 64] >> (b % 64)) & 1) << (b - lo);
  return v;
}}
#define NV_PUT(w, reg, field, v) nv_put128((w), reg##_##field##_LO, reg##_##field##_HI, (uint64_t)(v))
#define NV_GET_E(w, reg, field) nv_get128((w), reg##_##field##_LO, reg##_##field##_HI)

"""

def main(out_path: str):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tinygrad"))
  import tinygrad.runtime.autogen.nv_regs as nv_regs
  from tinygrad.runtime.autogen import nv_src

  # An entry's architecture may be a TUPLE of them. Four registers this tree needs are defined for both gh100 and ga102,
  # and they agree on the address while differing only in which bitfields each table bothers to name - so one macro is
  # correct and the field sets are merged. The address is ASSERTED equal rather than assumed: a register that really
  # moved between architectures cannot share a macro, and this fails loudly instead of emitting one of the two.
  def addr_key(base, off):
    if base is None and off is None: return ("fmt",)
    if callable(off): return ("arr",) + tuple(int(base) + int(off(i)) for i in range(4))
    return ("reg", int(base) + int(off))

  lines = []
  for mod, arch, name in WANTED:
    arches = arch if isinstance(arch, tuple) else (arch,)
    entry, fields, key = None, {}, None
    for a in arches:
      table = getattr(getattr(nv_regs, mod), a or "regs")
      if name not in table: raise SystemExit(f"{name} is not in {mod}/{a or 'regs'}: upstream moved it")
      e = table[name]
      # a plain constant is a named VALUE a field takes, not a register: it has no address and no fields of its own
      if not isinstance(e, tuple):
        if len(arches) > 1: raise SystemExit(f"{name} is a constant; naming several architectures for it means nothing")
        lines.append(f"// {mod}/{a or 'published'}")
        lines.append(f"#define {name} {int(e):#x}")
        lines.append("")
        entry = "constant"; break
      k = addr_key(e[0], e[1])
      if entry is None: entry, key = e, k
      elif k != key: raise SystemExit(f"{name} is at {key} in {arches[0]} but {k} in {a}: it moved, so it cannot share a macro")
      for f, r in e[2].items():
        if fields.get(f, r) != r: raise SystemExit(f"{name}.{f} is {fields[f]} in one architecture and {r} in {a}")
        fields[f] = r
    if entry == "constant": continue
    base, off, _ = entry
    lines.append(f"// {mod}/{'+'.join(a or 'published' for a in arches)}")
    if base is None and off is None:
      # a memory format rather than a register: the fields are the whole definition, and some of them are past bit 63
      lines.append(f"// {name}: a page table entry's layout, not an address")
    elif callable(off):
      # an array of identical registers. emit the arithmetic rather than the first element, and only after checking the
      # upstream lambda really is a stride, so a non-uniform layout fails here instead of addressing the wrong register
      addrs = [int(base) + int(off(i)) for i in range(4)]
      stride = addrs[1] - addrs[0]
      if any(addrs[i] - addrs[i - 1] != stride for i in range(2, 4)):
        raise SystemExit(f"{name} is indexed but not evenly spaced: {[hex(a) for a in addrs]}")
      lines.append(f"#define {name}(i) ({addrs[0]:#010x}u + (uint64_t)(i) * {stride:#x}u)")
    else:
      lines.append(f"#define {name} {int(base) + int(off):#010x}")
    for f, (lo, hi) in sorted(fields.items()):
      lines.append(f"#define {name}_{f.upper()}_LO {lo}")
      lines.append(f"#define {name}_{f.upper()}_HI {hi}")
    lines.append("")

  # The class methods the command stream is built from. These are not registers and have no address: a method number is
  # an offset inside a channel's class, which is why they come from the sdk autogen rather than the register tables.
  import tinygrad.runtime.autogen.nv_570 as nv_gpu
  METHODS = ["NVC56F_SEM_ADDR_LO", "NVC56F_NON_STALL_INTERRUPT", "NVC6C0_SET_OBJECT",
             "NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A", "NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A",
             # a launch is these two methods and nothing else: the descriptor's address, then the order to run it
             "NVC6C0_SEND_PCAS_A", "NVC6C0_SEND_SIGNALING_PCAS2_B",
             "NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE",
             # the copy engine: a transfer, and the semaphore it releases, which is not the one the other engines use
             "NVC6B5_OFFSET_IN_UPPER", "NVC6B5_LINE_LENGTH_IN", "NVC6B5_LAUNCH_DMA", "NVC6B5_SET_SEMAPHORE_A",
             # the barrier a compute batch opens with, which makes the shader caches forget what they hold
             "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI",
             # where the shader local memory the die needs actually is, and how much of it each cluster gets
             "NVC6C0_SET_SHADER_LOCAL_MEMORY_A", "NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A",
             # Inline-to-memory: the data rides in the pushbuffer and the compute engine writes it wherever asked. For
             # small host->device copies this replaces a copy-engine batch, a doorbell and a wait with methods in a
             # batch that is going out anyway - Session A priced the seven tiny copies at the start of every decode
             # token at ~370 us, about 50 us each including their two syncs.
             "NVC6C0_LINE_LENGTH_IN", "NVC6C0_LINE_COUNT", "NVC6C0_OFFSET_OUT_UPPER", "NVC6C0_OFFSET_OUT",
             "NVC6C0_LAUNCH_DMA", "NVC6C0_LOAD_INLINE_DATA"]
  FLAG_FIELDS = ["NVC56F_SEM_EXECUTE_PAYLOAD_SIZE", "NVC56F_SEM_EXECUTE_OPERATION", "NVC56F_SEM_EXECUTE_RELEASE_WFI",
                 "NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP",
                 "NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE", "NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT",
                 "NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT", "NVC6B5_LAUNCH_DMA_FLUSH_ENABLE",
                 "NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE",
                 "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION", "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA",
                 "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT",
                 "NVC6C0_LAUNCH_DMA_DST_MEMORY_LAYOUT", "NVC6C0_LAUNCH_DMA_COMPLETION_TYPE",
                 "NVC6C0_LAUNCH_DMA_SYSMEMBAR_DISABLE", "NVC6C0_LAUNCH_DMA_REDUCTION_ENABLE"]
  FLAG_VALUES = ["NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT", "NVC56F_SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ",
                 "NVC56F_SEM_EXECUTE_OPERATION_RELEASE", "NVC56F_SEM_EXECUTE_RELEASE_WFI_EN",
                 "NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_DIS",
                 "NVC6B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED", "NVC6B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH",
                 "NVC6B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH", "NVC6B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE",
                 "NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE",
                 "NVC6B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE",
                 "NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN",
                 "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_TRUE",
                 "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA_TRUE",
                 "NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT_TRUE",
                 "NVC6C0_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH", "NVC6C0_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_DISABLE",
                 "NVC6C0_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_ONLY",
                 "NVC6C0_LAUNCH_DMA_SYSMEMBAR_DISABLE_FALSE", "NVC6C0_LAUNCH_DMA_REDUCTION_ENABLE_FALSE"]
  lines.append("// class methods: an offset inside a channel's class, not an address")
  for m in METHODS: lines.append(f"#define {m} {getattr(nv_gpu, m):#x}")
  for f in FLAG_FIELDS:
    hi, lo = getattr(nv_gpu, f)
    lines.append(f"#define {f}_LO {lo}")
    lines.append(f"#define {f}_HI {hi}")
  for v in FLAG_VALUES: lines.append(f"#define {v} {getattr(nv_gpu, v)}")
  lines.append("")

  # The kernel descriptor, which is a bitfield structure rather than a struct: every field is a [lo, hi] bit range inside
  # a 384 byte block, and several of them are not byte aligned. Generated for the same reason the registers are - so a
  # field's position is never typed in from reading a header.
  # Emitted for BOTH architectures. A Blackwell descriptor is 384 bytes of QMD v5 and an Ampere one 256 bytes of v3,
  # and they disagree about more than size: v5 calls the grid CTA_RASTER_* and shifts the program address, v3 does not.
  # So the two field sets are kept apart by name (TINYNV_QMD_* and TINYNV_QMD_V3_*) rather than reconciled here, and
  # qmd.c chooses between them at run time from the chip. Nothing about the v5 spelling changes: it is the one the
  # committed reference descriptors were byte-compared against.
  QMD_INDEXED_MAX = 8
  def emit_qmd(prefix, version, size_bytes, tag, value_prefix):
    qmd_plain = {n[len(prefix):]: v for n, v in vars(nv_gpu).items() if n.startswith(prefix) and isinstance(v, tuple)}
    qmd_indexed = {n[len(prefix):]: v for n, v in vars(nv_gpu).items() if n.startswith(prefix) and callable(v)}
    if not qmd_plain: raise SystemExit(f"no descriptor fields under {prefix}: upstream renamed the class")
    lines.append(f"// the kernel descriptor's fields, as [lo, hi] bit positions inside the block")
    lines.append(f"#define TINYNV_QMD_{tag}VERSION {version}")
    lines.append(f"#define TINYNV_QMD_{tag}BYTES {size_bytes}")
    for name, (hi, lo) in sorted(qmd_plain.items()):
      lines.append(f"#define TINYNV_QMD_{tag}{name}_LO {lo}")
      lines.append(f"#define TINYNV_QMD_{tag}{name}_HI {hi}")
    for name, fn in sorted(qmd_indexed.items()):
      for i in range(QMD_INDEXED_MAX):
        hi, lo = fn(i)
        lines.append(f"#define TINYNV_QMD_{tag}{name}_{i}_LO {lo}")
        lines.append(f"#define TINYNV_QMD_{tag}{name}_{i}_HI {hi}")
    # the named values those fields take. upstream spells them as the field name with the value appended, which collides
    # with a field of the same prefix - QMD_TYPE_GRID_CTA next to a field QMD_TYPE - so they get a prefix of their own.
    qmd_values = {n[len(prefix):]: v for n, v in vars(nv_gpu).items() if n.startswith(prefix) and isinstance(v, int)}
    for name, v in sorted(qmd_values.items()): lines.append(f"#define {value_prefix}{name} {v}")
    # every field's width, checked here rather than assumed in C: the encoder reads and writes a 64 bit window at the
    # field's first byte, which is only correct while no field needs more than that.
    widest = max((hi - lo + 1) + (lo % 8) for hi, lo in list(qmd_plain.values()) + [fn(i) for fn in qmd_indexed.values()
                                                                                   for i in range(QMD_INDEXED_MAX)])
    if widest > 64: raise SystemExit(f"a descriptor field spans {widest} bits from its first byte: the encoder assumes 64")
    top = max(hi for hi, _ in list(qmd_plain.values()) + [fn(i) for fn in qmd_indexed.values() for i in range(QMD_INDEXED_MAX)])
    if top >= size_bytes * 8: raise SystemExit(f"{prefix} has a field at bit {top}, past the {size_bytes} byte descriptor")
    lines.append(f"#define TINYNV_QMD_{tag}WIDEST_SPAN {widest}")
    # The same fields again as a list a test can walk, so a byte that differs from the oracle can be reported as the field
    # that lives there rather than as an offset. Nothing in the library expands this; it costs nothing unless used.
    lines.append(f"#define TINYNV_QMD_{tag}FIELDS(X) \\")
    everything = sorted([(n, hi, lo) for n, (hi, lo) in qmd_plain.items()] +
                        [(f"{n}_{i}", *fn(i)) for n, fn in qmd_indexed.items() for i in range(QMD_INDEXED_MAX)])
    for n, hi, lo in everything: lines.append(f'  X("{n}", {lo}, {hi}) \\')
    lines.append("")
    lines.append("")
    return len(qmd_plain) + len(qmd_indexed) * QMD_INDEXED_MAX, len(qmd_values)

  nfields, nvalues = emit_qmd("NVCEC0_QMDV05_00_", 5, 0x60 * 4, "", "TINYNV_QMDV_")
  f3, v3 = emit_qmd("NVC6C0_QMDV03_00_", 3, 0x40 * 4, "V3_", "TINYNV_QMDV3_")
  nfields += f3; nvalues += v3

  # Upstream folds one register's fields into another's table here and there - NV_PGC6_AON_SECURE_SCRATCH_GROUP_05
  # carries the priv_level_mask fields as well as its own - so the same macro can be generated twice. C accepts an
  # IDENTICAL redefinition silently and refuses a differing one, which means the harmless case is invisible and the
  # dangerous case only shows up as a compile error in a generated file. Both are decided here instead: a repeat that
  # agrees is dropped, and one that disagrees stops the generator with the two values.
  seen, deduped = {}, []
  for ln in lines:
    if ln.startswith("#define "):
      rest = ln[len("#define "):]
      cut = rest.index(" ") if " " in rest else len(rest)
      nm, val = rest[:cut], rest[cut:].strip()
      if nm in seen:
        if seen[nm] != val: raise SystemExit(f"{nm} would be defined as {seen[nm]} and again as {val}")
        continue
      seen[nm] = val
    deduped.append(ln)
  lines = deduped

  commit = nv_src["nv_570"].rstrip(".tar.gz").split("/")[-1]
  text = HEAD.format(commit=commit) + "\n".join(lines) + "\n#endif\n"
  pathlib.Path(out_path).write_text(text)
  print(f"wrote {out_path}: {len(WANTED)} registers, {len(METHODS)} class methods and "
        f"{nfields} descriptor fields and {nvalues} values "
        f"from open-gpu-kernel-modules {commit[:12]}")

if __name__ == "__main__":
  main(sys.argv[1] if len(sys.argv) > 1 else "cuda-shim/libtinynv/src/nv_regs.h")
