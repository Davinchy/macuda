#!/usr/bin/env python3
""": source-level comparison of the firmware-facing definitions that cannot be compiled
standalone (they live behind NVIDIA's port layer or inside .c files). Extracts each typedef/function
body from BOTH pinned trees, normalizes comments/whitespace, and reports same/changed with the
normalized text as evidence. No GPU contact; pure file inspection.

Usage: python3 abi_text.py <570-tree> <610-tree> > text_diff.json
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path


def strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    return s


def read(p: Path) -> str:
    return strip_comments(p.read_text(errors="replace"))


def extract_typedef(text: str, name: str) -> str | None:
    # typedef struct NAME { ... } NAME;  /  typedef struct { ... } NAME;  / typedef union ...
    pat = re.compile(
        r"typedef\s+(?:struct|union)\s+(?:[A-Za-z_0-9]+\s*)?\{(?:(?!typedef).)*?\}\s*" + re.escape(name) + r"\s*;",
        re.S,
    )
    m = pat.search(text)
    if not m:
        return None
    body = m.group(0)
    return re.sub(r"\s+", " ", body).strip()


def extract_function(text: str, name: str) -> str | None:
    # NVIDIA style: static NV_STATUS\n_name\n(\n args\n)\n{ ... }  — match balanced braces
    idx = text.find(name + "\n(")
    if idx < 0:
        idx = text.find(name + "(")
    if idx < 0:
        return None
    brace = text.find("{", idx)
    if brace < 0:
        return None
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                body = text[idx : i + 1]
                return re.sub(r"\s+", " ", body).strip()
    return None


def extract_enum(text: str, name: str) -> str | None:
    pat = re.compile(r"typedef\s+enum\s+(?:[A-Za-z_0-9]+\s*)?\{(?:(?!typedef).)*?\}\s*" + re.escape(name) + r"\s*;", re.S)
    m = pat.search(text)
    return re.sub(r"\s+", " ", m.group(0)).strip() if m else None


def extract_define(text: str, name: str) -> str | None:
    m = re.search(r"#define\s+" + re.escape(name) + r"\s+([^\n]+(?:\\\n[^\n]+)*)", text)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


# (item id, relative path, kind, symbol)
ITEMS = [
    # RPC envelope (port-layer headers; verified by replay hashes in the 570 baseline)
    ("rpc_message_header_v03_00", "src/nvidia/generated/g_rpc-message-header.h", "typedef", "rpc_message_header_v03_00"),
    ("rpc_message_rpc_union_field_v03_00", "src/nvidia/generated/g_rpc-message-header.h", "typedef", "rpc_message_rpc_union_field_v03_00"),
    ("GSP_MSG_QUEUE_ELEMENT", "src/nvidia/inc/kernel/gpu/gsp/message_queue_priv.h", "typedef", "GSP_MSG_QUEUE_ELEMENT"),
    ("GspSystemInfo", "src/nvidia/inc/kernel/gpu/gsp/gsp_static_config.h", "typedef", "GspSystemInfo"),
    ("PACKED_REGISTRY_ENTRY", "src/nvidia/generated/g_os_nvoc.h", "typedef", "PACKED_REGISTRY_ENTRY"),
    ("PACKED_REGISTRY_TABLE", "src/nvidia/generated/g_os_nvoc.h", "typedef", "PACKED_REGISTRY_TABLE"),
    # RPC request payloads
    ("rpc_gsp_rm_alloc_v03_00", "src/nvidia/generated/g_rpc-structures.h", "typedef", "rpc_gsp_rm_alloc_v03_00"),
    ("rpc_gsp_rm_control_v03_00", "src/nvidia/generated/g_rpc-structures.h", "typedef", "rpc_gsp_rm_control_v03_00"),
    ("rpc_set_page_directory_v1E_05", "src/nvidia/generated/g_rpc-structures.h", "typedef", "rpc_set_page_directory_v1E_05"),
    ("rpc_unloading_guest_driver_v1F_07", "src/nvidia/generated/g_rpc-structures.h", "typedef", "rpc_unloading_guest_driver_v1F_07"),
    # registry RPC payload is PACKED_REGISTRY_TABLE (compared above); rpc_set_registry_v17_00 is print-only.
    ("rpc_run_cpu_sequencer_v17_00", "src/nvidia/generated/g_rpc-structures.h", "typedef", "rpc_run_cpu_sequencer_v17_00"),
    # 610 replacements for the sequencer
    ("rpc_load_exec_generic_bootloader_v1_00", "src/nvidia/generated/g_rpc-structures.h", "typedef", "rpc_load_exec_generic_bootloader_v1_00"),
    ("rpc_load_exec_hs_binary_v1_00", "src/nvidia/generated/g_rpc-structures.h", "typedef", "rpc_load_exec_hs_binary_v1_00"),
    ("GspLoadExecGenericBootloaderParams", "src/nvidia/arch/nvalloc/common/inc/rmgspseq.h", "typedef", "GspLoadExecGenericBootloaderParams"),
    ("GspLoadExecHsBinaryParams", "src/nvidia/arch/nvalloc/common/inc/rmgspseq.h", "typedef", "GspLoadExecHsBinaryParams"),
    # sequencer opcode set (570 only; removed in 610)
    ("GSP_SEQ_BUF_OPCODE", "src/nvidia/arch/nvalloc/common/inc/rmgspseq.h", "enum", "GSP_SEQ_BUF_OPCODE"),
    ("GSP_SEQ_BUF_REG_SAVE_SIZE", "src/nvidia/arch/nvalloc/common/inc/rmgspseq.h", "define", "GSP_SEQ_BUF_REG_SAVE_SIZE"),
    # firmware reservation policy
    ("GSP_FW_HEAP_PARAM_CLIENT_ALLOC_SIZE", "src/nvidia/inc/kernel/gpu/gsp/gsp_fw_heap.h", "define", "GSP_FW_HEAP_PARAM_CLIENT_ALLOC_SIZE"),
    ("_kgspCalculateFwHeapSize", "src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c", "function", "_kgspCalculateFwHeapSize"),
    # non-WPR heap HAL bodies (AD102 = ed6b8b in 570, d505ea in 610; GB20x = 1bb8e3 in 570, b5d989 in 610)
    ("kgspGetNonWprHeapSize_ed6b8b", "src/nvidia/generated/g_kernel_gsp_nvoc.h", "function", "kgspGetNonWprHeapSize_ed6b8b"),
    ("kgspGetNonWprHeapSize_d505ea", "src/nvidia/generated/g_kernel_gsp_nvoc.h", "function", "kgspGetNonWprHeapSize_d505ea"),
    ("kgspGetNonWprHeapSize_1bb8e3", "src/nvidia/generated/g_kernel_gsp_nvoc.h", "function", "kgspGetNonWprHeapSize_1bb8e3"),
    ("kgspGetNonWprHeapSize_b5d989", "src/nvidia/generated/g_kernel_gsp_nvoc.h", "function", "kgspGetNonWprHeapSize_b5d989"),
]


def main() -> None:
    t570, t610 = Path(sys.argv[1]), Path(sys.argv[2])
    out = []
    for item_id, rel, kind, sym in ITEMS:
        rec = {"id": item_id, "path": rel, "kind": kind, "symbol": sym}
        texts = {}
        for tag, root in (("570", t570), ("610", t610)):
            p = root / rel
            if not p.is_file():
                texts[tag] = None
                continue
            t = read(p)
            if kind == "typedef":
                texts[tag] = extract_typedef(t, sym)
            elif kind == "enum":
                texts[tag] = extract_enum(t, sym)
            elif kind == "function":
                texts[tag] = extract_function(t, sym)
            else:
                texts[tag] = extract_define(t, sym)
        a, b = texts["570"], texts["610"]
        if a is None and b is None:
            rec["status"] = "not-found-in-either"
        elif a is None:
            rec["status"] = "added-in-610"
        elif b is None:
            rec["status"] = "removed-in-610"
        elif re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b):
            rec["status"] = "same"   # identical modulo whitespace
        else:
            rec["status"] = "changed"
        rec["text_570"] = a
        rec["text_610"] = b
        out.append(rec)
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()