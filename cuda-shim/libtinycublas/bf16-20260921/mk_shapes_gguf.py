# A SHAPES-ONLY bf16 GGUF: Qwen2.5-3B-Instruct's exact dimensions and tokenizer, copied from the official Q8_0 GGUF's
# metadata, with block_count 2, every weight zero, and the output head tied to token_embd as the bf16 conversion of the HF
# snapshot (aa8e7253, tie_word_embeddings true) has it. It exists only so the shim's trace can log the GEMM calls ggml makes,
# on the null device, without re-creating the 6.2 GB bf16 model: a GEMM's shape depends on the dimensions, not the weights.
#   python3 mk_shapes_gguf.py <llama.cpp dir> <q8_0 gguf> <out gguf>
import sys
sys.path.insert(0, sys.argv[1] + "/gguf-py")
import numpy as np, gguf
q8, out = sys.argv[2], sys.argv[3]
r = gguf.GGUFReader(q8)
w = gguf.GGUFWriter(out, "qwen2")
for name, f in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture": continue
    val = f.contents()
    if name == "qwen2.block_count": val = 2
    if name == "general.file_type": val = int(gguf.LlamaFileType.MOSTLY_BF16)
    if name == "general.name": val = "SYNTHETIC qwen2.5-3b dims, 2 layers, zero bf16 weights (GEMM shapes only)"
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(name, val, vt, sub_type=st)
n = 0
for t in r.tensors:
    if t.name == "output.weight": continue
    if t.name.startswith("blk.") and int(t.name.split(".")[1]) >= 2: continue
    shape = tuple(reversed([int(x) for x in t.shape]))
    if len(shape) >= 2: w.add_tensor(t.name, np.zeros(shape, dtype=np.uint16), raw_shape=shape, raw_dtype=gguf.GGMLQuantizationType.BF16)
    else: w.add_tensor(t.name, np.zeros(shape, dtype=np.float32))
    n += 1
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out, n, "tensors")
