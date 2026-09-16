# M2 proof (stdlib only): load an EXTERNALLY-nvcc-compiled cubin and launch it through tinygrad's NV runtime (QMD path).
import sys, array
from tinygrad import Device, dtypes
from tinygrad.device import TinyELF, Buffer
dev = Device["NV"]
print("device", dev.device, "arch", dev.arch, "compute_class", hex(dev.iface.compute_class))
cubin = open(sys.argv[1], "rb").read()
sig = ((None,0,dtypes.float32,()), (None,1,dtypes.float32,()), (None,2,dtypes.float32,()), (None,3,dtypes.int32,()))
obj = TinyELF(lib=cubin, name="vecadd", target=dev.renderer.target, signature=sig)
prog = dev.runtime(obj)
print("program loaded: regs=%d smem=%d" % (prog.regs_usage, prog.shmem_usage))
n = 256
a = Buffer("NV", n, dtypes.float32).allocate(); b = Buffer("NV", n, dtypes.float32).allocate(); c = Buffer("NV", n, dtypes.float32).allocate()
av = array.array('f', [float(i) for i in range(n)]); bv = array.array('f', [float(i*10) for i in range(n)])
dev.allocator._copyin(a._buf, memoryview(av)); dev.allocator._copyin(b._buf, memoryview(bv)); dev.synchronize()
import os
g = tuple(int(x) for x in os.environ.get("G","4,1,1").split(","))
l = tuple(int(x) for x in os.environ.get("L","64,1,1").split(","))
print("launch global(grid)=%s local(block)=%s" % (g,l))
prog(a._buf, b._buf, c._buf, global_size=g, local_size=l, vals=(n,), wait=True)
out = bytearray(n*4); dev.allocator._copyout(memoryview(out), c._buf); dev.synchronize()
outf = memoryview(out).cast('f')
exp = [av[i]+bv[i] for i in range(n)]
errs = [abs(outf[i]-exp[i]) for i in range(n)]; ok = max(errs) < 1e-4
ncorrect = sum(1 for i in range(n) if errs[i] < 1e-4)
print("correct:", ncorrect, "/", n, "| spot [0,63,64,127,128,192,255]:", [round(outf[i],0) for i in [0,63,64,127,128,192,255]])
print("c[:6]   =", [round(outf[i],1) for i in range(6)])
print("expected=", [round(exp[i],1) for i in range(6)])
print("MATCH:", ok, "| max abs err", max(errs))
sys.exit(0 if ok else 1)
