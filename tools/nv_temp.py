# Boot the GPU firmware (powers the thermal domain), then read the on-die temperature sensor over ~15s.
# Reads only; exits clean (GSP resident, bus master cleared by the stable-tree fix). nouveau decode: temp=(reg&0x1fff8)>>8, valid=bit29.
import time
from tinygrad import Device
t0 = time.perf_counter()
dev = Device["NV"]
nvdev = dev.iface.dev_impl
print(f"booted in {time.perf_counter()-t0:.1f}s; reading NV_THERM 0x020460")
def decode(raw): return bool(raw & 0x20000000), (raw & 0x0001fff8) >> 8
# primary: nouveau's offset
ok = False
for i in range(20):
    raw = nvdev.rreg(0x020460); valid, temp = decode(raw)
    live = (raw & 0xffff0000) != 0xbadf0000
    print(f"  t+{i*0.75:4.1f}s  0x020460=0x{raw:08x}  {'LIVE' if live else 'gated(0xBADF)'}  valid={valid}  temp={temp}C")
    if live and 15 <= temp <= 120: ok = True
    time.sleep(0.75)
if not ok:
    print("0x020460 not giving a plausible temp on Blackwell; scanning the NV_THERM window 0x020400-0x0205fc for a stable 20-100C reading...")
    cands = {}
    for off in range(0x020400, 0x020600, 4):
        raw = nvdev.rreg(off)
        if (raw & 0xffff0000) == 0xbadf0000 or raw in (0,0xffffffff): continue
        t = (raw & 0x0001fff8) >> 8
        if 20 <= t <= 100: cands[off] = (raw, t)
    for off,(raw,t) in list(cands.items())[:20]: print(f"  candidate 0x{off:06x}=0x{raw:08x} -> {t}C")
    if not cands: print("  no plausible temperature register found by MMIO scan; would need the GSP thermal RM control")
