# E3: standalone FLR in its own process (the #16454 recipe). Clears Bus Master (read-back), FLR via dext, polls config + fresh BAR0
# until the chip answers and GFW reports boot complete, then verifies/repairs COMMAND + BARs against the pre-FLR snapshot.
import sys, time; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvmini import MiniNVDev, pci
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
d = MiniNVDev(timeout=60)
pre = d.snapshot_cfg(); s0 = d.state(); print("pre :", s0); print("cfg :", {hex(k): f"{v:08x}" for k, v in pre.items()})
cmd = d.command(); d.cfgw(pci.PCI_COMMAND, cmd & ~pci.PCI_COMMAND_MASTER, 2); rb = d.command()
print(f"bus master cleared: 0x{cmd:04x} -> 0x{rb:04x} {'ok' if not rb & pci.PCI_COMMAND_MASTER else 'FAILED'}")
t0 = time.perf_counter(); d.pd.reset(); t_flr = time.perf_counter() - t0
print(f"FLR rpc returned in {t_flr:.3f}s")
# 1) config space back?
t1 = time.perf_counter()
while True:
  v = d.cfg(0x00)
  if v == 0x2b8510de: break
  if time.perf_counter() - t1 > BUDGET: print(f"config VID/DID still 0x{v:08x} after {BUDGET}s"); sys.exit(3)
  time.sleep(0.01)
print(f"config space sane after {time.perf_counter()-t1:.3f}s; COMMAND now 0x{d.command():04x}")
post = d.snapshot_cfg(); diff = {hex(k): (f"{pre[k]:08x}", f"{post[k]:08x}") for k in pre if pre[k] != post[k]}
print("cfg diff pre->post:", diff or "none")
if any(k in diff for k in ('0x10','0x14','0x18','0x1c','0x20','0x24')):
  print("BARs changed by FLR: restoring from snapshot"); [d.cfgw(off, pre[off]) for off in (0x10,0x14,0x18,0x1c,0x20,0x24)]
if not d.command() & pci.PCI_COMMAND_MEMORY: print("MEM decode off after FLR: re-enabling"); d.cfgw(pci.PCI_COMMAND, d.command() | pci.PCI_COMMAND_MEMORY, 2)
# 2) fresh BAR0 mapping, wait for chip + GFW
d.remap_bar0(); t2 = time.perf_counter()
while True:
  s = d.state()
  if not s['mmio_all_ones'] and s['gfw_booted']: break
  if time.perf_counter() - t2 > BUDGET: print(f"GFW not booted after {BUDGET}s: {s}"); sys.exit(4)
  time.sleep(0.05)
print(f"chip answering + GFW boot complete {time.perf_counter()-t2:.3f}s after config sane; post:", s)
print(f"E3 OK: total {time.perf_counter()-t0:.2f}s. Bus master left {'ON' if d.command() & pci.PCI_COMMAND_MASTER else 'OFF'} (tinygrad re-enables it at boot).")
d.close()
