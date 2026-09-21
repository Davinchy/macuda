# E2: map BAR0 and READ registers only (chip id, WPR2 = is GSP resident, GFW boot-complete). No writes, no reset.
import sys, time; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvmini import MiniNVDev
st = time.perf_counter(); d = MiniNVDev()
print(f"connected+BAR0 mapped in {time.perf_counter()-st:.2f}s; BAR0 {d.pd.bar_info(0)[1]>>20} MB  BAR1 {d.pd.bar_info(1)[1]>>20} MB  BAR3 {d.pd.bar_info(3)[1]>>20} MB")
s = d.state(); print("state:", s)
if s['mmio_all_ones']: print("MMIO reads all-ones: link/BAR problem"); sys.exit(2)
print("verdict:", {'0x1b': "GB2xx alive;", '0x19': "AD1xx alive;", '0x17': "GA1xx alive;"}.get(s['arch'], f"unexpected arch {s['arch']};"),
      "GSP RESIDENT (warm: a boot needs FLR first)" if s['gsp_resident'] else "GSP not resident (cold: first boot after power-on)",
      "| GFW boot complete" if s['gfw_booted'] else "| GFW NOT reporting boot complete")
d.close()
