# Minimal NV register access through the TinyGPU server, without booting anything. Mirrors NVDev._early_ip_init's includes.
import time, socket, tinygrad.runtime.autogen.nv_regs
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.support.system import APLRemotePCIDevice, System
from tinygrad.runtime.support.nv.nvdev import NVReg

class MiniNVDev:
  def __init__(self, timeout=20, map_bar0=True):
    self.pd = APLRemotePCIDevice("NV", System.pci_scan_bus(0x10de, ((0xff00, (0x2b00,)),), 0x03)[0]); self.pd.sock.settimeout(timeout)
    self.mmio = self.pd.map_bar(0, fmt='I') if map_bar0 else None
    for name, arch in (("nv_ref", ""), ("dev_fb", "tu102"), ("dev_gc6_island", "ga102"), ("dev_therm", "gb202")): self.include(name, arch)
  def include(self, name, arch):
    for k, v in getattr(getattr(tinygrad.runtime.autogen.nv_regs, name), arch or 'regs').items():
      self.__dict__[k] = NVReg(self, *v) if isinstance(v, tuple) else v
  def reg(self, name): return self.__dict__[name]
  def rreg(self, addr): return self.mmio[addr // 4]
  def wreg(self, addr, value): self.mmio[addr // 4] = value
  def remap_bar0(self): self.mmio = self.pd.map_bar(0, fmt='I')
  # config helpers
  def cfg(self, off, size=4): return self.pd.read_config(off, size)
  def cfgw(self, off, val, size=4): self.pd.write_config(off, val, size)
  def command(self): return self.cfg(pci.PCI_COMMAND, 2)
  def snapshot_cfg(self): return {off: self.cfg(off) for off in (0x00, 0x04, 0x0c, 0x10, 0x14, 0x18, 0x1c, 0x20, 0x24, 0x3c)}
  def gfw_booted(self):  # NV_FLCN_COT.wait_for_reset condition for GB202 (ip.py:286): THERM I2CS scratch == 0xff
    return self.NV_THERM_I2CS_SCRATCH.read() == 0xff
  def gfw_scratch(self): return self.NV_THERM_I2CS_SCRATCH.read()
  def state(self):
    b0, b42 = self.NV_PMC_BOOT_0.read(), self.NV_PMC_BOOT_42.read_bitfields()
    wpr2 = self.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read()
    return dict(boot0=f"0x{b0:08x}", arch=f"0x{b42['architecture']:02x}", impl=b42['implementation'], mmio_all_ones=(b0 == 0xffffffff),
                wpr2_addr_hi=f"0x{wpr2:08x}", gsp_resident=(wpr2 != 0), gfw_booted=self.gfw_booted(), therm_scratch=f"0x{self.gfw_scratch():08x}")
  def close(self): self.pd.sock.close()
