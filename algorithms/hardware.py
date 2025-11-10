from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class DeviceSpec:
    name: str
    type: str          # 'cpu' | 'npu' | 'pim'
    tflops: float      # peak TFLOPS (FP16-equivalent)
    mem_bw_GBs: float  # memory bandwidth GB/s (HBM/DRAM/near-memory)
    onchip_bw_GBs: float  # on-chip SRAM/shared mem BW GB/s
    mem_capacity_GB: float

class Cluster:
    def __init__(self):
        self.devices: Dict[str, DeviceSpec] = {}
        self.links: Dict[Tuple[str,str], Tuple[float, str]] = {}

    def add_device(self, dev: DeviceSpec):
        self.devices[dev.name] = dev

    def connect(self, a: str, b: str, bw_GBs: float, link_type: str = "PCIe"):
        self.links[(a,b)] = (bw_GBs, link_type)
        self.links[(b,a)] = (bw_GBs, link_type)

    def get_link_bw(self, a: str, b: str) -> float:
        if a == b:
            return max(self.devices[a].onchip_bw_GBs, self.devices[a].mem_bw_GBs)
        return self.links.get((a,b), (12.0, "PCIe"))[0]

    def get_link_type(self, a: str, b: str) -> str:
        if a == b: return "LOCAL"
        return self.links.get((a,b), (12.0, "PCIe"))[1]

    def devices_by_type(self, t: str) -> List[DeviceSpec]: #传进来t是cpu/npu/pim，筛选出是这个dev的所有设备
        return [d for d in self.devices.values() if d.type == t]

def demo_cluster() -> Cluster:
    c = Cluster()
    c.add_device(DeviceSpec("CPU0","cpu", tflops=3.0,  mem_bw_GBs=51.2, onchip_bw_GBs=128.0, mem_capacity_GB=0.009))
    c.add_device(DeviceSpec("NPU0","npu", tflops=10.0, mem_bw_GBs=51.2, onchip_bw_GBs=128.0, mem_capacity_GB=0.003))
    # Two PIM stacks as example
    c.add_device(DeviceSpec("PIM0","pim", tflops=1.0,  mem_bw_GBs=32.0, onchip_bw_GBs=512, mem_capacity_GB=1))
    c.add_device(DeviceSpec("PIM1","pim", tflops=1.0,  mem_bw_GBs=32.0, onchip_bw_GBs=512, mem_capacity_GB=1))
    # Links
    # c.connect("CPU0","NPU0", 24.0, "PCIe")
    # c.connect("CPU0","PIM0", 24.0, "PCIe")
    # c.connect("CPU0","PIM1", 24.0, "PCIe")
    # c.connect("NPU0","PIM0", 100.0, "NVLink")
    # c.connect("NPU0","PIM1", 100.0, "NVLink")
    return c
