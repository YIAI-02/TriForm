from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import json

@dataclass
class DeviceSpec:
    name: str
    type: str          # 'cpu' | 'npu' | 'pim'
    tflops: float      # peak TFLOPS (FP16-equivalent)
    mem_bw_GBs: float  # memory bandwidth GB/s (HBM/DRAM/near-memory)
    mem_capacity_GB: float
    pim_type: Optional[str] = None
    attached_npu: Optional[str] = None # 若 pim_type 为 'dram'/'hbm'，表明这个 PIM-DRAM 绑定到哪一个 NPU

class Cluster:
    def __init__(self):
        self.devices: Dict[str, DeviceSpec] = {}
        self.links: Dict[Tuple[str,str], float] = {}

    def add_device(self, dev: DeviceSpec):
        self.devices[dev.name] = dev

    def connect(self, a: str, b: str, bw_GBs: float):
        self.links[(a,b)] = bw_GBs
        self.links[(b,a)] = bw_GBs

    def get_link_bw(self, a: str, b: str) -> float:
        if a == b:
            return self.devices[a].mem_bw_GBs
        return self.links.get((a,b), 0.0)

    def devices_by_type(self, t: str) -> List[DeviceSpec]: #传进来t是cpu/npu/pim，筛选出是这个dev的所有设备
        return [d for d in self.devices.values() if d.type == t]

def demo_cluster(cfg: Dict | None = None) -> Cluster:
    c = Cluster()
    hw_cfg = None
    if isinstance(cfg, dict):
        hw_json_path = cfg.get('hardware_json')
        if isinstance(hw_json_path, str) and hw_json_path.strip():
            try:
                p = Path(hw_json_path)
                raw = json.loads(p.read_text(encoding='utf-8'))
                if isinstance(raw, dict):
                    print(f"DEBUG: Loaded hardware config from {hw_json_path}")
                    hw_cfg = raw.get('hardware') or raw.get('cluster') or raw
            except Exception:
                hw_cfg = None
        if hw_cfg is None:
            hw_cfg = cfg.get('hardware') or cfg.get('cluster')
    if not hw_cfg:
        print("DEBUG: Using built-in demo hardware config")
        c.add_device(DeviceSpec("CPU0","cpu",tflops=2.0,mem_bw_GBs=51.2,mem_capacity_GB=0.009,))
        c.add_device(DeviceSpec("NPU0","npu",tflops=10.0,mem_bw_GBs=51.2,mem_capacity_GB=0.003,))

        # Multi-PIM support (no inter-PIM communication assumed):
        # You can override the number of PIMs by providing cfg['num_pim'] (or 'pim_count').
        num_pim = 1
        if isinstance(cfg, dict):
            try:
                num_pim = int(cfg.get('num_pim', cfg.get('pim_count', 1)) or 1)
            except Exception:
                num_pim = 1
        num_pim = max(1, num_pim)

        # Add PIM devices (default: identical PIMA{i})
        for i in range(num_pim):
            name = f"PIMA{i}"
            c.add_device(DeviceSpec(name,"pim",tflops=1.0,mem_bw_GBs=32.0,mem_capacity_GB=1.0,pim_type='accel'))

        # Links
        c.connect("CPU0","NPU0", 32.0)
        for i in range(num_pim):
            c.connect("CPU0", f"PIMA{i}", 32.0)
        return c

    # Parse loaded hw_cfg
    for d in hw_cfg.get('devices', []):
        c.add_device(DeviceSpec(
            name=d['name'],
            type=d['type'],
            tflops=float(d.get('tflops', 0)),
            mem_bw_GBs=float(d.get('mem_bw_GBs', 0)),
            mem_capacity_GB=float(d.get('mem_capacity_GB', 0)),
            pim_type=d.get('pim_type'),
            attached_npu=d.get('attached_npu')
        ))
    for l in hw_cfg.get('links', []):
        c.connect(l['a'], l['b'], float(l['bw_GBs']))
    return c
