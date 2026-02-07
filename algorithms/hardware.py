from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import json


@dataclass(frozen=True)
class LinkSpec:
    """Point-to-point link specification.

    Supports the LogGP/AHEAD-style link model used by LLMCompass:
        T = L + O + n_hat / B
        n_hat = n + ceil(n / MaxPayload) * FlitSize

    All sizes are in bytes, time in seconds.
    """

    bw_GBs: float
    latency_s: float = 0.0
    overhead_s: float = 0.0
    flit_size_B: int = 16
    max_payload_B: int = 256

@dataclass
class DeviceSpec:
    name: str
    type: str          # 'cpu' | 'npu' | 'pim'
    tflops: float      # peak TFLOPS (FP16-equivalent)
    mem_bw_GBs: float  # memory bandwidth GB/s (HBM/DRAM/near-memory)
    mem_capacity_GB: float
    pim_type: Optional[str] = None
    attached_npu: Optional[str] = None
    # Optional device arch/model tags (used by external latency submodels like LLMCompass)
    arch: Optional[str] = None
    llmcompass_kind: Optional[str] = None  # e.g., 'a100' | 'tpuv3'
    # Optional near-memory access latency (used by CostModel.pim_mem_time in fast-mode)
    pim_read_latency_ns: float = 0.0
    pim_write_latency_ns: float = 0.0

class Cluster:
    def __init__(self):
        self.devices: Dict[str, DeviceSpec] = {}
        self.links: Dict[Tuple[str, str], float] = {}
        self.link_specs: Dict[Tuple[str, str], LinkSpec] = {}

    def add_device(self, dev: DeviceSpec):
        self.devices[dev.name] = dev

    def connect(
        self,
        a: str,
        b: str,
        bw_GBs: float | LinkSpec,
        *,
        latency_s: float = 0.0,
        overhead_s: float = 0.0,
        flit_size_B: int = 16,
        max_payload_B: int = 256,
    ) -> None:
        if isinstance(bw_GBs, LinkSpec):
            spec = bw_GBs
        else:
            spec = LinkSpec(
                bw_GBs=float(bw_GBs),
                latency_s=float(latency_s),
                overhead_s=float(overhead_s),
                flit_size_B=int(flit_size_B),
                max_payload_B=int(max_payload_B),
            )
        self.links[(a, b)] = float(spec.bw_GBs)
        self.links[(b, a)] = float(spec.bw_GBs)
        self.link_specs[(a, b)] = spec
        self.link_specs[(b, a)] = spec

    def get_link_spec(self, a: str, b: str) -> LinkSpec:
        """Return a LinkSpec for (a->b). If not present, returns a bw=0 spec."""
        if a == b:
            # Intra-device access is not modeled as a link transfer; keep packet overhead disabled.
            return LinkSpec(bw_GBs=float(self.devices[a].mem_bw_GBs), flit_size_B=0, max_payload_B=0)
        spec = self.link_specs.get((a, b))
        if spec is None:
            bw = float(self.links.get((a, b), 0.0))
            return LinkSpec(bw_GBs=bw)
        return spec

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
        raise ValueError(
            "Missing hardware config. "
            "Provide cfg['hardware'] or cfg['cluster'], or set cfg['hardware_json'] to a valid JSON file path."
        )

    # Parse loaded hw_cfg
    for d in hw_cfg.get('devices', []):
        c.add_device(DeviceSpec(
            name=d['name'],
            type=d['type'],
            tflops=float(d.get('tflops', 0)),
            mem_bw_GBs=float(d.get('mem_bw_GBs', 0)),
            mem_capacity_GB=float(d.get('mem_capacity_GB', 0)),
            pim_type=d.get('pim_type'),
            attached_npu=d.get('attached_npu'),
            pim_read_latency_ns=float(d.get('pim_read_latency_ns', d.get('read_latency_ns', 0.0)) or 0.0),
            pim_write_latency_ns=float(d.get('pim_write_latency_ns', d.get('write_latency_ns', 0.0)) or 0.0),
            arch=d.get('arch') or d.get('model') or d.get('device_arch'),
            llmcompass_kind=d.get('llmcompass_kind') or d.get('llmcompass') or d.get('llmcompass_device'),
        ))
    # Optional: global defaults for link-level comm parameters.
    link_defaults = {}
    try:
        link_defaults = hw_cfg.get('link_defaults') or hw_cfg.get('comm_defaults') or hw_cfg.get('comm') or {}
        if not isinstance(link_defaults, dict):
            link_defaults = {}
    except Exception:
        link_defaults = {}

    def _read_time_s(obj: dict, key_s: str, key_us: str, key_ns: str, default: float) -> float:
        try:
            if key_s in obj and obj[key_s] is not None:
                return float(obj[key_s])
            if key_us in obj and obj[key_us] is not None:
                return float(obj[key_us]) * 1e-6
            if key_ns in obj and obj[key_ns] is not None:
                return float(obj[key_ns]) * 1e-9
        except Exception:
            pass
        return float(default)

    d_flit = int(link_defaults.get('flit_size_B', link_defaults.get('flit_size', 16)) or 16)
    d_maxp = int(link_defaults.get('max_payload_B', link_defaults.get('max_payload', 256)) or 256)
    d_lat = _read_time_s(link_defaults, 'latency_s', 'latency_us', 'latency_ns', 0.0)
    d_ovh = _read_time_s(link_defaults, 'overhead_s', 'overhead_us', 'overhead_ns', 0.0)

    for l in hw_cfg.get('links', []):
        if not isinstance(l, dict):
            continue
        a = l.get('a')
        b = l.get('b')
        if not a or not b:
            continue
        bw = float(l.get('bw_GBs', l.get('bw', l.get('bandwidth_GBs', 0.0))) or 0.0)
        lat = _read_time_s(l, 'latency_s', 'latency_us', 'latency_ns', d_lat)
        ovh = _read_time_s(l, 'overhead_s', 'overhead_us', 'overhead_ns', d_ovh)
        flit = int(l.get('flit_size_B', l.get('flit_size', d_flit)) or d_flit)
        maxp = int(l.get('max_payload_B', l.get('max_payload', d_maxp)) or d_maxp)
        c.connect(str(a), str(b), float(bw), latency_s=float(lat), overhead_s=float(ovh), flit_size_B=int(flit), max_payload_B=int(maxp))
    return c