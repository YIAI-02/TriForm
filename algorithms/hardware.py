from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
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
    pim_memory: Dict[str, Any] = field(default_factory=dict)
    cpu_read_latency_ns: float = 0.0
    cpu_write_latency_ns: float = 0.0
    cpu_access_bytes_B: int = 64
    freq_ghz: float = 0.0

class Cluster:
    def __init__(self):
        self.devices: Dict[str, DeviceSpec] = {}
        self.links: Dict[Tuple[str, str], float] = {}
        self.link_specs: Dict[Tuple[str, str], LinkSpec] = {}
        self.topology: str = 'fc'
        self.default_link_spec: Optional[LinkSpec] = None
        self.pim_memory: Dict[str, Any] = {}

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
        """Return a LinkSpec for (a->b)."""
        a = str(a); b = str(b)
        if a == b:
            # Intra-device access is not modeled as a link transfer; keep packet overhead disabled.
            return LinkSpec(bw_GBs=float(self.devices[a].mem_bw_GBs), flit_size_B=0, max_payload_B=0)

        spec = self.link_specs.get((a, b))
        if spec is not None:
            return spec

        # FC fabrics: unspecified pairs use the shared default link spec.
        if str(getattr(self, 'topology', '')).strip().lower() == 'fc' and self.default_link_spec is not None:
            return self.default_link_spec

        bw = float(self.links.get((a, b), 0.0))
        return LinkSpec(bw_GBs=bw)

    def get_link_bw(self, a: str, b: str) -> float:
        a = str(a); b = str(b)
        if a == b:
            return float(self.devices[a].mem_bw_GBs)
        bw = self.links.get((a, b), None)
        if bw is not None:
            return float(bw)
        if str(getattr(self, 'topology', '')).strip().lower() == 'fc' and self.default_link_spec is not None:
            return float(getattr(self.default_link_spec, 'bw_GBs', 0.0) or 0.0)
        return 0.0

    def devices_by_type(self, t: str) -> List[DeviceSpec]: #传进来t是cpu/npu/pim，筛选出是这个dev的所有设备
        return [d for d in self.devices.values() if d.type == t]


def _pim_capacity_bytes(pim_mem_cfg: Dict[str, Any]) -> Optional[int]:
    """Compute total PIM memory capacity in bytes from an address map."""
    if not isinstance(pim_mem_cfg, dict):
        return None

    addr = pim_mem_cfg.get('addr_map') or pim_mem_cfg.get('address_map') or pim_mem_cfg.get('addrmap')
    if not isinstance(addr, dict):
        return None

    unit = str(pim_mem_cfg.get('addr_map_unit', pim_mem_cfg.get('addr_map_units', 'bits')) or 'bits').strip().lower()

    def _get_int(*keys: str) -> Optional[int]:
        for k in keys:
            if k in addr and addr[k] is not None:
                try:
                    return int(addr[k])
                except Exception:
                    return None
        return None

    row = _get_int('line', 'lines', 'row', 'rows')
    ch = _get_int('channel', 'channels')
    bk = _get_int('bank', 'banks')
    col = _get_int('column', 'columns')
    off = _get_int('offset', 'offset_bits', 'offset_bytes')

    # Explicit override if user provides a capacity directly.
    cap_b = pim_mem_cfg.get('capacity_bytes') or pim_mem_cfg.get('capacity_B')
    if cap_b is not None:
        try:
            return int(cap_b)
        except Exception:
            pass

    if unit in ('bits', 'bit'):
        # Treat each field as number of address bits.
        if None in (row, ch, bk, col, off):
            return None
        total_bits = int(row) + int(ch) + int(bk) + int(col) + int(off)
        if total_bits < 0 or total_bits > 62:
            return None
        return int(1) << int(total_bits)

    # Otherwise treat fields as counts.
    if None in (row, ch, bk, col):
        return None
    # Line bytes.
    line_bytes = pim_mem_cfg.get('line_bytes') or pim_mem_cfg.get('line_bytes_B') or pim_mem_cfg.get('line_size_B')
    if line_bytes is None:
        # Interpret `offset` as bytes when unit != bits.
        if off is None:
            return None
        line_bytes = off
    try:
        line_bytes_i = max(1, int(line_bytes))
    except Exception:
        return None

    try:
        return int(row) * int(ch) * int(bk) * int(col) * int(line_bytes_i)
    except Exception:
        return None


def _check_pim_capacity_matches(dev_name: str, mem_capacity_GB: float, pim_mem_cfg: Dict[str, Any]) -> None:
    """Validate that address-map-derived capacity matches mem_capacity_GB."""
    try:
        cap_b = _pim_capacity_bytes(pim_mem_cfg)
    except Exception:
        cap_b = None
    if cap_b is None:
        raise ValueError(
            f"PIM device '{dev_name}' is missing a valid pim_memory.addr_map for capacity check. "
            f"Please set devices[].pim_memory={{'addr_map_unit':..., 'addr_map':{{row/channel/bank/column/offset}}}}."
        )

    cap_gib = float(cap_b) / float(1024 ** 3) / 8 
    target = float(mem_capacity_GB or 0.0)
    # Allow a tiny tolerance (floating JSON values).
    tol = max(1e-6, abs(target) * 1e-6)
    if abs(cap_gib - target) > tol:
        raise ValueError(
            f"PIM device '{dev_name}' capacity mismatch: mem_capacity_GB={target} GiB, "
            f"addr_map_capacity={cap_gib:.6f} GiB (bytes={cap_b}). "
            f"Please ensure channel*bank*column*line*(line_bytes) matches mem_capacity_GB."
        )

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

    # Topology
    topo = None
    try:
        topo = hw_cfg.get('topology') or (hw_cfg.get('interconnect') or {}).get('topology')
    except Exception:
        topo = None
    if not topo:
        # Best-effort inference: if all links touch a single CPU host, treat as star; else fc.
        try:
            devs = hw_cfg.get('devices', []) or []
            cpu_names = [str(d.get('name')) for d in devs if str(d.get('type','')).lower() == 'cpu' and d.get('name') is not None]
            host = cpu_names[0] if cpu_names else None
            links = hw_cfg.get('links', []) or []
            non_host_links = 0
            for lk in links:
                if not isinstance(lk, dict):
                    continue
                a = str(lk.get('a',''))
                b = str(lk.get('b',''))
                if host and a != host and b != host:
                    non_host_links += 1
            topo = 'star' if (host and non_host_links == 0) else 'fc'
        except Exception:
            topo = 'fc'
    c.topology = str(topo).strip().lower()

    # Normalize common topology aliases so downstream checks can simply use 'fc'/'star'.
    if c.topology in ('fully_connected', 'fully-connected', 'full', 'mesh'):
        c.topology = 'fc'
    elif c.topology in ('host', 'host_star', 'host-centric', 'host_centric'):
        c.topology = 'star'

    try:
        pm = hw_cfg.get('pim_memory') or hw_cfg.get('pim_mem') or {}
        if isinstance(pm, dict):
            c.pim_memory = pm
    except Exception:
        c.pim_memory = {}

    # Parse loaded hw_cfg
    for d in hw_cfg.get('devices', []):
        if not isinstance(d, dict):
            continue

        dev_type = str(d.get('type', '') or '').strip().lower()
        pim_mem = d.get('pim_memory') or d.get('pim_mem') or {}
        cpu_access_raw = (
            d.get('cpu_access_bytes_B')
            or d.get('cpu_access_bytes')
            or d.get('cpu_line_bytes_B')
            or d.get('cpu_cacheline_B')
            or d.get('cacheline_B')
        )
        try:
            cpu_access_bytes_B = max(1, int(cpu_access_raw)) if cpu_access_raw is not None else 64
        except Exception:
            cpu_access_bytes_B = 64
        dev = DeviceSpec(
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
            pim_memory=(pim_mem if isinstance(pim_mem, dict) else {}),

            cpu_read_latency_ns=float(d.get('cpu_read_latency_ns', d.get('read_latency_ns', 0.0)) or 0.0),
            cpu_write_latency_ns=float(d.get('cpu_write_latency_ns', d.get('write_latency_ns', 0.0)) or 0.0),
            cpu_access_bytes_B=int(cpu_access_bytes_B),
            freq_ghz=float(d.get('freq_ghz', d.get('clock_ghz', d.get('pim_freq_ghz', 0.0))) or 0.0),
        )

        # Capacity consistency check (required):
        #   channel * bank * column * line (and offset/line_bytes) must match mem_capacity_GB.
        if str(dev.type).strip().lower() == 'pim':
            _check_pim_capacity_matches(str(dev.name), float(dev.mem_capacity_GB), dev.pim_memory)

        c.add_device(dev)
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

    # -----------------------------
    # FC topology: single shared bandwidth for all (unspecified) pairs
    # -----------------------------
    def _read_bw_gbs(obj: Any, *keys: str) -> Optional[float]:
        if not isinstance(obj, dict):
            return None
        for k in keys:
            if k in obj and obj[k] is not None:
                try:
                    v = float(obj[k])
                    if v > 0.0:
                        return float(v)
                except Exception:
                    continue
        return None

    if c.topology == 'fc':
        bw_fc = _read_bw_gbs(
            hw_cfg,
            'fc_bw_GBs', 'fc_bw', 'fc_bandwidth_GBs', 'fc_bandwidth',
            'link_bw_GBs', 'link_bw', 'link_bandwidth_GBs', 'link_bandwidth',
            'bw_GBs', 'bw', 'bandwidth_GBs', 'bandwidth',
        )
        if bw_fc is None:
            try:
                bws: List[float] = []
                for lk in hw_cfg.get('links', []) or []:
                    if not isinstance(lk, dict):
                        continue
                    v = lk.get('bw_GBs', lk.get('bw', lk.get('bandwidth_GBs', lk.get('bandwidth', None))))
                    if v is None:
                        continue
                    fv = float(v)
                    if fv > 0.0:
                        bws.append(fv)
                if bws:
                    bw_fc = float(min(bws))
            except Exception:
                bw_fc = None

        if bw_fc is not None and float(bw_fc) > 0.0:
            # Use the same latency/overhead/packet parameters as link_defaults unless explicitly overridden per-link.
            c.default_link_spec = LinkSpec(
                bw_GBs=float(bw_fc),
                latency_s=float(d_lat),
                overhead_s=float(d_ovh),
                flit_size_B=int(d_flit),
                max_payload_B=int(d_maxp),
            )
        else:
            # If users declare FC but provide neither a common bandwidth nor any explicit links,
            # the comm model becomes unusable. Fail fast with a helpful error.
            has_any_link = False
            try:
                for lk in hw_cfg.get('links', []) or []:
                    if isinstance(lk, dict) and lk.get('a') and lk.get('b'):
                        has_any_link = True
                        break
            except Exception:
                has_any_link = False
            if not has_any_link and len(getattr(c, 'devices', {}) or {}) > 1:
                raise ValueError(
                    "FC topology requires a common link bandwidth or explicit links. "
                    "Please set one of: hardware.fc_bw_GBs / hardware.link_bw_GBs / hardware.interconnect.bw_GBs / "
                    "hardware.link_defaults.bw_GBs, or provide explicit hardware.links[]."
                )

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
