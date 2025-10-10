from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

@dataclass
class LRUCache:
    capacity: int
    used: int = 0
    items: Dict[str, int] = field(default_factory=dict)  # weight_id -> size
    order: List[str] = field(default_factory=list)       # LRU order (oldest first)

    def has(self, wid: str) -> bool:
        return wid in self.items

    def touch(self, wid: str):
        if wid in self.order:
            self.order.remove(wid)
        self.order.append(wid)

    def add(self, wid: str, size: int):
        # evict until fit
        while self.used + size > self.capacity and self.order:
            ev = self.order.pop(0)
            evsz = self.items.pop(ev, 0)
            self.used -= evsz
        if self.used + size <= self.capacity:
            self.items[wid] = size
            self.order.append(wid)
            self.used += size
            return True
        return False

@dataclass
class BufferManager:
    # Host-side storage format per weight_id: 'ND' | 'npu-opt' | 'pim-opt'
    host_format: Dict[str, str] = field(default_factory=dict)
    # Conversion throughputs (GB/s) for format conversion
    conv_bw_GBs: Dict[Tuple[str,str], float] = field(default_factory=dict)
    # Per-device caches (LRU by bytes)
    device_cache: Dict[str, LRUCache] = field(default_factory=dict)
    # Statistics for recommendation
    stats_convert_bytes: Dict[Tuple[str,str], int] = field(default_factory=dict)
    stats_weight_loads: Dict[Tuple[str,str,str], int] = field(default_factory=dict)  # (wid, from_fmt, to_fmt) -> total_bytes

    def device_native_fmt(self, dev_type: str) -> str:
        return "pim-opt" if dev_type == "pim" else ("npu-opt" if dev_type == "npu" else "ND")

    def get_host_fmt(self, wid: str) -> str:
        return self.host_format.get(wid, "ND")

    def set_host_fmt(self, wid: str, fmt: str):
        self.host_format[wid] = fmt

    def ensure_device_cache(self, dev_name: str, capacity_bytes: int):
        if dev_name not in self.device_cache:
            self.device_cache[dev_name] = LRUCache(capacity=capacity_bytes)

    def is_cached(self, dev_name: str, wid: str) -> bool:
        c = self.device_cache.get(dev_name)
        return c.has(wid) if c else False

    def mark_cached(self, dev_name: str, wid: str, size: int):
        c = self.device_cache[dev_name]
        c.add(wid, size)
        c.touch(wid)

    def convert_time(self, from_fmt: str, to_fmt: str, size_bytes: int) -> float:
        if from_fmt == to_fmt:
            return 0.0
        bw = self.conv_bw_GBs.get((from_fmt, to_fmt), 0.0)
        if bw <= 0:  # if not provided, assume symmetric fallback
            bw = self.conv_bw_GBs.get((to_fmt, from_fmt), 0.0)
        if bw <= 0:
            # conservative: 10 GB/s default
            bw = 10.0
        self.stats_convert_bytes[(from_fmt, to_fmt)] = self.stats_convert_bytes.get((from_fmt, to_fmt), 0) + size_bytes
        return size_bytes / (bw * 1e9)

    def avg_weight_load_time(self, wid: str, size_bytes: int, dev_type: str) -> float:
        # Estimate conversion (host -> device-native) + DMA (host->device) on first touch
        from_fmt = self.get_host_fmt(wid)
        to_fmt = self.device_native_fmt(dev_type)
        conv_t = self.convert_time(from_fmt, to_fmt, size_bytes)
        # DMA modeled elsewhere by link bw; here return just conv time proxy for rank
        return conv_t

    def record_weight_load(self, wid: str, from_fmt: str, to_fmt: str, size_bytes: int, dev_name: str):
        key = (wid, from_fmt, to_fmt)
        self.stats_weight_loads[key] = self.stats_weight_loads.get(key, 0) + size_bytes

    def recommend_host_formats(self) -> Dict[str, str]:
        # Simple heuristic: if more bytes converted to pim-opt than to npu-opt, store as pim-opt, else npu-opt
        per_weight_bytes = {}
        for (wid, src, dst), bytes_amt in self.stats_weight_loads.items():
            d = per_weight_bytes.setdefault(wid, {"npu-opt":0, "pim-opt":0, "ND":0})
            d[dst] += bytes_amt
        rec = {}
        for wid, m in per_weight_bytes.items():
            # choose argmax among pim-opt/npu-opt/ND
            best_fmt = max(m.items(), key=lambda kv: kv[1])[0]
            rec[wid] = best_fmt
        return rec
