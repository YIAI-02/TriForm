from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Set

@dataclass
class LRUCache:
    """Simple LRU cache with pin support (pinned items are never evicted)."""
    capacity: int
    used: int = 0 
    #field(default_factory=dict) creates a new dict for each instance，如果只是写空，所有的LRUCache实例会共享同一个字典，导致缓存行为异常
    items: Dict[str, int] = field(default_factory=dict)  # weight_id -> size 当前缓存的权重，[weight_id] = size
    order: List[str] = field(default_factory=list)       # LRU order (oldest first) 记录缓存的顺序
    pinned: Set[str] = field(default_factory=set)        # pinned weight_ids 被pinned的权重集和，不会被淘汰

    def has(self, wid: str) -> bool:
        return wid in self.items

    def touch(self, wid: str): #更新顺序，最近被访问的放到最后
        if wid in self.order:
            try:
                self.order.remove(wid)
            except ValueError:
                pass
        self.order.append(wid)

    def pin(self, wid: str):
        if wid in self.items:
            self.pinned.add(wid)
            self.touch(wid)

    def add(self, wid: str, size: int, pinned: bool = False) -> bool:
        # Evict only non-pinned items until fits
        if wid in self.items:
            # already present; optionally pin
            if pinned:
                self.pinned.add(wid)
            self.touch(wid)
            return True
        while self.used + size > self.capacity and self.order: #order里面有值就返回true
            ev = self.order.pop(0)
            if ev in self.pinned:
                # skip pinned, move to end to prevent infinite loop
                self.order.append(ev)
                # If all are pinned we cannot evict further
                if all(x in self.pinned for x in self.order):
                    break #直接跳出while循环
                continue #跳过当前pinned 项，继续下一个ev
            evsz = self.items.pop(ev, 0)
            self.used -= evsz
        if self.used + size <= self.capacity:
            self.items[wid] = size
            self.order.append(wid)
            self.used += size
            if pinned:
                self.pinned.add(wid)
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

    def get_host_fmt(self, wid: str) -> str: #获取在主存中的存储格式
        return self.host_format.get(wid, "ND")

    def set_host_fmt(self, wid: str, fmt: str): #设置在主存中的存储格式
        self.host_format[wid] = fmt

    def ensure_device_cache(self, dev_name: str, capacity_bytes: int): #给每个设备创建一个cache
        if dev_name not in self.device_cache:
            self.device_cache[dev_name] = LRUCache(capacity=capacity_bytes)

    def is_cached(self, dev_name: str, wid: str) -> bool: #检查某个权重在某个设备上是否已经缓存
        c = self.device_cache.get(dev_name)
        return c.has(wid) if c else False

    def mark_cached(self, dev_name: str, wid: str, size: int, pinned: bool = False):
        c = self.device_cache[dev_name]
        c.add(wid, size, pinned=pinned)
        c.touch(wid)
        if pinned:
            c.pin(wid)

    def convert_time(self, from_fmt: str, to_fmt: str, size_bytes: int) -> float:
        if from_fmt == to_fmt:
            return 0.0
        bw = self.conv_bw_GBs.get((from_fmt, to_fmt), 0.0)
        if bw <= 0:
            # conservative: 10 GB/s default
            bw = 10.0
        self.stats_convert_bytes[(from_fmt, to_fmt)] = self.stats_convert_bytes.get((from_fmt, to_fmt), 0) + size_bytes #get用法，如果这个key已经存在就返回原来的value，如果不存在就默认返回0
        return size_bytes / (bw * 1e9)

    def avg_weight_load_time(self, wid: str, size_bytes: int, dev_type: str) -> float:
        # Estimate conversion (host -> device-native) + DMA (host->device) on first touch
        from_fmt = self.get_host_fmt(wid)
        to_fmt = self.device_native_fmt(dev_type)
        conv_t = self.convert_time(from_fmt, to_fmt, size_bytes)
        # DMA modeled elsewhere by link bw; here return just conv time proxy for rank
        return conv_t