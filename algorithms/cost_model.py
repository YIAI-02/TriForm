from __future__ import annotations
from typing import Optional
from config import (
    GB_TO_DEVICE_BW, DEVICE_TO_GB_BW, DEVICE_TO_DEVICE_BW,
    DIRECT_INTERCONNECTS, INTERCONNECT_MODE,
    FORMAT_CONVERSION_PER_BYTE, DEVICE_PREFERRED_FORMAT,
    DEVICE_PEAK_THROUGHPUT, DEVICE_LAUNCH_OVERHEAD,
    FORMAT_SIZE_MULTIPLIER, INTRA_DEVICE_PASS_BW
)

class CostModel:
    """
    - same device op→op: L2/HBM/PIM_bw
    - cross-device op→op: direct-connect: NVLink3/PCIe/CXL or HOST relay
    - weight in HOST, GB→device
    """
    def __init__(self) -> None:
        self.gb2dev_bw = GB_TO_DEVICE_BW
        self.dev2gb_bw = DEVICE_TO_GB_BW
        merged = dict(DEVICE_TO_DEVICE_BW)
        merged.update(DIRECT_INTERCONNECTS.get(INTERCONNECT_MODE, {}))
        self.d2d_bw = merged
        self.conv_cost = FORMAT_CONVERSION_PER_BYTE
        self.pref_fmt = DEVICE_PREFERRED_FORMAT
        self.thr = DEVICE_PEAK_THROUGHPUT
        self.launch_over = DEVICE_LAUNCH_OVERHEAD
        self.scale = FORMAT_SIZE_MULTIPLIER
        self.local_bw = INTRA_DEVICE_PASS_BW

    # ---- compute ----
    def compute_time(self, work: float, device_type: str, op_name: Optional[str] = None) -> float:
        return self.launch_over[device_type] + work / self.thr[device_type]

    # ---- conversion ----
    def format_size(self, size_bytes: int, fmt: str) -> int:
        return int(size_bytes * self.scale[fmt])

    def format_conversion_time(self, src_fmt: str, dst_fmt: str, size_bytes: int) -> float:
        if src_fmt == dst_fmt:
            return 0.0
        per_b = self.conv_cost[(src_fmt, dst_fmt)]
        eff = max(self.format_size(size_bytes, src_fmt), self.format_size(size_bytes, dst_fmt))
        return per_b * eff

    # ---- transfers ----
    def gb_to_device_time(self, device_type: str, size_bytes: int) -> float:
        return size_bytes / self.gb2dev_bw[("GB", device_type)]

    def device_to_gb_time(self, device_type: str, size_bytes: int) -> float:
        return size_bytes / self.dev2gb_bw[(device_type, "GB")]

    def device_to_device_time(self, src: str, dst: str, size_bytes: int) -> float:
        if src == dst:
            return 0.0
        bw = self.d2d_bw.get((src, dst))
        if bw is None:
            return self.device_to_gb_time(src, size_bytes) + self.gb_to_device_time(dst, size_bytes)
        return size_bytes / bw

    def local_pass_time(self, device_type: str, size_bytes: int) -> float:
        return size_bytes / self.local_bw[device_type]

    # ---- compose ----
    def gb_move_and_format(self, device_type: str, size_bytes: int, src_fmt: str, dst_fmt: Optional[str] = None) -> float:
        if dst_fmt is None:
            dst_fmt = self.pref_fmt[device_type]
        return self.gb_to_device_time(device_type, self.format_size(size_bytes, src_fmt)) + \
               self.format_conversion_time(src_fmt, dst_fmt, size_bytes)

    def inter_device_comm(self, src_dev: str, dst_dev: str, size_bytes: int, src_fmt: str, dst_fmt: Optional[str] = None) -> float:
        if dst_fmt is None:
            dst_fmt = self.pref_fmt[dst_dev]
        if src_dev == "HYBRID":
            return self.gb_to_device_time(dst_dev, size_bytes) + self.format_conversion_time(src_fmt, dst_fmt, size_bytes)
        if dst_dev == "HYBRID":
            return self.device_to_gb_time(src_dev, size_bytes)
        return self.device_to_device_time(src_dev, dst_dev, size_bytes) + \
               self.format_conversion_time(src_fmt, dst_fmt, size_bytes)
