# hardware.py
from __future__ import annotations
from typing import List, Tuple, Dict

class Device:
    def __init__(self, dev_type: str, dev_id: int) -> None:
        assert dev_type in ("NPU", "PIM")
        self.type = dev_type
        self.id = dev_id
        self.available_time: float = 0.0
        self.timeline: List[Dict] = []

    def reserve(self, task_id: str, start: float, end: float) -> None:
        self.timeline.append({"task": task_id, "start": start, "end": end})
        self.available_time = end

class HardwareManager:
    def __init__(self, npu_count: int, pim_count: int) -> None:
        self.npus: List[Device] = [Device("NPU", i) for i in range(npu_count)]
        self.pims: List[Device] = [Device("PIM", i) for i in range(pim_count)]

    def devices_of_type(self, dev_type: str) -> List[Device]:
        return self.npus if dev_type == "NPU" else self.pims

    def earliest_available(self, dev_type: str) -> Tuple[Device, float]:
        devs = self.devices_of_type(dev_type)
        dev = min(devs, key=lambda d: d.available_time)
        return dev, dev.available_time

    def reserve(self, dev: Device, task_id: str, start: float, end: float) -> None:
        dev.reserve(task_id, start, end)

    def hybrid_earliest_available(self) -> Tuple[Device, Device, float]:
        npu, ta = self.earliest_available("NPU")
        pim, tb = self.earliest_available("PIM")
        return npu, pim, max(ta, tb)

    def reserve_hybrid(self, npu: Device, pim: Device, task_id: str, start: float, end: float) -> None:
        npu.reserve(task_id, start, end)
        pim.reserve(task_id, start, end)
