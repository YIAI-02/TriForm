
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import csv
from pathlib import Path

@dataclass
class StatsRecorder:
    phase: Optional[str] = None
    op_device_events: List[Dict[str, Any]] = field(default_factory=list)
    op_cp_events: List[Dict[str, Any]] = field(default_factory=list)
    comm_events: List[Dict[str, Any]] = field(default_factory=list)

    # ---- phase control ----
    def set_phase(self, phase: Optional[str]) -> None:
        self.phase = phase

    def reset(self) -> None:
        self.op_device_events.clear()
        self.op_cp_events.clear()
        self.comm_events.clear()

    # ---- logging APIs ----
    def log_op_device(self, *, nid: str, op: str, device: str, device_type: str,
                      start: float, end: float, mode: str = 'single', extra: Optional[Dict[str, Any]] = None) -> None:
        try:
            dur = float(end) - float(start)
        except Exception:
            dur = None
        self.op_device_events.append({
            'phase': self.phase,
            'node_id': nid,
            'op': op,
            'device': device,
            'device_type': device_type,
            'mode': mode,
            'start': float(start),
            'end': float(end),
            'duration': float(dur) if dur is not None else None,
            **(extra or {})
        })

    def log_op_cp(self, *, nid: str, op: str, start: float, end: float, mode: str = 'single',
                  extra: Optional[Dict[str, Any]] = None) -> None:
        try:
            dur = float(end) - float(start)
        except Exception:
            dur = None
        self.op_cp_events.append({
            'phase': self.phase,
            'node_id': nid,
            'op': op,
            'mode': mode,
            'start': float(start),
            'end': float(end),
            'duration': float(dur) if dur is not None else None,
            **(extra or {})
        })

    def log_comm(self, *, src: str, dst: str, bytes: int, start: float, end: float,
                 tag: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        try:
            dur = float(end) - float(start)
        except Exception:
            dur = None
        def _type(name: Optional[str]) -> str:
            n = (name or '').lower()
            if 'npu' in n: return 'npu'
            if 'pim' in n: return 'pim'
            if 'cpu' in n or 'host' in n: return 'cpu'
            return 'other'
        self.comm_events.append({
            'phase': self.phase,
            'src': src, 'src_type': _type(src),
            'dst': dst, 'dst_type': _type(dst),
            'bytes': int(bytes) if bytes is not None else 0,
            'start': float(start),
            'end': float(end),
            'duration': float(dur) if dur is not None else None,
            'tag': tag,
            **(extra or {})
        })

    # ---- CSV export ----
    def dump_csv(self, ops_csv_path: Path | str, comms_csv_path: Path | str) -> None:
        ops_csv_path = Path(ops_csv_path)
        comms_csv_path = Path(comms_csv_path)
        ops_csv_path.parent.mkdir(parents=True, exist_ok=True)
        comms_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) operator device occupancy events
        if self.op_device_events:
            fieldnames = ['phase','node_id','op','device','device_type','mode','start','end','duration']
            with ops_csv_path.open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for e in self.op_device_events:
                    w.writerow({k: e.get(k) for k in fieldnames})
        else:
            # still create empty file with header
            fieldnames = ['phase','node_id','op','device','device_type','mode','start','end','duration']
            with ops_csv_path.open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()

        # 2) communication events
        if self.comm_events:
            fieldnames = ['phase','src','src_type','dst','dst_type','bytes','start','end','duration','tag']
            with comms_csv_path.open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for e in self.comm_events:
                    w.writerow({k: e.get(k) for k in fieldnames})
        else:
            fieldnames = ['phase','src','src_type','dst','dst_type','bytes','start','end','duration','tag']
            with comms_csv_path.open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
