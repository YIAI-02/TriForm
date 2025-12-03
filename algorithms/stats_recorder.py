
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import csv
from pathlib import Path
from collections import defaultdict
from typing import Tuple


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
    
    def dump_overlap_csv(
        self,
        segments_csv_path: Path | str,
        summary_csv_path: Path | str | None = None,
        *,
        include_idle: bool = False,
        include_all_phase: bool = True,
        comm_tags: set[str] | None = None,
    ) -> None:
        segments_csv_path = Path(segments_csv_path)
        segments_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if summary_csv_path is not None:
            summary_csv_path = Path(summary_csv_path)
            summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # ---- 收集区间：phase -> {npu/pim/comm: [(s,e), ...]} ----
        buckets: dict[str, dict[str, list[Tuple[float, float]]]] = defaultdict(
            lambda: {"npu": [], "pim": [], "comm": []}
        )

        def _phase(x) -> str:
            return str(x) if (x is not None and x != "") else "unknown"

        def _add(ph: str, kind: str, s: float, e: float):
            s = float(s); e = float(e)
            if e <= s:
                return
            buckets[ph][kind].append((s, e))
            if include_all_phase:
                buckets["ALL"][kind].append((s, e))

        for e in self.op_device_events:
            ph = _phase(e.get("phase"))
            dt = (e.get("device_type") or "").lower()
            if dt == "npu":
                _add(ph, "npu", e["start"], e["end"])
            elif dt == "pim":
                _add(ph, "pim", e["start"], e["end"])

        for e in self.comm_events:
            if comm_tags is not None and e.get("tag") not in comm_tags:
                continue
            ph = _phase(e.get("phase"))
            _add(ph, "comm", e["start"], e["end"])

        # ---- label：8 个互斥区域（含 idle / 三者 overlap）----
        def _label(n: bool, p: bool, c: bool) -> str:
            if n and (not p) and (not c): return "NPU_only"
            if (not n) and p and (not c): return "PIM_only"
            if (not n) and (not p) and c: return "COMM_only"
            if n and p and (not c): return "NPU+PIM"
            if n and (not p) and c: return "NPU+COMM"
            if (not n) and p and c: return "PIM+COMM"
            if n and p and c: return "NPU+PIM+COMM"
            return "IDLE"

        labels = ["IDLE","NPU_only","PIM_only","COMM_only","NPU+PIM","NPU+COMM","PIM+COMM","NPU+PIM+COMM"]

        # ---- segments 输出 + 汇总 ----
        seg_rows: list[dict] = []
        summary_rows: list[dict] = []

        for ph, kinds in buckets.items():
            # events: (t, dn, dp, dc)
            ev: list[Tuple[float, int, int, int]] = []
            for s, e in kinds["npu"]:
                ev.append((s, +1, 0, 0)); ev.append((e, -1, 0, 0))
            for s, e in kinds["pim"]:
                ev.append((s, 0, +1, 0)); ev.append((e, 0, -1, 0))
            for s, e in kinds["comm"]:
                ev.append((s, 0, 0, +1)); ev.append((e, 0, 0, -1))

            if not ev:
                continue

            ev.sort(key=lambda x: x[0])
            first_t = ev[0][0]
            last_t = ev[-1][0]

            n_cnt = p_cnt = c_cnt = 0
            prev_t = first_t
            totals = defaultdict(float)

            i = 0
            last_emitted = None  # 用于合并相邻同 label 段
            while i < len(ev):
                t = ev[i][0]
                dt = t - prev_t
                if dt > 0:
                    lab = _label(n_cnt > 0, p_cnt > 0, c_cnt > 0)
                    totals[lab] += dt

                    if include_idle or lab != "IDLE":
                        if (
                            last_emitted is not None
                            and last_emitted["phase"] == ph
                            and last_emitted["label"] == lab
                            and last_emitted["end"] == prev_t
                        ):
                            # 合并
                            last_emitted["end"] = t
                            last_emitted["duration"] += dt
                        else:
                            row = {
                                "phase": ph,
                                "start": prev_t,
                                "end": t,
                                "duration": dt,
                                "npu_active": int(n_cnt > 0),
                                "pim_active": int(p_cnt > 0),
                                "comm_active": int(c_cnt > 0),
                                "label": lab,
                            }
                            seg_rows.append(row)
                            last_emitted = row

                # apply all deltas at t
                while i < len(ev) and ev[i][0] == t:
                    n_cnt += ev[i][1]
                    p_cnt += ev[i][2]
                    c_cnt += ev[i][3]
                    i += 1
                prev_t = t

            makespan = last_t - first_t
            summary = {
                "phase": ph,
                "t_start": first_t,
                "t_end": last_t,
                "makespan": makespan,
            }
            for lab in labels:
                summary[lab] = totals.get(lab, 0.0)
            summary["ACTIVE_any"] = makespan - totals.get("IDLE", 0.0)
            summary_rows.append(summary)

        # ---- 写 CSV ----
        seg_fields = ["phase","start","end","duration","npu_active","pim_active","comm_active","label"]
        with segments_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=seg_fields)
            w.writeheader()
            for r in seg_rows:
                w.writerow(r)

        if summary_csv_path is not None:
            sum_fields = ["phase","t_start","t_end","makespan"] + labels + ["ACTIVE_any"]
            with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=sum_fields)
                w.writeheader()
                for r in summary_rows:
                    w.writerow({k: r.get(k) for k in sum_fields})

    
