# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
from pathlib import Path
import csv

@dataclass
class StatsRecorder:
    phase: Optional[str] = None
    op_device_events: List[Dict[str, Any]] = field(default_factory=list)
    op_cp_events: List[Dict[str, Any]] = field(default_factory=list)
    comm_events: List[Dict[str, Any]] = field(default_factory=list)

    # -------------------- phase control --------------------
    def set_phase(self, phase: Optional[str]) -> None:
        self.phase = phase

    def reset(self) -> None:
        self.op_device_events.clear()
        self.op_cp_events.clear()
        self.comm_events.clear()

    # -------------------- logging --------------------
    def log_op_device(
        self,
        *,
        nid: str,
        op: str,
        device: str,
        device_type: str,
        start: float,
        end: float,
        mode: str = "single",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        s = float(start)
        e = float(end)
        self.op_device_events.append(
            {
                "phase": self.phase,
                "node_id": nid,
                "op": op,
                "device": device,
                "device_type": device_type,
                "mode": mode,
                "start": s,
                "end": e,
                "duration": float(e - s),
                **(extra or {}),
            }
        )

    def log_op_cp(
        self,
        *,
        nid: str,
        op: str,
        start: float,
        end: float,
        mode: str = "single",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        s = float(start)
        e = float(end)
        self.op_cp_events.append(
            {
                "phase": self.phase,
                "node_id": nid,
                "op": op,
                "mode": mode,
                "start": s,
                "end": e,
                "duration": float(e - s),
                **(extra or {}),
            }
        )

    def log_comm(
        self,
        *,
        src: str,
        dst: str,
        bytes: int,
        start: float,
        end: float,
        tag: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        def _type(name: Optional[str]) -> str:
            n = (name or "").lower()
            if "npu" in n:
                return "npu"
            if "pim" in n:
                return "pim"
            if "cpu" in n or "host" in n:
                return "cpu"
            return "other"

        s = float(start)
        e = float(end)
        self.comm_events.append(
            {
                "phase": self.phase,
                "src": src,
                "src_type": _type(src),
                "dst": dst,
                "dst_type": _type(dst),
                "bytes": int(bytes) if bytes is not None else 0,
                "start": s,
                "end": e,
                "duration": float(e - s),
                "tag": tag,
                **(extra or {}),
            }
        )

    # -------------------- 1) raw trace export --------------------
    def dump_trace_csv(self, ops_csv_path: Path | str, comms_csv_path: Path | str) -> None:
        """导出原始 ops/comms 两张表"""
        ops_csv_path = Path(ops_csv_path)
        comms_csv_path = Path(comms_csv_path)
        ops_csv_path.parent.mkdir(parents=True, exist_ok=True)
        comms_csv_path.parent.mkdir(parents=True, exist_ok=True)

        ops_fields = [
            "phase", "node_id", "op", "device", "device_type", "mode",
            "start", "end", "duration"
        ]
        with ops_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ops_fields)
            w.writeheader()
            for e in self.op_device_events:
                w.writerow({k: e.get(k) for k in ops_fields})

        comm_fields = [
            "phase", "src", "src_type", "dst", "dst_type",
            "bytes", "start", "end", "duration", "tag"
        ]
        with comms_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=comm_fields)
            w.writeheader()
            for e in self.comm_events:
                w.writerow({k: e.get(k) for k in comm_fields})

    # -------------------- 2) overlap export (stride-aware) --------------------
    @staticmethod
    def _phase_norm(phase: Optional[str]) -> str:
        return str(phase) if (phase is not None and phase != "") else "unknown"

    @staticmethod
    def _dev_cat(device_type: str) -> str:
        dt = (device_type or "").lower()
        if "npu" in dt:
            return "n"
        if "pim" in dt:
            return "p"
        return ""

    @staticmethod
    def _label(n_on: bool, p_on: bool, c_on: bool) -> str:
        if n_on and (not p_on) and (not c_on):
            return "NPU_only"
        if (not n_on) and p_on and (not c_on):
            return "PIM_only"
        if (not n_on) and (not p_on) and c_on:
            return "COMM_only"
        if n_on and p_on and (not c_on):
            return "NPU+PIM"
        if n_on and (not p_on) and c_on:
            return "NPU+COMM"
        if (not n_on) and p_on and c_on:
            return "PIM+COMM"
        if n_on and p_on and c_on:
            return "NPU+PIM+COMM"
        return "IDLE"

    @staticmethod
    def _effective_decode_multiplier(
        decode_stride: int,
        decode_len: Optional[int],
        decode_multiplier: Optional[float],
    ) -> float:

        if decode_multiplier is not None:
            try:
                v = float(decode_multiplier)
                return max(1.0, v)
            except Exception:
                pass

        try:
            s = int(decode_stride)
        except Exception:
            s = 1
        if s <= 1:
            return 1.0

        if decode_len is None:
            return float(s)

        try:
            L = int(decode_len)
        except Exception:
            L = 0
        if L <= 0:
            return float(s)

        # sample indices: t%stride==0 OR t==L-1
        n0 = (max(0, L - 1) // s) + 1           # multiples of stride: 0, s, 2s, ...
        extra_last = 0 if ((L - 1) % s == 0) else 1
        n_samples = n0 + extra_last
        if n_samples <= 0:
            return float(s)

        return float(L) / float(n_samples)

    def dump_csv(
        self,
        segments_csv_path: Path | str,
        summary_csv_path: Path | str,
        *,
        include_idle: bool = True,
        include_all_phase: bool = True,
        normalize_phase_t0: bool = True,
        decode_stride: int = 1,
        decode_len: Optional[int] = None,
        decode_multiplier: Optional[float] = None,
        comm_tags: Optional[set[str]] = None,
    ) -> None:
    
        segments_csv_path = Path(segments_csv_path)
        summary_csv_path = Path(summary_csv_path)
        segments_csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 收集区间：phase -> n/p/c -> [(s,e)...]
        buckets: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(
            lambda: {"n": [], "p": [], "c": []}
        )

        def _add(ph: str, cat: str, s: float, e: float):
            s = float(s); e = float(e)
            if e <= s:
                return
            buckets[ph][cat].append((s, e))
            if include_all_phase:
                buckets["ALL"][cat].append((s, e))

        for e in self.op_device_events:
            ph = self._phase_norm(e.get("phase"))
            cat = self._dev_cat(str(e.get("device_type") or ""))
            if not cat:
                continue
            _add(ph, cat, e.get("start", 0.0), e.get("end", 0.0))

        for e in self.comm_events:
            if comm_tags is not None and e.get("tag") not in comm_tags:
                continue
            ph = self._phase_norm(e.get("phase"))
            _add(ph, "c", e.get("start", 0.0), e.get("end", 0.0))

        dec_mul = self._effective_decode_multiplier(decode_stride, decode_len, decode_multiplier)

        seg_rows: List[Dict[str, Any]] = []
        sum_rows: List[Dict[str, Any]] = []

        # 输出列固定（兼容你现在的 overlap_summary.csv 用法）
        sum_fields = [
            "phase", "t_start", "t_end", "makespan",
            "IDLE",
            "NPU_only", "PIM_only", "COMM_only",
            "NPU+PIM", "NPU+COMM", "PIM+COMM", "NPU+PIM+COMM",
            "ACTIVE_any",
            "decode_multiplier",
        ]

        for ph, kinds in buckets.items():
            events: List[Tuple[float, int, int, int]] = []
            for s, e in kinds["n"]:
                events.append((s, +1, 0, 0))
                events.append((e, -1, 0, 0))
            for s, e in kinds["p"]:
                events.append((s, 0, +1, 0))
                events.append((e, 0, -1, 0))
            for s, e in kinds["c"]:
                events.append((s, 0, 0, +1))
                events.append((e, 0, 0, -1))

            if not events:
                continue

            events.sort(key=lambda x: x[0])
            raw_t0 = events[0][0]
            raw_t1 = events[-1][0]
            base = raw_t0 if normalize_phase_t0 else 0.0

            n_cnt = p_cnt = c_cnt = 0
            prev_t = raw_t0
            i = 0

            totals = defaultdict(float)
            last_emitted = None

            while i < len(events):
                t = events[i][0]
                dt = t - prev_t
                if dt > 0:
                    lab = self._label(n_cnt > 0, p_cnt > 0, c_cnt > 0)
                    totals[lab] += dt

                    if include_idle or lab != "IDLE":
                        s_out = prev_t - base
                        e_out = t - base
                        if (
                            last_emitted is not None
                            and last_emitted["phase"] == ph
                            and last_emitted["label"] == lab
                            and last_emitted["end"] == s_out
                        ):
                            last_emitted["end"] = e_out
                            last_emitted["duration"] += dt
                        else:
                            row = {
                                "phase": ph,
                                "start": s_out,
                                "end": e_out,
                                "duration": dt,
                                "npu_active": int(n_cnt > 0),
                                "pim_active": int(p_cnt > 0),
                                "comm_active": int(c_cnt > 0),
                                "label": lab,
                            }
                            seg_rows.append(row)
                            last_emitted = row

                # apply all deltas at t
                while i < len(events) and events[i][0] == t:
                    n_cnt += events[i][1]
                    p_cnt += events[i][2]
                    c_cnt += events[i][3]
                    i += 1
                prev_t = t

            makespan = raw_t1 - raw_t0

            # summary：decode 行做加权
            scale = dec_mul if ph == "decode" else 1.0

            npu_only = totals.get("NPU_only", 0.0) * scale
            pim_only = totals.get("PIM_only", 0.0) * scale
            comm_only = totals.get("COMM_only", 0.0) * scale
            np = totals.get("NPU+PIM", 0.0) * scale
            nc = totals.get("NPU+COMM", 0.0) * scale
            pc = totals.get("PIM+COMM", 0.0) * scale
            npc = totals.get("NPU+PIM+COMM", 0.0) * scale

            active_any = npu_only + pim_only + comm_only + np + nc + pc + npc

            idle = 0.0
            if include_idle:
                # 如果加权稍微超出 makespan，按比例缩回去，避免负 IDLE
                if makespan > 0 and active_any > makespan and active_any > 0:
                    shrink = makespan / active_any
                    npu_only *= shrink
                    pim_only *= shrink
                    comm_only *= shrink
                    np *= shrink
                    nc *= shrink
                    pc *= shrink
                    npc *= shrink
                    active_any = makespan
                idle = max(0.0, makespan - active_any)

            sum_rows.append(
                {
                    "phase": ph,
                    "t_start": 0.0 if normalize_phase_t0 else raw_t0,
                    "t_end": makespan if normalize_phase_t0 else raw_t1,
                    "makespan": makespan,
                    "IDLE": idle,
                    "NPU_only": npu_only,
                    "PIM_only": pim_only,
                    "COMM_only": comm_only,
                    "NPU+PIM": np,
                    "NPU+COMM": nc,
                    "PIM+COMM": pc,
                    "NPU+PIM+COMM": npc,
                    "ACTIVE_any": active_any,
                    "decode_multiplier": (dec_mul if ph == "decode" else 1.0),
                }
            )

        # 写 segments
        seg_fields = ["phase", "start", "end", "duration", "npu_active", "pim_active", "comm_active", "label"]
        with segments_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=seg_fields)
            w.writeheader()
            for r in seg_rows:
                w.writerow({k: r.get(k) for k in seg_fields})

        # 写 summary
        with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sum_fields)
            w.writeheader()
            for r in sum_rows:
                w.writerow({k: r.get(k) for k in sum_fields})
