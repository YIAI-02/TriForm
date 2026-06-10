# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
from pathlib import Path
import csv
import logging
import time
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class StatsRecorder:
    phase: Optional[str] = None
    op_device_events: List[Dict[str, Any]] = field(default_factory=list)
    op_cp_events: List[Dict[str, Any]] = field(default_factory=list)
    comm_events: List[Dict[str, Any]] = field(default_factory=list)
    _op_seq: int = 0
    _comm_seq: int = 0

    # -------------------- phase control --------------------
    def set_phase(self, phase: Optional[str]) -> None:
        self.phase = phase

    def reset(self) -> None:
        self.op_device_events.clear()
        self.op_cp_events.clear()
        self.comm_events.clear()
        self._op_seq = 0
        self._comm_seq = 0

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
        self._op_seq += 1
        phase = self._phase_norm(self.phase)
        nid_s = str(nid)
        op_s = str(op)
        dev_s = str(device)
        event_id = f"op-{self._op_seq:08d}"
        row = {
            "event_type": "op",
            "event_id": event_id,
            "timeline_id": event_id,
            "phase": phase,
            "node_id": nid_s,
            "op": op_s,
            "device": dev_s,
            "device_type": str(device_type),
            "mode": mode,
            "track": dev_s,
            "lane": dev_s,
            "label": f"{nid_s} | {op_s}",
            "tooltip": f"op={op_s}; node={nid_s}; device={dev_s}; start={s:.9g}s; end={e:.9g}s",
            "start": s,
            "end": e,
            "duration": float(e - s),
        }
        row.update(extra or {})
        self.op_device_events.append(row)

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
                "phase": self._phase_norm(self.phase),
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
            # Treat typical accelerator names as NPU as well (e.g., GPU0).
            if "npu" in n or "gpu" in n or "cuda" in n:
                return "npu"
            if "pim" in n:
                return "pim"
            if "cpu" in n or "host" in n:
                return "cpu"
            return "other"

        s = float(start)
        e = float(end)
        self._comm_seq += 1
        event_id = f"comm-{self._comm_seq:08d}"
        src_s = str(src)
        dst_s = str(dst)
        row = {
            "event_type": "comm",
            "event_id": event_id,
            "comm_id": event_id,
            "timeline_id": event_id,
            "phase": self._phase_norm(self.phase),
            "src": src_s,
            "src_type": _type(src_s),
            "dst": dst_s,
            "dst_type": _type(dst_s),
            "src_device": src_s,
            "src_device_type": _type(src_s),
            "dst_device": dst_s,
            "dst_device_type": _type(dst_s),
            "link": f"{src_s}->{dst_s}",
            "track": f"COMM:{src_s}->{dst_s}",
            "lane": f"COMM:{src_s}->{dst_s}",
            "bytes": int(bytes) if bytes is not None else 0,
            "start": s,
            "end": e,
            "duration": float(e - s),
            "tag": tag,
        }
        row.update(extra or {})
        # Re-assert physical link fields after merging extra metadata so that
        # graph-level fields named src/dst cannot accidentally hide devices.
        row["src"] = src_s
        row["dst"] = dst_s
        row["src_type"] = _type(src_s)
        row["dst_type"] = _type(dst_s)
        row["src_device"] = src_s
        row["dst_device"] = dst_s
        row["src_device_type"] = _type(src_s)
        row["dst_device_type"] = _type(dst_s)
        row["link"] = f"{src_s}->{dst_s}"
        row["track"] = row.get("track") or f"COMM:{src_s}->{dst_s}"
        row["lane"] = row.get("lane") or f"COMM:{src_s}->{dst_s}"
        self.comm_events.append(row)

    # -------------------- 1) trace export --------------------
    @staticmethod
    def _bytes_human(n: Any) -> str:
        try:
            v = float(int(n or 0))
        except Exception:
            return ""
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        i = 0
        while v >= 1024.0 and i < len(units) - 1:
            v /= 1024.0
            i += 1
        if i == 0:
            return f"{int(v)} {units[i]}"
        return f"{v:.3f} {units[i]}"

    @staticmethod
    def _infer_payload(tag: Any) -> str:
        t = str(tag or "").lower()
        if t.startswith("act_") or t in ("act", "activation"):
            return "activation"
        if t.startswith("kv_") or t == "kv":
            return "kv_cache"
        if "weight" in t:
            return "weight"
        if t in ("reduce", "gather", "scatter", "transfer", "allreduce", "all_reduce"):
            return "collective" if t != "transfer" else "activation"
        return "comm"

    @staticmethod
    def _infer_action(tag: Any, payload: str) -> str:
        t = str(tag or "").lower()
        if t.startswith("act_"):
            return t[len("act_"):]
        if t.startswith("kv_"):
            return t[len("kv_"):]
        if t.startswith("weight_"):
            return t[len("weight_"):]
        if t:
            return t
        return payload or "comm"

    @staticmethod
    def _get_first(row: Dict[str, Any], *keys: str) -> Any:
        for k in keys:
            if k in row:
                v = row.get(k)
                if v is not None and str(v) != "":
                    return v
        return None

    @staticmethod
    def _op_sort_key(row: Dict[str, Any]) -> Tuple[float, float, str]:
        def f(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0
        return (f(row.get("start")), f(row.get("end")), str(row.get("event_id", "")))

    def _build_op_index(self, op_rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        idx: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for r in op_rows:
            phase = self._phase_norm(r.get("phase"))
            nid = str(r.get("node_id", "") or "")
            if nid:
                idx[(phase, nid)].append(r)
        for k in list(idx.keys()):
            idx[k].sort(key=self._op_sort_key)
        return idx

    def _lookup_op_for_comm(
        self,
        op_idx: Dict[Tuple[str, str], List[Dict[str, Any]]],
        phase: str,
        node_id: Any,
        *,
        role: str,
        comm_start: float,
        comm_end: float,
    ) -> Optional[Dict[str, Any]]:
        nid = str(node_id or "")
        if not nid:
            return None
        rows = op_idx.get((self._phase_norm(phase), nid)) or []
        if not rows:
            return None
        try:
            cs = float(comm_start)
            ce = float(comm_end)
        except Exception:
            cs = ce = 0.0

        def st(r: Dict[str, Any]) -> float:
            try:
                return float(r.get("start", 0.0))
            except Exception:
                return 0.0

        def en(r: Dict[str, Any]) -> float:
            try:
                return float(r.get("end", 0.0))
            except Exception:
                return 0.0

        # Prefer an op interval overlapping the transfer. This helps weight/KV
        # loads whose time is folded into the consumer op interval.
        overlap = [r for r in rows if st(r) <= ce + 1e-15 and en(r) >= cs - 1e-15]
        if overlap:
            return sorted(overlap, key=lambda r: (abs(st(r) - cs), abs(en(r) - ce)))[0]
        if role == "producer":
            before = [r for r in rows if en(r) <= cs + 1e-15]
            if before:
                return max(before, key=en)
        if role == "consumer":
            after = [r for r in rows if st(r) >= ce - 1e-15]
            if after:
                return min(after, key=st)
        return rows[0]

    def _prepare_ops_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for i, e in enumerate(self.op_device_events, start=1):
            r = dict(e) if isinstance(e, dict) else {}
            r.setdefault("event_type", "op")
            r.setdefault("event_id", f"op-{i:08d}")
            r.setdefault("timeline_id", r.get("event_id"))
            r["phase"] = self._phase_norm(r.get("phase"))
            r.setdefault("track", str(r.get("device", "")))
            r.setdefault("lane", str(r.get("track", r.get("device", ""))))
            if not r.get("label"):
                r["label"] = f"{r.get('node_id','')} | {r.get('op','')}"
            if not r.get("tooltip"):
                r["tooltip"] = (
                    f"op={r.get('op','')}; node={r.get('node_id','')}; "
                    f"device={r.get('device','')}; start={r.get('start','')}s; end={r.get('end','')}s"
                )
            rows.append(r)
        return rows

    def _prepare_comm_rows(self, op_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        op_idx = self._build_op_index(op_rows)
        out: List[Dict[str, Any]] = []
        for i, e in enumerate(self.comm_events, start=1):
            r = dict(e) if isinstance(e, dict) else {}
            r.setdefault("event_type", "comm")
            r.setdefault("event_id", f"comm-{i:08d}")
            r.setdefault("comm_id", r.get("event_id"))
            r.setdefault("timeline_id", r.get("event_id"))
            phase = self._phase_norm(r.get("phase"))
            r["phase"] = phase
            tag = r.get("tag", "")
            payload = str(r.get("payload") or self._infer_payload(tag))
            action = str(r.get("action") or self._infer_action(tag, payload))
            r["payload"] = payload
            r["action"] = action
            try:
                cs = float(r.get("start", 0.0))
            except Exception:
                cs = 0.0
            try:
                ce = float(r.get("end", cs))
            except Exception:
                ce = cs

            producer_node = self._get_first(r, "producer_node_id", "prod_node", "src_node_id")
            consumer_node = self._get_first(r, "consumer_node_id", "cons_node", "dst_node_id")

            # Weight and KV-cache traffic are not edge activations, but they
            # still have a graph-side consumer that can be joined to ops.csv.
            if payload == "weight":
                consumer_node = consumer_node or self._get_first(r, "node_id")
                wid = self._get_first(r, "weight_id", "wid")
                producer_node = producer_node or (f"WEIGHT:{wid}" if wid else "WEIGHT")
            elif payload == "kv_cache":
                if action in ("load", "read", "reload", "local_read", "local_load"):
                    consumer_node = consumer_node or self._get_first(r, "node_id")
                    role = str(self._get_first(r, "kv_role") or "KV")
                    place = str(self._get_first(r, "kv_place", "kv_place_used") or "")
                    shard = self._get_first(r, "kv_seq_shard", "kv_head_start")
                    suffix = f":shard{shard}" if shard is not None else ""
                    producer_node = producer_node or f"KV_CACHE:{place}:{role}{suffix}"
                elif action in ("write", "store"):
                    producer_node = producer_node or self._get_first(r, "node_id")
                    place = str(self._get_first(r, "kv_place", "kv_place_used") or str(r.get("dst", "")))
                    consumer_node = consumer_node or f"KV_CACHE:{place}"
            elif payload in ("activation", "collective"):
                producer_node = producer_node or self._get_first(r, "source_node", "input_node")
                consumer_node = consumer_node or self._get_first(r, "node_id", "comm_node_id")

            prod_op_row = self._lookup_op_for_comm(op_idx, phase, producer_node, role="producer", comm_start=cs, comm_end=ce)
            cons_op_row = self._lookup_op_for_comm(op_idx, phase, consumer_node, role="consumer", comm_start=cs, comm_end=ce)

            producer_op = self._get_first(r, "producer_op", "prod_op", "src_op")
            consumer_op = self._get_first(r, "consumer_op", "cons_op", "dst_op")
            if prod_op_row is not None:
                producer_op = producer_op or prod_op_row.get("op")
                r["producer_device"] = r.get("producer_device") or prod_op_row.get("device")
                r["producer_device_type"] = r.get("producer_device_type") or prod_op_row.get("device_type")
                r["producer_start"] = r.get("producer_start") or prod_op_row.get("start")
                r["producer_end"] = r.get("producer_end") or prod_op_row.get("end")
            if cons_op_row is not None:
                consumer_op = consumer_op or cons_op_row.get("op")
                r["consumer_device"] = r.get("consumer_device") or cons_op_row.get("device")
                r["consumer_device_type"] = r.get("consumer_device_type") or cons_op_row.get("device_type")
                r["consumer_start"] = r.get("consumer_start") or cons_op_row.get("start")
                r["consumer_end"] = r.get("consumer_end") or cons_op_row.get("end")

            if not producer_op:
                if str(producer_node or "").startswith("WEIGHT:"):
                    producer_op = "WEIGHT"
                elif str(producer_node or "").startswith("KV_CACHE:"):
                    producer_op = "KV_CACHE"
                elif producer_node:
                    producer_op = str(producer_node)
            if not consumer_op:
                if str(consumer_node or "").startswith("KV_CACHE:"):
                    consumer_op = "KV_CACHE"
                elif consumer_node:
                    consumer_op = str(consumer_node)

            r["producer_node_id"] = str(producer_node or "")
            r["producer_op"] = str(producer_op or "")
            r["consumer_node_id"] = str(consumer_node or "")
            r["consumer_op"] = str(consumer_op or "")
            # Short aliases that make a CSV concat with ops.csv convenient.
            r["src_node_id"] = r["producer_node_id"]
            r["src_op"] = r["producer_op"]
            r["src_node_device"] = r.get("producer_device", "")
            r["dst_node_id"] = r["consumer_node_id"]
            r["dst_op"] = r["consumer_op"]
            r["dst_node_device"] = r.get("consumer_device", "")

            src_dev = str(r.get("src_device", r.get("src", "")) or "")
            dst_dev = str(r.get("dst_device", r.get("dst", "")) or "")
            r["src_device"] = src_dev
            r["dst_device"] = dst_dev
            r.setdefault("link", f"{src_dev}->{dst_dev}")
            r.setdefault("track", f"COMM:{src_dev}->{dst_dev}")
            r.setdefault("lane", r.get("track"))
            if "bytes_nd" not in r or r.get("bytes_nd") in (None, ""):
                r["bytes_nd"] = r.get("bytes", 0)
            r["bytes_human"] = self._bytes_human(r.get("bytes", 0))

            if r["producer_node_id"] and r["consumer_node_id"]:
                r["edge_id"] = f"{phase}:{r['producer_node_id']}->{r['consumer_node_id']}"
                r["edge"] = f"{r['producer_node_id']} -> {r['consumer_node_id']}"
            else:
                r["edge_id"] = ""
                r["edge"] = ""

            if not r.get("label"):
                left = r["producer_node_id"] or src_dev
                right = r["consumer_node_id"] or dst_dev
                r["label"] = f"{payload}/{action}: {left} -> {right} ({r['bytes_human']})"
            if not r.get("tooltip"):
                r["tooltip"] = (
                    f"payload={payload}; action={action}; tag={tag}; route={r.get('route','')}; hop={r.get('hop','')}; "
                    f"link={src_dev}->{dst_dev}; producer={r['producer_node_id']}[{r['producer_op']}]; "
                    f"consumer={r['consumer_node_id']}[{r['consumer_op']}]; bytes={r.get('bytes',0)}; "
                    f"start={r.get('start','')}s; end={r.get('end','')}s"
                )
            out.append(r)
        return out

    @staticmethod
    def _ordered_fields(base_fields: List[str], rows: List[Dict[str, Any]]) -> List[str]:
        seen = set()
        fields: List[str] = []
        for k in base_fields:
            if k not in seen:
                fields.append(k)
                seen.add(k)
        keys = set()
        for e in rows:
            if isinstance(e, dict):
                keys.update(e.keys())
        for k in sorted(keys):
            if k not in seen:
                fields.append(k)
                seen.add(k)
        return fields

    def dump_trace_csv(self, ops_csv_path: Path | str, comms_csv_path: Path | str) -> None:
        """Export ops and communication traces in a joinable timeline schema.

        ``ops.csv`` and ``comm.csv`` now share common fields
        (event_type/event_id/phase/track/lane/label/start/end/duration), while
        ``comm.csv`` additionally carries producer/consumer node ids and op
        names whenever a transfer is attached to graph operators.
        """
        ops_csv_path = Path(ops_csv_path)
        comms_csv_path = Path(comms_csv_path)
        ops_csv_path.parent.mkdir(parents=True, exist_ok=True)
        comms_csv_path.parent.mkdir(parents=True, exist_ok=True)

        op_rows = self._prepare_ops_rows()
        comm_rows = self._prepare_comm_rows(op_rows)

        base_ops_fields = [
            "event_type", "event_id", "timeline_id", "phase",
            "node_id", "op", "device", "device_type", "mode",
            "track", "lane", "label", "tooltip",
            "start", "end", "duration",
        ]
        ops_fields = self._ordered_fields(base_ops_fields, op_rows)
        with ops_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ops_fields)
            w.writeheader()
            for e in op_rows:
                w.writerow({k: (e.get(k) if isinstance(e, dict) else "") for k in ops_fields})

        base_comm_fields = [
            "event_type", "event_id", "comm_id", "timeline_id", "phase",
            "tag", "payload", "action", "route", "hop",
            "src", "src_type", "dst", "dst_type",
            "src_device", "src_device_type", "dst_device", "dst_device_type",
            "link", "track", "lane", "label", "tooltip",
            "producer_node_id", "producer_op", "producer_device", "producer_device_type", "producer_start", "producer_end",
            "consumer_node_id", "consumer_op", "consumer_device", "consumer_device_type", "consumer_start", "consumer_end",
            "src_node_id", "src_op", "src_node_device", "dst_node_id", "dst_op", "dst_node_device",
            "edge_id", "edge",
            "bytes", "bytes_human", "bytes_nd", "src_fmt", "wire_fmt", "dst_fmt",
            "start", "end", "duration",
        ]
        comm_fields = self._ordered_fields(base_comm_fields, comm_rows)
        with comms_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=comm_fields)
            w.writeheader()
            for e in comm_rows:
                w.writerow({k: (e.get(k) if isinstance(e, dict) else "") for k in comm_fields})

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

        # Collect intervals: phase -> n/p/c -> [(start, end), ...].
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

        # Keep output columns stable for overlap_summary.csv consumers.
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

            # Weight decode rows when building the summary table.
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
                # Clamp slight overflow back to the makespan to avoid negative idle time.
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

        # Write segment rows.
        seg_fields = ["phase", "start", "end", "duration", "npu_active", "pim_active", "comm_active", "label"]
        with segments_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=seg_fields)
            w.writeheader()
            for r in seg_rows:
                w.writerow({k: r.get(k) for k in seg_fields})

        # Write summary rows.
        with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sum_fields)
            w.writeheader()
            for r in sum_rows:
                w.writerow({k: r.get(k) for k in sum_fields})


# ---------------------------------------------------------------------------
#  PIM / Simulation logging
# ---------------------------------------------------------------------------
class SimulationLogger:
    """A lightweight logger for trace-based simulations (e.g., PIM/Ramulator).

    Notes:
      - Writes to both python logging (debug) and a dedicated log file.
      - Tracks unique simulated op configurations for quick coverage summaries.
    """

    def __init__(self, log_file: Optional[Path | str] = None):
        self.log_file: Path = Path(log_file) if (log_file is not None) else Path("pim_simulation.log")
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.simulated_ops: Dict[str, set[Tuple[int, int, int, int, int]]] = defaultdict(set)
        self.lock = Lock()

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_file.open("w", encoding="utf-8")

        self._log("=" * 80)
        self._log(f"PIM Simulation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 80 + "\n")

    def _log(self, message: str) -> None:
        with self.lock:
            logger.debug(str(message))
            self._log_handle.write(str(message) + "\n")
            self._log_handle.flush()

    def start_simulation(self) -> None:
        self.start_time = time.time()
        self._log("\n" + "=" * 80)
        self._log(f"Simulation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        self._log("=" * 80 + "\n")

    def end_simulation(self) -> None:
        self.end_time = time.time()
        elapsed = (self.end_time - self.start_time) if (self.start_time is not None) else 0.0
        self._log("\n" + "=" * 80)
        self._log(f"Simulation Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        self._log(f"Total Simulation Time: {elapsed:.3f} seconds ({elapsed / 60.0:.2f} minutes)")
        self._log("=" * 80 + "\n")
        self._print_statistics()

    def record_simulation(
        self,
        op: str,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        seqlen: Optional[int],
    ) -> None:
        with self.lock:
            cfg = (int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim), int(seqlen or 0))
            self.simulated_ops[str(op)].add(cfg)

    def _print_statistics(self) -> None:
        self._log("\n" + "=" * 80)
        self._log("Simulated Operations Summary")
        self._log("=" * 80)

        total_unique = sum(len(v) for v in self.simulated_ops.values())
        self._log(f"\nTotal unique operations simulated: {total_unique}")
        self._log(f"Total operation types: {len(self.simulated_ops)}\n")

        for op in sorted(self.simulated_ops.keys()):
            configs = self.simulated_ops[op]
            self._log(f"\n{op.upper()}:")
            self._log(f"  - Unique configurations: {len(configs)}")
            for cfg in sorted(configs):
                dim, n_heads, n_kv_heads, ffn_dim, seqlen = cfg
                self._log(
                    "    * "
                    f"dim={dim}, heads={n_heads}, kv_heads={n_kv_heads}, "
                    f"ffn_dim={ffn_dim}, seqlen={(seqlen if seqlen > 0 else 'None')}"
                )

        self._log("\n" + "=" * 80 + "\n")

    def close(self) -> None:
        if getattr(self, "_log_handle", None) and (not self._log_handle.closed):
            self._log_handle.close()


_sim_logger: Optional[SimulationLogger] = None

def get_simulation_logger(log_file: Optional[Path | str] = None) -> SimulationLogger:
    """Get a process-wide singleton SimulationLogger.

    Keeping this here (instead of cost_model) prevents mixing *instrumentation*
    with *latency/cost modeling logic*.
    """
    global _sim_logger
    if _sim_logger is None:
        _sim_logger = SimulationLogger(log_file)
    return _sim_logger


def reset_simulation_logger() -> None:
    """Close and reset the global SimulationLogger (use between runs/tests)."""
    global _sim_logger
    if _sim_logger is not None:
        try:
            _sim_logger.close()
        finally:
            _sim_logger = None
