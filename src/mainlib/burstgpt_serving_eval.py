from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed
from .shared import *
from .evaluate import _eval_one_baseline, _run_strategy_once
from .storage import _artifact_tag_token


@dataclass
class BurstGPTRequest:
    request_id: str
    arrival_s: float
    input_len: int
    output_len: int
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "request_id": self.request_id,
            "arrival_s": float(self.arrival_s),
            "input_len": int(self.input_len),
            "output_len": int(self.output_len),
        }
        if self.model:
            out["model"] = self.model
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


@dataclass(frozen=True)
class ProfileKey:
    policy: str
    batch: int
    prefill_len: int
    decode_len: int
    decode_horizon_len: int


@dataclass
class ShapeProfile:
    policy: str
    batch: int
    prefill_len: int
    decode_len: int
    decode_horizon_len: int
    prefill_time_s: float
    decode_step_times_s: List[float]
    decode_time_s: float
    pim_strategy: Optional[str] = None

    def first_decode_step_s(self) -> float:
        if self.decode_step_times_s:
            return float(self.decode_step_times_s[0])
        return float(self.decode_time_s) / float(max(1, self.decode_len))

    def decode_prefix_s(self, n_tokens: int) -> float:
        n = max(0, int(n_tokens))
        if n <= 0:
            return 0.0
        steps = self.decode_step_times_s
        if not steps:
            return float(self.decode_time_s) * float(n) / float(max(1, self.decode_len))
        if len(steps) >= n:
            return float(sum(steps[:n]))
        tail = float(steps[-1])
        return float(sum(steps) + tail * (n - len(steps)))

    def tbt_after_first(self, n_tokens: int) -> List[float]:
        n = max(0, int(n_tokens))
        if n <= 1:
            return []
        steps = self.decode_step_times_s
        if not steps:
            avg = float(self.decode_time_s) / float(max(1, self.decode_len))
            return [avg for _ in range(n - 1)]
        out: List[float] = []
        for i in range(1, n):
            out.append(float(steps[i] if i < len(steps) else steps[-1]))
        return out


# ---------------------------------------------------------------------------
# BurstGPT loader
# ---------------------------------------------------------------------------

def _norm_key(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").strip().lower())


def _lookup(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    norm = {_norm_key(k): v for k, v in row.items()}
    for name in names:
        if name in row:
            return row[name]
        nk = _norm_key(name)
        if nk in norm:
            return norm[nk]
    return None


def _safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    try:
        s = str(v).strip().replace(",", "")
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        s = str(v).strip().replace(",", "")
        if not s:
            return default
        x = float(s)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _model_filter_ok(model: str, model_filter: Optional[str]) -> bool:
    raw = str(model_filter or "").strip()
    if not raw:
        return True
    allowed = {x.strip().lower() for x in raw.replace(",", " ").split() if x.strip()}
    return str(model or "").strip().lower() in allowed


def load_burstgpt_csv(
    path: str | Path,
    *,
    max_requests: Optional[int] = None,
    skip_zero_output: bool = True,
    model_filter: Optional[str] = None,
    arrival_time_scale: float = 1.0,
    min_input_len: int = 1,
    min_output_len: int = 1,
    max_input_len: Optional[int] = None,
    max_output_len: Optional[int] = None,
) -> List[BurstGPTRequest]:
    """Read BurstGPT CSV columns: Timestamp, Request tokens, Response tokens."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"BurstGPT CSV not found: {p}")

    reqs: List[BurstGPTRequest] = []
    skipped = 0
    t0: Optional[float] = None
    scale = float(arrival_time_scale or 1.0)
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0

    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            ts = _safe_float(_lookup(row, ["Timestamp", "time", "arrival_s", "arrival", "submit_time"]))
            inp = _safe_int(_lookup(row, ["Request tokens", "request_tokens", "prompt_tokens", "input_len", "input_tokens"]))
            out = _safe_int(_lookup(row, ["Response tokens", "response_tokens", "completion_tokens", "output_len", "output_tokens"]))
            model = str(_lookup(row, ["Model", "model"]) or "")
            if ts is None or inp is None or out is None:
                skipped += 1
                continue
            if not _model_filter_ok(model, model_filter):
                skipped += 1
                continue
            if skip_zero_output and int(out) <= 0:
                skipped += 1
                continue

            raw_inp, raw_out = int(inp), int(out)
            inp = max(int(min_input_len), raw_inp)
            out = max(int(min_output_len), raw_out)
            clipped = False
            if max_input_len is not None and int(max_input_len) > 0 and inp > int(max_input_len):
                inp = int(max_input_len)
                clipped = True
            if max_output_len is not None and int(max_output_len) > 0 and out > int(max_output_len):
                out = int(max_output_len)
                clipped = True

            if t0 is None:
                t0 = float(ts)
            arrival = max(0.0, (float(ts) - float(t0)) * scale)
            metadata: Dict[str, Any] = {"source_row": row_idx}
            if clipped:
                metadata["clipped"] = True
                metadata["raw_input_len"] = raw_inp
                metadata["raw_output_len"] = raw_out

            reqs.append(BurstGPTRequest(
                request_id=f"burstgpt-{row_idx}",
                arrival_s=arrival,
                input_len=int(inp),
                output_len=int(out),
                model=model,
                metadata=metadata,
            ))
            if max_requests is not None and int(max_requests) > 0 and len(reqs) >= int(max_requests):
                break

    if not reqs:
        raise ValueError(
            f"No usable BurstGPT requests loaded from {p}. Skipped rows={skipped}. "
            "Check the CSV header and whether Response tokens are all zero after filtering."
        )
    reqs.sort(key=lambda r: (r.arrival_s, r.request_id))
    return reqs


# ---------------------------------------------------------------------------
# Statistics and bucketing
# ---------------------------------------------------------------------------

def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    qf = float(q)
    if qf > 1.0:
        qf /= 100.0
    qf = min(1.0, max(0.0, qf))
    if len(vals) == 1:
        return float(vals[0])
    pos = qf * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - pos) + vals[hi] * (pos - lo))


def _stats(values: Iterable[float]) -> Dict[str, Any]:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return {"count": 0, "mean_s": None, "p50_s": None, "p90_s": None, "p95_s": None, "p99_s": None}
    return {
        "count": len(vals),
        "mean_s": float(sum(vals) / len(vals)),
        "p50_s": _percentile(vals, 0.50),
        "p90_s": _percentile(vals, 0.90),
        "p95_s": _percentile(vals, 0.95),
        "p99_s": _percentile(vals, 0.99),
        "min_s": float(vals[0]),
        "max_s": float(vals[-1]),
    }


def _length_stats(values: Iterable[int]) -> Dict[str, Any]:
    vals = sorted(int(v) for v in values)
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "mean": float(sum(vals) / len(vals)),
        "p50": _percentile([float(v) for v in vals], 0.50),
        "p90": _percentile([float(v) for v in vals], 0.90),
        "p95": _percentile([float(v) for v in vals], 0.95),
        "p99": _percentile([float(v) for v in vals], 0.99),
        "min": int(vals[0]),
        "max": int(vals[-1]),
    }


def _ceil_bucket(x: int, bucket: int) -> int:
    x = max(1, int(x))
    b = max(1, int(bucket or 1))
    return int(math.ceil(x / b) * b)


def _policy_is_baseline(policy: str) -> bool:
    return _normalize_baseline_name(policy) in _BASELINE_TOKENS


def _display_label(policy: str) -> str:
    tok = _normalize_baseline_name(policy) if _policy_is_baseline(policy) else _normalize_algo_name(policy)
    return _policy_label(tok)


def _extract_decode_steps(decode_steps: Any, decode_len: int, decode_time_s: float) -> List[float]:
    out: List[float] = []
    if isinstance(decode_steps, list):
        for step in decode_steps:
            if isinstance(step, Mapping):
                try:
                    out.append(float(step.get("step_time", 0.0) or 0.0))
                except Exception:
                    pass
    if not out and int(decode_len) > 0:
        out = [float(decode_time_s) / float(max(1, int(decode_len))) for _ in range(int(decode_len))]
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# Serving replay
# ---------------------------------------------------------------------------

class BurstGPTServingEvaluator:
    def __init__(self, cfg: Mapping[str, Any], requests: Sequence[BurstGPTRequest], result_dir: str | Path):
        self.cfg = dict(cfg)
        self.requests = sorted(list(requests), key=lambda r: (r.arrival_s, r.request_id))
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.profile_cache: Dict[ProfileKey, ShapeProfile] = {}

        self.max_batch = max(1, int(self.cfg.get("serving_batch_size", 1) or 1))
        self.batch_timeout_s = max(0.0, float(self.cfg.get("batch_timeout_s", 0.0) or 0.0))
        self.prompt_bucket_size = max(1, int(self.cfg.get("prompt_bucket_size", 128) or 128))
        self.output_bucket_size = max(1, int(self.cfg.get("output_bucket_size", 16) or 16))
        self.decode_sample_stride = max(1, int(self.cfg.get("decode_sample_stride", 8) or 8))
        self.decode_plan_refresh_stride = max(0, int(self.cfg.get("decode_plan_refresh_stride", 8) or 8))

        self.output_horizon = str(self.cfg.get("output_horizon", self.cfg.get("output_horizon_policy", "p90")) or "p90").lower()
        self.output_horizon_fixed = int(self.cfg.get("output_horizon_fixed", 0) or 0)
        self.output_horizon_p90 = max(1, int(math.ceil(float(_percentile([float(r.output_len) for r in self.requests], 0.90) or 1))))

    def _horizon_for_shape(self, actual_decode_len: int) -> int:
        actual = max(1, int(actual_decode_len))
        if self.output_horizon in {"oracle", "actual"}:
            return actual
        if self.output_horizon in {"fixed", "constant"}:
            return max(1, int(self.output_horizon_fixed or self.output_horizon_p90))
        # Default: p90 of BurstGPT output lengths. This avoids giving Bifocal
        # per-request future knowledge while still supplying a workload-level hint.
        return self.output_horizon_p90

    def _profile_key(self, policy: str, group: Sequence[BurstGPTRequest]) -> ProfileKey:
        batch = max(1, len(group))
        prefill_len = _ceil_bucket(max(int(r.input_len) for r in group), self.prompt_bucket_size)
        decode_len = _ceil_bucket(max(int(r.output_len) for r in group), self.output_bucket_size)
        horizon = _ceil_bucket(self._horizon_for_shape(decode_len), self.output_bucket_size)
        return ProfileKey(str(policy), batch, prefill_len, decode_len, horizon)

    def _run_shape_profile(self, key: ProfileKey) -> ShapeProfile:
        cached = self.profile_cache.get(key)
        if cached is not None:
            return cached

        print(
            f"[burstgpt-profile] NEW policy={key.policy} batch={key.batch} "
            f"prefill={key.prefill_len} decode={key.decode_len} "
            f"horizon={key.decode_horizon_len} cached_profiles={len(self.profile_cache)}",
            flush=True,
        )

        cfg_run = dict(self.cfg)
        cfg_run["batch"] = int(key.batch)
        cfg_run["prefill_len"] = int(key.prefill_len)
        cfg_run["decode_len"] = int(key.decode_len)
        cfg_run["decode_horizon_len"] = int(key.decode_horizon_len)
        cfg_run["decode_sample_stride"] = int(self.decode_sample_stride)
        cfg_run["decode_plan_refresh_stride"] = int(self.decode_plan_refresh_stride)

        tag = f"burst_{_artifact_tag_token(key.policy)}_b{key.batch}_p{key.prefill_len}_d{key.decode_len}_h{key.decode_horizon_len}"
        policy_dir = self.result_dir / "profiles" / _artifact_tag_token(key.policy)
        policy_dir.mkdir(parents=True, exist_ok=True)
        cfg_run["result_dir"] = str(policy_dir)
        cfg_run["simulation_log_file"] = str(policy_dir / f"pim_sim_{tag}.txt")

        if _policy_is_baseline(key.policy):
            result = _eval_one_baseline(cfg_run, key.policy, artifact_tag=tag)
        else:
            result = _run_strategy_once(key.policy, cfg_run, artifact_tag=tag)

        decode_time_s = float(result.get("decode_time_s", 0.0) or 0.0)
        prof = ShapeProfile(
            policy=key.policy,
            batch=key.batch,
            prefill_len=key.prefill_len,
            decode_len=key.decode_len,
            decode_horizon_len=key.decode_horizon_len,
            prefill_time_s=float(result.get("prefill_time_s", 0.0) or 0.0),
            decode_step_times_s=_extract_decode_steps(result.get("decode_steps"), key.decode_len, decode_time_s),
            decode_time_s=decode_time_s,
            pim_strategy=result.get("pim_strategy"),
        )
        self.profile_cache[key] = prof
        print(
            f"[burstgpt-profile] DONE policy={key.policy} batch={key.batch} "
            f"prefill={key.prefill_len} decode={key.decode_len} "
            f"prefill_time={prof.prefill_time_s:.6f}s decode_time={prof.decode_time_s:.6f}s",
            flush=True,
        )
        return prof

    def _next_group(self, idx: int, server_free_s: float) -> Tuple[List[BurstGPTRequest], int, float]:
        n = len(self.requests)
        if idx >= n:
            return [], idx, server_free_s

        first = self.requests[idx]
        start = max(float(server_free_s), float(first.arrival_s))

        # Optional, conservative micro-batching: if requests arrive within a short
        # timeout of the oldest request, wait only until the last included arrival.
        if self.max_batch > 1 and self.batch_timeout_s > 0.0:
            deadline = float(first.arrival_s) + self.batch_timeout_s
            j = idx
            while j < n and (j - idx) < self.max_batch and float(self.requests[j].arrival_s) <= deadline:
                start = max(start, float(self.requests[j].arrival_s))
                j += 1

        group: List[BurstGPTRequest] = []
        j = idx
        while j < n and len(group) < self.max_batch and float(self.requests[j].arrival_s) <= start + 1e-12:
            group.append(self.requests[j])
            j += 1
        if not group:
            group = [first]
            j = idx + 1
            start = max(start, float(first.arrival_s))
        return group, j, start

    def evaluate_policy(self, policy: str) -> Dict[str, Any]:
        policy_token = _normalize_baseline_name(policy) if _policy_is_baseline(policy) else _normalize_algo_name(policy)
        start_profile_count = len(self.profile_cache)
        records: List[Dict[str, Any]] = []
        token_rows: List[Dict[str, Any]] = []
        batch_rows: List[Dict[str, Any]] = []

        idx = 0
        server_free = 0.0
        while idx < len(self.requests):
            group, next_idx, batch_start = self._next_group(idx, server_free)
            if not group:
                break
            key = self._profile_key(policy_token, group)
            prof = self._run_shape_profile(key)
            max_out = max(int(r.output_len) for r in group)
            service_s = float(prof.prefill_time_s + prof.decode_prefix_s(max_out))
            batch_finish = float(batch_start + service_s)

            batch_rows.append({
                "policy": _display_label(policy_token),
                "batch_id": len(batch_rows),
                "batch_start_s": batch_start,
                "batch_finish_s": batch_finish,
                "batch_size": len(group),
                "prefill_len_bucket": key.prefill_len,
                "decode_len_bucket": key.decode_len,
                "decode_horizon_len": key.decode_horizon_len,
                "service_time_s": service_s,
                "pim_strategy": prof.pim_strategy,
            })

            for req in group:
                wait_s = max(0.0, float(batch_start - req.arrival_s))
                first_step = prof.first_decode_step_s() if req.output_len > 0 else 0.0
                decode_until_req = prof.decode_prefix_s(req.output_len)
                ttft_s = float(wait_s + prof.prefill_time_s + first_step)
                e2e_s = float(wait_s + prof.prefill_time_s + decode_until_req)
                gaps = prof.tbt_after_first(req.output_len)
                tbt_mean = float(sum(gaps) / len(gaps)) if gaps else None
                records.append({
                    "policy": _display_label(policy_token),
                    "request_id": req.request_id,
                    "arrival_s": float(req.arrival_s),
                    "start_s": float(batch_start),
                    "finish_s": float(batch_start + prof.prefill_time_s + decode_until_req),
                    "queue_wait_s": wait_s,
                    "ttft_s": ttft_s,
                    "tbt_mean_s": tbt_mean,
                    "e2e_latency_s": e2e_s,
                    "input_len": int(req.input_len),
                    "output_len": int(req.output_len),
                    "model": req.model,
                    "batch_size": len(group),
                    "prefill_len_bucket": key.prefill_len,
                    "decode_len_bucket": key.decode_len,
                    "decode_horizon_len": key.decode_horizon_len,
                })
                for token_idx, gap in enumerate(gaps, start=2):
                    token_rows.append({
                        "policy": _display_label(policy_token),
                        "request_id": req.request_id,
                        "token_idx": token_idx,
                        "tbt_s": float(gap),
                    })

            server_free = batch_finish
            idx = next_idx

        first_arrival = min((r.arrival_s for r in self.requests), default=0.0)
        last_finish = max((r["finish_s"] for r in records), default=first_arrival)
        makespan = max(0.0, float(last_finish - first_arrival))
        summary = {
            "policy": _display_label(policy_token),
            "request_count": len(records),
            "batch_count": len(batch_rows),
            "new_shape_profiles": len(self.profile_cache) - start_profile_count,
            "serving_batch_size": self.max_batch,
            "batch_timeout_s": self.batch_timeout_s,
            "prompt_bucket_size": self.prompt_bucket_size,
            "output_bucket_size": self.output_bucket_size,
            "output_horizon": self.output_horizon,
            "output_horizon_p90": self.output_horizon_p90,
            "makespan_s": makespan,
            "throughput_req_s": (float(len(records) / makespan) if makespan > 0 else None),
            "ttft": _stats(r["ttft_s"] for r in records),
            "tbt_token": _stats(r["tbt_s"] for r in token_rows),
            "tbt_request_mean": _stats(r["tbt_mean_s"] for r in records if r.get("tbt_mean_s") is not None),
            "e2e_latency": _stats(r["e2e_latency_s"] for r in records),
            "queue_wait": _stats(r["queue_wait_s"] for r in records),
        }
        return {"summary": summary, "records": records, "token_rows": token_rows, "batch_rows": batch_rows}


def _trace_summary(requests: Sequence[BurstGPTRequest]) -> Dict[str, Any]:
    arrivals = sorted(float(r.arrival_s) for r in requests)
    inter = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
    span = (arrivals[-1] - arrivals[0]) if len(arrivals) >= 2 else 0.0
    return {
        "request_count": len(requests),
        "arrival_span_s": float(span),
        "effective_request_rate_req_s": (float(len(requests) / span) if span > 0 else None),
        "input_len": _length_stats(r.input_len for r in requests),
        "output_len": _length_stats(r.output_len for r in requests),
        "interarrival_s": _stats(inter),
    }

def _policy_slug(policy: str) -> str:
    raw = str(policy or "policy")
    slug = re.sub(r"[^A-Za-z0-9_.+-]+", "_", raw).strip("_")
    return slug or "policy"


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _eval_policy_worker(args: Tuple[Dict[str, Any], List[BurstGPTRequest], str, str]) -> Dict[str, Any]:
    """Evaluate one policy in an independent process.

    Each worker has its own evaluator and profile cache.  This avoids sharing
    mutable scheduler state across processes and prevents output-file races by
    writing policy-specific CSV files.
    """
    cfg, requests, out_dir_s, policy = args
    out_dir = Path(out_dir_s)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _policy_slug(policy)

    print(f"[burstgpt-worker] START policy={_display_policy_name(policy)} pid={os.getpid()}", flush=True)

    # Each process must construct its own evaluator.  Do not try to reuse the
    # parent process evaluator/profile_cache: scheduler state and cost-model
    # state are mutable and are not safe to share across processes.
    evaluator = BurstGPTServingEvaluator(cfg, requests, out_dir)
    res = evaluator.evaluate_policy(policy)

    records_path = out_dir / f"burstgpt_per_request_{slug}.csv"
    token_path = out_dir / f"burstgpt_token_tbt_{slug}.csv"
    batch_path = out_dir / f"burstgpt_batches_{slug}.csv"
    summary_path = out_dir / f"burstgpt_policy_summary_{slug}.json"

    _write_csv(records_path, res["records"])
    _write_csv(token_path, res["token_rows"])
    _write_csv(batch_path, res["batch_rows"])
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(res["summary"], f, ensure_ascii=False, indent=2)

    print(f"[burstgpt-worker] DONE policy={_display_policy_name(policy)} pid={os.getpid()}", flush=True)
    return {
        "policy": policy,
        "summary": res["summary"],
        "records_path": str(records_path),
        "token_path": str(token_path),
        "batch_path": str(batch_path),
        "summary_path": str(summary_path),
    }



def evaluate_burstgpt_suite(
    cfg: Mapping[str, Any],
    *,
    algos: Sequence[str],
    baselines: Sequence[str],
    result_dir: str | Path,
    combined_out: Optional[str | Path] = None,
) -> Dict[str, Any]:
    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = cfg.get("burstgpt_csv") or cfg.get("workload_path") or cfg.get("request_trace_path")
    if not csv_path:
        raise ValueError("Missing BurstGPT CSV path. Use --burstgpt_csv /path/to/BurstGPT_without_fails_1.csv")

    requests = load_burstgpt_csv(
        csv_path,
        max_requests=(None if cfg.get("num_requests", None) in (None, "", 0) else int(cfg.get("num_requests"))),
        skip_zero_output=bool(cfg.get("skip_zero_output", True)),
        model_filter=cfg.get("burstgpt_model_filter", None),
        arrival_time_scale=float(cfg.get("arrival_time_scale", 1.0) or 1.0),
        min_input_len=int(cfg.get("min_input_len", 1) or 1),
        min_output_len=int(cfg.get("min_output_len", 1) or 1),
        max_input_len=(None if cfg.get("max_input_len", None) in (None, "", 0) else int(cfg.get("max_input_len"))),
        max_output_len=(None if cfg.get("max_output_len", None) in (None, "", 0) else int(cfg.get("max_output_len"))),
    )

    trace_info = _trace_summary(requests)
    print(
        "[burstgpt-evaluate] loaded "
        f"{len(requests)} requests; "
        f"input_len p50/p90/max={trace_info['input_len'].get('p50')}/"
        f"{trace_info['input_len'].get('p90')}/{trace_info['input_len'].get('max')}; "
        f"output_len p50/p90/max={trace_info['output_len'].get('p50')}/"
        f"{trace_info['output_len'].get('p90')}/{trace_info['output_len'].get('max')}; "
        f"arrival_span_s={trace_info.get('arrival_span_s')}",
        flush=True,
    )

    req_path = out_dir / "burstgpt_requests_used.jsonl"
    with req_path.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req.as_dict(), ensure_ascii=False) + "\n")

    policies: List[str] = []
    for b in baselines:
        tok = _normalize_baseline_name(b)
        if tok and tok not in policies:
            policies.append(tok)
    for a in algos:
        tok = _normalize_algo_name(a)
        if tok and tok not in policies:
            policies.append(tok)
    if not policies:
        policies = ["PD", "Bifocal"]

    # Read policy-level parallelism from JSON.  This is intentionally configured
    # in the JSON rather than in Slurm/CLI, so experiments remain reproducible.
    parallel_workers = int(cfg.get("policy_parallel_workers", cfg.get("parallel_workers", 1)) or 1)
    parallel_workers = max(1, min(parallel_workers, len(policies), os.cpu_count() or 1))
    print(
        f"[burstgpt-evaluate] policy_parallel_workers={parallel_workers} "
        f"json_value={cfg.get('policy_parallel_workers', cfg.get('parallel_workers', None))} "
        f"policies={len(policies)} cpu_count={os.cpu_count()}",
        flush=True,
    )

    summaries: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []
    all_tokens: List[Dict[str, Any]] = []
    all_batches: List[Dict[str, Any]] = []

    if parallel_workers <= 1 or len(policies) <= 1:
        print(f"[burstgpt-evaluate] policy evaluation mode=serial policies={len(policies)}", flush=True)
        evaluator = BurstGPTServingEvaluator(cfg, requests, out_dir)
        for p in policies:
            print(f"[burstgpt-evaluate] policy={_display_policy_name(p)} requests={len(requests)}", flush=True)
            res = evaluator.evaluate_policy(p)
            summaries.append(res["summary"])
            all_records.extend(res["records"])
            all_tokens.extend(res["token_rows"])
            all_batches.extend(res["batch_rows"])
    else:
        print(
            f"[burstgpt-evaluate] policy evaluation mode=parallel "
            f"workers={parallel_workers} policies={len(policies)}",
            flush=True,
        )
        futures = []
        with ProcessPoolExecutor(max_workers=parallel_workers) as pool:
            for p in policies:
                futures.append(pool.submit(_eval_policy_worker, (dict(cfg), list(requests), str(out_dir), p)))
            policy_results: List[Dict[str, Any]] = []
            for fut in as_completed(futures):
                policy_results.append(fut.result())

        # Preserve the input policy order in the final summary/table.
        by_policy = {str(item["policy"]): item for item in policy_results}
        for p in policies:
            item = by_policy[str(p)]
            summaries.append(item["summary"])
            all_records.extend(_read_csv_rows(Path(item["records_path"])))
            all_tokens.extend(_read_csv_rows(Path(item["token_path"])))
            all_batches.extend(_read_csv_rows(Path(item["batch_path"])))
 
    _write_csv(out_dir / "burstgpt_per_request.csv", all_records)
    _write_csv(out_dir / "burstgpt_token_tbt.csv", all_tokens)
    _write_csv(out_dir / "burstgpt_batches.csv", all_batches)

    payload = {
        "config": dict(cfg),
        "burstgpt_csv": str(csv_path),
        "requests_used_jsonl": str(req_path),
        "trace_summary": _trace_summary(requests),
        "summaries": summaries,
    }
    out_json = Path(combined_out) if combined_out else (out_dir / "burstgpt_serving_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n=== BurstGPT Serving Comparison ===")
    header = (
        f"{'Policy':<18} {'Req':>6} {'Thr(req/s)':>10} "
        f"{'TTFT p50':>10} {'TTFT p90':>10} {'TBT p50':>10} {'TBT p90':>10} "
        f"{'E2E p50':>10} {'E2E p90':>10}"
    )
    print(header)
    print("-" * len(header))

    def _get(s: Mapping[str, Any], section: str, key: str) -> float:
        val = (s.get(section) or {}).get(key)
        return float(val) if val is not None else float("nan")

    for s in summaries:
        thr = s.get("throughput_req_s")
        print(
            f"{s.get('policy',''):<18} {int(s.get('request_count', 0)):>6} "
            f"{(float(thr) if thr is not None else float('nan')):>10.4f} "
            f"{_get(s, 'ttft', 'p50_s'):>10.4f} {_get(s, 'ttft', 'p90_s'):>10.4f} "
            f"{_get(s, 'tbt_token', 'p50_s'):>10.4f} {_get(s, 'tbt_token', 'p90_s'):>10.4f} "
            f"{_get(s, 'e2e_latency', 'p50_s'):>10.4f} {_get(s, 'e2e_latency', 'p90_s'):>10.4f}"
        )
    print(f"[burstgpt-evaluate] summary saved to: {out_json}", flush=True)
    return payload


__all__ = [
    "BurstGPTRequest",
    "load_burstgpt_csv",
    "BurstGPTServingEvaluator",
    "evaluate_burstgpt_suite",
]
