#!/usr/bin/env python3
"""Run reviewer-facing Bifocal component ablations.

The script uses the normal ``src/main.py evaluate`` path and toggles Bifocal
components through runtime CLI overrides.  It is intentionally small-sweep: the
built-in workload list mirrors representative Section-6 configurations instead
of enumerating all 16 component subsets.

Typical use:

  python3 commands/run_bifocal_component_ablation.py \
    --config src/examples/evaluate_len_sweep_config_npu.json \
    --hardware-json src/examples/hardware_1npu_2aim.json \
    --best-json output/sweep_bifocal_all/best_result.json \
    --variant-suite minimal \
    --outdir output/bifocal_component_ablation \
    --resume

For the unknown-output-length reviewer question, compare oracle and fixed
planning horizons:

  python3 commands/run_bifocal_component_ablation.py \
    --horizon-suite oracle,fixed --fixed-horizon 256
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CONFIG = SRC_DIR / "examples" / "evaluate_len_sweep_config_npu.json"
DEFAULT_HW = SRC_DIR / "examples" / "hardware_1npu_2aim.json"

# Representative Section-6-style points: Fig. 7 uses HP32, Llama-7B b16 and
# Qwen-1.8B b8, prefill in {128,1024}, decode in {128,512,1024}.
DEFAULT_WORKLOADS = [
    "llama:7b:fp16:16:128:128",
    "llama:7b:fp16:16:128:512",
    "llama:7b:fp16:16:128:1024",
    "llama:7b:fp16:16:1024:128",
    "llama:7b:fp16:16:1024:512",
    "llama:7b:fp16:16:1024:1024",
    "qwen:1.8b:fp16:8:128:128",
    "qwen:1.8b:fp16:8:128:512",
    "qwen:1.8b:fp16:8:128:1024",
    "qwen:1.8b:fp16:8:1024:128",
    "qwen:1.8b:fp16:8:1024:512",
    "qwen:1.8b:fp16:8:1024:1024",
]
QUICK_WORKLOADS = [
    "llama:7b:fp16:16:128:512",
    "llama:7b:fp16:16:1024:512",
    "qwen:1.8b:fp16:8:128:512",
    "qwen:1.8b:fp16:8:1024:512",
]

# Each variant is a small number of one-factor removals or controlled additions.
# EFT is not a toggle: it is the base score term.  The remaining flags add or
# remove lookahead, phase-reuse, and token-amortization terms.
VARIANTS: Dict[str, Dict[str, Any]] = {
    "EFT-only": {
        "lookahead": False,
        "phase": False,
        "token": False,
        "description": "Immediate earliest-finish score only.",
    },
    "+Lookahead": {
        "lookahead": True,
        "phase": False,
        "token": False,
        "description": "EFT plus DAG-window lookahead only.",
    },
    "+Lookahead+Phase": {
        "lookahead": True,
        "phase": True,
        "token": False,
        "description": "Adds phase-reuse bias to EFT+lookahead.",
    },
    "+Lookahead+Token": {
        "lookahead": True,
        "phase": False,
        "token": True,
        "description": "Adds token-amortization bias to EFT+lookahead.",
    },
    "Full": {
        "lookahead": True,
        "phase": True,
        "token": True,
        "description": "Full Bifocal score.",
    },
    "Full-w/o-Lookahead": {
        "lookahead": False,
        "phase": True,
        "token": True,
        "description": "Full score with DAG-window lookahead disabled.",
    },
    "Full-w/o-Phase": {
        "lookahead": True,
        "phase": False,
        "token": True,
        "description": "Full score with phase-reuse bias disabled.",
    },
    "Full-w/o-Token": {
        "lookahead": True,
        "phase": True,
        "token": False,
        "description": "Full score with token-amortization bias disabled.",
    },
}
VARIANT_SUITES = {
    "minimal": ["EFT-only", "Full-w/o-Lookahead", "Full-w/o-Phase", "Full-w/o-Token", "Full"],
    "incremental": ["EFT-only", "+Lookahead", "+Lookahead+Phase", "+Lookahead+Token", "Full"],
    "both": [
        "EFT-only",
        "+Lookahead",
        "+Lookahead+Phase",
        "+Lookahead+Token",
        "Full-w/o-Lookahead",
        "Full-w/o-Phase",
        "Full-w/o-Token",
        "Full",
    ],
}

RESULT_FIELDS = [
    "timestamp",
    "variant",
    "variant_description",
    "model_family",
    "model_variant",
    "dtype",
    "batch",
    "prefill_len",
    "decode_len",
    "horizon_mode",
    "decode_horizon_len",
    "hardware_json",
    "h",
    "gamma",
    "lambda",
    "plan_hint_max",
    "eta",
    "amort_alpha",
    "amort_rmin",
    "amort_reuse_prob",
    "lookahead_enable",
    "phase_reuse_enable",
    "token_amort_enable",
    "returncode",
    "prefill_time_s",
    "decode_time_s",
    "total_time_s",
    "pd_total_time_s",
    "speedup_vs_pd",
    "combined_json",
    "log_path",
]


def _parse_workload(spec: str) -> Dict[str, Any]:
    parts = [p.strip() for p in str(spec).split(":")]
    if len(parts) not in (6, 7):
        raise argparse.ArgumentTypeError(
            "workload must be family:variant:dtype:batch:prefill:decode[:horizon], "
            f"got {spec!r}"
        )
    family, variant, dtype, batch, prefill, decode = parts[:6]
    out = {
        "model_family": family,
        "model_variant": variant,
        "dtype": dtype,
        "batch": int(batch),
        "prefill_len": int(prefill),
        "decode_len": int(decode),
    }
    if len(parts) == 7:
        out["decode_horizon_len"] = int(parts[6])
    return out


def _read_config_scalar(config_py: Path, name: str, default: Any, cast: Any) -> Any:
    try:
        txt = config_py.read_text(encoding="utf-8")
        m = re.search(rf"^\s*{re.escape(name)}\s*(?::[^=\n]+)?=\s*([^#\n]+)", txt, re.M)
        if not m:
            return default
        raw = m.group(1).strip()
        if cast is bool:
            return raw.lower() in {"true", "1", "yes", "on"}
        return cast(raw)
    except Exception:
        return default


def _load_best_params(best_json: Optional[Path]) -> Dict[str, Any]:
    if not best_json:
        return {}
    if not best_json.exists():
        raise FileNotFoundError(f"--best-json does not exist: {best_json}")
    with best_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    params = data.get("params", data)
    if not isinstance(params, Mapping):
        return {}
    out: Dict[str, Any] = {}
    aliases = {
        "h": "h",
        "gamma": "gamma",
        "lambda": "lambda",
        "lambda_": "lambda",
        "plan_hint_max": "plan_hint_max",
        "eta": "eta",
        "amort_alpha": "amort_alpha",
        "amort_rmin": "amort_rmin",
        "amort_reuse_prob": "amort_reuse_prob",
    }
    for k, dest in aliases.items():
        if k in params and params[k] is not None:
            out[dest] = params[k]
    return out


def _bool_cli(v: bool) -> str:
    return "true" if bool(v) else "false"


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _gmean(vals: Iterable[float]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v)) and float(v) > 0]
    if not xs:
        return None
    return float(math.exp(sum(math.log(x) for x in xs) / len(xs)))


def _find_combined_json_paths(log_text: str, cwd: Path) -> List[Path]:
    out: List[Path] = []
    for pat in (
        re.compile(r"Combined comparison saved to:\s*(.+?\.json)\s*$", re.M),
        re.compile(r"\[REPORT\]\s*Combined comparison saved to:\s*(.+?\.json)\s*$", re.M),
    ):
        for m in pat.finditer(log_text):
            p = Path(m.group(1).strip())
            if not p.is_absolute():
                p = (cwd / p).resolve()
            if p not in out:
                out.append(p)
    return out


def _parse_policy_metrics(json_path: Path, policy: str = "algo:Bifocal") -> Optional[Dict[str, float]]:
    if not json_path.exists():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data.get("results", []):
            if str(row.get("policy", "")).strip() == policy:
                return {
                    "prefill": float(row.get("prefill_time_s", 0.0)),
                    "decode": float(row.get("decode_time_s", 0.0)),
                    "total": float(row.get("total_time_s", 0.0)),
                }
    except Exception:
        return None
    return None


def _parse_from_stdout(log_text: str, policy: str = "algo:Bifocal") -> Optional[Dict[str, float]]:
    pat = re.compile(
        rf"^\s*{re.escape(policy)}\s+\S+\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$",
        re.M,
    )
    matches = list(pat.finditer(log_text))
    if not matches:
        return None
    m = matches[-1]
    return {"prefill": float(m.group(1)), "decode": float(m.group(2)), "total": float(m.group(3))}


def _parse_metrics(log_text: str, cwd: Path, policy: str) -> Tuple[Optional[Dict[str, float]], str]:
    for p in reversed(_find_combined_json_paths(log_text, cwd)):
        res = _parse_policy_metrics(p, policy)
        if res is not None:
            return res, str(p)
    res = _parse_from_stdout(log_text, policy)
    return res, ""


def _row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("variant"),
        row.get("model_family"),
        row.get("model_variant"),
        int(row.get("batch", 0)),
        int(row.get("prefill_len", 0)),
        int(row.get("decode_len", 0)),
        row.get("horizon_mode"),
        int(row.get("decode_horizon_len", 0)),
    )


def _load_done(results_csv: Path) -> set[Tuple[Any, ...]]:
    if not results_csv.exists():
        return set()
    out: set[Tuple[Any, ...]] = set()
    try:
        with results_csv.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if str(r.get("returncode", "")) == "0" and r.get("total_time_s"):
                    out.add(_row_key(r))
    except Exception:
        pass
    return out


def _append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _cpu_count_fallback() -> int:
    try:
        return max(1, int(os.cpu_count() or 1))
    except Exception:
        return 1


def _resolve_jobs(requested: int, planned_runs: int) -> int:
    """Choose the number of concurrent subprocesses for the outer sweep.

    Each unit of work invokes ``src/main.py evaluate`` in a separate subprocess.
    The Bifocal scheduling algorithm inside one subprocess is left unchanged;
    only independent workload/variant/horizon runs are parallelized.
    """
    if planned_runs <= 0:
        return 1
    if requested and int(requested) > 0:
        return max(1, min(int(requested), int(planned_runs)))

    # ``--jobs 0`` means auto. On Slurm, respect the CPU allocation; otherwise
    # fall back to the visible CPU count. Cap the auto default conservatively so
    # that memory-heavy simulations do not oversubscribe a node by accident.
    cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    try:
        cpu_budget = int(cpus) if cpus else _cpu_count_fallback()
    except Exception:
        cpu_budget = _cpu_count_fallback()
    auto_jobs = max(1, min(8, cpu_budget, int(planned_runs)))
    return auto_jobs


def _sanitize_token(s: Any) -> str:
    raw = str(s)
    raw = re.sub(r"[^A-Za-z0-9_.+-]+", "_", raw).strip("_")
    return raw or "x"


def _variant_list(args: argparse.Namespace) -> List[str]:
    if args.variants:
        vals: List[str] = []
        for item in args.variants:
            vals.extend(tok for tok in str(item).split(",") if tok)
        bad = [v for v in vals if v not in VARIANTS]
        if bad:
            raise SystemExit(f"Unknown variants: {bad}; choices={list(VARIANTS)}")
        return vals
    return list(VARIANT_SUITES[args.variant_suite])


def _horizon_items(workload: Mapping[str, Any], args: argparse.Namespace) -> List[Tuple[str, int]]:
    actual = int(workload["decode_len"])
    explicit = workload.get("decode_horizon_len")
    modes = [m.strip().lower() for m in str(args.horizon_suite).replace("+", ",").split(",") if m.strip()]
    out: List[Tuple[str, int]] = []
    for mode in modes:
        if mode in {"oracle", "actual"}:
            out.append(("oracle", int(actual)))
        elif mode in {"fixed", "constant"}:
            out.append((f"fixed{int(args.fixed_horizon)}", int(args.fixed_horizon)))
        elif mode == "workload" and explicit is not None:
            out.append(("workload", int(explicit)))
        else:
            raise SystemExit(f"Unsupported horizon mode {mode!r}; use oracle,fixed,workload")
    # de-duplicate while keeping order
    seen = set()
    uniq: List[Tuple[str, int]] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _run_one(
    args: argparse.Namespace,
    workload: Mapping[str, Any],
    variant_name: str,
    horizon_mode: str,
    horizon_len: int,
    params: Mapping[str, Any],
    outdir: Path,
) -> Dict[str, Any]:
    variant = VARIANTS[variant_name]
    ts = dt.datetime.now().isoformat(timespec="seconds")

    tag = "_".join([
        _sanitize_token(variant_name),
        _sanitize_token(workload["model_family"]),
        _sanitize_token(workload["model_variant"]),
        f"b{workload['batch']}",
        f"p{workload['prefill_len']}",
        f"d{workload['decode_len']}",
        _sanitize_token(horizon_mode),
        f"h{horizon_len}",
    ])
    run_dir = outdir / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    cmd = [
        sys.executable,
        str(SRC_DIR / "main.py"),
        "evaluate",
        "--config", str(Path(args.config).resolve()),
        "--model_family", str(workload["model_family"]),
        "--model_variant", str(workload["model_variant"]),
        "--dtype", str(workload["dtype"]),
        "--batch", str(int(workload["batch"])),
        "--prefill_len", str(int(workload["prefill_len"])),
        "--decode_len", str(int(workload["decode_len"])),
        "--decode_horizon_len", str(int(horizon_len)),
        "--decode_sample_stride", str(int(args.decode_sample_stride)),
        "--decode_plan_refresh_stride", str(int(args.decode_plan_refresh_stride)),
        "--hardware_json", str(Path(args.hardware_json).resolve()),
        "--result_dir", str(run_dir / "eval"),
        "--algo", "Bifocal",
        "--baselines", str(args.baselines),
        "--npu_backend", str(args.npu_backend),
        "--bifocal-ready-score-enable", _bool_cli(True),
        "--bifocal-lookahead-enable", _bool_cli(variant["lookahead"]),
        "--bifocal-phase-reuse-enable", _bool_cli(variant["phase"]),
        "--bifocal-token-amort-enable", _bool_cli(variant["token"]),
        "--bifocal-h", str(int(params["h"])),
        "--bifocal-gamma", str(float(params["gamma"])),
        "--bifocal-lambda", str(float(params["lambda"])),
        "--bifocal-plan-hint-max", str(int(params["plan_hint_max"])),
        "--bifocal-eta", str(float(params["eta"])),
        "--bifocal-amort-alpha", str(float(params["amort_alpha"])),
        "--bifocal-amort-rmin", str(float(params["amort_rmin"])),
        "--bifocal-amort-reuse-prob", str(float(params["amort_reuse_prob"])),
    ]
    if bool(args.pim_fast_mode):
        cmd.append("--pim_fast_mode")
    if bool(args.debug):
        cmd.append("--debug")

    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    # Prevent each concurrent subprocess from internally spawning many BLAS/OpenMP
    # threads. This makes outer parallelism predictable on Slurm nodes.
    threads_per_run = max(1, int(getattr(args, "threads_per_run", 1) or 1))
    for env_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[env_name] = str(threads_per_run)
    # Write child output directly to its per-run log instead of buffering it in
    # memory. This matters when many evaluate subprocesses run concurrently.
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("# cmd: " + " ".join(cmd) + "\n\n")
        lf.flush()
        p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    bifocal, combined_json = _parse_metrics(log_text, PROJECT_ROOT, "algo:Bifocal")
    pd, _ = _parse_metrics(log_text, PROJECT_ROOT, "algo:PD")

    row: Dict[str, Any] = {
        "timestamp": ts,
        "variant": variant_name,
        "variant_description": variant.get("description", ""),
        **workload,
        "horizon_mode": horizon_mode,
        "decode_horizon_len": int(horizon_len),
        "hardware_json": str(Path(args.hardware_json).resolve()),
        "h": int(params["h"]),
        "gamma": float(params["gamma"]),
        "lambda": float(params["lambda"]),
        "plan_hint_max": int(params["plan_hint_max"]),
        "eta": float(params["eta"]),
        "amort_alpha": float(params["amort_alpha"]),
        "amort_rmin": float(params["amort_rmin"]),
        "amort_reuse_prob": float(params["amort_reuse_prob"]),
        "lookahead_enable": bool(variant["lookahead"]),
        "phase_reuse_enable": bool(variant["phase"]),
        "token_amort_enable": bool(variant["token"]),
        "returncode": int(p.returncode),
        "combined_json": combined_json,
        "log_path": str(log_path),
    }
    if bifocal is not None:
        row.update({
            "prefill_time_s": bifocal["prefill"],
            "decode_time_s": bifocal["decode"],
            "total_time_s": bifocal["total"],
        })
    if pd is not None:
        row["pd_total_time_s"] = pd["total"]
        if bifocal is not None and bifocal["total"] > 0:
            row["speedup_vs_pd"] = float(pd["total"] / bifocal["total"])
    return row


def _load_result_rows(results_csv: Path) -> List[Dict[str, Any]]:
    if not results_csv.exists():
        return []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f) if r.get("total_time_s")]


def _workload_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("model_family"),
        row.get("model_variant"),
        row.get("dtype"),
        int(row.get("batch", 0)),
        int(row.get("prefill_len", 0)),
        int(row.get("decode_len", 0)),
        row.get("horizon_mode"),
        int(row.get("decode_horizon_len", 0)),
    )


def _summarize(results_csv: Path, outdir: Path) -> None:
    rows = _load_result_rows(results_csv)
    if not rows:
        return

    # Normalize numeric fields.
    for r in rows:
        for k in ("batch", "prefill_len", "decode_len", "decode_horizon_len"):
            if k in r and r[k] != "":
                r[k] = int(float(r[k]))
        for k in ("prefill_time_s", "decode_time_s", "total_time_s", "pd_total_time_s", "speedup_vs_pd"):
            if k in r and r[k] != "":
                r[k] = float(r[k])

    by_wv: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_wv.setdefault(_workload_key(r), {})[str(r["variant"])] = r

    paired_rows: List[Dict[str, Any]] = []
    for wk, vr in by_wv.items():
        full = vr.get("Full")
        eft = vr.get("EFT-only")
        if not full and not eft:
            continue
        for variant, r in sorted(vr.items()):
            total = float(r["total_time_s"])
            item = {
                "variant": variant,
                "model_family": wk[0],
                "model_variant": wk[1],
                "dtype": wk[2],
                "batch": wk[3],
                "prefill_len": wk[4],
                "decode_len": wk[5],
                "horizon_mode": wk[6],
                "decode_horizon_len": wk[7],
                "total_time_s": total,
            }
            if eft is not None and float(eft["total_time_s"]) > 0:
                item["speedup_vs_eft"] = float(eft["total_time_s"]) / total
            if full is not None and float(full["total_time_s"]) > 0:
                item["slowdown_vs_full_pct"] = 100.0 * (total / float(full["total_time_s"]) - 1.0)
            paired_rows.append(item)
    _write_csv(outdir / "paired_workload_effects.csv", paired_rows)

    summary_rows: List[Dict[str, Any]] = []
    variants = sorted(set(str(r["variant"]) for r in rows))
    for v in variants:
        prs = [r for r in paired_rows if r.get("variant") == v]
        speedups = [_safe_float(r.get("speedup_vs_eft")) for r in prs]
        slowdowns = [_safe_float(r.get("slowdown_vs_full_pct")) for r in prs]
        slowdowns_f = [x for x in slowdowns if x is not None]
        summary_rows.append({
            "variant": v,
            "workloads": len(prs),
            "gmean_speedup_vs_eft": _gmean(x for x in speedups if x is not None),
            "median_slowdown_vs_full_pct": statistics.median(slowdowns_f) if slowdowns_f else None,
            "max_slowdown_vs_full_pct": max(slowdowns_f) if slowdowns_f else None,
            "benefit_count_vs_eft_gt_1pct": sum(1 for x in speedups if x is not None and x > 1.01),
            "regression_count_vs_eft_lt_minus_1pct": sum(1 for x in speedups if x is not None and x < 0.99),
        })
    _write_csv(outdir / "summary_by_variant.csv", summary_rows)

    component_specs = [
        ("Lookahead", "Full-w/o-Lookahead"),
        ("PhaseReuse", "Full-w/o-Phase"),
        ("TokenAmort", "Full-w/o-Token"),
    ]
    comp_rows: List[Dict[str, Any]] = []
    comp_by_workload: List[Dict[str, Any]] = []
    for comp, ablated in component_specs:
        speedups: List[float] = []
        for wk, vr in by_wv.items():
            full = vr.get("Full")
            ab = vr.get(ablated)
            if full is None or ab is None:
                continue
            ft = float(full["total_time_s"])
            at = float(ab["total_time_s"])
            if ft <= 0 or at <= 0:
                continue
            sp = at / ft
            speedups.append(sp)
            comp_by_workload.append({
                "component": comp,
                "model_family": wk[0],
                "model_variant": wk[1],
                "dtype": wk[2],
                "batch": wk[3],
                "prefill_len": wk[4],
                "decode_len": wk[5],
                "horizon_mode": wk[6],
                "decode_horizon_len": wk[7],
                "speedup_when_enabled": sp,
                "enabled_total_time_s": ft,
                "ablated_total_time_s": at,
                "active_gt_1pct": sp > 1.01,
                "harmful_lt_minus_1pct": sp < 0.99,
            })
        comp_rows.append({
            "component": comp,
            "paired_workloads": len(speedups),
            "gmean_speedup_when_enabled": _gmean(speedups),
            "median_speedup_when_enabled": statistics.median(speedups) if speedups else None,
            "min_speedup_when_enabled": min(speedups) if speedups else None,
            "max_speedup_when_enabled": max(speedups) if speedups else None,
            "active_count_gt_1pct": sum(1 for x in speedups if x > 1.01),
            "neutral_count_abs_le_1pct": sum(1 for x in speedups if 0.99 <= x <= 1.01),
            "harmful_count_lt_minus_1pct": sum(1 for x in speedups if x < 0.99),
        })
    _write_csv(outdir / "component_effects.csv", comp_rows)
    _write_csv(outdir / "component_effects_by_workload.csv", comp_by_workload)

    # Coarse regime split to make conditional effects easy to discuss.
    regime_rows: List[Dict[str, Any]] = []
    buckets: Dict[Tuple[str, str, str, str, str], List[float]] = {}
    for r in comp_by_workload:
        key = (
            str(r["component"]),
            f"{r['model_family']}-{r['model_variant']}",
            "prefill_long" if int(r["prefill_len"]) >= 1024 else "prefill_short",
            "decode_long" if int(r["decode_len"]) >= 512 else "decode_short",
            f"b{r['batch']}",
        )
        buckets.setdefault(key, []).append(float(r["speedup_when_enabled"]))
    for key, vals in sorted(buckets.items()):
        regime_rows.append({
            "component": key[0],
            "model": key[1],
            "prefill_regime": key[2],
            "decode_regime": key[3],
            "batch": key[4],
            "workloads": len(vals),
            "gmean_speedup_when_enabled": _gmean(vals),
            "active_count_gt_1pct": sum(1 for x in vals if x > 1.01),
            "neutral_count_abs_le_1pct": sum(1 for x in vals if 0.99 <= x <= 1.01),
            "harmful_count_lt_minus_1pct": sum(1 for x in vals if x < 0.99),
        })
    _write_csv(outdir / "component_effects_by_regime.csv", regime_rows)


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Bifocal component ablation suite", allow_abbrev=False)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON config used by src/main.py evaluate")
    ap.add_argument("--hardware-json", "--hardware_json", dest="hardware_json", default=str(DEFAULT_HW), help="HP32/HP64/etc hardware JSON")
    ap.add_argument("--outdir", default=str(PROJECT_ROOT / "output" / "bifocal_component_ablation"))
    ap.add_argument("--npu-backend", "--npu_backend", dest="npu_backend", default="fast_mode", choices=["fast", "fast_mode", "lut", "ascend_310b_json", "llmcompass"])
    ap.add_argument("--pim-fast-mode", "--pim_fast_mode", action="store_true")
    ap.add_argument("--baselines", default="PD", help="Baselines passed to evaluate; keep PD for speedup_vs_pd")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true", help="Use four workloads instead of the full representative Section-6 list")
    ap.add_argument("--workload", "--workloads", dest="workloads", action="append", help="family:variant:dtype:batch:prefill:decode[:horizon]; repeatable")
    ap.add_argument("--variant-suite", choices=sorted(VARIANT_SUITES), default="minimal")
    ap.add_argument("--variants", nargs="*", help="Explicit variants. Choices: " + ", ".join(VARIANTS))
    ap.add_argument("--horizon-suite", default="oracle", help="Comma list: oracle,fixed,workload")
    ap.add_argument("--fixed-horizon", type=int, default=256, help="Non-oracle planning horizon for --horizon-suite fixed")
    ap.add_argument("--decode-sample-stride", type=int, default=8)
    ap.add_argument("--decode-plan-refresh-stride", type=int, default=8)
    ap.add_argument("--best-json", type=str, help="best_result.json produced by sweep_bifocal_all_params.py")
    ap.add_argument("--config-py", default=str(SRC_DIR / "config.py"), help="config.py used only for default hparams when --best-json/CLI does not set them")
    ap.add_argument("--h", type=int)
    ap.add_argument("--gamma", type=float)
    ap.add_argument("--lambda", "--lambda_", dest="lambda_", type=float)
    ap.add_argument("--plan-hint-max", "--plan_hint_max", dest="plan_hint_max", type=int)
    ap.add_argument("--eta", type=float)
    ap.add_argument("--amort-alpha", "--amort_alpha", dest="amort_alpha", type=float)
    ap.add_argument("--amort-rmin", "--amort_rmin", dest="amort_rmin", type=float)
    ap.add_argument("--amort-reuse-prob", "--amort_reuse_prob", dest="amort_reuse_prob", type=float)
    ap.add_argument("--max-runs", type=int, default=0, help="Debug cap after resume filtering; 0=unlimited")
    ap.add_argument("--jobs", type=int, default=1, help="Concurrent independent evaluate subprocesses. Use 0 for conservative auto based on SLURM_CPUS_PER_TASK.")
    ap.add_argument("--threads-per-run", type=int, default=1, help="OMP/BLAS threads assigned to each evaluate subprocess")
    return ap


def main() -> int:
    args = make_parser().parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    results_csv = outdir / "component_ablation_results.csv"

    cfg_py = Path(args.config_py)
    if not cfg_py.is_absolute():
        cfg_py = (PROJECT_ROOT / cfg_py).resolve()
    params = {
        "h": _read_config_scalar(cfg_py, "SCHED_JOINT_LK_H", 3, int),
        "gamma": _read_config_scalar(cfg_py, "SCHED_JOINT_LK_GAMMA", 0.2, float),
        "lambda": _read_config_scalar(cfg_py, "SCHED_JOINT_LK_CONSIST_LAMBDA", 1.0, float),
        "plan_hint_max": _read_config_scalar(cfg_py, "SCHED_JOINT_LK_PLAN_HINT_MAX", 3, int),
        "eta": _read_config_scalar(cfg_py, "SCHED_WEIGHT_BIAS_ETA", 0.1, float),
        "amort_alpha": _read_config_scalar(cfg_py, "SCHED_DECODE_AMORT_ALPHA", 1.0, float),
        "amort_rmin": _read_config_scalar(cfg_py, "SCHED_DECODE_AMORT_RMIN", 1.0, float),
        "amort_reuse_prob": _read_config_scalar(cfg_py, "SCHED_DECODE_AMORT_REUSE_PROB", 1.0, float),
    }
    params.update(_load_best_params(Path(args.best_json).resolve() if args.best_json else None))
    # CLI has highest priority.
    for src, dest in [
        (args.h, "h"),
        (args.gamma, "gamma"),
        (args.lambda_, "lambda"),
        (args.plan_hint_max, "plan_hint_max"),
        (args.eta, "eta"),
        (args.amort_alpha, "amort_alpha"),
        (args.amort_rmin, "amort_rmin"),
        (args.amort_reuse_prob, "amort_reuse_prob"),
    ]:
        if src is not None:
            params[dest] = src

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "variant_suite": args.variant_suite,
        "variants": _variant_list(args),
        "hparams": params,
        "horizon_suite": args.horizon_suite,
        "fixed_horizon": args.fixed_horizon,
        "config": str(Path(args.config).resolve()),
        "hardware_json": str(Path(args.hardware_json).resolve()),
        "jobs": int(args.jobs),
        "threads_per_run": int(args.threads_per_run),
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if float(params.get("gamma", 0.0)) == 0.0 and any(VARIANTS[v]["lookahead"] for v in _variant_list(args)):
        print("[warn] gamma=0: lookahead-enabled variants will numerically collapse to EFT for the window term.", file=sys.stderr)
    if float(params.get("lambda", 0.0)) == 0.0 and float(params.get("eta", 0.0)) == 0.0:
        print("[warn] lambda=eta=0: phase-reuse variants will be intentionally neutral.", file=sys.stderr)

    workload_specs = args.workloads or (QUICK_WORKLOADS if args.quick else DEFAULT_WORKLOADS)
    workloads = [_parse_workload(w) for w in workload_specs]
    variants = _variant_list(args)
    done = _load_done(results_csv) if args.resume else set()

    planned = []
    for wl in workloads:
        for horizon_mode, horizon_len in _horizon_items(wl, args):
            for variant in variants:
                stub = {
                    "variant": variant,
                    **wl,
                    "horizon_mode": horizon_mode,
                    "decode_horizon_len": horizon_len,
                }
                if args.resume and _row_key(stub) in done:
                    continue
                planned.append((wl, variant, horizon_mode, horizon_len))
    if args.max_runs:
        planned = planned[: int(args.max_runs)]
        print(f"[info] applied --max-runs={args.max_runs}; remaining planned_runs={len(planned)}")

    jobs = _resolve_jobs(int(args.jobs), len(planned))
    print(f"[info] outdir={outdir}")
    print(f"[info] workloads={len(workloads)} variants={len(variants)} planned_runs={len(planned)}")
    print(f"[info] jobs={jobs} threads_per_run={max(1, int(args.threads_per_run))}")
    print(f"[info] hparams={params}")

    if planned and jobs == 1:
        for idx, (wl, variant, horizon_mode, horizon_len) in enumerate(planned, start=1):
            print(
                f"\n=== run {idx}/{len(planned)} variant={variant} "
                f"model={wl['model_family']}-{wl['model_variant']} b={wl['batch']} "
                f"p={wl['prefill_len']} d={wl['decode_len']} horizon={horizon_mode}:{horizon_len} ===",
                flush=True,
            )
            row = _run_one(args, wl, variant, horizon_mode, horizon_len, params, outdir)
            _append_csv(results_csv, row)
            if row.get("total_time_s"):
                msg = f"[ok] total={float(row['total_time_s']):.6f}s"
                if row.get("speedup_vs_pd"):
                    msg += f" speedup_vs_PD={float(row['speedup_vs_pd']):.4f}x"
                print(msg)
            else:
                print(f"[warn] metrics missing rc={row.get('returncode')} log={row.get('log_path')}")
    elif planned:
        print(f"[info] launching {len(planned)} independent runs with {jobs} workers", flush=True)
        completed = 0
        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            future_to_desc = {}
            for wl, variant, horizon_mode, horizon_len in planned:
                desc = (wl, variant, horizon_mode, horizon_len)
                fut = ex.submit(_run_one, args, wl, variant, horizon_mode, horizon_len, params, outdir)
                future_to_desc[fut] = desc
            for fut in cf.as_completed(future_to_desc):
                wl, variant, horizon_mode, horizon_len = future_to_desc[fut]
                completed += 1
                try:
                    row = fut.result()
                except Exception as exc:
                    # Keep the sweep alive even if one child setup fails before it
                    # can write its own log. The failed workload can be rerun with
                    # --resume after fixing the cause.
                    row = {
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                        "variant": variant,
                        "variant_description": VARIANTS[variant].get("description", ""),
                        **wl,
                        "horizon_mode": horizon_mode,
                        "decode_horizon_len": int(horizon_len),
                        "hardware_json": str(Path(args.hardware_json).resolve()),
                        "h": int(params["h"]),
                        "gamma": float(params["gamma"]),
                        "lambda": float(params["lambda"]),
                        "plan_hint_max": int(params["plan_hint_max"]),
                        "eta": float(params["eta"]),
                        "amort_alpha": float(params["amort_alpha"]),
                        "amort_rmin": float(params["amort_rmin"]),
                        "amort_reuse_prob": float(params["amort_reuse_prob"]),
                        "lookahead_enable": bool(VARIANTS[variant]["lookahead"]),
                        "phase_reuse_enable": bool(VARIANTS[variant]["phase"]),
                        "token_amort_enable": bool(VARIANTS[variant]["token"]),
                        "returncode": -998,
                        "log_path": f"EXCEPTION: {exc!r}",
                    }
                _append_csv(results_csv, row)
                prefix = (
                    f"[{completed}/{len(planned)}] variant={variant} "
                    f"model={wl['model_family']}-{wl['model_variant']} b={wl['batch']} "
                    f"p={wl['prefill_len']} d={wl['decode_len']} horizon={horizon_mode}:{horizon_len}"
                )
                if row.get("total_time_s"):
                    msg = f"{prefix} ok total={float(row['total_time_s']):.6f}s"
                    if row.get("speedup_vs_pd"):
                        msg += f" speedup_vs_PD={float(row['speedup_vs_pd']):.4f}x"
                    print(msg, flush=True)
                else:
                    print(f"{prefix} warn metrics missing rc={row.get('returncode')} log={row.get('log_path')}", flush=True)

    _summarize(results_csv, outdir)
    print(f"\n[info] results: {results_csv}")
    print(f"[info] summary: {outdir / 'component_effects.csv'}")
    print(f"[info] by workload: {outdir / 'component_effects_by_workload.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
