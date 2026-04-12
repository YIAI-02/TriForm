#!/usr/bin/env python3
"""
Sweep Bifocal-related scheduler knobs by editing config.py in place,
launching an external evaluation shell script, and collecting the
algo:Bifocal metrics.

Compared with the original Bifocal sweep script, this version:
  1) scans all current Bifocal knobs listed in config.py;
  2) supports bool/int/float parameters;
  3) prefers parsing the combined comparison JSON emitted by main.py;
  4) falls back to robust stdout-table regex parsing when needed;
  5) supports grid/random over discrete candidate lists.

Example:
python3 commands/sweep_bifocal_all_params.py \
  --mode grid \
  --config-py ./config.py \
  --h 2 3 4 \
  --gamma 0.2 0.4 0.6 0.8 \
  --lambda 0 1 2 4 \
  --plan_hint_max 1 \
  --eta 0 5 10 \
  --amort_enable false \
  --objective total \
  --outdir ./output/sweep_bifocal_all \
  --resume \
  --config ./src/examples/evaluate_test_config.json

Notes:
- This sweep edits config.py in place, so its own file flag is --config-py.
- Unknown args are passed through to the runner (typically main.py evaluate),
  so the standard runtime flag --config keeps its JSON-config meaning.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_COMMAND_SCRIPT = PROJECT_ROOT / "commands" / "command_single_evaluate.sh"

# ---- knobs we will edit in config.py ----
PARAM_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("h", "SCHED_JOINT_LK_H", "int"),
    ("gamma", "SCHED_JOINT_LK_GAMMA", "float"),
    ("lambda", "SCHED_JOINT_LK_CONSIST_LAMBDA", "float"),
    ("plan_hint_max", "SCHED_JOINT_LK_PLAN_HINT_MAX", "int"),
    ("eta", "SCHED_WEIGHT_BIAS_ETA", "float"),
    ("amort_enable", "SCHED_DECODE_AMORT_ENABLE", "bool"),
    ("amort_alpha", "SCHED_DECODE_AMORT_ALPHA", "float"),
    ("amort_rmin", "SCHED_DECODE_AMORT_RMIN", "float"),
    ("amort_reuse_prob", "SCHED_DECODE_AMORT_REUSE_PROB", "float"),
)

PARAM_NAME_TO_CFG = {k: cfg_name for k, cfg_name, _ in PARAM_SPECS}
PARAM_NAME_TO_TYPE = {k: typ for k, _, typ in PARAM_SPECS}
RESULT_FIELD_ORDER = [name for name, _, _ in PARAM_SPECS]
TARGET_POLICY_DEFAULT = "algo:Bifocal"


def _looks_like_python_config_path(value: str) -> bool:
    s = str(value or "").strip()
    if not s:
        return False
    name = Path(s).name.lower()
    return s.lower().endswith(".py") or name == "config.py"


def _rewrite_legacy_config_flag(argv: Sequence[str]) -> List[str]:
    """
    Keep backward compatibility for the old sweep-local --config flag while
    freeing standard runner --config (JSON) for pass-through args.

    - --config config.py        -> --config-py config.py
    - --config=./path/foo.py    -> --config-py=./path/foo.py
    - --config ./examples/x.json stays untouched and is forwarded to runner
    """
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = str(argv[i])
        if tok == "--config" and i + 1 < len(argv):
            val = str(argv[i + 1])
            if _looks_like_python_config_path(val):
                out.extend(["--config-py", val])
                i += 2
                continue
        if tok.startswith("--config="):
            val = tok.split("=", 1)[1]
            if _looks_like_python_config_path(val):
                out.append(f"--config-py={val}")
                i += 1
                continue
        out.append(tok)
        i += 1
    return out


def _validate_candidate_axes(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    axes = {
        "h": args.h,
        "gamma": args.gamma,
        "lambda": args.lambda_,
        "plan_hint_max": args.plan_hint_max,
        "eta": args.eta,
        "amort_enable": args.amort_enable,
        "amort_alpha": args.amort_alpha,
        "amort_rmin": args.amort_rmin,
        "amort_reuse_prob": args.amort_reuse_prob,
    }
    empty = [name for name, vals in axes.items() if len(list(vals)) == 0]
    if empty:
        ap.error("candidate list cannot be empty: " + ", ".join(empty))


def _script_supports_passthrough(script_path: Path) -> bool:
    try:
        txt = _read_text(script_path)
    except Exception:
        return False
    return any(token in txt for token in ('"$@"', "$@", "${@}"))


def _fmt_num(v: float) -> str:
    if not math.isfinite(v):
        raise ValueError(f"non-finite value: {v}")
    return f"{v:.12g}"


def _fmt_py_literal(value: Any, kind: str) -> str:
    kind = str(kind).lower()
    if kind == "bool":
        return "True" if bool(value) else "False"
    if kind == "int":
        return str(int(value))
    if kind == "float":
        return _fmt_num(float(value))
    raise ValueError(f"unsupported literal kind: {kind}")


def _parse_bool_token(token: str) -> bool:
    s = str(token).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", "disable", "disabled"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool token: {token}")


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def _replace_scalar_param_line(text: str, name: str, new_value: Any, kind: str) -> str:
    """
    Replace a scalar config line like:
      NAME: float = 0.6
      NAME = True
    while preserving trailing comments.
    """
    pat = re.compile(
        rf"^(\s*{re.escape(name)}\s*(?::[^\n=]+)?=\s*)"
        rf"([^#\n]+?)"
        rf"(\s*(?:#.*)?)$",
        re.MULTILINE,
    )
    if not pat.search(text):
        raise RuntimeError(f"Parameter line not found in config.py: {name}")
    return pat.sub(
        lambda m: f"{m.group(1)}{_fmt_py_literal(new_value, kind)}{m.group(3)}",
        text,
        count=1,
    )


def update_config(config_path: Path, values: Dict[str, Any]) -> None:
    txt = _read_text(config_path)
    for key, cfg_name, kind in PARAM_SPECS:
        if key not in values:
            continue
        txt = _replace_scalar_param_line(txt, cfg_name, values[key], kind)
    _write_text(config_path, txt)


def sanitize_tag(s: str) -> str:
    s = s.replace("=", "_").replace(":", "_").replace("/", "_")
    s = s.replace(" ", "")
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    return s


def run_command(script_path: Path, workdir: Path, extra_args: Sequence[str], env: Dict[str, str]) -> Tuple[int, str]:
    cmd = ["bash", str(script_path)]
    cmd += list(extra_args)
    p = subprocess.run(
        cmd,
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
    )
    out = ""
    if p.stdout:
        out += p.stdout
    if p.stderr:
        out += ("\n" if out else "") + p.stderr
    return p.returncode, out


def _coerce_combo_key(params: Dict[str, Any]) -> Tuple[Any, ...]:
    out: List[Any] = []
    for name in RESULT_FIELD_ORDER:
        kind = PARAM_NAME_TO_TYPE[name]
        v = params[name]
        if kind == "bool":
            out.append(bool(v))
        elif kind == "int":
            out.append(int(v))
        else:
            out.append(float(v))
    return tuple(out)


def load_done_keys(results_csv: Path) -> set:
    done = set()
    if not results_csv.exists():
        return done
    try:
        with results_csv.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    params: Dict[str, Any] = {}
                    for name in RESULT_FIELD_ORDER:
                        kind = PARAM_NAME_TO_TYPE[name]
                        raw = row[name]
                        if kind == "bool":
                            params[name] = _parse_bool_token(raw)
                        elif kind == "int":
                            params[name] = int(raw)
                        else:
                            params[name] = float(raw)
                    rep = int(row.get("repeat_idx", "1"))
                    done.add((_coerce_combo_key(params), rep))
                except Exception:
                    continue
    except Exception:
        pass
    return done


def append_result(results_csv: Path, row: Dict[str, Any]) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    fieldnames = [
        "timestamp",
        "run_id",
        "repeat_idx",
        *RESULT_FIELD_ORDER,
        "objective_name",
        "objective",
        "prefill",
        "decode",
        "total",
        "returncode",
        "metrics_source",
        "metrics_json",
        "log_path",
    ]
    with results_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def save_best_json(best_path: Path, payload: Dict[str, Any]) -> None:
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _parse_metrics_from_combined_json(json_path: Path, target_policy: str) -> Optional[Dict[str, Any]]:
    if not json_path.exists():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        for row in results:
            if str(row.get("policy", "")).strip() == str(target_policy):
                return {
                    "prefill": float(row["prefill_time_s"]),
                    "decode": float(row["decode_time_s"]),
                    "total": float(row["total_time_s"]),
                    "source": "combined_json",
                    "json_path": str(json_path),
                }
    except Exception:
        return None
    return None


def _find_combined_json_paths(log_text: str, workdir: Path) -> List[Path]:
    paths: List[Path] = []
    patterns = [
        re.compile(r"Combined comparison saved to:\s*(.+?\.json)\s*$", re.MULTILINE),
        re.compile(r"\[REPORT\]\s*Combined comparison saved to:\s*(.+?\.json)\s*$", re.MULTILINE),
    ]
    for pat in patterns:
        for m in pat.finditer(log_text):
            raw = m.group(1).strip()
            p = Path(raw)
            if not p.is_absolute():
                p = (workdir / p).resolve()
            if p not in paths:
                paths.append(p)
    return paths


def _parse_metrics_from_stdout_table(log_text: str, target_policy: str) -> Optional[Dict[str, Any]]:
    # main.py prints:
    # Policy                   PIM        Prefill(s)    Decode(s)     Total(s)
    # algo:Bifocal             host          1.2345       2.3456       3.5801
    # Need to tolerate any non-space PIM token in the middle.
    patterns = [
        re.compile(
            rf"^\s*{re.escape(target_policy)}\s+\S+\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$",
            re.MULTILINE,
        ),
        # Backward-compatible fallback without the PIM column.
        re.compile(
            rf"^\s*{re.escape(target_policy)}\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$",
            re.MULTILINE,
        ),
    ]
    for pat in patterns:
        matches = list(pat.finditer(log_text))
        if matches:
            m = matches[-1]
            return {
                "prefill": float(m.group(1)),
                "decode": float(m.group(2)),
                "total": float(m.group(3)),
                "source": "stdout_table",
                "json_path": "",
            }
    return None


def parse_bifocal_metrics(log_text: str, workdir: Path, target_policy: str) -> Optional[Dict[str, Any]]:
    # Prefer JSON emitted by main.py because it's much more robust.
    for json_path in reversed(_find_combined_json_paths(log_text, workdir)):
        metrics = _parse_metrics_from_combined_json(json_path, target_policy)
        if metrics is not None:
            return metrics
    return _parse_metrics_from_stdout_table(log_text, target_policy)


def build_grid_combos(args: argparse.Namespace) -> List[Dict[str, Any]]:
    axes = [
        list(args.h),
        list(args.gamma),
        list(args.lambda_),
        list(args.plan_hint_max),
        list(args.eta),
        list(args.amort_enable),
        list(args.amort_alpha),
        list(args.amort_rmin),
        list(args.amort_reuse_prob),
    ]
    names = RESULT_FIELD_ORDER
    combos: List[Dict[str, Any]] = []
    for values in itertools.product(*axes):
        combos.append({k: v for k, v in zip(names, values)})
    return combos


def build_random_combos(args: argparse.Namespace) -> List[Dict[str, Any]]:
    all_combos = build_grid_combos(args)
    rnd = random.Random(args.seed)
    rnd.shuffle(all_combos)
    if args.trials <= 0 or args.trials >= len(all_combos):
        return all_combos
    return all_combos[: int(args.trials)]


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Sweep all Bifocal scheduler parameters",
        allow_abbrev=False,
    )
    ap.add_argument(
        "--config-py", "--config_py", "--scheduler-config-py",
        dest="config_py",
        default="config.py",
        help="path to the config.py file edited in place by this sweep script",
    )
    ap.add_argument("--script", default=str(DEFAULT_COMMAND_SCRIPT), help="bash script to run")
    ap.add_argument("--workdir", default=str(PROJECT_ROOT / "src"), help="working directory")
    ap.add_argument("--target-policy", "--target_policy", default=TARGET_POLICY_DEFAULT, help="policy row to extract, e.g. algo:Bifocal")

    ap.add_argument("--mode", choices=["grid", "random"], default="grid")
    ap.add_argument("--objective", choices=["total", "decode", "prefill"], default="total")

    # Explicit discrete candidate lists for all Bifocal-related knobs.
    ap.add_argument("--h", type=int, nargs="*", default=[2, 3, 4], help="candidate SCHED_JOINT_LK_H values")
    ap.add_argument("--gamma", type=float, nargs="*", default=[0.2, 0.4, 0.6, 0.8], help="candidate SCHED_JOINT_LK_GAMMA values")
    ap.add_argument("--lambda", "--lambda_", dest="lambda_", type=float, nargs="*", default=[0.0, 1.0, 2.0, 4.0], help="candidate SCHED_JOINT_LK_CONSIST_LAMBDA values")
    ap.add_argument("--plan-hint-max", "--plan_hint_max", dest="plan_hint_max", type=int, nargs="*", default=[1, 3, 5], help="candidate SCHED_JOINT_LK_PLAN_HINT_MAX values")
    ap.add_argument("--eta", type=float, nargs="*", default=[1.0, 5.0, 10.0], help="candidate SCHED_WEIGHT_BIAS_ETA values")
    ap.add_argument("--amort-enable", "--amort_enable", dest="amort_enable", type=_parse_bool_token, nargs="*", default=[True, False], help="candidate SCHED_DECODE_AMORT_ENABLE values")
    ap.add_argument("--amort-alpha", "--amort_alpha", dest="amort_alpha", type=float, nargs="*", default=[2.0, 4.0, 6.0], help="candidate SCHED_DECODE_AMORT_ALPHA values")
    ap.add_argument("--amort-rmin", "--amort_rmin", dest="amort_rmin", type=float, nargs="*", default=[0.5, 1.0, 2.0], help="candidate SCHED_DECODE_AMORT_RMIN values")
    ap.add_argument("--amort-reuse-prob", "--amort_reuse_prob", dest="amort_reuse_prob", type=float, nargs="*", default=[0.5, 1.0], help="candidate SCHED_DECODE_AMORT_REUSE_PROB values")

    ap.add_argument("--trials", type=int, default=256, help="random mode: number of sampled combinations")
    ap.add_argument("--seed", type=int, default=0, help="random mode seed")
    ap.add_argument("--repeat", type=int, default=1, help="repeat each combo N times")
    ap.add_argument("--max-runs", "--max_runs", type=int, default=0, help="cap total runs (0=unlimited)")
    ap.add_argument("--outdir", default=str(PROJECT_ROOT / "output" / "sweep_bifocal_all"), help="output directory for logs/results")
    ap.add_argument("--resume", action="store_true", help="skip combos already in results.csv")
    return ap


def main() -> int:
    ap = make_parser()
    argv = _rewrite_legacy_config_flag(sys.argv[1:])
    args, extra = ap.parse_known_args(argv)
    _validate_candidate_axes(ap, args)

    workdir = Path(args.workdir).resolve()
    config_path = Path(args.config_py)
    if not config_path.is_absolute():
        config_path = (workdir / config_path).resolve()
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = (workdir / script_path).resolve()
    outdir = Path(args.outdir).resolve()
    results_csv = outdir / "results.csv"
    best_json = outdir / "best_result.json"

    if not config_path.exists():
        print(f"[err] config not found: {config_path}", file=sys.stderr)
        return 2
    if not script_path.exists():
        print(f"[err] script not found: {script_path}", file=sys.stderr)
        return 2

    combos = build_grid_combos(args) if args.mode == "grid" else build_random_combos(args)
    total_combo_count = len(combos)

    done_keys = load_done_keys(results_csv) if args.resume else set()

    orig_text = _read_text(config_path)
    backup_path = config_path.with_suffix(config_path.suffix + ".bifocal_all_sweep_bak")
    shutil.copy2(config_path, backup_path)

    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")

    best: Dict[str, Any] = {
        "objective_name": args.objective,
        "objective": float("inf"),
        "params": None,
        "log_path": None,
        "metrics_source": None,
        "metrics_json": None,
        "prefill": None,
        "decode": None,
        "total": None,
    }

    print(f"[info] mode={args.mode} combos={total_combo_count} repeat={args.repeat} objective={args.objective}")
    print(f"[info] script={script_path}")
    print(f"[info] workdir={workdir}")
    print(f"[info] outdir={outdir}")
    if extra:
        print(f"[info] pass-through args to runner: {' '.join(extra)}")
        if not _script_supports_passthrough(script_path):
            print(
                f"[warn] runner script does not appear to forward $@: {script_path} ; "
                'pass-through args may be ignored unless the script appends "$@" to its main command.'
            )

    run_id = 0
    executed = 0

    try:
        for params in combos:
            combo_key = _coerce_combo_key(params)
            for rep in range(1, int(args.repeat) + 1):
                if (combo_key, rep) in done_keys:
                    continue

                run_id += 1
                if args.max_runs and executed >= int(args.max_runs):
                    print("[info] reached --max-runs cap, stop.")
                    return 0

                executed += 1
                update_config(config_path, params)

                tag = "_".join(
                    [
                        f"h={params['h']}",
                        f"g={_fmt_num(float(params['gamma']))}",
                        f"lam={_fmt_num(float(params['lambda']))}",
                        f"ph={params['plan_hint_max']}",
                        f"eta={_fmt_num(float(params['eta']))}",
                        f"ae={'T' if bool(params['amort_enable']) else 'F'}",
                        f"aa={_fmt_num(float(params['amort_alpha']))}",
                        f"ar={_fmt_num(float(params['amort_rmin']))}",
                        f"arp={_fmt_num(float(params['amort_reuse_prob']))}",
                        f"r={rep}",
                    ]
                )
                run_dir = outdir / "runs" / f"{run_id:06d}_{sanitize_tag(tag)}"
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / "run.log"

                ts = dt.datetime.now().isoformat(timespec="seconds")
                print(
                    "\n=== [{rid}] {ts} h={h} gamma={g} lambda={lam} plan_hint_max={ph} eta={eta} "
                    "amort_enable={ae} amort_alpha={aa} amort_rmin={ar} amort_reuse_prob={arp} rep={rep} ===".format(
                        rid=run_id,
                        ts=ts,
                        h=params["h"],
                        g=_fmt_num(float(params["gamma"])),
                        lam=_fmt_num(float(params["lambda"])),
                        ph=params["plan_hint_max"],
                        eta=_fmt_num(float(params["eta"])),
                        ae=params["amort_enable"],
                        aa=_fmt_num(float(params["amort_alpha"])),
                        ar=_fmt_num(float(params["amort_rmin"])),
                        arp=_fmt_num(float(params["amort_reuse_prob"])),
                        rep=rep,
                    )
                )

                rc, out = run_command(script_path, workdir=workdir, extra_args=extra, env=env)

                with log_path.open("w", encoding="utf-8") as f:
                    f.write(f"# cmd: bash {script_path}\n")
                    if extra:
                        f.write(f"# extra: {' '.join(extra)}\n")
                    f.write(f"# time: {ts}\n")
                    f.write("# params:\n")
                    for key in RESULT_FIELD_ORDER:
                        f.write(f"#   {PARAM_NAME_TO_CFG[key]} = {_fmt_py_literal(params[key], PARAM_NAME_TO_TYPE[key])}\n")
                    f.write(f"# repeat = {rep}\n\n")
                    f.write(out)

                metrics = parse_bifocal_metrics(out, workdir=workdir, target_policy=args.target_policy)

                row: Dict[str, Any] = {
                    "timestamp": ts,
                    "run_id": run_id,
                    "repeat_idx": rep,
                    **params,
                    "objective_name": args.objective,
                    "objective": "",
                    "prefill": "",
                    "decode": "",
                    "total": "",
                    "returncode": rc,
                    "metrics_source": "",
                    "metrics_json": "",
                    "log_path": str(log_path),
                }

                if metrics is None:
                    print(f"[warn] cannot parse {args.target_policy} metrics. rc={rc} log={log_path}")
                    append_result(results_csv, row)
                    if rc != 0:
                        print("[warn] command failed (non-zero). continue.")
                    continue

                obj = float(metrics[args.objective])
                row.update(
                    {
                        "objective": obj,
                        "prefill": float(metrics["prefill"]),
                        "decode": float(metrics["decode"]),
                        "total": float(metrics["total"]),
                        "metrics_source": str(metrics.get("source", "")),
                        "metrics_json": str(metrics.get("json_path", "")),
                    }
                )
                append_result(results_csv, row)

                print(
                    f"[ok] {args.target_policy} prefill={float(metrics['prefill']):.4f} "
                    f"decode={float(metrics['decode']):.4f} total={float(metrics['total']):.4f} "
                    f"-> objective({args.objective})={obj:.4f} "
                    f"source={metrics.get('source', '')}"
                )

                if obj < float(best["objective"]):
                    best.update(
                        {
                            "objective": obj,
                            "params": dict(params),
                            "log_path": str(log_path),
                            "metrics_source": str(metrics.get("source", "")),
                            "metrics_json": str(metrics.get("json_path", "")),
                            "prefill": float(metrics["prefill"]),
                            "decode": float(metrics["decode"]),
                            "total": float(metrics["total"]),
                        }
                    )
                    save_best_json(best_json, best)
                    print(
                        "[best] objective={obj:.6f} at h={h} gamma={g} lambda={lam} plan_hint_max={ph} eta={eta} "
                        "amort_enable={ae} amort_alpha={aa} amort_rmin={ar} amort_reuse_prob={arp}".format(
                            obj=obj,
                            h=params["h"],
                            g=_fmt_num(float(params["gamma"])),
                            lam=_fmt_num(float(params["lambda"])),
                            ph=params["plan_hint_max"],
                            eta=_fmt_num(float(params["eta"])),
                            ae=params["amort_enable"],
                            aa=_fmt_num(float(params["amort_alpha"])),
                            ar=_fmt_num(float(params["amort_rmin"])),
                            arp=_fmt_num(float(params["amort_reuse_prob"])),
                        )
                    )
                    print(f"[best] log: {best['log_path']}")

    finally:
        _write_text(config_path, orig_text)
        print(f"\n[info] restored {config_path}")
        print(f"[info] backup saved at {backup_path}")
        print(f"[info] results at {results_csv}")
        if best["params"] is not None:
            print(f"[info] best summary at {best_json}")
            print(f"[info] best objective={best['objective']:.6g} params={best['params']}")
            print(f"[info] best log={best['log_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
