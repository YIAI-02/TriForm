#!/usr/bin/env python3
# run_bifocal_one.py
"""Run one Bifocal config (gamma/lambda/eta) once.

This is designed to be launched many times in parallel by a shell script.

Typical usage (inside an isolated workdir / per-job checkout):
  python3 run_bifocal.py \
    --config config.py \
    --script ./commands/command_single_evaluate.sh \
    --workdir . \
    --gamma 0.2 --lambda_ 2 --eta 0 \
    --objective total \
    --outdir ./output/sweep_bifocal_parallel

Notes:
- IMPORTANT: do NOT run multiple instances in the same workdir, because it edits config.py.
  Use per-job copies (rsync/cp/git-worktree) to avoid races.
- It will write per-run artifacts under: <outdir>/runs/<run_id>_<tag>/
  including: run.log, result.json
"""

import argparse
import datetime as dt
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_COMMAND_SCRIPT = PROJECT_ROOT / "commands" / "command_single_evaluate.sh"

# ---- knobs we will edit in config.py ----
P_GAMMA = "SCHED_JOINT_LK_GAMMA"
P_LAMBDA = "SCHED_JOINT_LK_CONSIST_LAMBDA"
P_ETA = "SCHED_WEIGHT_BIAS_ETA"

# ---- parse target ----
TARGET_POLICY = "algo:Bifocal"


def _fmt_num(v: float) -> str:
    """Format number for python literal writing."""
    if not math.isfinite(v):
        raise ValueError(f"non-finite value: {v}")
    return f"{v:.12g}"


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def replace_param_line(text: str, name: str, new_value: float) -> Tuple[str, float]:
    """Replace a line like 'NAME = 0' (or with annotation) keeping trailing comment."""
    pat = re.compile(
        rf"^(\s*{re.escape(name)}\s*(?::[^\n=]+)?=\s*)"
        rf"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        rf"(\s*(?:#.*)?)$",
        re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"Parameter line not found in config.py: {name}")
    old_val = float(m.group(2))
    new_text = pat.sub(lambda mm: f"{mm.group(1)}{_fmt_num(new_value)}{mm.group(3)}", text, count=1)
    return new_text, old_val


def update_config(config_path: Path, gamma: float, lam: float, eta: float) -> Dict[str, float]:
    """Update 3 params in config.py, return old values."""
    txt = read_text(config_path)
    txt, old_gamma = replace_param_line(txt, P_GAMMA, gamma)
    txt, old_lam = replace_param_line(txt, P_LAMBDA, lam)
    txt, old_eta = replace_param_line(txt, P_ETA, eta)
    write_text(config_path, txt)
    return {P_GAMMA: old_gamma, P_LAMBDA: old_lam, P_ETA: old_eta}


def sanitize_tag(s: str) -> str:
    # safe folder name
    s = s.replace("=", "_").replace(":", "_").replace("/", "_")
    s = s.replace(" ", "")
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    return s


def run_command(script_path: Path, workdir: Path, extra_args: List[str], env: dict) -> Tuple[int, str]:
    cmd = ["bash", str(script_path)] + list(extra_args)
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


def parse_bifocal_metrics(log_text: str) -> Optional[Dict[str, float]]:
    """Parse last occurrence of:
        algo:Bifocal   prefill   decode   total
    """
    pat = re.compile(
        rf"^\s*{re.escape(TARGET_POLICY)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        re.MULTILINE,
    )
    matches = list(pat.finditer(log_text))
    if not matches:
        return None
    m = matches[-1]
    return {
        "prefill": float(m.group(1)),
        "decode": float(m.group(2)),
        "total": float(m.group(3)),
    }


def _default_run_id() -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    rnd = random.randint(0, 99999)
    return f"{ts}_pid{pid}_{rnd:05d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one Bifocal config once (for parallel launching).")
    ap.add_argument("--config", default="config.py", help="path to config.py")
    ap.add_argument("--script", default=str(DEFAULT_COMMAND_SCRIPT), help="bash script to run")
    ap.add_argument("--workdir", default=str(SCRIPT_DIR), help="working directory (where command script runs)")

    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--lambda_", type=float, required=True, help="consist lambda")
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--repeat-idx", type=int, default=1)

    ap.add_argument("--objective", choices=["total", "decode", "prefill"], default="total")
    ap.add_argument("--outdir", default=str(PROJECT_ROOT / "output" / "sweep_bifocal_parallel"), help="output root dir")
    ap.add_argument("--run-id", default="", help="optional run id; default uses timestamp+pid")
    ap.add_argument("--tag", default="", help="optional tag for folder name")
    ap.add_argument("--no-restore", action="store_true", help="do not restore config.py (NOT recommended)")
    args, extra = ap.parse_known_args()

    workdir = Path(args.workdir).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (workdir / config_path).resolve()
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = (workdir / script_path).resolve()
    outdir = Path(args.outdir).resolve()

    if not config_path.exists():
        print(f"[err] config not found: {config_path}", file=sys.stderr)
        sys.exit(2)
    if not script_path.exists():
        print(f"[err] script not found: {script_path}", file=sys.stderr)
        sys.exit(2)

    g = float(args.gamma)
    lam = float(args.lambda_)
    eta = float(args.eta)
    rep = int(args.repeat_idx)

    # basic validity
    if not (0.0 <= g <= 1.0):
        print(f"[err] gamma must be within [0,1], got {g}", file=sys.stderr)
        sys.exit(2)
    if lam < 0 or eta < 0:
        print(f"[err] lambda_/eta must be >=0, got lambda_={lam}, eta={eta}", file=sys.stderr)
        sys.exit(2)

    run_id = args.run_id.strip() or _default_run_id()
    tag = args.tag.strip() or f"g={_fmt_num(g)}_lam={_fmt_num(lam)}_eta={_fmt_num(eta)}_r={rep}"

    run_dir = outdir / "runs" / f"{sanitize_tag(run_id)}_{sanitize_tag(tag)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    result_path = run_dir / "result.json"

    # backup config (per-job)
    orig_text = read_text(config_path)
    backup_path = config_path.with_suffix(config_path.suffix + ".one_bak")
    try:
        shutil.copy2(config_path, backup_path)
    except Exception:
        # backup is best-effort
        pass

    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    # expose params to command_single.sh (optional; harmless if unused)
    env["SWEEP_GAMMA"] = _fmt_num(g)
    env["SWEEP_LAMBDA"] = _fmt_num(lam)
    env["SWEEP_ETA"] = _fmt_num(eta)
    env["SWEEP_REPEAT_IDX"] = str(rep)
    env["SWEEP_RUN_DIR"] = str(run_dir)
    env["SWEEP_OUTDIR"] = str(outdir)

    ts = dt.datetime.now().isoformat(timespec="seconds")

    rc = 1
    out = ""
    metrics: Optional[Dict[str, float]] = None

    try:
        update_config(config_path, gamma=g, lam=lam, eta=eta)

        print(f"=== [{run_id}] {ts} {tag} ===")
        rc, out = run_command(script_path, workdir=workdir, extra_args=extra, env=env)

        # write log
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"# cmd: bash {script_path}\n")
            if extra:
                f.write(f"# extra: {' '.join(extra)}\n")
            f.write(f"# time: {ts}\n")
            f.write(
                f"# {P_GAMMA}={_fmt_num(g)} {P_LAMBDA}={_fmt_num(lam)} {P_ETA}={_fmt_num(eta)} repeat={rep}\n\n"
            )
            f.write(out)

        metrics = parse_bifocal_metrics(out)

        if metrics is None:
            print(f"[warn] cannot find '{TARGET_POLICY}' row in log. rc={rc} log={log_path}")
        else:
            obj = float(metrics[args.objective])
            print(
                f"[ok] Bifocal prefill={metrics['prefill']:.4f} "
                f"decode={metrics['decode']:.4f} total={metrics['total']:.4f} -> objective({args.objective})={obj:.4f}"
            )

        result = {
            "timestamp": ts,
            "run_id": run_id,
            "repeat_idx": rep,
            "gamma": g,
            "lambda": lam,
            "eta": eta,
            "objective_name": args.objective,
            "prefill": (metrics or {}).get("prefill"),
            "decode": (metrics or {}).get("decode"),
            "total": (metrics or {}).get("total"),
            "returncode": rc,
            "log_path": str(log_path),
            "workdir": str(workdir),
            "script": str(script_path),
            "config": str(config_path),
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Also print a single-line JSON to stdout for easy grep/collect
        print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False))

    finally:
        if not args.no_restore:
            try:
                write_text(config_path, orig_text)
            except Exception as e:
                print(f"[warn] failed to restore {config_path}: {e}", file=sys.stderr)

    # propagate command return code
    sys.exit(rc)


if __name__ == "__main__":
    main()
