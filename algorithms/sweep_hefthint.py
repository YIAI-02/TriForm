#!/usr/bin/env python3
# sweep_hefthint.py
'''
 python3 sweep_hefthint.py   --mode grid   --gamma 0 0.2 0.4 0.6   --lambda_ 0 1 1.5 2 2.5 3 4   --eta 0 --objective total   --outdir ./output/sweep_hefthint_manual_2aim   --resume
'''
import argparse
import csv
import datetime as dt
import itertools
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- knobs we will edit in config.py ----
P_GAMMA = "SCHED_JOINT_LK_GAMMA"
P_LAMBDA = "SCHED_JOINT_LK_CONSIST_LAMBDA"
P_ETA = "SCHED_WEIGHT_BIAS_ETA"

# ---- parse target ----
TARGET_POLICY = "algo:hefthint"


def _fmt_num(v: float) -> str:
    """Format number for python literal writing."""
    if not math.isfinite(v):
        raise ValueError(f"non-finite value: {v}")
    # Use compact formatting; keep enough precision.
    return f"{v:.12g}"


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def replace_param_line(text: str, name: str, new_value: float) -> Tuple[str, float]:
    """
    Replace a line like:
      NAME: float = 0
      NAME = 0
    keeping any trailing comment.
    Return (new_text, old_value).
    """
    # Match whole line, capture old numeric, keep comment.
    pat = re.compile(
        rf"^(\s*{re.escape(name)}\s*(?::[^\n=]+)?=\s*)"
        rf"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        rf"(\s*(?:#.*)?)$",
        re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"在 config.py 中没找到参数行：{name}")
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


def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [float(a)]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def geomspace(a: float, b: float, n: int) -> List[float]:
    """Geometric space (log-spaced) for positive numbers."""
    if a <= 0 or b <= 0:
        raise ValueError("geomspace requires a>0 and b>0")
    if n <= 1:
        return [float(a)]
    r = (b / a) ** (1.0 / (n - 1))
    out = [a * (r ** i) for i in range(n)]
    return out


def uniq_sorted(xs: List[float], *, tol: float = 0.0) -> List[float]:
    """Unique (with optional tolerance), then sort."""
    ys = []
    for x in xs:
        keep = True
        for y in ys:
            if tol > 0 and abs(x - y) <= tol:
                keep = False
                break
            if tol == 0 and x == y:
                keep = False
                break
        if keep:
            ys.append(x)
    return sorted(ys)


def sanitize_tag(s: str) -> str:
    # safe folder name
    s = s.replace("=", "_").replace(":", "_").replace("/", "_")
    s = s.replace(" ", "")
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    return s


def run_command(script_path: Path, workdir: Path, extra_args: List[str], env: dict) -> Tuple[int, str]:
    cmd = ["bash", str(script_path)]
    cmd += extra_args
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


def parse_hefthint_metrics(log_text: str) -> Optional[Dict[str, float]]:
    """
    Parse last occurrence of algo:hefthint line:
      algo:hefthint   prefill   decode   total
    """
    # Allow leading spaces; capture 3 floats.
    pat = re.compile(
        rf"^\s*{re.escape(TARGET_POLICY)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        re.MULTILINE,
    )
    matches = list(pat.finditer(log_text))
    if not matches:
        return None
    m = matches[-1]
    prefill = float(m.group(1))
    decode = float(m.group(2))
    total = float(m.group(3))
    return {"prefill": prefill, "decode": decode, "total": total}


def load_done_keys(results_csv: Path) -> set:
    done = set()
    if not results_csv.exists():
        return done
    try:
        with results_csv.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    g = float(row["gamma"])
                    lam = float(row["lambda"])
                    eta = float(row["eta"])
                    rep = int(row.get("repeat_idx", "1"))
                    done.add((g, lam, eta, rep))
                except Exception:
                    continue
    except Exception:
        pass
    return done


def append_result(results_csv: Path, row: dict) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    with results_csv.open("a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "timestamp",
            "run_id",
            "repeat_idx",
            "gamma",
            "lambda",
            "eta",
            "objective",
            "prefill",
            "decode",
            "total",
            "returncode",
            "log_path",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Grid/random sweep for hefthint Total(s)")
    ap.add_argument("--config", default="config.py", help="path to config.py")
    ap.add_argument("--script", default="./command_single.sh", help="bash script to run")
    ap.add_argument("--workdir", default=".", help="working directory")

    # value generation
    ap.add_argument("--mode", choices=["grid", "random"], default="grid")
    ap.add_argument("--objective", choices=["total", "decode", "prefill"], default="total")

    # explicit lists (if provided, override generators)
    ap.add_argument("--gamma", type=float, nargs="*", default=None, help="explicit gamma list (0..1)")
    ap.add_argument("--lambda_", type=float, nargs="*", default=None, help="explicit lambda list (>0)")
    ap.add_argument("--eta", type=float, nargs="*", default=None, help="explicit eta list (>0)")

    # generators
    ap.add_argument("--gamma-lin", type=float, nargs=3, metavar=("START", "END", "N"),
                    default=[0.0, 1.0, 11], help="gamma linspace start end n")
    ap.add_argument("--lambda-log", type=float, nargs=3, metavar=("START", "END", "N"),
                    default=[0.1, 100.0, 7], help="lambda geomspace start end n (positive)")
    ap.add_argument("--eta-log", type=float, nargs=3, metavar=("START", "END", "N"),
                    default=[0.1, 100.0, 7], help="eta geomspace start end n (positive)")
    ap.add_argument("--include-zero", action="store_true", help="also try 0 for lambda/eta (baseline)")

    # random mode
    ap.add_argument("--trials", type=int, default=50, help="random trials (mode=random)")
    ap.add_argument("--seed", type=int, default=0)

    # misc
    ap.add_argument("--repeat", type=int, default=1, help="repeat each combo N times (reduce randomness)")
    ap.add_argument("--max-runs", type=int, default=0, help="cap total runs (0=unlimited)")
    ap.add_argument("--outdir", default="./output/sweep_hefthint", help="output directory for logs/results")
    ap.add_argument("--resume", action="store_true", help="skip combos already in results.csv")
    args, extra = ap.parse_known_args()

    config_path = Path(args.workdir) / args.config
    script_path = Path(args.workdir) / args.script
    workdir = Path(args.workdir)
    outdir = Path(args.outdir)
    results_csv = outdir / "results.csv"

    if not config_path.exists():
        print(f"[err] config not found: {config_path}", file=sys.stderr)
        sys.exit(2)
    if not script_path.exists():
        print(f"[err] script not found: {script_path}", file=sys.stderr)
        sys.exit(2)

    # Prepare candidate lists
    if args.gamma is not None and len(args.gamma) > 0:
        gammas = list(args.gamma)
    else:
        gs, ge, gn = args.gamma_lin
        gammas = linspace(gs, ge, int(gn))

    if args.lambda_ is not None and len(args.lambda_) > 0:
        lambdas = list(args.lambda_)
    else:
        ls, le, ln = args.lambda_log
        lambdas = geomspace(ls, le, int(ln))

    if args.eta is not None and len(args.eta) > 0:
        etas = list(args.eta)
    else:
        es, ee, en = args.eta_log
        etas = geomspace(es, ee, int(en))

    # clamp/clean
    gammas = [float(max(0.0, min(1.0, g))) for g in gammas]
    lambdas = [float(x) for x in lambdas]
    etas = [float(x) for x in etas]
    if args.include_zero:
        lambdas = [0.0] + lambdas
        etas = [0.0] + etas

    gammas = uniq_sorted(gammas)
    lambdas = uniq_sorted(lambdas)
    etas = uniq_sorted(etas)

    # Determine combo iterator
    if args.mode == "grid":
        combos = list(itertools.product(gammas, lambdas, etas))
    else:
        rnd = random.Random(args.seed)
        # random: gamma uniform[0,1], lambda/eta log-uniform within provided min/max
        gmin, gmax = 0.0, 1.0
        lmin, lmax = min([x for x in lambdas if x > 0.0], default=0.1), max([x for x in lambdas if x > 0.0], default=100.0)
        emin, emax = min([x for x in etas if x > 0.0], default=0.1), max([x for x in etas if x > 0.0], default=100.0)
        combos = []
        for _ in range(int(args.trials)):
            g = rnd.uniform(gmin, gmax)
            # log-uniform
            lam = math.exp(rnd.uniform(math.log(lmin), math.log(lmax)))
            eta = math.exp(rnd.uniform(math.log(emin), math.log(emax)))
            if args.include_zero and rnd.random() < 0.05:
                lam = 0.0
            if args.include_zero and rnd.random() < 0.05:
                eta = 0.0
            combos.append((g, lam, eta))

    # resume support
    done_keys = load_done_keys(results_csv) if args.resume else set()

    # backup config
    orig_text = read_text(config_path)
    backup_path = config_path.with_suffix(config_path.suffix + ".sweep_bak")
    shutil.copy2(config_path, backup_path)

    # env: reduce randomness a bit
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")

    best = {"objective": float("inf"), "gamma": None, "lambda": None, "eta": None, "log_path": None}

    run_id = 0
    try:
        for (g, lam, eta) in combos:
            # basic validity
            if not (0.0 <= g <= 1.0):
                continue
            if lam < 0 or eta < 0:
                continue

            for rep in range(1, int(args.repeat) + 1):
                if (g, lam, eta, rep) in done_keys:
                    continue

                run_id += 1
                if args.max_runs and run_id > int(args.max_runs):
                    print("[info] reached --max-runs cap, stop.")
                    return

                # update config
                update_config(config_path, gamma=g, lam=lam, eta=eta)

                tag = f"g={_fmt_num(g)}_lam={_fmt_num(lam)}_eta={_fmt_num(eta)}_r={rep}"
                run_dir = outdir / "runs" / f"{run_id:06d}_{sanitize_tag(tag)}"
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / "run.log"

                ts = dt.datetime.now().isoformat(timespec="seconds")
                print(f"\n=== [{run_id}] {ts} {tag} ===")

                rc, out = run_command(script_path, workdir=workdir, extra_args=extra, env=env)

                # write log
                with log_path.open("w", encoding="utf-8") as f:
                    f.write(f"# cmd: bash {script_path}\n")
                    if extra:
                        f.write(f"# extra: {' '.join(extra)}\n")
                    f.write(f"# time: {ts}\n")
                    f.write(f"# {P_GAMMA}={_fmt_num(g)} {P_LAMBDA}={_fmt_num(lam)} {P_ETA}={_fmt_num(eta)} repeat={rep}\n\n")
                    f.write(out)

                metrics = parse_hefthint_metrics(out)
                if metrics is None:
                    print(f"[warn] cannot find '{TARGET_POLICY}' row in log. rc={rc} log={log_path}")
                    row = {
                        "timestamp": ts,
                        "run_id": run_id,
                        "repeat_idx": rep,
                        "gamma": g,
                        "lambda": lam,
                        "eta": eta,
                        "objective": "",
                        "prefill": "",
                        "decode": "",
                        "total": "",
                        "returncode": rc,
                        "log_path": str(log_path),
                    }
                    append_result(results_csv, row)
                    if rc != 0:
                        print("[warn] command failed (non-zero). continue.")
                    continue

                obj = float(metrics[args.objective])
                print(f"[ok] hefthint prefill={metrics['prefill']:.4f} decode={metrics['decode']:.4f} total={metrics['total']:.4f}  -> objective={obj:.4f}")
                row = {
                    "timestamp": ts,
                    "run_id": run_id,
                    "repeat_idx": rep,
                    "gamma": g,
                    "lambda": lam,
                    "eta": eta,
                    "objective": obj,
                    "prefill": metrics["prefill"],
                    "decode": metrics["decode"],
                    "total": metrics["total"],
                    "returncode": rc,
                    "log_path": str(log_path),
                }
                append_result(results_csv, row)

                if obj < best["objective"]:
                    best.update({"objective": obj, "gamma": g, "lambda": lam, "eta": eta, "log_path": str(log_path)})
                    print(f"[best] objective={best['objective']:.4f} at gamma={g:.6g} lambda={lam:.6g} eta={eta:.6g}")
                    print(f"[best] log: {best['log_path']}")

    finally:
        # restore config
        write_text(config_path, orig_text)
        print(f"\n[info] restored {config_path}")
        print(f"[info] backup saved at {backup_path}")
        print(f"[info] results at {results_csv}")
        if best["gamma"] is not None:
            print(f"[info] best objective={best['objective']:.6g} gamma={best['gamma']} lambda={best['lambda']} eta={best['eta']}")
            print(f"[info] best log={best['log_path']}")


if __name__ == "__main__":
    main()
