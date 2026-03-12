from __future__ import annotations

"""
python avg_speedup.py ../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst64_rst64
"""

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional, Tuple

PAIR_RE = re.compile(r"(?:baseline_compare|best_summary)_(\d+)x(\d+)\.json$")
REP_PREFILLS = {128, 1024}
REP_DECODES = {128, 256, 512, 1024}


def parse_pair(path: Path) -> Tuple[int, int]:
    m = PAIR_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse workload pair from file name: {path}")
    return int(m.group(1)), int(m.group(2))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_total_from_results(data: dict, policy: str) -> Optional[float]:
    for item in data.get("results", []):
        if item.get("policy") == policy:
            value = item.get("total_time_s")
            if value is not None:
                return float(value)
    return None


def get_total_from_best_summary(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    data = read_json(path)
    best_times = data.get("best_times", {})
    value = best_times.get("total")
    return None if value is None else float(value)


def resolve_total_times(exp_dir: Path, baseline_path: Path) -> dict:
    data = read_json(baseline_path)
    prefill, decode = parse_pair(baseline_path)

    pd_total = get_total_from_results(data, "algo:pd")
    heft_total = get_total_from_results(data, "algo:heft")
    hefthint_total = get_total_from_results(data, "algo:hefthint")

    # 如果 baseline_compare 里缺字段，就回退到 algo_heft / algo_hefthint 下的 best_summary
    if heft_total is None:
        heft_total = get_total_from_best_summary(
            exp_dir / "algo_heft" / f"best_summary_{prefill}x{decode}.json"
        )
    if hefthint_total is None:
        hefthint_total = get_total_from_best_summary(
            exp_dir / "algo_hefthint" / f"best_summary_{prefill}x{decode}.json"
        )

    if pd_total is None or heft_total is None or hefthint_total is None:
        raise RuntimeError(
            f"Missing total_time_s for {exp_dir.name} @ {prefill}x{decode}: "
            f"pd={pd_total}, heft={heft_total}, hefthint={hefthint_total}"
        )

    if heft_total <= hefthint_total:
        this_work_total = heft_total
        this_work_method = "heft"
    else:
        this_work_total = hefthint_total
        this_work_method = "hefthint"

    return {
        "model_dir": exp_dir.name,
        "batch": data.get("config", {}).get("batch"),
        "prefill_len": prefill,
        "decode_len": decode,
        "pd_total_s": pd_total,
        "heft_total_s": heft_total,
        "hefthint_total_s": hefthint_total,
        "this_work_total_s": this_work_total,
        "this_work_method": this_work_method,
        "speedup_pd_over_this_work": pd_total / this_work_total,
        "representative_8": prefill in REP_PREFILLS and decode in REP_DECODES,
    }


def find_experiment_dirs(root: Path) -> List[Path]:
    # 情况1：root 本身就是一个实验目录
    if list(root.glob("baseline_compare_*.json")):
        return [root]

    # 情况2：root 是父目录，下面有多个实验目录
    dirs = [p for p in root.iterdir() if p.is_dir() and list(p.glob("baseline_compare_*.json"))]
    if not dirs:
        raise FileNotFoundError(
            f"No experiment directory found under {root}. "
            "Expected either baseline_compare_*.json directly under root, "
            "or subdirectories containing them."
        )
    return sorted(dirs)


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float("nan") if not values else mean(values)


def summarize(rows: List[dict]) -> dict:
    rows = sorted(rows, key=lambda x: (x["prefill_len"], x["decode_len"]))
    rep_rows = [r for r in rows if r["representative_8"]]

    return {
        "num_cases_all": len(rows),
        "avg_speedup_all": safe_mean(r["speedup_pd_over_this_work"] for r in rows),
        "ratio_of_sums_all": (
            sum(r["pd_total_s"] for r in rows) / sum(r["this_work_total_s"] for r in rows)
            if rows else float("nan")
        ),
        "num_cases_rep8": len(rep_rows),
        "avg_speedup_rep8": safe_mean(r["speedup_pd_over_this_work"] for r in rep_rows),
        "ratio_of_sums_rep8": (
            sum(r["pd_total_s"] for r in rep_rows) / sum(r["this_work_total_s"] for r in rep_rows)
            if rep_rows else float("nan")
        ),
    }


def print_report(exp_dir: Path) -> None:
    rows = [resolve_total_times(exp_dir, p) for p in sorted(exp_dir.glob("baseline_compare_*.json"))]
    rows = sorted(rows, key=lambda x: (x["prefill_len"], x["decode_len"]))
    summary = summarize(rows)

    print(f"\n=== {exp_dir} ===")
    print(
        "prefill decode  pd_total_s  heft_total_s  hefthint_total_s  "
        "this_work  this_work_total_s  speedup"
    )
    for r in rows:
        print(
            f"{r['prefill_len']:>7} {r['decode_len']:>6}  "
            f"{r['pd_total_s']:>10.6f}  "
            f"{r['heft_total_s']:>12.6f}  "
            f"{r['hefthint_total_s']:>15.6f}  "
            f"{r['this_work_method']:>9}  "
            f"{r['this_work_total_s']:>17.6f}  "
            f"{r['speedup_pd_over_this_work']:>7.4f}"
        )

    print("\nSummary")
    print(f"  #cases (all): {summary['num_cases_all']}")
    print(f"  avg speedup over all cases      = {summary['avg_speedup_all']:.4f}x")
    print(f"  ratio(sum(pd), sum(this work)) = {summary['ratio_of_sums_all']:.4f}x")
    print(f"  #cases (representative 8): {summary['num_cases_rep8']}")
    print(f"  avg speedup over rep-8 cases    = {summary['avg_speedup_rep8']:.4f}x")
    print(f"  ratio(sum(pd), sum(this work))  = {summary['ratio_of_sums_rep8']:.4f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute speedup = pd / min(heft, hefthint) from an unpacked experiment directory."
    )
    parser.add_argument(
        "root",
        type=Path,
        help=(
            "Either one experiment directory, or a parent directory containing "
            "multiple experiment directories."
        ),
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    for exp_dir in find_experiment_dirs(root):
        print_report(exp_dir)


if __name__ == "__main__":
    main()