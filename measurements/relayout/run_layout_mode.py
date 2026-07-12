#!/usr/bin/env python3
"""
export PROJECT_ROOT=/path/to/TriForm
export RAMU_REPO=${PROJECT_ROOT}/src
export RAMU_EXE=${PROJECT_ROOT}/src/ramulator2
export RELAYOUT_DIR=${PROJECT_ROOT}/measurements/relayout
export LD_LIBRARY_PATH="${PROJECT_ROOT}/submodules/CENT/aim_simulator${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python3 ${RELAYOUT_DIR}/run_layout_mode.py \
  --mode pimopt_rw_host \
  --repo ${RAMU_REPO} \
  --exe ${RAMU_EXE} \
  --config ${RELAYOUT_DIR}/linear_to_pimopt_gddr6.yaml \
  --sweep 512x512 4096x4096 4096x14336 14336x4096 \
  --workdir ${RELAYOUT_DIR}/runs_pimopt_rw_host_sweep

mode
    linear_to_pimopt 
    pimopt_rw_dma
    pimopt_rw_host
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

from gen_layout_mode_trace import PRESETS, TraceConfig, generate_trace

DEFAULT_TCK_PS = 500.0


def parse_simple_stats(text: str) -> dict[str, int | float | str]:
    stats: dict[str, int | float | str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_]+):\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s*$", line)
        if m:
            k, v = m.group(1), m.group(2)
            stats[k] = float(v) if "." in v else int(v)
    return stats


def parse_shape(s: str) -> tuple[int, int]:
    try:
        r, c = s.lower().split("x")
        rows, cols = int(r), int(c)
        if rows <= 0 or cols <= 0:
            raise ValueError
        return rows, cols
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"invalid shape '{s}', expected ROWSxCOLS") from exc


def run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )


def resolve_repo_dir_and_exe(repo_arg: Path, exe_arg: Path) -> tuple[Path, Path]:
    repo_path = repo_arg.resolve()
    exe_path = exe_arg.resolve()

    if repo_path.is_file():
        repo_dir = repo_path.parent
    else:
        repo_dir = repo_path

    if not repo_dir.exists() or not repo_dir.is_dir():
        raise FileNotFoundError(f"repo/cwd directory not found: {repo_dir}")
    if not exe_path.exists() or not exe_path.is_file():
        raise FileNotFoundError(f"executable not found: {exe_path}")

    return repo_dir, exe_path


def resolve_single_shape(args: argparse.Namespace) -> tuple[int, int]:
    if args.rows is not None and args.cols is not None:
        return args.rows, args.cols
    return PRESETS[args.preset]


def resolve_sweep_shapes(args: argparse.Namespace) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []

    if args.shape:
        shapes.extend(args.shape)
    if args.preset_list:
        shapes.extend(PRESETS[name] for name in args.preset_list)

    if not shapes:
        shapes.append(resolve_single_shape(args))

    deduped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for shape in shapes:
        if shape not in seen:
            deduped.append(shape)
            seen.add(shape)
    return deduped


def make_cfg(args: argparse.Namespace, rows: int, cols: int) -> TraceConfig:
    return TraceConfig(
        rows=rows,
        cols=cols,
        dtype_bytes=args.dtype_bytes,
        burst_bytes=args.burst_bytes,
        channels=args.channels,
        banks=args.banks,
        gpr_count=args.gpr_count,
    )


def make_tag(mode: str, rows: int, cols: int, dtype_bytes: int, custom_tag: str | None = None) -> str:
    if custom_tag:
        return custom_tag
    return f"{mode}_{rows}x{cols}_fp{dtype_bytes * 8}"


def run_one(
    *,
    mode: str,
    repo_dir: Path,
    exe: Path,
    config: Path,
    tck_ps: float,
    workdir: Path,
    tag: str,
    cfg: TraceConfig,
    sweep_index: int | None = None,
    sweep_count: int | None = None,
    skip_sim: bool = False,
) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    run_dir = workdir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_path = run_dir / f"{mode}.trace"
    stdout_path = run_dir / "sim_stdout.txt"
    summary_path = run_dir / "summary.json"

    generate_trace(cfg, trace_path, mode)

    summary: dict[str, object] = {
        "mode": mode,
        "shape": [cfg.rows, cfg.cols],
        "dtype_bytes": cfg.dtype_bytes,
        "matrix_bytes": cfg.matrix_bytes,
        "num_bursts": cfg.num_bursts,
        "burst_bytes": cfg.burst_bytes,
        "channels": cfg.channels,
        "banks": cfg.banks,
        "trace_path": str(trace_path),
        "config_path": str(config.resolve()),
        "repo": str(repo_dir),
        "exe": str(exe),
        "tCK_ps": tck_ps,
        "payload_bytes": cfg.matrix_bytes,
        "dram_traffic_bytes": 2 * cfg.matrix_bytes,
    }
    if sweep_index is not None and sweep_count is not None:
        summary["sweep_index"] = sweep_index
        summary["sweep_count"] = sweep_count

    if skip_sim:
        summary["sim_skipped"] = True
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    sim = run_cmd(
        [str(exe), "-f", str(config.resolve()), "-t", str(trace_path)],
        cwd=exe.parent,
    )
    stdout_path.write_text(
        sim.stdout + ("\n[stderr]\n" + sim.stderr if sim.stderr else ""),
        encoding="utf-8",
    )
    if sim.returncode != 0:
        raise RuntimeError(f"simulation failed for {tag}; see {stdout_path}\n{sim.stderr}")

    stats = parse_simple_stats(sim.stdout)
    if "memory_system_cycles" not in stats:
        raise RuntimeError(f"cannot find memory_system_cycles for {tag}; see {stdout_path}")

    total_cycles = int(stats["memory_system_cycles"])
    total_time_s = total_cycles * tck_ps * 1e-12
    total_payload_bytes = cfg.matrix_bytes
    total_dram_bytes = 2 * cfg.matrix_bytes

    summary.update(
        {
            "memory_system_cycles": total_cycles,
            "total_time_s": total_time_s,
            "total_time_ms": total_time_s * 1e3,
            "payload_equivalent_bandwidth_GBps": total_payload_bytes / total_time_s / 1e9,
            "dram_traffic_equivalent_bandwidth_GBps": total_dram_bytes / total_time_s / 1e9,
            "raw_stats_subset": {
                k: v
                for k, v in stats.items()
                if k.startswith("memory_system_cycles") or k.startswith("total_num_")
            },
        }
    )

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_single_summary(summary: dict) -> None:
    print(json.dumps(summary, indent=2))


def print_sweep_progress(summary: dict) -> None:
    shape = summary["shape"]
    prefix = ""
    if "sweep_index" in summary and "sweep_count" in summary:
        prefix = f"[{summary['sweep_index']}/{summary['sweep_count']}] "

    if summary.get("sim_skipped"):
        print(f"{prefix}{shape[0]}x{shape[1]}: trace generated")
        return

    print(
        f"{prefix}{shape[0]}x{shape[1]}: "
        f"time={summary['total_time_ms']:.6f} ms, "
        f"payload_bw={summary['payload_equivalent_bandwidth_GBps']:.3f} GB/s, "
        f"dram_bw={summary['dram_traffic_equivalent_bandwidth_GBps']:.3f} GB/s"
    )


def build_aggregate(per_run: list[dict], mode: str, tck_ps: float) -> dict:
    simulated_runs = [x for x in per_run if not x.get("sim_skipped")]
    agg: dict[str, object] = {
        "mode": mode,
        "num_runs": len(per_run),
        "num_simulated_runs": len(simulated_runs),
        "tCK_ps": tck_ps,
        "shapes": [x["shape"] for x in per_run],
        "per_run": per_run,
    }

    if not simulated_runs:
        agg["aggregate"] = {"sim_skipped": True}
        return agg

    total_payload = sum(int(x["payload_bytes"]) for x in simulated_runs)
    total_dram = sum(int(x["dram_traffic_bytes"]) for x in simulated_runs)
    total_time_s = sum(float(x["total_time_s"]) for x in simulated_runs)

    agg["aggregate"] = {
        "total_payload_bytes": total_payload,
        "total_dram_traffic_bytes": total_dram,
        "total_time_s": total_time_s,
        "total_time_ms": total_time_s * 1e3,
        "aggregate_payload_equivalent_bandwidth_GBps": total_payload / total_time_s / 1e9,
        "aggregate_dram_traffic_equivalent_bandwidth_GBps": total_dram / total_time_s / 1e9,
        "arithmetic_mean_payload_bandwidth_GBps": statistics.mean(
            float(x["payload_equivalent_bandwidth_GBps"]) for x in simulated_runs
        ),
        "arithmetic_mean_dram_bandwidth_GBps": statistics.mean(
            float(x["dram_traffic_equivalent_bandwidth_GBps"]) for x in simulated_runs
        ),
        "median_payload_bandwidth_GBps": statistics.median(
            float(x["payload_equivalent_bandwidth_GBps"]) for x in simulated_runs
        ),
        "median_dram_bandwidth_GBps": statistics.median(
            float(x["dram_traffic_equivalent_bandwidth_GBps"]) for x in simulated_runs
        ),
    }
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["linear_to_pimopt", "pimopt_rw_host", "pimopt_rw_dma"], required=True)
    ap.add_argument("--repo", type=Path, required=True, help="directory to use as cwd for ramulator")
    ap.add_argument("--exe", type=Path, required=True, help="compiled ramulator2 executable")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("linear_to_pimopt_gddr6.yaml"))
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), default="demo")
    ap.add_argument("--preset-list", choices=sorted(PRESETS.keys()), nargs="*", default=None,
                    help="optional sweep presets, e.g. --preset-list demo llama3_proj llama3_ffn_up")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--shape", "--sweep", dest="shape", type=parse_shape, nargs="*", default=None,
                    help="optional multi-shape sweep, e.g. --shape 512x512 4096x4096 4096x14336 or --sweep ...")
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--burst-bytes", type=int, default=32)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--banks", type=int, default=16)
    ap.add_argument("--gpr-count", type=int, default=16)
    ap.add_argument("--tck-ps", type=float, default=DEFAULT_TCK_PS)
    ap.add_argument("--workdir", type=Path, default=Path("./layout_mode_runs"))
    ap.add_argument("--tag", type=str, default="", help="single-run custom tag; ignored during multi-shape sweep")
    ap.add_argument("--aggregate-name", type=str, default="batch_summary.json")
    ap.add_argument("--skip-sim", action="store_true", help="only generate trace(s), do not launch ramulator")
    args = ap.parse_args()

    repo_dir, exe = resolve_repo_dir_and_exe(args.repo, args.exe)
    config = args.config.resolve()
    if not config.exists():
        raise FileNotFoundError(f"config not found: {config}")

    shapes = resolve_sweep_shapes(args)
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    is_sweep = len(shapes) > 1 or bool(args.shape) or bool(args.preset_list)
    per_run: list[dict] = []

    for idx, (rows, cols) in enumerate(shapes, start=1):
        cfg = make_cfg(args, rows, cols)
        custom_tag = args.tag if not is_sweep else None
        tag = make_tag(args.mode, rows, cols, cfg.dtype_bytes, custom_tag)
        summary = run_one(
            mode=args.mode,
            repo_dir=repo_dir,
            exe=exe,
            config=config,
            tck_ps=args.tck_ps,
            workdir=workdir,
            tag=tag,
            cfg=cfg,
            sweep_index=idx if is_sweep else None,
            sweep_count=len(shapes) if is_sweep else None,
            skip_sim=args.skip_sim,
        )
        per_run.append(summary)

        if is_sweep:
            print_sweep_progress(summary)
        else:
            print_single_summary(summary)

    if is_sweep:
        agg = build_aggregate(per_run, args.mode, args.tck_ps)
        out_json = workdir / args.aggregate_name
        out_json.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print("\n=== aggregate ===")
        print(json.dumps(agg["aggregate"], indent=2))
        print(f"\nwritten: {out_json}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
