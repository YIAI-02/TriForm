#!/usr/bin/env python3
"""Batch runner for speedup.py over sweep_models_lens outputs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Sequence, Tuple

ModelEntry = Tuple[str, str, Path]


def _iter_model_dirs(output_root: Path) -> Iterator[ModelEntry]:
    """Yield (hardware, stride, model_dir) triples that contain algo_* subfolders."""
    for hw_dir in sorted(p for p in output_root.glob("hw_*") if p.is_dir()):
        for stride_dir in sorted(p for p in hw_dir.iterdir() if p.is_dir()):
            for model_dir in sorted(p for p in stride_dir.iterdir() if p.is_dir()):
                if any(p.is_dir() for p in model_dir.glob("algo_*")):
                    yield (hw_dir.name, stride_dir.name, model_dir)


def _build_command(
    *,
    python_bin: str,
    speedup_script: Path,
    model_dir: Path,
    outfile: Path,
    ncols: int,
    sharey: bool,
    algos: Sequence[str],
) -> Sequence[str]:
    cmd = [
        python_bin,
        str(speedup_script),
        "--grid-best",
        "--root",
        str(model_dir),
        "--ncols",
        str(ncols),
        "--outfile",
        str(outfile),
    ]
    if sharey:
        cmd.append("--sharey")
    if algos:
        cmd.extend(["--algos", ",".join(algos)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(__file__).resolve().parents[1] / "algorithms" / "output" / "lens_eval_sweep"
    parser.add_argument("--output-root", type=Path, default=default_output,
                        help="Root produced by sweep_models_lens (default: %(default)s)")
    parser.add_argument("--speedup-script", type=Path, default=Path(__file__).parent / "speedup.py",
                        help="Path to speedup.py (default: %(default)s)")
    parser.add_argument("--python-bin", type=str, default=sys.executable,
                        help="Python interpreter used to invoke speedup.py")
    parser.add_argument("--ncols", type=int, default=2,
                        help="Number of subplot columns for each figure")
    parser.add_argument("--sharey", dest="sharey", action="store_true",
                        help="Share Y axis between subplots (default)")
    parser.add_argument("--no-sharey", dest="sharey", action="store_false",
                        help="Disable Y axis sharing")
    parser.set_defaults(sharey=True)
    parser.add_argument("--algos", type=str, default=None,
                        help="Comma separated algo names to forward to speedup.py (optional)")
    parser.add_argument("--outfile-name", type=str, default="speedup_grid.pdf",
                        help="Filename placed inside each model_dir for the generated figure")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    speedup_script = args.speedup_script.resolve()
    if not speedup_script.exists():
        raise SystemExit(f"speedup.py not found: {speedup_script}")

    algos = [] if not args.algos else [s.strip() for s in args.algos.split(',') if s.strip()]

    print(f"Scanning for model results under: {output_root}")
    found_any = False
    for hw_name, stride_name, model_dir in _iter_model_dirs(output_root):
        found_any = True
        outfile = (model_dir / args.outfile_name).resolve()
        outfile.parent.mkdir(parents=True, exist_ok=True)
        cmd = _build_command(
            python_bin=args.python_bin,
            speedup_script=speedup_script,
            model_dir=model_dir,
            outfile=outfile,
            ncols=args.ncols,
            sharey=args.sharey,
            algos=algos,
        )
        pretty_cmd = " ".join(cmd)
        print(f"[HW={hw_name}][{stride_name}][{model_dir.name}] -> {outfile.name}")
        if args.dry_run:
            print(f"  DRY-RUN: {pretty_cmd}")
            continue
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as err:
            print(f"  ERROR: Command failed with code {err.returncode}")
            raise

    if not found_any:
        print("No eligible model directories found (missing algo_* folders?).")


if __name__ == "__main__":
    main()
