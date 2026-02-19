#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python3 ./verify/gen_job_tsv.py \
  --input ./algorithms/output/evaluate_single_test/ \
  --output ./verify/jobs_sweep.tsv
'''
import argparse
import os
import re
import sys
from pathlib import Path

TRACE_RE = re.compile(
    r"^(?P<algo>.+?)_+prefill-(?P<prefill>\d+)xdecode_(?P<decode_len>\d+)_(?P<trace>ops|comms)_trace\.csv$"
)

STRIDE_RE = re.compile(r"(?:^|_)s(?P<stride>\d+)$")
DTYPE_RE = re.compile(r"^(?P<model>.+?)_(?:int|fp|bf)\d+", re.IGNORECASE)
BATCH_RE = re.compile(r"^(?P<model>.+?)_b\d+", re.IGNORECASE)
SIZE_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?[bkBK]$")

def to_repo_rel_posix(p: Path, repo_root: Path) -> str:
    rp = os.path.relpath(str(p), start=str(repo_root))
    rp = rp.replace(os.sep, "/")
    if not rp.startswith("."):
        rp = "./" + rp
    return rp


def infer_out_dir_relpath(rel_to_eval: Path) -> Path:
    parts = rel_to_eval.parts
    algo_idx = None
    for i, part in enumerate(parts):
        if part.startswith("algo_") or part.startswith("algo-"):
            algo_idx = i
            break

    if algo_idx is None:
        if len(parts) >= 3:
            algo_idx = len(parts) - 2
        else:
            algo_idx = max(0, len(parts) - 1)

    return Path(*parts[:algo_idx])


def extract_decode_stride(rel_to_eval: Path) -> int | None:
    for part in rel_to_eval.parts[:-1]:
        m = STRIDE_RE.search(part)
        if m:
            return int(m.group("stride"))
    return None


def infer_model_name_from_dir(model_dir_name: str) -> str | None:
    """Infer model name from the model directory name under evaluate_single_test.

    Heuristics (in order):
      1) <model>_(int|fp|bf)\d+...  -> <model>
      2) <model>_b\d+...            -> <model>
      3) first two tokens if token2 looks like a size (e.g. 7b, 13b, 1.3b)
      4) fallback: first token
    """
    name = str(model_dir_name)
    m = DTYPE_RE.match(name)
    if m:
        return m.group("model")
    m = BATCH_RE.match(name)
    if m:
        return m.group("model")

    parts = [p for p in name.split("_") if p]
    if len(parts) >= 2 and SIZE_TOKEN_RE.match(parts[1]):
        return "_".join(parts[:2])
    if parts:
        return parts[0]
    return None


def find_model_cfg_path(repo_root: Path, model_name: str) -> Path | None:
    """Find ./configs/<model>*.json for the given model name."""
    cfg_dir = repo_root / "configs"
    if not cfg_dir.exists():
        return None

    # preferred naming conventions
    for cand in [cfg_dir / f"{model_name}_shape.json", cfg_dir / f"{model_name}.json"]:
        if cand.exists():
            return cand

    # fallback: any json starting with model_name
    cands = sorted(cfg_dir.glob(f"{model_name}*.json"))
    return cands[0] if cands else None

def main():
    ap = argparse.ArgumentParser(
        description="Scan evaluate_single_test top dir and generate job.tsv (ops/comms pairs)."
    )
    ap.add_argument(
        "--input",
        "-i",
        required=True,
    )
    ap.add_argument(
        "--output",
        "-o",
        default="job.tsv",
    )
    ap.add_argument(
        "--verify-base",
        default=None,
        help=(
            "defualt./verify/<input_dir_name> "
        ),
    )
    args = ap.parse_args()

    eval_root = Path(args.input).resolve()
    if not eval_root.exists() or not eval_root.is_dir():
        print(f"[ERR] --input is not path: {eval_root}", file=sys.stderr)
        sys.exit(2)

    repo_root = Path.cwd().resolve()

    verify_base = (
        Path(args.verify_base).resolve()
        if args.verify_base
        else (repo_root / "verify" / eval_root.name).resolve()
    )

    rows = []
    missing_pairs = 0
    unmatched = 0
    missing_stride = 0

    for ops_path in eval_root.rglob("*_ops_trace.csv"):
        m = TRACE_RE.match(ops_path.name)
        if not m:
            unmatched += 1
            continue

        algo = m.group("algo").rstrip("_")
        prefill_len = int(m.group("prefill"))
        decode_len = int(m.group("decode_len"))

        comms_name = re.sub(r"_ops_trace\.csv$", "_comms_trace.csv", ops_path.name)
        comms_path = ops_path.with_name(comms_name)

        if not comms_path.exists():
            missing_pairs += 1
            print(f"[WARN] 找不到配对 comms_trace: {comms_path}", file=sys.stderr)
            continue

        rel_to_eval = ops_path.relative_to(eval_root)

        decode_stride = extract_decode_stride(rel_to_eval)
        if decode_stride is None:
            missing_stride += 1
            decode_stride = decode_len
            print(
                f"[WARN] can not find *_s<stride>, fallback to decode_len={decode_len} as decode_stride: {rel_to_eval}",
                file=sys.stderr,
            )

        # out_dir = verify_base / <hardware>/<model_dir>
        out_dir_rel = infer_out_dir_relpath(rel_to_eval)
        out_dir_abs = verify_base / out_dir_rel

        model_dir_name = out_dir_rel.parts[-1] if out_dir_rel.parts else ""
        model_name = infer_model_name_from_dir(model_dir_name) or ""
        cfg_path = find_model_cfg_path(repo_root, model_name) if model_name else None
        if cfg_path is None:
            print(
                f"[WARN] cannot locate model cfg under ./configs for model={model_name!r} (from dir={model_dir_name!r}); set cfg='-' in tsv",
                file=sys.stderr,
            )
            cfg_rel = "-"
        else:
            cfg_rel = to_repo_rel_posix(cfg_path, repo_root)

        rows.append(
            {
                "schedule_csv": to_repo_rel_posix(ops_path, repo_root),
                "comms_csv": to_repo_rel_posix(comms_path, repo_root),
                "prefix": f"{algo}_{prefill_len}x{decode_len}",
                "out_dir": to_repo_rel_posix(out_dir_abs, repo_root),
                "prefill_len": str(prefill_len),
                "decode_stride": str(decode_stride),
                "cfg": cfg_rel,
            }
        )
    rows.sort(key=lambda r: (r["out_dir"], r["prefix"], r["schedule_csv"]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["schedule_csv", "comms_csv", "prefix", "out_dir", "prefill_len", "decode_stride", "cfg"]
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# " + "\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r[h] for h in header) + "\n")

    print(f"[OK] write: {out_path}  (rows={len(rows)})", file=sys.stderr)
    if missing_pairs:
        print(f"[INFO] skip {missing_pairs} ops_trace (missing comms_trace match)", file=sys.stderr)
    if unmatched:
       print(f"[INFO] have {unmatched} *_ops_trace.csv unmatched csv", file=sys.stderr)
    if missing_stride:
        print(f"[INFO] cannot find {missing_stride} *_s<stride>, fallback to decode_len as stride", file=sys.stderr)


if __name__ == "__main__":
    main()
