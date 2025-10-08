#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, csv, json, subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from aim_shared import load_model_shape

def shlex_join(cmd: List[str]) -> str:
    def q(x: str) -> str:
        if any(c in x for c in ' "\'\\'):
            return '"' + x.replace('"', r'\"') + '"'
        return x
    return " ".join(q(x) for x in cmd)

def run_cmd(cmd: List[str], cwd: Path | None = None) -> None:
    print(f"[CMD] {shlex_join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(cwd) if cwd else None).returncode
    if rc != 0:
        raise RuntimeError(f"Command failed: {rc} :: {shlex_join(cmd)}")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s: return []
    out: List[int] = []
    for tok in s.split(","):
        t = tok.strip()
        if not t: continue
        out.append(int(t))
    return sorted(set(out))

def merge_csvs(csv_files: List[Path], out_csv: Path) -> None:
    headers: List[str] = []
    rows: List[dict] = []
    for p in csv_files:
        if not p.exists(): continue
        with p.open("r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            if not headers:
                headers = rd.fieldnames or []
            rows.extend(list(rd))
    if not rows:
        raise RuntimeError("No rows to merge")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] merged -> {out_csv}")

def sample_pairs(vs: List[int], ns: List[int], limit: int | None) -> Tuple[List[int], List[int]]:
    pairs = [(v,n) for v in vs for n in ns]
    if not limit or len(pairs) <= limit: return vs, ns
    step = max(1, len(pairs)//limit)
    s = pairs[::step][:limit]
    return sorted(set(v for v,_ in s)), sorted(set(n for _,n in s))

def call_01(py01: Path, pim_cfg: Path, out_dir: Path, ops: str, seqlens, v_dims, m_cols, with_af, dim, n_heads, n_kv_heads, model_shape):
    cmd = [sys.executable, str(py01),
           "--pim-config", str(pim_cfg), "--ops", ops, "--out-dir", str(out_dir)]
    if seqlens: cmd += ["--seqlens", ",".join(map(str, seqlens))]
    if v_dims: cmd += ["--vector-dims", ",".join(map(str, v_dims))]
    if m_cols: cmd += ["--matrix-cols", ",".join(map(str, m_cols))]
    if with_af: cmd += ["--with-af"]
    if dim is not None: cmd += ["--dim", str(dim)]
    if n_heads is not None: cmd += ["--n-heads", str(n_heads)]
    if n_kv_heads is not None: cmd += ["--n-kv-heads", str(n_kv_heads)]
    if model_shape is not None: cmd += ["--model-shape", str(model_shape)]
    run_cmd(cmd)

def call_02(py02: Path, traces_dir: Path, ram_bin: Path, ram_cfg: Path, out_csv: Path, metric_regex: str | None, extra_args: str | None, cmd_template: str | None):
    cmd = [sys.executable, str(py02), "--traces-dir", str(traces_dir),
           "--ramulator-bin", str(ram_bin), "--config", str(ram_cfg), "--out-csv", str(out_csv)]
    if metric_regex: cmd += ["--metric-regex", metric_regex]
    if extra_args: cmd += ["--extra-args", extra_args]
    if cmd_template: cmd += ["--cmd-template", cmd_template]
    run_cmd(cmd)

def call_03_fit(py03: Path, results_csv: Path, out_model: Path, out_summary_csv: Path):
    cmd = [sys.executable, str(py03), "fit", "--results-csv", str(results_csv), "--out", str(out_model), "--summary-csv", str(out_summary_csv)]
    run_cmd(cmd)

def call_03_predict(py03: Path, model_json: Path, op: str, L, V, N, H, model_shape: Path | None):
    cmd = [sys.executable, str(py03), "predict", "--model", str(model_json), "--op", op]
    if L is not None: cmd += ["--seqlen", str(L)]
    if V is not None: cmd += ["--vector-dim", str(V)]
    if N is not None: cmd += ["--matrix-col", str(N)]
    if H is not None: cmd += ["--n-heads", str(H)]
    if model_shape is not None and H is None:
        cmd += ["--model-shape", str(model_shape)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout); print(out.stderr, file=sys.stderr)
        raise RuntimeError("predict failed")
    s = out.stdout.strip().splitlines()[-1].strip()
    import re as _re
    m = _re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    return float(m.group(1)) if m else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pim-config", type=Path, required=True)
    ap.add_argument("--ramulator-bin", type=Path, required=True)
    ap.add_argument("--ramulator-config", type=Path, required=True)
    ap.add_argument("--model-shape", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("auto_run_") / datetime.now().strftime("%Y%m%d_%H%M%S"))

    ap.add_argument("--seqlens", type=str, default="16,32,64,128,256,384,512,768,1024,1536,2048,3072,4096")
    ap.add_argument("--vector-dims", type=str, default="64,96,128,192,256,384,512,768,1024")
    ap.add_argument("--matrix-cols", type=str, default="64,128,192,256,384,512,768,1024,1536,2048,3072,4096")
    ap.add_argument("--max-weight-combos", type=int, default=0)

    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--n-kv-heads", type=int, default=None)

    ap.add_argument("--metric-regex", type=str, default=None)
    ap.add_argument("--extra-args", type=str, default=None)
    ap.add_argument("--cmd-template", type=str, default=None)

    ap.add_argument("--light-grid", action="store_true")
    ap.add_argument("--gen-predictions", action="store_true")

    here = Path(__file__).resolve().parent
    ap.add_argument("--py01", type=Path, default=here / "01_gentrace.py")
    ap.add_argument("--py02", type=Path, default=here / "02_run_ramulator.py")
    ap.add_argument("--py03", type=Path, default=here / "03_fit_latency_model.py")

    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir.resolve())

    shape = load_model_shape(args.model_shape) if args.model_shape else None
    dim = args.dim if args.dim is not None else (shape["dim"] if shape else 256)
    n_heads = args.n_heads if args.n_heads is not None else (shape["n_heads"] if shape else 8)
    n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else (shape["n_kv_heads"] if shape else n_heads)

    if args.light_grid:
        seqlens = parse_int_list("16,32,64,128,256,512,1024")
        v_dims  = parse_int_list("64,128,256,512,768,1024")
        m_cols  = parse_int_list("64,128,256,512,1024,2048,4096")
    else:
        seqlens = parse_int_list(args.seqlens)
        v_dims  = parse_int_list(args.vector_dims)
        m_cols  = parse_int_list(args.matrix_cols)

    if shape and shape.get("seq_length"):
        Lmax = int(shape["seq_length"])
        seqlens = [L for L in seqlens if L <= Lmax] or [min(512, Lmax)]

    if args.max_weight_combos and args.max_weight_combos > 0:
        v_dims, m_cols = sample_pairs(v_dims, m_cols, args.max_weight_combos)

    (out_dir / "args.json").write_text(json.dumps({
        "seqlens": seqlens, "vector_dims": v_dims, "matrix_cols": m_cols,
        "dim": dim, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
        "pim_config": str(args.pim_config),
        "ramulator_bin": str(args.ramulator_bin),
        "ramulator_config": str(args.ramulator_config),
        "model_shape": str(args.model_shape) if args.model_shape else None
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    traces_root = ensure_dir(out_dir / "traces")
    results_root = ensure_dir(out_dir / "results")

    # 1) traces
    call_01(args.py01, args.pim_config, ensure_dir(traces_root / "score_output"),
            "score,output", seqlens, None, None, False, dim, n_heads, n_kv_heads, args.model_shape)
    call_01(args.py01, args.pim_config, ensure_dir(traces_root / "weight_noaf"),
            "weight", None, v_dims, m_cols, False, dim, n_heads, n_kv_heads, args.model_shape)
    call_01(args.py01, args.pim_config, ensure_dir(traces_root / "weight_withaf"),
            "weight", None, v_dims, m_cols, True, dim, n_heads, n_kv_heads, args.model_shape)

    # 2) ramulator
    call_02(args.py02, traces_root / "score_output", args.ramulator_bin, args.ramulator_config, results_root / "score_output.csv", args.metric_regex, args.extra_args, args.cmd_template)
    call_02(args.py02, traces_root / "weight_noaf", args.ramulator_bin, args.ramulator_config, results_root / "weight.csv", args.metric_regex, args.extra_args, args.cmd_template)
    call_02(args.py02, traces_root / "weight_withaf", args.ramulator_bin, args.ramulator_config, results_root / "weight_af.csv", args.metric_regex, args.extra_args, args.cmd_template)

    # 3) fit
    merged_csv = results_root / "all_results.csv"
    merge_csvs([results_root / "score_output.csv", results_root / "weight.csv", results_root / "weight_af.csv"], merged_csv)

    model_json = results_root / "model_formula.json"
    summary_csv = results_root / "model_fit_summary.csv"
    call_03_fit(args.py03, merged_csv, model_json, summary_csv)

    # 4) 写 model_fit_summary.json（供算法侧读取）并复制一份到 measurements/pim/out_run/results/
    try:
        obj = json.loads(model_json.read_text(encoding="utf-8"))
        summary_json = results_root / "model_fit_summary.json"
        summary_json.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        print(f"[ok] wrote {summary_json}")
        # copy to measurements/
        meas_dir = Path("measurements/pim/out_run/results")
        meas_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(summary_json, meas_dir / "model_fit_summary.json")
        print(f"[ok] copied -> {meas_dir / 'model_fit_summary.json'}")
    except Exception as e:
        print("[warn] cannot write/copy summary json:", e)

    # 5) （可选）基于公式做预测
    if args.gen_preditctions or args.gen_predictions:
        preds_csv = results_root / "predictions.csv"
        with preds_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["op_label","op","with_af","seqlen","vector_dim","matrix_col","n_heads","predicted_cycles"])
            for L in seqlens:
                w.writerow(["score","score",0,L,"","",n_heads,f"{call_03_predict(args.py03, model_json, 'score', L, None, None, n_heads, args.model_shape):.3f}"])
                w.writerow(["output","output",0,L,"","",n_heads,f"{call_03_predict(args.py03, model_json, 'output', L, None, None, n_heads, args.model_shape):.3f}"])
            for V in v_dims:
                for N in m_cols:
                    w.writerow(["weight","weight",0,"",V,N,n_heads,f"{call_03_predict(args.py03, model_json, 'weight', None, V, N, n_heads, args.model_shape):.3f}"])
                    w.writerow(["weight_af","weight",1,"",V,N,n_heads,f"{call_03_predict(args.py03, model_json, 'weight_af', None, V, N, n_heads, args.model_shape):.3f}"])
        print(f"[ok] wrote predictions -> {preds_csv}")

    print("==== Done ====")
    print(f"Output_dir: {out_dir}")

if __name__ == "__main__":
    main()
