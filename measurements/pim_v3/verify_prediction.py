#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import subprocess
import math

from aim_shared import load_model_shape

def shlex_join(cmd: List[str]) -> str:
    def q(x: str) -> str:
        if any(c in x for c in ' "\'\\'):
            return '"' + x.replace('"', r'\"') + '"'
        return x
    return " ".join(q(x) for x in cmd)

def run_cmd(cmd: List[str]) -> None:
    print(f"[CMD] {shlex_join(cmd)}")
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {shlex_join(cmd)} (rc={p.returncode})")

def read_predictions(pred_csv: Path) -> List[Dict[str, Any]]:
    with pred_csv.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return list(rd)

def group_points(rows: List[Dict[str, Any]]) -> Dict[str, Set[Tuple]]:
    g: Dict[str, Set[Tuple]] = {"score": set(), "output": set(), "weight": set(), "weight_af": set()}
    for r in rows:
        op = r.get("op_label") or r.get("op") or ""
        op = op.strip()
        if op not in g:
            continue
        if op in ("score","output"):
            L = r.get("seqlen") or ""
            if L != "":
                g[op].add((int(L),))
        elif op == "weight":
            v = r.get("vector_dim") or ""
            n = r.get("matrix_col") or ""
            if v != "" and n != "":
                g[op].add((int(v), int(n)))
        elif op == "weight_af":
            v = r.get("vector_dim") or ""
            n = r.get("matrix_col") or ""
            if v != "" and n != "":
                g[op].add((int(v), int(n)))
    return g

def write_list(lst, path: Path):
    path.write_text("\n".join(map(str, lst)), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="验证 predictions.csv 的有效性：重新真实跑 01+02，并与预测对比。")
    ap.add_argument("--predictions-csv", type=Path, required=True)
    ap.add_argument("--pim-config", type=Path, required=True)
    ap.add_argument("--ramulator-bin", type=Path, required=True)
    ap.add_argument("--ramulator-config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-shape", type=Path, default=None, help="用于确定 dim/n_heads/n_kv_heads")
    ap.add_argument("--py01", type=Path, default=Path(__file__).resolve().parent / "01_gentrace.py")
    ap.add_argument("--py02", type=Path, default=Path(__file__).resolve().parent / "02_run_ramulator.py")
    args = ap.parse_args()

    shape = load_model_shape(args.model_shape) if args.model_shape else None
    dim = shape["dim"] if shape else None
    n_heads = shape["n_heads"] if shape else None
    n_kv_heads = shape["n_kv_heads"] if shape else None

    out_dir = args.out_dir.resolve()
    (out_dir).mkdir(parents=True, exist_ok=True)
    traces_root = out_dir / "traces_verify"
    (traces_root).mkdir(parents=True, exist_ok=True)
    results_root = out_dir / "results_verify"
    (results_root).mkdir(parents=True, exist_ok=True)

    # 读取预测点并分组
    preds = read_predictions(args.predictions_csv)
    groups = group_points(preds)
    print({k: len(v) for k,v in groups.items()})

    # 生成 trace：score/output
    if groups["score"] or groups["output"]:
        seqlens = sorted({t[0] for t in (groups["score"] | groups["output"])})
        cmd = [sys.executable, str(args.py01),
               "--pim-config", str(args.pim_config),
               "--ops", "score,output",
               "--out-dir", str(traces_root / "score_output"),
               "--seqlens", ",".join(map(str, seqlens))]
        if args.model_shape: cmd += ["--model-shape", str(args.model_shape)]
        if dim is not None: cmd += ["--dim", str(dim)]
        if n_heads is not None: cmd += ["--n-heads", str(n_heads)]
        if n_kv_heads is not None: cmd += ["--n-kv-heads", str(n_kv_heads)]
        run_cmd(cmd)
        # run 02
        cmd = [sys.executable, str(args.py02),
               "--traces-dir", str(traces_root / "score_output"),
               "--ramulator-bin", str(args.ramulator_bin),
               "--config", str(args.ramulator_config),
               "--out-csv", str(results_root / "score_output.csv")]
        run_cmd(cmd)

    # weight（no AF）
    if groups["weight"]:
        vset = sorted({t[0] for t in groups["weight"]})
        nset = sorted({t[1] for t in groups["weight"]})
        cmd = [sys.executable, str(args.py01),
               "--pim-config", str(args.pim_config),
               "--ops", "weight",
               "--out-dir", str(traces_root / "weight_noaf"),
               "--vector-dims", ",".join(map(str, vset)),
               "--matrix-cols", ",".join(map(str, nset))]
        if args.model_shape: cmd += ["--model-shape", str(args.model_shape)]
        if dim is not None: cmd += ["--dim", str(dim)]
        if n_heads is not None: cmd += ["--n-heads", str(n_heads)]
        if n_kv_heads is not None: cmd += ["--n-kv-heads", str(n_kv_heads)]
        run_cmd(cmd)
        cmd = [sys.executable, str(args.py02),
               "--traces-dir", str(traces_root / "weight_noaf"),
               "--ramulator-bin", str(args.ramulator_bin),
               "--config", str(args.ramulator_config),
               "--out-csv", str(results_root / "weight.csv")]
        run_cmd(cmd)

    # weight_af
    if groups["weight_af"]:
        vset = sorted({t[0] for t in groups["weight_af"]})
        nset = sorted({t[1] for t in groups["weight_af"]})
        cmd = [sys.executable, str(args.py01),
               "--pim-config", str(args.pim_config),
               "--ops", "weight",
               "--with-af",
               "--out-dir", str(traces_root / "weight_withaf"),
               "--vector-dims", ",".join(map(str, vset)),
               "--matrix-cols", ",".join(map(str, nset))]
        if args.model_shape: cmd += ["--model-shape", str(args.model_shape)]
        if dim is not None: cmd += ["--dim", str(dim)]
        if n_heads is not None: cmd += ["--n-heads", str(n_heads)]
        if n_kv_heads is not None: cmd += ["--n-kv-heads", str(n_kv_heads)]
        run_cmd(cmd)
        cmd = [sys.executable, str(args.py02),
               "--traces-dir", str(traces_root / "weight_withaf"),
               "--ramulator-bin", str(args.ramulator_bin),
               "--config", str(args.ramulator_config),
               "--out-csv", str(results_root / "weight_af.csv")]
        run_cmd(cmd)

    # 合并实际结果
    actuals: List[Dict[str, Any]] = []
    for name in ["score_output.csv", "weight.csv", "weight_af.csv"]:
        p = results_root / name
        if not p.exists(): continue
        with p.open("r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            actuals.extend(list(rd))

    # 构建索引（只保留预测中出现的点）
    def key_of(row: Dict[str, Any]) -> Tuple:
        op = (row.get("op_label") or row.get("op") or "").strip()
        if op == "":
            op = "weight" if "weight" in (row.get("trace","")) else op
        waf = str(row.get("with_af", "0")).strip().lower()
        if (op == "weight" and waf in ("1","true")):
            op = "weight_af"
        if op in ("score","output"):
            return (op, int(row.get("seqlen", 0) or 0), None, None)
        else:
            return (op, None, int(row.get("vector_dim", 0) or 0), int(row.get("matrix_col", 0) or 0))

    actual_map: Dict[Tuple, float] = {}
    for r in actuals:
        k = key_of(r)
        cyc = r.get("cycles","")
        if cyc not in ("", None):
            try:
                actual_map[k] = float(cyc)
            except Exception:
                pass

    # 读取预测并对齐比较
    compare_rows: List[Dict[str, Any]] = []
    for r in preds:
        op = r.get("op_label") or r.get("op") or ""
        op = op.strip()
        if op in ("score","output"):
            L = int(r.get("seqlen", 0) or 0)
            k = (op, L, None, None)
        elif op == "weight":
            v = int(r.get("vector_dim", 0) or 0); n = int(r.get("matrix_col", 0) or 0)
            k = (op, None, v, n)
        elif op == "weight_af":
            v = int(r.get("vector_dim", 0) or 0); n = int(r.get("matrix_col", 0) or 0)
            k = (op, None, v, n)
        else:
            continue
        pred = float(r.get("predicted_cycles", 0) or 0)
        act = actual_map.get(k, float("nan"))
        diff = float("nan") if math.isnan(act) else (act - pred)
        rel = float("nan") if (math.isnan(act) or act == 0) else (diff / act)
        compare_rows.append({
            "op_label": op,
            "seqlen": r.get("seqlen",""),
            "vector_dim": r.get("vector_dim",""),
            "matrix_col": r.get("matrix_col",""),
            "predicted_cycles": f"{pred:.3f}",
            "actual_cycles": ("" if math.isnan(act) else f"{act:.3f}"),
            "error": ("" if math.isnan(diff) else f"{diff:.3f}"),
            "rel_error": ("" if math.isnan(rel) else f"{rel:.6f}")
        })

    # 写出对比 CSV
    comp_csv = out_dir / "verification_compare.csv"
    with comp_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["op_label","seqlen","vector_dim","matrix_col","predicted_cycles","actual_cycles","error","rel_error"])
        w.writeheader()
        w.writerows(compare_rows)
    print(f"[ok] wrote {comp_csv}")

    # 计算指标
    def metrics_for(op: str) -> Dict[str, Any]:
        rows = [r for r in compare_rows if r["op_label"] == op and r["actual_cycles"] != ""]
        if not rows:
            return {"op_label": op, "count": 0, "rmse": "", "mape": "", "r2": ""}
        preds = [float(r["predicted_cycles"]) for r in rows]
        acts = [float(r["actual_cycles"]) for r in rows]
        diffs = [a - p for a,p in zip(acts, preds)]
        rmse = (sum(d*d for d in diffs)/len(diffs))**0.5
        ape = [abs((a-p)/a) for a,p in zip(acts,preds) if a != 0]
        mape = sum(ape)/len(ape) if ape else float("nan")
        mean_a = sum(acts)/len(acts)
        ss_res = sum((a - p)**2 for a,p in zip(acts,preds))
        ss_tot = sum((a - mean_a)**2 for a in acts)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else float("nan")
        return {"op_label": op, "count": len(rows), "rmse": f"{rmse:.6g}", "mape": f"{mape:.6g}", "r2": f"{r2:.6g}"}

    summ_rows = [metrics_for(op) for op in ["score","output","weight","weight_af"]]
    # overall
    rows_all = [r for r in compare_rows if r["actual_cycles"] != ""]
    if rows_all:
        preds = [float(r["predicted_cycles"]) for r in rows_all]
        acts = [float(r["actual_cycles"]) for r in rows_all]
        diffs = [a - p for a,p in zip(acts, preds)]
        rmse = (sum(d*d for d in diffs)/len(diffs))**0.5
        ape = [abs((a-p)/a) for a,p in zip(acts,preds) if a != 0]
        mape = sum(ape)/len(ape) if ape else float("nan")
        mean_a = sum(acts)/len(acts)
        ss_res = sum((a - p)**2 for a,p in zip(acts,preds))
        ss_tot = sum((a - mean_a)**2 for a in acts)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else float("nan")
        summ_rows.append({"op_label": "ALL", "count": len(rows_all), "rmse": f"{rmse:.6g}", "mape": f"{mape:.6g}", "r2": f"{r2:.6g}"})
    else:
        summ_rows.append({"op_label": "ALL", "count": 0, "rmse": "", "mape": "", "r2": ""})

    summ_csv = out_dir / "verification_summary.csv"
    with summ_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["op_label","count","rmse","mape","r2"])
        w.writeheader()
        w.writerows(summ_rows)
    print(f"[ok] wrote {summ_csv}")

if __name__ == "__main__":
    main()
