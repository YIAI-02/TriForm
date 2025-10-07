#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动批处理脚本：
1) 调用 01_gentrace.py 生成多种规模的 trace（包含四种算子：score、output、weight、weight+AF=weight_af）。
2) 调用 02_run_ramulator.py 跑 ramulator，拿到 cycles 并把特征统计拼到 CSV。
3) 调用 03_fit_latency_model.py fit（*基于尺寸变量的显式公式*），得到 per-op 的公式模型 model.json 与摘要 CSV。
4) （可选）用 03 的 predict（基于公式，不需要再生成 trace）在一批尺寸上生成预测结果表 predictions.csv。

使用示例：
python auto_fit_pipeline.py \
  --pim-config /path/to/pim_config.json \
  --ramulator-bin /path/to/ramulator \
  --ramulator-config /path/to/ramulator_config.cfg \
  --model-shape ../configs/mpt_shape.json \
  --out-dir out_run \
  --seqlens 16,32,64,128,256,384,512,768,1024,1536,2048,3072,4096 \
  --vector-dims 64,96,128,192,256,384,512,768,1024 \
  --matrix-cols 64,128,192,256,384,512,768,1024,1536,2048,3072,4096 \
  --gen-predictions

如果算力有限，可用 --light-grid 来生成较少的 trace。
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from aim_shared import load_model_shape

# ----------------------------- 工具函数 -----------------------------

def shlex_join(cmd: List[str]) -> str:
    """在不依赖 shlex 的情况下简单拼接命令，主要用于日志展示"""
    def q(x: str) -> str:
        if any(c in x for c in ' "\'\\'):
            return '"' + x.replace('"', r'\"') + '"'
        return x
    return " ".join(q(x) for x in cmd)

def run_cmd(cmd: List[str], cwd: Path | None = None) -> int:
    print(f"[CMD] {shlex_join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {shlex_join(cmd)}")
    return proc.returncode

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_csv_headers(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        import csv as _csv
        r = _csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            return []
    return header

def merge_csvs(csv_files: List[Path], out_csv: Path) -> None:
    """将多个 CSV 合并为一个，列名取并集；缺失列补空值"""
    import csv as _csv
    headers_list = [parse_csv_headers(p) for p in csv_files if p.exists() and p.stat().st_size > 0]
    union_headers: List[str] = []
    for hs in headers_list:
        for h in hs:
            if h not in union_headers:
                union_headers.append(h)
    if not union_headers:
        raise RuntimeError("没有可合并的 CSV（为空或不存在）。")

    with out_csv.open("w", newline="", encoding="utf-8") as wf:
        w = _csv.DictWriter(wf, fieldnames=union_headers)
        w.writeheader()
        for p in csv_files:
            if not p.exists() or p.stat().st_size == 0:
                continue
            with p.open("r", newline="", encoding="utf-8") as f:
                r = _csv.DictReader(f)
                for row in r:
                    # 标记来源，方便后续排查
                    row.setdefault("source_csv", str(p))
                    # 兼容：如果有 with_af 列但值为空，补成 False/0
                    if "with_af" in row and (row["with_af"] in ("", None)):
                        row["with_af"] = "0"
                    # 统一 op_label：四种算子
                    op = row.get("op", "") or ""
                    with_af = row.get("with_af", "0")
                    if op == "weight" and str(with_af) in ("1", "True", "true"):
                        row["op_label"] = "weight_af"
                    elif op in ("score", "output", "weight"):
                        row["op_label"] = op
                    else:
                        row["op_label"] = op or "unknown"
                    w.writerow({h: row.get(h, "") for h in union_headers})
    print(f"[ok] merged -> {out_csv}")

def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out: List[int] = []
    for tok in s.split(","):
        t = tok.strip()
        if not t:
            continue
        if "-" in t:
            a, b = t.split("-", 1)
            a, b = int(a), int(b)
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(a, b - 1, -1))
        else:
            out.append(int(t))
    # 去重并排序
    out = sorted(set(out))
    return out

def sample_pairs(vs: List[int], ns: List[int], limit: int | None) -> Tuple[List[int], List[int]]:
    """如果 vs×ns 太大，用均匀抽样减少组合数量"""
    all_pairs = [(v, n) for v in vs for n in ns]
    if limit is None or limit <= 0 or len(all_pairs) <= limit:
        return vs, ns
    step = max(1, len(all_pairs) // limit)
    sampled = all_pairs[::step][:limit]
    sv = sorted(set(v for v, _ in sampled))
    sn = sorted(set(n for _, n in sampled))
    return sv, sn

# -------------------------- 主流程封装 --------------------------

def call_01_gentrace(py01: Path,
                     pim_config: Path,
                     out_dir: Path,
                     ops: str,
                     seqlens: List[int] | None,
                     vector_dims: List[int] | None,
                     matrix_cols: List[int] | None,
                     with_af: bool,
                     dim: int,
                     n_heads: int,
                     n_kv_heads: int | None,
                     model_shape: Path | None) -> None:
    cmd = [sys.executable, str(py01),
           "--pim-config", str(pim_config),
           "--ops", ops,
           "--out-dir", str(out_dir)]
    if dim is not None:
        cmd += ["--dim", str(dim)]
    if n_heads is not None:
        cmd += ["--n-heads", str(n_heads)]
    if n_kv_heads is not None:
        cmd += ["--n-kv-heads", str(n_kv_heads)]
    if model_shape is not None:
        cmd += ["--model-shape", str(model_shape)]
    if seqlens:
        cmd += ["--seqlens", ",".join(map(str, seqlens))]
    if vector_dims:
        cmd += ["--vector-dims", ",".join(map(str, vector_dims))]
    if matrix_cols:
        cmd += ["--matrix-cols", ",".join(map(str, matrix_cols))]
    if with_af:
        cmd += ["--with-af"]
    run_cmd(cmd)

def call_02_run_ramulator(py02: Path,
                          traces_dir: Path,
                          ramulator_bin: Path,
                          ramulator_cfg: Path,
                          out_csv: Path,
                          metric_regex: str | None,
                          extra_args: str | None,
                          cmd_template: str | None) -> None:
    cmd = [sys.executable, str(py02),
           "--traces-dir", str(traces_dir),
           "--ramulator-bin", str(ramulator_bin),
           "--config", str(ramulator_cfg),
           "--out-csv", str(out_csv)]
    if metric_regex:
        cmd += ["--metric-regex", metric_regex]
    if extra_args:
        cmd += ["--extra-args", extra_args]
    if cmd_template:
        cmd += ["--cmd-template", cmd_template]
    run_cmd(cmd)

def call_03_fit(py03: Path, results_csv: Path, out_model: Path, summary_csv: Path) -> None:
    cmd = [sys.executable, str(py03), "fit",
           "--results-csv", str(results_csv),
           "--out", str(out_model),
           "--summary-csv", str(summary_csv)]
    run_cmd(cmd)

def call_03_predict(py03: Path,
                    model_json: Path,
                    op: str,
                    seqlen: int | None,
                    vector_dim: int | None,
                    matrix_col: int | None,
                    n_heads: int | None,
                    model_shape: Path | None) -> float:
    cmd = [sys.executable, str(py03), "predict",
           "--model", str(model_json),
           "--op", op]
    if seqlen is not None:
        cmd += ["--seqlen", str(seqlen)]
    if vector_dim is not None:
        cmd += ["--vector-dim", str(vector_dim)]
    if matrix_col is not None:
        cmd += ["--matrix-col", str(matrix_col)]
    if n_heads is not None:
        cmd += ["--n-heads", str(n_heads)]
    if model_shape is not None and n_heads is None:
        cmd += ["--model-shape", str(model_shape)]
    # 捕获输出
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"predict failed code={proc.returncode}")
    out = proc.stdout.strip().splitlines()[-1].strip()
    try:
        return float(out)
    except Exception:
        import re as _re
        m = _re.search(r"([0-9]+(?:\.[0-9]+)?)", out)
        if not m:
            raise RuntimeError(f"cannot parse predicted cycles from: {out!r}")
        return float(m.group(1))

# ------------------------------- CLI -------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pim-config", type=Path, required=True, help="01_gentrace 所需的 PIM 配置 JSON")
    ap.add_argument("--ramulator-bin", type=Path, required=True)
    ap.add_argument("--ramulator-config", type=Path, required=True)
    ap.add_argument("--model-shape", type=Path, default=None, help="从 ../configs/*_shape.json 读取 dim/n_heads/n_kv_heads/seq_length")
    ap.add_argument("--out-dir", type=Path, default=Path("auto_run_") / datetime.now().strftime("%Y%m%d_%H%M%S"))

    # 网格（尽量多的数据）。可以用 --light-grid 走一个较小网格
    ap.add_argument("--seqlens", type=str,
                    default="16,32,48,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,896,1024,1280,1536,1792,2048,2560,3072,3584,4096")
    ap.add_argument("--vector-dims", type=str,
                    default="64,96,128,192,256,384,512,768,1024")
    ap.add_argument("--matrix-cols", type=str,
                    default="64,128,192,256,384,512,768,1024,1536,2048,3072,4096")
    ap.add_argument("--max-weight-combos", type=int, default=0,
                    help=">0 则对 (vector_dim, matrix_col) 进行均匀抽样，限制组合个数")

    # Transformer 结构参数（若提供 --model-shape 则以 shape 为准）
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--n-kv-heads", type=int, default=None)

    # 02 的可选项
    ap.add_argument("--metric-regex", type=str, default=None)
    ap.add_argument("--extra-args", type=str, default=None)
    ap.add_argument("--cmd-template", type=str, default=None)

    # 是否精简网格
    ap.add_argument("--light-grid", action="store_true",
                    help="使用一个较小的网格（更快）")

    # 是否生成预测表
    ap.add_argument("--gen-predictions", action="store_true",
                    help="基于**公式模型**，在一批尺寸上做预测并导出 predictions.csv")

    # 三个子脚本的路径（默认取当前脚本所在目录下的同名文件）
    here = Path(__file__).resolve().parent
    ap.add_argument("--py01", type=Path, default=here / "01_gentrace.py")
    ap.add_argument("--py02", type=Path, default=here / "02_run_ramulator.py")
    ap.add_argument("--py03", type=Path, default=here / "03_fit_latency_model.py")
    return ap

# ------------------------------- 主入口 -------------------------------

def main() -> None:
    args = build_parser().parse_args()

    out_dir: Path = args.out_dir.resolve()
    ensure_dir(out_dir)

    # 解析 shape（若给）
    shape = None
    if args.model_shape:
        try:
            shape = load_model_shape(args.model_shape)
        except Exception as e:
            print(f"[warn] 读取 --model-shape 失败：{e}；将使用命令行/默认参数。", file=sys.stderr)

    # 解析/覆盖 模型形状
    dim = args.dim if args.dim is not None else (shape["dim"] if shape else 256)
    n_heads = args.n_heads if args.n_heads is not None else (shape["n_heads"] if shape else 8)
    n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else (shape["n_kv_heads"] if shape else n_heads)

    # 解析网格
    if args.light_grid:
        seqlens = parse_int_list("16,32,64,128,256,512,1024")
        v_dims  = parse_int_list("64,128,256,512,768,1024")
        m_cols  = parse_int_list("64,128,256,512,1024,2048,4096")
    else:
        seqlens = parse_int_list(args.seqlens)
        v_dims  = parse_int_list(args.vector_dims)
        m_cols  = parse_int_list(args.matrix_cols)

    # 若 shape 指定了 seq_length，则裁掉超过该长度的 seqlen 点
    if shape and shape.get("seq_length"):
        Lmax = int(shape["seq_length"])
        seqlens = [L for L in seqlens if L <= Lmax] or [min(512, Lmax)]

    # weight 组合抽样（可选）
    if args.max_weight_combos and args.max_weight_combos > 0:
        v_dims, m_cols = sample_pairs(v_dims, m_cols, args.max_weight_combos)

    # 保存参数
    (out_dir / "args.json").write_text(json.dumps({
        "seqlens": seqlens, "vector_dims": v_dims, "matrix_cols": m_cols,
        "dim": dim, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
        "pim_config": str(args.pim_config),
        "ramulator_bin": str(args.ramulator_bin),
        "ramulator_config": str(args.ramulator_config),
        "metric_regex": args.metric_regex,
        "extra_args": args.extra_args,
        "cmd_template": args.cmd_template,
        "light_grid": bool(args.light_grid),
        "model_shape": str(args.model_shape) if args.model_shape else None
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------
    # Step 1: 生成 trace（四种）
    # ---------------------------------------------------
    traces_root = ensure_dir(out_dir / "traces")

    # 1a) score + output（与 seqlen 相关）
    traces_score_output = ensure_dir(traces_root / "score_output")
    call_01_gentrace(args.py01, args.pim_config, traces_score_output,
                     ops="score,output",
                     seqlens=seqlens,
                     vector_dims=None, matrix_cols=None,
                     with_af=False,
                     dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                     model_shape=args.model_shape)

    # 1b) weight（不带 AF）
    traces_weight = ensure_dir(traces_root / "weight_noaf")
    call_01_gentrace(args.py01, args.pim_config, traces_weight,
                     ops="weight",
                     seqlens=None,
                     vector_dims=v_dims, matrix_cols=m_cols,
                     with_af=False,
                     dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                     model_shape=args.model_shape)

    # 1c) weight（带 AF）
    traces_weight_af = ensure_dir(traces_root / "weight_withaf")
    call_01_gentrace(args.py01, args.pim_config, traces_weight_af,
                     ops="weight",
                     seqlens=None,
                     vector_dims=v_dims, matrix_cols=m_cols,
                     with_af=True,
                     dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                     model_shape=args.model_shape)

    # ---------------------------------------------------
    # Step 2: 跑 ramulator -> 结果 CSV
    # ---------------------------------------------------
    results_root = ensure_dir(out_dir / "results")
    csv_score_output = results_root / "score_output.csv"
    csv_weight       = results_root / "weight.csv"
    csv_weight_af    = results_root / "weight_af.csv"

    call_02_run_ramulator(args.py02, traces_score_output, args.ramulator_bin, args.ramulator_config,
                          out_csv=csv_score_output,
                          metric_regex=args.metric_regex, extra_args=args.extra_args,
                          cmd_template=args.cmd_template)

    call_02_run_ramulator(args.py02, traces_weight, args.ramulator_bin, args.ramulator_config,
                          out_csv=csv_weight,
                          metric_regex=args.metric_regex, extra_args=args.extra_args,
                          cmd_template=args.cmd_template)

    call_02_run_ramulator(args.py02, traces_weight_af, args.ramulator_bin, args.ramulator_config,
                          out_csv=csv_weight_af,
                          metric_regex=args.metric_regex, extra_args=args.extra_args,
                          cmd_template=args.cmd_template)

    # ---------------------------------------------------
    # Step 3: 合并 CSV + 拟合（03 fit）
    # ---------------------------------------------------
    merged_csv = results_root / "all_results.csv"
    merge_csvs([csv_score_output, csv_weight, csv_weight_af], merged_csv)

    model_json = results_root / "model_formula.json"
    summary_csv = results_root / "model_fit_summary.csv"
    # 使用尺寸变量的显式公式进行拟合
    call_03_fit(args.py03, merged_csv, model_json, summary_csv)

    # ---------------------------------------------------
    # Step 4: （可选）用模型在一批尺寸上做预测，导出 predictions.csv（不需要任何 trace）
    # ---------------------------------------------------
    if args.gen_predictions:
        preds_csv = results_root / "predictions.csv"
        with preds_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["op_label", "op", "with_af", "seqlen", "vector_dim", "matrix_col", "n_heads", "predicted_cycles"])

            # score
            for L in seqlens:
                cyc = call_03_predict(args.py03, model_json,
                                      op="score", seqlen=L, vector_dim=None, matrix_col=None,
                                      n_heads=n_heads, model_shape=args.model_shape)
                w.writerow(["score", "score", 0, L, "", "", n_heads, f"{cyc:.3f}"])

            # output
            for L in seqlens:
                cyc = call_03_predict(args.py03, model_json,
                                      op="output", seqlen=L, vector_dim=None, matrix_col=None,
                                      n_heads=n_heads, model_shape=args.model_shape)
                w.writerow(["output", "output", 0, L, "", "", n_heads, f"{cyc:.3f}"])

            # weight（两种：无 AF / 有 AF）
            for v in v_dims:
                for n in m_cols:
                    cyc0 = call_03_predict(args.py03, model_json,
                                           op="weight", seqlen=None, vector_dim=v, matrix_col=n,
                                           n_heads=n_heads, model_shape=args.model_shape)
                    w.writerow(["weight", "weight", 0, "", v, n, n_heads, f"{cyc0:.3f}"])

                    cyc1 = call_03_predict(args.py03, model_json,
                                           op="weight_af", seqlen=None, vector_dim=v, matrix_col=n,
                                           n_heads=n_heads, model_shape=args.model_shape)
                    w.writerow(["weight_af", "weight", 1, "", v, n, n_heads, f"{cyc1:.3f}"])
        print(f"[ok] wrote predictions -> {preds_csv}")

    print("\n==== 完成 ====")
    print(f"输出目录：{out_dir}")
    print(f"- traces/score_output, traces/weight_noaf, traces/weight_withaf")
    print(f"- results/score_output.csv, results/weight.csv, results/weight_af.csv")
    print(f"- results/all_results.csv (合并)")
    print(f"- results/model_formula.json (四算子公式模型)")
    print(f"- results/model_fit_summary.csv (每算子一条可读公式 + 指标)")
    if args.gen_predictions:
        print(f"- results/predictions.csv (基于公式模型的尺寸-延迟表)")

if __name__ == "__main__":
    main()
