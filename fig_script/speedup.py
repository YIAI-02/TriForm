'''
python speedup.py --grid-best \
  --root ../algorithms/output/kv_cache_v2/llama_7b_INT8_b1 \
  --algos pd,attn_on_pim,weights_on_pim,facil,attacc,ianus,neupims,heft \
  --ncols 2 --sharey \
  --outfile ./prefill_decode_speedup_grid_heft_llama_7b_int8_b1.pdf

  python speedup.py --grid-best \
  --root ../algorithms/output/len_sweep/palm_8b_INT8_b4 \
  --algos pd,attn_on_pim,weights_on_pim,facil,attacc,ianus,neupims,heft \
  --ncols 2 --sharey \
  --outfile ./prefill_decode_speedup_grid_heft_palm_8b_INT8_b4.pdf

'''
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

def _order_policies(policies: List[str]) -> List[str]:
    group1 = ["weights_on_pim", "attn_on_pim", "pd"]
    group2 = ["neupims", "ianus", "facil", "attacc"]
    algos  = [p for p in policies if p.startswith("algo:")]
    ordered = [p for p in group1 if p in policies] + [p for p in group2 if p in policies] + algos
    leftovers = [p for p in policies if p not in ordered]
    return ordered + leftovers

def _compute_speeds(time_map: Dict[str, Tuple[float,float,float]], ordered: List[str]):
    # time_map: policy -> (prefill, decode, total)
    prefill = [time_map[p][0] for p in ordered]
    decode  = [time_map[p][1] for p in ordered]
    e2e     = [time_map[p][2] for p in ordered]
    pmax = max(prefill) if prefill else 1.0
    dmax = max(decode)  if decode  else 1.0
    emax = max(e2e)     if e2e     else 1.0
    def s(mx, t): return (np.inf if t == 0 else mx/ t)
    return [s(pmax,t) for t in prefill], [s(dmax,t) for t in decode], [s(emax,t) for t in e2e]

COL_PREFILL = "#1d2e53"
COL_DECODE  = "#395aad"
COL_E2E     = "#84b4fc"

def _annotate(ax, bars, values, ymax):
    for b, v in zip(bars, values):
        if v is None or (not np.isfinite(v)):  # nan/inf 不标
            continue
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02 * ymax,
                f"{v:.2f}×", ha="center", va="bottom", fontsize=9, rotation=90)

def _plot_one(ax, time_map: Dict[str, Tuple[float,float,float]], *, title: str):
    policies = list(time_map.keys())
    ordered  = _order_policies(policies)
    ps, ds, es = _compute_speeds(time_map, ordered)
    x = np.arange(len(ordered))
    group_width = 0.85
    bar_w = group_width / 3.0
    offs = np.array([-bar_w, 0.0, +bar_w])
    bars_p = ax.bar(x + offs[0], ps, width=bar_w, label="Prefill",
                    color=COL_PREFILL, edgecolor="black", linewidth=0.8, zorder=3)
    bars_d = ax.bar(x + offs[1], [np.nan if np.isinf(v) else v for v in ds], width=bar_w, label="Decode",
                    color=COL_DECODE,  edgecolor="black", linewidth=0.8, zorder=3)
    bars_e = ax.bar(x + offs[2], [np.nan if np.isinf(v) else v for v in es], width=bar_w, label="End-to-End",
                    color=COL_E2E,     edgecolor="black", linewidth=0.8, zorder=3)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.1, alpha=0.9, zorder=2)
    ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.9, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(ordered, rotation=30, ha="right")
    ax.set_title(title, pad=12)
    finite_vals = [v for v in (ps+ds+es) if np.isfinite(v)]
    ymax = (max(finite_vals) if finite_vals else 1.0) * 1.35
    ax.set_ylim(0, ymax)
    _annotate(ax, bars_p, ps, ymax); _annotate(ax, bars_d, ds, ymax); _annotate(ax, bars_e, es, ymax)
    return bars_p, bars_d, bars_e, ymax

# ===== 新增：从 algo_* 目录读取 best_summary_*.json，按 (S,T) 画子图 =====
def _load_best_summary(path: Path) -> Tuple[int,int,float,float,float]:
    """返回 (S, T, prefill, decode, total)。S/T 优先从 JSON 的 config 取，否则从文件名 'best_summary_SxT.json' 解析。"""
    obj = json.loads(path.read_text(encoding='utf-8'))
    bt = obj.get('best_times') or {}
    prefill = float(bt.get('prefill', 0.0)); decode = float(bt.get('decode', 0.0)); total = float(bt.get('total', prefill+decode))
    S = obj.get('config', {}).get('prefill_len'); T = obj.get('config', {}).get('decode_len')
    if (S is None or T is None):
        # best_summary_128x1024.json -> (128, 1024)
        stem = path.stem  # e.g. best_summary_128x1024
        try:
            tag = stem.split('_')[-1]
            S, T = [int(x) for x in tag.split('x')]
        except Exception:
            S = S or -1; T = T or -1
    return int(S), int(T), prefill, decode, total

def _gather_from_algos(root: Path, algos: List[str]):
    """
    返回：cases -> { policy -> (prefill, decode, total) }
      cases: List[(S,T)]（按 S,T 排序）
      policy: 形如 'algo:astar'，顺序按 group1/group2/algo 规则
    """
    by_case: Dict[Tuple[int,int], Dict[str, Tuple[float,float,float]]] = {}
    for a in algos:
        adir = root / f"algo_{a}"
        if not adir.is_dir(): 
            continue
        for f in sorted(adir.glob("best_summary_*.json")):
            S, T, pf, de, to = _load_best_summary(f)
            key = (S, T)
            by_case.setdefault(key, {})
            by_case[key][f"algo:{a}"] = (pf, de, to)
    cases = sorted(by_case.keys())
    return cases, by_case

def main():
    ap = argparse.ArgumentParser()
    # 单图/compare 兼容参数
    ap.add_argument('--json', type=str, default=None, help='单图：baseline_compare.json 路径（默认保持原相对路径）')
    ap.add_argument('--grid', action='store_true', help='compare-grid：批量读取 baseline_compare_*.json 画子图')
    ap.add_argument('--dir', type=str, default='./output/len_sweep', help='compare-grid：compare 文件所在目录')
    ap.add_argument('--pattern', type=str, default='baseline_compare_*.json', help='compare-grid：glob 模式')
    ap.add_argument('--ncols', type=int, default=2, help='子图列数')
    ap.add_argument('--sharey', action='store_true', help='子图共享 Y 轴（便于横向比较）')
    ap.add_argument('--outfile', type=str, default=None, help='输出图片路径（pdf/png）')
    # 新增：从 algo_* 目录读取 best_summary_* 的子图模式
    ap.add_argument('--grid-best', action='store_true', help='从 algo_* 目录下的 best_summary_*.json 聚合绘制子图')
    ap.add_argument('--root', type=str, default='../algorithms/output/len_sweep', help='algo_* 根目录')
    ap.add_argument('--algos', type=str, default='', help='逗号分隔：要绘制的算法名（不含前缀），例如 "heft,astar,ga,attn_on_pim"')
    args = ap.parse_args()

    if args.grid_best:
        root = Path(args.root)
        algos = [s.strip() for s in (args.algos or '').split(',') if s.strip()]
        if not algos:
            # 未指定则自动发现 algo_* 目录
            algos = [p.name.replace('algo_', '') for p in root.glob('algo_*') if p.is_dir()]
        cases, by_case = _gather_from_algos(root, algos)
        if not cases:
            raise SystemExit(f"No best_summary_* found under: {root}")
        n = len(cases); ncols = max(1, int(args.ncols)); nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 3.5*nrows), squeeze=False)
        legend = None
        for i, (S,T) in enumerate(cases):
            r, c = divmod(i, ncols); ax = axes[r][c]
            time_map = by_case[(S,T)]
            bars_p, bars_d, bars_e, _ = _plot_one(ax, time_map, title=f"S={S}, T={T}")
            ax.set_xlabel(f"prefill={S}, decode={T}")
            if legend is None:
                legend = (bars_p, bars_d, bars_e)
        # 关闭多余子图
        for j in range(i+1, nrows*ncols):
            axes[j//ncols][j%ncols].axis('off')
        if legend is not None:
            lp, ld, le = legend
            fig.legend((lp, ld, le), ('Prefill','Decode','End-to-End'),
                       ncol=3, loc='upper center', frameon=True, framealpha=0.9)
        plt.tight_layout(rect=[0,0,1,0.95])
        out = Path(args.outfile) if args.outfile else (root / 'prefill_decode_speedup_grid_from_best_llama_7b_fp16_b32.pdf')
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=220, bbox_inches='tight')
        print(f"Saved grid(best) to: {out}")
        return

    # ===== 兼容：老的 compare-grid / 单图（与原脚本一致） =====
    if args.grid:
        files = sorted(Path(args.dir).glob(args.pattern))
        if not files:
            raise SystemExit(f"No compare json found: {Path(args.dir) / args.pattern}")
        # 先预读求全局 ymax（sharey）
        cache = []; gmax = 0.0
        for fp in files:
            data = json.loads(Path(fp).read_text(encoding='utf-8'))
            results = data.get('results', [])
            # policy -> (prefill, decode, total)
            tm = {r['policy']: (float(r['prefill_time_s']), float(r['decode_time_s']), float(r['total_time_s'])) for r in results}
            ords = _order_policies(list(tm.keys()))
            ps, ds, es = _compute_speeds(tm, ords)
            finite = [v for v in (ps+ds+es) if np.isfinite(v)]
            gmax = max(gmax, max(finite) if finite else 1.0)
            cache.append((fp, tm, data.get('config', {})))
        n = len(cache); ncols = max(1, int(args.ncols)); nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4.4*nrows), squeeze=False)
        legend = None
        for i, (fp, tm, cfg) in enumerate(cache):
            r, c = divmod(i, ncols); ax = axes[r][c]
            S = cfg.get('prefill_len','?'); T = cfg.get('decode_len','?')
            bars_p, bars_d, bars_e, _ = _plot_one(ax, tm, title=f"S={S}, T={T}  ({fp.name})")
            ax.set_xlabel(f"prefill={S}, decode={T}")
            if legend is None:
                legend = (bars_p, bars_d, bars_e)
        for j in range(i+1, nrows*ncols):
            axes[j//ncols][j%ncols].axis('off')
        if legend is not None:
            lp, ld, le = legend
            fig.legend((lp, ld, le), ('Prefill','Decode','End-to-End'),
                       ncol=3, loc='upper center', frameon=True, framealpha=0.9)
        plt.tight_layout(rect=[0,0,1,0.95])
        out = Path(args.outfile) if args.outfile else (Path(args.dir) / 'prefill_decode_speedup_grid.pdf')
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=220, bbox_inches='tight')
        print(f"Saved grid to: {out}")
        return

    # 单图（保持与原脚本一致）
    json_path = Path(args.json) if args.json else Path("../algorithms/output/baseline_compare.json")
    data = json.loads(json_path.read_text(encoding='utf-8'))
    results = data["results"]
    policies = [r["policy"] for r in results]
    ordered = _order_policies(policies)
    time_map = {r["policy"]: (float(r["prefill_time_s"]), float(r["decode_time_s"]), float(r["total_time_s"])) for r in results}
    fig, ax = plt.subplots(figsize=(12, 5.6))
    _plot_one(ax, time_map, title="Prefill & Decode Speedup")
    out_path = Path(args.outfile) if args.outfile else Path("./prefill_decode_speedup.pdf")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
