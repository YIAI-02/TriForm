'''
python speedup.py --grid-best \
  --root /Users/yangjiaqi/WW/project_1/python/TriForm_bak/TriForm/algorithms/output/baseline_sweep_pima/llama_7b_INT8_b1\
  --algos pd,attn_on_pim,weights_on_pim,facil,attacc,ianus,neupims,heft \
  --ncols 2 --sharey \
  --outfile ./pima_heft_llama_7b_int8_b1.pdf
'''
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional

def _order_policies(policies: List[str]) -> List[str]:
    group1 = ["weights_on_pim", "attn_on_pim", "pd"]
    group2 = ["neupims", "ianus", "facil", "attacc"]
    algos  = [p for p in policies if p.startswith("algo:")]
    ordered = [p for p in group1 if p in policies] + [p for p in group2 if p in policies] + algos
    leftovers = [p for p in policies if p not in ordered]
    combined = ordered + leftovers
    if "algo:heft" in combined:
        combined = [p for p in combined if p != "algo:heft"] + ["algo:heft"]
    return combined

def _compute_multiples(time_map: Dict[str, Tuple[float,float,float]], ordered: List[str]):
    pd_prefill = None
    for policy, times in time_map.items():
        if policy == "algo:pd":
            pd_prefill = times[0]
            break

    if pd_prefill is None or pd_prefill == 0:
        pd_prefill = 1.0

    prefill = [time_map[p][0] for p in ordered]
    decode  = [time_map[p][1] for p in ordered]
    e2e     = [time_map[p][2] for p in ordered]

    def m(t): return (np.inf if pd_prefill == 0 else t / pd_prefill)
    return [m(t) for t in prefill], [m(t) for t in decode], [m(t) for t in e2e]

COL_PREFILL = "#1d2e53"
COL_DECODE  = "#395aad"
COL_E2E     = "#84b4fc"

def _annotate(ax, bars, values, ymax, is_prefill=False):
    for b, v in zip(bars, values):
        if v is None or (not np.isfinite(v)):  # nan/inf 不标
            continue
        height = b.get_height()
        if height <= 0:
            continue
        if is_prefill:
            y = height + 0.02 * ymax
            va = "bottom"
            color = "black"
        else:
            pad = max(0.05 * height, 0.02 * ymax)
            y = max(height - pad, height * 0.35)
            va = "top"
            color = "white"  # For visibility inside bar
        ax.text(
            b.get_x() + b.get_width() / 2,
            y,
            f"{v:.2f}×",
            ha="center",
            va=va,
            fontsize=9,
            rotation=90,
            color=color,
        )

def _plot_one(ax, time_map: Dict[str, Tuple[float,float,float]], *, title: str, ymax_override: Optional[float] = None):
    policies = list(time_map.keys())
    ordered  = _order_policies(policies)
    ps, ds, es = _compute_multiples(time_map, ordered)
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

    # Draw lighter gray dashed horizontal line at heft's decode multiple to indicate minimum decode latency
    if "algo:heft" in ordered:
        heft_idx = ordered.index("algo:heft")
        ax.axhline(ds[heft_idx], linestyle="--", color="black", linewidth=1.0, alpha=0.7, zorder=4)

    ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.9, zorder=1)
    labels = [p.replace("algo:", "") for p in ordered]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title, pad=12)
    finite_vals = [v for v in (ps+ds+es) if np.isfinite(v) and v > 0]
    if ymax_override is not None:
        ymax = max(ymax_override, 1e-6)
        ax.set_ylim(0.0, ymax)
    elif finite_vals:
        ymax = max(finite_vals) * 1.2
        if ymax <= 0:
            ymax = 1.0
        ax.set_ylim(0.0, ymax)
    else:
        ymax = 1.0
        ax.set_ylim(0.0, ymax)
    _annotate(ax, bars_p, ps, ymax, is_prefill=True)
    _annotate(ax, bars_d, ds, ymax)
    _annotate(ax, bars_e, es, ymax)
    return bars_p, bars_d, bars_e, ymax

# ===== 从 algo_* 目录读取 best_summary_*.json，按 (S,T) 画子图 =====
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


def _prepare_compare_cache(files: List[Path]):
    cache = []
    ymax_candidates: List[float] = []
    for fp in files:
        data = json.loads(Path(fp).read_text(encoding='utf-8'))
        results = data.get('results', [])
        tm = {r['policy']: (float(r['prefill_time_s']), float(r['decode_time_s']), float(r['total_time_s'])) for r in results}
        cache.append((Path(fp), tm, data.get('config', {})))
        ords = _order_policies(list(tm.keys()))
        ps, ds, es = _compute_multiples(tm, ords)
        finite = [v for v in (ps + ds + es) if np.isfinite(v) and v > 0]
        if finite:
            ymax_candidates.append(max(finite) * 1.2)
    return cache, ymax_candidates


def _plot_compare_grid(files: List[Path], *, ncols: int, sharey: bool, outfile: Path):
    files = sorted(files)
    cache, ymax_candidates = _prepare_compare_cache(files)
    if not cache:
        return None
    n = len(cache)
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols
    share_limit = max(ymax_candidates) if (sharey and ymax_candidates) else None
    share_axes = bool(sharey and share_limit is not None)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4.4*nrows), squeeze=False, sharey=share_axes)
    legend = None
    for i, (fp, tm, cfg) in enumerate(cache):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        S = cfg.get('prefill_len', '?')
        T = cfg.get('decode_len', '?')
        bars_p, bars_d, bars_e, _ = _plot_one(ax, tm, title=f"S={S}, T={T}  ({fp.name})", ymax_override=share_limit)
        ax.set_xlabel(f"prefill={S}, decode={T}")
        if legend is None:
            legend = (bars_p, bars_d, bars_e)
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    if legend is not None:
        lp, ld, le = legend
        fig.legend((lp, ld, le), ('Prefill', 'Decode', 'End-to-End'),
                   ncol=3, loc='upper center', frameon=True, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=220, bbox_inches='tight')
    print(f"Saved grid to: {outfile}")
    return outfile


def _extract_path_metadata(json_path: Path, root: Path):
    try:
        rel = json_path.relative_to(root)
    except ValueError:
        rel = json_path
    parts = rel.parts
    hardware = parts[0] if len(parts) > 0 else None
    scenario = parts[1] if len(parts) > 1 else None
    model_dir = parts[2] if len(parts) > 2 else None
    model_family = model_variant = dtype = None
    batch = None
    if model_dir:
        tokens = model_dir.split('_')
        if tokens:
            model_family = tokens[0]
        if len(tokens) >= 2:
            model_variant = tokens[1]
        if len(tokens) >= 3:
            dtype = tokens[2]
        if len(tokens) >= 4 and tokens[3].startswith('b'):
            try:
                batch = int(tokens[3][1:])
            except ValueError:
                batch = tokens[3]
    stride_val = None
    if scenario and scenario.startswith('st'):
        try:
            stride_val = int(scenario[2:])
        except ValueError:
            stride_val = scenario
    return {
        'hardware': hardware,
        'scenario': scenario,
        'model_dir': model_dir,
        'model_family': model_family,
        'model_variant': model_variant,
        'dtype': dtype,
        'batch': batch,
        'stride_value': stride_val,
    }


CSV_COLUMNS = [
    'hardware',
    'scenario',
    'model_dir',
    'model_family',
    'model_variant',
    'dtype',
    'batch',
    'prefill_len',
    'decode_len',
    'decode_sample_stride',
    'policy',
    'prefill_time_s',
    'first_token_latency_s',
    'decode_time_s',
    'decode_time_per_token_s',
    'total_time_s',
    'prefill_multiple_vs_pd_prefill',
    'decode_multiple_vs_pd_prefill',
    'total_multiple_vs_pd_prefill',
    'json_file',
]


def _rows_from_compare_file(json_path: Path, *, lens_root: Path):
    data = json.loads(json_path.read_text(encoding='utf-8'))
    config = data.get('config', {})
    results = data.get('results', [])
    time_map = {r['policy']: (float(r.get('prefill_time_s', 0.0)),
                              float(r.get('decode_time_s', 0.0)),
                              float(r.get('total_time_s', 0.0))) for r in results}
    ordered = _order_policies(list(time_map.keys()))
    ps, ds, es = _compute_multiples(time_map, ordered)
    multiples = {policy: (ps[idx], ds[idx], es[idx]) for idx, policy in enumerate(ordered)}
    meta = _extract_path_metadata(json_path, lens_root)
    stride = config.get('decode_sample_stride', meta.get('stride_value'))
    rows = []
    try:
        rel_path = str(json_path.relative_to(lens_root))
    except ValueError:
        rel_path = str(json_path)
    for r in results:
        pol = r['policy']
        pm, dm, em = multiples.get(pol, (np.nan, np.nan, np.nan))
        prefill_time = float(r.get('prefill_time_s', 0.0))
        decode_time = float(r.get('decode_time_s', 0.0))
        decode_len = config.get('decode_len')
        try:
            decode_per_token = (decode_time / decode_len) if decode_len else np.nan
        except ZeroDivisionError:
            decode_per_token = np.nan
        rows.append({
            'hardware': meta.get('hardware'),
            'scenario': meta.get('scenario'),
            'model_dir': meta.get('model_dir'),
            'model_family': meta.get('model_family') or config.get('model_family'),
            'model_variant': meta.get('model_variant') or config.get('model_variant'),
            'dtype': meta.get('dtype') or config.get('dtype'),
            'batch': meta.get('batch') or config.get('batch'),
            'prefill_len': config.get('prefill_len'),
            'decode_len': config.get('decode_len'),
            'decode_sample_stride': stride,
            'policy': pol,
            'prefill_time_s': prefill_time,
            'first_token_latency_s': prefill_time,
            'decode_time_s': decode_time,
            'decode_time_per_token_s': decode_per_token,
            'total_time_s': float(r.get('total_time_s', 0.0)),
            'prefill_multiple_vs_pd_prefill': pm,
            'decode_multiple_vs_pd_prefill': dm,
            'total_multiple_vs_pd_prefill': em,
            'json_file': rel_path,
        })
    return rows


def _write_lens_csv(rows: List[Dict[str, object]], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _scan_lens_eval(root: Path, *, ncols: int, sharey: bool, csv_out: Path):
    root = root.resolve()
    if not root.exists():
        raise SystemExit(f"lens_eval root does not exist: {root}")
    json_files = sorted(root.rglob('baseline_compare_*.json'))
    if not json_files:
        raise SystemExit(f"No baseline_compare_*.json found under: {root}")
    by_dir: Dict[Path, List[Path]] = {}
    rows: List[Dict[str, object]] = []
    for fp in json_files:
        rows.extend(_rows_from_compare_file(fp, lens_root=root))
        by_dir.setdefault(fp.parent, []).append(fp)
    for dir_path in sorted(by_dir.keys()):
        files = sorted(by_dir[dir_path])
        out_pdf = dir_path / 'prefill_decode_latency_multiples_vs_pd_prefill_grid.pdf'
        _plot_compare_grid(files, ncols=ncols, sharey=sharey, outfile=out_pdf)
        print(f"[lens-eval] processed {dir_path.relative_to(root)}")
    _write_lens_csv(rows, csv_out)
    print(f"Saved lens-eval CSV to: {csv_out}")

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
    # lens_eval 批量处理
    ap.add_argument('--lens-sweep', action='store_true', help='遍历 output/lens_eval_sweep 下的 baseline_compare json 并绘图+导出 CSV')
    ap.add_argument('--lens-root', type=str, default='../algorithms/output/lens_eval_sweep', help='lens_eval_sweep 根目录')
    ap.add_argument('--lens-csv', type=str, default=None, help='汇总 CSV 输出路径')
    args = ap.parse_args()

    if args.lens_sweep:
        lens_root = Path(args.lens_root)
        csv_out = Path(args.lens_csv) if args.lens_csv else (lens_root / 'lens_eval_summary.csv')
        _scan_lens_eval(lens_root, ncols=args.ncols, sharey=args.sharey, csv_out=csv_out)
        return

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
        out = Path(args.outfile) if args.outfile else (root / 'prefill_decode_latency_multiples_vs_pd_prefill_grid_from_best_llama_7b_fp16_b32.pdf')
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=220, bbox_inches='tight')
        print(f"Saved grid(best) to: {out}")
        return

    # ===== 兼容：老的 compare-grid / 单图（与原脚本一致） =====
    if args.grid:
        files = sorted(Path(args.dir).glob(args.pattern))
        if not files:
            raise SystemExit(f"No compare json found: {Path(args.dir) / args.pattern}")
        out = Path(args.outfile) if args.outfile else (Path(args.dir) / 'prefill_decode_latency_multiples_vs_pd_prefill_grid.pdf')
        _plot_compare_grid(files, ncols=args.ncols, sharey=args.sharey, outfile=out)
        return

    # 单图（保持与原脚本一致）
    json_path = Path(args.json) if args.json else Path("../algorithms/output/baseline_compare.json")
    data = json.loads(json_path.read_text(encoding='utf-8'))
    results = data["results"]
    policies = [r["policy"] for r in results]
    ordered = _order_policies(policies)
    time_map = {r["policy"]: (float(r["prefill_time_s"]), float(r["decode_time_s"]), float(r["total_time_s"])) for r in results}
    fig, ax = plt.subplots(figsize=(12, 5.6))
    _plot_one(ax, time_map, title="Prefill & Decode Latency Multiples (vs PD Prefill)")
    out_path = Path(args.outfile) if args.outfile else Path("./prefill_decode_latency_multiples_vs_pd_prefill.pdf")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
