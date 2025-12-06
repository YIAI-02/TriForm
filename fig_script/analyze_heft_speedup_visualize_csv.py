#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

PALETTE = ["#0024FF", "#0349FF", "#006CFE", "#0092FE",
           "#FF3F04", "#FE5D00", "#FE8000", "#FFBF02"]

# Diverging for log2(speedup): <1 (blue) | >1 (warm)
DIVERGING_SPEEDUP = ["#0024FF", "#0349FF", "#006CFE", "#0092FE",
                     "#FFBF02", "#FE8000", "#FE5D00", "#FF3F04"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def color_cycle(n):
    return [PALETTE[i % len(PALETTE)] for i in range(n)]

def read_csv_robust(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # common "index column"
    if df.columns[0].startswith("Unnamed:"):
        df = df.rename(columns={df.columns[0]: "index"})
    return df

def savefig(out_path: Path):
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_barh_categories(df, cat_col, val_col, out_path, title, topn=25, positive_color="#006CFE", negative_color="#FE8000"):
    d = df[[cat_col, val_col]].copy()
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna(subset=[val_col])
    if d.empty:
        return
    d = d.sort_values(val_col, ascending=False).head(topn)
    y = d[cat_col].astype(str).values
    x = d[val_col].values
    colors = [positive_color if v >= 0 else negative_color for v in x]

    plt.figure(figsize=(10, max(4, 0.35 * len(d))))
    plt.barh(y[::-1], x[::-1], color=colors[::-1])
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.xlabel(val_col)
    savefig(out_path)

def plot_summary_by_baseline(p: Path, out_dir: Path):
    df = read_csv_robust(p)
    if "baseline_algo" not in df.columns:
        # maybe stored as index
        df = df.rename(columns={df.columns[0]: "baseline_algo"})
    for c in ["mean", "median", "p90"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mean"]).sort_values("mean", ascending=False)

    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    colors = color_cycle(len(df))
    plt.barh(df["baseline_algo"].astype(str)[::-1], df["mean"][::-1], color=colors[::-1])
    if "median" in df.columns:
        plt.scatter(df["median"], df["baseline_algo"].astype(str), color=PALETTE[4], s=18, label="median")
        plt.legend(loc="lower right")
    plt.axvline(1.0, linewidth=1)
    plt.title("Summary: mean speedup_total by baseline")
    plt.xlabel("mean(speedup_total)  (baseline/heft)")
    savefig(out_dir / "_viz" / "summary_by_baseline_mean.png")

def plot_all_pairs_speedup(p: Path, out_dir: Path):
    df = read_csv_robust(p)
    if "baseline_algo" not in df.columns or "speedup_total" not in df.columns:
        return
    df["speedup_total"] = pd.to_numeric(df["speedup_total"], errors="coerce")
    df = df.dropna(subset=["speedup_total"])

    # boxplot by baseline (top N by sample count)
    counts = df["baseline_algo"].value_counts()
    baselines = counts.head(16).index.tolist()
    sub = df[df["baseline_algo"].isin(baselines)].copy()

    data = [sub[sub["baseline_algo"] == b]["speedup_total"].values for b in baselines]

    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(data, labels=baselines, patch_artist=True, showfliers=False)
    cols = color_cycle(len(baselines))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
    plt.axhline(1.0, linewidth=1)
    plt.title("Distribution: speedup_total by baseline (top 16 by count)")
    plt.ylabel("speedup_total (baseline/heft)")
    plt.xticks(rotation=30, ha="right")
    savefig(out_dir / "_viz" / "all_pairs_speedup_boxplot.png")

def plot_spearman(p: Path):
    df = read_csv_robust(p)
    if "feature" not in df.columns or "spearman" not in df.columns:
        return
    df["spearman"] = pd.to_numeric(df["spearman"], errors="coerce")
    df = df.dropna(subset=["spearman"]).sort_values("abs_spearman", ascending=True)

    out_dir = p.parent / "_viz"
    plt.figure(figsize=(10, max(4, 0.4 * len(df))))
    colors = [PALETTE[1] if v >= 0 else PALETTE[6] for v in df["spearman"].values]
    plt.barh(df["feature"].astype(str), df["spearman"].values, color=colors)
    plt.axvline(0, linewidth=1)
    plt.title("Spearman correlation with speedup_total")
    plt.xlabel("spearman")
    savefig(out_dir / f"{p.stem}_bar.png")

def plot_drop_importance(p: Path):
    df = read_csv_robust(p)
    if "group" not in df.columns or "drop_in_r2" not in df.columns:
        return
    df["drop_in_r2"] = pd.to_numeric(df["drop_in_r2"], errors="coerce")
    df = df.dropna(subset=["drop_in_r2"]).sort_values("drop_in_r2", ascending=True)

    out_dir = p.parent / "_viz"
    plt.figure(figsize=(10, max(4, 0.4 * len(df))))
    plt.barh(df["group"].astype(str), df["drop_in_r2"].values, color=PALETTE[2])
    plt.axvline(0, linewidth=1)
    plt.title("Drop-column importance (R² drop) on log(speedup_total)")
    plt.xlabel("drop_in_r2 (bigger => more important)")
    savefig(out_dir / f"{p.stem}_bar.png")

def plot_topk(p: Path):
    df = read_csv_robust(p)
    if "speedup_total" not in df.columns:
        return
    df["speedup_total"] = pd.to_numeric(df["speedup_total"], errors="coerce")
    df = df.dropna(subset=["speedup_total"]).sort_values("speedup_total", ascending=False)

    out_dir = p.parent / "_viz"
    y = df["speedup_total"].values
    x = np.arange(1, len(y) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o", color=PALETTE[0])
    plt.axhline(1.0, linewidth=1)
    plt.title(f"TopK speedup_total rank plot ({p.parent.name})")
    plt.xlabel("rank (1=best)")
    plt.ylabel("speedup_total (baseline/heft)")
    savefig(out_dir / f"{p.stem}_rank.png")

def plot_series_csv(p: Path, guess_title: str):
    df = read_csv_robust(p)
    # handle "key,value" style
    if df.shape[1] < 2:
        return

    # pick first non-numeric column as category, first numeric as value
    num_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            num_cols.append(c)
    if not num_cols:
        return
    val_col = num_cols[0]
    cat_col = [c for c in df.columns if c != val_col][0]

    out_dir = p.parent / "_viz"
    plot_barh_categories(df, cat_col, val_col, out_dir / f"{p.stem}_barh.png", guess_title, topn=25,
                         positive_color=PALETTE[3], negative_color=PALETTE[4])

def plot_eta_effects(p: Path, out_dir: Path):
    df = read_csv_robust(p)
    if not set(["baseline_algo", "cat", "eta_squared"]).issubset(df.columns):
        return
    df["eta_squared"] = pd.to_numeric(df["eta_squared"], errors="coerce")
    pv = df.pivot_table(index="cat", columns="baseline_algo", values="eta_squared", aggfunc="mean")
    if pv.empty:
        return

    cmap = ListedColormap(PALETTE)
    bounds = np.linspace(0, 1, 9)
    norm = BoundaryNorm(bounds, cmap.N)
    data = pv.values
    data = np.ma.masked_invalid(data)

    plt.figure(figsize=(12, max(4, 0.35 * len(pv.index))))
    plt.imshow(data, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    plt.colorbar(label="eta_squared (0~1)")
    plt.xticks(range(len(pv.columns)), [str(c) for c in pv.columns], rotation=30, ha="right")
    plt.yticks(range(len(pv.index)), [str(i) for i in pv.index])
    plt.title("Categorical effect size (eta²): baseline_algo × category")
    savefig(out_dir / "_viz" / "eta_squared_heatmap.png")

def plot_mean_speedup_batch_prefill_decode(p: Path):
    df = read_csv_robust(p)
    need = {"batch", "prefill_len", "decode_len", "speedup_total"}
    if not need.issubset(df.columns):
        return
    for c in ["batch", "prefill_len", "decode_len", "speedup_total"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["batch", "prefill_len", "decode_len", "speedup_total"])
    if df.empty:
        return

    out_dir = p.parent / "_viz"
    cmap = ListedColormap(DIVERGING_SPEEDUP)

    # bins on log2(speedup)
    bounds = np.array([-np.inf, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, np.inf])
    norm = BoundaryNorm(bounds, cmap.N)

    for b in sorted(df["batch"].unique()):
        sub = df[df["batch"] == b]
        pv = sub.pivot_table(index="prefill_len", columns="decode_len", values="speedup_total", aggfunc="mean")
        if pv.empty:
            continue
        arr = pv.values
        arr = np.log2(arr)  # diverging around 0 == speedup=1
        arr = np.ma.masked_invalid(arr)

        plt.figure(figsize=(10, 6))
        plt.imshow(arr, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        plt.colorbar(label="log2(speedup_total)")
        plt.xticks(range(len(pv.columns)), [str(c) for c in pv.columns], rotation=30, ha="right")
        plt.yticks(range(len(pv.index)), [str(i) for i in pv.index])
        plt.title(f"Mean speedup_total heatmap (batch={int(b)})")
        plt.xlabel("decode_len")
        plt.ylabel("prefill_len")
        savefig(out_dir / f"{p.stem}_heatmap_b{int(b)}.png")

def dispatch(csv_path: Path, root_out: Path):
    name = csv_path.name
    if name == "summary_by_baseline.csv":
        plot_summary_by_baseline(csv_path, root_out)
    elif name == "all_pairs_speedup.csv":
        plot_all_pairs_speedup(csv_path, root_out)
    elif name == "eta_squared_categorical_effects.csv":
        plot_eta_effects(csv_path, root_out)
    elif name == "spearman_corr_numeric.csv":
        plot_spearman(csv_path)
    elif name == "drop_column_importance_ridge.csv":
        plot_drop_importance(csv_path)
    elif name == "topk_speedup_total.csv":
        plot_topk(csv_path)
    elif name == "mean_speedup_by_batch_prefill_decode.csv":
        plot_mean_speedup_batch_prefill_decode(csv_path)
    elif name in ("mean_speedup_by_hardware_tag.csv", "mean_speedup_by_model_variant.csv"):
        plot_series_csv(csv_path, guess_title=name.replace(".csv",""))
    else:
        # generic fallback: if it looks like key/value -> barh
        try:
            df = read_csv_robust(csv_path)
            if df.shape[1] == 2:
                plot_series_csv(csv_path, guess_title=csv_path.stem)
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="analysis output dir (contains csvs)")
    args = ap.parse_args()

    root_out = Path(args.out)
    csvs = sorted(root_out.rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV found in {root_out}")

    for p in csvs:
        try:
            dispatch(p, root_out)
        except Exception as e:
            print(f"[WARN] {p}: {e}")

    print("DONE. Check *_viz/*.png under:", root_out)

if __name__ == "__main__":
    main()
