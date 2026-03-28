
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================
# python plot_exp1_batch.py
# ============================================================
ROOT_DIR = Path("../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst2_rst2")
FILE_GLOB = "baseline_compare_*.json"

# 算法顺序/显示名
ALGO_ORDER = [
    "pd",
    "attn_on_pim",
    "ianus",
    "facil",
    "attacc",
    "hefthint",
]
ALGO_DISPLAY = {
    "ianus": "PD+FFN",
    "attacc": "PD+Attention",
    "facil": "PD+Linear",
    "pd": "PD",
    "attn_on_pim": "AF",
    "hefthint": "HEFT-Hint",
}

# 模型顺序；None 表示自动排序
MODEL_ORDER = None

# 一行固定 3 个 subplot
SUBPLOTS_PER_ROW = 3
SUBPLOT_ORDER = [
    (1024, 128),
    (1024, 512),
    (1024, 1024),
]

LATENCY_FIELD = "total_time_s"
TIME_UNIT = "s"  # "s" 或 "ms"

# speedup 定义：
# 1) "vs_first_batch": 同一算法、同一模型、同一 Lin/Lout，以最小 batch 为基准
# 2) "vs_ref_algo"  : 同一模型、同一 Lin/Lout、同一 batch，以参考算法为基准
SPEEDUP_MODE = "vs_ref_algo"
SPEEDUP_REF_ALGO = "pd"

# 颜色
BAR_COLORS = ["#aee4ad", "#add9e4", "#bdade4", "#e4adb5"]
LINE_COLOR = "black"

# 字体：Roboto Condensed Regular
# 如果系统已经安装 Roboto Condensed，可保持 None
# 如果没有安装，填本机 RobotoCondensed-Regular.ttf 的绝对路径
FONT_PATH = None
FONT_FAMILY = "Roboto Condensed"
FONT_WEIGHT = "regular"

# 版式
FIGSIZE_PER_SUBPLOT_W = 4.8
FIGSIZE_PER_SUBPLOT_H = 1.55     # 保持整体扁平
BAR_WIDTH = 0.4                  # 当前的一半
BAR_CENTER_STEP = 0.4            # 保持柱子中心间距不变，只缩窄柱宽
BAR_INNER_GAP = 0
ALGO_GROUP_GAP = 0.2
INSET_Y0 = 0.40                  # speedup 轴往下压，和柱状图重合更多
INSET_HEIGHT = 0.38             # speedup 显示区域更大，但不改整图高度
TOP_HEADROOM = 1.12
SPEED_HEADROOM = 1.12

# speedup y 轴范围：用数据自适应范围，让不同 speedup 更明显
SPEEDUP_RANGE_MODE = "tight"   # "tight" or "zero_based"
SPEEDUP_PAD_FRAC = 0.08
SPEEDUP_MIN_PAD = 0.03
SPEEDUP_MIN_SPAN = 0.20

# 子图间距 / 边距
LEFT_MARGIN = 0.07
RIGHT_MARGIN = 0.975
BOTTOM_MARGIN = 0.22
TOP_MARGIN = 0.80
WSPACE = 0.08                    # 不同模型列之间留一点距离
HSPACE = 0.24                    # 同一模型列内的三个子图之间留一点距离

# 线条
LINE_WIDTH = 1.4
MARKER = "o"
MARKER_SIZE = 3.4

# 边框
AX_SPINE_COLOR = "black"
AX_SPINE_WIDTH = 1.0
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.8

# 字号
FONTSIZE_TITLE = 12
FONTSIZE_SUPTITLE = 20
FONTSIZE_YLABEL = 12
FONTSIZE_TICK = 11
FONTSIZE_BATCH_TICK = 11
FONTSIZE_ALGO_GROUP = 11
FONTSIZE_MODEL_ROW = 12
FONTSIZE_MODEL_COL = 13
FONTSIZE_LEGEND = 11
FONTSIZE_SPEED_TICK = 10
FONTSIZE_SPEED_LABEL = 11
FONTSIZE_SPEED_VALUE = 8

# 旋转角度
ALGO_LABEL_ROTATION = 0
BATCH_LABEL_ROTATION = 0

# x 轴双层标签
BATCH_TICK_PAD = 1
ALGO_AXIS_OUTWARD = 12
ALGO_TICK_PAD = 2
MODEL_TITLE_FIG_PAD = 0.06
LEGEND_BBOX_Y = 1.04
LEGEND_NCOL = None
LEGEND_COLUMNSPACING = 1.2
LEGEND_HANDLETEXTPAD = 0.5

# x 轴是否显示 batch 标注（这里按需求关掉，只保留算法名）
SHOW_BATCH_TICK_LABELS = False

# speedup 数值
SHOW_SPEEDUP_VALUE = True
SPEEDUP_VALUE_FMT = "{:.2f}"
SPEEDUP_VALUE_YOFFSET_FRAC = 0.03
SPEEDUP_VALUE_XOFFSET_PT = 4
SPEEDUP_VALUE_YOFFSET_PT = 3
SPEEDUP_VALUE_ROTATION = 45

# 其他
SHOW_SUPTITLE = False
SUPTITLE = "Latency + Speedup"
SHOW_SPEEDUP_LABEL_ONLY_ON_LAST_VISIBLE_COL = True
SHARE_LATENCY_Y_WITHIN_MODEL = True
SHARE_SPEEDUP_Y_WITHIN_MODEL = True
SHOW_XLABEL_ONLY_ON_BOTTOM_ROW = True
SHOW_SPEEDUP_TICKS_ONLY_ON_LAST_VISIBLE_COL = True
TITLE_PAD = 8

SAVE_PATH = "../../figs/exp1/llama_batch.pdf"
DPI = 300
SAVE_TRANSPARENT = False


# ============================================================
# 全局字体
# ============================================================
def setup_global_font():
    font_name = FONT_FAMILY
    if FONT_PATH:
        fm.fontManager.addfont(FONT_PATH)
        font_name = fm.FontProperties(fname=FONT_PATH).get_name()

    plt.rcParams["font.family"] = font_name
    plt.rcParams["font.weight"] = FONT_WEIGHT
    plt.rcParams["axes.titleweight"] = FONT_WEIGHT
    plt.rcParams["axes.labelweight"] = FONT_WEIGHT
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


# ============================================================
# 数据读取
# ============================================================
def normalize_algo(policy: str) -> str:
    return str(policy).split(":", 1)[-1].strip()


def parse_lin_lout(json_path: Path):
    m = re.search(r"(\d+)x(\d+)", json_path.stem)
    if not m:
        raise ValueError(f"无法从文件名解析 Lin/Lout: {json_path.name}")
    return int(m.group(1)), int(m.group(2))


def parse_model_batch_from_path(json_path: Path):
    """
    例如：
    llama_7b_fp16_b8_s2 -> model=llama_7b_fp16, batch=8
    qwen_1.8b_fp16_b1_s2 -> model=qwen_1.8b_fp16, batch=1
    """
    pat = re.compile(r"(?P<model>.+?)_b(?P<batch>\d+)(?:_s\d+)?$")
    for part in reversed(json_path.parts):
        m = pat.match(part)
        if m:
            return m.group("model"), int(m.group("batch"))
    return None, None


def parse_model_batch_from_json(obj: dict):
    cfg = obj.get("config", {})
    family = cfg.get("model_family")
    variant = cfg.get("model_variant")
    dtype = cfg.get("dtype")
    batch = cfg.get("batch")

    if family and variant and dtype and batch is not None:
        return f"{family}_{variant}_{dtype}", int(batch)

    result_dir = cfg.get("result_dir", "")
    if result_dir:
        pat = re.compile(r"(?P<model>.+?)_b(?P<batch>\d+)(?:_s\d+)?$")
        for part in reversed(Path(result_dir).parts):
            m = pat.match(part)
            if m:
                return m.group("model"), int(m.group("batch"))

    return None, None


def read_latency(item: dict):
    if LATENCY_FIELD in item:
        return float(item[LATENCY_FIELD])
    if "latency" in item:
        return float(item["latency"])
    if "total_time_s" in item:
        return float(item["total_time_s"])
    if "prefill_time_s" in item and "decode_time_s" in item:
        return float(item["prefill_time_s"]) + float(item["decode_time_s"])
    raise KeyError(f"结果里没有找到 {LATENCY_FIELD} / latency / total_time_s")


def to_unit(x):
    if TIME_UNIT == "s":
        return x
    if TIME_UNIT == "ms":
        return x * 1000.0
    raise ValueError("TIME_UNIT 只能是 's' 或 'ms'")


def load_dataframe(root_dir: Path):
    rows = []
    files = sorted(root_dir.rglob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"{root_dir} 下没找到 {FILE_GLOB}")

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)

        lin, lout = parse_lin_lout(fp)

        model, batch = parse_model_batch_from_path(fp)
        if model is None or batch is None:
            model, batch = parse_model_batch_from_json(obj)
        if model is None or batch is None:
            raise ValueError(f"无法从路径或 JSON config 解析 model/batch: {fp}")

        for item in obj.get("results", []):
            algo = normalize_algo(item.get("policy", ""))
            if ALGO_ORDER and algo not in ALGO_ORDER:
                continue

            rows.append({
                "model": model,
                "batch": int(batch),
                "lin": int(lin),
                "lout": int(lout),
                "algo": algo,
                "latency": to_unit(read_latency(item)),
                "file": str(fp),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("读出来的数据为空，请检查 FILE_GLOB / ALGO_ORDER / LATENCY_FIELD")
    return df


def compute_speedup(df: pd.DataFrame):
    df = df.copy()

    if SPEEDUP_MODE == "vs_first_batch":
        base = (
            df.sort_values("batch")
              .groupby(["model", "lin", "lout", "algo"], as_index=False)
              .first()[["model", "lin", "lout", "algo", "latency"]]
              .rename(columns={"latency": "base_latency"})
        )
        df = df.merge(base, on=["model", "lin", "lout", "algo"], how="left")
        df["speedup"] = df["base_latency"] / df["latency"]

    elif SPEEDUP_MODE == "vs_ref_algo":
        ref = (
            df[df["algo"] == SPEEDUP_REF_ALGO][["model", "lin", "lout", "batch", "latency"]]
            .rename(columns={"latency": "ref_latency"})
        )
        df = df.merge(ref, on=["model", "lin", "lout", "batch"], how="left")
        df["speedup"] = df["ref_latency"] / df["latency"]

    else:
        raise ValueError("SPEEDUP_MODE 只能是 'vs_first_batch' 或 'vs_ref_algo'")

    return df


def compute_speedup_ylim(values: np.ndarray):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    spd_min = float(np.nanmin(vals))
    spd_max = float(np.nanmax(vals))

    if SPEEDUP_RANGE_MODE == "tight":
        span = max(spd_max - spd_min, SPEEDUP_MIN_SPAN)
        pad = max(span * SPEEDUP_PAD_FRAC, SPEEDUP_MIN_PAD)
        lower = max(0.0, spd_min - pad)
        upper = spd_max + pad
    elif SPEEDUP_RANGE_MODE == "zero_based":
        lower = 0.0
        upper = spd_max * SPEED_HEADROOM if spd_max > 0 else 1.0
    else:
        raise ValueError("SPEEDUP_RANGE_MODE 只能是 'tight' 或 'zero_based'")

    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


# ============================================================
# 绘图
# ============================================================
def build_orders(df: pd.DataFrame):
    models = sorted(df["model"].unique()) if MODEL_ORDER is None else [m for m in MODEL_ORDER if m in set(df["model"])]
    algos = [a for a in ALGO_ORDER if a in set(df["algo"])] if ALGO_ORDER else sorted(df["algo"].unique())
    batches = sorted(df["batch"].unique())

    if SUBPLOT_ORDER is None:
        pairs = sorted(set(zip(df["lin"], df["lout"])), key=lambda x: (x[0], x[1]))
        if len(pairs) > SUBPLOTS_PER_ROW:
            print(f"[Warning] 自动检测到 {len(pairs)} 个 Lin/Lout，只显示前 {SUBPLOTS_PER_ROW} 个；如需固定顺序，请设置 SUBPLOT_ORDER")
            pairs = pairs[:SUBPLOTS_PER_ROW]
    else:
        valid = set(zip(df["lin"], df["lout"]))
        pairs = [p for p in SUBPLOT_ORDER if p in valid]

    return models, algos, batches, pairs


def make_x_positions(algos, batches):
    pos = {}
    centers = {}
    cursor = 0.0
    step = BAR_CENTER_STEP + BAR_INNER_GAP
    for algo in algos:
        xs = []
        for i, b in enumerate(batches):
            x = cursor + i * step
            pos[(algo, b)] = x
            xs.append(x)
        centers[algo] = float(np.mean(xs))
        cursor = xs[-1] + step + ALGO_GROUP_GAP
    return pos, centers


def style_axis_frame(ax):
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(AX_SPINE_COLOR)
        ax.spines[side].set_linewidth(AX_SPINE_WIDTH)


def add_group_separators(ax, algos, batches, pos):
    for i in range(len(algos) - 1):
        xl = pos[(algos[i], batches[-1])]
        xr = pos[(algos[i + 1], batches[0])]
        ax.axvline((xl + xr) / 2.0, color="0.75", linewidth=0.8, linestyle="--", zorder=0)


def add_algo_axis(ax, algos, centers):
    secax = ax.secondary_xaxis("bottom")
    secax.set_xticks(
        [centers[a] for a in algos],
        [ALGO_DISPLAY.get(a, a) for a in algos],
    )
    secax.spines["bottom"].set_position(("outward", ALGO_AXIS_OUTWARD))
    secax.spines["bottom"].set_visible(False)
    secax.tick_params(
        axis="x",
        length=0,
        pad=ALGO_TICK_PAD,
        labelsize=FONTSIZE_ALGO_GROUP,
        rotation=ALGO_LABEL_ROTATION,
    )
    return secax


def add_model_column_titles(fig, axes, models):
    for c, model in enumerate(models):
        bbox = axes[0][c].get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2.0,
            bbox.y1 + MODEL_TITLE_FIG_PAD,
            model,
            ha="center",
            va="bottom",
            fontsize=FONTSIZE_MODEL_COL,
            fontweight=FONT_WEIGHT,
        )


def annotate_speedup_values(ax2, xs, ys, ylim):
    if not SHOW_SPEEDUP_VALUE:
        return

    y0, y1 = ylim
    y_offset = (y1 - y0) * SPEEDUP_VALUE_YOFFSET_FRAC
    for x, y in zip(xs, ys):
        if np.isnan(y):
            continue
        ax2.annotate(
            SPEEDUP_VALUE_FMT.format(y),
            xy=(x, y + y_offset),
            xytext=(SPEEDUP_VALUE_XOFFSET_PT, SPEEDUP_VALUE_YOFFSET_PT),
            textcoords="offset points",
            ha="left",
            va="bottom",
            rotation=SPEEDUP_VALUE_ROTATION,
            rotation_mode="anchor",
            fontsize=FONTSIZE_SPEED_VALUE,
            color=LINE_COLOR,
            clip_on=False,
            zorder=5,
        )


def plot_one_subplot(
    ax,
    sub_df,
    model,
    lin,
    lout,
    algos,
    batches,
    pos,
    centers,
    xlim,
    latency_ylim=None,
    speedup_ylim=None,
    show_ylabel=False,
    show_yticklabels=True,
    show_xlabels=True,
    show_speedup_axis=True,
    show_speedup_label=False,
):
    latency_map = {(r.algo, int(r.batch)): float(r.latency) for r in sub_df.itertuples()}
    speedup_map = {(r.algo, int(r.batch)): float(r.speedup) for r in sub_df.itertuples()}

    # 柱状图：颜色按 batch size，黑色边框
    for bi, b in enumerate(batches):
        xs, ys = [], []
        for algo in algos:
            xs.append(pos[(algo, b)])
            ys.append(latency_map.get((algo, b), np.nan))

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        mask = ~np.isnan(ys)
        if mask.any():
            ax.bar(
                xs[mask],
                ys[mask],
                width=BAR_WIDTH,
                color=BAR_COLORS[bi % len(BAR_COLORS)],
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=2,
            )

    xticks = [pos[(algo, b)] for algo in algos for b in batches]

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    if SHOW_BATCH_TICK_LABELS and show_xlabels:
        xticklabels = [f"b{b}" for algo in algos for b in batches]
        ax.set_xticklabels(xticklabels, fontsize=FONTSIZE_BATCH_TICK, rotation=BATCH_LABEL_ROTATION)
        ax.tick_params(axis="x", pad=BATCH_TICK_PAD)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_title(f"Lin={lin}, Lout={lout}", fontsize=FONTSIZE_TITLE, pad=TITLE_PAD)

    if latency_ylim is not None:
        ax.set_ylim(*latency_ylim)

    if show_ylabel:
        unit_str = "s" if TIME_UNIT == "s" else "ms"
        ax.set_ylabel(f"Latency ({unit_str})", fontsize=FONTSIZE_YLABEL)

    if not show_yticklabels:
        ax.tick_params(axis="y", labelleft=False)

    if show_xlabels:
        add_algo_axis(ax, algos, centers)
    else:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    add_group_separators(ax, algos, batches, pos)
    style_axis_frame(ax)

    # speedup inset：放到更靠下一点的位置
    ax2 = ax.inset_axes([0.0, INSET_Y0, 1.0, INSET_HEIGHT])
    ax2.patch.set_alpha(0.0)

    for algo in algos:
        xs, ys = [], []
        for b in batches:
            xs.append(pos[(algo, b)])
            ys.append(speedup_map.get((algo, b), np.nan))

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        mask = ~np.isnan(ys)
        if mask.any():
            ax2.plot(
                xs[mask],
                ys[mask],
                color=LINE_COLOR,
                linewidth=LINE_WIDTH,
                marker=MARKER,
                markersize=MARKER_SIZE,
                zorder=4,
            )
            annotate_speedup_values(
                ax2=ax2,
                xs=xs[mask],
                ys=ys[mask],
                ylim=speedup_ylim,
            )

    ax2.set_xlim(*xlim)
    if speedup_ylim is not None:
        ax2.set_ylim(*speedup_ylim)

    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(show_speedup_axis)
    if show_speedup_axis:
        ax2.spines["right"].set_color(AX_SPINE_COLOR)
        ax2.spines["right"].set_linewidth(AX_SPINE_WIDTH)

    ax2.yaxis.tick_right()
    ax2.tick_params(axis="x", bottom=False, labelbottom=False)

    if show_speedup_axis:
        ax2.tick_params(axis="y", labelsize=FONTSIZE_SPEED_TICK)
    else:
        ax2.tick_params(axis="y", right=False, labelright=False)

    if show_speedup_axis and show_speedup_label:
        ax2.set_ylabel("Speedup", fontsize=FONTSIZE_SPEED_LABEL, rotation=270, labelpad=12)
        ax2.yaxis.set_label_position("right")


def plot_grid(df: pd.DataFrame):
    setup_global_font()

    df = compute_speedup(df)
    models, algos, batches, pairs = build_orders(df)
    if not models or not algos or not batches or not pairs:
        raise ValueError("模型 / 算法 / batch / LinLout 为空，请检查数据")

    # 新布局：一列一个模型；同一列里垂直堆叠不同的 Lin/Lout
    nrows, ncols = len(pairs), len(models)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIGSIZE_PER_SUBPLOT_W * ncols, FIGSIZE_PER_SUBPLOT_H * nrows),
        squeeze=False,
        sharex="col",
        sharey="col",
    )

    pos, centers = make_x_positions(algos, batches)

    all_xs = [pos[(algo, b)] for algo in algos for b in batches]
    x_margin = max(BAR_CENTER_STEP, BAR_WIDTH) * 0.65
    xlim = (min(all_xs) - x_margin, max(all_xs) + x_margin)

    latency_ylim_map, speedup_ylim_map = {}, {}
    pair_set = set(pairs)
    for model in models:
        col_df = df[
            (df["model"] == model)
            & (df.apply(lambda x: (x["lin"], x["lout"]) in pair_set, axis=1))
        ]

        lat_max = float(np.nanmax(col_df["latency"].values)) if not col_df.empty else 1.0
        latency_ylim_map[model] = (0.0, lat_max * TOP_HEADROOM if lat_max > 0 else 1.0)
        speedup_ylim_map[model] = compute_speedup_ylim(col_df["speedup"].values)

    for c, model in enumerate(models):
        for r, (lin, lout) in enumerate(pairs):
            ax = axes[r][c]
            sub_df = df[
                (df["model"] == model)
                & (df["lin"] == lin)
                & (df["lout"] == lout)
            ].copy()

            show_xlabels = (r == nrows - 1) if SHOW_XLABEL_ONLY_ON_BOTTOM_ROW else True
            show_ylabel = (c == 0 and r == nrows // 2)
            show_speedup_axis = (r == 0)
            show_speedup_label = (r == 0)

            plot_one_subplot(
                ax=ax,
                sub_df=sub_df,
                model=model,
                lin=lin,
                lout=lout,
                algos=algos,
                batches=batches,
                pos=pos,
                centers=centers,
                xlim=xlim,
                latency_ylim=latency_ylim_map[model] if SHARE_LATENCY_Y_WITHIN_MODEL else None,
                speedup_ylim=speedup_ylim_map[model] if SHARE_SPEEDUP_Y_WITHIN_MODEL else (0.0, 1.0),
                show_ylabel=show_ylabel,
                show_yticklabels=True,
                show_xlabels=show_xlabels,
                show_speedup_axis=show_speedup_axis,
                show_speedup_label=show_speedup_label,
            )

    legend_handles = [
        Patch(
            facecolor=BAR_COLORS[i % len(BAR_COLORS)],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            label=f"b{b}",
        )
        for i, b in enumerate(batches)
    ]
    legend_handles.append(
        Line2D([0], [0], color=LINE_COLOR, marker=MARKER, linewidth=LINE_WIDTH, label="Speedup")
    )

    legend_ncol = len(legend_handles) if LEGEND_NCOL is None else min(LEGEND_NCOL, len(legend_handles))
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_BBOX_Y),
        ncol=legend_ncol,
        frameon=False,
        fontsize=FONTSIZE_LEGEND,
        columnspacing=LEGEND_COLUMNSPACING,
        handletextpad=LEGEND_HANDLETEXTPAD,
    )

    if SHOW_SUPTITLE:
        fig.suptitle(SUPTITLE, fontsize=FONTSIZE_SUPTITLE, y=1.04)

    fig.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        bottom=BOTTOM_MARGIN,
        top=TOP_MARGIN,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    add_model_column_titles(fig, axes, models)

    plt.savefig(SAVE_PATH, dpi=DPI, bbox_inches="tight", transparent=SAVE_TRANSPARENT)
    plt.show()
    print(f"Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    df = load_dataframe(ROOT_DIR)
    plot_grid(df)
