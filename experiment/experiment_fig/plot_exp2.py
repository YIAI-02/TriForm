#!/usr/bin/env python3
"""
python plot_exp2.py \
  --panel "1 NPU 2 PIM=../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst64_rst64" \
  --panel "1 NPU 4 PIM=../../algorithms/output/exp2/4shards/hw_hardware_1npu_4aim/sst64_rst64" \
  --panel "1 NPU 8 PIM=../../algorithms/output/exp2/8shards/hw_hardware_1npu_8aim/sst64_rst64" \
  --model "llama_7b" \
  --batches 1 4 8 16 \
  --prefills 128 256 512 1024\
  --decodes 128 256 512 1024 \
  --baseline pd \
  --reference-panel "1 NPU 2 PIM" \
  --output ../../figs/exp2/llama_7b_heatmap.pdf

"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_COLORS = [
    "#aee4ad",
    "#e4ddad",
    "#e4adb5",
    "#bdade4",
]
AUTO_ALGORITHMS = ["heft", "hefthint"]
HARDWARE_UNIT_PATTERN = re.compile(r"(?P<count>\d+)\s*[_\-\s]?(?P<unit>gpu|npu|aim|pim)\b", re.IGNORECASE)

_JSON_CACHE: Dict[Path, dict] = {}
_PANEL_FILE_CACHE: Dict[Path, List[Path]] = {}


def parse_panel(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f'--panel 参数格式必须是 "标签=路径"，收到: {text}'
        )
    label, path = text.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(
            f'--panel 参数格式必须是 "标签=路径"，收到: {text}'
        )
    return label, path


def load_json(path: Path) -> dict:
    path = path.expanduser().resolve()
    if path not in _JSON_CACHE:
        with path.open("r", encoding="utf-8") as f:
            _JSON_CACHE[path] = json.load(f)
    return _JSON_CACHE[path]


def get_panel_candidate_files(panel_root: Path) -> List[Path]:
    panel_root = panel_root.expanduser().resolve()
    if panel_root not in _PANEL_FILE_CACHE:
        if not panel_root.exists():
            _PANEL_FILE_CACHE[panel_root] = []
        else:
            _PANEL_FILE_CACHE[panel_root] = list(panel_root.rglob("baseline_compare_*.json"))
    return _PANEL_FILE_CACHE[panel_root]


def normalize_policy_name(name: Optional[str]) -> str:
    if name is None:
        return ""
    name = str(name).strip()
    if ":" in name:
        name = name.split(":", 1)[1]
    return name.strip().lower()


def normalize_text_key(value: Optional[object]) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_model_name(value: Optional[object]) -> str:
    return normalize_text_key(value)


def stringify_option(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = " ".join(str(x) for x in value)
    value = str(value).strip()
    return value or None


def candidate_matches_model(path: Path, data: dict, model: Optional[str]) -> bool:
    model_key = normalize_model_name(model)
    if not model_key:
        return True

    compact_model_key = model_key.replace("_", "")
    pattern = re.compile(rf"(^|_){re.escape(model_key)}(_|$)")
    cfg = data.get("config", {})
    search_texts = [str(path)]

    for key in ("result_dir", "model", "model_name", "shape_file"):
        value = cfg.get(key)
        if value:
            search_texts.append(str(value))

    family = cfg.get("model_family")
    variant = cfg.get("model_variant")
    if family or variant:
        search_texts.append(f"{family or ''}_{variant or ''}")

    for text in search_texts:
        normalized = normalize_text_key(text)
        if pattern.search(normalized):
            return True
        if compact_model_key and compact_model_key in normalized.replace("_", ""):
            return True
    return False


def find_policy_result(data: dict, algorithm_name: str) -> Optional[dict]:
    target = normalize_policy_name(algorithm_name)
    for item in data.get("results", []):
        policy = normalize_policy_name(item.get("policy"))
        if policy == target:
            return item
    return None


def score_candidate(
    path: Path,
    data: dict,
    batch: int,
    prefill: int,
    decode: int,
    model: Optional[str] = None,
) -> int:
    score = 0
    cfg = data.get("config", {})
    if cfg.get("batch") == batch:
        score += 100
    if cfg.get("prefill_len") == prefill:
        score += 100
    if cfg.get("decode_len") == decode:
        score += 100

    path_text = str(path).lower()
    if path.name.lower() == f"baseline_compare_{prefill}x{decode}.json".lower():
        score += 10
    if re.search(rf"(^|[_/\-])b{batch}([_/\-]|$)", path_text):
        score += 5
    if cfg.get("result_dir") and f"b{batch}" in str(cfg.get("result_dir")).lower():
        score += 3
    if model and candidate_matches_model(path, data, model):
        score += 50
    return score


def find_case_json(
    panel_root: Path,
    batch: int,
    prefill: int,
    decode: int,
    model: Optional[str] = None,
    verbose: bool = False,
) -> Optional[Path]:
    panel_root = panel_root.expanduser().resolve()
    exact_name = f"baseline_compare_{prefill}x{decode}.json"

    all_candidates = get_panel_candidate_files(panel_root)
    if not all_candidates:
        return None

    exact_candidates = [path for path in all_candidates if path.name == exact_name]
    candidate_groups: List[List[Path]] = []
    if exact_candidates:
        candidate_groups.append(exact_candidates)
    if len(exact_candidates) != len(all_candidates):
        candidate_groups.append(all_candidates)
    elif not exact_candidates:
        candidate_groups.append(all_candidates)

    for candidates in candidate_groups:

        scored: List[Tuple[int, Path]] = []
        for candidate_path in candidates:
            try:
                data = load_json(candidate_path)
            except Exception:
                continue

            cfg = data.get("config", {})
            if cfg.get("batch") not in (None, batch):
                continue
            if cfg.get("prefill_len") not in (None, prefill):
                continue
            if cfg.get("decode_len") not in (None, decode):
                continue
            if not candidate_matches_model(candidate_path, data, model):
                continue

            score = score_candidate(candidate_path, data, batch, prefill, decode, model=model)
            scored.append((score, candidate_path))

        if scored:
            scored.sort(key=lambda x: (-x[0], len(str(x[1]))))
            if verbose:
                extra = f", model={model}" if model else ""
                print(
                    f"[find_case_json] batch={batch}, prefill={prefill}, decode={decode}{extra} -> {scored[0][1]}"
                )
            return scored[0][1]

    if verbose and model:
        print(
            f"[find_case_json] batch={batch}, prefill={prefill}, decode={decode}, model={model} -> no match"
        )
    return None


def format_latency(seconds: Optional[float]) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "NA"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 10:
        return f"{seconds:.3f}s"
    return f"{seconds:.2f}s"


def format_ratio(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.2f}"


def format_compact_annotation(value: float, fmt: str) -> str:
    try:
        if "{" in fmt:
            return fmt.format(value)
        return format(value, fmt)
    except Exception:
        return f"{value:.2f}"


def luminance(rgb: Tuple[float, float, float]) -> float:
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def pick_text_color(cmap, norm, value: float) -> str:
    rgba = cmap(norm(value))
    return "white" if luminance(rgba[:3]) < 0.55 else "black"


def select_algorithm_result(
    data: dict,
    algorithm: Optional[str],
    latency_field: str,
    path: Path,
) -> dict:
    if algorithm:
        algo = find_policy_result(data, algorithm)
        if algo is None:
            raise KeyError(f"算法 {algorithm} 在 {path} 中不存在")
        return algo

    candidates = []
    for priority, candidate_name in enumerate(AUTO_ALGORITHMS):
        candidate = find_policy_result(data, candidate_name)
        if candidate is None:
            continue
        try:
            candidate_latency = float(candidate[latency_field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isnan(candidate_latency):
            continue
        candidates.append((candidate_latency, priority, candidate))

    if not candidates:
        raise KeyError(
            f"默认模式下在 {path} 中找不到可用算法：{', '.join(AUTO_ALGORITHMS)}"
        )

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def get_metric_from_file(
    path: Path,
    algorithm: Optional[str],
    baseline_algorithm: str,
    latency_field: str,
) -> Dict[str, float]:
    data = load_json(path)
    algo = select_algorithm_result(data, algorithm, latency_field, path)
    base = find_policy_result(data, baseline_algorithm)

    if base is None:
        raise KeyError(f"baseline 算法 {baseline_algorithm} 在 {path} 中不存在")

    algo_latency = float(algo[latency_field])
    base_latency = float(base[latency_field])

    return {
        "algo_latency": algo_latency,
        "base_latency": base_latency,
        "speedup_vs_baseline": base_latency / algo_latency if algo_latency != 0 else float("nan"),
    }


def collect_panel_data(
    panel_root: Path,
    batch: int,
    prefill_lengths: List[int],
    decode_lengths: List[int],
    algorithm: Optional[str],
    baseline_algorithm: str,
    latency_field: str,
    model: Optional[str] = None,
    verbose: bool = False,
):
    speedup = np.full((len(prefill_lengths), len(decode_lengths)), np.nan, dtype=float)
    latency = np.full((len(prefill_lengths), len(decode_lengths)), np.nan, dtype=float)
    source_files: Dict[Tuple[int, int], Optional[Path]] = {}

    for i, prefill in enumerate(prefill_lengths):
        for j, decode in enumerate(decode_lengths):
            case_path = find_case_json(panel_root, batch, prefill, decode, model=model, verbose=verbose)
            source_files[(i, j)] = case_path
            if case_path is None:
                continue
            try:
                metric = get_metric_from_file(case_path, algorithm, baseline_algorithm, latency_field)
            except Exception as e:
                if verbose:
                    print(f"[WARN] {e}")
                continue
            speedup[i, j] = metric["speedup_vs_baseline"]
            latency[i, j] = metric["algo_latency"]

    return speedup, latency, source_files


def compute_speedup_vs_reference(
    panel_latency: np.ndarray,
    reference_latency: np.ndarray,
) -> np.ndarray:
    out = np.full_like(panel_latency, np.nan, dtype=float)
    mask = (~np.isnan(panel_latency)) & (~np.isnan(reference_latency)) & (panel_latency != 0)
    out[mask] = reference_latency[mask] / panel_latency[mask]
    return out


def build_reference_speedup_maps(
    latencies: Dict[Tuple[int, str], np.ndarray],
    batches: List[int],
    panel_labels: List[str],
    reference_panel: str,
) -> Dict[Tuple[int, str], np.ndarray]:
    out: Dict[Tuple[int, str], np.ndarray] = {}
    for batch in batches:
        reference_latency = latencies[(batch, reference_panel)]
        for panel_label in panel_labels:
            out[(batch, panel_label)] = compute_speedup_vs_reference(
                latencies[(batch, panel_label)],
                reference_latency,
            )
    return out


def infer_vmin_vmax(
    all_heatmaps: List[np.ndarray],
    vmin: Optional[float],
    vmax: Optional[float],
) -> Tuple[float, float]:
    valid_arrays = [arr[~np.isnan(arr)] for arr in all_heatmaps if np.any(~np.isnan(arr))]
    values = np.concatenate(valid_arrays, axis=0) if valid_arrays else np.array([1.0])
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def build_colormap(colors: List[str]) -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list("user_custom_heatmap", colors, N=256)
    cmap.set_bad(color="#f2f2f2")
    return cmap


def draw_annotated_heatmap(
    ax,
    data: np.ndarray,
    latency: np.ndarray,
    speedup_vs_reference: np.ndarray,
    decode_lengths: List[int],
    prefill_lengths: List[int],
    title: str,
    cmap,
    norm,
):
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto", origin="lower")
    ax.set_title(title, fontsize=11, pad=3)
    ax.set_xticks(range(len(decode_lengths)))
    ax.set_xticklabels([str(x) for x in decode_lengths], fontsize=10)
    ax.set_yticks(range(len(prefill_lengths)))
    ax.set_yticklabels([str(y) for y in prefill_lengths], fontsize=10)

    ax.set_xticks(np.arange(-0.5, len(decode_lengths), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(prefill_lengths), 1), minor=True)
    ax.grid(which="minor", color="#aaaaaa", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            lat = latency[i, j]
            ref = speedup_vs_reference[i, j]
            if np.isnan(value):
                cell_text = "NA"
                text_color = "black"
            else:
                cell_text = (
                    f"{format_ratio(value)}\n"
                    f"{format_latency(lat)}\n"
                    f"({format_ratio(ref)}x)"
                )
                text_color = pick_text_color(cmap, norm, float(value))
            ax.text(
                j,
                i,
                cell_text,
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
                linespacing=1.12,
                fontweight="semibold",
            )
    return im


def draw_compact_heatmap(
    ax,
    data: np.ndarray,
    decode_lengths: List[int],
    prefill_lengths: List[int],
    cmap,
    norm,
    show_xlabels: bool,
    show_ylabels_right: bool,
    tick_fontsize: float = 7,
    x_tick_rotation: float = 90,
    annotate: bool = False,
    annotation_fmt: str = "{:.2f}",
    annotation_fontsize: float = 6,
):
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="equal", origin="lower")

    ax.set_xticks(range(len(decode_lengths)))
    if show_xlabels:
        ax.set_xticklabels([str(x) for x in decode_lengths], fontsize=tick_fontsize, rotation=x_tick_rotation)
        for label in ax.get_xticklabels():
            label.set_ha("center")
            label.set_va("top")
    else:
        ax.set_xticklabels([])

    ax.set_yticks(range(len(prefill_lengths)))
    if show_ylabels_right:
        ax.set_yticklabels([str(y) for y in prefill_lengths], fontsize=tick_fontsize)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", labelright=True, labelleft=False, right=False, left=False, length=0, pad=1)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", labelright=False, labelleft=False, right=False, left=False, length=0)

    ax.tick_params(axis="x", labelbottom=show_xlabels, bottom=False, top=False, length=0, pad=1)

    ax.set_xticks(np.arange(-0.5, len(decode_lengths), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(prefill_lengths), 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linestyle="-", linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if np.isnan(value):
                    text = "NA"
                    text_color = "black"
                else:
                    text = format_compact_annotation(float(value), annotation_fmt)
                    text_color = pick_text_color(cmap, norm, float(value))
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=annotation_fontsize,
                    color=text_color,
                    fontweight="semibold",
                )
    return im


def add_group_title(fig, axes, text: str, dy: float = 0.02, fontsize: int = 16):
    valid_axes = [ax for ax in axes if ax is not None]
    if not valid_axes:
        return
    x0 = min(ax.get_position().x0 for ax in valid_axes)
    x1 = max(ax.get_position().x1 for ax in valid_axes)
    y1 = max(ax.get_position().y1 for ax in valid_axes)
    fig.text((x0 + x1) / 2, y1 + dy, text, ha="center", va="bottom", fontsize=fontsize)


def load_config_file(path: Optional[str]) -> dict:
    if not path:
        return {}
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_hardware_signature(text: str) -> Optional[Tuple[int, int]]:
    counts = {"gpu": 0, "npu": 0, "aim": 0, "pim": 0}
    found = False
    for match in HARDWARE_UNIT_PATTERN.finditer(text.lower()):
        count = int(match.group("count"))
        unit = match.group("unit").lower()
        counts[unit] += count
        found = True
    if not found:
        return None
    device_count = counts["gpu"] + counts["npu"]
    memory_count = counts["aim"] + counts["pim"]
    return device_count, memory_count


def panel_hardware_score(label: str, path: Path) -> Tuple[float, float, float, str]:
    signature = parse_hardware_signature(label)
    if signature is None:
        signature = parse_hardware_signature(str(path))
    if signature is None:
        return (float("inf"), float("inf"), float("inf"), label)

    device_count, memory_count = signature
    if device_count > 0 and memory_count > 0:
        total_units = device_count * memory_count
    else:
        total_units = device_count + memory_count
    return (float(total_units), float(device_count), float(memory_count), label)


def infer_min_hardware_panel(panels: Dict[str, Path], verbose: bool = False) -> str:
    scored = [(panel_hardware_score(label, path), label) for label, path in panels.items()]
    scored.sort(key=lambda item: item[0])
    selected = scored[0][1]
    if verbose:
        pretty_scores = ", ".join(f"{label}:{score[:3]}" for score, label in scored)
        print(f"[infer_min_hardware_panel] selected={selected}; scores={pretty_scores}")
    return selected


def merge_settings(args, cfg: dict) -> dict:
    panels = dict(cfg.get("panels", {}))
    for label, path in (args.panel or []):
        panels[label] = path

    colors = args.colors or cfg.get("colors") or DEFAULT_COLORS

    cfg_model = cfg.get("model")
    if not cfg_model:
        family = stringify_option(cfg.get("model_family"))
        variant = stringify_option(cfg.get("model_variant"))
        if family and variant:
            cfg_model = f"{family} {variant}"

    merged = {
        "panels": panels,
        "batches": args.batches or cfg.get("batches"),
        "prefill_lengths": args.prefills or cfg.get("prefill_lengths"),
        "decode_lengths": args.decodes or cfg.get("decode_lengths"),
        "algorithm": args.algorithm or cfg.get("algorithm"),
        "model": stringify_option(args.model) or stringify_option(cfg_model),
        "baseline_algorithm": args.baseline or cfg.get("baseline_algorithm", "pd"),
        "reference_panel": args.reference_panel or cfg.get("reference_panel"),
        "latency_field": args.latency_field or cfg.get("latency_field", "total_time_s"),
        "title": args.title or cfg.get("title", "Experiment: Speedup"),
        "subtitle": args.subtitle or cfg.get("subtitle", ""),
        "output": args.output or cfg.get("output", "custom_heatmap.png"),
        "batch_cols": args.batch_cols or cfg.get("batch_cols", 2),
        "figsize_scale": args.figsize_scale or cfg.get("figsize_scale", 1.0),
        "colors": colors,
        "annotated_cbar_label": args.cbar_label or cfg.get("annotated_cbar_label") or cfg.get("cbar_label"),
        "compact_cbar_label": cfg.get("compact_cbar_label"),
        "compact_cell_width": args.compact_cell_width if args.compact_cell_width is not None else cfg.get("compact_cell_width", 0.10),
        "compact_cell_height": args.compact_cell_height if args.compact_cell_height is not None else cfg.get("compact_cell_height", 0.10),
        "compact_tick_fontsize": args.compact_tick_fontsize if args.compact_tick_fontsize is not None else cfg.get("compact_tick_fontsize", 7),
        "compact_outer_label_fontsize": args.compact_outer_label_fontsize if args.compact_outer_label_fontsize is not None else cfg.get("compact_outer_label_fontsize", 10),
        "compact_cbar_tick_fontsize": args.compact_cbar_tick_fontsize if args.compact_cbar_tick_fontsize is not None else cfg.get("compact_cbar_tick_fontsize", 8),
        "compact_show_annotations": args.compact_annotate or cfg.get("compact_show_annotations", False),
        "compact_annotation_fontsize": args.compact_annotation_fontsize if args.compact_annotation_fontsize is not None else cfg.get("compact_annotation_fontsize", 6),
        "compact_annotation_fmt": args.compact_annotation_fmt or cfg.get("compact_annotation_fmt", "{:.2f}"),
        "compact_x_tick_rotation": args.compact_x_tick_rotation if args.compact_x_tick_rotation is not None else cfg.get("compact_x_tick_rotation", 90),
        "vmin": args.vmin if args.vmin is not None else cfg.get("vmin"),
        "vmax": args.vmax if args.vmax is not None else cfg.get("vmax"),
        "compact_vmin": cfg.get("compact_vmin", args.vmin if args.vmin is not None else None),
        "compact_vmax": cfg.get("compact_vmax", args.vmax if args.vmax is not None else None),
        "verbose": args.verbose or cfg.get("verbose", False),
        "xlabel": args.xlabel or cfg.get("xlabel", "decode length"),
        "ylabel": args.ylabel or cfg.get("ylabel", "prefill length"),
        "right_note": args.right_note or cfg.get("right_note"),
    }
    return merged


def validate_settings(s: dict):
    if not s["panels"]:
        raise SystemExit("至少要提供一个 panel。可用 --panel '标签=路径'，或在 config.json 里写 panels。")
    if not s["batches"]:
        raise SystemExit("请提供 batches。")
    if not s["prefill_lengths"]:
        raise SystemExit("请提供 prefill_lengths。")
    if not s["decode_lengths"]:
        raise SystemExit("请提供 decode_lengths。")
    if not s["reference_panel"]:
        s["reference_panel"] = list(s["panels"].keys())[0]
    if s["reference_panel"] not in s["panels"]:
        raise SystemExit(f"reference_panel={s['reference_panel']} 不在 panels 中。")


def resolve_output_paths(output: str) -> Tuple[Path, Path]:
    annotated_path = Path(output).expanduser().resolve()
    if annotated_path.suffix:
        compact_name = f"{annotated_path.stem}_compact_vs_min_hardware{annotated_path.suffix}"
    else:
        compact_name = f"{annotated_path.name}_compact_vs_min_hardware.png"
    compact_path = annotated_path.with_name(compact_name)
    return annotated_path, compact_path


def pretty_latency_field_name(latency_field: str) -> str:
    mapping = {
        "total_time_s": "latency",
        "prefill_time_s": "prefill latency",
        "decode_time_s": "decode latency",
    }
    return mapping.get(latency_field, latency_field)


def get_default_annotated_note(settings: dict) -> str:
    return (
        f"Top: speedup vs. {settings['baseline_algorithm']}\n"
        f"Middle: {pretty_latency_field_name(settings['latency_field'])}\n"
        f"Bottom: speedup vs. reference panel ({settings['reference_panel']})"
    )


def get_default_compact_note(min_hardware_panel: str) -> str:
    return (
        f"Color only: speedup vs. min-hardware panel\n"
        f"Reference: {min_hardware_panel}"
    )


def render_annotated_figure(
    settings: dict,
    panel_labels: List[str],
    batches: List[int],
    prefill_lengths: List[int],
    decode_lengths: List[int],
    heatmaps: Dict[Tuple[int, str], np.ndarray],
    latencies: Dict[Tuple[int, str], np.ndarray],
    reference_speedups: Dict[Tuple[int, str], np.ndarray],
    cmap,
    norm,
    output_path: Path,
):
    batch_cols = max(1, int(settings["batch_cols"]))
    batch_rows = math.ceil(len(batches) / batch_cols)

    panel_w = 3.4 * float(settings["figsize_scale"])
    panel_h = 2.9 * float(settings["figsize_scale"])
    fig_w = batch_cols * len(panel_labels) * panel_w + 2.4
    fig_h = batch_rows * panel_h + 1.7
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        batch_rows,
        batch_cols,
        left=0.08,
        right=0.90,
        top=0.80,
        bottom=0.11,
        wspace=0.12,
        hspace=0.42,
    )

    batch_axes_groups = []
    for idx, batch in enumerate(batches):
        r = idx // batch_cols
        c = idx % batch_cols
        inner = outer[r, c].subgridspec(1, len(panel_labels), wspace=0.12)
        group_axes = []
        for j, panel_label in enumerate(panel_labels):
            ax = fig.add_subplot(inner[0, j])
            draw_annotated_heatmap(
                ax=ax,
                data=heatmaps[(batch, panel_label)],
                latency=latencies[(batch, panel_label)],
                speedup_vs_reference=reference_speedups[(batch, panel_label)],
                decode_lengths=decode_lengths,
                prefill_lengths=prefill_lengths,
                title=panel_label,
                cmap=cmap,
                norm=norm,
            )

            if j == 0:
                ax.set_ylabel(settings["ylabel"], fontsize=11)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel(settings["xlabel"], fontsize=11)
            group_axes.append(ax)
        batch_axes_groups.append((batch, group_axes))

    title_font = 15 if len(panel_labels) * len(batches) <= 2 else 22
    fig.text(0.04, 0.965, settings["title"], fontsize=title_font, fontweight="bold", ha="left", va="top")
    if settings["subtitle"]:
        fig.text(0.04, 0.925, settings["subtitle"], fontsize=9, ha="left", va="top")

    annotated_note = settings["right_note"] or get_default_annotated_note(settings)
    fig.text(0.70, 0.965, annotated_note, fontsize=9.5, ha="left", va="top")

    plt.draw()
    for batch, group_axes in batch_axes_groups:
        add_group_title(fig, group_axes, f"Batch_size = {batch}", dy=0.025, fontsize=11)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.30, 0.018, 0.36])
    cbar = fig.colorbar(sm, cax=cax, format=mpl.ticker.FormatStrFormatter("%.2f"))
    annotated_cbar_label = settings["annotated_cbar_label"] or f"speedup vs. {settings['baseline_algorithm']}"
    cbar.set_label(annotated_cbar_label, rotation=270, labelpad=16, fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_compact_figure(
    settings: dict,
    panel_labels: List[str],
    batches: List[int],
    prefill_lengths: List[int],
    decode_lengths: List[int],
    compact_heatmaps: Dict[Tuple[int, str], np.ndarray],
    compact_reference_panel: str,
    cmap,
    norm,
    output_path: Path,
):
    nrows = len(batches)
    ncols = len(panel_labels)

    cell_w = float(settings["compact_cell_width"])
    cell_h = float(settings["compact_cell_height"])
    tick_fontsize = float(settings["compact_tick_fontsize"])
    outer_label_fontsize = float(settings["compact_outer_label_fontsize"])
    cbar_tick_fontsize = float(settings["compact_cbar_tick_fontsize"])
    annotate = bool(settings["compact_show_annotations"])
    annotation_fontsize = float(settings["compact_annotation_fontsize"])
    annotation_fmt = str(settings["compact_annotation_fmt"])
    x_tick_rotation = float(settings["compact_x_tick_rotation"])

    panel_w = max(0.9, len(decode_lengths) * cell_w)
    panel_h = max(0.9, len(prefill_lengths) * cell_h)

    gap_w = 0.12
    gap_h = 0.12
    top_pad = 0.42
    right_pad = 0.82
    left_pad = 0.82
    xlabels_pad = 0.62 if abs(x_tick_rotation) >= 45 else 0.38
    cbar_gap = 0.18
    cbar_h = 0.12
    bottom_pad = 0.18

    fig_w = left_pad + ncols * panel_w + max(0, ncols - 1) * gap_w + right_pad
    fig_h = top_pad + nrows * panel_h + max(0, nrows - 1) * gap_h + xlabels_pad + cbar_gap + cbar_h + bottom_pad
    fig = plt.figure(figsize=(fig_w, fig_h))

    grid_left = left_pad / fig_w
    grid_right = 1 - right_pad / fig_w
    grid_top = 1 - top_pad / fig_h
    grid_bottom = (bottom_pad + cbar_h + cbar_gap + xlabels_pad) / fig_h
    grid_wspace = gap_w / panel_w if panel_w > 0 else 0.08
    grid_hspace = gap_h / panel_h if panel_h > 0 else 0.08

    outer = fig.add_gridspec(
        nrows,
        ncols,
        left=grid_left,
        right=grid_right,
        top=grid_top,
        bottom=grid_bottom,
        wspace=grid_wspace,
        hspace=grid_hspace,
    )

    axes_grid: List[List[plt.Axes]] = []
    for i, batch in enumerate(batches):
        row_axes = []
        for j, panel_label in enumerate(panel_labels):
            ax = fig.add_subplot(outer[i, j])
            draw_compact_heatmap(
                ax=ax,
                data=compact_heatmaps[(batch, panel_label)],
                decode_lengths=decode_lengths,
                prefill_lengths=prefill_lengths,
                cmap=cmap,
                norm=norm,
                show_xlabels=(i == nrows - 1),
                show_ylabels_right=(j == ncols - 1),
                tick_fontsize=tick_fontsize,
                x_tick_rotation=x_tick_rotation,
                annotate=annotate,
                annotation_fmt=annotation_fmt,
                annotation_fontsize=annotation_fontsize,
            )
            row_axes.append(ax)
        axes_grid.append(row_axes)

    plt.draw()

    for j, panel_label in enumerate(panel_labels):
        ax = axes_grid[0][j]
        pos = ax.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.006,
            panel_label,
            ha="center",
            va="bottom",
            fontsize=outer_label_fontsize,
        )

    for i, batch in enumerate(batches):
        ax = axes_grid[i][0]
        pos = ax.get_position()
        fig.text(
            max(0.01, pos.x0 - 0.02),
            (pos.y0 + pos.y1) / 2,
            f"Batch={batch}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=outer_label_fontsize,
        )

    all_axes = [ax for row in axes_grid for ax in row]
    grid_x0 = min(ax.get_position().x0 for ax in all_axes)
    grid_x1 = max(ax.get_position().x1 for ax in all_axes)
    cbar_y = bottom_pad / fig_h
    cbar_h_rel = cbar_h / fig_h
    cbar_x_pad = 0.04 * max(0.01, grid_x1 - grid_x0)
    cbar_x0 = grid_x0 + cbar_x_pad
    cbar_w = max(0.12, (grid_x1 - grid_x0) - 2 * cbar_x_pad)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([cbar_x0, cbar_y, cbar_w, cbar_h_rel])
    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        format=mpl.ticker.FormatStrFormatter("%.2f"),
    )
    if settings["compact_cbar_label"]:
        cbar.set_label(settings["compact_cbar_label"], fontsize=tick_fontsize)
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize, length=0, pad=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "根据 batch / prefill / decode / 模型 / 算法 自动搜索 baseline_compare_*.json，"
            "并同时生成带标注热力图与紧凑无标注热力图。"
        )
    )
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件")
    parser.add_argument("--panel", type=parse_panel, action="append", help='可重复，格式：--panel "1 NPU 2 PIM=output/..."')
    parser.add_argument("--batches", nargs="+", type=int)
    parser.add_argument("--prefills", nargs="+", type=int)
    parser.add_argument("--decodes", nargs="+", type=int)
    parser.add_argument("--model", nargs="+", default=None, help='按模型筛选结果，例如 --model llama 70b 或 --model llama_70b')
    parser.add_argument("--algorithm", type=str, help="显式指定算法；若不指定，则自动在 heft / hefthint 中选更快者")
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--reference-panel", type=str, default=None)
    parser.add_argument("--latency-field", type=str, default=None, choices=["total_time_s", "prefill_time_s", "decode_time_s"])
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--subtitle", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="主图输出路径；同时会额外生成一张 compact_vs_min_hardware 图")
    parser.add_argument("--batch-cols", type=int, default=None)
    parser.add_argument("--figsize-scale", type=float, default=None)
    parser.add_argument("--colors", nargs="+", default=None, help="自定义颜色，示例：#bdade4 #e4adb5 ...")
    parser.add_argument("--cbar-label", type=str, default=None, help="带数字主图的 colorbar 标签")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--compact-cell-width", type=float, default=None, help="compact 图中单个 cell 的宽度（英寸）")
    parser.add_argument("--compact-cell-height", type=float, default=None, help="compact 图中单个 cell 的高度（英寸）")
    parser.add_argument("--compact-tick-fontsize", type=float, default=None, help="compact 图内部 tick label 字号")
    parser.add_argument("--compact-outer-label-fontsize", type=float, default=None, help="compact 图外侧 panel / batch label 字号")
    parser.add_argument("--compact-cbar-tick-fontsize", type=float, default=None, help="compact 图 colorbar 刻度字号")
    parser.add_argument("--compact-annotate", action="store_true", help="在 compact 图的 cell 内显示数值标注")
    parser.add_argument("--compact-annotation-fontsize", type=float, default=None, help="compact 图 cell 内标注字号")
    parser.add_argument("--compact-annotation-fmt", type=str, default=None, help='compact 图标注格式，例如 "{:.2f}"')
    parser.add_argument("--compact-x-tick-rotation", type=float, default=None, help="compact 图底部 x 轴 label 旋转角度")
    parser.add_argument("--xlabel", type=str, default=None)
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--right-note", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config_file(args.config)
    settings = merge_settings(args, cfg)
    validate_settings(settings)

    panels: Dict[str, Path] = {label: Path(path) for label, path in settings["panels"].items()}
    panel_labels = list(panels.keys())
    batches = list(map(int, settings["batches"]))
    prefill_lengths = list(map(int, settings["prefill_lengths"]))
    decode_lengths = list(map(int, settings["decode_lengths"]))

    speedup_vs_baseline_maps: Dict[Tuple[int, str], np.ndarray] = {}
    latencies: Dict[Tuple[int, str], np.ndarray] = {}

    for batch in batches:
        for panel_label, panel_path in panels.items():
            heat, lat, _ = collect_panel_data(
                panel_root=panel_path,
                batch=batch,
                prefill_lengths=prefill_lengths,
                decode_lengths=decode_lengths,
                algorithm=settings["algorithm"],
                baseline_algorithm=settings["baseline_algorithm"],
                latency_field=settings["latency_field"],
                model=settings["model"],
                verbose=settings["verbose"],
            )
            speedup_vs_baseline_maps[(batch, panel_label)] = heat
            latencies[(batch, panel_label)] = lat

    reference_speedup_maps = build_reference_speedup_maps(
        latencies=latencies,
        batches=batches,
        panel_labels=panel_labels,
        reference_panel=settings["reference_panel"],
    )

    min_hardware_panel = infer_min_hardware_panel(panels, verbose=settings["verbose"])
    min_hardware_speedup_maps = build_reference_speedup_maps(
        latencies=latencies,
        batches=batches,
        panel_labels=panel_labels,
        reference_panel=min_hardware_panel,
    )

    annotated_output_path, compact_output_path = resolve_output_paths(settings["output"])

    cmap = build_colormap(settings["colors"])

    annotated_arrays = [speedup_vs_baseline_maps[(batch, label)] for batch in batches for label in panel_labels]
    annotated_vmin, annotated_vmax = infer_vmin_vmax(annotated_arrays, settings["vmin"], settings["vmax"])
    annotated_norm = mpl.colors.Normalize(vmin=annotated_vmin, vmax=annotated_vmax)

    compact_arrays = [min_hardware_speedup_maps[(batch, label)] for batch in batches for label in panel_labels]
    compact_vmin, compact_vmax = infer_vmin_vmax(compact_arrays, settings["compact_vmin"], settings["compact_vmax"])
    compact_norm = mpl.colors.Normalize(vmin=compact_vmin, vmax=compact_vmax)

    render_annotated_figure(
        settings=settings,
        panel_labels=panel_labels,
        batches=batches,
        prefill_lengths=prefill_lengths,
        decode_lengths=decode_lengths,
        heatmaps=speedup_vs_baseline_maps,
        latencies=latencies,
        reference_speedups=reference_speedup_maps,
        cmap=cmap,
        norm=annotated_norm,
        output_path=annotated_output_path,
    )

    render_compact_figure(
        settings=settings,
        panel_labels=panel_labels,
        batches=batches,
        prefill_lengths=prefill_lengths,
        decode_lengths=decode_lengths,
        compact_heatmaps=min_hardware_speedup_maps,
        compact_reference_panel=min_hardware_panel,
        cmap=cmap,
        norm=compact_norm,
        output_path=compact_output_path,
    )

    print(f"[OK] saved annotated: {annotated_output_path}")
    print(f"[OK] saved compact:   {compact_output_path}")
    print(f"[INFO] compact figure reference panel (min hardware): {min_hardware_panel}")


if __name__ == "__main__":
    main()
