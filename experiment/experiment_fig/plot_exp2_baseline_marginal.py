#!/usr/bin/env python3
"""
python plot_exp2_baseline_marginal.py \
  --model-folder llama_7b_fp16_b16_s8 \
  --prefills 128 512 1024 2048 \
  --decodes 128 256 512 1024 \
  --reference-panel 0 \
  --panel-root 0=../../algorithms/output/exp2/npu_only/npu/hw_hardware_1npu/sst8_rst8/ \
  --panel-root 2=../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst8_rst8/ \
  --panel-root 4=../../algorithms/output/exp2/4shards/hw_hardware_1npu_4aim/sst8_rst8/ \
  --panel-root 8=../../algorithms/output/exp2/8shards/hw_hardware_1npu_8aim/sst8_rst8/ \
  --output ../../figs/exp2/llama_7b_fp16_b16_s8_marginal.pdf

python plot_exp2_baseline_marginal.py \
  --model-folder llama_13b_fp16_b16_s8 \
  --prefills 128 512 1024 2048 \
  --decodes 128 256 512 1024 \
  --reference-panel 0 \
  --panel-root 0=../../algorithms/output/exp2/npu_only/npu/hw_hardware_1npu/sst8_rst8/ \
  --panel-root 2=../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst8_rst8/ \
  --panel-root 4=../../algorithms/output/exp2/4shards/hw_hardware_1npu_4aim/sst8_rst8/ \
  --panel-root 8=../../algorithms/output/exp2/8shards/hw_hardware_1npu_8aim/sst8_rst8/ \
  --output ../../figs/exp2/llama_13b_fp16_b16_s8_marginal.pdf

python plot_exp2_baseline_marginal.py \
  --model-folder llama_70b_fp16_b16_s8 \
  --prefills 128 512 1024 2048 \
  --decodes 128 256 512 1024 \
  --reference-panel 0 \
  --panel-root 0=../../algorithms/output/exp2/npu_only/npu/hw_hardware_1npu/sst8_rst8/ \
  --panel-root 2=../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst8_rst8/ \
  --panel-root 4=../../algorithms/output/exp2/4shards/hw_hardware_1npu_4aim/sst8_rst8/ \
  --panel-root 8=../../algorithms/output/exp2/8shards/hw_hardware_1npu_8aim/sst8_rst8/ \
  --output ../../figs/exp2/llama_70b_fp16_b16_s8_marginal.pdf

others:
python plot_exp2_baseline_marginal.py \
  --model-folder llama_7b_fp16_b1_s8 \
  --prefills 128 512 1024 2048 \
  --decodes 128 256 512 1024 \
  --reference-panel 0 \
  --panel-root 0=../../algorithms/output/exp2/npu_only/npu/hw_hardware_1npu/sst8_rst8/ \
  --panel-root 2=../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst8_rst8/ \
  --panel-root 4=../../algorithms/output/exp2/4shards/hw_hardware_1npu_4aim/sst8_rst8/ \
  --panel-root 8=../../algorithms/output/exp2/8shards/hw_hardware_1npu_8aim/sst8_rst8/ \
  --panel-policy 2=heft \
  --panel-policy 4=myalgo \
  --panel-policy 8=otheralgo \
  --output ../../figs/exp2/llama_7b_fp16_b1_s8_marginal_mixed_policy.pdf

"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

ARIAL_FONT_FAMILY = "Arial"
MIN_FONT_PT = 7.0


def apply_global_plot_style() -> None:
    plt.rcParams.update({
        "font.family": [ARIAL_FONT_FAMILY],
        "font.sans-serif": [ARIAL_FONT_FAMILY],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def enforce_figure_fonts(
    fig: plt.Figure,
    *,
    min_font_pt: float = MIN_FONT_PT,
    font_family: str = ARIAL_FONT_FAMILY,
) -> None:
    for text in fig.findobj(Text):
        try:
            current_size = float(text.get_fontsize())
        except (TypeError, ValueError):
            current_size = min_font_pt
        text.set_fontfamily(font_family)
        text.set_fontsize(max(min_font_pt, current_size))


apply_global_plot_style()


DEFAULT_PALETTE = [
    "#5837a8",
    "#a83747",
    "#3791a8",
    "#3760a9",
    "#39a937",
    "#a89a37",
]

DEFAULT_NONZERO_POLICIES = ("algo:heft", "algo:hefthint")
ZERO_PANEL_POLICY = "algo:hefthint"
FILENAME_RE = re.compile(r"baseline_compare_(\d+)x(\d+)\.json$")


@dataclass(frozen=True)
class Record:
    path: Path
    panel_count: int
    prefill_len: int
    decode_len: int
    batch: Optional[int]
    model_key: Optional[str]
    selected_policy: str
    best_total_time_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot speedup-vs-panel curves from baseline_compare_*.json. "
            "Panel 0 is always normalized with algo:hefthint, while non-zero panels "
            "can use either the default policy pool or per-panel policies via "
            "--panel-policy PANEL=POLICY. Supports either one recursive --root, or "
            "repeated --panel-root PANEL=PATH when each panel setting lives under "
            "a different directory."
        )
    )

    root_group = parser.add_mutually_exclusive_group(required=True)
    root_group.add_argument(
        "--root",
        help="Single top-level directory to search recursively.",
    )
    root_group.add_argument(
        "--panel-root",
        action="append",
        default=None,
        metavar="PANEL=PATH",
        help=(
            "Repeatable explicit panel-path mapping, e.g. "
            "--panel-root 0=/data/1NPU0PIM --panel-root 2=/data/1NPU2PIM. "
            "Useful when different panel settings are stored under different roots."
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model key to match from JSON config, e.g. llama_7b_fp16. Ignored when --model-folder is set.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size to filter from JSON config, e.g. 1.",
    )
    parser.add_argument(
        "--model-folder",
        default=None,
        help=(
            "Exact model folder name to match in the path, e.g. llama_7b_fp16_b1_s8. "
            "When set, it is used as the primary path filter."
        ),
    )
    parser.add_argument(
        "--folder-contains",
        default=None,
        help="Optional extra substring that must appear somewhere in the file path.",
    )
    parser.add_argument(
        "--prefills",
        type=int,
        nargs="+",
        required=True,
        help="One or more prefill lengths to plot as subplots, e.g. 128 256 512.",
    )
    parser.add_argument(
        "--decodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional decode lengths to include. Default: all discovered decode lengths.",
    )
    parser.add_argument(
        "--reference-panel",
        type=int,
        default=None,
        help=(
            "Reference panel count used for speedup = T_ref / T_current. "
            "Default: the smallest discovered panel count."
        ),
    )
    parser.add_argument(
        "--panel-regex",
        default=r"(?P<count>\d+)aim",
        help=(
            "Regex used to extract panel count from any path component in --root mode. "
            "Must contain either a named group 'count' or a first capturing group. "
            "Default matches hw_hardware_1npu_4aim -> 4. Ignored in --panel-root mode."
        ),
    )
    parser.add_argument(
        "--panel-policy",
        action="append",
        default=None,
        metavar="PANEL=POLICY[,POLICY...]",
        help=(
            "Repeatable non-zero panel policy override, e.g. --panel-policy 2=heft "
            "--panel-policy 4=algo:beam. When multiple policies are given with commas, "
            "the best total_time_s among them is used. Panel 0 always uses algo:hefthint."
        ),
    )
    parser.add_argument(
        "--default-policies",
        nargs="+",
        default=list(DEFAULT_NONZERO_POLICIES),
        help=(
            "Default candidate policies for non-zero panels when --panel-policy is not set. "
            "Accepts names with or without the algo: prefix. "
            f"Default: {' '.join(DEFAULT_NONZERO_POLICIES)}"
        ),
    )
    parser.add_argument(
        "--colors",
        nargs="*",
        default=DEFAULT_PALETTE,
        help="Color list used by decode length curves.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path, e.g. /tmp/b1_prefill_128_256.png",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=1.0,
        help="Global figure size scale. Default: 1.0",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom figure title.",
    )
    parser.add_argument(
        "--xlabel",
        default="Panel count",
        help="X-axis label.",
    )
    parser.add_argument(
        "--ylabel",
        default=None,
        help="Y-axis label. Default is generated automatically.",
    )
    parser.add_argument(
        "--marginal-mode",
        choices=["delta", "per_panel"],
        default="delta",
        help=(
            "How to annotate marginal utility numbers: "
            "'delta' = adjacent speedup increment, "
            "'per_panel' = adjacent speedup increment divided by panel increment."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable optional smoothing even if scipy is available.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print matched files and skipped reasons.",
    )

    args = parser.parse_args()
    args.default_policies = normalize_policy_list(args.default_policies, arg_name="--default-policies")
    args.panel_policy_map = parse_panel_policy_specs(args.panel_policy)
    return args


def infer_model_key(config: dict) -> Optional[str]:
    family = config.get("model_family")
    variant = config.get("model_variant")
    dtype = config.get("dtype")
    if family and variant and dtype:
        return f"{family}_{variant}_{dtype}"
    result_dir = config.get("result_dir")
    if isinstance(result_dir, str):
        # Try to recover from .../llama_7b_fp16_b1_s8
        tail = Path(result_dir).name
        m = re.match(r"([a-zA-Z0-9]+_[a-zA-Z0-9]+_[a-zA-Z0-9]+)_b\d+(?:_|$)", tail)
        if m:
            return m.group(1)
    return None


def extract_panel_count(path: Path, panel_re: re.Pattern[str]) -> int:
    for part in reversed(path.parts):
        match = panel_re.search(part)
        if match:
            if "count" in match.groupdict():
                return int(match.group("count"))
            return int(match.group(1))
    raise ValueError(
        f"Cannot extract panel count from path: {path}\n"
        f"Please adjust --panel-regex."
    )


def normalize_policy_name(policy_text: str) -> str:
    policy = policy_text.strip()
    if not policy:
        raise ValueError("Policy name cannot be empty.")
    if not policy.startswith("algo:"):
        policy = f"algo:{policy}"
    return policy


def normalize_policy_list(policies: Sequence[str], arg_name: str) -> Tuple[str, ...]:
    normalized: List[str] = []
    seen = set()
    for policy_text in policies:
        policy = normalize_policy_name(policy_text)
        if policy in seen:
            continue
        normalized.append(policy)
        seen.add(policy)
    if not normalized:
        raise ValueError(f"{arg_name} must contain at least one policy.")
    return tuple(normalized)


def parse_panel_policy_specs(specs: Optional[Sequence[str]]) -> Dict[int, Tuple[str, ...]]:
    parsed: Dict[int, Tuple[str, ...]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --panel-policy value: {spec!r}. Expected format PANEL=POLICY "
                "or PANEL=POLICY1,POLICY2"
            )
        panel_text, policy_text = spec.split("=", 1)
        panel_text = panel_text.strip()
        policy_text = policy_text.strip()
        if not panel_text:
            raise ValueError(f"Invalid --panel-policy value: {spec!r}. Missing PANEL.")
        if not policy_text:
            raise ValueError(f"Invalid --panel-policy value: {spec!r}. Missing POLICY.")

        try:
            panel_value = int(panel_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid panel value in --panel-policy {spec!r}. PANEL must be an integer."
            ) from exc

        policies = normalize_policy_list(policy_text.split(","), arg_name="--panel-policy")

        if panel_value == 0:
            if policies != (ZERO_PANEL_POLICY,):
                print(
                    f"[warn] panel 0 always uses {ZERO_PANEL_POLICY}; ignoring override {spec!r}",
                    file=sys.stderr,
                )
            continue

        if panel_value in parsed:
            raise ValueError(
                f"Duplicate panel value {panel_value} in --panel-policy. "
                "Each panel value can only be specified once."
            )
        parsed[panel_value] = policies

    return parsed


def policies_for_panel(panel_count: int, args: argparse.Namespace) -> Tuple[str, ...]:
    if panel_count == 0:
        return (ZERO_PANEL_POLICY,)
    return args.panel_policy_map.get(panel_count, args.default_policies)


def parse_panel_root_specs(specs: Sequence[str]) -> List[Tuple[int, Path]]:
    parsed: Dict[int, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --panel-root value: {spec!r}. Expected format PANEL=PATH, "
                "for example 4=/data/1NPU4PIM"
            )
        panel_text, path_text = spec.split("=", 1)
        panel_text = panel_text.strip()
        path_text = path_text.strip()
        if not panel_text:
            raise ValueError(f"Invalid --panel-root value: {spec!r}. Missing PANEL.")
        if not path_text:
            raise ValueError(f"Invalid --panel-root value: {spec!r}. Missing PATH.")
        try:
            panel_value = int(panel_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid panel value in --panel-root {spec!r}. PANEL must be an integer."
            ) from exc

        root = Path(path_text).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"panel root does not exist: {root}")
        if panel_value in parsed:
            raise ValueError(
                f"Duplicate panel value {panel_value} in --panel-root. "
                "Each panel value can only be specified once."
            )
        parsed[panel_value] = root

    return sorted(parsed.items(), key=lambda kv: kv[0])


def select_total_time_for_panel(
    payload: dict,
    json_path: Path,
    panel_count: int,
    args: argparse.Namespace,
) -> Tuple[str, float]:
    target_policies = policies_for_panel(panel_count, args)
    candidates: List[Tuple[str, float]] = []
    available_policies = set()

    for item in payload.get("results", []):
        policy = item.get("policy")
        if isinstance(policy, str):
            available_policies.add(policy)
        if policy in target_policies:
            total = item.get("total_time_s")
            if total is not None:
                candidates.append((str(policy), float(total)))

    if not candidates:
        available_text = ", ".join(sorted(available_policies)) if available_policies else "none"
        raise ValueError(
            f"No requested policy total_time_s found in {json_path} for panel={panel_count}. "
            f"Requested policies={list(target_policies)}, available policies={available_text}"
        )

    best_policy, best_total_time_s = min(candidates, key=lambda item: item[1])
    return best_policy, best_total_time_s


def load_record(
    path: Path,
    args: argparse.Namespace,
    panel_re: Optional[re.Pattern[str]] = None,
    panel_count_override: Optional[int] = None,
) -> Record:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    file_match = FILENAME_RE.search(path.name)
    if file_match:
        prefill_len = int(file_match.group(1))
        decode_len = int(file_match.group(2))
    else:
        prefill_len = int(config.get("prefill_len"))
        decode_len = int(config.get("decode_len"))

    if panel_count_override is not None:
        panel_count = panel_count_override
    else:
        if panel_re is None:
            raise ValueError("panel_re must be provided when panel_count_override is not set")
        panel_count = extract_panel_count(path, panel_re)

    batch = config.get("batch")
    if batch is not None:
        batch = int(batch)
    model_key = infer_model_key(config)
    selected_policy, best_total_time_s = select_total_time_for_panel(
        payload=payload,
        json_path=path,
        panel_count=panel_count,
        args=args,
    )

    return Record(
        path=path,
        panel_count=panel_count,
        prefill_len=prefill_len,
        decode_len=decode_len,
        batch=batch,
        model_key=model_key,
        selected_policy=selected_policy,
        best_total_time_s=best_total_time_s,
    )


def matches_filters(record: Record, path: Path, args: argparse.Namespace) -> bool:
    if args.model_folder is not None and args.model_folder not in path.parts:
        return False

    if args.folder_contains and args.folder_contains not in str(path):
        return False

    if args.model_folder is None and args.model is not None:
        if record.model_key != args.model:
            return False

    if args.batch is not None and record.batch != args.batch:
        return False

    if record.prefill_len not in set(args.prefills):
        return False

    if args.decodes is not None and len(args.decodes) > 0 and record.decode_len not in set(args.decodes):
        return False

    return True


def discover_records(args: argparse.Namespace) -> List[Record]:
    search_roots: List[Tuple[Optional[int], Path]] = []
    panel_re: Optional[re.Pattern[str]] = None

    if args.panel_root:
        search_roots = [(panel_value, root) for panel_value, root in parse_panel_root_specs(args.panel_root)]
    else:
        root = Path(args.root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root}")
        panel_re = re.compile(args.panel_regex)
        search_roots = [(None, root)]

    matched: List[Record] = []
    prefill_set = set(args.prefills)
    decode_set = set(args.decodes) if args.decodes else None

    for panel_override, root in search_roots:
        if args.verbose:
            mode_text = f"panel={panel_override}" if panel_override is not None else "auto-panel"
            print(f"[root] {mode_text} root={root}", file=sys.stderr)

        for path in root.rglob("baseline_compare_*.json"):
            if not FILENAME_RE.search(path.name):
                continue
            if args.verbose:
                print(f"[scan] {path}", file=sys.stderr)
            try:
                record = load_record(
                    path,
                    args=args,
                    panel_re=panel_re,
                    panel_count_override=panel_override,
                )
            except Exception as exc:
                if args.verbose:
                    print(f"[skip] {path}: {exc}", file=sys.stderr)
                continue

            if record.prefill_len not in prefill_set:
                continue
            if decode_set is not None and record.decode_len not in decode_set:
                continue
            if matches_filters(record, path, args):
                matched.append(record)
                if args.verbose:
                    print(
                        f"[match] panel={record.panel_count} prefill={record.prefill_len} "
                        f"decode={record.decode_len} batch={record.batch} model={record.model_key} "
                        f"policy={record.selected_policy} path={path}",
                        file=sys.stderr,
                    )

    if not matched:
        debug_bits = []
        if args.panel_root:
            debug_bits.append(f"panel_roots={args.panel_root}")
        else:
            debug_bits.append(f"root={Path(args.root).expanduser().resolve()}")
        if args.model_folder:
            debug_bits.append(f"model_folder={args.model_folder}")
        if args.model:
            debug_bits.append(f"model={args.model}")
        if args.batch is not None:
            debug_bits.append(f"batch={args.batch}")
        debug_bits.append(f"prefills={sorted(prefill_set)}")
        if decode_set:
            debug_bits.append(f"decodes={sorted(decode_set)}")
        raise RuntimeError(
            "No matching baseline_compare_*.json files found.\n" + ", ".join(debug_bits)
        )

    # Reject ambiguous duplicates for the same panel/prefill/decode.
    seen: Dict[Tuple[int, int, int], Record] = {}
    for rec in matched:
        key = (rec.panel_count, rec.prefill_len, rec.decode_len)
        if key in seen:
            other = seen[key]
            raise RuntimeError(
                "Duplicate matches found for the same (panel, prefill, decode): "
                f"{key}\n - {other.path}\n - {rec.path}\n"
                "Please narrow the search roots or use --model-folder / --folder-contains."
            )
        seen[key] = rec

    return sorted(matched, key=lambda r: (r.prefill_len, r.decode_len, r.panel_count))


def maybe_smooth(xs: Sequence[float], ys: Sequence[float], disable: bool) -> Tuple[np.ndarray, np.ndarray]:
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if disable or len(xs_arr) < 3:
        return xs_arr, ys_arr

    try:
        from scipy.interpolate import PchipInterpolator  # type: ignore
    except Exception:
        return xs_arr, ys_arr

    if np.any(np.diff(xs_arr) <= 0):
        return xs_arr, ys_arr

    x_dense = np.linspace(xs_arr.min(), xs_arr.max(), 200)
    y_dense = PchipInterpolator(xs_arr, ys_arr)(x_dense)
    return x_dense, y_dense


def build_speedup_table(
    records: Sequence[Record],
    prefills: Sequence[int],
    reference_panel: Optional[int],
) -> Tuple[Dict[int, Dict[int, List[Tuple[int, float]]]], int, List[int]]:
    time_table: Dict[int, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    all_panels = set()

    for rec in records:
        time_table[rec.prefill_len][rec.decode_len][rec.panel_count] = rec.best_total_time_s
        all_panels.add(rec.panel_count)

    if not all_panels:
        raise RuntimeError("No panel counts were extracted from matched files.")

    if reference_panel is None:
        reference_panel = min(all_panels)

    speedup_table: Dict[int, Dict[int, List[Tuple[int, float]]]] = defaultdict(dict)
    for prefill in prefills:
        decode_map = time_table.get(prefill, {})
        for decode_len, panel_to_time in decode_map.items():
            if reference_panel not in panel_to_time:
                # Without the reference point, this curve cannot be normalized.
                continue
            ref_time = panel_to_time[reference_panel]
            curve = []
            for panel_count, current_time in sorted(panel_to_time.items()):
                if current_time <= 0:
                    continue
                speedup = ref_time / current_time
                curve.append((panel_count, speedup))
            if curve:
                speedup_table[prefill][decode_len] = curve

    return speedup_table, reference_panel, sorted(all_panels)


def choose_decode_order(
    speedup_table: Dict[int, Dict[int, List[Tuple[int, float]]]],
    prefills: Sequence[int],
    user_decodes: Optional[Sequence[int]],
) -> List[int]:
    if user_decodes:
        return list(user_decodes)

    discovered = set()
    for prefill in prefills:
        discovered.update(speedup_table.get(prefill, {}).keys())
    return sorted(discovered)


def pretty_gain(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}"


def plot_speedup_figure(
    speedup_table: Dict[int, Dict[int, List[Tuple[int, float]]]],
    prefills: Sequence[int],
    decode_order: Sequence[int],
    reference_panel: int,
    all_panels: Sequence[int],
    args: argparse.Namespace,
) -> Path:
    prefills = list(prefills)
    n = len(prefills)
    cols = min(4, n)
    rows = math.ceil(n / 4)

    fig_w = max(4.6 * cols * args.figscale, 5.0)
    fig_h = max(3.8 * rows * args.figscale, 3.8)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    color_map = {
        decode_len: args.colors[idx % len(args.colors)]
        for idx, decode_len in enumerate(decode_order)
    }

    if args.title:
        fig_title = args.title
    else:
        left = args.model_folder or args.model or "matched model"
        batch_text = f"b{args.batch}" if args.batch is not None else "batch=*"
        fig_title = f"{left} | {batch_text} | speedup vs ref={reference_panel}"

    fig.suptitle(fig_title, fontsize=13, y=0.995)

    subplot_ranges: Dict[int, Tuple[float, float]] = {}
    for prefill in prefills:
        ys = []
        for decode_len in decode_order:
            curve = speedup_table.get(prefill, {}).get(decode_len)
            if curve:
                ys.extend(y for _, y in curve)
        if ys:
            y_min = min(ys)
            y_max = max(ys)
        else:
            y_min, y_max = 0.95, 1.05
        if abs(y_max - y_min) < 1e-9:
            y_min -= 0.05
            y_max += 0.05
        subplot_ranges[prefill] = (y_min, y_max)

    handles = {}
    x_ticks = sorted(all_panels)

    for idx, prefill in enumerate(prefills):
        ax = axes[idx // cols][idx % cols]
        curves = {
            decode_len: speedup_table.get(prefill, {}).get(decode_len)
            for decode_len in decode_order
            if speedup_table.get(prefill, {}).get(decode_len)
        }

        if not curves:
            ax.text(
                0.5,
                0.5,
                f"No data for prefill={prefill}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_axis_off()
            continue

        y_min, y_max = subplot_ranges[prefill]
        y_span = y_max - y_min
        x_diffs = np.diff(x_ticks) if len(x_ticks) > 1 else np.array([1.0])
        x_step = float(np.min(x_diffs)) if len(x_diffs) else 1.0
        n_curves = len(curves)
        if n_curves == 1:
            curve_offsets = [0.0]
        else:
            curve_offsets = np.linspace(-0.16 * x_step, 0.16 * x_step, n_curves)

        ax.axhline(1.0, color="0.70", linestyle="--", linewidth=1.0, zorder=0)

        plotted_curve_idx = 0
        for decode_len in decode_order:
            curve = curves.get(decode_len)
            if not curve:
                continue

            color = color_map[decode_len]
            xs = [x for x, _ in curve]
            ys = [y for _, y in curve]

            smooth_x, smooth_y = maybe_smooth(xs, ys, args.no_smooth)
            (line,) = ax.plot(
                smooth_x,
                smooth_y,
                color=color,
                linewidth=2.0,
                alpha=0.95,
                label=f"decode={decode_len}",
                zorder=2,
            )
            ax.scatter(xs, ys, color=color, s=30, zorder=3)

            handles[decode_len] = line

            x_offset = curve_offsets[min(plotted_curve_idx, len(curve_offsets) - 1)]
            text_base_offset = (plotted_curve_idx - (n_curves - 1) / 2.0) * 0.05 * y_span

            for point_idx in range(1, len(xs)):
                x_prev, y_prev = xs[point_idx - 1], ys[point_idx - 1]
                x_curr, y_curr = xs[point_idx], ys[point_idx]
                gain = y_curr - y_prev
                panel_delta = x_curr - x_prev

                if args.marginal_mode == "per_panel":
                    label_value = gain / panel_delta if panel_delta != 0 else float("nan")
                else:
                    label_value = gain

                arrow_x = x_curr + x_offset
                ax.annotate(
                    "",
                    xy=(arrow_x, y_curr),
                    xytext=(arrow_x, y_prev),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        lw=1.1,
                        alpha=0.85,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=4,
                )

                extra_y = ((point_idx % 2) - 0.5) * 0.04 * y_span
                text_y = y_prev + 0.5 * (y_curr - y_prev) + text_base_offset + extra_y
                text_x = arrow_x + (0.07 * x_step if plotted_curve_idx % 2 == 0 else -0.07 * x_step)
                ha = "left" if plotted_curve_idx % 2 == 0 else "right"

                ax.text(
                    text_x,
                    text_y,
                    pretty_gain(label_value),
                    color=color,
                    fontsize=8.8,
                    ha=ha,
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=0.20),
                    zorder=5,
                )

            plotted_curve_idx += 1

        ax.set_title(f"prefill={prefill}", fontsize=11)
        ax.set_xticks(x_ticks)
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.45)
        ax.set_xlim(min(x_ticks) - 0.2 * x_step, max(x_ticks) + 0.35 * x_step)
        ax.set_ylim(y_min - 0.12 * y_span, y_max + 0.15 * y_span)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_axis_off()

    ylabel = args.ylabel or (
        f"Speedup vs ref={reference_panel}"
        if args.reference_panel is not None
        else f"Speedup vs min panel ({reference_panel})"
    )
    fig.supxlabel(args.xlabel)
    fig.supylabel(ylabel)

    if handles:
        ordered_handles = [handles[d] for d in decode_order if d in handles]
        ordered_labels = [f"decode={d}" for d in decode_order if d in handles]
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(6, max(1, len(ordered_handles))),
            frameon=False,
        )

    enforce_figure_fonts(fig)
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.88))

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    records = discover_records(args)
    speedup_table, reference_panel, all_panels = build_speedup_table(
        records=records,
        prefills=args.prefills,
        reference_panel=args.reference_panel,
    )

    available_prefills = [p for p in args.prefills if speedup_table.get(p)]
    missing_prefills = [p for p in args.prefills if not speedup_table.get(p)]
    if not available_prefills:
        raise RuntimeError(
            "None of the requested prefills have usable curves after normalization. "
            "This usually means the reference panel file is missing for those prefills."
        )
    if missing_prefills:
        print(
            f"[warn] These prefills had no usable curves and will be left blank: {missing_prefills}",
            file=sys.stderr,
        )

    decode_order = choose_decode_order(speedup_table, args.prefills, args.decodes)
    if not decode_order:
        raise RuntimeError("No decode lengths remained after filtering.")

    output_path = plot_speedup_figure(
        speedup_table=speedup_table,
        prefills=args.prefills,
        decode_order=decode_order,
        reference_panel=reference_panel,
        all_panels=all_panels,
        args=args,
    )
    print(output_path)


if __name__ == "__main__":
    main()
