#!/usr/bin/env python3

"""
python ./experiment/experiment_fig/plot_exp5_ablation.py \
  --input ./output/bifocal_component_ablation_llama7b_qwen1p8b_b8/component_ablation_results.csv \
  --outdir ./figs/supp_exp/exp5_ablation \
  --select qwen:1.8b:8:128:512 \
  --select qwen:1.8b:8:1024:512 \
  --select llama:7b:8:1024:1024
"""
from pathlib import Path
import argparse
import zipfile

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Plot configuration
# -----------------------------
VARIANT_ORDER = [
    "EFT-only",
    "Full-w/o-Lookahead",
    "Full-w/o-Phase",
    "Full-w/o-Token",
    "Full",
]

VARIANT_SHORT = [
    "EFT",
    "w/o LA",
    "w/o Phase",
    "w/o Token",
    "Full",
]

# User-specified color palette
COLORS = [
    "#38a836",
    "#3660a8",
    "#a83646",
    "#5736a8",
    "#3690a8",
]

# Default three selected workloads
DEFAULT_SELECTIONS = [
    {
        "model_family": "qwen",
        "model_variant": "1.8b",
        "batch": 8,
        "prefill_len": 128,
        "decode_len": 512,
        "label": "[Qwen,128,512]",
    },
    {
        "model_family": "qwen",
        "model_variant": "1.8b",
        "batch": 8,
        "prefill_len": 1024,
        "decode_len": 512,
        "label": "[Qwen,1024,512]",
    },
    {
        "model_family": "llama",
        "model_variant": "7b",
        "batch": 8,
        "prefill_len": 1024,
        "decode_len": 1024,
        "label": "[Llama,1024,1024]",
    },
]


def load_component_results(input_path: Path) -> pd.DataFrame:
    """
    Load component_ablation_results.csv from:
      1. a .zip file,
      2. an extracted ablation result directory,
      3. a direct CSV path.
    """
    if input_path.is_file() and input_path.suffix == ".zip":
        with zipfile.ZipFile(input_path, "r") as zf:
            candidates = [
                name for name in zf.namelist()
                if name.endswith("component_ablation_results.csv")
            ]
            if not candidates:
                raise FileNotFoundError(
                    "Cannot find component_ablation_results.csv inside zip."
                )

            with zf.open(candidates[0]) as f:
                return pd.read_csv(f)

    if input_path.is_file() and input_path.name == "component_ablation_results.csv":
        return pd.read_csv(input_path)

    if input_path.is_dir():
        csv_path = input_path / "component_ablation_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find {csv_path}")
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Unsupported input path: {input_path}")


def make_default_label(model_family: str, prefill_len: int, decode_len: int) -> str:
    """
    Build default-style workload label.

    Examples:
      qwen, 128, 512     -> [Qwen,128,512]
      llama, 1024, 1024  -> [Llama,1024,1024]
    """
    family_map = {
        "qwen": "Qwen",
        "llama": "Llama",
    }

    model_name = family_map.get(model_family.lower(), model_family)

    return f"[{model_name},{prefill_len},{decode_len}]"


def parse_selection(selection: str) -> dict:
    """
    Parse customized workload selection.

    Supported formats:
      model_family:model_variant:batch:prefill:decode
      model_family:model_variant:batch:prefill:decode:label

    Important:
      The label passed from command line is ignored.
      We always generate default-style labels such as:
        [Qwen,128,512]
        [Llama,1024,1024]
    """
    parts = selection.split(":", 5)

    if len(parts) not in {5, 6}:
        raise ValueError(
            "Invalid --select format. Expected either:\n"
            "  model_family:model_variant:batch:prefill:decode\n"
            "or:\n"
            "  model_family:model_variant:batch:prefill:decode:label"
        )

    model_family, model_variant, batch, prefill, decode = parts[:5]

    batch = int(batch)
    prefill = int(prefill)
    decode = int(decode)

    label = make_default_label(model_family, prefill, decode)

    return {
        "model_family": model_family,
        "model_variant": model_variant,
        "batch": batch,
        "prefill_len": prefill,
        "decode_len": decode,
        "label": label,
    }


def build_plot_df(df: pd.DataFrame, selections: list[dict]) -> pd.DataFrame:
    """
    Build a compact dataframe for plotting.

    speedup_vs_pd = pd_total_time_s / total_time_s
    """
    rows = []

    required_cols = {
        "model_family",
        "model_variant",
        "batch",
        "prefill_len",
        "decode_len",
        "variant",
        "pd_total_time_s",
        "total_time_s",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for spec in selections:
        sub = df[
            (df["model_family"] == spec["model_family"]) &
            (df["model_variant"] == spec["model_variant"]) &
            (df["batch"] == spec["batch"]) &
            (df["prefill_len"] == spec["prefill_len"]) &
            (df["decode_len"] == spec["decode_len"])
        ].copy()

        if sub.empty:
            raise ValueError(f"No rows found for workload: {spec}")

        sub = sub.set_index("variant").reindex(VARIANT_ORDER).reset_index()

        if sub["total_time_s"].isna().any():
            found_variants = sorted(
                df[
                    (df["model_family"] == spec["model_family"]) &
                    (df["model_variant"] == spec["model_variant"]) &
                    (df["batch"] == spec["batch"]) &
                    (df["prefill_len"] == spec["prefill_len"]) &
                    (df["decode_len"] == spec["decode_len"])
                ]["variant"].unique()
            )
            raise ValueError(
                f"Missing one or more ablation variants for workload {spec}.\n"
                f"Expected: {VARIANT_ORDER}\n"
                f"Found: {found_variants}"
            )

        sub["workload_label"] = spec["label"]
        sub["speedup_vs_pd"] = sub["pd_total_time_s"] / sub["total_time_s"]

        rows.append(sub)

    return pd.concat(rows, ignore_index=True)


def plot_fig_a(plot_df: pd.DataFrame, selections: list[dict], outdir: Path) -> None:
    """
    Plot Fig. (a): Overall speedup over PD.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "legend.fontsize": 12,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    fig, ax = plt.subplots(figsize=(4.8, 3))

    for i, spec in enumerate(selections):
        sub = (
            plot_df[plot_df["workload_label"] == spec["label"]]
            .set_index("variant")
            .reindex(VARIANT_ORDER)
            .reset_index()
        )

        x = range(len(VARIANT_ORDER))
        y = sub["speedup_vs_pd"].tolist()

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=COLORS[i % len(COLORS)],
            label=spec["label"],
        )

    ax.set_xticks(range(len(VARIANT_SHORT)))
    ax.set_xticklabels(VARIANT_SHORT, rotation=15)

    ax.set_ylabel("Speedup over PD")
    # ax.set_xlabel("Ablation variant")

    ax.grid(True, axis="y", alpha=0.25)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()

    png_path = outdir / "bifocal_ablation.png"
    pdf_path = outdir / "bifocal_ablation.pdf"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to ablation_v2.zip, extracted ablation directory, "
            "or component_ablation_results.csv"
        ),
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for generated figure.",
    )

    parser.add_argument(
        "--select",
        action="append",
        default=[],
        help=(
            "Optional workload selection. Supported formats:\n"
            "  model_family:model_variant:batch:prefill:decode\n"
            "or:\n"
            "  model_family:model_variant:batch:prefill:decode:label\n"
            "The input label will be ignored and regenerated automatically."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    df = load_component_results(input_path)

    if args.select:
        selections = [parse_selection(s) for s in args.select]
    else:
        selections = DEFAULT_SELECTIONS

    plot_df = build_plot_df(df, selections)

    outdir.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(outdir / "fig_a_plot_data.csv", index=False)

    plot_fig_a(plot_df, selections, outdir)


if __name__ == "__main__":
    main()