# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

json_path = Path("../algorithms/output/baseline_compare.json")
with open(json_path, "r") as f:
    data = json.load(f)

results = data["results"]
policies      = [r["policy"] for r in results]

group1 = ["weights_on_pim", "attn_on_pim", "pd"]
group2 = ["neupims", "ianus", "facil", "attacc"]
algos = [p for p in policies if p.startswith("algo:")]
ordered_policies = [p for p in group1 if p in policies] + [p for p in group2 if p in policies] + algos
leftovers = [p for p in policies if p not in ordered_policies]
ordered_policies += leftovers

prefill_times_map = {r["policy"]: float(r["prefill_time_s"]) for r in results}
decode_times_map  = {r["policy"]: float(r["decode_time_s"])  for r in results}
e2e_times_map     = {p: prefill_times_map[p] + decode_times_map[p] for p in policies}

prefill_times = [prefill_times_map[p] for p in ordered_policies]
decode_times  = [decode_times_map[p]  for p in ordered_policies]
e2e_times     = [e2e_times_map[p]     for p in ordered_policies]

prefill_max = max(prefill_times)
decode_max  = max(decode_times)
e2e_max = max(e2e_times)

def speedup(max_time, t):
    if t == 0:
        return np.inf
    return max_time / t

prefill_speed = [speedup(prefill_max, t) for t in prefill_times]
decode_speed  = [speedup(decode_max,  t) for t in decode_times]
e2e_speed  = [speedup(e2e_max,  t) for t in e2e_times]

COL_PREFILL = "#1d2e53"
COL_DECODE  = "#395aad"
COL_E2E = "#84b4fc"

fig, ax = plt.subplots(figsize=(12, 5.6)) 
x = np.arange(len(policies))
group_width = 0.85
bar_w = group_width / 3.0
offsets = np.array([-bar_w, 0.0, +bar_w]) 
bars_p = plt.bar(x + offsets[0], prefill_speed, width=bar_w, label="Prefill",
                 color=COL_PREFILL, edgecolor="black", linewidth=0.8, zorder=3)

decode_heights = [np.nan if np.isinf(v) else v for v in decode_speed]
bars_d = plt.bar(x + offsets[1], decode_heights, width=bar_w, label="Decode",
                 color=COL_DECODE, edgecolor="black", linewidth=0.8, zorder=3)

e2e_heights = [np.nan if np.isinf(v) else v for v in e2e_speed]
bars_e = plt.bar(x + offsets[2], e2e_heights, width=bar_w, label="End-to-End",
                 color=COL_E2E, edgecolor="black", linewidth=0.8, zorder=3)

ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.1, alpha=0.9, zorder=2)
ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.9, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(ordered_policies, rotation=30, ha="right")
ax.set_title("Prefill & Decode Speedup", pad=12)

finite_vals = [v for v in prefill_speed + decode_speed + e2e_speed if np.isfinite(v)]
ymax = max(finite_vals) if finite_vals else 1.0
ax.set_ylim(0, ymax * 1.35)
plt.margins(x=0.02)

def annotate(ax, bars, values):
    top = ymax
    for b, v in zip(bars, values):
        if np.isnan(v) or np.isinf(v):
            continue
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02 * top,
                f"{v:.2f}×", ha="center", va="bottom", fontsize=9, rotation=90)

legend = ax.legend(
    ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.99),
    frameon=True, framealpha=0.9
)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_edgecolor("#dddddd")
annotate(ax, bars_p, prefill_speed)
annotate(ax, bars_d, decode_speed)
annotate(ax, bars_e, e2e_speed)

plt.tight_layout()

out_path = Path("./prefill_decode_speedup.pdf")
plt.savefig(out_path, dpi=220, bbox_inches="tight")

print(f"Saved to: {out_path}")
