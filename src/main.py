"""CLI entrypoint and backward-compatible re-exports.

The original main.py had grown into a multi-thousand-line module. The CLI and
its helpers now live under mainlib/ so the public entrypoint stays compact.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _purge_local_bytecode() -> None:
    """Delete stale local bytecode before importing the rest of the project."""
    skip = str(os.environ.get("TRIFORM_SKIP_PYCACHE_PURGE", "") or "").strip().lower()
    if skip in {"1", "true", "yes", "on"}:
        return

    sys.dont_write_bytecode = True
    src_dir = Path(__file__).resolve().parent
    roots = [src_dir, src_dir.parent / "commands"]
    for base in roots:
        if not base.exists():
            continue
        for pycache_dir in sorted(base.rglob("__pycache__"), reverse=True):
            shutil.rmtree(pycache_dir, ignore_errors=True)
        for pyc_path in base.rglob("*.pyc"):
            try:
                pyc_path.unlink()
            except OSError:
                pass


_purge_local_bytecode()

from mainlib.app import main
from mainlib.baselines import (
    PD_BASELINES,
    _BASELINE_REGISTRY,
    _apply_policy_on_graph,
    _arith_intensity,
    _baseline_partitioned_attn,
    _baseline_partitioned_linear,
    _baseline_partitioned_ffn,
    _baseline_neu_pims,
    _is_attention_node,
    _is_gemv_like,
    _is_kv_rw,
    _is_op,
    register_baseline,
)
from mainlib.cli import (
    _apply_runtime_config_overrides,
    _load_cfg_from_json,
    _normalize_list_field,
    parse_args,
)
from mainlib.burstgpt_serving_eval import evaluate_burstgpt_suite, load_burstgpt_csv
from mainlib.evaluate import (
    _ensure_dir,
    _eval_one_baseline,
    _label_summary,
    _run_strategy_once,
    _save_best_json,
    evaluate_suite,
)
from mainlib.graph_utils import (
    _clone_graph,
    _cluster_type_count,
    _fallback_npu_to_cpu_if_needed,
    _fallback_pim_to_cpu_if_needed,
)
from mainlib.kv_policy import (
    _apply_kv_place_constraints,
    _build_cost_model_for_run,
    _compute_kv_plan_info,
    _effective_tp_qkv,
    _estimate_total_time_for_label,
    _fmt_kv_policy_scores,
    _infer_kv_dtype_bytes_from_graph,
    _infer_kv_place_from_label,
    _make_label_from_kv_plan,
    _make_label_given_kv_place,
    _normalize_kv_place,
    _normalize_npu_backend,
    auto_select_kv_policy,
)
from mainlib.log_utils import (
    _debug,
    _emit_weight_suggest_al_log,
    _emit_weight_suggest_progress,
    _is_key_weight_suggest_al_message,
    _render_log_message,
    _reset_weight_suggest_al_logger,
    _set_weight_suggest_debug_summary_only,
    _setup_weight_suggest_al_logger,
)
from mainlib.runner import run
from mainlib.simulator import (
    _make_scheduler,
    _serialize_schedule,
    simulate_decode_progressive,
    simulate_prefill,
)
from mainlib.storage import (
    ALL_PASSES_RESULT_PATH,
    BEST_PASS_SUMMARY_PATH,
    _artifact_tag_token,
    _build_result_dir,
    _build_tag,
    _build_uniform_weight_storage_map,
    _collect_weight_ids_from_graph,
    _normalize_weight_storage_fmt,
    _result_stride_for_naming,
    _storage_mode_display_name,
    _weight_map_format_counts,
    _weight_map_summary,
)
from mainlib.weight_formats import (
    _as_bool,
    _base_weight_family,
    _block_local_weight_name,
    _build_weight_blocks,
    _coerce_fraction,
    _coerce_positive_int,
    _dominant_block_fmt,
    _normalize_block_span_overrides,
    _normalize_reload_count_mode,
    _resolve_block_layer_span,
    _sa_make_neighbor_map,
    _split_layer_prefixed_weight_id,
    _strip_weight_shard_suffix,
    _weight_block_key,
    mapping_diff_ratio,
)

__all__ = [
    "ALL_PASSES_RESULT_PATH",
    "BEST_PASS_SUMMARY_PATH",
    "PD_BASELINES",
    "auto_select_kv_policy",
    "evaluate_suite",
    "evaluate_burstgpt_suite",
    "load_burstgpt_csv",
    "main",
    "mapping_diff_ratio",
    "parse_args",
    "register_baseline",
    "run",
    "simulate_decode_progressive",
    "simulate_prefill",
]


if __name__ == "__main__":
    main()
