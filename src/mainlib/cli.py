"""CLI argument parsing and runtime-config normalization helpers."""

from __future__ import annotations

import sys

from .shared import *


_INPUT_PATH_KEYS = {
    'hardware_json',
    'pim_config_path',
    'ramulator_config_path',
    'burstgpt_csv',
    'workload_path',
    'request_trace_path',
}

_OUTPUT_PATH_KEYS = {
    'result_dir',
    'simulation_log_file',
    'dump_graph_dir',
    'all_passes_json',
    'best_summary_json',
    'weight_format_json',
    'baseline_out',
    'serve_out',
}


def _parse_bool_token(token: Any) -> bool:
    s = str(token).strip().lower()
    if s in {'1', 'true', 't', 'yes', 'y', 'on', 'enable', 'enabled'}:
        return True
    if s in {'0', 'false', 'f', 'no', 'n', 'off', 'disable', 'disabled'}:
        return False
    raise argparse.ArgumentTypeError(f'invalid boolean token: {token!r}')


def _add_bifocal_override_args(sp) -> None:
    sp.add_argument('--decode_horizon_len', type=int, help='Planning horizon visible to Bifocal token-amortization; defaults to decode_len.')
    sp.add_argument('--bifocal-ready-score-enable', '--bifocal_ready_score_enable', dest='bifocal_ready_score_enable', type=_parse_bool_token)
    sp.add_argument('--bifocal-lookahead-enable', '--bifocal_lookahead_enable', dest='bifocal_lookahead_enable', type=_parse_bool_token)
    sp.add_argument('--bifocal-phase-reuse-enable', '--bifocal_phase_reuse_enable', dest='bifocal_phase_reuse_enable', type=_parse_bool_token)
    sp.add_argument('--bifocal-token-amort-enable', '--bifocal_token_amort_enable', dest='bifocal_token_amort_enable', type=_parse_bool_token)
    sp.add_argument('--bifocal-h', '--bifocal_h', dest='bifocal_h', type=int, help='Override SCHED_JOINT_LK_H.')
    sp.add_argument('--bifocal-gamma', '--bifocal_gamma', dest='bifocal_gamma', type=float, help='Override SCHED_JOINT_LK_GAMMA.')
    sp.add_argument('--bifocal-lambda', '--bifocal_lambda', dest='bifocal_lambda', type=float, help='Override SCHED_JOINT_LK_CONSIST_LAMBDA.')
    sp.add_argument('--bifocal-plan-hint-max', '--bifocal_plan_hint_max', dest='bifocal_plan_hint_max', type=int, help='Override SCHED_JOINT_LK_PLAN_HINT_MAX.')
    sp.add_argument('--bifocal-eta', '--bifocal_eta', dest='bifocal_eta', type=float, help='Override SCHED_WEIGHT_BIAS_ETA.')
    sp.add_argument('--bifocal-amort-alpha', '--bifocal_amort_alpha', dest='bifocal_amort_alpha', type=float, help='Override SCHED_DECODE_AMORT_ALPHA.')
    sp.add_argument('--bifocal-amort-rmin', '--bifocal_amort_rmin', dest='bifocal_amort_rmin', type=float, help='Override SCHED_DECODE_AMORT_RMIN.')
    sp.add_argument('--bifocal-amort-reuse-prob', '--bifocal_amort_reuse_prob', dest='bifocal_amort_reuse_prob', type=float, help='Override SCHED_DECODE_AMORT_REUSE_PROB.')


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode')

    sp_eval = sub.add_parser('evaluate', help='Run selected algorithms and baselines; outputs go under result_dir.')
    sp_eval.add_argument('--config', required=True, type=str, help='Path to a JSON config with run parameters.')
    sp_eval.add_argument('--debug', action='store_true', help='Enable verbose logging.')
    sp_eval.add_argument('--model_family', type=str)
    sp_eval.add_argument('--model_variant', type=str)
    sp_eval.add_argument('--dtype', type=str)
    sp_eval.add_argument('--batch', type=int)
    sp_eval.add_argument('--prefill_len', type=int)
    sp_eval.add_argument('--decode_len', type=int)
    sp_eval.add_argument('--decode_sample_stride', type=int)
    sp_eval.add_argument(
        '--decode_plan_refresh_stride',
        type=int,
        help='Run a full decode search every N tokens; hidden tokens replay a fixed plan. 0 means freeze after warmup.',
    )
    sp_eval.add_argument('--result_dir', type=str)
    sp_eval.add_argument('--hardware_json', type=str, help='Path to a JSON file with hardware topology (devices + links).')
    sp_eval.add_argument('--algo', type=str, help='Algorithm list, for example "HEFT,Bifocal" or a single name.')
    sp_eval.add_argument('--baselines', type=str, help='Baseline list, for example "PD,AF,PD+Linear".')
    sp_eval.add_argument(
        '--npu_backend',
        type=str,
        default=None,
        choices=['fast', 'fast_mode', 'lut', 'ascend_310b_json', 'llmcompass'],
        help='NPU operator-latency backend: fast/lut/llmcompass. Must be explicitly specified in config JSON or CLI.',
    )
    sp_eval.add_argument('--pim_fast_mode', action='store_true', default=None)
    sp_eval.add_argument(
        '--pim-weight-load-overlap-ratio',
        dest='pim_weight_load_overlap_ratio',
        type=float,
        help='Override config.PIM_WEIGHT_LOAD_OVERLAP_RATIO in [0,1] for this process only.',
    )
    sp_eval.add_argument(
        '--weight-load-compute-overlap-ratio',
        dest='weight_load_compute_overlap_ratio',
        type=float,
        help='Override config.WEIGHT_LOAD_COMPUTE_OVERLAP_RATIO in [0,1] for this process only.',
    )
    sp_eval.add_argument('--tp_qkv', type=int, help='Tensor-parallel shard size for Q/K/V generation and attention head sharding.')
    sp_eval.add_argument('--tp_ffn', type=int, help='Tensor-parallel shard size for FFN intermediate dimension.')
    sp_eval.add_argument('--tp_moe', type=int, help='Expert-parallel shard size for MoE experts / Mixtral routing.')
    _add_bifocal_override_args(sp_eval)

    sp_ws = sub.add_parser('weight-suggest', help='Run multi-pass SA to suggest weight formats and fixed baseline experiments.')
    sp_ws.add_argument('--config', required=True, type=str, help='Path to a JSON config with run parameters.')
    sp_ws.add_argument('--debug', action='store_true', help='Enable verbose logging.')
    sp_ws.add_argument('--model_family', type=str)
    sp_ws.add_argument('--model_variant', type=str)
    sp_ws.add_argument('--dtype', type=str)
    sp_ws.add_argument('--batch', type=int)
    sp_ws.add_argument('--prefill_len', type=int)
    sp_ws.add_argument('--decode_len', type=int)
    sp_ws.add_argument('--decode_sample_stride', type=int)
    sp_ws.add_argument(
        '--decode_plan_refresh_stride',
        type=int,
        help='Run a full decode search every N tokens; hidden tokens replay a fixed plan. 0 means freeze after warmup.',
    )
    sp_ws.add_argument('--result_dir', type=str)
    sp_ws.add_argument('--hardware_json', type=str, help='Path to a JSON file with hardware topology (devices + links).')
    sp_ws.add_argument('--algo', type=str, help='Algorithm list, for example "HEFT,Bifocal".')
    sp_ws.add_argument('--all_passes_json', type=str, help='Override path for all passes JSON.')
    sp_ws.add_argument('--best_summary_json', type=str, help='Override path for best pass summary JSON.')
    sp_ws.add_argument('--weight_format_json', type=str, help='Override path for accepted weight format JSON.')
    sp_ws.add_argument(
        '--npu_backend',
        type=str,
        default=None,
        choices=['fast', 'fast_mode', 'lut', 'ascend_310b_json', 'llmcompass'],
        help='NPU operator-latency backend: fast/lut/llmcompass. Must be explicitly specified in config JSON or CLI.',
    )
    sp_ws.add_argument('--pim_fast_mode', action='store_true', default=None)
    sp_ws.add_argument(
        '--pim-weight-load-overlap-ratio',
        dest='pim_weight_load_overlap_ratio',
        type=float,
        help='Override config.PIM_WEIGHT_LOAD_OVERLAP_RATIO in [0,1] for this process only.',
    )
    sp_ws.add_argument(
        '--weight-load-compute-overlap-ratio',
        dest='weight_load_compute_overlap_ratio',
        type=float,
        help='Override config.WEIGHT_LOAD_COMPUTE_OVERLAP_RATIO in [0,1] for this process only.',
    )
    sp_ws.add_argument('--tp_qkv', type=int, help='Tensor-parallel shard size for Q/K/V generation and attention head sharding.')
    sp_ws.add_argument('--tp_ffn', type=int, help='Tensor-parallel shard size for FFN intermediate dimension.')
    sp_ws.add_argument('--tp_moe', type=int, help='Expert-parallel shard size for MoE experts / Mixtral routing.')
    _add_bifocal_override_args(sp_ws)
    sp_ws.add_argument(
        '--format_outer_max_iters',
        type=int,
        help='Deprecated compatibility knob. If format_block_change_percent is unset, percent is derived as 1/format_outer_max_iters.',
    )
    sp_ws.add_argument('--format_block_change_percent', type=float, help='At most this fraction of total blocks may change per outer iteration.')
    sp_ws.add_argument('--format_inner_max_blocks', type=int, help='AL inner sweep cap (0 means no cap).')
    sp_ws.add_argument('--format_nd_margin_init', type=float, help='AL initial ND band in [0,1].')
    sp_ws.add_argument('--format_nd_margin_decay', type=float, help='AL ND band decay per outer iteration.')
    sp_ws.add_argument('--format_nd_margin_min', type=float, help='AL minimum ND band.')
    sp_ws.add_argument('--format_inner_improve_eps', type=float, help='AL accept change if new_total + eps < old_total.')
    sp_ws.add_argument('--format_outer_stop_eps', type=float, help='AL stop when outer_n is worse than outer_{n-1} by eps.')
    sp_ws.add_argument('--format_block_layer_span', type=int, help='Group the same weight across every N layers into one block.')
    sp_ws.add_argument(
        '--format_reload_count_mode',
        type=str,
        choices=['raw', 'per_device', 'soft_per_device'],
        help='Use raw reload totals or normalize by the number of devices of each type when comparing NPU vs PIM reload pressure.',
    )


    sp_burst = sub.add_parser('burstgpt-evaluate', help='Replay a BurstGPT CSV trace and report TTFT/TBT/E2E p50/p90.')
    sp_burst.add_argument('--config', required=True, type=str, help='Path to a JSON config with model/hardware/policy settings.')
    sp_burst.add_argument('--debug', action='store_true', help='Enable verbose logging.')
    sp_burst.add_argument('--model_family', type=str)
    sp_burst.add_argument('--model_variant', type=str)
    sp_burst.add_argument('--dtype', type=str)
    sp_burst.add_argument('--result_dir', type=str)
    sp_burst.add_argument('--hardware_json', type=str, help='Path to a JSON file with hardware topology (devices + links).')
    sp_burst.add_argument('--algo', type=str, help='Algorithm list, for example "Bifocal".')
    sp_burst.add_argument('--baselines', type=str, help='Baseline list, for example "PD,AF,PD+Linear,PD+Attn,PD+FFN".')
    sp_burst.add_argument('--npu_backend', type=str, default=None, choices=['fast', 'fast_mode', 'lut', 'ascend_310b_json', 'llmcompass'])
    sp_burst.add_argument('--pim_fast_mode', action='store_true', default=None)
    sp_burst.add_argument('--pim-weight-load-overlap-ratio', dest='pim_weight_load_overlap_ratio', type=float)
    sp_burst.add_argument('--weight-load-compute-overlap-ratio', dest='weight_load_compute_overlap_ratio', type=float)
    sp_burst.add_argument('--tp_qkv', type=int)
    sp_burst.add_argument('--tp_ffn', type=int)
    sp_burst.add_argument('--tp_moe', type=int)
    _add_bifocal_override_args(sp_burst)
    sp_burst.add_argument('--decode_sample_stride', type=int)
    sp_burst.add_argument('--decode_plan_refresh_stride', type=int)

    # BurstGPT trace controls.
    sp_burst.add_argument('--burstgpt_csv', type=str, help='Path to BurstGPT_without_fails_*.csv or BurstGPT_*.csv.')
    sp_burst.add_argument('--workload_path', type=str, help='Alias of --burstgpt_csv for compatibility.')
    sp_burst.add_argument('--request_trace_path', type=str, help='Alias of --burstgpt_csv for compatibility.')
    sp_burst.add_argument('--num_requests', type=int, help='Use only the first N valid requests from the CSV.')
    sp_burst.add_argument('--arrival_time_scale', type=float, help='Scale BurstGPT timestamps; 0.1 compresses time by 10x and increases load.')
    sp_burst.add_argument('--burstgpt_model_filter', type=str, help='Optional model filter, e.g. "ChatGPT" or "GPT-4".')
    sp_burst.add_argument('--skip_zero_output', action='store_true', default=None)
    sp_burst.add_argument('--min_input_len', type=int)
    sp_burst.add_argument('--min_output_len', type=int)
    sp_burst.add_argument('--max_input_len', type=int)
    sp_burst.add_argument('--max_output_len', type=int)

    # Serving replay controls.
    sp_burst.add_argument('--serving_batch_size', type=int, help='Maximum FCFS micro-batch size. Use 1 for no batching.')
    sp_burst.add_argument('--batch_timeout_s', type=float, help='Optional max wait to form a micro-batch.')
    sp_burst.add_argument('--prompt_bucket_size', type=int, help='Round prompt lengths up to this multiple before profiling.')
    sp_burst.add_argument('--output_bucket_size', type=int, help='Round output lengths up to this multiple before profiling.')
    sp_burst.add_argument('--output_horizon', type=str, choices=['p90', 'fixed', 'oracle'], help='Decode horizon hint seen by Bifocal. p90 is the non-oracle default.')
    sp_burst.add_argument('--output_horizon_fixed', type=int, help='Fixed decode horizon when --output_horizon fixed is used.')
    sp_burst.add_argument('--serve_out', type=str, help='Output summary JSON path.')

    args, _unknown = parser.parse_known_args()
    if args.mode is None:
        parser.error("Please specify a mode: 'evaluate', 'weight-suggest', or 'burstgpt-evaluate'.")
    return args


def _normalize_list_field(val) -> list[str]:
    items: list[str] = []
    seq = val if isinstance(val, list) else [val]
    for item in seq:
        if item is None:
            continue
        for tok in str(item).replace(',', ' ').split():
            tok = tok.strip()
            if tok:
                items.append(tok)
    return items


def _resolve_project_path(value: Any, *, config_path: str | None = None, must_exist: bool) -> str:
    raw = str(value or '').strip()
    if not raw:
        return raw

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    src_root = Path(__file__).resolve().parent.parent
    candidates = []
    if config_path:
        config_dir = Path(config_path).expanduser().resolve().parent
        candidates.append((config_dir / candidate).resolve())
    candidates.append((src_root / candidate).resolve())
    candidates.append((Path.cwd() / candidate).resolve())

    if must_exist:
        for cand in candidates:
            if cand.exists():
                return str(cand)
        return str(candidates[0])

    return str(candidates[0])


def _resolve_cfg_paths(cfg: Dict, *, config_path: str | None = None) -> Dict:
    out = dict(cfg)
    for key in _INPUT_PATH_KEYS:
        if out.get(key):
            out[key] = _resolve_project_path(out[key], config_path=config_path, must_exist=True)
    for key in _OUTPUT_PATH_KEYS:
        if out.get(key):
            out[key] = _resolve_project_path(out[key], config_path=config_path, must_exist=False)
    return out


def _load_cfg_from_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config JSON must be an object/dict, got: {type(raw).__name__}")
    cfg = dict(raw)
    cfg['_config_path'] = str(Path(path).expanduser().resolve())
    cfg = _resolve_cfg_paths(cfg, config_path=cfg['_config_path'])
    return cfg


def _apply_runtime_config_overrides(cfg: Dict) -> Dict[str, Any]:
    """Apply per-run overrides to the imported config module without editing config.py on disk."""
    import config as _runtime_config

    applied: Dict[str, Any] = {}

    def _apply_ratio(cfg_key: str, runtime_key: str) -> None:
        if cfg_key not in cfg or cfg.get(cfg_key) is None:
            return
        try:
            ratio = float(cfg.get(cfg_key))
        except Exception as exc:
            raise ValueError(
                f"Invalid {cfg_key}={cfg.get(cfg_key)!r}; expected a float in [0, 1]"
            ) from exc
        if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"{cfg_key} must be within [0, 1], got {ratio!r}")
        setattr(_runtime_config, runtime_key, float(ratio))
        applied[runtime_key] = float(ratio)

    _apply_ratio('pim_weight_load_overlap_ratio', 'PIM_WEIGHT_LOAD_OVERLAP_RATIO')
    _apply_ratio('weight_load_compute_overlap_ratio', 'WEIGHT_LOAD_COMPUTE_OVERLAP_RATIO')

    bifocal_map = {
        'bifocal_ready_score_enable': ('SCHED_BIFOCAL_READY_SCORE_ENABLE', bool),
        'bifocal_lookahead_enable': ('SCHED_BIFOCAL_LOOKAHEAD_ENABLE', bool),
        'bifocal_phase_reuse_enable': ('SCHED_BIFOCAL_PHASE_REUSE_ENABLE', bool),
        'bifocal_token_amort_enable': ('SCHED_BIFOCAL_TOKEN_AMORT_ENABLE', bool),
        'bifocal_h': ('SCHED_JOINT_LK_H', int),
        'bifocal_gamma': ('SCHED_JOINT_LK_GAMMA', float),
        'bifocal_lambda': ('SCHED_JOINT_LK_CONSIST_LAMBDA', float),
        'bifocal_plan_hint_max': ('SCHED_JOINT_LK_PLAN_HINT_MAX', int),
        'bifocal_eta': ('SCHED_WEIGHT_BIAS_ETA', float),
        'bifocal_amort_alpha': ('SCHED_DECODE_AMORT_ALPHA', float),
        'bifocal_amort_rmin': ('SCHED_DECODE_AMORT_RMIN', float),
        'bifocal_amort_reuse_prob': ('SCHED_DECODE_AMORT_REUSE_PROB', float),
    }

    def _coerce_runtime_value(raw: Any, caster: Any) -> Any:
        if caster is bool:
            return bool(_parse_bool_token(raw))
        return caster(raw)

    modules = [_runtime_config]
    for mod_name in ('scheduler.scheduler_common', 'scheduler.scheduler_bifocal'):
        try:
            mod = sys.modules.get(mod_name)
            if mod is not None:
                modules.append(mod)
        except Exception:
            pass

    for cfg_key, (runtime_key, caster) in bifocal_map.items():
        if cfg_key not in cfg or cfg.get(cfg_key) is None:
            continue
        val = _coerce_runtime_value(cfg.get(cfg_key), caster)
        if runtime_key == 'SCHED_JOINT_LK_H' and int(val) < 1:
            raise ValueError(f'{cfg_key} must be >= 1, got {val!r}')
        if runtime_key in {'SCHED_JOINT_LK_GAMMA', 'SCHED_DECODE_AMORT_REUSE_PROB'}:
            fval = float(val)
            if not math.isfinite(fval) or fval < 0.0 or fval > 1.0:
                raise ValueError(f'{cfg_key} must be within [0, 1], got {val!r}')
        if runtime_key in {'SCHED_JOINT_LK_CONSIST_LAMBDA', 'SCHED_WEIGHT_BIAS_ETA', 'SCHED_DECODE_AMORT_ALPHA', 'SCHED_DECODE_AMORT_RMIN'}:
            fval = float(val)
            if not math.isfinite(fval) or fval < 0.0:
                raise ValueError(f'{cfg_key} must be non-negative, got {val!r}')
        for mod in modules:
            try:
                setattr(mod, runtime_key, val)
            except Exception:
                pass
        applied[runtime_key] = val

    return applied
