"""CLI argument parsing and runtime-config normalization helpers."""

from __future__ import annotations

from .shared import *


_INPUT_PATH_KEYS = {
    'hardware_json',
    'shape_file',
    'pim_config_path',
    'ramulator_config_path',
    'burstgpt_csv',
    'workload_path',
    'request_trace_path',
    'gpu_runtime_model_json',
}

_OUTPUT_PATH_KEYS = {
    'result_dir',
    'simulation_log_file',
    'dump_graph_dir',
    'all_passes_json',
    'best_summary_json',
    'hetinfer_prior_out',
    'weight_format_json',
    'baseline_out',
    'serve_out',
}


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
    sp_eval.add_argument('--pim_fast_mode', action=argparse.BooleanOptionalAction, default=None)
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
    sp_eval.add_argument(
        '--hetinfer-prior-out',
        dest='hetinfer_prior_out',
        type=str,
        help='Write a separate dops.hetinfer_prior.v1 artifact (Bifocal only). A directory gets an automatic filename.',
    )

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
    sp_ws.add_argument('--pim_fast_mode', action=argparse.BooleanOptionalAction, default=None)
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
    sp_ws.add_argument(
        '--hetinfer-prior-out',
        dest='hetinfer_prior_out',
        type=str,
        help='After choosing the best layout, re-evaluate it with Bifocal and write dops.hetinfer_prior.v1.',
    )
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
    sp_burst.add_argument('--pim_fast_mode', action=argparse.BooleanOptionalAction, default=None)
    sp_burst.add_argument('--pim-weight-load-overlap-ratio', dest='pim_weight_load_overlap_ratio', type=float)
    sp_burst.add_argument('--weight-load-compute-overlap-ratio', dest='weight_load_compute_overlap_ratio', type=float)
    sp_burst.add_argument('--tp_qkv', type=int)
    sp_burst.add_argument('--tp_ffn', type=int)
    sp_burst.add_argument('--tp_moe', type=int)
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

    runtime_model_path = cfg.get('gpu_runtime_model_json')
    if runtime_model_path not in (None, ''):
        backend = str(cfg.get('npu_backend', '') or '').strip().lower().replace('-', '_')
        if backend not in {'fast', 'fast_mode', 'fastmode'}:
            raise ValueError(
                "gpu_runtime_model_json is only valid with npu_backend=fast; "
                f"got npu_backend={cfg.get('npu_backend')!r}"
            )
        path = Path(str(runtime_model_path)).expanduser()
        try:
            runtime_model = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            raise ValueError(f"Failed to read gpu_runtime_model_json={path}: {exc}") from exc
        if not isinstance(runtime_model, dict):
            raise ValueError("gpu_runtime_model_json must contain a JSON object")
        if runtime_model.get('schema') != 'dops.gpu_runtime_model.v1':
            raise ValueError(
                "gpu_runtime_model_json schema must equal 'dops.gpu_runtime_model.v1'"
            )
        prefix = str(runtime_model.get('device_name_prefix', '') or '').strip()
        if not prefix:
            raise ValueError("gpu_runtime_model_json.device_name_prefix must be non-empty")
        compute = runtime_model.get('compute_utilization')
        launch = runtime_model.get('kernel_launch_overhead')
        if not isinstance(compute, dict) or not compute:
            raise ValueError("gpu_runtime_model_json.compute_utilization must be a non-empty object")
        if not isinstance(launch, dict) or not launch:
            raise ValueError("gpu_runtime_model_json.kernel_launch_overhead must be a non-empty object")

        compute_root = dict(getattr(_runtime_config, 'COMPUTE_UTILIZATION', {}) or {})
        compute_by_name = dict(compute_root.get('by_device_name', compute_root.get('by_name', {})) or {})
        compute_by_name[prefix] = dict(compute)
        compute_root['by_device_name'] = compute_by_name
        _runtime_config.COMPUTE_UTILIZATION = compute_root

        launch_root = dict(getattr(_runtime_config, 'KERNEL_LAUNCH_OVERHEAD', {}) or {})
        launch_by_name = dict(launch_root.get('by_device_name', launch_root.get('by_name', {})) or {})
        launch_by_name[prefix] = dict(launch)
        launch_root['by_device_name'] = launch_by_name
        _runtime_config.KERNEL_LAUNCH_OVERHEAD = launch_root
        applied['GPU_RUNTIME_MODEL'] = {
            'path': str(path.resolve()),
            'device_name_prefix': prefix,
            'source_fit_sha256': runtime_model.get('source_fit_sha256'),
        }
    return applied
