"""High-level CLI application entrypoint."""

from __future__ import annotations

from .shared import *
from .cli import (
    _apply_runtime_config_overrides,
    _load_cfg_from_json,
    _normalize_list_field,
    _resolve_cfg_paths,
    parse_args,
)
from .evaluate import evaluate_suite
from .kv_policy import _ensure_weight_suggest_supported, _normalize_npu_backend
from .log_utils import _set_weight_suggest_debug_summary_only, _setup_weight_suggest_al_logger
from .runner import run
from .storage import _build_result_dir, _build_tag


def main():
    args = parse_args()

    if getattr(args, 'mode', None) in ('evaluate', 'weight-suggest'):
        cfg = _load_cfg_from_json(getattr(args, 'config'))
        requested_debug = bool(getattr(args, 'debug', False)) or cfg.get('debug', False)
        cfg['debug'] = bool(requested_debug)
        cfg['_weight_suggest_debug_summary_only'] = bool(getattr(args, 'mode', None) == 'weight-suggest')
        _set_weight_suggest_debug_summary_only(
            cfg['_weight_suggest_debug_summary_only'],
            emit_progress=bool(requested_debug),
        )
        if cfg['_weight_suggest_debug_summary_only']:
            cfg['_requested_debug'] = bool(requested_debug)
            cfg['debug'] = bool(requested_debug)

        override_fields = [
            'model_family',
            'model_variant',
            'dtype',
            'batch',
            'prefill_len',
            'decode_len',
            'decode_sample_stride',
            'decode_plan_refresh_stride',
            'tp_qkv',
            'tp_ffn',
            'tp_moe',
            'result_dir',
            'hardware_json',
            'algo',
            'baselines',
            'all_passes_json',
            'best_summary_json',
            'weight_format_json',
            'npu_backend',
            'pim_fast_mode',
            'pim_weight_load_overlap_ratio',
            'weight_load_compute_overlap_ratio',
            'format_outer_max_iters',
            'format_block_change_percent',
            'format_inner_max_blocks',
            'format_nd_margin_init',
            'format_nd_margin_decay',
            'format_nd_margin_min',
            'format_inner_improve_eps',
            'format_outer_stop_eps',
            'format_block_layer_span',
            'format_reload_count_mode',
        ]
        for key in override_fields:
            val = getattr(args, key, None)
            if val is not None:
                cfg[key] = val

        cfg = _resolve_cfg_paths(cfg, config_path=cfg.get('_config_path'))
        cfg['dtype'] = normalize_dtype_token(cfg.get('dtype', 'fp16'), default='fp16')

        runtime_cfg_overrides = _apply_runtime_config_overrides(cfg)
        if runtime_cfg_overrides:
            print(f"[runtime-config] applied {json.dumps(runtime_cfg_overrides, ensure_ascii=False, sort_keys=True)}")

        if cfg.get('npu_backend', None) is None:
            raise ValueError("Missing required config key: 'npu_backend'. Choose from: fast, ascend_310b_lut, ascend_310b_json, llmcompass")
        cfg['npu_backend'] = _normalize_npu_backend(cfg.get('npu_backend'))

        if args.mode == 'weight-suggest':
            _ensure_weight_suggest_supported(cfg)

        result_dir = str(_build_result_dir(cfg, cfg.get('result_dir') or './output'))
        cfg['result_dir'] = result_dir
        if args.mode == 'weight-suggest':
            ws_al_log_path = cfg.get('weight_suggest_al_log_path')
            if not ws_al_log_path:
                ws_al_log_path = str(Path(result_dir) / "weight_suggest_al_debug.txt")
            cfg['weight_suggest_al_log_path'] = str(ws_al_log_path)
        else:
            _setup_weight_suggest_al_logger(None)

        tag = f"{int(cfg.get('prefill_len', 0))}x{int(cfg.get('decode_len', 0))}"
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        setup_logging(bool(cfg.get('debug', False)), log_file=str(Path(result_dir) / "driver_debug.txt"))
        if args.mode == 'weight-suggest':
            _setup_weight_suggest_al_logger(cfg.get('weight_suggest_al_log_path'))

        if cfg.get('decode_sample_stride', None) is not None:
            try:
                cfg['decode_sample_stride'] = int(cfg['decode_sample_stride'])
            except Exception:
                pass

        if args.mode == 'weight-suggest':
            algo_field = cfg.get('algo', 'HEFT')
            if isinstance(algo_field, list):
                algo_chosen = str(algo_field[0]) if algo_field else 'HEFT'
            else:
                parts = [t for t in str(algo_field).replace(',', ' ').split() if t]
                algo_chosen = parts[0] if parts else 'HEFT'

            tag = _build_tag(cfg)
            if isinstance(cfg.get('tag'), str) and cfg['tag'].strip():
                tag = f"{tag}_{cfg['tag'].strip()}"

            if not cfg.get('all_passes_json'):
                cfg['all_passes_json'] = str(Path(result_dir) / f"all_passes_{tag}.json")
            if not cfg.get('best_summary_json'):
                cfg['best_summary_json'] = str(Path(result_dir) / f"best_summary_{tag}.json")
            if not cfg.get('weight_format_json'):
                cfg['weight_format_json'] = str(Path(result_dir) / f"weight_storage_suggestion_{tag}.json")

            cfg['simulation_log_file'] = str(Path(result_dir) / f"pim_sim_{tag}.txt")

            print(f"[weight-suggest] algo={_display_policy_name(algo_chosen)} result_dir={result_dir} tag={tag}")
            run(cfg)
            return

        if args.mode == 'evaluate':
            algos = _normalize_list_field(cfg.get('algo', 'HEFT'))
            baselines = _normalize_list_field(cfg.get('baselines', 'PD,weights_on_pim,AF'))
            baseline_out = cfg.get('baseline_out') or str(Path(result_dir) / f"baseline_compare_{tag}.json")
            print(
                f"[evaluate] algos={[ _display_policy_name(a) for a in algos ]} "
                f"baselines={[ _display_policy_name(b) for b in baselines ]} "
                f"result_dir={result_dir} tag={tag}"
            )
            evaluate_suite(
                cfg,
                algos=algos,
                baselines=baselines,
                result_dir=result_dir,
                debug=cfg['debug'],
                combined_out=baseline_out,
            )
            return
