"""Weight-suggest search entrypoint extracted from main.py."""

from __future__ import annotations

from .shared import *
from .log_utils import _debug
from .storage import (
    ALL_PASSES_RESULT_PATH,
    BEST_PASS_SUMMARY_PATH,
    _artifact_tag_token,
    _best_summary_config_snapshot,
    _build_result_dir,
    _build_tag,
    _build_uniform_weight_storage_map,
    _collect_weight_ids_from_graph,
    _normalize_weight_storage_fmt,
    _resolve_hetinfer_prior_output,
    _storage_mode_display_name,
    _weight_map_summary,
)
from hetinfer_prior import build_artifact as build_hetinfer_prior_artifact
from hetinfer_prior import write_artifact as write_hetinfer_prior_artifact
from .weight_formats import (
    _build_weight_blocks,
    _coerce_fraction,
    _dominant_block_fmt,
    _normalize_reload_count_mode,
    _sa_make_neighbor_map,
    mapping_diff_ratio,
)
from .graph_utils import _cluster_type_count
from .kv_policy import (
    _ensure_weight_suggest_supported,
    _apply_kv_place_constraints,
    _build_cost_model_for_run,
    _fmt_kv_policy_scores,
    _infer_kv_place_from_label,
    auto_select_kv_policy,
)
from .simulator import simulate_decode_progressive, simulate_prefill
from .evaluate import _eval_one_baseline, _run_strategy_once


def _comparison_cfg_without_prior_export(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Isolate fixed comparison runs from the selected-layout prior output.

    ``weight-suggest`` writes the Het-Infer prior exactly once, immediately
    after re-evaluating ``best_map``.  The later Bifocal+Linear/Dual comparison
    runs must not inherit the same output path and overwrite that artifact.
    """

    out = dict(cfg)
    out.pop("hetinfer_prior_out", None)
    return out


def run(cfg: Dict):
    _ensure_weight_suggest_supported(cfg)

    #--------------------------------------------
    # 0: init all hardware settings
    #--------------------------------------------
    result_dir = Path(cfg.get('result_dir') or _build_result_dir(cfg, './output/weight_suggestions'))
    result_dir.mkdir(parents=True, exist_ok=True)
    weight_format_path = Path(cfg.get('weight_format_json') or (result_dir / 'weight_storage_suggestion.json'))
    compare_path = Path(
        cfg.get('weight_format_compare_json')
        or weight_format_path.with_name(weight_format_path.stem + '_compare' + weight_format_path.suffix)
    )
    cluster = demo_cluster(cfg)
    cfg['topology'] = getattr(cluster, 'topology', cfg.get('topology', None))
    prefill_len = int(cfg.get('prefill_len', 128))
    batch = int(cfg.get('batch', 1))
    graph, shape = build_graph(cfg)
    sim_log_file = cfg.get('simulation_log_file', str(result_dir / 'pim_simulation.txt'))
    cost = _build_cost_model_for_run(
        cfg,
        cluster,
        shape,
        simulation_log_file=sim_log_file,
        debug_traces=False,
    )
    cost.logger.start_simulation()

    fmt_map: Dict[str, str] = {}
    best_total: float | None = None
    best_map: Dict[str, str] = {}
    best_pass: int = -1
    all_pass_records: List[Dict] = []
    buffer_mgr = GlobalMemoryManager()
    search_start_mode = 'ND'
    fixed_baseline_experiments = [
        {
            'experiment_id': 'PD+Linear',
            'display_name': 'algo:PD+Linear',
            'algo': 'PD',
            'storage_fmt': 'ND',
            'runner': 'baseline',
        },
        {
            'experiment_id': 'PD+Dual',
            'display_name': 'algo:PD+Dual',
            'algo': 'PD',
            'storage_fmt': 'DUAL',
            'runner': 'baseline',
        },
        {
            'experiment_id': 'Bifocal+Linear',
            'display_name': 'algo:Bifocal+Linear',
            'algo': 'Bifocal',
            'storage_fmt': 'ND',
            'runner': 'strategy',
        },
        {
            'experiment_id': 'Bifocal+Dual',
            'display_name': 'algo:Bifocal+Dual',
            'algo': 'Bifocal',
            'storage_fmt': 'DUAL',
            'runner': 'strategy',
        },
    ]

    # Choose scheduler class for the tuning run (default: HEFT).
    algo_raw = cfg.get('algo', 'HEFT')
    if isinstance(algo_raw, list):
        algo_name = str(algo_raw[0]) if algo_raw else 'HEFT'
    else:
        algo_name = str(algo_raw)
    algo_name = _normalize_algo_name((algo_name.replace(',', ' ').split()[:1] or ['HEFT'])[0])

    SchedCls = HEFTScheduler
    if algo_name == 'Bifocal':
        if BifocalScheduler is None:
            raise ImportError("BifocalScheduler is not available. Please export it from the scheduler package.")
        SchedCls = BifocalScheduler
    elif algo_name not in ('HEFT', 'Naive', ''):
        _debug(f"[weight-suggest] Unknown algorithm '{algo_name}', fallback to HEFTScheduler")

    label = auto_select_kv_policy(
        strategy=algo_name,
        cfg=cfg,
        cluster=cluster,
        cost=cost,
        graph=graph,
        shape=shape,
    )

    sel = getattr(label, 'kv_policy_selected', 'unknown')
    sc = getattr(label, 'kv_policy_scores', {})
    msg = _fmt_kv_policy_scores(sc)
    if msg:
        _debug(str(f"[KV-SELECT] selected={sel} | {msg}"))
    else:
        _debug(str(f"[KV-SELECT] selected={sel}"))

    kv_place = _infer_kv_place_from_label(label)
    graph_kv = _apply_kv_place_constraints(graph, kv_place)

    # ------------------------------------------------------------
    # 1: block-CD + 2-layer BFS/beam search
    # ------------------------------------------------------------
    legacy_outer_max_cfg = cfg.get('format_outer_max_iters', cfg.get('outer_max_iters', None))
    block_change_percent_cfg = cfg.get('format_block_change_percent', None)
    if block_change_percent_cfg is None:
        try:
            legacy_outer_max = int(legacy_outer_max_cfg or 0)
        except Exception:
            legacy_outer_max = 0
        if legacy_outer_max > 0:
            block_change_percent = float(1.0 / float(legacy_outer_max))
        else:
            block_change_percent = 0.20
    else:
        block_change_percent = _coerce_fraction(block_change_percent_cfg, default=0.20)
        if block_change_percent <= 0.0:
            block_change_percent = 0.20

    inner_max_blocks = int(cfg.get('format_inner_max_blocks', 0) or 0)  # 0 => no cap
    inner_improve_eps = float(cfg.get('format_inner_improve_eps', 1e-6) or 0.0)
    outer_stop_eps = float(cfg.get('format_outer_stop_eps', 0.0) or 0.0)
    block_layer_span = int(cfg.get('format_block_layer_span', 8) or 0)
    reload_count_mode_cfg = cfg.get('format_reload_count_mode', None)
    if reload_count_mode_cfg is None and ('format_normalize_reload_by_device_count' in cfg):
        reload_count_mode_cfg = 'per_device' if bool(cfg.get('format_normalize_reload_by_device_count')) else 'raw'
    reload_count_alpha = float(cfg.get('format_reload_device_count_alpha', 1.0) or 1.0)
    reload_count_mode = _normalize_reload_count_mode(reload_count_mode_cfg)
    type_device_counts = {
        'npu': max(1, _cluster_type_count(cluster, 'npu')),
        'pim': max(1, _cluster_type_count(cluster, 'pim')),
    }

    # Stable blocks built from model graph weight ids.
    all_wids = sorted({str(n.weight_id) for n in graph.nodes.values() if getattr(n, 'weight_id', None)})
    blocks = _build_weight_blocks(all_wids, layer_span=block_layer_span)
    outer_max = max(1, int(math.ceil(1.0 / float(block_change_percent))))
    max_outer_block_changes = 0
    if blocks:
        max_outer_block_changes = min(
            len(blocks),
            max(1, int(math.ceil(float(block_change_percent) * float(len(blocks))))),
        )

    _debug(
        f"[AL] init: weights={len(all_wids)} blocks={len(blocks)} "
        f"block_change_percent={block_change_percent:.3f} outer_max={outer_max} outer_topk={max_outer_block_changes} "
        f"inner_max_blocks={('inf' if not inner_max_blocks else int(inner_max_blocks))} "
        f"inner_eps={inner_improve_eps:g} outer_stop_eps={outer_stop_eps:g} "
        f"block_layer_span={block_layer_span} reload_count_mode={reload_count_mode} "
        f"device_counts(npu={type_device_counts['npu']}, pim={type_device_counts['pim']}) "
        f"search_start_mode={search_start_mode} baseline_experiments="
        f"{[spec['experiment_id'] for spec in fixed_baseline_experiments]}"
    )

    def _normalize_wlc(wstats: Dict) -> Dict[str, Dict[str, int]]:
        raw = (wstats or {}).get('weight_load_counts', {}) or {}
        out: Dict[str, Dict[str, int]] = {}
        for wid, cnts in raw.items():
            try:
                out[str(wid)] = {str(k): int(v) for k, v in (cnts or {}).items()}
            except Exception:
                out[str(wid)] = {}
        return out

    def _normalize_reload_count(dev_type: str, raw_cnt: int | float) -> float:
        value = float(raw_cnt or 0.0)

        if reload_count_mode == 'per_device':
            denom = float(max(1, int(type_device_counts.get(str(dev_type), 1) or 1)))
            return value / denom

        if reload_count_mode == 'soft_per_device':
            denom = float(max(1, int(type_device_counts.get(str(dev_type), 1) or 1)))
            alpha = max(0.0, float(reload_count_alpha))
            return value / (denom ** alpha)

        return value

    def _block_reload_counts(wlc: Dict[str, Dict[str, int]]) -> Dict[str, Tuple[float, float]]:
        """Return {block_key: (npu_pressure, pim_pressure)} aggregated across blocks."""
        out: Dict[str, Tuple[float, float]] = {}
        for bkey, wids in blocks.items():
            npu = 0.0
            pim = 0.0
            for w in wids:
                c = wlc.get(str(w), {}) or {}
                npu += _normalize_reload_count('npu', c.get('npu', 0) or 0)
                pim += _normalize_reload_count('pim', c.get('pim', 0) or 0)
            out[str(bkey)] = (float(npu), float(pim))
        return out

    def _apply_block_fmt(map_in: Dict[str, str], bkey: str, fmt: str) -> Dict[str, str]:
        """
        Set the storage format of a given “block” uniformly to fmt, and return a new format map (without modifying the original map)
        """
        out = dict(map_in or {})
        wids = blocks.get(str(bkey), [])
        for w in wids:
            if fmt == 'ND':
                out.pop(str(w), None)
            else:
                out[str(w)] = str(fmt)
        return out

    def _current_block_fmt(map_in: Dict[str, str], bkey: str) -> str:
        wids = blocks.get(str(bkey), [])
        if not wids:
            return 'ND'
        return str((map_in or {}).get(str(wids[0]), 'ND'))

    def _map_stats(map_in: Dict[str, str]) -> Dict[str, Any]:
        return _weight_map_summary(all_wids, map_in)

    def _effective_counts_for_cost(raw_counts: Dict[str, int | float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for dev_type, cnt in dict(raw_counts or {}).items():
            eff = float(_normalize_reload_count(str(dev_type), float(cnt or 0.0)))
            if eff > 0.0:
                out[str(dev_type)] = float(eff)
        return out

    def _rank_outer_block_updates(
        map_in: Dict[str, str],
        wstats: Dict,
        sched_eval: Any,
        *,
        only_if_current_nd: bool,
    ) -> List[Dict[str, Any]]:
        """Rank outer-step block flips by estimated cost reduction."""
        wlc = _normalize_wlc(wstats)
        try:
            chain_hits = {
                str(k): float(v or 0.0)
                for k, v in ((wstats or {}).get('weight_chain_hits', {}) or {}).items()
            }
        except Exception:
            chain_hits = {}
        max_chain_hits = float(max(chain_hits.values(), default=0.0)) if chain_hits else 0.0

        ranked: List[Dict[str, Any]] = []
        eps = 1e-12
        for bkey, wids in blocks.items():
            cur_fmt = _current_block_fmt(map_in, bkey)
            if only_if_current_nd and cur_fmt != 'ND':
                continue

            cost_by_fmt: Dict[str, float] = {'ND': 0.0, 'NZ': 0.0, 'PIM-OPT': 0.0}
            for wid in wids:
                counts_eff = _effective_counts_for_cost(wlc.get(str(wid), {}) or {})
                if not counts_eff:
                    continue
                for fmt in ('ND', 'NZ', 'PIM-OPT'):
                    cost_by_fmt[str(fmt)] += float(
                        sched_eval._estimate_weight_host_to_device_cost(
                            str(wid),
                            counts_eff,
                            str(fmt),
                            lookahead_beta=0.0,
                            max_chain_hits=max_chain_hits,
                            chain_hits=chain_hits,
                        )
                    )

            cur_cost = float(cost_by_fmt.get(cur_fmt, cost_by_fmt.get('ND', 0.0)))
            best_fmt = str(cur_fmt)
            best_cost = float(cur_cost)
            # Keep the current format on ties; for ND blocks this means we only
            # promote to NZ/PIM-OPT when the surrogate cost is strictly smaller.
            for fmt in ('ND', 'NZ', 'PIM-OPT'):
                trial_cost = float(cost_by_fmt.get(fmt, 0.0))
                if trial_cost + eps < best_cost:
                    best_cost = float(trial_cost)
                    best_fmt = str(fmt)

            gain = float(cur_cost - best_cost)
            if best_fmt != cur_fmt and gain > eps:
                ranked.append({
                    'block': str(bkey),
                    'cur_fmt': str(cur_fmt),
                    'next_fmt': str(best_fmt),
                    'gain': float(gain),
                    'cur_cost': float(cur_cost),
                    'next_cost': float(best_cost),
                    'costs': {str(k): float(v) for k, v in cost_by_fmt.items()},
                })

        ranked.sort(
            key=lambda item: (
                -float(item.get('gain', 0.0) or 0.0),
                -float(item.get('cur_cost', 0.0) or 0.0),
                str(item.get('block', '')),
            )
        )
        return ranked

    def _apply_ranked_outer_block_updates(
        map_in: Dict[str, str],
        ranked_updates: List[Dict[str, Any]],
        *,
        max_changes: int,
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        out = dict(map_in or {})
        applied: List[Dict[str, Any]] = []
        limit = max(0, int(max_changes))
        if limit <= 0 or not ranked_updates:
            return out, applied

        for item in ranked_updates[:limit]:
            out = _apply_block_fmt(out, str(item.get('block', '')), str(item.get('next_fmt', 'ND')))
            applied.append(dict(item))
        return out, applied

    def _evaluate_map(fmt_map_eval: Dict[str, str], *, tag: str) -> Tuple[float, float, float, Any, Any, Dict, Any]:
        """Run prefill+decode simulation under a given host format map."""
        sched = SchedCls(cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
        if str(tag) == 'hetinfer_prior_best_layout':
            enable_capture = getattr(sched, 'enable_hetinfer_candidate_capture', None)
            if callable(enable_capture):
                enable_capture(True)
        sched.reset_state()
        sched.set_storage_format_map(fmt_map_eval)
        graph_eval = graph_kv
        prefill_time, prefill_ser = simulate_prefill(sched, cfg, graph_eval)
        decode_time, decode_ser = simulate_decode_progressive(sched, cfg, graph_eval, prefill_end=prefill_time)
        total_time = float(prefill_time + decode_time)
        wstats = sched.export_weight_stats()
        return (total_time, float(prefill_time), float(decode_time), prefill_ser, decode_ser, wstats, sched)

    def _record(pass_id: int, total: float, prefill_t: float, decode_t: float, prefill_ser: Any, decode_ser: Any, fm: Dict[str, str], wstats: Dict, *, note: str):
        all_pass_records.append({
            'search_format': str(search_start_mode),
            'role': 'search',
            'pass': int(pass_id),
            'note': str(note),
            'times': {'prefill': float(prefill_t), 'decode': float(decode_t), 'total': float(total)},
            'schedules': {'prefill': prefill_ser, 'decode_steps': decode_ser},
            'formats': dict(fm or {}),
            'format_summary': _map_stats(fm),
            'weights': dict(wstats or {}),
            'pim_trace': list(getattr(getattr(cost, 'logger', None), 'pim_trace', []) or []),
        })

    def _inner_sweep(
        map_in: Dict[str, str],
        base_total: float,
        base_prefill: float,
        base_decode: float,
        base_prefill_ser: Any,
        base_decode_ser: Any,
        base_wstats: Dict,
        base_sched: Any,
        *,
        sweep_id: int,
    ) -> Tuple[Dict[str, str], float, float, float, Any, Any, Dict, Any]:
        """Inner sweep: try per-block format flips (NZ->ND->PIM-OPT)."""
        cur_map = dict(map_in or {})
        cur_total = float(base_total)
        cur_prefill = float(base_prefill)
        cur_decode = float(base_decode)
        cur_prefill_ser = base_prefill_ser
        cur_decode_ser = base_decode_ser
        cur_wstats = dict(base_wstats or {})
        cur_sched = base_sched

        wlc = _normalize_wlc(cur_wstats)
        blk_cnt = _block_reload_counts(wlc)

        # Candidate blocks: stored as NZ but used mostly on PIM, or vice versa.
        candidates: List[Tuple[float, float, str]] = []  # (severity, total_cnt, bkey)
        for bkey, (npu, pim) in blk_cnt.items():
            fmt = _current_block_fmt(cur_map, bkey)
            if fmt == 'NZ' and pim > npu:
                candidates.append((float(pim - npu), float(npu + pim), str(bkey)))
            elif fmt == 'PIM-OPT' and npu > pim:
                candidates.append((float(npu - pim), float(npu + pim), str(bkey)))

        # Try the most "wrong" blocks first.
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))

        _debug(
            f"[AL] inner{sweep_id}: start cur_total={cur_total:.6f}s "
            f"candidates={len(candidates)} max_blocks={('inf' if not inner_max_blocks else int(inner_max_blocks))} "
            f"eps={inner_improve_eps:g}"
        )

        if not candidates:
            _debug(f"[AL] inner{sweep_id}: no candidates; skip.")
            return (cur_map, cur_total, cur_prefill, cur_decode, cur_prefill_ser, cur_decode_ser, cur_wstats, cur_sched)

        tried = 0
        accepted_cnt = 0
        for _, _, bkey in candidates:
            if inner_max_blocks and tried >= inner_max_blocks:
                break
            tried += 1

            fmt0 = _current_block_fmt(cur_map, bkey)
            # Two-layer BFS along the line: NZ -> ND -> PIM-OPT (or reverse)
            if fmt0 == 'NZ':
                fmt_chain = ['ND', 'PIM-OPT']
            elif fmt0 == 'PIM-OPT':
                fmt_chain = ['ND', 'NZ']
            else:
                continue

            try:
                _npu_r, _pim_r = blk_cnt.get(str(bkey), (0, 0))
            except Exception:
                _npu_r, _pim_r = 0, 0
            _debug(
                f"[AL] inner{sweep_id}: try#{tried}/{len(candidates)} "
                f"block={bkey} fmt0={fmt0} reload(npu={float(_npu_r):.3f}, pim={float(_pim_r):.3f}) "
                f"chain={fmt_chain} cur_total={cur_total:.6f}s"
            )

            accepted = False
            best_trial: float | None = None
            for fmt1 in fmt_chain:
                cand_map = _apply_block_fmt(cur_map, bkey, fmt1)
                total_time, prefill_time, decode_time, prefill_ser, decode_ser, wstats, sched_eval = _evaluate_map(
                    cand_map,
                    tag=f"inner{sweep_id}_blk_{bkey}_{fmt0}_to_{fmt1}",
                )
                try:
                    best_trial = float(total_time) if best_trial is None else min(float(best_trial), float(total_time))
                except Exception:
                    pass
                if float(total_time) + float(inner_improve_eps) < float(cur_total):
                    old_total = float(cur_total)
                    cur_map = cand_map
                    cur_total = float(total_time)
                    cur_prefill = float(prefill_time)
                    cur_decode = float(decode_time)
                    cur_prefill_ser = prefill_ser
                    cur_decode_ser = decode_ser
                    cur_wstats = dict(wstats or {})
                    cur_sched = sched_eval
                    accepted = True
                    accepted_cnt += 1
                    _debug(
                        f"[AL] inner{sweep_id}: ACCEPT block={bkey} {fmt0}->{fmt1} "
                        f"total {old_total:.6f}s -> {cur_total:.6f}s (delta={cur_total - old_total:+.6f}s)"
                    )
                    break
                else:
                    # Keep the logs light: only show a few early rejects.
                    if tried <= 3:
                        _debug(
                            f"[AL] inner{sweep_id}: reject block={bkey} {fmt0}->{fmt1} "
                            f"trial_total={float(total_time):.6f}s (cur={cur_total:.6f}s)"
                        )
            if accepted:
                # Refresh counts after an accepted change (keeps subsequent tests meaningful).
                wlc = _normalize_wlc(cur_wstats)
                blk_cnt = _block_reload_counts(wlc)
            else:
                if best_trial is not None and tried <= 3:
                    _debug(
                        f"[AL] inner{sweep_id}: best_trial_for_block={float(best_trial):.6f}s (no accept; cur={cur_total:.6f}s)"
                    )

        _debug(
            f"[AL] inner{sweep_id}: done tried={tried} accepted={accepted_cnt} final_total={cur_total:.6f}s"
        )
        return (cur_map, cur_total, cur_prefill, cur_decode, cur_prefill_ser, cur_decode_ser, cur_wstats, cur_sched)

    # -------------------------------
    # ND is the only real search start.
    # -------------------------------
    fmt_map = {}
    _debug(f"[AL][{search_start_mode}] outer0: start (all weights ND)")
    total_time0, prefill_time0, decode_time0, prefill_time0_ser, decode_time0_ser, wst0, sched0 = _evaluate_map(
        fmt_map,
        tag='outer0_all_nd',
    )
    _debug(
        f"[AL][{search_start_mode}] outer0: done total={float(total_time0):.6f}s prefill={float(prefill_time0):.6f}s decode={float(decode_time0):.6f}s"
    )
    _record(0, total_time0, prefill_time0, decode_time0, prefill_time0_ser, decode_time0_ser, fmt_map, wst0, note='outer0_all_nd')

    best_total = float(total_time0)
    best_map = dict(fmt_map)
    best_pass = 0

    prev_outer_total = float(total_time0)
    prev_outer_map = dict(fmt_map)
    prev_wstats = dict(wst0)
    prev_sched = sched0

    # -------------------------------
    # outer iterations: cost-ranked top-K block changes + unchanged inner sweep
    # -------------------------------
    for outer_it in range(1, max(1, outer_max) + 1):
        ranked_updates = _rank_outer_block_updates(
            prev_outer_map,
            prev_wstats,
            prev_sched,
            only_if_current_nd=True,
        )

        if not ranked_updates:
            _debug(
                f"[AL][{search_start_mode}] outer{outer_it}: no outer blocks need modification; stop."
            )
            break

        cand_map, applied_updates = _apply_ranked_outer_block_updates(
            prev_outer_map,
            ranked_updates,
            max_changes=max_outer_block_changes,
        )

        diff_ratio = mapping_diff_ratio(prev_outer_map, cand_map)
        if diff_ratio == 0.0 or not applied_updates:
            _debug(
                f"[AL][{search_start_mode}] outer{outer_it}: ranked candidates exist but no block was applied; stop."
            )
            break

        try:
            keys = set(prev_outer_map.keys()) | set(cand_map.keys())
            changed = sum((1 for k in keys if prev_outer_map.get(k) != cand_map.get(k)))
        except Exception:
            changed = -1

        top0 = applied_updates[0]
        _debug(
            f"[AL][{search_start_mode}] outer{outer_it}: apply blocks={len(applied_updates)}/{max_outer_block_changes} "
            f"changed_weights={changed} diff_ratio={diff_ratio:.3f} prev_total={float(prev_outer_total):.6f}s "
            f"top1={top0.get('block')} {top0.get('cur_fmt')}->{top0.get('next_fmt')} gain={float(top0.get('gain', 0.0)):.6e}"
        )

        total_time_k, prefill_time_k, decode_time_k, prefill_time_k_ser, decode_time_k_ser, wst_k, sched_k = _evaluate_map(
            cand_map,
            tag=f'outer{outer_it}_baseline',
        )
        _debug(
            f"[AL][{search_start_mode}] outer{outer_it}: baseline total={float(total_time_k):.6f}s prefill={float(prefill_time_k):.6f}s decode={float(decode_time_k):.6f}s"
        )
        outer_k_base_total = float(total_time_k)
        outer_k_base_map = dict(cand_map)
        (
            cand_map,
            total_k,
            prefill_time_k,
            decode_time_k,
            prefill_time_k_ser,
            decode_time_k_ser,
            wst_k,
            sched_k,
        ) = _inner_sweep(
            cand_map,
            total_time_k,
            prefill_time_k,
            decode_time_k,
            prefill_time_k_ser,
            decode_time_k_ser,
            wst_k,
            sched_k,
            sweep_id=outer_it,
        )

        # IMPORTANT: use the post-inner-sweep total for decisions/records.
        total_time_k = float(total_k)
        _debug(
            f"[AL][{search_start_mode}] outer{outer_it}: after inner total={float(total_time_k):.6f}s "
            f"(delta={float(total_time_k) - outer_k_base_total:+.6f}s, map_diff={mapping_diff_ratio(outer_k_base_map, cand_map):.3f})"
        )

        # Revert + stop on regression. If we only revert without stopping, the
        # next outer pass would simply re-propose the same top-ranked blocks.
        if float(total_time_k) > float(prev_outer_total) + float(outer_stop_eps):
            _debug(
                f"[AL][{search_start_mode}] outer{outer_it}: total {total_time_k:.6f}s is worse than prev {prev_outer_total:.6f}s; revert and stop."
            )
            break

        # Accept this outer iteration.
        prev_outer_total = float(total_time_k)
        prev_outer_map = dict(cand_map)
        prev_wstats = dict(wst_k)
        prev_sched = sched_k
        _record(
            outer_it,
            total_time_k,
            prefill_time_k,
            decode_time_k,
            prefill_time_k_ser,
            decode_time_k_ser,
            cand_map,
            wst_k,
            note=f'outer{outer_it}_after_inner',
        )

        if best_total is None or float(total_time_k) < float(best_total):
            best_total = float(total_time_k)
            best_map = dict(cand_map)
            best_pass = int(outer_it)

    best_rec: Dict[str, Any]
    improvements: List[Dict[str, Any]] = []
    if all_pass_records:
        best_idx = min(range(len(all_pass_records)), key=lambda i: float(all_pass_records[i]['times']['total']))
        best_rec = dict(all_pass_records[best_idx])
        best_total_rec = float(best_rec['times']['total'])
        best_map = dict(best_rec.get('formats') or {})
        best_pass = int(best_rec.get('pass', best_pass))
        for rec in all_pass_records:
            total_time = float(rec['times']['total'])
            delta = float(total_time - best_total_rec)
            pct = delta / total_time * 100.0 if total_time > 0 else 0.0
            improvements.append({
                'pass': rec.get('pass', -1),
                'total_time': float(total_time),
                'delta_seconds_vs_best': float(delta),
                'delta_percent_vs_that_pass': float(pct),
            })
    else:
        best_total_rec = float(best_total or 0.0)
        best_rec = {
            'search_format': str(search_start_mode),
            'role': 'search',
            'pass': int(best_pass),
            'note': 'outer0_all_nd',
            'times': {'prefill': float(prefill_time0), 'decode': float(decode_time0), 'total': float(best_total_rec)},
            'schedules': {'prefill': prefill_time0_ser, 'decode_steps': decode_time0_ser},
            'formats': dict(best_map or {}),
            'format_summary': _map_stats(best_map),
            'weights': dict(wst0 or {}),
            'pim_trace': list(getattr(getattr(cost, 'logger', None), 'pim_trace', []) or []),
        }
        improvements.append({
            'pass': int(best_pass),
            'total_time': float(best_total_rec),
            'delta_seconds_vs_best': 0.0,
            'delta_percent_vs_that_pass': 0.0,
        })

    # Persist best ND-search map and reports.
    weight_format_path.parent.mkdir(parents=True, exist_ok=True)
    with open(weight_format_path, 'w', encoding='utf-8') as f:
        json.dump(best_map or {}, f, indent=2, sort_keys=True)
    _debug(str(f'[INFO] Best weight storage map (ND search) saved to: {weight_format_path}'))

    full_map = {str(w): str((best_map or {}).get(str(w), 'ND')) for w in all_wids}
    full_path = weight_format_path.with_name(weight_format_path.stem + "_full" + weight_format_path.suffix)
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(full_map, f, indent=2, sort_keys=True)
    _debug(str(f'[INFO] Full weight storage map (ND search) saved to: {full_path}'))

    # Optional control-plane artifact: re-evaluate only the selected best
    # layout so candidate scores correspond to the map that is handed to
    # Het-Infer. Existing AE summaries/timelines remain unchanged.
    hetinfer_prior_path = None
    requested_prior_out = cfg.get('hetinfer_prior_out')
    if requested_prior_out not in (None, ''):
        if algo_name != 'Bifocal':
            raise ValueError('--hetinfer-prior-out in weight-suggest requires algo=Bifocal')
        (
            _,
            _,
            _,
            _,
            _,
            _,
            prior_sched,
        ) = _evaluate_map(dict(best_map or {}), tag='hetinfer_prior_best_layout')
        exporter = getattr(prior_sched, 'export_hetinfer_candidate_records', None)
        if not callable(exporter):
            raise RuntimeError('Bifocal scheduler does not expose Het-Infer candidate records')
        candidate_records = list(exporter() or [])
        if not candidate_records:
            raise RuntimeError('Bifocal produced no exact candidate records for Het-Infer export')
        output_path = _resolve_hetinfer_prior_output(
            str(requested_prior_out),
            result_dir=str(result_dir),
            tag=_build_tag(cfg),
        )
        if output_path is None:
            raise RuntimeError('failed to resolve Het-Infer prior output path')
        prior_cfg = dict(cfg)
        prior_cfg['algo'] = algo_name
        artifact = build_hetinfer_prior_artifact(
            cfg=prior_cfg,
            graph=graph_kv,
            cluster=cluster,
            shape=shape,
            candidate_records=candidate_records,
        )
        hetinfer_prior_path = str(
            write_hetinfer_prior_artifact(
                artifact,
                output_path,
                candidate_records=candidate_records,
            )
        )
        _debug(f'[Het-Infer] Saved best-layout placement prior to: {hetinfer_prior_path}')

    # ------------------------------------------------------------
    # 2: fixed baseline experiments
    # ------------------------------------------------------------
    nd_initial_rec = dict(all_pass_records[0]) if all_pass_records else dict(best_rec)
    nd_initial_total = float((nd_initial_rec.get('times') or {}).get('total', 0.0) or 0.0)
    compare_rows: List[Dict[str, Any]] = [
        {
            'experiment_id': 'search_nd_tuned',
            'display_name': f"algo:{_display_policy_name(algo_name or 'HEFT')} + search(best map)",
            'format': 'ND',
            'algo': str(algo_name or 'HEFT'),
            'storage_mode': 'AL-search(best map)',
            'storage_format': 'MIXED',
            'role': 'search',
            'search_executed': True,
            'comparison_only': False,
            'initial_times': dict(nd_initial_rec.get('times') or {}),
            'initial_total_s': float(nd_initial_total),
            'best_total_s': float(best_total_rec),
            'delta_vs_initial_s': float(best_total_rec - nd_initial_total),
            'delta_vs_initial_pct': float(((best_total_rec - nd_initial_total) / nd_initial_total * 100.0) if nd_initial_total > 0 else 0.0),
            'best_pass': int(best_rec.get('pass', best_pass)),
            'best_format_summary': dict(best_rec.get('format_summary') or _map_stats(best_map)),
            'best_weight_format_json': str(weight_format_path),
            'best_weight_format_full_json': str(full_path),
        }
    ]

    baseline_experiment_ids = [str(spec.get('experiment_id', '')) for spec in fixed_baseline_experiments]
    for spec in fixed_baseline_experiments:
        exp_id = str(spec.get('experiment_id', '') or '')
        runner = str(spec.get('runner', 'strategy') or 'strategy').strip().lower()
        algo_for_row = str(spec.get('algo', '') or '')
        storage_fmt = _normalize_weight_storage_fmt(spec.get('storage_fmt', 'ND'))
        storage_mode_name = _storage_mode_display_name(storage_fmt)

        _debug(
            f"[BASELINE][{exp_id}] start algo={algo_for_row} storage={storage_mode_name}"
        )
        comparison_cfg = _comparison_cfg_without_prior_export(cfg)

        if runner == 'baseline':
            result = _eval_one_baseline(
                comparison_cfg,
                algo_for_row,
                shared_graph=graph,
                shared_shape=shape,
                uniform_weight_storage_fmt=storage_fmt,
                artifact_tag=exp_id,
            )
        else:
            result = _run_strategy_once(
                algo_for_row,
                comparison_cfg,
                shared_graph=graph,
                shared_shape=shape,
                uniform_weight_storage_fmt=storage_fmt,
                artifact_tag=exp_id,
            )

        init_times = {
            'prefill': float(result.get('prefill_time_s', 0.0) or 0.0),
            'decode': float(result.get('decode_time_s', 0.0) or 0.0),
            'total': float(result.get('total_time_s', 0.0) or 0.0),
        }
        _debug(
            f"[BASELINE][{exp_id}] done total={float(init_times['total']):.6f}s "
            f"prefill={float(init_times['prefill']):.6f}s decode={float(init_times['decode']):.6f}s"
        )
        compare_rows.append({
            'experiment_id': exp_id,
            'display_name': str(spec.get('display_name', exp_id) or exp_id),
            'format': str(storage_fmt),
            'algo': str(algo_for_row),
            'storage_mode': str(result.get('weight_storage_mode', storage_mode_name) or storage_mode_name),
            'storage_format': str(result.get('weight_storage_format', storage_fmt) or storage_fmt),
            'role': 'baseline',
            'search_executed': False,
            'comparison_only': False,
            'initial_times': dict(init_times),
            'initial_total_s': float(init_times['total']),
            'best_total_s': float(init_times['total']),
            'delta_vs_initial_s': 0.0,
            'delta_vs_initial_pct': 0.0,
            'best_pass': 0,
            'best_format_summary': dict(result.get('weight_storage_map_summary') or _map_stats(_build_uniform_weight_storage_map(graph, storage_fmt))),
            'pim_strategy': result.get('pim_strategy'),
            'pim_strategy_scores': result.get('pim_strategy_scores'),
        })

    comparison_payload = {
        'config': {
            'model_family': cfg.get('model_family'),
            'model_variant': cfg.get('model_variant'),
            'dtype': cfg.get('dtype'),
            'batch': cfg.get('batch'),
            'prefill_len': cfg.get('prefill_len'),
            'decode_len': cfg.get('decode_len'),
            'search_format': str(search_start_mode),
            'compare_only_formats': [],
            'baseline_experiment_ids': list(baseline_experiment_ids),
            'format_block_change_percent': float(block_change_percent),
            'format_outer_max_iters': int(outer_max),
            'format_inner_max_blocks': cfg.get('format_inner_max_blocks', 0),
            'format_nd_margin_init': cfg.get('format_nd_margin_init', 0.60),
            'format_nd_margin_decay': cfg.get('format_nd_margin_decay', 0.85),
            'format_nd_margin_min': cfg.get('format_nd_margin_min', 0.05),
            'format_inner_improve_eps': cfg.get('format_inner_improve_eps', 1e-6),
            'format_outer_stop_eps': cfg.get('format_outer_stop_eps', 0.0),
            'pim_weight_load_overlap_ratio': cfg.get('pim_weight_load_overlap_ratio', None),
            'weight_load_compute_overlap_ratio': cfg.get('weight_load_compute_overlap_ratio', None),
        },
        'rows': compare_rows,
        'search_format': str(search_start_mode),
        'compare_only_formats': [],
        'baseline_experiment_ids': list(baseline_experiment_ids),
        'best_pass': int(best_rec.get('pass', best_pass)),
        'best_total_s': float(best_total_rec),
        'best_weight_format_json': str(weight_format_path),
        'best_weight_format_full_json': str(full_path),
    }
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    with open(compare_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_payload, f, ensure_ascii=False, indent=2)
    _debug(str(f'[REPORT] Weight-format comparison saved to: {compare_path}'))

    all_path = Path(cfg.get('all_passes_json', ALL_PASSES_RESULT_PATH))
    best_path = Path(cfg.get('best_summary_json', BEST_PASS_SUMMARY_PATH))
    all_path.parent.mkdir(parents=True, exist_ok=True)
    with open(all_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'config': {
                    'model_family': cfg.get('model_family'),
                    'model_variant': cfg.get('model_variant'),
                    'dtype': cfg.get('dtype'),
                    'batch': cfg.get('batch'),
                    'prefill_len': cfg.get('prefill_len'),
                    'decode_len': cfg.get('decode_len'),
                    'search_format': str(search_start_mode),
                    'compare_only_formats': [],
                    'baseline_experiment_ids': list(baseline_experiment_ids),
                    'pim_weight_load_overlap_ratio': cfg.get('pim_weight_load_overlap_ratio', None),
                    'weight_load_compute_overlap_ratio': cfg.get('weight_load_compute_overlap_ratio', None),
                },
                'weight_format_comparison': comparison_payload,
                'passes': all_pass_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    _debug(str(f'[REPORT] All passes (ND search only) saved to: {all_path}'))

    best_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'config': _best_summary_config_snapshot(cfg),
                'search_format': str(search_start_mode),
                'best_pass': int(best_rec.get('pass', best_pass)),
                'best_times': best_rec.get('times', {}),
                'best_formats': best_rec.get('formats', {}),
                'best_format_summary': best_rec.get('format_summary', {}),
                'best_weights': best_rec.get('weights', {}),
                'prefill_schedule': best_rec.get('schedules', {}).get('prefill'),
                'decode_steps': best_rec.get('schedules', {}).get('decode_steps'),
                'improvements_vs_each_pass': improvements,
                'weight_format_comparison': compare_rows,
                'baseline_experiment_ids': list(baseline_experiment_ids),
                'pim_weight_load_overlap_ratio': cfg.get('pim_weight_load_overlap_ratio', None),
                'weight_load_compute_overlap_ratio': cfg.get('weight_load_compute_overlap_ratio', None),
                'best_weight_format_json': str(weight_format_path),
                'best_weight_format_full_json': str(full_path),
                'weight_format_compare_json': str(compare_path),
                'hetinfer_prior_path': hetinfer_prior_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    _debug(str(f'[REPORT] Best pass summary (ND search only) saved to: {best_path}'))

    print("\n=== Weight-Suggest Format Comparison ===")
    header = (
        f"{'Experiment':<40} {'Role':<10} {'Outer0(s)':>12} {'Best(s)':>12} {'Delta(s)':>12} "
        f"{'BestPass':>9} {'ND':>8} {'NZ':>8} {'PIM':>8} {'DUAL':>8}"
    )
    print(header)
    print('-' * len(header))
    for row in compare_rows:
        counts = dict((row.get('best_format_summary') or {}).get('counts', {}) or {})
        print(
            f"{str(row.get('display_name', row.get('experiment_id', row.get('format', '')))):<40} "
            f"{str(row.get('role', '')):<10} "
            f"{float(row.get('initial_total_s', 0.0)):>12.4f} "
            f"{float(row.get('best_total_s', 0.0)):>12.4f} "
            f"{float(row.get('delta_vs_initial_s', 0.0)):>12.4f} "
            f"{int(row.get('best_pass', -1)):>9d} "
            f"{int(counts.get('ND', 0)):>8d} "
            f"{int(counts.get('NZ', 0)):>8d} "
            f"{int(counts.get('PIM-OPT', 0)):>8d} "
            f"{int(counts.get('DUAL', 0)):>8d}"
        )
    print(
        f"[weight-suggest] search_format={str(search_start_mode)} "
        f"pim_weight_load_overlap_ratio={cfg.get('pim_weight_load_overlap_ratio', None)} "
        f"weight_load_compute_overlap_ratio={cfg.get('weight_load_compute_overlap_ratio', None)} "
        f"best_total={float(best_total_rec):.6f}s"
    )

    # AL mode terminates here (skip legacy multi-pass loop below).
    try:
        cost.logger.end_simulation()
        cost.logger.close()
    except Exception:
        pass

    return
