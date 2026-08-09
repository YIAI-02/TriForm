"""Strategy and baseline evaluation helpers shared by CLI commands."""

from __future__ import annotations

from .shared import *
from .log_utils import _debug
from .storage import (
    _artifact_tag_token,
    _best_summary_config_snapshot,
    _build_uniform_weight_storage_map,
    _collect_weight_ids_from_graph,
    _normalize_weight_storage_fmt,
    _resolve_hetinfer_prior_output,
    _storage_mode_display_name,
    _weight_map_summary,
)
from hetinfer_prior import build_artifact as build_hetinfer_prior_artifact
from hetinfer_prior import write_artifact as write_hetinfer_prior_artifact
from .graph_utils import _clone_graph, _fallback_npu_to_cpu_if_needed, _fallback_pim_to_cpu_if_needed
from .baselines import PD_BASELINES, _BASELINE_REGISTRY, _apply_policy_on_graph
from .kv_policy import (
    _apply_kv_place_constraints,
    _build_cost_model_for_run,
    _fmt_kv_policy_scores,
    _infer_kv_place_from_label,
    auto_select_kv_policy,
)
from .simulator import _make_scheduler, simulate_decode_progressive, simulate_prefill

def _eval_one_baseline(
    cfg: Dict,
    policy: str,
    *,
    shared_graph: TaskGraph | None = None,
    shared_shape: Any = None,
    uniform_weight_storage_fmt: str | None = None,
    artifact_tag: str | None = None,
) -> Dict:
    reset_simulation_logger()

    cluster = demo_cluster(cfg)
    cfg['topology'] = getattr(cluster, 'topology', cfg.get('topology', None))
    if shared_graph is not None and shared_shape is not None:
        graph, shape = shared_graph, shared_shape
    else:
        graph, shape = build_graph(cfg)

    policy_token = _normalize_baseline_name(policy)
    policy_name = _display_policy_name(policy_token)

    batch = int(cfg.get("batch", 1))
    prefill_len = int(cfg.get("prefill_len", 128))
    decode_len = int(cfg.get("decode_len", 32))

    base_dir = Path(cfg["result_dir"])
    tag_tok = _artifact_tag_token(artifact_tag)
    algo_dir_name = _policy_dir_name(policy_token) + (f"__{tag_tok}" if tag_tok else "")
    algo_dir = base_dir / algo_dir_name
    algo_dir.mkdir(parents=True, exist_ok=True)

    if tag_tok:
        sim_log_path = algo_dir / f"pim_simulation_{tag_tok}.txt"
    else:
        sim_log_path = Path(cfg.get(
            "simulation_log_file",
            algo_dir / "pim_simulation.txt",
        ))

    cost = _build_cost_model_for_run(
        cfg,
        cluster,
        shape,
        simulation_log_file=sim_log_path,
    )

    try:
        cost.logger.start_simulation()
    except Exception:
        pass

    # Build separate prefill and decode graphs for the baseline policy.
    pol = policy_token
    if pol in _BASELINE_REGISTRY:
        g_prefill = _BASELINE_REGISTRY[pol](graph, phase="prefill")
        g_decode = _BASELINE_REGISTRY[pol](graph, phase="decode")
    else:
        g_prefill = _apply_policy_on_graph(graph, policy, phase="prefill")
        g_decode = _apply_policy_on_graph(graph, policy, phase="decode")

    _fallback_npu_to_cpu_if_needed(g_prefill, cluster)
    _fallback_npu_to_cpu_if_needed(g_decode, cluster)
    _fallback_pim_to_cpu_if_needed(g_prefill, cluster)
    _fallback_pim_to_cpu_if_needed(g_decode, cluster)

    uses_partitioned_decode = pol in PD_BASELINES

    best: Dict[str, Any] | None = None
    best_label = None
    best_prefill_ser = None
    best_decode_ser = None
    best_sched = None

    label = auto_select_kv_policy(
        strategy="Naive",
        cfg=cfg,
        cluster=cluster,
        cost=cost,
        graph=g_prefill,
        graph_decode=g_decode,
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
    g_prefill = _apply_kv_place_constraints(g_prefill, kv_place)
    g_decode = _apply_kv_place_constraints(g_decode, kv_place)

    sched = _make_scheduler("Naive", cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=GlobalMemoryManager())
    weight_fmt_map = _build_uniform_weight_storage_map(graph, uniform_weight_storage_fmt)
    sched.set_storage_format_map(weight_fmt_map)
    t_prefill, prefill_ser = simulate_prefill(sched, cfg, g_prefill)

    # The PD baseline includes a one-time KV migration cost from host to PIM.
    t_kv_move = 0.0
    if uses_partitioned_decode and label.kv_in_pim and label.kv_total_bytes > 0:
        host = cluster.devices_by_type("cpu")[0]
        pim_list = cluster.devices_by_type("pim")
        if pim_list:
            per = label.kv_total_bytes // max(1, len(pim_list))
            for d in pim_list:
                t_kv_move = max(t_kv_move, cost.comm_cost(host, d, per))

    t_decode, decode_ser = simulate_decode_progressive(
        sched, cfg, g_decode, prefill_end=t_prefill
    )

    # decode_time_effective = float(t_decode + (t_kv_move if uses_partitioned_decode else 0.0))
    decode_time_effective = float(t_decode)
    total_time = float(t_prefill + decode_time_effective)

    best = {
        "prefill_time_s": float(t_prefill),
        "decode_time_s": decode_time_effective,
        "total_time_s": total_time,
        "kv_in_pim": bool(getattr(label, "kv_in_pim", False)),
        "kv_total_bytes": int(getattr(label, "kv_total_bytes", 0) or 0),
        "pim_weight_capacity_bytes": int(getattr(label, "pim_weight_capacity_bytes", 0) or 0),
    }
    best_label = label
    best_prefill_ser = prefill_ser
    best_decode_ser = decode_ser
    best_sched = sched

    try:
        if best_sched is not None and hasattr(best_sched, "stats"):
            mode_tok = _artifact_tag_token(_storage_mode_display_name(uniform_weight_storage_fmt))
            prefix = f"{policy_name}" + (f"_{mode_tok}" if mode_tok else "") + f"_prefill-{prefill_len}xdecode_{decode_len}"
            trace_ops = algo_dir / f"{prefix}_ops_trace.csv"
            trace_comms = algo_dir / f"{prefix}_comms_trace.csv"
            trace_ops.parent.mkdir(parents=True, exist_ok=True)
            best_sched.stats.dump_trace_csv(
                trace_ops,
                trace_comms,
            )
            # Record trace paths on the plan label for downstream scripts.
            try:
                setattr(best_label, 'trace_ops_csv', str(trace_ops))
                setattr(best_label, 'trace_comms_csv', str(trace_comms))
            except Exception:
                pass
    except Exception:
        pass

    try:
        cost.logger.end_simulation()
        cost.logger.close()
    except Exception:
        pass

    pim_trace = None
    try:
        if best_sched is not None and hasattr(best_sched, "pim_trace"):
            pim_trace = list(getattr(best_sched, "pim_trace") or [])
    except Exception:
        pim_trace = None

    return {
        "policy": _policy_label(policy_token),
        "pim_strategy": getattr(best_label, 'kv_policy_selected', None),
        "pim_strategy_scores": getattr(best_label, 'kv_policy_scores', None),
        "prefill_time_s": best["prefill_time_s"],
        "decode_time_s": best["decode_time_s"],
        "total_time_s": best["total_time_s"],
        "batch": batch,
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "prefill_schedule": best_prefill_ser,
        "decode_steps": best_decode_ser,
        "pim_trace": pim_trace,
        "kv_in_pim": best.get("kv_in_pim", False),
        "kv_total_bytes": best.get("kv_total_bytes", 0),
        "pim_weight_capacity_bytes": best.get("pim_weight_capacity_bytes", 0),
        "weight_storage_mode": _storage_mode_display_name(uniform_weight_storage_fmt),
        "weight_storage_format": _normalize_weight_storage_fmt(uniform_weight_storage_fmt or 'ND'),
        "weight_storage_map_summary": _weight_map_summary(_collect_weight_ids_from_graph(graph), weight_fmt_map),
        "label": best_label,
    }

def _run_strategy_once(
    strategy: str,
    cfg: Dict,
    *,
    shared_graph: TaskGraph | None = None,
    shared_shape: Any = None,
    uniform_weight_storage_fmt: str | None = None,
    artifact_tag: str | None = None,
) -> Dict:
    cluster = demo_cluster(cfg)
    cfg['topology'] = getattr(cluster, 'topology', cfg.get('topology', None))
    if shared_graph is not None and shared_shape is not None:
        graph, shape = _clone_graph(shared_graph), shared_shape
    else:
        graph, shape = build_graph(cfg)

    strategy_token = _normalize_algo_name(strategy)
    strategy_name = _display_policy_name(strategy_token)

    # If there is no NPU in the hardware topology, fall back NPU ops to CPU.
    _fallback_npu_to_cpu_if_needed(graph, cluster)

    batch = int(cfg.get("batch", 1))
    prefill_len = int(cfg.get("prefill_len", 128))
    decode_len = int(cfg.get("decode_len", 32))

    reset_simulation_logger()

    tag_tok = _artifact_tag_token(artifact_tag)
    if tag_tok:
        result_dir = Path(cfg.get("result_dir", "./output/strategy_results"))
        result_dir.mkdir(parents=True, exist_ok=True)
        sim_log_path = result_dir / f"pim_simulation_{str(strategy_name)}_{tag_tok}.txt"
    else:
        sim_log_path = Path(cfg.get(
            "simulation_log_file",
            "./output/pim_simulation.txt",
        ))

    cost = _build_cost_model_for_run(
        cfg,
        cluster,
        shape,
        simulation_log_file=sim_log_path,
    )

    try:
        cost.logger.start_simulation()
    except Exception:
        pass

    best: Dict[str, Any] | None = None
    best_sched = None
    best_prefill_ser = None
    best_decode_ser = None
    best_label = None

    label = auto_select_kv_policy(
        strategy=strategy_token,
        cfg=cfg,
        cluster=cluster,
        cost=cost,
        graph=graph,
        shape=shape,
        capture_best_schedule=True,
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

    kv_in_pim = bool(getattr(label, "kv_in_pim", False))
    kv_total_bytes = int(getattr(label, "kv_total_bytes", 0) or 0)
    kv_weight_cap = int(getattr(label, "pim_weight_capacity_bytes", 0) or 0)
    sim_best = getattr(label, "_kv_policy_best_sim", None)
    weight_fmt_map = _build_uniform_weight_storage_map(graph, uniform_weight_storage_fmt)
    if not weight_fmt_map and isinstance(sim_best, dict) and sim_best.get("sched") is not None:
        sched = sim_best.get("sched")
        t_prefill = float(sim_best.get("prefill_s", 0.0) or 0.0)
        t_decode = float(sim_best.get("decode_s", 0.0) or 0.0)
        prefill_ser = sim_best.get("prefill_schedule")
        decode_ser = sim_best.get("decode_steps")
        total_time = float(sim_best.get("total_s", t_prefill + t_decode) or (t_prefill + t_decode))
    else:
        buffer_mgr = GlobalMemoryManager()
        sched = _make_scheduler(
            strategy_token,
            cluster,
            cost,
            label,
            batch=batch,
            seq_len=prefill_len,
            buffer=buffer_mgr,
            rand_seed=cfg.get("scheduler_seed"),
        )
        if cfg.get('hetinfer_prior_out') not in (None, ''):
            enable_capture = getattr(sched, 'enable_hetinfer_candidate_capture', None)
            if callable(enable_capture):
                enable_capture(True)
        sched.reset_state()
        sched.set_storage_format_map(weight_fmt_map)

        t_prefill, prefill_ser = simulate_prefill(sched, cfg, graph_kv)
        t_decode, decode_ser = simulate_decode_progressive(
            sched, cfg, graph_kv, prefill_end=t_prefill
        )
        total_time = float(t_prefill + t_decode)

    best = {
        "prefill_time_s": float(t_prefill),
        "decode_time_s": float(t_decode),
        "total_time_s": total_time,
        "kv_in_pim": kv_in_pim,
        "kv_total_bytes": kv_total_bytes,
        "pim_weight_capacity_bytes": kv_weight_cap,
    }
    best_sched = sched
    best_prefill_ser = prefill_ser
    best_decode_ser = decode_ser
    best_label = label

    try:
        if best_sched is not None and hasattr(best_sched, "stats"):
            mode_tok = _artifact_tag_token(_storage_mode_display_name(uniform_weight_storage_fmt))
            prefix = f"{strategy_name}" + (f"_{mode_tok}" if mode_tok else "") + f"_prefill-{prefill_len}xdecode_{decode_len}"
            result_dir = Path(cfg.get("result_dir", "./output/strategy_results"))
            result_dir.mkdir(parents=True, exist_ok=True)
            trace_ops = result_dir / f"{prefix}_ops_trace.csv"
            trace_comms = result_dir / f"{prefix}_comms_trace.csv"
            best_sched.stats.dump_trace_csv(
                trace_ops,
                trace_comms,
            )
            # Record trace paths on the plan label for downstream scripts.
            try:
                setattr(best_label, 'trace_ops_csv', str(trace_ops))
                setattr(best_label, 'trace_comms_csv', str(trace_comms))
            except Exception:
                pass

    except Exception:
        pass

    try:
        cost.logger.end_simulation()
        cost.logger.close()
    except Exception:
        pass

    pim_trace = None
    try:
        if best_sched is not None and hasattr(best_sched, "pim_trace"):
            pim_trace = list(getattr(best_sched, "pim_trace") or [])
    except Exception:
        pim_trace = None

    hetinfer_prior_path = None
    requested_prior_out = cfg.get("hetinfer_prior_out")
    if requested_prior_out not in (None, "") and strategy_token == "Bifocal":
        exporter = getattr(best_sched, "export_hetinfer_candidate_records", None)
        if not callable(exporter):
            raise RuntimeError("Bifocal scheduler does not expose Het-Infer candidate records")
        candidate_records = list(exporter() or [])
        if not candidate_records:
            raise RuntimeError("Bifocal produced no exact candidate records for Het-Infer export")
        output_path = _resolve_hetinfer_prior_output(
            str(requested_prior_out),
            result_dir=str(cfg.get("result_dir", "./output")),
            tag=f"{int(cfg.get('prefill_len', 0) or 0)}x{int(cfg.get('decode_len', 0) or 0)}",
        )
        if output_path is None:
            raise RuntimeError("failed to resolve Het-Infer prior output path")
        prior_cfg = dict(cfg)
        # ``cfg['algo']`` may be a multi-algorithm list. Provenance must name
        # the concrete scheduler whose candidate scores are in this file.
        prior_cfg["algo"] = strategy_token
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
        _debug(f"[Het-Infer] Saved versioned placement prior to: {hetinfer_prior_path}")

    return {
        "policy": _policy_label(strategy_token),
        "strategy": strategy_name,
        "pim_strategy": getattr(best_label, 'kv_policy_selected', None),
        "pim_strategy_scores": getattr(best_label, 'kv_policy_scores', None),
        "prefill_time_s": best["prefill_time_s"],
        "decode_time_s": best["decode_time_s"],
        "total_time_s": best["total_time_s"],
        "batch": batch,
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "prefill_schedule": best_prefill_ser,
        "decode_steps": best_decode_ser,
        "pim_trace": pim_trace,
        "kv_in_pim": best.get("kv_in_pim", False),
        "kv_total_bytes": best.get("kv_total_bytes", 0),
        "pim_weight_capacity_bytes": best.get("pim_weight_capacity_bytes", 0),
        "weight_storage_mode": _storage_mode_display_name(uniform_weight_storage_fmt),
        "weight_storage_format": _normalize_weight_storage_fmt(uniform_weight_storage_fmt or 'ND'),
        "weight_storage_map_summary": _weight_map_summary(_collect_weight_ids_from_graph(graph), weight_fmt_map),
        "label": best_label,
        "hetinfer_prior_path": hetinfer_prior_path,
    }

def _ensure_dir(p:Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p

def _label_summary(label: PlanLabel | None) -> Dict[str, Any]:
    if label is None:
        return {}
    out = {
        'kv_place': str(getattr(label, 'kv_place', 'pim' if bool(getattr(label, 'kv_in_pim', False)) else 'host')),
        'kv_in_npu': bool(getattr(label, 'kv_in_npu', False)),
        'kv_in_pim': bool(getattr(label, 'kv_in_pim', False)),
        'kv_total_bytes': int(getattr(label, 'kv_total_bytes', 0) or 0),
        'kv_total_bytes_all': int(getattr(label, 'kv_total_bytes_all', getattr(label, 'kv_total_bytes_raw', 0)) or 0),
        'pim_weight_capacity_bytes': int(getattr(label, 'pim_weight_capacity_bytes', 0) or 0),
        'pinned_fc_on_pim': sorted(list(getattr(label, 'pinned_fc_on_pim', set()) or [])),
    }

    # Optional: record trace/stat artifact locations if the caller populated them.
    try:
        for attr in (
            'trace_ops_csv',
            'trace_comms_csv',
        ):
            val = getattr(label, attr, None)
            if val:
                out[attr] = str(val)
    except Exception:
        pass

    return out

def _save_best_json(algo_dir: Path, tag: str, policy: str, *, times: Dict, prefill_schedule=None, decode_steps=None, cfg: Dict|None=None, label: PlanLabel | None = None):
    payload = {
        'policy': policy,
        'pim_strategy': times.get('pim_strategy', 'unknown'),
        'config': _best_summary_config_snapshot(cfg),
        'best_times': {'prefill': float(times.get('prefill_time_s', 0.0)), 'decode': float(times.get('decode_time_s', 0.0)), 'total': float(times.get('total_time_s', 0.0))},
    }

    label_dict = _label_summary(label) or _label_summary(times.get('label'))
    if label_dict:
        payload['plan_label'] = label_dict

    if prefill_schedule is not None:
        payload['prefill_schedule'] = prefill_schedule
    if decode_steps is not None:
        payload['decode_steps'] = decode_steps
    pim_trace = times.get('pim_trace')
    if pim_trace is not None:
        payload['pim_trace'] = pim_trace

    # Also record the KV policy comparison numbers if present.
    if 'pim_strategy_scores' in times:
        payload['pim_strategy_scores'] = times.get('pim_strategy_scores')
    if times.get('hetinfer_prior_path'):
        payload['hetinfer_prior_path'] = str(times.get('hetinfer_prior_path'))
    path = algo_dir / f"best_summary_{tag}.json"
    with open(path,'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _debug(f"[Strategy] Saved best summary to: {path}")
    return path


def evaluate_suite(cfg: Dict, *, algos: List[str], baselines: List[str], result_dir: str | None, debug: bool, combined_out: str):
    base_dir = _ensure_dir(Path(result_dir or './output/len_sweep'))
    tag = f"{int(cfg.get('prefill_len', 0))}x{int(cfg.get('decode_len', 0))}"
    results: List[Dict] = []

    cluster = demo_cluster(cfg)
    cfg['topology'] = getattr(cluster, 'topology', cfg.get('topology', None))
    shared_graph, shared_shape = build_graph(cfg)

    blist: List[str] = []
    for item in baselines:
        token = _normalize_baseline_name(item)
        if token and token not in blist:
            blist.append(token)

    for b in blist:
        algo_dir = _ensure_dir(base_dir / _policy_dir_name(b))
        try:
            setup_logging(bool(cfg.get('debug', False)), log_file=str(algo_dir / f"debug_{tag}.txt"))
        except Exception:
            logger.error(f"Failed to setup logging for baseline '{b}'")
        cfg_b = dict(cfg)
        cfg_b['result_dir'] = str(base_dir)
        cfg_b['simulation_log_file'] = str(algo_dir / f"pim_sim_{tag}.txt")
        r = _eval_one_baseline(
            cfg_b,
            b,
            shared_graph=shared_graph,
            shared_shape=shared_shape,
        )
        _save_best_json(
            algo_dir,
            tag,
            policy=_policy_label(b),
            times=r,
            cfg=cfg_b,
            prefill_schedule=r.get('prefill_schedule'),
            decode_steps=r.get('decode_steps'),
            label=r.get('label'),
        )
        results.append({
            'policy': _policy_label(b),
            'pim_strategy': r.get('pim_strategy'),
            'kv_in_pim': bool(r.get('kv_in_pim', False)),
            'kv_total_bytes': int(r.get('kv_total_bytes', 0) or 0),
            'pim_weight_capacity_bytes': int(r.get('pim_weight_capacity_bytes', 0) or 0),
            **{k: r[k] for k in ('prefill_time_s', 'decode_time_s', 'total_time_s')},
        })

    alist: List[str] = []
    for item in algos:
        token = _normalize_algo_name(item)
        if token and token not in alist:
            alist.append(token)

    if cfg.get('hetinfer_prior_out') not in (None, '') and 'Bifocal' not in alist:
        raise ValueError('--hetinfer-prior-out requires Bifocal in the evaluate algorithm list')

    for a in alist:
        algo_dir = _ensure_dir(base_dir / _policy_dir_name(a))
        try:
            setup_logging(bool(cfg.get('debug', False)), log_file=str(algo_dir / f"debug_{tag}.txt"))
        except Exception:
            logger.error(f"Failed to setup logging for algorithm '{a}'")
        cfg_a = dict(cfg)
        cfg_a['simulation_log_file'] = str(algo_dir / f"pim_sim_{tag}.txt")
        cfg_a['result_dir'] = str(algo_dir)
        res = _run_strategy_once(a, cfg_a, shared_graph=shared_graph, shared_shape=shared_shape)
        _save_best_json(
            algo_dir,
            tag,
            policy=res.get('policy', _policy_label(a)),
            times=res,
            prefill_schedule=res.get('prefill_schedule'),
            decode_steps=res.get('decode_steps'),
            cfg=cfg_a,
            label=res.get('label'),
        )
        results.append({
            'policy': _policy_label(a),
            'pim_strategy': res.get('pim_strategy'),
            'pim_strategy_scores': res.get('pim_strategy_scores'),
            'kv_in_pim': bool(res.get('kv_in_pim', False)),
            'kv_total_bytes': int(res.get('kv_total_bytes', 0) or 0),
            'pim_weight_capacity_bytes': int(res.get('pim_weight_capacity_bytes', 0) or 0),
            'hetinfer_prior_path': res.get('hetinfer_prior_path'),
            **{k: res[k] for k in ('prefill_time_s', 'decode_time_s', 'total_time_s')},
        })

    if results:
        os.makedirs(os.path.dirname(combined_out), exist_ok=True)
        with open(combined_out, 'w', encoding='utf-8') as f:
            json.dump({'config': cfg, 'results': results}, f, ensure_ascii=False, indent=2)
        print(f"[REPORT] Combined comparison saved to: {combined_out}")

    print("\n=== Strategy/Baseline Comparison ===")
    header = f"{'Policy':<24} {'PIM':<8} {'Prefill(s)':>12} {'Decode(s)':>12} {'Total(s)':>12}"
    print(header)
    print('-' * len(header))
    for r in results:
        pim_s = str(r.get('pim_strategy') or '')
        print(f"{r['policy']:<24} {pim_s:<8} {r['prefill_time_s']:>12.4f} {r['decode_time_s']:>12.4f} {r['total_time_s']:>12.4f}")
