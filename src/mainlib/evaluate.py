"""Strategy and baseline evaluation helpers shared by CLI commands."""

from __future__ import annotations

from .shared import *
from .log_utils import _debug
from .storage import (
    _artifact_tag_token,
    _build_uniform_weight_storage_map,
    _collect_weight_ids_from_graph,
    _normalize_weight_storage_fmt,
    _storage_mode_display_name,
    _weight_map_summary,
)
from .graph_utils import _clone_graph, _fallback_npu_to_cpu_if_needed, _fallback_pim_to_cpu_if_needed
from .baselines import (
    PD_BASELINES,
    _BASELINE_REGISTRY,
    _FC_OPS,
    _PAPI_ATTN_OPS,
    _PAISE_ATTN_OPS,
    _apply_policy_on_graph,
    _op_name,
)
from .kv_policy import (
    _apply_kv_place_constraints,
    _build_cost_model_for_run,
    _fmt_kv_policy_scores,
    _infer_kv_place_from_label,
    auto_select_kv_policy,
)
from .simulator import _make_scheduler, simulate_decode_progressive, simulate_prefill


def _invoke_baseline_policy(fn: Callable[..., TaskGraph], graph: TaskGraph, *, phase: str, cfg: Dict, cost: CostModel, cluster: Cluster, shape: Any) -> TaskGraph:
    """Invoke a baseline policy with optional context.

    Older baseline functions accept only ``(graph, phase=...)``.  Newer
    cost-model-aware baselines, such as PAPI/PAISE-inspired, can additionally
    consume cfg/cost/cluster/shape without changing the public registry API.
    """
    try:
        return fn(graph, phase=phase, cfg=cfg, cost=cost, cluster=cluster, shape=shape)
    except TypeError as exc:
        msg = str(exc)
        if 'unexpected keyword argument' not in msg and 'positional' not in msg:
            raise
        return fn(graph, phase=phase)

_MAX_RUNTIME_PROBE_NODES_PER_OP = 2
_RUNTIME_PAPI_TLP = 1  # this simulator does not model speculative decoding

def _runtime_probe_baselines_requested(baselines: List[str] | Tuple[str, ...] | None, policy: str | None = None) -> bool:
    toks: set[str] = set()
    for item in baselines or []:
        try:
            toks.add(_normalize_baseline_name(item))
        except Exception:
            pass
    if policy is not None:
        try:
            toks.add(_normalize_baseline_name(policy))
        except Exception:
            pass
    return bool({'PAPI-inspired', 'PAISE-inspired'} & toks)


def _runtime_probe_nodes(graph: TaskGraph, op_names: set[str], *, max_per_op: int = _MAX_RUNTIME_PROBE_NODES_PER_OP) -> List[TaskNode]:
    wanted = {str(x).upper() for x in op_names}
    counts: Dict[str, int] = {x: 0 for x in wanted}
    nodes: List[TaskNode] = []
    for n in getattr(graph, 'nodes', {}).values():
        op = _op_name(n)
        if op not in wanted:
            continue
        if counts.get(op, 0) >= max_per_op:
            continue
        nodes.append(n)
        counts[op] = counts.get(op, 0) + 1
        if wanted and all(counts.get(x, 0) >= max_per_op for x in wanted):
            break
    return nodes


def _runtime_probe_candidate_batches(batch: int) -> List[int]:
    b = max(1, int(batch or 1))
    vals = {1, b}
    x = 1
    while x < b:
        vals.add(x)
        x *= 2
    # Include a midpoint for non-power-of-two batches without exploding probe cost.
    vals.add(max(1, (b + 1) // 2))
    return sorted(v for v in vals if 1 <= v <= b)


def _runtime_probe_safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _runtime_probe_has_weight(node: TaskNode) -> bool:
    try:
        return bool(getattr(node, 'weight_id', None)) and int(getattr(node, 'weight_size', 0) or 0) > 0
    except Exception:
        return False


def _runtime_probe_storage_fmt(cfg: Dict | None, uniform_weight_storage_fmt: str | None = None) -> str:
    """Return the weight-storage format that the final scheduler will use.

    Runtime calibration must not invent a manual PAPI/PAISE override.  It should
    use the same uniform storage mode as the actual run; if the caller does not
    pass one, the scheduler's default is ND.
    """
    raw = uniform_weight_storage_fmt
    if raw in (None, '') and isinstance(cfg, dict):
        raw = cfg.get('_runtime_probe_weight_storage_fmt', None)
    if raw in (None, ''):
        raw = 'ND'
    try:
        return str(_normalize_weight_storage_fmt(raw or 'ND'))
    except Exception:
        return 'ND'


def _runtime_probe_weight_components(
    cost: CostModel,
    node: TaskNode,
    dev: DeviceSpec,
    label: PlanLabel,
    *,
    batch: int,
    seq_len: int,
    phase: str,
    weight_storage_fmt: str,
) -> Dict[str, Any]:
    """Scheduler-consistent steady-state latency for one node on one device.

    The old inspired-baseline calibration used CostModel.node_device_cost(), which
    is only the bare kernel path.  For FC/linear nodes it misses the same
    weight-format compute-stage terms that SchedulerBase._weight_load_time() adds
    through CostModel.weighted_compute_stage(), most importantly NPU ND->NZ->ZN/ZZ
    conversion.  This helper makes the probe match the final scheduler's
    per-token, weights-resident compute path while still reporting raw and cold
    estimates for diagnostics.
    """
    raw_s: float | None = None
    try:
        raw_s = float(cost.node_device_cost(node, dev, label, int(batch), int(seq_len), str(phase)))
    except Exception:
        raw_s = None

    if not _runtime_probe_has_weight(node):
        return {
            'total_s': raw_s if raw_s is not None and math.isfinite(raw_s) else float('inf'),
            'raw_kernel_s': raw_s,
            'weighted_compute_s': raw_s,
            'weight_storage_fmt': None,
            'host_src_fmt': None,
            'resident_weight_fmt': None,
            'compute_fmt': None,
            'compute_backend': None,
            'compute_rule': 'raw_node_device_cost',
            'compute_b1_s': None,
            'compute_b2_s': None,
            'launch_overhead_s': None,
            'cold_weight_load_comm_s': 0.0,
            'cold_weight_local_load_s': 0.0,
            'cold_total_s': raw_s,
            'diagnostic': 'weightless_node_raw_cost',
        }

    storage_fmt = _runtime_probe_storage_fmt(None, weight_storage_fmt)
    try:
        host_src_fmt = str(cost.weight_host_source_format(str(storage_fmt), dev))
        resident_fmt = str(cost.weight_resident_format(str(host_src_fmt), dev))
        stage = cost.weighted_compute_stage(
            node,
            dev,
            label,
            int(batch),
            int(seq_len),
            str(phase),
            resident_weight_fmt=str(resident_fmt),
        )

        cold_comm_s = 0.0
        cold_local_s = 0.0
        try:
            host = cost.get_host_device()
            wsize = int(getattr(node, 'weight_size', 0) or 0)
            wire_bytes = int(cost.weight_transfer_comm_bytes(int(wsize), str(storage_fmt), dev_or_type=dev))
            cold_comm_s = float(cost.comm_cost(host, dev, int(wire_bytes)))
            if str(getattr(dev, 'type', '') or '').lower() == 'pim':
                cold_local_s = float(cost.pim_local_weight_load_time(int(wsize), str(host_src_fmt), dev=dev))
        except Exception:
            cold_comm_s = 0.0
            cold_local_s = 0.0

        total_s = float(stage.total_s)
        return {
            'total_s': float(total_s),
            'raw_kernel_s': raw_s,
            'weighted_compute_s': float(stage.total_s),
            'weight_storage_fmt': str(storage_fmt),
            'host_src_fmt': str(host_src_fmt),
            'resident_weight_fmt': str(resident_fmt),
            'compute_fmt': str(stage.compute_fmt),
            'compute_backend': str(stage.backend),
            'compute_rule': str(stage.combine_rule),
            'compute_b1_s': float(stage.b1_s),
            'compute_b2_s': float(stage.b2_s),
            'launch_overhead_s': float(stage.launch_overhead_s),
            # Diagnostic cold-start terms.  They are not charged into total_s
            # because calibration decides steady-state decode placement; final
            # scheduling still accounts for real cache misses and transfer queues.
            'cold_weight_load_comm_s': float(cold_comm_s),
            'cold_weight_local_load_s': float(cold_local_s),
            'cold_total_s': float(total_s + cold_comm_s + cold_local_s),
            'diagnostic': 'weighted_compute_stage_steady_state',
        }
    except Exception as exc:
        return {
            'total_s': raw_s if raw_s is not None and math.isfinite(raw_s) else float('inf'),
            'raw_kernel_s': raw_s,
            'weighted_compute_s': raw_s,
            'weight_storage_fmt': str(storage_fmt),
            'host_src_fmt': None,
            'resident_weight_fmt': None,
            'compute_fmt': None,
            'compute_backend': None,
            'compute_rule': 'fallback_raw_node_device_cost',
            'compute_b1_s': None,
            'compute_b2_s': None,
            'launch_overhead_s': None,
            'cold_weight_load_comm_s': 0.0,
            'cold_weight_local_load_s': 0.0,
            'cold_total_s': raw_s,
            'diagnostic': f'weighted_compute_stage_failed: {exc}',
        }


def _runtime_best_device_type_probe(
    cost: CostModel,
    cluster: Cluster,
    node: TaskNode,
    dev_type: str,
    *,
    batch: int,
    seq_len: int,
    phase: str = 'decode',
    weight_storage_fmt: str = 'ND',
) -> Dict[str, Any]:
    try:
        devs = list(cluster.devices_by_type(dev_type) or [])
    except Exception:
        devs = []
    if not devs:
        return {'total_s': float('inf'), 'device': None, 'device_type': str(dev_type)}

    label = PlanLabel(kv_in_pim=True, pim_mode='runtime_inspired_probe', kv_place='pim')
    best: Dict[str, Any] | None = None
    for dev in devs:
        comp = _runtime_probe_weight_components(
            cost,
            node,
            dev,
            label,
            batch=int(batch),
            seq_len=int(seq_len),
            phase=str(phase),
            weight_storage_fmt=str(weight_storage_fmt),
        )
        total_s = float(comp.get('total_s', float('inf')) or float('inf'))
        if best is None or total_s < float(best.get('total_s', float('inf'))):
            best = {
                'device': str(getattr(dev, 'name', '') or ''),
                'device_type': str(getattr(dev, 'type', dev_type) or dev_type),
                **comp,
            }
    return best if best is not None else {'total_s': float('inf'), 'device': None, 'device_type': str(dev_type)}


def _runtime_best_device_type_cost(
    cost: CostModel,
    cluster: Cluster,
    node: TaskNode,
    dev_type: str,
    *,
    batch: int,
    seq_len: int,
    phase: str = 'decode',
    weight_storage_fmt: str = 'ND',
) -> float:
    probe = _runtime_best_device_type_probe(
        cost, cluster, node, dev_type, batch=batch, seq_len=seq_len, phase=phase, weight_storage_fmt=weight_storage_fmt
    )
    try:
        return float(probe.get('total_s', float('inf')))
    except Exception:
        return float('inf')


def _runtime_avg_probe_for_nodes(
    cost: CostModel,
    cluster: Cluster,
    nodes: List[TaskNode],
    *,
    batch: int,
    seq_len: int,
    phase: str = 'decode',
    weight_storage_fmt: str = 'ND',
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    npu_vals: List[float] = []
    pim_vals: List[float] = []
    npu_raw_vals: List[float] = []
    pim_raw_vals: List[float] = []

    for n in nodes:
        npu_probe = _runtime_best_device_type_probe(
            cost, cluster, n, 'npu', batch=batch, seq_len=seq_len, phase=phase, weight_storage_fmt=weight_storage_fmt
        )
        pim_probe = _runtime_best_device_type_probe(
            cost, cluster, n, 'pim', batch=batch, seq_len=seq_len, phase=phase, weight_storage_fmt=weight_storage_fmt
        )
        npu_s = float(npu_probe.get('total_s', float('inf')) or float('inf'))
        pim_s = float(pim_probe.get('total_s', float('inf')) or float('inf'))
        ratio = (npu_s / pim_s) if math.isfinite(npu_s) and math.isfinite(pim_s) and pim_s > 0.0 else None
        npu_raw = _runtime_probe_safe_float(npu_probe.get('raw_kernel_s'))
        pim_raw = _runtime_probe_safe_float(pim_probe.get('raw_kernel_s'))

        rows.append({
            'node_id': str(getattr(n, 'id', '') or ''),
            'op': _op_name(n),
            # npu_s/pim_s are the values used for calibration.
            'npu_s': float(npu_s) if math.isfinite(npu_s) else None,
            'pim_s': float(pim_s) if math.isfinite(pim_s) else None,
            'npu_over_pim': float(ratio) if ratio is not None else None,
            'pim_better': bool(ratio is not None and ratio > 1.0),
            # Raw kernel-only values are kept to diagnose any old/new mismatch.
            'npu_raw_kernel_s': float(npu_raw) if npu_raw is not None else None,
            'pim_raw_kernel_s': float(pim_raw) if pim_raw is not None else None,
            'npu_probe': npu_probe,
            'pim_probe': pim_probe,
        })
        if math.isfinite(npu_s):
            npu_vals.append(float(npu_s))
        if math.isfinite(pim_s):
            pim_vals.append(float(pim_s))
        if npu_raw is not None:
            npu_raw_vals.append(float(npu_raw))
        if pim_raw is not None:
            pim_raw_vals.append(float(pim_raw))

    avg_npu = (sum(npu_vals) / len(npu_vals)) if npu_vals else float('inf')
    avg_pim = (sum(pim_vals) / len(pim_vals)) if pim_vals else float('inf')
    avg_ratio = (avg_npu / avg_pim) if math.isfinite(avg_npu) and math.isfinite(avg_pim) and avg_pim > 0.0 else None
    avg_npu_raw = (sum(npu_raw_vals) / len(npu_raw_vals)) if npu_raw_vals else None
    avg_pim_raw = (sum(pim_raw_vals) / len(pim_raw_vals)) if pim_raw_vals else None
    return {
        'probe_cost_mode': 'scheduler_weighted_compute_stage_steady_state',
        'weight_storage_fmt': str(weight_storage_fmt),
        'avg_npu_s': float(avg_npu) if math.isfinite(avg_npu) else None,
        'avg_pim_s': float(avg_pim) if math.isfinite(avg_pim) else None,
        'avg_npu_over_pim': float(avg_ratio) if avg_ratio is not None else None,
        'avg_npu_raw_kernel_s': float(avg_npu_raw) if avg_npu_raw is not None else None,
        'avg_pim_raw_kernel_s': float(avg_pim_raw) if avg_pim_raw is not None else None,
        'pim_better': bool(avg_ratio is not None and avg_ratio > 1.0),
        'nodes': rows,
    }


def _runtime_calibrate_inspired_baseline_hparams(
    cfg: Dict,
    *,
    cluster: Cluster,
    graph: TaskGraph,
    shape: Any,
    cost: CostModel | None = None,
    result_dir: Path | None = None,
    tag: str | None = None,
    save_report: bool = True,
    uniform_weight_storage_fmt: str | None = None,
) -> Dict[str, Any]:
    """Calibrate PAPI/PAISE-inspired knobs for this single evaluate run.

    The routine mutates ``cfg`` with the effective runtime-calibrated knobs and
    returns a compact probe report.  It is intentionally evaluated once per
    workload, not once per baseline, so PAPI/PAISE and DOPS consume the same
    input graph and hardware model.
    """
    probe_weight_storage_fmt = _runtime_probe_storage_fmt(cfg, uniform_weight_storage_fmt)
    if (
        cfg.get('_inspired_baseline_runtime_hparams_applied')
        and str(cfg.get('_inspired_baseline_runtime_probe_weight_storage_fmt', 'ND')) == str(probe_weight_storage_fmt)
    ):
        return dict(cfg.get('_inspired_baseline_runtime_probe') or {})

    owns_cost = False
    if cost is None:
        owns_cost = True
        probe_log = Path(cfg.get('simulation_log_file') or './output/pim_simulation.txt')
        if result_dir is not None:
            probe_log = Path(result_dir) / f"inspired_runtime_probe_{tag or 'run'}.pim.log"
        cost = _build_cost_model_for_run(cfg, cluster, shape, simulation_log_file=probe_log)
        try:
            cost.logger.start_simulation()
        except Exception:
            pass

    batch = max(1, int(cfg.get('batch', 1) or 1))
    prefill_len = max(1, int(cfg.get('prefill_len', 128) or 128))
    decode_len = max(0, int(cfg.get('decode_len', 0) or 0))
    seq_probe = prefill_len
    tlp = int(_RUNTIME_PAPI_TLP)

    fc_nodes = _runtime_probe_nodes(graph, set(_FC_OPS))
    paise_attn_nodes = _runtime_probe_nodes(graph, set(_PAISE_ATTN_OPS))
    papi_attn_nodes = _runtime_probe_nodes(graph, set(_PAPI_ATTN_OPS))

    papi_threshold_probe: List[Dict[str, Any]] = []
    positive_rlp_xtlp: List[int] = []
    for cand_batch in _runtime_probe_candidate_batches(batch):
        probe = _runtime_avg_probe_for_nodes(
            cost,
            cluster,
            fc_nodes,
            batch=int(cand_batch),
            seq_len=seq_probe,
            phase='decode',
            weight_storage_fmt=str(probe_weight_storage_fmt),
        )
        rlp_x_tlp = int(cand_batch) * int(tlp)
        if bool(probe.get('pim_better')):
            positive_rlp_xtlp.append(rlp_x_tlp)
        papi_threshold_probe.append({
            'batch': int(cand_batch),
            'rlp': int(cand_batch),
            'tlp': int(tlp),
            'rlp_x_tlp': int(rlp_x_tlp),
            **probe,
        })

    if positive_rlp_xtlp:
        papi_alpha = float(max(positive_rlp_xtlp))
        papi_alpha_note = 'largest_measured_rlp_x_tlp_with_pim_better_fc'
    else:
        # The smallest feasible RLP*TLP in this simulator is 1.  If PIM is not
        # faster even at that point, the calibrated threshold is below the
        # feasible domain.  Encoding it as nextafter(1, 0) avoids a hard-coded
        # alpha=0 while preserving the PAPI decision rule for every feasible
        # RLP*TLP value.
        try:
            papi_alpha = float(math.nextafter(1.0, 0.0))
        except AttributeError:
            papi_alpha = 0.9999999999999999
        papi_alpha_note = 'below_min_feasible_rlp_x_tlp_no_fc_offload'

    paise_fc_probe = _runtime_avg_probe_for_nodes(
        cost,
        cluster,
        fc_nodes,
        batch=batch,
        seq_len=seq_probe,
        phase='decode',
        weight_storage_fmt=str(probe_weight_storage_fmt),
    )
    paise_attention_probe = _runtime_avg_probe_for_nodes(
        cost,
        cluster,
        paise_attn_nodes,
        batch=batch,
        seq_len=seq_probe,
        phase='decode',
        weight_storage_fmt=str(probe_weight_storage_fmt),
    )
    papi_attention_probe = _runtime_avg_probe_for_nodes(
        cost,
        cluster,
        papi_attn_nodes,
        batch=batch,
        seq_len=seq_probe,
        phase='decode',
        weight_storage_fmt=str(probe_weight_storage_fmt),
    )

    paise_fc_policy = 'always' if bool(paise_fc_probe.get('pim_better')) else 'never'
    current_rlp = batch
    current_rlp_x_tlp = int(current_rlp) * int(tlp)
    fc_on_pim_for_current_papi = bool(float(current_rlp_x_tlp) <= float(papi_alpha))

    effective_config = {
        'papi_tlp': int(tlp),
        'papi_rlp': int(current_rlp),
        'papi_fc_threshold_alpha': float(papi_alpha),
        'papi_alpha_source': 'runtime_scheduler_weighted_probe',
        'inspired_probe_weight_storage_fmt': str(probe_weight_storage_fmt),
        'paise_policy_seq_len': int(seq_probe),
        'paise_min_speedup': 0.0,
        'paise_fc_policy': str(paise_fc_policy),
        'paise_attention_policy': 'always',
        'paise_policy_source': 'runtime_scheduler_weighted_probe_for_fc_attention_forced_by_paper_semantics',
    }

    report: Dict[str, Any] = {
        'scope': 'per_evaluate_runtime_calibration',
        'config_summary': {
            'model_family': cfg.get('model_family'),
            'model_variant': cfg.get('model_variant'),
            'dtype': cfg.get('dtype'),
            'batch': int(batch),
            'prefill_len': int(prefill_len),
            'decode_len': int(decode_len),
            'npu_backend': cfg.get('npu_backend'),
            'pim_fast_mode': bool(cfg.get('pim_fast_mode', False)),
            'hardware_json': cfg.get('hardware_json'),
            'weight_storage_fmt': str(probe_weight_storage_fmt),
            'probe_cost_mode': 'scheduler_weighted_compute_stage_steady_state',
        },
        'effective_config': effective_config,
        'papi': {
            'rule': 'FC_on_PIM iff RLP*TLP <= papi_fc_threshold_alpha',
            'current_rlp_x_tlp': int(current_rlp_x_tlp),
            'fc_on_pim_for_current_workload': bool(fc_on_pim_for_current_papi),
            'alpha_note': papi_alpha_note,
            'threshold_probe': papi_threshold_probe,
            'attention_qk_softmax_sv_probe': papi_attention_probe,
        },
        'paise': {
            'fc_policy': str(paise_fc_policy),
            'attention_policy': 'always',
            'fc_probe': paise_fc_probe,
            'attention_qk_sv_probe': paise_attention_probe,
            'notes': [
                'PAISE-inspired does not add explicit gamma(DLA) or beta(idle-bank) terms to the placement rule.',
                'Runtime calibration uses scheduler-consistent weighted_compute_stage steady-state costs for weight-bearing ops.',
                'Raw node_device_cost values are still reported as *_raw_kernel_s diagnostics only.',
                'Final timing still uses the existing DOPS scheduler/cost-model paths unchanged.',
            ],
        },
    }

    cfg.update(effective_config)
    cfg['_runtime_probe_weight_storage_fmt'] = str(probe_weight_storage_fmt)
    cfg['_inspired_baseline_runtime_hparams_applied'] = True
    cfg['_inspired_baseline_runtime_probe_weight_storage_fmt'] = str(probe_weight_storage_fmt)
    cfg['_inspired_baseline_runtime_probe'] = report

    if save_report and result_dir is not None:
        try:
            result_dir.mkdir(parents=True, exist_ok=True)
            out_path = result_dir / f"inspired_runtime_hparams_{tag or 'run'}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            cfg['_inspired_baseline_runtime_probe_json'] = str(out_path)
            print(f"[inspired-runtime] calibrated PAPI/PAISE knobs saved to: {out_path}", flush=True)
        except Exception:
            pass

    try:
        print(
            f"[inspired-runtime] PAPI alpha={float(papi_alpha):.6g} "
            f"({papi_alpha_note}); RLP*TLP={current_rlp_x_tlp}; "
            f"PAPI_FC_on_PIM={fc_on_pim_for_current_papi}; "
            f"PAISE_FC={paise_fc_policy}; PAISE_Attn=always; "
            f"probe_weight_fmt={probe_weight_storage_fmt}",
            flush=True,
        )
    except Exception:
        pass

    if owns_cost:
        try:
            cost.logger.end_simulation()
            cost.logger.close()
        except Exception:
            pass

    return report


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

    pol = policy_token
    if pol in {'PAPI-inspired', 'PAISE-inspired'}:
        try:
            _runtime_calibrate_inspired_baseline_hparams(
                cfg,
                cluster=cluster,
                graph=graph,
                shape=shape,
                cost=cost,
                result_dir=Path(cfg.get('result_dir', './output')),
                tag=f"{prefill_len}x{decode_len}",
                save_report=True,
                uniform_weight_storage_fmt=uniform_weight_storage_fmt,
            )
        except Exception as exc:
            logger.warning('[inspired-runtime] runtime calibration failed for %s: %s', pol, exc)

    # Build separate prefill and decode graphs for the baseline policy.
    if pol in _BASELINE_REGISTRY:
        g_prefill = _invoke_baseline_policy(
            _BASELINE_REGISTRY[pol], graph, phase="prefill", cfg=cfg, cost=cost, cluster=cluster, shape=shape
        )
        g_decode = _invoke_baseline_policy(
            _BASELINE_REGISTRY[pol], graph, phase="decode", cfg=cfg, cost=cost, cluster=cluster, shape=shape
        )
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
        "inspired_baseline_runtime_probe": cfg.get('_inspired_baseline_runtime_probe'),
        "inspired_baseline_runtime_probe_json": cfg.get('_inspired_baseline_runtime_probe_json'),
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
        sched = _make_scheduler(strategy_token, cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
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
        'config': {'batch': int((cfg or {}).get('batch', 1)), 'prefill_len': int((cfg or {}).get('prefill_len', 0)), 'decode_len': int((cfg or {}).get('decode_len', 0)), 'dtype': (cfg or {}).get('dtype')},
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

    policy_display = _display_policy_name(str(policy).replace('algo:', '', 1))
    if policy_display in {'PAPI-inspired', 'PAISE-inspired'}:
        probe = times.get('inspired_baseline_runtime_probe') or (cfg or {}).get('_inspired_baseline_runtime_probe')
        if probe:
            payload['inspired_baseline_runtime_probe'] = probe
        probe_json = times.get('inspired_baseline_runtime_probe_json') or (cfg or {}).get('_inspired_baseline_runtime_probe_json')
        if probe_json:
            payload['inspired_baseline_runtime_probe_json'] = str(probe_json)

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

    inspired_runtime_probe = None
    if _runtime_probe_baselines_requested(blist):
        probe_log_path = base_dir / f"inspired_runtime_probe_{tag}.pim.log"
        probe_cost = _build_cost_model_for_run(
            cfg,
            cluster,
            shared_shape,
            simulation_log_file=probe_log_path,
        )
        try:
            probe_cost.logger.start_simulation()
        except Exception:
            pass
        inspired_runtime_probe = _runtime_calibrate_inspired_baseline_hparams(
            cfg,
            cluster=cluster,
            graph=shared_graph,
            shape=shared_shape,
            cost=probe_cost,
            result_dir=base_dir,
            tag=tag,
            save_report=True,
        )
        try:
            probe_cost.logger.end_simulation()
            probe_cost.logger.close()
        except Exception:
            pass

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
        row = {
            'policy': _policy_label(b),
            'pim_strategy': r.get('pim_strategy'),
            'kv_in_pim': bool(r.get('kv_in_pim', False)),
            'kv_total_bytes': int(r.get('kv_total_bytes', 0) or 0),
            'pim_weight_capacity_bytes': int(r.get('pim_weight_capacity_bytes', 0) or 0),
            **{k: r[k] for k in ('prefill_time_s', 'decode_time_s', 'total_time_s')},
        }
        if _normalize_baseline_name(b) in {'PAPI-inspired', 'PAISE-inspired'}:
            row['inspired_runtime_hparams'] = {
                'papi_tlp': cfg_b.get('papi_tlp'),
                'papi_rlp': cfg_b.get('papi_rlp'),
                'papi_fc_threshold_alpha': cfg_b.get('papi_fc_threshold_alpha'),
                'papi_alpha_source': cfg_b.get('papi_alpha_source'),
                'inspired_probe_weight_storage_fmt': cfg_b.get('inspired_probe_weight_storage_fmt'),
                'paise_policy_seq_len': cfg_b.get('paise_policy_seq_len'),
                'paise_fc_policy': cfg_b.get('paise_fc_policy'),
                'paise_attention_policy': cfg_b.get('paise_attention_policy'),
                'paise_policy_source': cfg_b.get('paise_policy_source'),
            }
            if cfg_b.get('_inspired_baseline_runtime_probe_json'):
                row['inspired_runtime_probe_json'] = cfg_b.get('_inspired_baseline_runtime_probe_json')
        results.append(row)

    alist: List[str] = []
    for item in algos:
        token = _normalize_algo_name(item)
        if token and token not in alist:
            alist.append(token)

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
            **{k: res[k] for k in ('prefill_time_s', 'decode_time_s', 'total_time_s')},
        })

    if results:
        os.makedirs(os.path.dirname(combined_out), exist_ok=True)
        cfg_for_report = {k: v for k, v in cfg.items() if k != '_inspired_baseline_runtime_probe'}
        payload = {'config': cfg_for_report, 'results': results}
        if inspired_runtime_probe is not None:
            payload['inspired_baseline_runtime_probe'] = inspired_runtime_probe
            if cfg.get('_inspired_baseline_runtime_probe_json'):
                payload['inspired_baseline_runtime_probe_json'] = cfg.get('_inspired_baseline_runtime_probe_json')
        with open(combined_out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[REPORT] Combined comparison saved to: {combined_out}")

    print("\n=== Strategy/Baseline Comparison ===")
    header = f"{'Policy':<24} {'PIM':<8} {'Prefill(s)':>12} {'Decode(s)':>12} {'Total(s)':>12}"
    print(header)
    print('-' * len(header))
    for r in results:
        pim_s = str(r.get('pim_strategy') or '')
        print(f"{r['policy']:<24} {pim_s:<8} {r['prefill_time_s']:>12.4f} {r['decode_time_s']:>12.4f} {r['total_time_s']:>12.4f}")
