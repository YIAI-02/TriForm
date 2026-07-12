"""Scheduler factory and phase simulation helpers."""

from __future__ import annotations

from .shared import *

def _serialize_schedule(schedule: List[ScheduledTask], *, phase: str, token_idx: int | None=None) -> List[Dict]:
    """Convert ScheduledTask list to JSON-friendly dicts."""
    out: List[Dict] = []
    for t in schedule:
        out.append({'node_id': t.node_id, 'device': t.device, 'start': float(t.start), 'finish': float(t.finish), 'duration': float(max(0.0, t.finish - t.start)), 'phase': phase, 'token_idx': token_idx})
    return out

def simulate_prefill(sched: SchedulerBase, cfg: Dict, graph: TaskGraph) -> tuple[float, List[Dict]]:
    """
    Simulate prefill phase: process entire prefix at once.
    current_length = prefill_len
    """
    prefill_len = int(cfg.get('prefill_len', 128))
    sched.set_seq_len(prefill_len)
    prefill_sched = sched.schedule(graph, phase='prefill')
    prefill_time = sched.makespan(prefill_sched)
    return (prefill_time, _serialize_schedule(prefill_sched, phase='prefill', token_idx=None))

def simulate_decode_progressive(sched: SchedulerBase, cfg: Dict, graph: TaskGraph, prefill_end: float) -> tuple[float, List[Dict]]:
    prefill_len = int(cfg.get('prefill_len', 128))
    decode_len = int(cfg.get('decode_len', 32))
    decode_horizon_len = int(cfg.get('decode_horizon_len', decode_len) or decode_len)
    global_end = float(prefill_end)
    steps_serialized: List[Dict] = []

    if isinstance(cfg, dict):
        dump_raw = cfg.get('decode_sample_stride', 1)
        refresh_cfg = cfg.get('decode_plan_refresh_stride', 64)
        dump_stride = int(1 if dump_raw is None else dump_raw)
        refresh_raw = int(64 if refresh_cfg is None else refresh_cfg)
    else:
        dump_stride = 1
        refresh_raw = 64

    dump_stride = max(1, int(dump_stride))
    refresh_stride = max(0, int(refresh_raw))
    current_plan: Dict[str, Any] | None = None

    def _set_decode_ctx(token_idx: int) -> None:
        setter = getattr(sched, 'set_decode_context', None)
        if callable(setter):
            try:
                setter(cur_token_idx=int(token_idx), total_decode_tokens=int(decode_horizon_len), cfg=cfg)
            except TypeError:
                try:
                    setter(cur_token_idx=int(token_idx), total_decode_tokens=int(decode_horizon_len))
                except Exception:
                    pass
            except Exception:
                pass

    def _clear_decode_ctx() -> None:
        clearer = getattr(sched, 'clear_decode_context', None)
        if callable(clearer):
            try:
                clearer()
            except Exception:
                pass

    def _need_exact_token(token_idx: int) -> bool:
        if token_idx < 2:
            return True
        if current_plan is None:
            return True
        if refresh_stride > 0 and (token_idx % refresh_stride) == 0:
            return True
        return False

    def _should_dump_schedule(token_idx: int) -> bool:
        if dump_stride <= 1:
            return True
        if token_idx in (0, 1, max(0, decode_len - 1)):
            return True
        return (token_idx % dump_stride) == 0

    def _validate_fixed_plan_or_raise(plan_obj: Any) -> Dict[str, Any]:
        try:
            plan_map = dict(plan_obj or {})
        except Exception as e:
            raise RuntimeError('Fixed decode plan is not mapping-like') from e

        raw_order = plan_map.get('order', None)
        if raw_order is None or isinstance(raw_order, (str, bytes)):
            raise RuntimeError("Fixed decode plan missing iterable 'order'")
        try:
            order = tuple(str(x) for x in raw_order)
        except Exception as e:
            raise RuntimeError("Fixed decode plan 'order' is not iterable") from e
        if not order:
            raise RuntimeError("Fixed decode plan has empty 'order'")
        if len(set(order)) != len(order):
            raise RuntimeError("Fixed decode plan 'order' contains duplicate node ids")

        raw_dev_map = plan_map.get('device_by_node', None)
        try:
            device_by_node = {str(k): str(v) for k, v in dict(raw_dev_map or {}).items()}
        except Exception as e:
            raise RuntimeError("Fixed decode plan missing mapping 'device_by_node'") from e
        if not device_by_node:
            raise RuntimeError("Fixed decode plan has empty 'device_by_node'")

        return {
            'order': order,
            'device_by_node': device_by_node,
        }

    def _refresh_plan_from_schedule(dec_sched: List[ScheduledTask]) -> None:
        nonlocal current_plan
        exporter = getattr(sched, 'export_fixed_plan', None)
        if not callable(exporter):
            raise RuntimeError('Scheduler does not implement export_fixed_plan()')
        plan_obj = exporter(dec_sched)
        current_plan = _validate_fixed_plan_or_raise(plan_obj)

    try:
        for t in range(decode_len):
            cur_len = int(prefill_len + t)
            sched.set_seq_len(cur_len)
            _set_decode_ctx(t)

            if _need_exact_token(t):
                dec_sched = sched.schedule(graph, phase='decode')
                _refresh_plan_from_schedule(dec_sched)
                estimated = False
            else:
                plan_runner = getattr(sched, 'schedule_with_plan', None)
                if not callable(plan_runner):
                    raise RuntimeError(
                        'decode fixed-plan replay requested, but scheduler does not implement schedule_with_plan()'
                    )
                if current_plan is None:
                    raise RuntimeError('decode fixed-plan replay requested before any valid fixed plan was prepared')
                dec_sched = plan_runner(graph, phase='decode', plan=current_plan)
                estimated = True

            token_end = float(sched.makespan(dec_sched))
            step_time = max(0.0, float(token_end - global_end))
            global_end = float(token_end)

            steps_serialized.append({
                't': int(t),
                'seq_len': int(cur_len),
                'step_time': float(step_time),
                'estimated': bool(estimated),
                'schedule': (
                    _serialize_schedule(dec_sched, phase='decode', token_idx=t)
                    if _should_dump_schedule(t)
                    else None
                ),
            })
    finally:
        _clear_decode_ctx()

    return (float(global_end - prefill_end), steps_serialized)

def _make_scheduler(
    name: str,
    cluster: Cluster,
    cost: CostModel,
    label: PlanLabel,
    batch: int,
    seq_len: int,
    buffer: GlobalMemoryManager,
    *,
    rand_seed: int | None = None,
):
    """Factory for scheduler strategies used by evaluate-suite."""

    strategy = _normalize_algo_name(name or 'HEFT')

    if strategy == 'Bifocal':
        if BifocalScheduler is None:
            raise ImportError(
                "BifocalScheduler is not available. Please export it from the scheduler package."
            )
        seed = None if rand_seed is None else int(rand_seed)
        return BifocalScheduler(
            cluster,
            cost,
            label,
            batch=batch,
            seq_len=seq_len,
            buffer=buffer,
            rand_seed=seed,
        )

    if strategy == 'HEFT':
        return HEFTScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer)

    if strategy == 'Naive':
        return NaiveTopoScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer)

    raise ValueError(f"Unknown scheduler strategy: {name}")
