from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        x = float(value)
        if x != x:
            return float(default)
        return x
    s = str(value).strip()
    if not s or s.lower() in {'nan', 'none', 'null'}:
        return float(default)
    try:
        x = float(s)
        if x != x:
            return float(default)
        return x
    except Exception:
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_to_float(value, float(default))))
    except Exception:
        return int(default)


def _safe_div(num: float, den: float) -> float:
    den_f = float(den or 0.0)
    if den_f <= 0.0:
        return 0.0
    return float(num) / den_f


def _dominant(a: float, b: float, *, left: str, right: str, eps: float = 1e-18) -> str:
    aa = float(a or 0.0)
    bb = float(b or 0.0)
    if aa <= eps and bb <= eps:
        return ''
    if abs(aa - bb) <= eps:
        return 'equal'
    return left if aa > bb else right


def enrich_weight_stage_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(row)
    wid = str(row.get('wid', '') or '').strip()
    weight_size_nd = _to_float(row.get('weight_size_nd', 0.0), 0.0)
    is_weighted = 1 if (wid or weight_size_nd > 0.0) else 0

    load_active_s = _to_float(row.get('load_active_s', 0.0), 0.0)
    load_total_s = _to_float(row.get('load_total_s', 0.0), 0.0)
    queue_wait_s = _to_float(row.get('queue_wait_s', 0.0), 0.0)
    compute_total_s = _to_float(row.get('compute_total_s', 0.0), 0.0)
    lc_saved_s = _to_float(row.get('lc_overlap_saved_s', 0.0), 0.0)

    l1_s = _to_float(row.get('load_l1_s', 0.0), 0.0)
    l2_s = _to_float(row.get('load_l2_s', 0.0), 0.0)
    l2_write_only_s = _to_float(row.get('load_l2_write_only_s', 0.0), 0.0)
    l2_pack_only_est_s = _to_float(
        row.get('load_l2_pack_only_est_s', max(0.0, l2_s - l2_write_only_s)),
        max(0.0, l2_s - l2_write_only_s),
    )

    b1_s = _to_float(row.get('b1_s', 0.0), 0.0)
    b2_s = _to_float(row.get('b2_s', 0.0), 0.0)
    launch_s = _to_float(row.get('launch_overhead_s', 0.0), 0.0)
    compute_rule = str(row.get('compute_rule', '') or '').strip().lower()

    lc_pair_sum_s = load_active_s + compute_total_s
    lc_modeled_s = max(0.0, lc_pair_sum_s - lc_saved_s)

    load_l1_l2_pair_sum_s = l1_s + l2_s
    load_l1_l2_saved_s = max(0.0, load_l1_l2_pair_sum_s - load_active_s)

    b_pair_sum_s = b1_s + b2_s
    if compute_rule == 'max':
        b_effective_driver = _dominant(b1_s, b2_s, left='B1', right='B2')
        b_core_effective_s = max(b1_s, b2_s)
    elif compute_rule == 'sum':
        b_effective_driver = 'sum'
        b_core_effective_s = b_pair_sum_s
    elif compute_rule == 'trace':
        b_effective_driver = 'trace'
        b_core_effective_s = max(0.0, b1_s)
    elif compute_total_s > 0.0 and b_pair_sum_s <= 0.0:
        b_effective_driver = 'direct'
        b_core_effective_s = max(0.0, compute_total_s - launch_s)
    else:
        b_effective_driver = _dominant(b1_s, b2_s, left='B1', right='B2')
        b_core_effective_s = max(0.0, compute_total_s - launch_s)

    out.update(
        {
            'is_weighted_op': int(is_weighted),
            'l_stage_s': float(load_active_s),
            'c_stage_s': float(compute_total_s),
            'l_stage_with_queue_s': float(load_total_s),
            'lc_pair_sum_s': float(lc_pair_sum_s),
            'lc_modeled_s': float(lc_modeled_s),
            'lc_value_dominant': _dominant(load_active_s, compute_total_s, left='L', right='C'),
            'load_share_of_lc_pair_sum': float(_safe_div(load_active_s, lc_pair_sum_s)),
            'compute_share_of_lc_pair_sum': float(_safe_div(compute_total_s, lc_pair_sum_s)),
            'lc_overlap_saved_ratio_of_pair_sum': float(_safe_div(lc_saved_s, lc_pair_sum_s)),
            'queue_share_of_total': float(_safe_div(queue_wait_s, _to_float(row.get('total_s', 0.0), 0.0))),
            'load_l1_l2_pair_sum_s': float(load_l1_l2_pair_sum_s),
            'load_l1_l2_saved_s': float(load_l1_l2_saved_s),
            'load_l1_l2_value_dominant': _dominant(l1_s, l2_s, left='L1', right='L2'),
            'load_l1_share_of_pair_sum': float(_safe_div(l1_s, load_l1_l2_pair_sum_s)),
            'load_l2_share_of_pair_sum': float(_safe_div(l2_s, load_l1_l2_pair_sum_s)),
            'load_l1_l2_saved_ratio_of_pair_sum': float(_safe_div(load_l1_l2_saved_s, load_l1_l2_pair_sum_s)),
            'load_l2_write_only_s': float(l2_write_only_s),
            'load_l2_pack_only_est_s': float(l2_pack_only_est_s),
            'b_pair_sum_s': float(b_pair_sum_s),
            'b_core_effective_s': float(b_core_effective_s),
            'b_value_dominant': _dominant(b1_s, b2_s, left='B1', right='B2'),
            'b_effective_driver': str(b_effective_driver),
            'b1_share_of_pair_sum': float(_safe_div(b1_s, b_pair_sum_s)),
            'b2_share_of_pair_sum': float(_safe_div(b2_s, b_pair_sum_s)),
            'launch_share_of_compute_total': float(_safe_div(launch_s, compute_total_s)),
        }
    )
    return out


def enrich_weight_stage_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [enrich_weight_stage_row(r) for r in rows]


def _default_aggregate_state() -> Dict[str, Any]:
    return {
        'weighted_op_count': 0,
        'actual_duration_s_sum': 0.0,
        'modeled_total_s_sum': 0.0,
        'queue_wait_s_sum': 0.0,
        'l_stage_s_sum': 0.0,
        'c_stage_s_sum': 0.0,
        'lc_pair_sum_s_sum': 0.0,
        'lc_overlap_saved_s_sum': 0.0,
        'load_l1_s_sum': 0.0,
        'load_l2_s_sum': 0.0,
        'load_l2_write_only_s_sum': 0.0,
        'load_l2_pack_only_est_s_sum': 0.0,
        'load_l1_l2_pair_sum_s_sum': 0.0,
        'load_l1_l2_saved_s_sum': 0.0,
        'b1_s_sum': 0.0,
        'b2_s_sum': 0.0,
        'launch_overhead_s_sum': 0.0,
        'b_pair_sum_s_sum': 0.0,
        'compute_total_s_sum': 0.0,
        'lc_dominant_L_count': 0,
        'lc_dominant_C_count': 0,
        'lc_dominant_equal_count': 0,
        'load_dominant_L1_count': 0,
        'load_dominant_L2_count': 0,
        'load_dominant_equal_count': 0,
        'b_dominant_B1_count': 0,
        'b_dominant_B2_count': 0,
        'b_dominant_equal_count': 0,
        'b_effective_driver_max_count': 0,
        'b_effective_driver_sum_count': 0,
        'b_effective_driver_trace_count': 0,
        'b_effective_driver_direct_count': 0,
        'compute_rule_max_count': 0,
        'compute_rule_sum_count': 0,
        'compute_rule_trace_count': 0,
        'compute_rule_direct_count': 0,
    }


def aggregate_weight_stage_rows(rows: Iterable[Mapping[str, Any]], group_by: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    keys = tuple(str(k) for k in (group_by or ()))

    for raw in rows:
        row = enrich_weight_stage_row(raw)
        if _to_int(row.get('is_weighted_op', 0), 0) <= 0:
            continue
        gkey = tuple(str(row.get(k, '') or '') for k in keys)
        if gkey not in groups:
            base = {k: v for k, v in zip(keys, gkey)}
            base.update(_default_aggregate_state())
            groups[gkey] = base
        agg = groups[gkey]

        agg['weighted_op_count'] += 1
        agg['actual_duration_s_sum'] += _to_float(row.get('duration', 0.0), 0.0)
        agg['modeled_total_s_sum'] += _to_float(row.get('total_s', 0.0), 0.0)
        agg['queue_wait_s_sum'] += _to_float(row.get('queue_wait_s', 0.0), 0.0)
        agg['l_stage_s_sum'] += _to_float(row.get('l_stage_s', 0.0), 0.0)
        agg['c_stage_s_sum'] += _to_float(row.get('c_stage_s', 0.0), 0.0)
        agg['lc_pair_sum_s_sum'] += _to_float(row.get('lc_pair_sum_s', 0.0), 0.0)
        agg['lc_overlap_saved_s_sum'] += _to_float(row.get('lc_overlap_saved_s', 0.0), 0.0)
        agg['load_l1_s_sum'] += _to_float(row.get('load_l1_s', 0.0), 0.0)
        agg['load_l2_s_sum'] += _to_float(row.get('load_l2_s', 0.0), 0.0)
        agg['load_l2_write_only_s_sum'] += _to_float(row.get('load_l2_write_only_s', 0.0), 0.0)
        agg['load_l2_pack_only_est_s_sum'] += _to_float(row.get('load_l2_pack_only_est_s', 0.0), 0.0)
        agg['load_l1_l2_pair_sum_s_sum'] += _to_float(row.get('load_l1_l2_pair_sum_s', 0.0), 0.0)
        agg['load_l1_l2_saved_s_sum'] += _to_float(row.get('load_l1_l2_saved_s', 0.0), 0.0)
        agg['b1_s_sum'] += _to_float(row.get('b1_s', 0.0), 0.0)
        agg['b2_s_sum'] += _to_float(row.get('b2_s', 0.0), 0.0)
        agg['launch_overhead_s_sum'] += _to_float(row.get('launch_overhead_s', 0.0), 0.0)
        agg['b_pair_sum_s_sum'] += _to_float(row.get('b_pair_sum_s', 0.0), 0.0)
        agg['compute_total_s_sum'] += _to_float(row.get('compute_total_s', 0.0), 0.0)

        lc_dom = str(row.get('lc_value_dominant', '') or '')
        if lc_dom == 'L':
            agg['lc_dominant_L_count'] += 1
        elif lc_dom == 'C':
            agg['lc_dominant_C_count'] += 1
        elif lc_dom == 'equal':
            agg['lc_dominant_equal_count'] += 1

        load_dom = str(row.get('load_l1_l2_value_dominant', '') or '')
        if load_dom == 'L1':
            agg['load_dominant_L1_count'] += 1
        elif load_dom == 'L2':
            agg['load_dominant_L2_count'] += 1
        elif load_dom == 'equal':
            agg['load_dominant_equal_count'] += 1

        b_dom = str(row.get('b_value_dominant', '') or '')
        if b_dom == 'B1':
            agg['b_dominant_B1_count'] += 1
        elif b_dom == 'B2':
            agg['b_dominant_B2_count'] += 1
        elif b_dom == 'equal':
            agg['b_dominant_equal_count'] += 1

        eff = str(row.get('b_effective_driver', '') or '').lower()
        if eff in {'b1', 'b2', 'equal', 'max'}:
            agg['b_effective_driver_max_count'] += 1
        elif eff == 'sum':
            agg['b_effective_driver_sum_count'] += 1
        elif eff == 'trace':
            agg['b_effective_driver_trace_count'] += 1
        elif eff == 'direct':
            agg['b_effective_driver_direct_count'] += 1

        rule = str(row.get('compute_rule', '') or '').lower()
        if rule == 'max':
            agg['compute_rule_max_count'] += 1
        elif rule == 'sum':
            agg['compute_rule_sum_count'] += 1
        elif rule == 'trace':
            agg['compute_rule_trace_count'] += 1
        elif rule in {'direct', ''}:
            agg['compute_rule_direct_count'] += 1

    out_rows: List[Dict[str, Any]] = []
    for agg in groups.values():
        n = float(agg.get('weighted_op_count', 0) or 0)
        row = dict(agg)
        row.update(
            {
                'avg_modeled_total_s': _safe_div(agg['modeled_total_s_sum'], n),
                'avg_actual_duration_s': _safe_div(agg['actual_duration_s_sum'], n),
                'avg_queue_wait_s': _safe_div(agg['queue_wait_s_sum'], n),
                'queue_wait_share_of_modeled_total': _safe_div(agg['queue_wait_s_sum'], agg['modeled_total_s_sum']),
                'load_share_of_lc_pair_sum': _safe_div(agg['l_stage_s_sum'], agg['lc_pair_sum_s_sum']),
                'compute_share_of_lc_pair_sum': _safe_div(agg['c_stage_s_sum'], agg['lc_pair_sum_s_sum']),
                'lc_overlap_saved_share_of_lc_pair_sum': _safe_div(agg['lc_overlap_saved_s_sum'], agg['lc_pair_sum_s_sum']),
                'lc_dominant_L_ratio': _safe_div(agg['lc_dominant_L_count'], n),
                'lc_dominant_C_ratio': _safe_div(agg['lc_dominant_C_count'], n),
                'lc_dominant_equal_ratio': _safe_div(agg['lc_dominant_equal_count'], n),
                'l1_share_of_load_pair_sum': _safe_div(agg['load_l1_s_sum'], agg['load_l1_l2_pair_sum_s_sum']),
                'l2_share_of_load_pair_sum': _safe_div(agg['load_l2_s_sum'], agg['load_l1_l2_pair_sum_s_sum']),
                'load_l1_l2_saved_share_of_load_pair_sum': _safe_div(agg['load_l1_l2_saved_s_sum'], agg['load_l1_l2_pair_sum_s_sum']),
                'load_dominant_L1_ratio': _safe_div(agg['load_dominant_L1_count'], n),
                'load_dominant_L2_ratio': _safe_div(agg['load_dominant_L2_count'], n),
                'load_dominant_equal_ratio': _safe_div(agg['load_dominant_equal_count'], n),
                'b1_share_of_b_pair_sum': _safe_div(agg['b1_s_sum'], agg['b_pair_sum_s_sum']),
                'b2_share_of_b_pair_sum': _safe_div(agg['b2_s_sum'], agg['b_pair_sum_s_sum']),
                'launch_share_of_compute_total': _safe_div(agg['launch_overhead_s_sum'], agg['compute_total_s_sum']),
                'b_dominant_B1_ratio': _safe_div(agg['b_dominant_B1_count'], n),
                'b_dominant_B2_ratio': _safe_div(agg['b_dominant_B2_count'], n),
                'b_dominant_equal_ratio': _safe_div(agg['b_dominant_equal_count'], n),
            }
        )
        out_rows.append(row)

    def _sort_key(r: Mapping[str, Any]) -> Tuple[Any, ...]:
        return tuple(str(r.get(k, '') or '') for k in keys) + (-_to_float(r.get('modeled_total_s_sum', 0.0), 0.0),)

    out_rows.sort(key=_sort_key)
    return out_rows


def read_csv_dicts(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open('r', newline='', encoding='utf-8') as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv_dicts(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows_list = [dict(r) for r in rows]
    fieldnames: List[str] = []
    seen = set()
    for r in rows_list:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(str(k))
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_list:
            w.writerow({k: r.get(k) for k in fieldnames})
    return p


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def summarize_weight_stage_trace_csv(
    ops_csv_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    output_stem: str | None = None,
    custom_groupings: Sequence[Sequence[str]] | None = None,
    write_enriched_ops_csv: bool = False,
) -> Dict[str, str]:
    ops_path = Path(ops_csv_path)
    out_root = Path(out_dir) if out_dir is not None else ops_path.parent
    stem = str(output_stem or ops_path.stem)

    rows = read_csv_dicts(ops_path)
    enriched = enrich_weight_stage_rows(rows)

    outputs: Dict[str, str] = {}
    if write_enriched_ops_csv:
        enriched_path = out_root / f'{stem}_enriched.csv'
        write_csv_dicts(enriched_path, enriched)
        outputs['enriched_csv'] = str(enriched_path)

    overall = aggregate_weight_stage_rows(enriched, [])
    by_phase = aggregate_weight_stage_rows(enriched, ['phase'])
    by_device = aggregate_weight_stage_rows(enriched, ['device_type'])
    by_op = aggregate_weight_stage_rows(enriched, ['op'])

    overall_path = out_root / f'{stem}_weight_stage_overall.csv'
    by_phase_path = out_root / f'{stem}_weight_stage_by_phase.csv'
    by_device_path = out_root / f'{stem}_weight_stage_by_device_type.csv'
    by_op_path = out_root / f'{stem}_weight_stage_by_op.csv'

    write_csv_dicts(overall_path, overall)
    write_csv_dicts(by_phase_path, by_phase)
    write_csv_dicts(by_device_path, by_device)
    write_csv_dicts(by_op_path, by_op)
    outputs.update(
        {
            'overall_csv': str(overall_path),
            'by_phase_csv': str(by_phase_path),
            'by_device_type_csv': str(by_device_path),
            'by_op_csv': str(by_op_path),
        }
    )

    custom_outputs: List[Dict[str, Any]] = []
    for grouping in list(custom_groupings or []):
        keys = [str(k).strip() for k in grouping if str(k).strip()]
        if not keys:
            continue
        rows_g = aggregate_weight_stage_rows(enriched, keys)
        suffix = '_'.join(keys)
        path_g = out_root / f'{stem}_weight_stage_by_{suffix}.csv'
        write_csv_dicts(path_g, rows_g)
        custom_outputs.append({'group_by': list(keys), 'csv': str(path_g)})

    summary_json_path = out_root / f'{stem}_weight_stage_summary.json'
    summary_payload = {
        'source_ops_csv': str(ops_path),
        'outputs': {
            **outputs,
            'custom_groupings': custom_outputs,
        },
        'overall': overall[0] if overall else {},
    }
    write_json(summary_json_path, summary_payload)
    outputs['summary_json'] = str(summary_json_path)
    return outputs


def _parse_group_by_args(values: Sequence[str] | None) -> List[List[str]]:
    out: List[List[str]] = []
    for item in list(values or []):
        keys = [x.strip() for x in str(item).split(',') if x.strip()]
        if keys:
            out.append(keys)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Summarize weighted-op L/C/B1/B2/L1/L2 proportions from an ops trace CSV.')
    ap.add_argument('--ops-csv', required=True, help='Path to *_ops_trace.csv')
    ap.add_argument('--out-dir', default=None, help='Directory for output summary files (default: same dir as ops csv).')
    ap.add_argument('--output-stem', default=None, help='Prefix/stem for output file names (default: input csv stem).')
    ap.add_argument('--group-by', action='append', default=[], help='Extra grouping, comma-separated. Example: --group-by phase,device_type,op')
    ap.add_argument('--write-enriched-csv', action='store_true', help='Also emit an enriched copy of the ops csv with derived stage columns.')
    args = ap.parse_args(list(argv) if argv is not None else None)

    outputs = summarize_weight_stage_trace_csv(
        args.ops_csv,
        out_dir=args.out_dir,
        output_stem=args.output_stem,
        custom_groupings=_parse_group_by_args(args.group_by),
        write_enriched_ops_csv=bool(args.write_enriched_csv),
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
