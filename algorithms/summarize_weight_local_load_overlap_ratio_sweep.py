#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _to_float(v: Any, default: float = float('nan')) -> float:
    try:
        if v in (None, ''):
            return float(default)
        x = float(v)
        if math.isfinite(x):
            return x
        return float(default)
    except Exception:
        return float(default)


def _to_int(v: Any, default: int = 0) -> int:
    try:
        if v in (None, ''):
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _ratio_sort_key(v: Any) -> Tuple[int, float, str]:
    s = str(v if v is not None else '').strip()
    if s == '':
        return (1, 0.0, s)
    try:
        x = float(s)
        if math.isfinite(x):
            return (0, -x, s)
    except Exception:
        pass
    return (1, 0.0, s)


def _stable_group_value(row: Dict[str, Any], key: str) -> str:
    val = row.get(key, '')
    if isinstance(val, float):
        if math.isfinite(val):
            return f"{val:.12g}"
        return ''
    return str(val if val is not None else '')


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float('nan')
    return float(sum(vals) / len(vals))


def _median(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float('nan')
    return float(statistics.median(vals))


def _enrich_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    ratio = _to_float(row.get('weight_local_load_overlap_ratio'), default=float('nan'))
    if math.isfinite(ratio):
        out['weight_local_load_overlap_ratio'] = f"{ratio:.12g}"
    nd_initial = _to_float(row.get('nd_initial'), default=float('nan'))
    nd_best = _to_float(row.get('nd_best'), default=float('nan'))
    iter_gain_s = _to_float(row.get('iter_gain_s'), default=float('nan'))
    iter_gain_pct = _to_float(row.get('iter_gain_pct'), default=float('nan'))
    if not math.isfinite(iter_gain_s) and math.isfinite(nd_initial) and math.isfinite(nd_best):
        iter_gain_s = float(nd_initial - nd_best)
    if not math.isfinite(iter_gain_pct) and math.isfinite(iter_gain_s) and math.isfinite(nd_initial) and nd_initial > 0.0:
        iter_gain_pct = float(iter_gain_s / nd_initial * 100.0)
    if math.isfinite(iter_gain_s):
        out['iter_gain_s'] = f"{iter_gain_s:.12g}"
    if math.isfinite(iter_gain_pct):
        out['iter_gain_pct'] = f"{iter_gain_pct:.12g}"
    return out


def _pick_best_row(rows: List[Dict[str, Any]], objective_field: str) -> Dict[str, Any]:
    def key(row: Dict[str, Any]) -> Tuple[int, float, str]:
        obj = _to_float(row.get(objective_field), default=float('inf'))
        return (0 if math.isfinite(obj) else 1, obj, str(row.get('params_json', '') or ''))
    return min(rows, key=key)


def _summarize_group(rows: List[Dict[str, Any]], *, objective_field: str) -> Dict[str, Any]:
    best = _pick_best_row(rows, objective_field)
    objectives = [_to_float(r.get(objective_field), default=float('nan')) for r in rows]
    totals = [_to_float(r.get('total'), default=float('nan')) for r in rows]
    iter_gains = [_to_float(r.get('iter_gain_s'), default=float('nan')) for r in rows]
    iter_gain_pcts = [_to_float(r.get('iter_gain_pct'), default=float('nan')) for r in rows]
    nd_initials = [_to_float(r.get('nd_initial'), default=float('nan')) for r in rows]
    nd_bests = [_to_float(r.get('nd_best'), default=float('nan')) for r in rows]
    finite_iter_gains = [x for x in iter_gains if math.isfinite(x)]
    positive_iter = [x for x in finite_iter_gains if x > 0.0]

    return {
        'runs': len(rows),
        'best_objective': _to_float(best.get(objective_field), default=float('nan')),
        'mean_objective': _mean(objectives),
        'median_objective': _median(objectives),
        'best_prefill': _to_float(best.get('prefill'), default=float('nan')),
        'best_decode': _to_float(best.get('decode'), default=float('nan')),
        'best_total': _to_float(best.get('total'), default=float('nan')),
        'mean_total': _mean(totals),
        'best_search_format': str(best.get('search_format', '') or ''),
        'best_pass': _to_int(best.get('best_pass'), default=-1),
        'best_nd_initial': _to_float(best.get('nd_initial'), default=float('nan')),
        'best_nd_best': _to_float(best.get('nd_best'), default=float('nan')),
        'mean_nd_initial': _mean(nd_initials),
        'mean_nd_best': _mean(nd_bests),
        'best_iter_gain_s': _to_float(best.get('iter_gain_s'), default=float('nan')),
        'best_iter_gain_pct': _to_float(best.get('iter_gain_pct'), default=float('nan')),
        'mean_iter_gain_s': _mean(iter_gains),
        'mean_iter_gain_pct': _mean(iter_gain_pcts),
        'max_iter_gain_s': max(finite_iter_gains) if finite_iter_gains else float('nan'),
        'positive_iter_gain_runs': len(positive_iter),
        'positive_iter_gain_fraction': (len(positive_iter) / len(finite_iter_gains)) if finite_iter_gains else float('nan'),
        'best_log_path': str(best.get('log_path', '') or ''),
        'best_weight_suggest_al_log_path': str(best.get('weight_suggest_al_log_path', '') or ''),
        'best_generated_config_json': str(best.get('generated_config_json', '') or ''),
        'best_weight_format_json': str(best.get('weight_format_json', '') or ''),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser(description='Summarize weight-suggest sweeps over WEIGHT_LOCAL_LOAD_OVERLAP_RATIO.')
    ap.add_argument('--results-csv', required=True, help='Path to results.csv produced by sweep_weight_suggest_params.py')
    ap.add_argument('--outdir', default='', help='Output directory. Defaults to <results_dir>/ratio_summary')
    ap.add_argument('--objective-field', default='objective', choices=['objective', 'total', 'decode', 'prefill'], help='Field used to choose the best row inside each group')
    ap.add_argument('--group-by', default='model,prefill_len,decode_len,batch,hardware_json,tp_qkv,tp_ffn,weight_local_load_overlap_ratio', help='Comma-separated keys for detailed grouped summary')
    args = ap.parse_args()

    results_csv = Path(args.results_csv).resolve()
    if not results_csv.exists():
        raise SystemExit(f'results.csv not found: {results_csv}')
    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else (results_csv.parent / 'ratio_summary').resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with results_csv.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        original_fieldnames = list(reader.fieldnames or [])
        rows = [_enrich_row(dict(row)) for row in reader]

    if not rows:
        raise SystemExit(f'no rows found in {results_csv}')

    detail_fieldnames = list(original_fieldnames)
    for extra in ('iter_gain_s', 'iter_gain_pct'):
        if extra not in detail_fieldnames:
            detail_fieldnames.append(extra)
    _write_csv(outdir / 'detail_enriched.csv', rows, detail_fieldnames)

    group_keys = [x.strip() for x in str(args.group_by).split(',') if x.strip()]
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_stable_group_value(row, k) for k in group_keys)
        grouped[key].append(row)

    group_rows: List[Dict[str, Any]] = []
    for key, subrows in grouped.items():
        payload = {group_keys[i]: key[i] for i in range(len(group_keys))}
        payload.update(_summarize_group(subrows, objective_field=args.objective_field))
        group_rows.append(payload)

    group_rows.sort(key=lambda r: tuple(
        _ratio_sort_key(r[k]) if k == 'weight_local_load_overlap_ratio' else str(r.get(k, '') or '')
        for k in group_keys
    ))

    group_fieldnames = list(group_keys) + [
        'runs', 'best_objective', 'mean_objective', 'median_objective',
        'best_prefill', 'best_decode', 'best_total', 'mean_total',
        'best_search_format', 'best_pass',
        'best_nd_initial', 'best_nd_best', 'mean_nd_initial', 'mean_nd_best',
        'best_iter_gain_s', 'best_iter_gain_pct', 'mean_iter_gain_s', 'mean_iter_gain_pct',
        'max_iter_gain_s', 'positive_iter_gain_runs', 'positive_iter_gain_fraction',
        'best_log_path', 'best_generated_config_json', 'best_weight_format_json',
        'best_log_path', 'best_weight_suggest_al_log_path', 'best_generated_config_json', 'best_weight_format_json',
    ]
    _write_csv(outdir / 'group_summary.csv', group_rows, group_fieldnames)

    ratio_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ratio_groups[_stable_group_value(row, 'weight_local_load_overlap_ratio')].append(row)

    ratio_rows: List[Dict[str, Any]] = []
    best_by_ratio: Dict[str, Dict[str, Any]] = {}
    for ratio, subrows in ratio_groups.items():
        payload = {'weight_local_load_overlap_ratio': ratio}
        summary = _summarize_group(subrows, objective_field=args.objective_field)
        payload.update(summary)
        ratio_rows.append(payload)
        best_by_ratio[ratio] = {
            'best_objective': summary['best_objective'],
            'best_total': summary['best_total'],
            'best_iter_gain_s': summary['best_iter_gain_s'],
            'best_iter_gain_pct': summary['best_iter_gain_pct'],
            'best_log_path': summary['best_log_path'],
            'best_weight_suggest_al_log_path': summary['best_weight_suggest_al_log_path'],
            'best_generated_config_json': summary['best_generated_config_json'],
        }

    ratio_rows.sort(key=lambda r: _ratio_sort_key(r.get('weight_local_load_overlap_ratio', '')))
    ratio_fieldnames = [
        'weight_local_load_overlap_ratio',
        'runs', 'best_objective', 'mean_objective', 'median_objective',
        'best_prefill', 'best_decode', 'best_total', 'mean_total',
        'best_search_format', 'best_pass',
        'best_nd_initial', 'best_nd_best', 'mean_nd_initial', 'mean_nd_best',
        'best_iter_gain_s', 'best_iter_gain_pct', 'mean_iter_gain_s', 'mean_iter_gain_pct',
        'max_iter_gain_s', 'positive_iter_gain_runs', 'positive_iter_gain_fraction',
        'best_log_path', 'best_generated_config_json', 'best_weight_format_json',
        'best_log_path', 'best_weight_suggest_al_log_path', 'best_generated_config_json', 'best_weight_format_json',
    ]
    _write_csv(outdir / 'ratio_summary.csv', ratio_rows, ratio_fieldnames)

    summary_json = {
        'results_csv': str(results_csv),
        'detail_enriched_csv': str(outdir / 'detail_enriched.csv'),
        'group_summary_csv': str(outdir / 'group_summary.csv'),
        'ratio_summary_csv': str(outdir / 'ratio_summary.csv'),
        'objective_field': args.objective_field,
        'group_by': group_keys,
        'row_count': len(rows),
        'ratio_count': len(ratio_rows),
        'best_by_ratio': best_by_ratio,
    }
    with (outdir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f'[summary] results_csv={results_csv}')
    print(f'[summary] detail_enriched_csv={outdir / "detail_enriched.csv"}')
    print(f'[summary] group_summary_csv={outdir / "group_summary.csv"}')
    print(f'[summary] ratio_summary_csv={outdir / "ratio_summary.csv"}')
    print(f'[summary] summary_json={outdir / "summary.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
