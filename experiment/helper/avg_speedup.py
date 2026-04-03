#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
递归扫描 baseline_compare_<prefill>x<decode>.json，
找出 heft / hefthint 中更优的那个，并计算其相对 pd 的加速比。

支持：
1. 输入可以是 exp1 层、某个 model 层，甚至单个 baseline_compare_*.json 文件。
2. 可按 prefill length / decode length 过滤。
3. 默认按 total_time_s 选择 heft / hefthint 里的优胜者。
4. 输出该优胜者相对 pd 的 prefill / decode / total 三个加速比。
5. 可选输出 phase-wise 最优（prefill / decode / total 分别选 heft 或 hefthint 中更快的那个）。

示例：
    python avg_speedup.py /lustre/home/2501111916/workspace/DOPS_0330_merge/TriForm/algorithms/output/exp1/hw_hardware_1npu_2aim/sst2_rst2
    python avg_speedup.py /path/to/llama_13b_fp16_b1_s8 --prefill 128 --decode 128
    python compare_heft_best_vs_pd.py /path/to/exp1 --choose-by total --show-phase-best
    python compare_heft_best_vs_pd.py /path/to/exp1 --format json > result.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FILE_RE = re.compile(r"baseline_compare_(\d+)x(\d+)\.json$", re.IGNORECASE)
TIME_KEYS = {
    "prefill": "prefill_time_s",
    "decode": "decode_time_s",
    "total": "total_time_s",
}
TARGET_POLICIES = {"pd", "heft", "hefthint"}


def normalize_policy(name: Any) -> str:
    s = str(name).strip().lower()
    if ":" in s:
        s = s.split(":")[-1]
    if "/" in s:
        s = s.split("/")[-1]
    return s


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def speedup_against(base: Optional[float], candidate: Optional[float]) -> Optional[float]:
    """返回 base / candidate。一般这里 base 是 pd 时间。"""
    if base is None or candidate is None:
        return None
    if candidate == 0:
        if base == 0:
            return 1.0
        return math.inf
    return base / candidate


def fmt_num(value: Optional[float], digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "NA"
    if math.isinf(value):
        return f"inf{suffix}"
    return f"{value:.{digits}f}{suffix}"


def read_json(path: Path) -> Dict[str, Any]:
    encodings = ["utf-8", "utf-8-sig"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"读取 JSON 失败: {path} ({last_err})")


def iter_candidate_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if FILE_RE.search(root.name):
            yield root
        return

    for p in root.rglob("baseline_compare_*.json"):
        if p.is_file() and FILE_RE.search(p.name):
            yield p


def extract_lengths(path: Path, data: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    m = FILE_RE.search(path.name)
    if m:
        return int(m.group(1)), int(m.group(2))

    config = data.get("config", {}) if isinstance(data, dict) else {}
    pf = config.get("prefill_len")
    dec = config.get("decode_len")
    try:
        pf_int = int(pf) if pf is not None else None
        dec_int = int(dec) if dec is not None else None
        return pf_int, dec_int
    except (TypeError, ValueError):
        return None, None



def better_record(existing: Optional[Dict[str, Any]], new_row: Dict[str, Any]) -> bool:
    """如果同一个 policy 出现多次，优先保留 total_time 更小的一条。"""
    if existing is None:
        return True
    old_total = safe_float(existing.get("total_time_s"))
    new_total = safe_float(new_row.get("total_time_s"))
    if old_total is None:
        return True
    if new_total is None:
        return False
    return new_total < old_total


def collect_policy_rows(results: Any) -> Dict[str, Dict[str, Any]]:
    picked: Dict[str, Dict[str, Any]] = {}
    if not isinstance(results, list):
        return picked

    for row in results:
        if not isinstance(row, dict):
            continue
        policy = normalize_policy(row.get("policy", ""))
        if policy in TARGET_POLICIES and better_record(picked.get(policy), row):
            picked[policy] = row
    return picked


def choose_fastest_policy(rows: Dict[str, Dict[str, Any]], metric: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    time_key = TIME_KEYS[metric]
    candidates = []
    for name in ("heft", "hefthint"):
        row = rows.get(name)
        if row is None:
            continue
        t = safe_float(row.get(time_key))
        if t is not None:
            candidates.append((name, t, row))
    if not candidates:
        return None, None
    name, _, row = min(candidates, key=lambda x: x[1])
    return name, row


def build_record(path: Path, root: Path, choose_by: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        data = read_json(path)
    except Exception as e:  # noqa: BLE001
        return None, str(e)

    prefill_len, decode_len = extract_lengths(path, data)
    rows = collect_policy_rows(data.get("results"))

    missing = [name for name in ("pd", "heft", "hefthint") if name not in rows]
    if missing:
        return None, f"缺少 policy: {', '.join(missing)}"

    pd_row = rows["pd"]
    chosen_name, chosen_row = choose_fastest_policy(rows, choose_by)
    if chosen_name is None or chosen_row is None:
        return None, f"无法按 {choose_by} 选出 heft/hefthint 中更快者"

    root_is_dir = root.is_dir()
    try:
        rel_path = str(path.relative_to(root)) if root_is_dir else path.name
    except ValueError:
        rel_path = str(path)

    pd_prefill = safe_float(pd_row.get("prefill_time_s"))
    pd_decode = safe_float(pd_row.get("decode_time_s"))
    pd_total = safe_float(pd_row.get("total_time_s"))

    ch_prefill = safe_float(chosen_row.get("prefill_time_s"))
    ch_decode = safe_float(chosen_row.get("decode_time_s"))
    ch_total = safe_float(chosen_row.get("total_time_s"))

    best_prefill_name, best_prefill_row = choose_fastest_policy(rows, "prefill")
    best_decode_name, best_decode_row = choose_fastest_policy(rows, "decode")
    best_total_name, best_total_row = choose_fastest_policy(rows, "total")

    record: Dict[str, Any] = {
        "path": str(path),
        "relative_path": rel_path,
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "choose_by": choose_by,
        "winner": chosen_name,
        "winner_times_s": {
            "prefill": ch_prefill,
            "decode": ch_decode,
            "total": ch_total,
        },
        "pd_times_s": {
            "prefill": pd_prefill,
            "decode": pd_decode,
            "total": pd_total,
        },
        "winner_speedup_vs_pd": {
            "prefill": speedup_against(pd_prefill, ch_prefill),
            "decode": speedup_against(pd_decode, ch_decode),
            "total": speedup_against(pd_total, ch_total),
        },
        "phase_best": {
            "prefill": {
                "winner": best_prefill_name,
                "time_s": safe_float(best_prefill_row.get("prefill_time_s")) if best_prefill_row else None,
                "speedup_vs_pd": speedup_against(
                    pd_prefill,
                    safe_float(best_prefill_row.get("prefill_time_s")) if best_prefill_row else None,
                ),
            },
            "decode": {
                "winner": best_decode_name,
                "time_s": safe_float(best_decode_row.get("decode_time_s")) if best_decode_row else None,
                "speedup_vs_pd": speedup_against(
                    pd_decode,
                    safe_float(best_decode_row.get("decode_time_s")) if best_decode_row else None,
                ),
            },
            "total": {
                "winner": best_total_name,
                "time_s": safe_float(best_total_row.get("total_time_s")) if best_total_row else None,
                "speedup_vs_pd": speedup_against(
                    pd_total,
                    safe_float(best_total_row.get("total_time_s")) if best_total_row else None,
                ),
            },
        },
        "raw_candidates_s": {
            "heft": {
                "prefill": safe_float(rows["heft"].get("prefill_time_s")),
                "decode": safe_float(rows["heft"].get("decode_time_s")),
                "total": safe_float(rows["heft"].get("total_time_s")),
            },
            "hefthint": {
                "prefill": safe_float(rows["hefthint"].get("prefill_time_s")),
                "decode": safe_float(rows["hefthint"].get("decode_time_s")),
                "total": safe_float(rows["hefthint"].get("total_time_s")),
            },
        },
    }
    return record, None



def filter_by_lengths(records: List[Dict[str, Any]], prefill: Optional[int], decode: Optional[int]) -> List[Dict[str, Any]]:
    filtered = []
    for r in records:
        if prefill is not None and r.get("prefill_len") != prefill:
            continue
        if decode is not None and r.get("decode_len") != decode:
            continue
        filtered.append(r)
    return filtered


def sort_records(records: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    if sort_by == "speedup_total":
        return sorted(
            records,
            key=lambda r: (
                -(r.get("winner_speedup_vs_pd", {}).get("total") or float("-inf")),
                r.get("relative_path", ""),
            ),
        )
    if sort_by == "prefill_decode":
        return sorted(
            records,
            key=lambda r: (
                r.get("prefill_len") if r.get("prefill_len") is not None else float("inf"),
                r.get("decode_len") if r.get("decode_len") is not None else float("inf"),
                r.get("relative_path", ""),
            ),
        )
    return sorted(records, key=lambda r: r.get("relative_path", ""))



def as_json_ready(records: List[Dict[str, Any]]) -> str:
    return json.dumps(records, ensure_ascii=False, indent=2)



def render_table(records: List[Dict[str, Any]]) -> str:
    headers = [
        "prefill",
        "decode",
        "winner",
        "spd_pf",
        "spd_dec",
        "spd_total",
        "relative_path",
    ]

    rows: List[List[str]] = []
    for r in records:
        spd = r.get("winner_speedup_vs_pd", {})
        rows.append(
            [
                str(r.get("prefill_len", "NA")),
                str(r.get("decode_len", "NA")),
                str(r.get("winner", "NA")),
                fmt_num(spd.get("prefill"), suffix="x"),
                fmt_num(spd.get("decode"), suffix="x"),
                fmt_num(spd.get("total"), suffix="x"),
                str(r.get("relative_path", "")),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: List[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)



def render_verbose(records: List[Dict[str, Any]]) -> str:
    chunks = []
    for r in records:
        spd = r.get("winner_speedup_vs_pd", {})
        phase_best = r.get("phase_best", {})
        chunks.append(
            "\n".join(
                [
                    f"[{r.get('relative_path')}]",
                    f"  prefill_len={r.get('prefill_len')}  decode_len={r.get('decode_len')}  choose_by={r.get('choose_by')}",
                    f"  winner={r.get('winner')}  speedup_vs_pd: prefill={fmt_num(spd.get('prefill'), suffix='x')}, decode={fmt_num(spd.get('decode'), suffix='x')}, total={fmt_num(spd.get('total'), suffix='x')}",
                    (
                        "  phase_best: "
                        f"prefill={phase_best.get('prefill', {}).get('winner')}({fmt_num(phase_best.get('prefill', {}).get('speedup_vs_pd'), suffix='x')}), "
                        f"decode={phase_best.get('decode', {}).get('winner')}({fmt_num(phase_best.get('decode', {}).get('speedup_vs_pd'), suffix='x')}), "
                        f"total={phase_best.get('total', {}).get('winner')}({fmt_num(phase_best.get('total', {}).get('speedup_vs_pd'), suffix='x')})"
                    ),
                ]
            )
        )
    return "\n\n".join(chunks)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "递归扫描 baseline_compare_<prefill>x<decode>.json，"
            "在 heft / hefthint 中选出更优者，并计算其相对 pd 的 prefill / decode / total 加速比。"
        )
    )
    parser.add_argument(
        "input_path",
        help="输入路径：可以是 exp1 层目录、某个 model 层目录，或者单个 baseline_compare_*.json 文件。",
    )
    parser.add_argument("--prefill", type=int, default=None, help="按 prefill length 过滤。")
    parser.add_argument("--decode", type=int, default=None, help="按 decode length 过滤。")
    parser.add_argument(
        "--choose-by",
        choices=["prefill", "decode", "total"],
        default="total",
        help="在 heft / hefthint 中按哪个时间指标选优，默认 total。",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="输出格式，默认 table。",
    )
    parser.add_argument(
        "--show-phase-best",
        action="store_true",
        help="额外输出 prefill / decode / total 三个阶段各自最优的算法及其相对 pd 的加速比。",
    )
    parser.add_argument(
        "--sort-by",
        choices=["path", "prefill_decode", "speedup_total"],
        default="path",
        help="结果排序方式，默认按路径。",
    )
    return parser



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root = Path(args.input_path).expanduser().resolve()
    if not root.exists():
        print(f"错误：路径不存在 -> {root}", file=sys.stderr)
        return 1

    candidate_files = list(iter_candidate_files(root))
    if not candidate_files:
        print("没有找到 baseline_compare_*.json 文件。", file=sys.stderr)
        return 1

    records: List[Dict[str, Any]] = []
    errors: List[Tuple[str, str]] = []

    for path in candidate_files:
        record, err = build_record(path, root, args.choose_by)
        if err is not None:
            errors.append((str(path), err))
            continue
        if record is not None:
            records.append(record)

    records = filter_by_lengths(records, args.prefill, args.decode)
    records = sort_records(records, args.sort_by)

    if not records:
        print("找到了 baseline_compare_*.json，但按当前过滤条件没有可用结果。", file=sys.stderr)
        if errors:
            print("\n被跳过的文件：", file=sys.stderr)
            for p, err in errors:
                print(f"  - {p}: {err}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(as_json_ready(records))
    else:
        print(render_table(records))
        if args.show_phase_best:
            print("\n详细信息：")
            print(render_verbose(records))

        best_total = max(
            records,
            key=lambda r: (r.get("winner_speedup_vs_pd", {}).get("total") or float("-inf")),
        )
        best_total_spd = best_total.get("winner_speedup_vs_pd", {}).get("total")
        print(
            "\n"
            f"总计：{len(records)} 个匹配结果；"
            f"最高 total speedup = {fmt_num(best_total_spd, suffix='x')} @ {best_total.get('relative_path')}"
        )

    if errors:
        print("\n以下文件被跳过：", file=sys.stderr)
        for p, err in errors:
            print(f"  - {p}: {err}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
