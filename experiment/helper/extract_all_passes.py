#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
递归扫描指定目录下的 JSON 文件，自动识别 all passes 结果文件，提取：
- model_family
- model_variant
- dtype
- batch
- prefill_len
- decode_len
- weight_format_comparison 中目标格式（默认 ND）的 initial_times.{prefill, decode, total}
- 目标格式（默认 ND）的 best_total_s（即搜索后的最终时间）

也支持可选展开所有 format row（如 NZ / PIM-OPT）的初始时间和 best_total_s --include-all-rows。

示例：
    python3 extract_all_passes.py /path/to/search -o summary.csv
    python3 extract_all_passes.py ../../algorithms/output/ws_hpc/shards/8w -o ../../algorithms/output/ws_hpc/shards/8w/summary.csv --include-all-rows
    python3 extract_all_passes.py /path/to/search -o summary.csv --target-format ND
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


NAME_REGEX = re.compile(r"all[\s_\-]*passes", re.IGNORECASE)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def normalize_format_name(fmt: str) -> str:
    fmt = (fmt or "UNKNOWN").strip()
    fmt = re.sub(r"[^0-9A-Za-z]+", "_", fmt)
    fmt = re.sub(r"_+", "_", fmt).strip("_")
    return fmt.upper() or "UNKNOWN"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def looks_like_all_passes(path: Path, data: Any) -> bool:
    if NAME_REGEX.search(path.name):
        return True
    if not isinstance(data, dict):
        return False
    wfc = data.get("weight_format_comparison")
    if not isinstance(wfc, dict):
        return False
    rows = wfc.get("rows")
    return isinstance(rows, list)


def merge_config(top_cfg: Any, inner_cfg: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if isinstance(inner_cfg, dict):
        merged.update(inner_cfg)
    if isinstance(top_cfg, dict):
        merged.update(top_cfg)
    return merged


def first_row_by_format(rows: Iterable[Dict[str, Any]], fmt: Optional[str]) -> Optional[Dict[str, Any]]:
    if not fmt:
        return None
    fmt_upper = str(fmt).strip().upper()
    for row in rows:
        if str(row.get("format", "")).strip().upper() == fmt_upper:
            return row
    return None


def extract_one(
    path: Path,
    data: Dict[str, Any],
    target_format: str = "ND",
    include_all_rows: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(data, dict):
        return None, "top-level JSON is not an object"

    top_cfg = data.get("config")
    wfc = data.get("weight_format_comparison")
    if not isinstance(wfc, dict):
        return None, "missing weight_format_comparison"

    rows = wfc.get("rows")
    if not isinstance(rows, list):
        return None, "weight_format_comparison.rows is not a list"

    cfg = merge_config(top_cfg, wfc.get("config"))

    search_format = wfc.get("search_format") or cfg.get("search_format")
    compare_only_formats = wfc.get("compare_only_formats") or cfg.get("compare_only_formats") or []
    if not isinstance(compare_only_formats, list):
        compare_only_formats = [compare_only_formats]

    # 优先用命令行指定的格式；若未找到，再回退到 search_format
    target_row = first_row_by_format(rows, target_format)
    if target_row is None and search_format:
        target_row = first_row_by_format(rows, search_format)

    initial_times = target_row.get("initial_times", {}) if isinstance(target_row, dict) else {}
    if not isinstance(initial_times, dict):
        initial_times = {}

    init_prefill = safe_float(initial_times.get("prefill"))
    init_decode = safe_float(initial_times.get("decode"))
    init_total = safe_float(initial_times.get("total"))

    final_total = None
    best_pass = None
    actual_target_format = target_format

    if isinstance(target_row, dict):
        final_total = safe_float(target_row.get("best_total_s"))
        best_pass = target_row.get("best_pass")
        actual_target_format = target_row.get("format") or target_format
    else:
        # 如果找不到目标 row，但顶层有 best_total_s，仍尝试保留
        final_total = safe_float(wfc.get("best_total_s"))
        best_pass = wfc.get("best_pass")

    improvement_s = None
    improvement_pct = None
    if init_total is not None and final_total is not None:
        improvement_s = init_total - final_total
        if init_total != 0:
            improvement_pct = improvement_s / init_total * 100.0

    result: Dict[str, Any] = {
        "json_path": str(path),
        "model_family": cfg.get("model_family"),
        "model_variant": cfg.get("model_variant"),
        "dtype": cfg.get("dtype"),
        "batch": cfg.get("batch"),
        "prefill_len": cfg.get("prefill_len"),
        "decode_len": cfg.get("decode_len"),
        "search_format": search_format,
        "compare_only_formats": ",".join(map(str, compare_only_formats)),
        f"{normalize_format_name(target_format).lower()}_row_found": target_row is not None,
        f"{normalize_format_name(target_format).lower()}_actual_row_format": actual_target_format,
        f"{normalize_format_name(target_format).lower()}_initial_prefill_s": init_prefill,
        f"{normalize_format_name(target_format).lower()}_initial_decode_s": init_decode,
        f"{normalize_format_name(target_format).lower()}_initial_total_s": init_total,
        f"{normalize_format_name(target_format).lower()}_final_total_s": final_total,
        f"{normalize_format_name(target_format).lower()}_best_pass": best_pass,
        f"{normalize_format_name(target_format).lower()}_improvement_s": improvement_s,
        f"{normalize_format_name(target_format).lower()}_improvement_pct": improvement_pct,
    }

    if include_all_rows:
        for row in rows:
            if not isinstance(row, dict):
                continue
            fmt = normalize_format_name(str(row.get("format", "UNKNOWN")))
            row_init = row.get("initial_times", {})
            if not isinstance(row_init, dict):
                row_init = {}
            result[f"{fmt.lower()}_initial_prefill_s"] = safe_float(row_init.get("prefill"))
            result[f"{fmt.lower()}_initial_decode_s"] = safe_float(row_init.get("decode"))
            result[f"{fmt.lower()}_initial_total_s"] = safe_float(row_init.get("total"))
            result[f"{fmt.lower()}_best_total_s"] = safe_float(row.get("best_total_s"))
            result[f"{fmt.lower()}_role"] = row.get("role")
            result[f"{fmt.lower()}_search_executed"] = row.get("search_executed")
            result[f"{fmt.lower()}_best_pass"] = row.get("best_pass")

    return result, None


def scan_json_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.json"):
        if path.is_file():
            yield path


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        # 即使为空，也写一个空文件，避免调用方误判
        out_path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_errors_csv(errors: List[Dict[str, Any]], out_path: Path) -> None:
    if not errors:
        return
    fieldnames = ["json_path", "error"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description="提取 all passes JSON 中的模型字段和 ND（或指定 format）的时间信息")
    parser.add_argument("root", type=Path, help="要扫描的根目录")
    parser.add_argument("-o", "--output", type=Path, default=Path("all_passes_summary.csv"), help="输出 CSV 文件路径")
    parser.add_argument("--target-format", default="ND", help="要提取的目标 format，默认 ND")
    parser.add_argument("--include-all-rows", action="store_true", help="额外展开所有 format row 的初始时间和 best_total_s")
    parser.add_argument("--filename-only", action="store_true", help="仅根据文件名匹配 all_passes，不按内容结构识别")
    parser.add_argument("--sort", action="store_true", help="按模型字段排序输出")
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists():
        print(f"[ERROR] 路径不存在: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"[ERROR] 不是目录: {root}", file=sys.stderr)
        return 2

    ok_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    total_json = 0
    matched_files = 0

    for path in scan_json_files(root):
        total_json += 1
        try:
            data = load_json(path)
        except Exception as e:  # noqa: BLE001
            # 只记录疑似目标文件的加载错误，避免普通 json 太多导致错误表噪音大
            name_likely = bool(NAME_REGEX.search(path.name))
            if name_likely:
                errors.append({"json_path": str(path), "error": f"json load failed: {e}"})
            continue

        if args.filename_only:
            is_target = bool(NAME_REGEX.search(path.name))
        else:
            is_target = looks_like_all_passes(path, data)

        if not is_target:
            continue

        matched_files += 1
        row, err = extract_one(
            path=path,
            data=data,
            target_format=args.target_format,
            include_all_rows=args.include_all_rows,
        )
        if err:
            errors.append({"json_path": str(path), "error": err})
            continue
        if row is not None:
            ok_rows.append(row)

    if args.sort and ok_rows:
        ok_rows.sort(
            key=lambda x: (
                str(x.get("model_family") or ""),
                str(x.get("model_variant") or ""),
                str(x.get("dtype") or ""),
                float(x.get("batch") or -1),
                float(x.get("prefill_len") or -1),
                float(x.get("decode_len") or -1),
                str(x.get("json_path") or ""),
            )
        )

    write_csv(ok_rows, args.output)
    error_path = args.output.with_name(args.output.stem + "_errors.csv")
    write_errors_csv(errors, error_path)

    print(f"[DONE] 扫描 JSON 文件数: {total_json}")
    print(f"[DONE] 识别为 all passes 的文件数: {matched_files}")
    print(f"[DONE] 成功提取条数: {len(ok_rows)}")
    print(f"[DONE] 输出 CSV: {args.output}")
    if errors:
        print(f"[WARN] 有 {len(errors)} 个文件提取失败，错误明细: {error_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
