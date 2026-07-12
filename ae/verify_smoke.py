#!/usr/bin/env python3
"""Validate the reviewer-facing DOPS analytical smoke test."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


EXPECTED_POLICIES = {"PD", "Bifocal"}
EXPECTED_CONFIG = {
    "model_family": "qwen",
    "model_variant": "1.8b",
    "batch": 1,
    "prefill_len": 8,
    "decode_len": 4,
    "npu_backend": "fast",
    "pim_fast_mode": True,
    "scheduler_seed": 0,
}


def fail(message: str) -> None:
    raise SystemExit(f"[AE] FAIL: {message}")


def exactly_one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        fail(f"expected exactly one {label}, found {len(paths)}")
    return paths[0]


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - reviewer-facing error path
        fail(f"cannot parse {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"{path} does not contain a JSON object")
    return payload


def validate_positive_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        fail(f"{label} is not numeric: {value!r}")
    if not math.isfinite(number) or number <= 0.0:
        fail(f"{label} must be finite and positive, got {number!r}")
    return number


def validate_trace(path: Path, label: str, required_columns: set[str]) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = required_columns - columns
            if missing:
                fail(f"{label} is missing CSV columns: {sorted(missing)}")

            phases: set[str] = set()
            rows = 0
            for row in reader:
                rows += 1
                phases.add(str(row.get("phase", "")).strip())
                duration_raw = row.get("duration")
                try:
                    duration = float(duration_raw or 0.0)
                except (TypeError, ValueError):
                    fail(f"{label} has a non-numeric duration at row {rows + 1}")
                if not math.isfinite(duration) or duration < 0.0:
                    fail(f"{label} has an invalid duration at row {rows + 1}")
    except OSError as exc:
        fail(f"cannot read {label}: {exc}")

    if rows < 10:
        fail(f"{label} has only {rows} data rows")
    if not {"prefill", "decode"}.issubset(phases):
        fail(f"{label} does not contain both prefill and decode rows")
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: verify_smoke.py OUTPUT_ROOT")

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        fail(f"output directory does not exist: {root}")

    combined = exactly_one(list(root.rglob("baseline_compare_*.json")), "combined comparison JSON")
    payload = load_json(combined)
    config = payload.get("config")
    if not isinstance(config, dict):
        fail(f"{combined} has no config object")
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            fail(f"combined config {key}={config.get(key)!r}, expected {expected!r}")

    results = payload.get("results")
    if not isinstance(results, list):
        fail(f"{combined} has no results list")

    observed: dict[str, dict[str, float]] = {}
    for row in results:
        if not isinstance(row, dict):
            fail("combined results contains a non-object entry")
        policy = str(row.get("policy", "")).strip()
        if policy.startswith("algo:"):
            policy = policy.removeprefix("algo:")
        if policy not in EXPECTED_POLICIES:
            fail(f"unexpected policy in combined results: {policy!r}")
        if policy in observed:
            fail(f"duplicate policy in combined results: {policy}")
        observed[policy] = {
            key: validate_positive_number(row.get(key), f"{policy}.{key}")
            for key in ("prefill_time_s", "decode_time_s", "total_time_s")
        }

    if set(observed) != EXPECTED_POLICIES:
        fail(f"expected policies {sorted(EXPECTED_POLICIES)}, observed {sorted(observed)}")

    trace_rows: dict[str, dict[str, int]] = {}
    for policy in sorted(EXPECTED_POLICIES):
        policy_dir = exactly_one(
            [path for path in root.rglob(f"algo_{policy}") if path.is_dir()],
            f"algo_{policy} directory",
        )
        summary_path = exactly_one(list(policy_dir.glob("best_summary_*.json")), f"{policy} best summary")
        summary = load_json(summary_path)
        if summary.get("policy") != f"algo:{policy}":
            fail(f"{summary_path} has unexpected policy {summary.get('policy')!r}")
        summary_config = summary.get("config")
        if not isinstance(summary_config, dict):
            fail(f"{summary_path} has no config object")
        for key in ("batch", "prefill_len", "decode_len"):
            if summary_config.get(key) != EXPECTED_CONFIG[key]:
                fail(f"{summary_path} has unexpected config value for {key}")

        best_times = summary.get("best_times")
        if not isinstance(best_times, dict):
            fail(f"{summary_path} has no best_times object")
        for summary_key, combined_key in (
            ("prefill", "prefill_time_s"),
            ("decode", "decode_time_s"),
            ("total", "total_time_s"),
        ):
            value = validate_positive_number(best_times.get(summary_key), f"{policy}.{summary_key}")
            if not math.isclose(value, observed[policy][combined_key], rel_tol=1e-12, abs_tol=1e-15):
                fail(f"{policy} summary and combined metrics disagree for {summary_key}")

        prefill_schedule = summary.get("prefill_schedule")
        decode_steps = summary.get("decode_steps")
        if not isinstance(prefill_schedule, list) or not prefill_schedule:
            fail(f"{summary_path} has no prefill schedule")
        if not isinstance(decode_steps, list) or not decode_steps:
            fail(f"{summary_path} has no decode steps")

        ops_trace = exactly_one(list(policy_dir.glob("*_ops_trace.csv")), f"{policy} operator trace")
        comms_trace = exactly_one(list(policy_dir.glob("*_comms_trace.csv")), f"{policy} communication trace")
        trace_rows[policy] = {
            "operator": validate_trace(
                ops_trace,
                f"{policy} operator trace",
                {"phase", "node_id", "op", "device", "device_type", "start", "end", "duration"},
            ),
            "communication": validate_trace(
                comms_trace,
                f"{policy} communication trace",
                {"phase", "src", "src_type", "dst", "dst_type", "bytes", "start", "end", "duration"},
            ),
        }

    print(
        json.dumps(
            {"comparison": str(combined), "metrics": observed, "trace_rows": trace_rows},
            indent=2,
            sort_keys=True,
        )
    )
    print("[AE] PASS: analytical fast-mode smoke test produced valid PD and Bifocal artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
