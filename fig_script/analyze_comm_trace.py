#!/usr/bin/env python3
"""Utility to analyze communication traces produced by TriForm.

The tool surfaces which data transfers dominate the traffic and highlights
useful aggregates (by phase, tag, and device pair).
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

Number = float | int


def _format_bytes(num_bytes: int) -> str:
    """Return a human friendly string for byte counts."""
    suffixes = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for suffix in suffixes:
        if abs(value) < 1024.0 or suffix == suffixes[-1]:
            return f"{value:,.2f} {suffix}"
        value /= 1024.0
    return f"{value:,.2f} TiB"


def _format_seconds(seconds: float) -> str:
    """Format seconds with microsecond precision."""
    return f"{seconds * 1e6:,.1f} µs" if seconds < 0.1 else f"{seconds:,.4f} s"


def load_comm_rows(path: Path) -> Iterable[Dict[str, Number]]:
    """Yield parsed rows from a comms trace CSV."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                yield {
                    "phase": row["phase"],
                    "src": row["src"],
                    "src_type": row["src_type"],
                    "dst": row["dst"],
                    "dst_type": row["dst_type"],
                    "bytes": int(row["bytes"]),
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "duration": float(row["duration"]),
                    "tag": row.get("tag", ""),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Failed to parse row: {row}") from exc


def summarize_comms(rows: Iterable[Dict[str, Number]]) -> None:
    totals_by_tag: Dict[str, int] = defaultdict(int)
    totals_by_phase_tag: Dict[Tuple[str, str], int] = defaultdict(int)
    totals_by_route: Dict[Tuple[str, str], int] = defaultdict(int)
    totals_by_type: Dict[Tuple[str, str], int] = defaultdict(int)

    durations_by_tag: Dict[str, list[float]] = defaultdict(list)
    bytes_per_event: list[int] = []

    total_bytes = 0
    total_events = 0

    top_event = None

    for row in rows:
        total_events += 1
        total_bytes += row["bytes"]
        bytes_per_event.append(row["bytes"])

        tag = row["tag"]
        phase = row["phase"]
        src = row["src"]
        dst = row["dst"]
        src_type = row["src_type"]
        dst_type = row["dst_type"]
        duration = row["duration"]

        totals_by_tag[tag] += row["bytes"]
        totals_by_phase_tag[(phase, tag)] += row["bytes"]
        totals_by_route[(src, dst)] += row["bytes"]
        totals_by_type[(src_type, dst_type)] += row["bytes"]

        durations_by_tag[tag].append(duration)

        if top_event is None or row["bytes"] > top_event["bytes"]:
            top_event = row

    if total_events == 0:
        print("No communication events found.")
        return

    print("=== Communication Trace Summary ===")
    print(f"Events: {total_events:,}")
    print(f"Total bytes: {_format_bytes(total_bytes)}")
    print(f"Mean event size: {_format_bytes(int(statistics.mean(bytes_per_event)))}")
    if len(bytes_per_event) > 1:
        print(
            "Stddev event size:"
            f" {_format_bytes(int(statistics.pstdev(bytes_per_event)))}"
        )
    print()

    # Top tag aggregates
    print("-- Bytes by tag --")
    for tag, bytes_total in sorted(
        totals_by_tag.items(), key=lambda item: item[1], reverse=True
    )[:8]:
        share = bytes_total / total_bytes
        durations = durations_by_tag[tag]
        avg_dur = statistics.mean(durations)
        print(
            f"{tag or '<empty>' :>10} : {_format_bytes(bytes_total)} "
            f"({share:.1%} of total), avg duration {_format_seconds(avg_dur)}"
        )
    print()

    # Top device routes
    print("-- Bytes by device route (src -> dst) --")
    for (src, dst), bytes_total in sorted(
        totals_by_route.items(), key=lambda item: item[1], reverse=True
    )[:10]:
        share = bytes_total / total_bytes
        print(
            f"{src:>10} -> {dst:<10} : {_format_bytes(bytes_total)} ({share:.1%})"
        )
    print()

    # Bytes by device type route
    print("-- Bytes by device type pair --")
    for (src_type, dst_type), bytes_total in sorted(
        totals_by_type.items(), key=lambda item: item[1], reverse=True
    ):
        share = bytes_total / total_bytes
        print(
            f"{src_type:>5} -> {dst_type:<5} : {_format_bytes(bytes_total)} ({share:.1%})"
        )
    print()

    # Phase + tag combinations to highlight hotspots
    print("-- Top phase/tag hotspots --")
    for (phase, tag), bytes_total in sorted(
        totals_by_phase_tag.items(), key=lambda item: item[1], reverse=True
    )[:8]:
        share = bytes_total / total_bytes
        print(
            f"{phase:>12} | {tag or '<empty>':>10} :"
            f" {_format_bytes(bytes_total)} ({share:.1%})"
        )
    print()

    if top_event:
        print("-- Largest single transfer --")
        print(
            f"{top_event['src']} -> {top_event['dst']} | tag={top_event['tag']} |"
            f" bytes={_format_bytes(top_event['bytes'])} |"
            f" duration={_format_seconds(top_event['duration'])} |"
            f" phase={top_event['phase']}"
        )



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze TriForm communication trace CSV files."
    )
    parser.add_argument(
        "trace",
        type=Path,
        help="Path to the *_comms_trace.csv file",
    )
    args = parser.parse_args()

    if not args.trace.exists():
        parser.error(f"Trace file not found: {args.trace}")

    rows = list(load_comm_rows(args.trace))
    summarize_comms(rows)


if __name__ == "__main__":
    main()
