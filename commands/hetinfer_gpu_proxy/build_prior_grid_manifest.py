"""Validate profile artifacts below a grid directory and write a stable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from hetinfer_prior import load_json, validate_artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to <grid-root>/prior_grid_manifest.json",
    )
    parser.add_argument(
        "--require-count",
        type=int,
        help="Fail unless exactly this many valid prior.json files are present",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parser().parse_args()
    root = args.grid_root.expanduser().resolve()
    entries: list[dict[str, Any]] = []
    seen_workloads: set[tuple[int, int, int]] = set()

    for path in sorted(root.glob("*/prior.json")):
        artifact = load_json(path)
        validate_artifact(artifact)
        workload = artifact["provenance"]["workload"]
        key = (
            int(workload["batch"]),
            int(workload["prefill_len"]),
            int(workload["decode_len"]),
        )
        if key in seen_workloads:
            raise ValueError(f"duplicate workload in grid: {key}")
        seen_workloads.add(key)
        config_snapshot = artifact["provenance"].get("config", {}).get("snapshot", {})
        entries.append(
            {
                "batch": key[0],
                "prefill_len": key[1],
                "decode_len": key[2],
                "artifact_id": artifact["artifact_id"],
                "profile_count": len(artifact["profiles"]),
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "provenance_status": artifact["provenance"]["status"],
                "evidence_class": config_snapshot.get("evidence_class"),
            }
        )

    if not entries:
        raise ValueError(f"no */prior.json artifacts found under {root}")
    if args.require_count is not None and len(entries) != args.require_count:
        raise ValueError(
            f"expected {args.require_count} artifacts under {root}, found {len(entries)}"
        )

    manifest = {
        "schema": "dops.hetinfer_prior_grid_manifest.v1",
        "root": str(root),
        "artifact_count": len(entries),
        "workload_axes": {
            "batch": sorted({entry["batch"] for entry in entries}),
            "prefill_len": sorted({entry["prefill_len"] for entry in entries}),
            "decode_len": sorted({entry["decode_len"] for entry in entries}),
        },
        "artifacts": sorted(
            entries,
            key=lambda entry: (
                entry["batch"],
                entry["prefill_len"],
                entry["decode_len"],
            ),
        ),
    }
    output = args.output or (root / "prior_grid_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(output)
    print(f"[GRID-MANIFEST] wrote {output} artifacts={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
