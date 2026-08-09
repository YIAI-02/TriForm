"""Validate one live-captured DOPS prior and print a compact ingestion summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from hetinfer_prior import load_artifact_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--expected-batch", type=int)
    parser.add_argument("--expected-prefill", type=int)
    parser.add_argument("--expected-decode", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    artifact, source_artifact = load_artifact_bundle(args.artifact)

    provenance = artifact["provenance"]
    workload = provenance["workload"]
    expected = {
        "batch": args.expected_batch,
        "prefill_len": args.expected_prefill,
        "decode_len": args.expected_decode,
    }
    for field, value in expected.items():
        if value is not None and int(workload[field]) != int(value):
            raise ValueError(
                f"provenance workload mismatch for {field}: "
                f"expected={value} actual={workload[field]}"
            )

    profiles = artifact["profiles"]
    operators = [
        op
        for profile in profiles
        for phase in ("prefill", "decode")
        for op in profile["phases"][phase]["operators"]
    ]
    dynamic = [op for op in operators if op["dynamic_eligible"]]
    if not dynamic:
        raise ValueError(
            "artifact has no dynamic_eligible operators for online selection"
        )
    if not all(
        score["dops_score_s"] is not None
        for op in dynamic
        for score in op["candidates"].values()
    ):
        raise ValueError("a dynamic candidate is missing dops_score_s")

    devices = sorted({device["name"] for device in provenance["hardware"]["devices"]})
    if "GPU0" not in devices or "PIM0" not in devices:
        raise ValueError(
            f"expected GPU0 and PIM0 in hardware provenance, got {devices}"
        )

    print(
        "[PRIOR-VALID] "
        f"artifact_id={artifact['artifact_id']} profiles={len(profiles)} "
        f"operators={len(operators)} dynamic={len(dynamic)} "
        f"devices={','.join(devices)} provenance={provenance['status']} "
        f"source={source_artifact.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
