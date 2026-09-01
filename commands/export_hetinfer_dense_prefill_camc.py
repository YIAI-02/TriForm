#!/usr/bin/env python3
"""Export the fixed b1-p128 full-Qwen prefill CAMC bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DOPS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DOPS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hetinfer_dense_prefill_camc_export import (  # noqa: E402
    export_dense_prefill_camc_bundle,
)
from hetinfer_prior import load_prior_artifact  # noqa: E402


def _load_json(path: Path, context: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Project full_qwen_1p8b b1-p128 network[0] and export its strict "
            "28-layer prefill CAMC profile"
        )
    )
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument("--tensor-bindings", required=True, type=Path)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    outputs = export_dense_prefill_camc_bundle(
        prior_artifact=load_prior_artifact(args.prior),
        network_manifest=_load_json(args.network, "network"),
        tensor_bindings=_load_json(args.tensor_bindings, "tensor_bindings"),
        hardware=_load_json(args.hardware, "hardware"),
        output_dir=args.output_dir,
    )
    for name, path in sorted(outputs.items()):
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
