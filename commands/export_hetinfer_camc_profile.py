#!/usr/bin/env python3
"""Export one strict CAMC profile from existing DOPS sidecars and a layer spec."""

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

from hetinfer_camc_profile_export import export_camc_profile  # noqa: E402
from hetinfer_prior import load_prior_artifact  # noqa: E402


def _load_json(path: Path, context: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build dops.hetinfer_camc_profile.v1 without copying t_service or "
            "t_move from the prior"
        )
    )
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument("--tensor-bindings", required=True, type=Path)
    parser.add_argument(
        "--layer-spec",
        required=True,
        type=Path,
        help=(
            "Explicit JSON deployment input; layer, shape, capability, order, "
            "home, and grouping fields are never inferred"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = export_camc_profile(
        prior_artifact=load_prior_artifact(args.prior),
        network_manifest=_load_json(args.network, "network"),
        tensor_bindings=_load_json(args.tensor_bindings, "tensor_bindings"),
        layer_spec=_load_json(args.layer_spec, "layer_spec"),
        output=args.output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
