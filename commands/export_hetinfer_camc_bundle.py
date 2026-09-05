#!/usr/bin/env python3
"""Export one validated five-file Het-Infer CAMC bundle."""

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

from hetinfer_camc_profile_export import export_camc_bundle  # noqa: E402
from hetinfer_prior import load_prior_artifact  # noqa: E402


def _load_json(path: Path, context: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and write prior, network, bindings, spec, and profile"
    )
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument("--tensor-bindings", required=True, type=Path)
    parser.add_argument("--layer-spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    outputs = export_camc_bundle(
        prior_artifact=load_prior_artifact(args.prior),
        network_manifest=_load_json(args.network, "network"),
        tensor_bindings=_load_json(args.tensor_bindings, "tensor_bindings"),
        layer_spec=_load_json(args.layer_spec, "layer_spec"),
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {name: str(path) for name, path in sorted(outputs.items())},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
