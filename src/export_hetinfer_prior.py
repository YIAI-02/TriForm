#!/usr/bin/env python3
"""Export a versioned Het-Infer prior from DOPS artifacts.

This compatibility tool can read an old ``best_summary``.  Such an export is
explicitly marked unscored: it never manufactures alternatives or timing
components that the old artifact did not contain.  Prefer the live
``evaluate --hetinfer-prior-out`` hook for DQN-ready candidate priors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from hardware import demo_cluster
from hetinfer_prior import build_artifact, load_json, write_artifact
from mainlib.cli import _load_cfg_from_json
from model_parser import build_graph


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-summary", required=True, help="Legacy or current DOPS best_summary JSON")
    parser.add_argument("--output", required=True, help="Output dops.hetinfer_prior.v1 JSON")
    parser.add_argument("--config", help="Full original DOPS input config (recommended)")
    parser.add_argument("--hardware-json", help="Override hardware_json in the config")
    parser.add_argument(
        "--candidate-records",
        help="Optional JSON array (or object field candidate_records) captured by Bifocal",
    )
    parser.add_argument("--model-revision", help="Model weights/config revision")
    parser.add_argument("--dops-revision", help="Override producer git revision")
    return parser


def _load_candidate_records(path: str | None) -> list[Dict[str, Any]]:
    if not path:
        return []
    import json

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("candidate_records", value.get("hetinfer_candidate_records"))
    if not isinstance(value, list):
        raise ValueError("--candidate-records must contain a JSON array")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("--candidate-records array entries must be objects")
    return value


def main() -> int:
    args = _parser().parse_args()
    summary = load_json(args.best_summary)
    cfg: Dict[str, Any] = {}
    if args.config:
        # Match the normal DOPS CLI: input paths in a config are relative to
        # the config file (with src-root/cwd fallbacks), not to whichever
        # directory happened to invoke this compatibility tool.
        cfg.update(_load_cfg_from_json(args.config))
    if isinstance(summary.get("config"), dict):
        # The explicit config wins; legacy summary fields only fill gaps.
        cfg = {**summary["config"], **cfg}
    if args.hardware_json:
        cfg["hardware_json"] = str(Path(args.hardware_json).expanduser().resolve())
    if args.model_revision:
        cfg["model_revision"] = str(args.model_revision)

    # Older summaries may contain an explicit JSON null for an optional shape
    # override.  Absence means "use the family/variant default" and is the
    # only representation accepted by the existing model parser.
    if cfg.get("shape_file") in (None, ""):
        cfg.pop("shape_file", None)

    graph = shape = cluster = None
    # Provenance recovery is best-effort only in legacy mode.  The resulting
    # artifact records missing fields and remains placement-only when inputs
    # are insufficient.
    try:
        graph, shape = build_graph(cfg)
    except Exception:
        graph = shape = None
    try:
        cluster = demo_cluster(cfg)
    except Exception:
        cluster = None

    records = _load_candidate_records(args.candidate_records)
    artifact = build_artifact(
        cfg=cfg,
        graph=graph,
        cluster=cluster,
        shape=shape,
        candidate_records=records or None,
        legacy_best_summary=None if records else summary,
        producer_revision=args.dops_revision,
    )
    output = write_artifact(artifact, args.output)
    print(f"[Het-Infer] wrote {output}")
    if artifact["provenance"]["status"] != "complete":
        print(
            "[Het-Infer] provenance is partial; missing="
            + ",".join(artifact["provenance"].get("missing_fields", []))
        )
    if any(not p["source"].get("candidate_scores_complete") for p in artifact["profiles"]):
        print("[Het-Infer] candidate scores are incomplete; use live Bifocal capture for DQN bootstrap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
