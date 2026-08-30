from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hetinfer_tensor_bindings_export import (  # noqa: E402
    build_tensor_bindings_manifest,
    build_tensor_bindings_manifest_from_artifacts,
    export_tensor_bindings_manifest,
)


def _operator(
    op_id: str,
    name: str,
    canonical_op_slot: str,
    layer_index: int | None,
) -> dict:
    attrs = {"canonical_op_slot": canonical_op_slot}
    if layer_index is not None:
        attrs["layer_index"] = layer_index
    return {
        "op_id": op_id,
        "network_metadata": {
            "name": name,
            "node_attrs": attrs,
        },
    }


def _input(
    *,
    consumer: str,
    producer: str | None,
    tensor: str,
    bytes_: int,
    semantics: str = "data",
) -> dict:
    return {
        "consumer_op_id": consumer,
        "producer_op_id": producer,
        "tensor_id": tensor,
        "semantics": semantics,
        "bytes": bytes_,
    }


def _snapshot() -> dict:
    return {
        "operators": [
            _operator("op-a", "IDENTITY", "embedding", None),
            _operator("op-b", "K", "k", 3),
            _operator("op-c", "V", "v", 3),
            _operator("op-d", "K_WRITE", "k_write", 3),
            _operator("op-e", "V_WRITE", "v_write", 3),
            _operator("op-f", "QK", "qk", 3),
            _operator("op-g", "SV", "sv", 3),
            _operator("op-h", "SOFTMAX", "softmax", 3),
            _operator("op-i", "O", "o", 3),
        ],
        "inputs": [
            _input(
                consumer="op-a",
                producer=None,
                tensor="opaque-001",
                bytes_=64,
            ),
            _input(
                consumer="op-f",
                producer=None,
                tensor="opaque-002",
                bytes_=2048,
            ),
            _input(
                consumer="op-g",
                producer=None,
                tensor="opaque-003",
                bytes_=2048,
            ),
            # The ordinary attention consumer appears before K_WRITE/V_WRITE.
            # Classification must use graph metadata across the whole fanout.
            _input(
                consumer="op-f",
                producer="op-b",
                tensor="opaque-004",
                bytes_=512,
            ),
            _input(
                consumer="op-d",
                producer="op-b",
                tensor="opaque-004",
                bytes_=512,
            ),
            _input(
                consumer="op-g",
                producer="op-c",
                tensor="opaque-005",
                bytes_=512,
            ),
            _input(
                consumer="op-e",
                producer="op-c",
                tensor="opaque-005",
                bytes_=512,
            ),
            _input(
                consumer="op-i",
                producer="op-h",
                tensor="opaque-006",
                bytes_=128,
            ),
            _input(
                consumer="op-i",
                producer="op-h",
                tensor="opaque-007",
                bytes_=0,
                semantics="barrier",
            ),
        ],
    }


def _prior_artifact() -> dict:
    return {
        "graph_id": "graph",
        "workload_id": "workload",
        "inputs": copy.deepcopy(_snapshot()["inputs"]),
    }


def _network_manifest() -> dict:
    operators = []
    for operator in _snapshot()["operators"]:
        attrs = operator["network_metadata"]["node_attrs"]
        operators.append(
            {
                "op_id": operator["op_id"],
                "layer_index": attrs.get("layer_index"),
                "canonical_op_slot": attrs["canonical_op_slot"],
            }
        )
    return {
        "networks": [
            {
                "graph_id": "graph",
                "workload_id": "workload",
                "operators": operators,
            }
        ]
    }


class HetInferTensorBindingsExportTests(unittest.TestCase):
    def test_uses_opaque_ids_and_unifies_kv_append_with_read_slots(self) -> None:
        payload = build_tensor_bindings_manifest(
            snapshots=[_snapshot(), _snapshot()],
            prior_artifact={"graph_id": "graph", "workload_id": "workload"},
        )
        self.assertEqual(
            set(payload),
            {
                "schema",
                "schema_version",
                "graph_id",
                "workload_id",
                "bindings",
            },
        )
        self.assertEqual(payload["schema"], "dops.hetinfer_tensor_bindings.v1")
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["graph_id"], "graph")
        self.assertEqual(payload["workload_id"], "workload")

        by_key = {
            (item["network_index"], item["tensor_id"]): item
            for item in payload["bindings"]
        }
        self.assertEqual(len(by_key), 12)
        self.assertNotIn((0, "opaque-007"), by_key)
        self.assertEqual(
            by_key[(0, "opaque-001")],
            {
                "network_index": 0,
                "tensor_id": "opaque-001",
                "layer_index": None,
                "canonical_tensor_slot": "request_input",
                "persistence": "request_input",
                "size_bytes": 64,
            },
        )
        self.assertEqual(
            (
                by_key[(0, "opaque-002")]["layer_index"],
                by_key[(0, "opaque-002")]["canonical_tensor_slot"],
                by_key[(0, "opaque-002")]["persistence"],
            ),
            (3, "k", "kv-read"),
        )
        self.assertEqual(
            (
                by_key[(0, "opaque-004")]["layer_index"],
                by_key[(0, "opaque-004")]["canonical_tensor_slot"],
                by_key[(0, "opaque-004")]["persistence"],
            ),
            (3, "k", "kv-append"),
        )
        self.assertEqual(
            (
                by_key[(0, "opaque-003")]["layer_index"],
                by_key[(0, "opaque-003")]["canonical_tensor_slot"],
                by_key[(0, "opaque-003")]["persistence"],
            ),
            (3, "v", "kv-read"),
        )
        self.assertEqual(
            (
                by_key[(0, "opaque-005")]["layer_index"],
                by_key[(0, "opaque-005")]["canonical_tensor_slot"],
                by_key[(0, "opaque-005")]["persistence"],
            ),
            (3, "v", "kv-append"),
        )
        self.assertEqual(
            (
                by_key[(0, "opaque-006")]["canonical_tensor_slot"],
                by_key[(0, "opaque-006")]["persistence"],
                by_key[(0, "opaque-006")]["size_bytes"],
            ),
            ("softmax", "transient", 128),
        )
        self.assertEqual(by_key[(1, "opaque-004")]["network_index"], 1)

    def test_writes_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "tensor_bindings.json"
            payload = build_tensor_bindings_manifest(
                snapshots=[_snapshot()],
                prior_artifact={"graph_id": "graph", "workload_id": "workload"},
            )
            self.assertEqual(
                export_tensor_bindings_manifest(
                    snapshots=[_snapshot()],
                    prior_artifact={
                        "graph_id": "graph",
                        "workload_id": "workload",
                    },
                    output=output,
                ),
                output,
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)

    def test_artifact_backfill_matches_snapshot_export(self) -> None:
        expected = build_tensor_bindings_manifest(
            snapshots=[_snapshot()],
            prior_artifact=_prior_artifact(),
        )
        self.assertEqual(
            build_tensor_bindings_manifest_from_artifacts(
                prior_artifact=_prior_artifact(),
                network_manifest=_network_manifest(),
            ),
            expected,
        )

    def test_artifact_backfill_rejects_mismatched_sidecars(self) -> None:
        network_manifest = _network_manifest()
        network_manifest["networks"][0]["workload_id"] = "other-workload"
        with self.assertRaisesRegex(RuntimeError, "workload_id"):
            build_tensor_bindings_manifest_from_artifacts(
                prior_artifact=_prior_artifact(),
                network_manifest=network_manifest,
            )

        prior_artifact = _prior_artifact()
        prior_artifact["inputs"][0]["consumer_op_id"] = "absent-op"
        with self.assertRaisesRegex(RuntimeError, "consumer_op_id"):
            build_tensor_bindings_manifest_from_artifacts(
                prior_artifact=prior_artifact,
                network_manifest=_network_manifest(),
            )


if __name__ == "__main__":
    unittest.main()
