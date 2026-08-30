from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from commands.export_hetinfer_full_suite import (  # noqa: E402
    DECODE_TOKENS,
    _backfill_tensor_bindings,
    _is_complete,
)


class HetInferFullSuiteExportTests(unittest.TestCase):
    def test_complete_requires_tensor_bindings_for_every_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            network_path = root / "network.json"
            tensor_bindings_path = root / "tensor_bindings.json"
            networks = [
                {
                    "graph_id": "graph",
                    "workload_id": "b1-p128-h128",
                    "workload": {
                        "batch": 1,
                        "sequence_length": 128,
                        "scheduled_tokens": 128,
                    }
                }
                for _ in range(1 + DECODE_TOKENS)
            ]
            network_path.write_text(
                json.dumps({"networks": networks}), encoding="utf-8"
            )
            tensor_bindings_path.write_text(
                json.dumps(
                    {
                        "schema": "dops.hetinfer_tensor_bindings.v1",
                        "schema_version": 1,
                        "graph_id": "graph",
                        "workload_id": "b1-p128-h128",
                        "bindings": [
                            {"network_index": index}
                            for index in range(len(networks))
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                _is_complete(network_path, tensor_bindings_path, 1, 128)
            )

            tensor_bindings_path.write_text(
                json.dumps(
                    {
                        "schema": "dops.hetinfer_tensor_bindings.v1",
                        "schema_version": 1,
                        "graph_id": "graph",
                        "workload_id": "b1-p128-h128",
                        "bindings": [
                            {"network_index": index}
                            for index in range(len(networks) - 1)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(
                _is_complete(network_path, tensor_bindings_path, 1, 128)
            )

    def test_backfills_a_complete_legacy_sidecar_without_rescheduling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior_path = root / "prior.json"
            network_path = root / "network.json"
            tensor_bindings_path = root / "tensor_bindings.json"
            networks = []
            inputs = []
            for index in range(1 + DECODE_TOKENS):
                op_id = f"opaque-op-{index}"
                networks.append(
                    {
                        "graph_id": "graph",
                        "workload_id": "b1-p128-h128",
                        "workload": {
                            "batch": 1,
                            "sequence_length": 128,
                            "scheduled_tokens": 128,
                        },
                        "operators": [
                            {
                                "op_id": op_id,
                                "layer_index": None,
                                "canonical_op_slot": "embedding",
                            }
                        ],
                    }
                )
                inputs.append(
                    {
                        "consumer_op_id": op_id,
                        "producer_op_id": None,
                        "tensor_id": f"opaque-tensor-{index}",
                        "semantics": "data",
                        "bytes": 64,
                    }
                )
            prior_path.write_text(
                json.dumps(
                    {
                        "graph_id": "graph",
                        "workload_id": "b1-p128-h128",
                        "inputs": inputs,
                    }
                ),
                encoding="utf-8",
            )
            network_path.write_text(
                json.dumps({"networks": networks}), encoding="utf-8"
            )

            _backfill_tensor_bindings(
                prior_path=prior_path,
                network_path=network_path,
                output=tensor_bindings_path,
            )

            self.assertTrue(
                _is_complete(network_path, tensor_bindings_path, 1, 128)
            )
            bindings = json.loads(
                tensor_bindings_path.read_text(encoding="utf-8")
            )["bindings"]
            self.assertEqual(len(bindings), 1 + DECODE_TOKENS)
            self.assertEqual(
                {item["persistence"] for item in bindings}, {"request_input"}
            )


if __name__ == "__main__":
    unittest.main()
