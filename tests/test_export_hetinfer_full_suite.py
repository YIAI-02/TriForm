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


def _network(index: int, op_id: str | None = None) -> dict:
    is_prefill = index == 0
    sequence_length = 128 if is_prefill else 127 + index
    return {
        "graph_id": "graph",
        "workload_id": "b1-p128-h128",
        "phase": "prefill" if is_prefill else "decode",
        "workload": {
            "batch": 1,
            "sequence_length": sequence_length,
            "past_kv_len": 0 if is_prefill else sequence_length,
            "query_len": 128 if is_prefill else 1,
            "scheduled_tokens": 128 if is_prefill else 1,
            "mean_context": float(sequence_length),
        },
        "operators": [
            {
                "op_id": op_id or f"op-{index}",
                "layer_index": None,
                "canonical_op_slot": "embedding",
                "operator_index": 0,
            }
        ],
    }


class HetInferFullSuiteExportTests(unittest.TestCase):
    def test_complete_requires_tensor_bindings_for_every_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            network_path = root / "network.json"
            tensor_bindings_path = root / "tensor_bindings.json"
            networks = [_network(index) for index in range(1 + DECODE_TOKENS)]
            network_path.write_text(
                json.dumps({"schema": "dops.hetinfer_network.v1", "schema_version": 1, "networks": networks}), encoding="utf-8"
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

    def test_backfills_a_complete_sidecar_without_rescheduling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior_path = root / "prior.json"
            network_path = root / "network.json"
            tensor_bindings_path = root / "tensor_bindings.json"
            networks = []
            inputs = []
            for index in range(1 + DECODE_TOKENS):
                op_id = f"opaque-op-{index}"
                networks.append(_network(index, op_id))
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
                json.dumps({"schema": "dops.hetinfer_network.v1", "schema_version": 1, "networks": networks}), encoding="utf-8"
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
