from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hetinfer_network_export import (  # noqa: E402
    build_network_manifest,
    export_network_manifest,
)
from model_parser import build_graph  # noqa: E402


def _snapshot() -> dict:
    return {
        "schedule_call_index": 1,
        "phase": "prefill",
        "devices": [],
        "operators": [
            {
                "op_id": "prefill:1:q_proj",
                "dependencies": [],
                "network_metadata": {
                    "name": "Q",
                    "phase": "prefill",
                    "batch": 1,
                    "seq_len": 8,
                    "node_attrs": {
                        "layer": 0,
                        "canonical_op_slot": "q",
                    },
                },
            }
        ],
        "inputs": [],
        "collective_contexts": [],
        "routes": [],
    }


class HetInferNetworkExportTests(unittest.TestCase):
    def test_qwen_full_graph_has_explicit_layer_and_global_slots(self) -> None:
        graph, shape = build_graph(
            {
                "model_family": "qwen",
                "model_variant": "1.8b",
                "batch": 1,
                "prefill_len": 128,
                "decode_len": 128,
                "dtype": "fp16",
            }
        )
        self.assertEqual(shape.layer_num, 28)
        self.assertEqual(len(graph.nodes), 28 * 17 + 3)
        self.assertEqual(
            {
                graph.nodes[name].attrs["canonical_op_slot"]
                for name in ("embedding", "final_norm", "lm_head")
            },
            {"embedding", "final_norm", "lm_head"},
        )
        layer_slots = {
            layer: {
                node.attrs["canonical_op_slot"]
                for node in graph.nodes.values()
                if node.attrs.get("layer_index") == layer
            }
            for layer in range(28)
        }
        self.assertEqual(len(layer_slots[0]), 17)
        self.assertTrue(
            all(layer_slots[layer] == layer_slots[0] for layer in range(1, 28))
        )

    def test_builds_and_writes_1npu2pim_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hardware = Path(tmp) / "hardware.json"
            hardware.write_text(
                json.dumps(
                    {
                        "hardware": {
                            "devices": [
                                {
                                    "name": "Ascend_910B_NPU0",
                                    "type": "npu",
                                    "mem_capacity_GB": 16,
                                },
                                {
                                    "name": "PIM0",
                                    "type": "pim",
                                    "mem_capacity_GB": 16,
                                },
                                {
                                    "name": "PIM1",
                                    "type": "pim",
                                    "mem_capacity_GB": 16,
                                },
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            cfg = {"hardware_json": str(hardware)}
            prior = {"graph_id": "qwen-1layer", "workload_id": "b1-s8"}
            payload = build_network_manifest(
                cfg=cfg,
                snapshots=[_snapshot()],
                prior_artifact=prior,
            )
            network = payload["networks"][0]
            self.assertEqual(network["graph_id"], "qwen-1layer")
            self.assertEqual(network["workload_id"], "b1-s8")
            self.assertEqual(
                network["policy_devices"],
                ["Ascend_910B_NPU0", "PIM0", "PIM1"],
            )
            self.assertEqual(
                network["workload"],
                {
                    "batch": 1,
                    "sequence_length": 8,
                    "scheduled_tokens": 8,
                    "mean_context": 8.0,
                },
            )
            self.assertEqual(network["operators"][0]["op_role"], "Q_PROJ")
            self.assertEqual(network["operators"][0]["layer_index"], 0)
            self.assertEqual(
                network["operators"][0]["canonical_op_slot"], "q"
            )

            output = Path(tmp) / "network.json"
            self.assertEqual(
                export_network_manifest(
                    cfg=cfg,
                    snapshots=[_snapshot()],
                    prior_artifact=prior,
                    output=output,
                ),
                output,
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
