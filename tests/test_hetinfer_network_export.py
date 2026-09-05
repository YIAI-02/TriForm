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
    _block_type,
    _op_role,
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

    def test_exports_moe_combine_and_sd_transfer_roles(self) -> None:
        self.assertEqual(_op_role("MoE_Combine", {}), "COMBINE")
        self.assertEqual(
            _block_type({}, "COMBINE", 0),
            "MOE_BLOCK",
        )
        self.assertEqual(
            _op_role("transfer", {"op_role": "CANDIDATE_TRANSFER"}),
            "CANDIDATE_TRANSFER",
        )
        self.assertEqual(
            _block_type({}, "CANDIDATE_TRANSFER", 0),
            "SD_COMPONENT",
        )

    def test_mixtral_graph_exports_all_experts_after_router_and_before_combine(self) -> None:
        graph, shape = build_graph(
            {
                "model_family": "mixtral",
                "model_variant": "8x7b",
                "batch": 1,
                "prefill_len": 16,
                "decode_len": 1,
                "dtype": "fp16",
                "tp": 1,
            }
        )

        self.assertEqual(shape.active_experts_per_layer, 8)
        self.assertEqual(shape.moe_pruned_experts_per_layer, 0)
        router = "L0_Router"
        combine = "L0_Combine"
        self.assertEqual(graph.predecessors(router), ["L0_LN2"])
        self.assertEqual(graph.nodes[router].attrs["moe_selection_policy"], "runtime")
        self.assertEqual(graph.nodes[combine].attrs["op_role"], "COMBINE")
        self.assertEqual(
            set(graph.predecessors(combine)),
            {f"L0_FFN_W2_E{expert_id}" for expert_id in range(8)},
        )
        self.assertIn(combine, graph.predecessors("L0_Add2"))

        for expert_id in range(8):
            for slot in ("FFN_W1", "FFN_W3"):
                node_id = f"L0_{slot}_E{expert_id}"
                self.assertEqual(graph.predecessors(node_id), [router])
                attrs = graph.nodes[node_id].attrs
                self.assertEqual(attrs["expert_id"], f"E{expert_id}")
                self.assertEqual(
                    attrs["placement_supernode"],
                    f"L0:expert:{expert_id}",
                )
                self.assertEqual(
                    attrs["parallel_group_hint"],
                    "L0:moe_experts",
                )
                self.assertNotIn("moe_token_fraction", attrs)

        order = graph.topological()
        position = {node_id: index for index, node_id in enumerate(order)}
        self.assertLess(position[router], position["L0_FFN_W1_E7"])
        self.assertLess(position["L0_FFN_W2_E7"], position[combine])

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
                    "past_kv_len": 0,
                    "query_len": 8,
                    "scheduled_tokens": 8,
                    "mean_context": 8.0,
                },
            )
            self.assertEqual(network["operators"][0]["op_role"], "Q_PROJ")
            self.assertEqual(network["operators"][0]["layer_index"], 0)
            self.assertEqual(
                network["operators"][0]["canonical_op_slot"], "q"
            )
            self.assertEqual(network["operators"][0]["operator_index"], 0)

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
