from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hetinfer_dense_prefill_camc_export import (  # noqa: E402
    GRAPH_ID,
    LAYER_COUNT,
    LAYER_SLOTS,
    WORKLOAD_ID,
    build_dense_prefill_camc_bundle,
    export_dense_prefill_camc_bundle,
)
from hetinfer_prior import DOPSPriorArtifact, validate_prior_artifact  # noqa: E402


def _op_id(layer_index: int, slot: str) -> str:
    names = {
        "ln": "LN",
        "q": "Q",
        "k": "K",
        "v": "V",
        "k_write": "K_write",
        "v_write": "V_write",
        "qk": "QK",
        "softmax": "Softmax",
        "sv": "SV",
        "o": "O",
        "add1": "Add1",
        "ln2": "LN2",
        "ffn_w1": "FFN_W1",
        "ffn_w3": "FFN_W3",
        "swiglu": "SwiGLU",
        "ffn_w2": "FFN_W2",
        "add2": "Add2",
    }
    return f"prefill:1:L{layer_index}_{names[slot]}"


def _dependencies(layer_index: int, slot: str) -> list[str]:
    residual = (
        "prefill:1:embedding"
        if layer_index == 0
        else _op_id(layer_index - 1, "add2")
    )
    current = lambda name: _op_id(layer_index, name)
    return {
        "ln": [residual],
        "q": [current("ln")],
        "k": [current("ln")],
        "v": [current("ln")],
        "k_write": [current("k")],
        "v_write": [current("v")],
        "qk": [current("q"), current("k")],
        "softmax": [current("qk")],
        "sv": [current("v"), current("softmax")],
        "o": [current("sv")],
        "add1": (
            [current("o")]
            if layer_index == 0
            else [residual, current("o")]
        ),
        "ln2": [current("add1")],
        "ffn_w1": [current("ln2")],
        "ffn_w3": [current("ln2")],
        "swiglu": [current("ffn_w1"), current("ffn_w3")],
        "ffn_w2": [current("swiglu")],
        "add2": [current("add1"), current("ffn_w2")],
    }[slot]


def _operator_role(slot: str) -> str:
    return {
        "ln": "LAYERNORM",
        "q": "Q_PROJ",
        "k": "K_PROJ",
        "v": "V_PROJ",
        "k_write": "KV_WRITE",
        "v_write": "KV_WRITE",
        "qk": "QK",
        "softmax": "SOFTMAX",
        "sv": "SV",
        "o": "O_PROJ",
        "add1": "OTHER",
        "ln2": "LAYERNORM",
        "ffn_w1": "FFN_UP",
        "ffn_w3": "FFN_UP",
        "swiglu": "ACTIVATION",
        "ffn_w2": "FFN_DOWN",
        "add2": "OTHER",
    }[slot]


def _artifacts() -> tuple[DOPSPriorArtifact, dict, dict, dict, dict]:
    network_operators = []
    prior_operators = []
    legal_devices: dict[str, tuple[str, ...]] = {}
    placement: dict[str, str] = {}
    for layer_index in range(LAYER_COUNT):
        raw_by_slot = {}
        for slot in LAYER_SLOTS:
            op_id = _op_id(layer_index, slot)
            dependencies = _dependencies(layer_index, slot)
            raw_by_slot[slot] = {
                "block_id": f"layer:{layer_index}",
                "block_type": "DENSE_BLOCK",
                "canonical_op_slot": slot,
                "dependencies": dependencies,
                "layer_index": layer_index,
                "op_id": op_id,
                "op_role": _operator_role(slot),
                "repeat_index": layer_index,
                "total_repeats": LAYER_COUNT,
            }
            legal = ("PIM0",) if slot in {"k_write", "v_write"} else (
                "Ascend_910B_NPU0",
                "PIM0",
            )
            legal_devices[op_id] = legal
            placement[op_id] = (
                "PIM0" if slot in {"k", "v", "k_write", "v_write"}
                else "Ascend_910B_NPU0"
            )
            prior_operators.append(
                {
                    "op_id": op_id,
                    "dependencies": dependencies,
                    "legal_devices": list(legal),
                }
            )
        network_operators.extend(
            raw_by_slot[slot]
            for slot in (
                "ln",
                "add1",
                "q",
                "k",
                "v",
                "k_write",
                "v_write",
                "qk",
                "softmax",
                "sv",
                "o",
                "ln2",
                "ffn_w1",
                "ffn_w3",
                "swiglu",
                "ffn_w2",
                "add2",
            )
        )

    globals_ = [
        {
            "block_id": "global",
            "block_type": "OTHER",
            "canonical_op_slot": "embedding",
            "dependencies": [],
            "layer_index": None,
            "op_id": "prefill:1:embedding",
            "op_role": "OTHER",
            "repeat_index": 0,
            "total_repeats": 1,
        },
        {
            "block_id": "global",
            "block_type": "OTHER",
            "canonical_op_slot": "final_norm",
            "dependencies": [_op_id(LAYER_COUNT - 1, "add2")],
            "layer_index": None,
            "op_id": "prefill:1:final_norm",
            "op_role": "LAYERNORM",
            "repeat_index": 0,
            "total_repeats": 1,
        },
        {
            "block_id": "global",
            "block_type": "OTHER",
            "canonical_op_slot": "lm_head",
            "dependencies": ["prefill:1:final_norm"],
            "layer_index": None,
            "op_id": "prefill:1:lm_head",
            "op_role": "Q_PROJ",
            "repeat_index": 0,
            "total_repeats": 1,
        },
    ]
    for operator in globals_:
        op_id = operator["op_id"]
        legal_devices[op_id] = ("Ascend_910B_NPU0",)
        placement[op_id] = "Ascend_910B_NPU0"
        prior_operators.append(
            {
                "op_id": op_id,
                "dependencies": operator["dependencies"],
                "legal_devices": ["Ascend_910B_NPU0"],
            }
        )
    network_operators.extend(globals_)
    for operator_index, operator in enumerate(network_operators):
        operator["operator_index"] = operator_index

    decode_id = "decode:2:only"
    prior_operators.append(
        {"op_id": decode_id, "dependencies": [], "legal_devices": ["PIM0"]}
    )
    legal_devices[decode_id] = ("PIM0",)
    placement[decode_id] = "PIM0"
    inputs = []
    legal_routes = []
    movement_times = []
    for operator in prior_operators:
        op_id = operator["op_id"]
        producers = operator["dependencies"] or [None]
        for input_index, producer in enumerate(producers):
            tensor_id = f"{producer or 'external'}->{op_id}:{input_index}"
            input_entry = {
                "consumer_op_id": op_id,
                "producer_op_id": producer,
                "tensor_id": tensor_id,
                "semantics": "data",
                "bytes": 16,
                "source_residencies": [
                    {"device_id": "Ascend_910B_NPU0", "layout": "row_major"}
                ],
                "destination_devices": list(legal_devices[op_id]),
            }
            inputs.append(input_entry)
            for destination in input_entry["destination_devices"]:
                route = {
                    "tensor_id": tensor_id,
                    "source_device_id": "Ascend_910B_NPU0",
                    "destination_device_id": destination,
                    "bytes": 16,
                    "layout": "row_major",
                }
                legal_routes.append(route)
                movement_times.append(
                    {
                        **route,
                        "duration_s": (
                            0.0 if destination == "Ascend_910B_NPU0" else 0.001
                        ),
                    }
                )
    prior_payload = {
        "schema": "dops.hetinfer_prior.v1",
        "schema_version": 1,
        "graph_id": GRAPH_ID,
        "workload_id": WORKLOAD_ID,
        "time_unit": "seconds",
        "devices": [
            {"device_id": "CPU0"},
            {"device_id": "Ascend_910B_NPU0"},
            {"device_id": "PIM0"},
        ],
        "operators": prior_operators,
        "inputs": inputs,
        "collective_contexts": [],
        "legal_movement_routes": legal_routes,
        "expert_placement": [
            {"op_id": op_id, "device_id": device_id}
            for op_id, device_id in placement.items()
        ],
        "t_service": [
            {"op_id": op_id, "device_id": device_id, "duration_s": 0.001}
            for op_id, devices in legal_devices.items()
            for device_id in devices
        ],
        "t_move": movement_times,
    }
    prior = validate_prior_artifact(prior_payload)
    network = {
        "schema": "dops.hetinfer_network.v1",
        "schema_version": 1,
        "networks": [
            {
                "graph_id": GRAPH_ID,
                "workload_id": WORKLOAD_ID,
                "phase": "prefill",
                "schedule_call_index": 0,
                "policy_devices": ["Ascend_910B_NPU0", "PIM0", "PIM1"],
                "device_memory_bytes": {
                    "Ascend_910B_NPU0": 1024,
                    "PIM0": 1024,
                    "PIM1": 1024,
                },
                "workload": {
                    "batch": 1,
                    "sequence_length": 128,
                    "past_kv_len": 0,
                    "query_len": 128,
                    "scheduled_tokens": 128,
                    "mean_context": 128.0,
                },
                "operators": network_operators,
            },
            {
                "graph_id": GRAPH_ID,
                "workload_id": WORKLOAD_ID,
                "phase": "decode",
                "operators": [
                    {"op_id": decode_id, "dependencies": []}
                ],
            },
        ],
    }
    bindings = {
        "schema": "dops.hetinfer_tensor_bindings.v1",
        "schema_version": 1,
        "graph_id": GRAPH_ID,
        "workload_id": WORKLOAD_ID,
        "bindings": [
            {
                "network_index": 0,
                "tensor_id": f"layer-{layer_index}",
                "layer_index": layer_index,
            }
            for layer_index in range(LAYER_COUNT)
        ]
        + [{"network_index": 1, "tensor_id": "decode", "layer_index": None}],
    }
    hardware = {
        "hardware": {
            "devices": [
                {"name": "CPU0", "type": "cpu", "tflops": 0.001, "mem_bw_GBs": 1},
                {
                    "name": "Ascend_910B_NPU0",
                    "type": "npu",
                    "tflops": 280,
                    "mem_bw_GBs": 819.2,
                },
                {"name": "PIM0", "type": "pim", "tflops": 16, "mem_bw_GBs": 16384},
            ]
        }
    }
    measured_capabilities = {
        "NPU": {
            "effective_compute_flops_per_s": 1234.0,
            "effective_bandwidth_bytes_per_s": 2345.0,
            "queue_count": 1,
        },
        "PIM": {
            "effective_compute_flops_per_s": 3456.0,
            "effective_bandwidth_bytes_per_s": 4567.0,
            "queue_count": 1,
        },
    }
    return prior, network, bindings, hardware, measured_capabilities


class DensePrefillCAMCExportTests(unittest.TestCase):
    def test_projects_true_28_layer_prefill_and_excludes_source_cpu(self) -> None:
        prior, network, bindings, hardware, measured = _artifacts()

        payloads = build_dense_prefill_camc_bundle(
            prior_artifact=prior,
            network_manifest=network,
            tensor_bindings=bindings,
            hardware=hardware,
            measured_domain_capabilities=measured,
        )
        projected_network = payloads["network"]
        projected_bindings = payloads["tensor_bindings"]
        profile = payloads["camc_profile"]

        self.assertEqual(len(network["networks"]), 2)
        self.assertEqual(len(payloads["prior"]["operators"]), 479)
        self.assertEqual(len(payloads["layer_spec"]["layers"]), 1)
        self.assertEqual(len(projected_network["networks"]), 1)
        operators = projected_network["networks"][0]["operators"]
        self.assertEqual(len(operators), 479)
        self.assertEqual(
            {operator["layer_index"] for operator in operators if operator["layer_index"] is not None},
            set(range(28)),
        )
        self.assertTrue(
            all(binding["network_index"] == 0 for binding in projected_bindings["bindings"])
        )
        self.assertEqual(
            profile["device_domains"],
            {"Ascend_910B_NPU0": "NPU", "PIM0": "PIM"},
        )
        layer = profile["layers"][0]
        self.assertEqual(len(layer["nodes"]), 479)
        self.assertEqual(layer["default_order"][0], "prefill:1:embedding")
        self.assertEqual(layer["default_order"][-1], "prefill:1:lm_head")
        self.assertLess(
            layer["default_order"].index(_op_id(0, "o")),
            layer["default_order"].index(_op_id(0, "add1")),
        )
        nodes = {node["op_id"]: node for node in layer["nodes"]}
        self.assertNotEqual(
            nodes[_op_id(0, "q")]["parallel_group_hint"],
            nodes[_op_id(1, "q")]["parallel_group_hint"],
        )
        self.assertEqual(
            nodes[_op_id(0, "q")]["operator_family"],
            nodes[_op_id(1, "q")]["operator_family"],
        )
        self.assertEqual(nodes[_op_id(0, "qk")]["kv_home"], "PIM0")
        self.assertEqual(
            layer["domain_capabilities"]["NPU"]["effective_compute_flops_per_s"],
            1234.0,
        )

    def test_rejects_a_network_that_is_not_28_complete_layers(self) -> None:
        prior, network, bindings, hardware, measured = _artifacts()
        invalid = copy.deepcopy(network)
        invalid["networks"][0]["operators"] = [
            operator for operator in invalid["networks"][0]["operators"]
            if operator.get("layer_index") != 27
        ]

        with self.assertRaisesRegex(RuntimeError, "exactly 479 operators"):
            build_dense_prefill_camc_bundle(
                prior_artifact=prior,
                network_manifest=invalid,
                tensor_bindings=bindings,
                hardware=hardware,
                measured_domain_capabilities=measured,
            )

    def test_rejects_wrong_dense_dependency_contract(self) -> None:
        prior, network, bindings, hardware, measured = _artifacts()
        invalid = copy.deepcopy(network)
        qk = next(
            operator
            for operator in invalid["networks"][0]["operators"]
            if operator["op_id"] == _op_id(7, "qk")
        )
        qk["dependencies"] = [_op_id(7, "q"), _op_id(7, "k_write")]

        with self.assertRaisesRegex(RuntimeError, "dense dependency contract"):
            build_dense_prefill_camc_bundle(
                prior_artifact=prior,
                network_manifest=invalid,
                tensor_bindings=bindings,
                hardware=hardware,
                measured_domain_capabilities=measured,
            )

    def test_export_writes_only_projected_inputs_and_profile(self) -> None:
        prior, network, bindings, hardware, measured = _artifacts()
        with tempfile.TemporaryDirectory() as tmp:
            outputs = export_dense_prefill_camc_bundle(
                prior_artifact=prior,
                network_manifest=network,
                tensor_bindings=bindings,
                hardware=hardware,
                measured_domain_capabilities=measured,
                output_dir=tmp,
            )

            self.assertEqual(
                set(outputs), {"prior", "network", "tensor_bindings", "layer_spec", "camc_profile"}
            )
            self.assertTrue((Path(tmp) / "prior.json").is_file())
            exported_prior = validate_prior_artifact(json.loads(outputs["prior"].read_text()))
            self.assertEqual(len(exported_prior.operator_ids), 479)
            self.assertEqual(
                len(json.loads(outputs["network"].read_text())["networks"]), 1
            )


if __name__ == "__main__":
    unittest.main()
