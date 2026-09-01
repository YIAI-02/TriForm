"""Build the fixed 28-layer Qwen-1.8B prefill-only CAMC inputs."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hetinfer_camc_profile_export import build_camc_profile
from hetinfer_prior import DOPSPriorArtifact, validate_prior_artifact


GRAPH_ID = "qwen-1.8b-28layer-1npu2pim"
WORKLOAD_ID = "b1-p128-h128"
LAYER_COUNT = 28
LAYER_SLOTS = (
    "ln",
    "q",
    "k",
    "v",
    "k_write",
    "v_write",
    "qk",
    "softmax",
    "sv",
    "o",
    "add1",
    "ln2",
    "ffn_w1",
    "ffn_w3",
    "swiglu",
    "ffn_w2",
    "add2",
)
GLOBAL_SLOTS = ("embedding", "final_norm", "lm_head")
WEIGHT_SLOTS = {
    "embedding",
    "q",
    "k",
    "v",
    "o",
    "ffn_w1",
    "ffn_w3",
    "ffn_w2",
    "lm_head",
}
KV_SLOTS = {"k", "v", "k_write", "v_write", "qk", "sv"}


def _object(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be an object")
    return value


def _array(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{context} must be an array")
    return value


def _positive_number(value: object, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RuntimeError(f"{context} must be a finite number > 0")
    return float(value)


def _selected_network(network_manifest: Mapping[str, Any]) -> dict[str, Any]:
    root = _object(network_manifest, "network")
    if root.get("schema") != "dops.hetinfer_network.v1":
        raise RuntimeError("network schema must be dops.hetinfer_network.v1")
    if root.get("schema_version") != 1:
        raise RuntimeError("network schema_version must be 1")
    networks = _array(root.get("networks"), "network.networks")
    if not networks:
        raise RuntimeError("network.networks must contain network_index 0")
    selected = copy.deepcopy(dict(_object(networks[0], "network[0]")))
    if selected.get("graph_id") != GRAPH_ID:
        raise RuntimeError(f"network[0].graph_id must be {GRAPH_ID!r}")
    if selected.get("workload_id") != WORKLOAD_ID:
        raise RuntimeError(f"network[0].workload_id must be {WORKLOAD_ID!r}")
    if selected.get("phase") != "prefill":
        raise RuntimeError("network[0] must be prefill")
    workload = _object(selected.get("workload"), "network[0].workload")
    expected_workload = {
        "batch": 1,
        "sequence_length": 128,
        "scheduled_tokens": 128,
    }
    for field, expected in expected_workload.items():
        if workload.get(field) != expected:
            raise RuntimeError(
                f"network[0].workload.{field} must equal {expected}"
            )
    _validate_dense_operators(selected)
    return selected


def _validate_dense_operators(network: Mapping[str, Any]) -> None:
    operators = _array(network.get("operators"), "network[0].operators")
    if len(operators) != LAYER_COUNT * len(LAYER_SLOTS) + len(GLOBAL_SLOTS):
        raise RuntimeError("network[0] must contain exactly 479 operators")
    layer_operators: dict[int, dict[str, Mapping[str, Any]]] = {
        index: {} for index in range(LAYER_COUNT)
    }
    global_operators: dict[str, Mapping[str, Any]] = {}
    op_ids: set[str] = set()
    for raw_operator in operators:
        operator = _object(raw_operator, "network[0].operator")
        op_id = operator.get("op_id")
        if not isinstance(op_id, str) or not op_id or op_id in op_ids:
            raise RuntimeError("network[0] operator op_id values must be unique")
        op_ids.add(op_id)
        slot = operator.get("canonical_op_slot")
        if not isinstance(slot, str) or not slot:
            raise RuntimeError(f"operator {op_id!r} lacks canonical_op_slot")
        op_role = operator.get("op_role")
        if not isinstance(op_role, str) or not op_role:
            raise RuntimeError(f"operator {op_id!r} lacks op_role")
        layer_index = operator.get("layer_index")
        if layer_index is None:
            if operator.get("block_type") != "OTHER":
                raise RuntimeError(f"global operator {op_id!r} must use block_type OTHER")
            if slot in global_operators:
                raise RuntimeError(f"duplicate global slot {slot!r}")
            global_operators[slot] = operator
            continue
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index not in layer_operators
        ):
            raise RuntimeError(f"operator {op_id!r} has invalid layer_index")
        if operator.get("block_id") != f"layer:{layer_index}":
            raise RuntimeError(f"operator {op_id!r} has invalid block_id")
        if operator.get("block_type") != "DENSE_BLOCK":
            raise RuntimeError(f"operator {op_id!r} must be a DENSE_BLOCK")
        if operator.get("repeat_index") != layer_index:
            raise RuntimeError(f"operator {op_id!r} has invalid repeat_index")
        if operator.get("total_repeats") != LAYER_COUNT:
            raise RuntimeError(f"operator {op_id!r} must declare 28 repeats")
        if slot in layer_operators[layer_index]:
            raise RuntimeError(f"duplicate layer {layer_index} slot {slot!r}")
        layer_operators[layer_index][slot] = operator
    if set(global_operators) != set(GLOBAL_SLOTS):
        raise RuntimeError("network[0] global slots must be embedding/final_norm/lm_head")
    for layer_index, layer in layer_operators.items():
        if set(layer) != set(LAYER_SLOTS):
            raise RuntimeError(
                f"network[0] layer {layer_index} must contain the 17 dense slots"
            )

    roles_by_slot = {
        slot: {
            str(layer_operators[layer_index][slot]["op_role"])
            for layer_index in range(LAYER_COUNT)
        }
        for slot in LAYER_SLOTS
    }
    inconsistent_roles = {
        slot: sorted(roles)
        for slot, roles in roles_by_slot.items()
        if len(roles) != 1
    }
    if inconsistent_roles:
        raise RuntimeError(
            "same-slot operators must share op_role across all layers: "
            f"{inconsistent_roles}"
        )

    def op_id(layer_index: int, slot: str) -> str:
        return str(layer_operators[layer_index][slot]["op_id"])

    previous_add2 = str(global_operators["embedding"]["op_id"])
    for layer_index, layer in layer_operators.items():
        expected_dependencies = {
            "ln": {previous_add2},
            "q": {op_id(layer_index, "ln")},
            "k": {op_id(layer_index, "ln")},
            "v": {op_id(layer_index, "ln")},
            "k_write": {op_id(layer_index, "k")},
            "v_write": {op_id(layer_index, "v")},
            "qk": {op_id(layer_index, "q"), op_id(layer_index, "k")},
            "softmax": {op_id(layer_index, "qk")},
            "sv": {op_id(layer_index, "v"), op_id(layer_index, "softmax")},
            "o": {op_id(layer_index, "sv")},
            "add1": (
                {op_id(layer_index, "o")}
                if layer_index == 0
                else {previous_add2, op_id(layer_index, "o")}
            ),
            "ln2": {op_id(layer_index, "add1")},
            "ffn_w1": {op_id(layer_index, "ln2")},
            "ffn_w3": {op_id(layer_index, "ln2")},
            "swiglu": {
                op_id(layer_index, "ffn_w1"),
                op_id(layer_index, "ffn_w3"),
            },
            "ffn_w2": {op_id(layer_index, "swiglu")},
            "add2": {
                op_id(layer_index, "add1"),
                op_id(layer_index, "ffn_w2"),
            },
        }
        for slot, operator in layer.items():
            raw_dependencies = _array(
                operator.get("dependencies"), f"{operator['op_id']}.dependencies"
            )
            actual_dependencies = set(raw_dependencies)
            if (
                len(actual_dependencies) != len(raw_dependencies)
                or actual_dependencies != expected_dependencies[slot]
            ):
                raise RuntimeError(
                    f"operator {operator['op_id']!r} violates the dense "
                    "dependency contract"
                )
        previous_add2 = op_id(layer_index, "add2")

    expected_global_dependencies = {
        "embedding": set(),
        "final_norm": {previous_add2},
        "lm_head": {str(global_operators["final_norm"]["op_id"])},
    }
    for slot, operator in global_operators.items():
        raw_dependencies = _array(
            operator.get("dependencies"), f"{operator['op_id']}.dependencies"
        )
        if (
            len(set(raw_dependencies)) != len(raw_dependencies)
            or set(raw_dependencies) != expected_global_dependencies[slot]
        ):
            raise RuntimeError(
                f"operator {operator['op_id']!r} violates the dense "
                "dependency contract"
            )


def _project_network(network: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "dops.hetinfer_network.v1",
        "schema_version": 1,
        "networks": [copy.deepcopy(dict(network))],
    }


def _project_tensor_bindings(
    tensor_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    root = _object(tensor_bindings, "tensor_bindings")
    if root.get("schema") != "dops.hetinfer_tensor_bindings.v1":
        raise RuntimeError(
            "tensor_bindings schema must be dops.hetinfer_tensor_bindings.v1"
        )
    if root.get("schema_version") != 1:
        raise RuntimeError("tensor_bindings schema_version must be 1")
    if root.get("graph_id") != GRAPH_ID or root.get("workload_id") != WORKLOAD_ID:
        raise RuntimeError("tensor_bindings graph/workload must match fixed prefill")
    selected = [
        copy.deepcopy(dict(_object(item, "tensor binding")))
        for item in _array(root.get("bindings"), "tensor_bindings.bindings")
        if _object(item, "tensor binding").get("network_index") == 0
    ]
    if not selected:
        raise RuntimeError("tensor_bindings must cover network_index 0")
    layer_indices = {
        item.get("layer_index") for item in selected if item.get("layer_index") is not None
    }
    if layer_indices != set(range(LAYER_COUNT)):
        raise RuntimeError("network_index 0 tensor bindings must cover all 28 layers")
    return {
        "schema": "dops.hetinfer_tensor_bindings.v1",
        "schema_version": 1,
        "graph_id": GRAPH_ID,
        "workload_id": WORKLOAD_ID,
        "bindings": selected,
    }


def _topological_order(network: Mapping[str, Any]) -> list[str]:
    operators = [
        _object(item, "network[0].operator")
        for item in _array(network.get("operators"), "network[0].operators")
    ]
    by_id = {str(operator["op_id"]): operator for operator in operators}
    indegree = {op_id: 0 for op_id in by_id}
    consumers = {op_id: [] for op_id in by_id}
    for op_id, operator in by_id.items():
        dependencies = _array(operator.get("dependencies"), f"{op_id}.dependencies")
        if any(dependency not in by_id for dependency in dependencies):
            raise RuntimeError(f"operator {op_id!r} has a dependency outside prefill")
        indegree[op_id] = len(dependencies)
        for dependency in dependencies:
            consumers[dependency].append(op_id)

    slot_rank = {slot: index for index, slot in enumerate(LAYER_SLOTS)}

    def order_key(op_id: str) -> tuple[int, int, str]:
        operator = by_id[op_id]
        layer_index = operator.get("layer_index")
        slot = str(operator["canonical_op_slot"])
        if slot == "embedding":
            return (-1, 0, op_id)
        if layer_index is not None:
            return (int(layer_index), slot_rank[slot], op_id)
        return (LAYER_COUNT, GLOBAL_SLOTS.index(slot), op_id)

    ready = sorted(
        (op_id for op_id, degree in indegree.items() if degree == 0),
        key=order_key,
    )
    result: list[str] = []
    while ready:
        op_id = ready.pop(0)
        result.append(op_id)
        for consumer in consumers[op_id]:
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                ready.append(consumer)
        ready.sort(key=order_key)
    if len(result) != len(by_id):
        raise RuntimeError("network[0] operators must form a DAG")
    return result


def _domain_contract(
    hardware: Mapping[str, Any],
    schedulable_devices: set[str],
) -> tuple[dict[str, str], dict[str, dict[str, float | int]]]:
    root = _object(hardware.get("hardware"), "hardware.hardware")
    raw_devices = _array(root.get("devices"), "hardware.hardware.devices")
    devices = {
        str(_object(item, "hardware device").get("name")): _object(
            item, "hardware device"
        )
        for item in raw_devices
    }
    if set(devices) & schedulable_devices != schedulable_devices:
        missing = sorted(schedulable_devices - set(devices))
        raise RuntimeError(f"hardware lacks schedulable devices: {missing}")
    domains: dict[str, str] = {}
    accumulators = {
        "NPU": {"compute": 0.0, "bandwidth": 0.0, "queues": 0},
        "PIM": {"compute": 0.0, "bandwidth": 0.0, "queues": 0},
    }
    for device_id in sorted(schedulable_devices):
        device = devices[device_id]
        device_type = device.get("type")
        if device_type not in {"npu", "pim"}:
            raise RuntimeError(
                f"schedulable device {device_id!r} must have type npu or pim"
            )
        domain = str(device_type).upper()
        domains[device_id] = domain
        accumulators[domain]["compute"] += (
            _positive_number(device.get("tflops"), f"{device_id}.tflops") * 1e12
        )
        accumulators[domain]["bandwidth"] += (
            _positive_number(device.get("mem_bw_GBs"), f"{device_id}.mem_bw_GBs")
            * 1e9
        )
        accumulators[domain]["queues"] += 1
    if set(domains.values()) != {"NPU", "PIM"}:
        raise RuntimeError("schedulable devices must contain NPU and PIM domains")
    capabilities = {
        domain: {
            "effective_compute_flops_per_s": values["compute"],
            "effective_bandwidth_bytes_per_s": values["bandwidth"],
            "queue_count": values["queues"],
        }
        for domain, values in accumulators.items()
    }
    return domains, capabilities


def _layer_spec(
    prior: DOPSPriorArtifact,
    network: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    operators = {
        str(_object(item, "network[0].operator")["op_id"]): _object(
            item, "network[0].operator"
        )
        for item in _array(network.get("operators"), "network[0].operators")
    }
    selected_ids = set(operators)
    schedulable_devices = {
        device_id
        for op_id in selected_ids
        for device_id in prior.legal_devices[op_id]
    }
    domains, capabilities = _domain_contract(hardware, schedulable_devices)
    kv_home_by_layer: dict[int, str] = {}
    for layer_index in range(LAYER_COUNT):
        write_nodes = [
            operator
            for operator in operators.values()
            if operator.get("layer_index") == layer_index
            and operator.get("canonical_op_slot") in {"k_write", "v_write"}
        ]
        homes = {
            prior.expert_device(str(operator["op_id"])) for operator in write_nodes
        }
        if len(write_nodes) != 2 or len(homes) != 1:
            raise RuntimeError(
                f"layer {layer_index} must have one shared K/V cache home"
            )
        kv_home = homes.pop()
        if domains.get(kv_home) != "PIM":
            raise RuntimeError(f"layer {layer_index} K/V cache home must be PIM")
        kv_home_by_layer[layer_index] = kv_home

    order = _topological_order(network)
    nodes = []
    for op_id in order:
        operator = operators[op_id]
        slot = str(operator["canonical_op_slot"])
        layer_index = operator.get("layer_index")
        parallel_group = None
        if layer_index is not None and slot in {"q", "k", "v"}:
            parallel_group = f"layer:{layer_index}:attention-qkv"
        elif layer_index is not None and slot in {"ffn_w1", "ffn_w3"}:
            parallel_group = f"layer:{layer_index}:ffn-up"
        nodes.append(
            {
                "op_id": op_id,
                "operator_family": slot,
                "placement_supernode": op_id,
                "parallel_group_hint": parallel_group,
                "weight_home": (
                    prior.expert_device(op_id) if slot in WEIGHT_SLOTS else None
                ),
                "kv_home": (
                    kv_home_by_layer[int(layer_index)]
                    if layer_index is not None and slot in KV_SLOTS
                    else None
                ),
                "expert_id": None,
                "expert_service_buckets": [],
            }
        )
    return {
        "graph_id": GRAPH_ID,
        "workload_id": WORKLOAD_ID,
        "device_domains": domains,
        "layers": [
            {
                "network_index": 0,
                "layer_class": "dense",
                "phase": "prefill",
                "shape_bucket": "b1-prefill128",
                "capability_basis": "compute",
                "domain_capabilities": capabilities,
                "default_order": order,
                "nodes": nodes,
            }
        ],
    }


def build_dense_prefill_camc_bundle(
    *,
    prior_artifact: DOPSPriorArtifact | Mapping[str, Any],
    network_manifest: Mapping[str, Any],
    tensor_bindings: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prior = (
        prior_artifact
        if isinstance(prior_artifact, DOPSPriorArtifact)
        else validate_prior_artifact(prior_artifact)
    )
    selected_network = _selected_network(network_manifest)
    projected_network = _project_network(selected_network)
    projected_bindings = _project_tensor_bindings(tensor_bindings)
    spec = _layer_spec(prior, selected_network, hardware)
    profile = build_camc_profile(
        prior_artifact=prior,
        network_manifest=projected_network,
        tensor_bindings=projected_bindings,
        layer_spec=spec,
        allow_prior_operator_subset=True,
    )
    return projected_network, projected_bindings, profile


def export_dense_prefill_camc_bundle(
    *,
    prior_artifact: DOPSPriorArtifact | Mapping[str, Any],
    network_manifest: Mapping[str, Any],
    tensor_bindings: Mapping[str, Any],
    hardware: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    network, bindings, profile = build_dense_prefill_camc_bundle(
        prior_artifact=prior_artifact,
        network_manifest=network_manifest,
        tensor_bindings=tensor_bindings,
        hardware=hardware,
    )
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "network": root / "network.json",
        "tensor_bindings": root / "tensor_bindings.json",
        "camc_profile": root / "camc_profile.json",
    }
    for name, payload in (
        ("network", network),
        ("tensor_bindings", bindings),
        ("camc_profile", profile),
    ):
        outputs[name].write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return outputs


__all__ = [
    "GLOBAL_SLOTS",
    "GRAPH_ID",
    "LAYER_COUNT",
    "LAYER_SLOTS",
    "WORKLOAD_ID",
    "build_dense_prefill_camc_bundle",
    "export_dense_prefill_camc_bundle",
]
