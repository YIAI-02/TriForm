"""Export opaque DOPS tensor identities with canonical runtime bindings."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TENSOR_BINDINGS_SCHEMA = "dops.hetinfer_tensor_bindings.v1"
TENSOR_BINDINGS_SCHEMA_VERSION = 1


def _snapshot_operator(operator: Mapping[str, Any]) -> dict[str, Any]:
    attrs = operator["network_metadata"]["node_attrs"]
    layer = attrs.get("layer_index", attrs.get("layer"))
    return {
        "layer_index": None if layer is None else int(layer),
        "canonical_tensor_slot": str(attrs["canonical_op_slot"]),
    }


def _bindings_for_network(
    *,
    inputs: Sequence[Mapping[str, Any]],
    operators: Mapping[str, Mapping[str, Any]],
    network_index: int,
) -> list[dict[str, Any]]:
    inputs_by_tensor: dict[str, list[Mapping[str, Any]]] = {}
    for item in inputs:
        if item["semantics"] == "barrier":
            continue
        inputs_by_tensor.setdefault(item["tensor_id"], []).append(item)

    bindings: list[dict[str, Any]] = []
    for tensor_id in sorted(inputs_by_tensor):
        inputs = inputs_by_tensor[tensor_id]
        representative = inputs[0]
        producer_op_id = representative["producer_op_id"]

        if producer_op_id is None:
            consumer = operators[representative["consumer_op_id"]]
            consumer_slot = consumer["canonical_tensor_slot"]
            if consumer_slot == "qk":
                layer_index = consumer["layer_index"]
                canonical_tensor_slot = "k"
                persistence = "kv-read"
            elif consumer_slot == "sv":
                layer_index = consumer["layer_index"]
                canonical_tensor_slot = "v"
                persistence = "kv-read"
            else:
                layer_index = None
                canonical_tensor_slot = "request_input"
                persistence = "request_input"
        else:
            producer = operators[producer_op_id]
            consumer_slots = {
                operators[item["consumer_op_id"]]["canonical_tensor_slot"]
                for item in inputs
            }
            layer_index = producer["layer_index"]
            if "k_write" in consumer_slots:
                canonical_tensor_slot = "k"
                persistence = "kv-append"
            elif "v_write" in consumer_slots:
                canonical_tensor_slot = "v"
                persistence = "kv-append"
            else:
                canonical_tensor_slot = producer["canonical_tensor_slot"]
                persistence = "transient"

        bindings.append(
            {
                "network_index": int(network_index),
                "tensor_id": tensor_id,
                "layer_index": layer_index,
                "canonical_tensor_slot": canonical_tensor_slot,
                "persistence": persistence,
                "size_bytes": int(representative["bytes"]),
            }
        )
    return bindings


def _snapshot_bindings(
    snapshot: Mapping[str, Any], *, network_index: int
) -> list[dict[str, Any]]:
    operators = {
        operator["op_id"]: _snapshot_operator(operator)
        for operator in snapshot["operators"]
    }
    return _bindings_for_network(
        inputs=snapshot["inputs"],
        operators=operators,
        network_index=network_index,
    )


def build_tensor_bindings_manifest(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    prior_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": TENSOR_BINDINGS_SCHEMA,
        "schema_version": TENSOR_BINDINGS_SCHEMA_VERSION,
        "graph_id": str(prior_artifact["graph_id"]),
        "workload_id": str(prior_artifact["workload_id"]),
        "bindings": [
            binding
            for network_index, snapshot in enumerate(snapshots)
            for binding in _snapshot_bindings(
                snapshot, network_index=network_index
            )
        ],
    }


def build_tensor_bindings_manifest_from_artifacts(
    *,
    prior_artifact: Mapping[str, Any],
    network_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    graph_id = str(prior_artifact["graph_id"])
    workload_id = str(prior_artifact["workload_id"])
    operators_by_network: list[dict[str, dict[str, Any]]] = []
    network_index_by_op_id: dict[str, int] = {}
    for network_index, network in enumerate(network_manifest["networks"]):
        if str(network["graph_id"]) != graph_id:
            raise RuntimeError(
                f"network[{network_index}] graph_id does not match the prior"
            )
        if str(network["workload_id"]) != workload_id:
            raise RuntimeError(
                f"network[{network_index}] workload_id does not match the prior"
            )
        operators = {
            operator["op_id"]: {
                "layer_index": operator["layer_index"],
                "canonical_tensor_slot": operator["canonical_op_slot"],
            }
            for operator in network["operators"]
        }
        operators_by_network.append(operators)
        for op_id in operators:
            if op_id in network_index_by_op_id:
                raise RuntimeError(f"duplicate network operator op_id: {op_id!r}")
            network_index_by_op_id[op_id] = network_index

    inputs_by_network: list[list[Mapping[str, Any]]] = [
        [] for _ in operators_by_network
    ]
    for item in prior_artifact["inputs"]:
        consumer_op_id = item["consumer_op_id"]
        if consumer_op_id not in network_index_by_op_id:
            raise RuntimeError(
                f"prior input consumer_op_id is absent from network.json: "
                f"{consumer_op_id!r}"
            )
        network_index = network_index_by_op_id[consumer_op_id]
        producer_op_id = item["producer_op_id"]
        if producer_op_id is not None:
            if producer_op_id not in network_index_by_op_id:
                raise RuntimeError(
                    f"prior input producer_op_id is absent from network.json: "
                    f"{producer_op_id!r}"
                )
            if network_index_by_op_id[producer_op_id] != network_index:
                raise RuntimeError(
                    "prior input producer and consumer belong to different networks"
                )
        inputs_by_network[network_index].append(item)

    return {
        "schema": TENSOR_BINDINGS_SCHEMA,
        "schema_version": TENSOR_BINDINGS_SCHEMA_VERSION,
        "graph_id": graph_id,
        "workload_id": workload_id,
        "bindings": [
            binding
            for network_index, operators in enumerate(operators_by_network)
            for binding in _bindings_for_network(
                inputs=inputs_by_network[network_index],
                operators=operators,
                network_index=network_index,
            )
        ],
    }


def export_tensor_bindings_manifest(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    prior_artifact: Mapping[str, Any],
    output: str | Path,
) -> Path:
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            build_tensor_bindings_manifest(
                snapshots=snapshots,
                prior_artifact=prior_artifact,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def export_tensor_bindings_manifest_from_artifacts(
    *,
    prior_artifact: Mapping[str, Any],
    network_manifest: Mapping[str, Any],
    output: str | Path,
) -> Path:
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            build_tensor_bindings_manifest_from_artifacts(
                prior_artifact=prior_artifact,
                network_manifest=network_manifest,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "TENSOR_BINDINGS_SCHEMA",
    "TENSOR_BINDINGS_SCHEMA_VERSION",
    "build_tensor_bindings_manifest",
    "build_tensor_bindings_manifest_from_artifacts",
    "export_tensor_bindings_manifest",
    "export_tensor_bindings_manifest_from_artifacts",
]
