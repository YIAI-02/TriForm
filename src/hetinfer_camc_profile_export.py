"""Build a strict CAMC deployment profile from existing Het-Infer sidecars."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hetinfer_prior import DOPSPriorArtifact, validate_prior_artifact


CAMC_PROFILE_SCHEMA = "dops.hetinfer_camc_profile.v1"
CAMC_PROFILE_SCHEMA_VERSION = 1
CAMC_DOMAINS = ("NPU", "PIM")
CAMC_PHASE_TO_NETWORK_PHASE = {
    "prefill": "prefill",
    "decode": "decode",
    "verify": "decode",
    "draft": "decode",
}


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be an object")
    return value


def _array(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{context} must be an array")
    return value


def _exact_fields(
    value: Mapping[str, Any], required: set[str], context: str
) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing or extra:
        raise RuntimeError(
            f"{context} fields mismatch: missing={missing!r}, extra={extra!r}"
        )


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{context} must be a non-empty string")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{context} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, context: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RuntimeError(f"{context} must be a finite number > 0")
    return value


def _header(value: Mapping[str, Any], schema: str, context: str) -> None:
    if value.get("schema") != schema:
        raise RuntimeError(f"{context} schema must equal {schema!r}")
    if value.get("schema_version") != 1:
        raise RuntimeError(f"{context} schema_version must equal 1")


def _sidecar_index(
    *,
    prior_artifact: DOPSPriorArtifact | Mapping[str, Any],
    network_manifest: Mapping[str, Any],
    tensor_bindings: Mapping[str, Any],
) -> tuple[
    DOPSPriorArtifact,
    str,
    str,
    list[dict[str, Any]],
    dict[str, tuple[str, ...]],
]:
    prior = (
        prior_artifact
        if isinstance(prior_artifact, DOPSPriorArtifact)
        else validate_prior_artifact(prior_artifact)
    )
    prior_payload = prior.payload
    graph_id = prior_payload["graph_id"]
    workload_id = prior_payload["workload_id"]
    dependencies = {
        item["op_id"]: tuple(item["dependencies"])
        for item in prior_payload["operators"]
    }

    network_root = _object(network_manifest, "network")
    _header(network_root, "dops.hetinfer_network.v1", "network")
    networks: list[dict[str, Any]] = []
    op_network: dict[str, int] = {}
    for network_index, raw_network in enumerate(
        _array(network_root.get("networks"), "network.networks")
    ):
        network = _object(raw_network, f"network[{network_index}]")
        if network.get("graph_id") != graph_id:
            raise RuntimeError(
                f"network[{network_index}] graph_id does not match the prior"
            )
        if network.get("workload_id") != workload_id:
            raise RuntimeError(
                f"network[{network_index}] workload_id does not match the prior"
            )
        phase = _string(network.get("phase"), f"network[{network_index}].phase")
        if phase not in {"prefill", "decode"}:
            raise RuntimeError(
                f"network[{network_index}].phase must be 'prefill' or 'decode'"
            )
        operator_ids: list[str] = []
        for raw_operator in _array(
            network.get("operators"), f"network[{network_index}].operators"
        ):
            operator = _object(raw_operator, f"network[{network_index}].operator")
            op_id = _string(
                operator.get("op_id"), f"network[{network_index}].operator.op_id"
            )
            if op_id in op_network:
                raise RuntimeError(f"duplicate network operator op_id: {op_id!r}")
            if op_id not in dependencies:
                raise RuntimeError(f"network has unknown prior op_id: {op_id!r}")
            operator_dependencies = [
                _string(
                    item,
                    f"network[{network_index}].operator.dependencies",
                )
                for item in _array(
                    operator.get("dependencies"),
                    f"network[{network_index}].operator.dependencies",
                )
            ]
            if len(operator_dependencies) != len(set(operator_dependencies)):
                raise RuntimeError(
                    f"network dependencies for {op_id!r} must contain unique strings"
                )
            if tuple(sorted(operator_dependencies)) != tuple(
                sorted(dependencies[op_id])
            ):
                raise RuntimeError(
                    f"network dependencies for {op_id!r} do not match the prior"
                )
            op_network[op_id] = network_index
            operator_ids.append(op_id)
        if not operator_ids:
            raise RuntimeError(f"network[{network_index}] has no operators")
        networks.append({"phase": phase, "operator_ids": operator_ids})

    if set(op_network) != set(prior.operator_ids):
        raise RuntimeError("network operators must exactly cover prior operators")
    for op_id, op_dependencies in dependencies.items():
        if any(op_network[dependency] != op_network[op_id] for dependency in op_dependencies):
            raise RuntimeError(
                f"prior dependencies for {op_id!r} cross network boundaries"
            )

    bindings_root = _object(tensor_bindings, "tensor_bindings")
    _header(
        bindings_root,
        "dops.hetinfer_tensor_bindings.v1",
        "tensor_bindings",
    )
    if bindings_root.get("graph_id") != graph_id:
        raise RuntimeError("tensor_bindings graph_id does not match the prior")
    if bindings_root.get("workload_id") != workload_id:
        raise RuntimeError("tensor_bindings workload_id does not match the prior")
    binding_networks = {
        _integer(
            _object(item, "tensor binding").get("network_index"),
            "tensor binding network_index",
        )
        for item in _array(
            bindings_root.get("bindings"), "tensor_bindings.bindings"
        )
    }
    if binding_networks != set(range(len(networks))):
        raise RuntimeError(
            "tensor_bindings must cover every network_index in network.json"
        )
    return prior, graph_id, workload_id, networks, dependencies


def _domain_capabilities(value: Any, context: str) -> dict[str, dict[str, Any]]:
    root = _object(value, context)
    _exact_fields(root, set(CAMC_DOMAINS), context)
    fields = {
        "effective_compute_flops_per_s",
        "effective_bandwidth_bytes_per_s",
        "queue_count",
    }
    result: dict[str, dict[str, Any]] = {}
    for domain in CAMC_DOMAINS:
        capability = _object(root[domain], f"{context}.{domain}")
        _exact_fields(capability, fields, f"{context}.{domain}")
        result[domain] = {
            "effective_compute_flops_per_s": _positive_number(
                capability["effective_compute_flops_per_s"],
                f"{context}.{domain}.effective_compute_flops_per_s",
            ),
            "effective_bandwidth_bytes_per_s": _positive_number(
                capability["effective_bandwidth_bytes_per_s"],
                f"{context}.{domain}.effective_bandwidth_bytes_per_s",
            ),
            "queue_count": _integer(
                capability["queue_count"],
                f"{context}.{domain}.queue_count",
                minimum=1,
            ),
        }
    return result


def _nullable_string(value: Any, context: str) -> str | None:
    return None if value is None else _string(value, context)


def _expert_service_buckets(
    value: Any,
    *,
    context: str,
    legal_devices: tuple[str, ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw_bucket in enumerate(_array(value, context)):
        bucket_context = f"{context}[{index}]"
        bucket = _object(raw_bucket, bucket_context)
        _exact_fields(
            bucket,
            {"min_tokens", "max_tokens", "service_time_s"},
            bucket_context,
        )
        min_tokens = _integer(
            bucket["min_tokens"], f"{bucket_context}.min_tokens", minimum=1
        )
        max_tokens = _integer(
            bucket["max_tokens"],
            f"{bucket_context}.max_tokens",
            minimum=min_tokens,
        )
        raw_service = _object(
            bucket["service_time_s"], f"{bucket_context}.service_time_s"
        )
        if set(raw_service) != set(legal_devices):
            raise RuntimeError(
                f"{bucket_context}.service_time_s must exactly cover legal devices"
            )
        result.append(
            {
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
                "service_time_s": {
                    device_id: _positive_number(
                        raw_service[device_id],
                        f"{bucket_context}.service_time_s[{device_id!r}]",
                    )
                    for device_id in legal_devices
                },
            }
        )
    result.sort(key=lambda item: (item["min_tokens"], item["max_tokens"]))
    for previous, current in zip(result, result[1:]):
        if current["min_tokens"] <= previous["max_tokens"]:
            raise RuntimeError(f"{context} token ranges must not overlap")
    return result


def _node_spec(
    value: Any,
    *,
    context: str,
    known_devices: set[str],
    legal_devices_by_op: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    node = _object(value, context)
    _exact_fields(
        node,
        {
            "op_id",
            "operator_family",
            "placement_supernode",
            "parallel_group_hint",
            "weight_home",
            "kv_home",
            "expert_id",
            "expert_service_buckets",
        },
        context,
    )
    op_id = _string(node["op_id"], f"{context}.op_id")
    if op_id not in legal_devices_by_op:
        raise RuntimeError(f"{context}.op_id is not present in the prior")
    expert_id = _nullable_string(node["expert_id"], f"{context}.expert_id")
    expert_service_buckets = _expert_service_buckets(
        node["expert_service_buckets"],
        context=f"{context}.expert_service_buckets",
        legal_devices=legal_devices_by_op[op_id],
    )
    if expert_id is None and expert_service_buckets:
        raise RuntimeError(
            f"{context}.expert_service_buckets must be empty for a non-expert node"
        )
    if expert_id is not None and not expert_service_buckets:
        raise RuntimeError(
            f"{context}.expert_service_buckets must be non-empty for an expert node"
        )
    result = {
        "op_id": op_id,
        "operator_family": _string(
            node["operator_family"], f"{context}.operator_family"
        ),
        "placement_supernode": _string(
            node["placement_supernode"], f"{context}.placement_supernode"
        ),
        "parallel_group_hint": _nullable_string(
            node["parallel_group_hint"], f"{context}.parallel_group_hint"
        ),
        "weight_home": _nullable_string(
            node["weight_home"], f"{context}.weight_home"
        ),
        "kv_home": _nullable_string(node["kv_home"], f"{context}.kv_home"),
        "expert_id": expert_id,
        "expert_service_buckets": expert_service_buckets,
    }
    for field in ("weight_home", "kv_home"):
        if result[field] is not None and result[field] not in known_devices:
            raise RuntimeError(
                f"{context}.{field} must be a declared physical device"
            )
    return result


def _validate_supernodes(
    *,
    node_specs: Mapping[str, Mapping[str, Any]],
    legal_devices: Mapping[str, tuple[str, ...]],
    default_devices: Mapping[str, str],
    dependencies: Mapping[str, tuple[str, ...]],
    context: str,
) -> None:
    members: dict[str, list[str]] = {}
    for op_id, node in node_specs.items():
        members.setdefault(node["placement_supernode"], []).append(op_id)

    hints: dict[str, str | None] = {}
    neighbors = {op_id: set() for op_id in node_specs}
    for op_id in node_specs:
        for dependency in dependencies[op_id]:
            if (
                node_specs[dependency]["placement_supernode"]
                == node_specs[op_id]["placement_supernode"]
            ):
                neighbors[op_id].add(dependency)
                neighbors[dependency].add(op_id)

    for supernode, op_ids in members.items():
        first = op_ids[0]
        expected_legal = legal_devices[first]
        expected_default = default_devices[first]
        expected_hint = node_specs[first]["parallel_group_hint"]
        for op_id in op_ids[1:]:
            if legal_devices[op_id] != expected_legal:
                raise RuntimeError(
                    f"{context} supernode {supernode!r} must have identical "
                    "legal_devices"
                )
            if default_devices[op_id] != expected_default:
                raise RuntimeError(
                    f"{context} supernode {supernode!r} must have one "
                    "default_device"
                )
            if node_specs[op_id]["parallel_group_hint"] != expected_hint:
                raise RuntimeError(
                    f"{context} supernode {supernode!r} must have one "
                    "parallel_group_hint"
                )
        hints[supernode] = expected_hint
        if len(op_ids) > 1:
            visited = {first}
            pending = [first]
            while pending:
                current = pending.pop()
                for neighbor in neighbors[current] - visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
            if visited != set(op_ids):
                raise RuntimeError(
                    f"{context} supernode {supernode!r} must be internally "
                    "dependency-connected"
                )

    contracted = {supernode: set() for supernode in members}
    for op_id in node_specs:
        destination = node_specs[op_id]["placement_supernode"]
        for dependency in dependencies[op_id]:
            source = node_specs[dependency]["placement_supernode"]
            if source != destination:
                contracted[source].add(destination)

    def reaches(source: str, target: str) -> bool:
        pending = list(contracted[source])
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current not in visited:
                visited.add(current)
                pending.extend(contracted[current] - visited)
        return False

    by_hint: dict[str, list[str]] = {}
    for supernode, hint in hints.items():
        if hint is None:
            continue
        for peer in by_hint.setdefault(hint, []):
            if reaches(supernode, peer) or reaches(peer, supernode):
                raise RuntimeError(
                    f"{context} parallel_group_hint {hint!r} requires mutually "
                    "unreachable supernodes"
                )
        by_hint[hint].append(supernode)


def build_camc_profile(
    *,
    prior_artifact: DOPSPriorArtifact | Mapping[str, Any],
    network_manifest: Mapping[str, Any],
    tensor_bindings: Mapping[str, Any],
    layer_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge explicit CAMC deployment annotations with validated DOPS sidecars."""

    prior, graph_id, workload_id, networks, dependencies = _sidecar_index(
        prior_artifact=prior_artifact,
        network_manifest=network_manifest,
        tensor_bindings=tensor_bindings,
    )
    spec = _object(layer_spec, "layer_spec")
    _exact_fields(
        spec,
        {"graph_id", "workload_id", "device_domains", "layers"},
        "layer_spec",
    )
    if spec["graph_id"] != graph_id:
        raise RuntimeError("layer_spec graph_id does not match the prior")
    if spec["workload_id"] != workload_id:
        raise RuntimeError("layer_spec workload_id does not match the prior")

    raw_domains = _object(spec["device_domains"], "layer_spec.device_domains")
    if set(raw_domains) != set(prior.device_ids):
        raise RuntimeError(
            "layer_spec.device_domains must exactly cover prior physical devices"
        )
    device_domains: dict[str, str] = {}
    for device_id in sorted(prior.device_ids):
        domain = raw_domains[device_id]
        if domain not in CAMC_DOMAINS:
            raise RuntimeError(
                f"layer_spec.device_domains[{device_id!r}] must be NPU or PIM"
            )
        device_domains[device_id] = domain
    if set(device_domains.values()) != set(CAMC_DOMAINS):
        raise RuntimeError("layer_spec.device_domains must contain NPU and PIM")

    layer_fields = {
        "network_index",
        "layer_class",
        "phase",
        "shape_bucket",
        "capability_basis",
        "domain_capabilities",
        "default_order",
        "nodes",
    }
    layers_by_index: dict[int, dict[str, Any]] = {}
    known_devices = set(prior.device_ids)
    for spec_index, raw_layer in enumerate(_array(spec["layers"], "layer_spec.layers")):
        context = f"layer_spec.layers[{spec_index}]"
        layer = _object(raw_layer, context)
        _exact_fields(layer, layer_fields, context)
        network_index = _integer(
            layer["network_index"], f"{context}.network_index"
        )
        if network_index >= len(networks):
            raise RuntimeError(f"{context}.network_index is out of range")
        if network_index in layers_by_index:
            raise RuntimeError(f"duplicate layer network_index: {network_index}")
        network = networks[network_index]
        phase = _string(layer["phase"], f"{context}.phase")
        if phase not in CAMC_PHASE_TO_NETWORK_PHASE:
            raise RuntimeError(
                f"{context}.phase must be prefill, decode, verify, or draft"
            )
        if CAMC_PHASE_TO_NETWORK_PHASE[phase] != network["phase"]:
            raise RuntimeError(f"{context}.phase does not match network phase")

        default_order = [
            _string(item, f"{context}.default_order")
            for item in _array(layer["default_order"], f"{context}.default_order")
        ]
        if len(default_order) != len(set(default_order)):
            raise RuntimeError(f"{context}.default_order contains duplicates")
        if set(default_order) != set(network["operator_ids"]):
            raise RuntimeError(
                f"{context}.default_order must exactly cover network operators"
            )
        position = {op_id: index for index, op_id in enumerate(default_order)}
        for op_id in default_order:
            if any(position[dependency] >= position[op_id] for dependency in dependencies[op_id]):
                raise RuntimeError(
                    f"{context}.default_order is not topological for {op_id!r}"
                )

        node_specs: dict[str, dict[str, Any]] = {}
        for node_index, raw_node in enumerate(
            _array(layer["nodes"], f"{context}.nodes")
        ):
            node = _node_spec(
                raw_node,
                context=f"{context}.nodes[{node_index}]",
                known_devices=known_devices,
                legal_devices_by_op=prior.legal_devices,
            )
            if node["op_id"] in node_specs:
                raise RuntimeError(f"duplicate layer node op_id: {node['op_id']!r}")
            node_specs[node["op_id"]] = node
        if set(node_specs) != set(default_order):
            raise RuntimeError(f"{context}.nodes must exactly cover network operators")
        _validate_supernodes(
            node_specs=node_specs,
            legal_devices=prior.legal_devices,
            default_devices=prior.expert_placement,
            dependencies=dependencies,
            context=context,
        )

        basis = _string(layer["capability_basis"], f"{context}.capability_basis")
        if basis not in {"compute", "bandwidth"}:
            raise RuntimeError(
                f"{context}.capability_basis must be 'compute' or 'bandwidth'"
            )
        layers_by_index[network_index] = {
            "network_index": network_index,
            "layer_class": _string(
                layer["layer_class"], f"{context}.layer_class"
            ),
            "phase": phase,
            "shape_bucket": _string(
                layer["shape_bucket"], f"{context}.shape_bucket"
            ),
            "capability_basis": basis,
            "domain_capabilities": _domain_capabilities(
                layer["domain_capabilities"],
                f"{context}.domain_capabilities",
            ),
            "default_order": default_order,
            "nodes": [
                {
                    **node_specs[op_id],
                    "legal_devices": list(prior.legal_devices[op_id]),
                    "default_device": prior.expert_placement[op_id],
                }
                for op_id in default_order
            ],
        }

    if set(layers_by_index) != set(range(len(networks))):
        raise RuntimeError(
            "layer_spec.layers must contain exactly one layer for every network"
        )
    return {
        "schema": CAMC_PROFILE_SCHEMA,
        "schema_version": CAMC_PROFILE_SCHEMA_VERSION,
        "graph_id": graph_id,
        "workload_id": workload_id,
        "device_domains": device_domains,
        "layers": [
            layers_by_index[network_index]
            for network_index in range(len(networks))
        ],
    }


def export_camc_profile(
    *,
    prior_artifact: DOPSPriorArtifact | Mapping[str, Any],
    network_manifest: Mapping[str, Any],
    tensor_bindings: Mapping[str, Any],
    layer_spec: Mapping[str, Any],
    output: str | Path,
) -> Path:
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            build_camc_profile(
                prior_artifact=prior_artifact,
                network_manifest=network_manifest,
                tensor_bindings=tensor_bindings,
                layer_spec=layer_spec,
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
    "CAMC_DOMAINS",
    "CAMC_PHASE_TO_NETWORK_PHASE",
    "CAMC_PROFILE_SCHEMA",
    "CAMC_PROFILE_SCHEMA_VERSION",
    "build_camc_profile",
    "export_camc_profile",
]
