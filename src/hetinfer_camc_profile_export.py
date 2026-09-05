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
SD_COMPONENTS = (
    "none",
    "target_decode",
    "draft_decode",
    "target_verify",
    "candidate_transfer",
)



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
    allow_prior_operator_subset: bool,
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
        context = f"network[{network_index}]"
        phase = _string(network.get("phase"), f"{context}.phase")
        if phase not in {"prefill", "decode"}:
            raise RuntimeError(f"{context}.phase must be 'prefill' or 'decode'")
        workload = _object(network.get("workload"), f"{context}.workload")
        _exact_fields(
            workload,
            {
                "batch",
                "sequence_length",
                "past_kv_len",
                "query_len",
                "scheduled_tokens",
                "mean_context",
            },
            f"{context}.workload",
        )
        batch_size = _integer(
            workload.get("batch"), f"{context}.workload.batch", minimum=1
        )
        sequence_length = _integer(
            workload.get("sequence_length"),
            f"{context}.workload.sequence_length",
            minimum=1,
        )
        past_kv_len = _integer(
            workload.get("past_kv_len"),
            f"{context}.workload.past_kv_len",
        )
        query_len = _integer(
            workload.get("query_len"),
            f"{context}.workload.query_len",
            minimum=1,
        )
        scheduled_tokens = _integer(
            workload.get("scheduled_tokens"),
            f"{context}.workload.scheduled_tokens",
            minimum=1,
        )
        _positive_number(
            workload.get("mean_context"),
            f"{context}.workload.mean_context",
        )
        if scheduled_tokens != batch_size * query_len:
            raise RuntimeError(
                f"{context}.workload.scheduled_tokens must equal batch * query_len"
            )
        operator_ids: list[str] = []
        operator_metadata: dict[str, dict[str, Any]] = {}
        for operator_index, raw_operator in enumerate(
            _array(network.get("operators"), f"{context}.operators")
        ):
            operator_context = f"{context}.operators[{operator_index}]"
            operator = _object(raw_operator, operator_context)
            op_id = _string(operator.get("op_id"), f"{operator_context}.op_id")
            if op_id in op_network:
                raise RuntimeError(f"duplicate network operator op_id: {op_id!r}")
            if op_id not in dependencies:
                raise RuntimeError(f"network has unknown prior op_id: {op_id!r}")
            declared_position = _integer(
                operator.get("operator_index"),
                f"{operator_context}.operator_index",
            )
            if declared_position != operator_index:
                raise RuntimeError(
                    f"{operator_context}.operator_index must equal its array position"
                )
            op_role = _string(
                operator.get("op_role"),
                f"{operator_context}.op_role",
            )
            layer_index = operator.get("layer_index")
            if layer_index is not None:
                layer_index = _integer(
                    layer_index,
                    f"{operator_context}.layer_index",
                )
            operator_dependencies = [
                _string(item, f"{operator_context}.dependencies")
                for item in _array(
                    operator.get("dependencies"),
                    f"{operator_context}.dependencies",
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
            operator_metadata[op_id] = {
                "layer_index": layer_index,
                "operator_index": operator_index,
                "op_role": op_role,
            }
        if not operator_ids:
            raise RuntimeError(f"{context} has no operators")
        networks.append(
            {
                "phase": phase,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "past_kv_len": past_kv_len,
                "query_len": query_len,
                "operator_ids": operator_ids,
                "operator_metadata": operator_metadata,
            }
        )

    selected_operator_ids = set(op_network)
    if not allow_prior_operator_subset and selected_operator_ids != set(
        prior.operator_ids
    ):
        raise RuntimeError("network operators must exactly cover prior operators")
    for op_id in selected_operator_ids:
        missing_dependencies = set(dependencies[op_id]) - selected_operator_ids
        if missing_dependencies:
            raise RuntimeError(
                f"selected network operator {op_id!r} has dependencies outside "
                f"the selected prior subset: {sorted(missing_dependencies)}"
            )
        if any(
            op_network[dependency] != op_network[op_id]
            for dependency in dependencies[op_id]
        ):
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


EXPERT_TIMING_SOURCES = (
    "exact_lut",
    "aim_simulator",
    "interpolated_lut",
)


def build_expert_service_lut(
    *,
    max_tokens: int,
    activation_bytes_per_token: int,
    npu_anchors: Mapping[str, Mapping[int, int | float]],
    pim_measurements: Mapping[str, Mapping[int, int | float]],
) -> list[dict[str, Any]]:
    """Build one dense integer n_e LUT without extrapolation."""

    maximum = _integer(max_tokens, "max_tokens", minimum=1)
    bytes_per_token = _integer(
        activation_bytes_per_token,
        "activation_bytes_per_token",
        minimum=1,
    )
    if not npu_anchors or not pim_measurements:
        raise RuntimeError("expert LUT requires NPU anchors and PIM measurements")
    overlap = set(npu_anchors) & set(pim_measurements)
    if overlap:
        raise RuntimeError(f"expert LUT device domains overlap: {sorted(overlap)}")
    anchors_by_device: dict[str, dict[int, float]] = {}
    for device_id, raw_anchors in npu_anchors.items():
        device = _string(device_id, "NPU device_id")
        anchors = {
            _integer(token_count, f"{device} anchor", minimum=1): float(
                _positive_number(duration, f"{device} anchor duration")
            )
            for token_count, duration in raw_anchors.items()
        }
        if set(anchors) - set(range(1, maximum + 1)):
            raise RuntimeError(f"{device} anchor token count exceeds LUT range")
        if 1 not in anchors or maximum not in anchors:
            raise RuntimeError(
                f"{device} NPU anchors must include 1 and max_tokens"
            )
        anchors_by_device[device] = anchors

    measurements_by_device: dict[str, dict[int, float]] = {}
    expected_counts = set(range(1, maximum + 1))
    for device_id, raw_measurements in pim_measurements.items():
        device = _string(device_id, "PIM device_id")
        measurements = {
            _integer(token_count, f"{device} measurement", minimum=1): float(
                _positive_number(duration, f"{device} measurement duration")
            )
            for token_count, duration in raw_measurements.items()
        }
        if set(measurements) != expected_counts:
            raise RuntimeError(
                f"{device} PIM measurements must cover every integer n_e"
            )
        measurements_by_device[device] = measurements

    def npu_value(
        anchors: Mapping[int, float],
        token_count: int,
    ) -> tuple[float, str]:
        if token_count in anchors:
            return anchors[token_count], "exact_lut"
        lower = max(point for point in anchors if point < token_count)
        upper = min(point for point in anchors if point > token_count)
        ratio = (token_count - lower) / (upper - lower)
        return (
            anchors[lower] + (anchors[upper] - anchors[lower]) * ratio,
            "interpolated_lut",
        )

    result: list[dict[str, Any]] = []
    for token_count in range(1, maximum + 1):
        service_time_s: dict[str, float] = {}
        timing_source: dict[str, str] = {}
        for device_id, anchors in anchors_by_device.items():
            duration, source = npu_value(anchors, token_count)
            service_time_s[device_id] = duration
            timing_source[device_id] = source
        for device_id, measurements in measurements_by_device.items():
            service_time_s[device_id] = measurements[token_count]
            timing_source[device_id] = "aim_simulator"
        result.append(
            {
                "min_tokens": token_count,
                "max_tokens": token_count,
                "activation_bytes": token_count * bytes_per_token,
                "service_time_s": service_time_s,
                "timing_source": timing_source,
            }
        )
    return result


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
            {
                "min_tokens",
                "max_tokens",
                "activation_bytes",
                "service_time_s",
                "timing_source",
            },
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
        if min_tokens != max_tokens:
            raise RuntimeError(
                f"{bucket_context} must describe one exact integer n_e"
            )
        raw_service = _object(
            bucket["service_time_s"], f"{bucket_context}.service_time_s"
        )
        raw_sources = _object(
            bucket["timing_source"], f"{bucket_context}.timing_source"
        )
        if set(raw_service) != set(legal_devices):
            raise RuntimeError(
                f"{bucket_context}.service_time_s must exactly cover legal devices"
            )
        if set(raw_sources) != set(legal_devices):
            raise RuntimeError(
                f"{bucket_context}.timing_source must exactly cover legal devices"
            )
        timing_source = {
            device_id: _string(
                raw_sources[device_id],
                f"{bucket_context}.timing_source[{device_id!r}]",
            )
            for device_id in legal_devices
        }
        invalid_sources = set(timing_source.values()) - set(EXPERT_TIMING_SOURCES)
        if invalid_sources:
            raise RuntimeError(
                f"{bucket_context}.timing_source has unsupported values "
                f"{sorted(invalid_sources)}"
            )
        result.append(
            {
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
                "activation_bytes": _integer(
                    bucket["activation_bytes"],
                    f"{bucket_context}.activation_bytes",
                    minimum=1,
                ),
                "service_time_s": {
                    device_id: _positive_number(
                        raw_service[device_id],
                        f"{bucket_context}.service_time_s[{device_id!r}]",
                    )
                    for device_id in legal_devices
                },
                "timing_source": timing_source,
            }
        )
    result.sort(key=lambda item: item["min_tokens"])
    counts = [item["min_tokens"] for item in result]
    if counts and counts != list(range(1, counts[-1] + 1)):
        raise RuntimeError(f"{context} must be a dense integer n_e LUT from 1")
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
            "layer_index",
            "operator_index",
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
    raw_layer_index = node["layer_index"]
    layer_index = (
        None
        if raw_layer_index is None
        else _integer(raw_layer_index, f"{context}.layer_index")
    )

    result = {
        "op_id": op_id,
        "layer_index": layer_index,
        "operator_index": _integer(
            node["operator_index"], f"{context}.operator_index"
        ),
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
        allowed = known_devices - ({"CPU0"} if field == "kv_home" else set())
        if result[field] is not None and result[field] not in allowed:
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


def _validate_moe_movement_lut(
    *,
    prior: DOPSPriorArtifact,
    node_specs: Mapping[str, Mapping[str, Any]],
    roles: Mapping[str, str],
    context: str,
) -> None:
    expert_ids = {
        node_specs[op_id]["expert_id"]
        for op_id, role in roles.items()
        if role == "EXPERT"
    }
    dispatch_experts: set[str] = set()
    combine_experts: set[str] = set()
    boundary_inputs: list[tuple[Mapping[str, Any], str]] = []
    for raw in prior.inputs:
        consumer_id = raw["consumer_op_id"]
        producer_id = raw["producer_op_id"]
        if consumer_id not in roles:
            continue
        expert_node_id = None
        if (
            roles[consumer_id] == "EXPERT"
            and producer_id is not None
            and roles.get(producer_id) == "ROUTER"
        ):
            expert_node_id = consumer_id
            dispatch_experts.add(node_specs[consumer_id]["expert_id"])
        elif (
            roles[consumer_id] == "COMBINE"
            and producer_id is not None
            and roles.get(producer_id) == "EXPERT"
        ):
            expert_node_id = producer_id
            combine_experts.add(node_specs[producer_id]["expert_id"])
        if expert_node_id is not None:
            boundary_inputs.append((raw, expert_node_id))

    if dispatch_experts != expert_ids or combine_experts != expert_ids:
        raise RuntimeError(
            f"{context} must expose Router-to-Expert Dispatch and "
            "Expert-to-Combine movement for every expert"
        )

    for raw, expert_node_id in boundary_inputs:
        for bucket in node_specs[expert_node_id]["expert_service_buckets"]:
            activation_bytes = bucket["activation_bytes"]
            for residency in raw["source_residencies"]:
                for destination in raw["destination_devices"]:
                    key = (
                        raw["tensor_id"],
                        residency["device_id"],
                        destination,
                        activation_bytes,
                        residency["layout"],
                    )
                    try:
                        prior.movement_time_s(*key)
                    except KeyError as exc:
                        raise RuntimeError(
                            f"{context} lacks exact MoE movement timing for {key!r}"
                        ) from exc


def build_camc_profile(
    *,
    prior_artifact: DOPSPriorArtifact | Mapping[str, Any],
    network_manifest: Mapping[str, Any],
    tensor_bindings: Mapping[str, Any],
    layer_spec: Mapping[str, Any],
    allow_prior_operator_subset: bool = False,
) -> dict[str, Any]:
    """Merge explicit CAMC deployment annotations with validated DOPS sidecars."""

    prior, graph_id, workload_id, networks, dependencies = _sidecar_index(
        prior_artifact=prior_artifact,
        network_manifest=network_manifest,
        tensor_bindings=tensor_bindings,
        allow_prior_operator_subset=allow_prior_operator_subset,
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

    selected_operator_ids = {
        op_id for network in networks for op_id in network["operator_ids"]
    }
    schedulable_devices = {
        device_id
        for op_id in selected_operator_ids
        for device_id in prior.legal_devices[op_id]
    }
    raw_domains = _object(spec["device_domains"], "layer_spec.device_domains")
    if set(raw_domains) != schedulable_devices:
        raise RuntimeError(
            "layer_spec.device_domains must exactly cover schedulable legal devices"
        )
    device_domains: dict[str, str] = {}
    for device_id in sorted(schedulable_devices):
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
        "batch_size",
        "sequence_length",
        "past_kv_len",
        "query_len",
        "router_top_k",
        "sd_component",
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

        batch_size = _integer(
            layer["batch_size"], f"{context}.batch_size", minimum=1
        )
        sequence_length = _integer(
            layer["sequence_length"],
            f"{context}.sequence_length",
            minimum=1,
        )
        past_kv_len = _integer(
            layer["past_kv_len"], f"{context}.past_kv_len"
        )
        query_len = _integer(
            layer["query_len"], f"{context}.query_len", minimum=1
        )
        for field, value in (
            ("batch_size", batch_size),
            ("sequence_length", sequence_length),
            ("past_kv_len", past_kv_len),
            ("query_len", query_len),
        ):
            if value != network[field]:
                raise RuntimeError(
                    f"{context}.{field} does not match network workload"
                )

        raw_top_k = layer["router_top_k"]
        router_top_k = (
            None
            if raw_top_k is None
            else _integer(raw_top_k, f"{context}.router_top_k", minimum=1)
        )
        layer_class = _string(
            layer["layer_class"], f"{context}.layer_class"
        )
        if layer_class == "moe" and router_top_k is None:
            raise RuntimeError(f"{context}.router_top_k is required for MoE")
        if layer_class != "moe" and router_top_k is not None:
            raise RuntimeError(f"{context}.router_top_k is only valid for MoE")

        sd_component = _string(
            layer["sd_component"], f"{context}.sd_component"
        )
        if sd_component not in SD_COMPONENTS:
            raise RuntimeError(
                f"{context}.sd_component must be one of {SD_COMPONENTS}"
            )
        component_contract = {
            "target_decode": ("decode", 1),
            "draft_decode": ("draft", 1),
        }
        if sd_component in component_contract:
            expected_phase, expected_query = component_contract[sd_component]
            if phase != expected_phase or query_len != expected_query:
                raise RuntimeError(
                    f"{context} has invalid {sd_component} phase/query_len"
                )
        if sd_component == "target_verify" and (
            phase != "verify" or query_len < 2
        ):
            raise RuntimeError(
                f"{context} target_verify requires verify phase and query_len >= 2"
            )
        if sd_component == "candidate_transfer" and phase != "decode":
            raise RuntimeError(
                f"{context} candidate_transfer requires decode phase"
            )
        required_component_role = {
            "target_decode": "KV_WRITE",
            "draft_decode": "KV_WRITE",
            "target_verify": "KV_WRITE",
            "candidate_transfer": "CANDIDATE_TRANSFER",
        }.get(sd_component)
        if required_component_role is not None:
            operator_roles = {
                metadata["op_role"]
                for metadata in network["operator_metadata"].values()
            }
            if required_component_role not in operator_roles:
                raise RuntimeError(
                    f"{context} {sd_component} requires op_role "
                    f"{required_component_role}"
                )

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
            if any(
                position[dependency] >= position[op_id]
                for dependency in dependencies[op_id]
            ):
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
            metadata = network["operator_metadata"][node["op_id"]]
            if node["layer_index"] != metadata["layer_index"]:
                raise RuntimeError(
                    f"{context} node {node['op_id']!r} layer_index mismatch"
                )
            if node["operator_index"] != metadata["operator_index"]:
                raise RuntimeError(
                    f"{context} node {node['op_id']!r} operator_index mismatch"
                )
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
        if layer_class == "moe":
            roles = {
                op_id: network["operator_metadata"][op_id]["op_role"]
                for op_id in default_order
            }
            layer_indices = sorted({node_specs[op_id]["layer_index"] for op_id in roles})
            for model_layer_index in layer_indices:
                layer_roles = {op_id: role for op_id, role in roles.items()
                               if node_specs[op_id]["layer_index"] == model_layer_index}
                routers = [
                    op_id for op_id, role in layer_roles.items() if role == "ROUTER"
                ]
                combines = [
                    op_id for op_id, role in layer_roles.items() if role == "COMBINE"
                ]
                expert_nodes = [
                    op_id for op_id, role in layer_roles.items() if role == "EXPERT"
                ]
                if len(routers) != 1 or len(combines) != 1 or not expert_nodes:
                    raise RuntimeError(
                        f"{context} MoE requires one Router, Expert nodes, and one Combine"
                    )
                expert_ids = {
                    node_specs[op_id]["expert_id"] for op_id in expert_nodes
                }
                if None in expert_ids:
                    raise RuntimeError(
                        f"{context} every EXPERT node requires expert_id and exact LUT"
                    )
                if router_top_k > len(expert_ids):
                    raise RuntimeError(
                        f"{context}.router_top_k exceeds exported experts"
                    )
                for op_id, node in node_specs.items():
                    if op_id not in layer_roles:
                        continue
                    if layer_roles[op_id] != "EXPERT" and node["expert_id"] is not None:
                        raise RuntimeError(
                            f"{context} non-EXPERT node {op_id!r} has expert_id"
                        )
                router_position = position[routers[0]]
                combine_position = position[combines[0]]
                if any(
                    not router_position < position[op_id] < combine_position
                    for op_id in expert_nodes
                ):
                    raise RuntimeError(
                        f"{context} requires Router before Experts before Combine"
                    )
                _validate_moe_movement_lut(
                    prior=prior,
                    node_specs=node_specs,
                    roles=layer_roles,
                    context=context,
                )

        basis = _string(layer["capability_basis"], f"{context}.capability_basis")
        if basis not in {"compute", "bandwidth"}:
            raise RuntimeError(
                f"{context}.capability_basis must be 'compute' or 'bandwidth'"
            )
        layers_by_index[network_index] = {
            "network_index": network_index,
            "layer_class": layer_class,
            "phase": phase,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "past_kv_len": past_kv_len,
            "query_len": query_len,
            "router_top_k": router_top_k,
            "sd_component": sd_component,
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
    allow_prior_operator_subset: bool = False,
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
                allow_prior_operator_subset=allow_prior_operator_subset,
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


def export_camc_bundle(
    *,
    prior_artifact: DOPSPriorArtifact | Mapping[str, Any],
    network_manifest: Mapping[str, Any],
    tensor_bindings: Mapping[str, Any],
    layer_spec: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    prior = (
        prior_artifact
        if isinstance(prior_artifact, DOPSPriorArtifact)
        else validate_prior_artifact(prior_artifact)
    )
    profile = build_camc_profile(
        prior_artifact=prior,
        network_manifest=network_manifest,
        tensor_bindings=tensor_bindings,
        layer_spec=layer_spec,
    )
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "prior": prior.payload,
        "network": dict(network_manifest),
        "tensor_bindings": dict(tensor_bindings),
        "layer_spec": dict(layer_spec),
        "camc_profile": profile,
    }
    outputs = {
        name: root / f"{name}.json"
        for name in payloads
    }
    for name, payload in payloads.items():
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
    "CAMC_DOMAINS",
    "CAMC_PHASE_TO_NETWORK_PHASE",
    "CAMC_PROFILE_SCHEMA",
    "CAMC_PROFILE_SCHEMA_VERSION",
    "EXPERT_TIMING_SOURCES",
    "SD_COMPONENTS",
    "build_camc_profile",
    "build_expert_service_lut",
    "export_camc_bundle",
    "export_camc_profile",
]
