"""Strict DOPS -> Het-Infer static prior contract.

This module defines the versioned file boundary only.  It deliberately does
not integrate with the DOPS scheduler or the Het-Infer online runtime.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PRIOR_SCHEMA = "dops.hetinfer_prior.v1"
PRIOR_SCHEMA_VERSION = 1
TIME_UNIT = "seconds"

_ROOT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "graph_id",
        "workload_id",
        "time_unit",
        "devices",
        "operators",
        "inputs",
        "collective_contexts",
        "legal_movement_routes",
        "expert_placement",
        "t_service",
        "t_move",
    }
)
_DEVICE_FIELDS = frozenset({"device_id"})
_OPERATOR_FIELDS = frozenset({"op_id", "dependencies", "legal_devices"})
_INPUT_FIELDS = frozenset(
    {
        "consumer_op_id",
        "producer_op_id",
        "tensor_id",
        "semantics",
        "bytes",
        "source_residencies",
        "destination_devices",
    }
)
_INPUT_SEMANTICS = frozenset({"data", "barrier", "collective_staging"})
_SOURCE_RESIDENCY_FIELDS = frozenset({"device_id", "layout"})
_COLLECTIVE_FIELDS = frozenset(
    {
        "op_id",
        "primitive",
        "topology",
        "canonical_device_id",
        "participant_device_ids",
        "output_device_ids",
        "resource_device_ids",
        "tensor_bytes",
        "internal_transport",
    }
)
_PLACEMENT_FIELDS = frozenset({"op_id", "device_id"})
_SERVICE_FIELDS = frozenset({"op_id", "device_id", "duration_s"})
_ROUTE_FIELDS = frozenset(
    {"tensor_id", "source_device_id", "destination_device_id", "bytes", "layout"}
)
_MOVE_FIELDS = _ROUTE_FIELDS | {"duration_s"}


class DOPSPriorValidationError(ValueError):
    """The static execution artifact is incomplete, ambiguous, or unsafe."""


class DOPSPriorArtifact:
    """Validated immutable-by-copy view of the three static DOPS tables."""

    def __init__(
        self,
        *,
        payload: Mapping[str, Any],
        device_ids: tuple[str, ...],
        operator_ids: tuple[str, ...],
        legal_devices: Mapping[str, tuple[str, ...]],
        inputs: tuple[Mapping[str, Any], ...],
        collective_contexts: Mapping[str, Mapping[str, Any]],
        placement: Mapping[str, str],
        service_times: Mapping[tuple[str, str], float],
        movement_times: Mapping[tuple[str, str, str, int, str], float],
    ) -> None:
        self._payload = copy.deepcopy(dict(payload))
        self.device_ids = device_ids
        self.operator_ids = operator_ids
        self.legal_devices = dict(legal_devices)
        self.inputs = tuple(copy.deepcopy(dict(entry)) for entry in inputs)
        self.collective_contexts = {
            op_id: copy.deepcopy(dict(context))
            for op_id, context in collective_contexts.items()
        }
        self.expert_placement = dict(placement)
        self._service_times = dict(service_times)
        self._movement_times = dict(movement_times)

    @property
    def payload(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)

    def expert_device(self, op_id: str) -> str:
        try:
            return self.expert_placement[op_id]
        except KeyError as exc:
            raise KeyError(f"unknown expert placement op_id: {op_id!r}") from exc

    def inputs_for(self, consumer_op_id: str) -> tuple[dict[str, Any], ...]:
        if consumer_op_id not in self.legal_devices:
            raise KeyError(f"unknown consumer op_id: {consumer_op_id!r}")
        return tuple(
            copy.deepcopy(entry)
            for entry in self.inputs
            if entry["consumer_op_id"] == consumer_op_id
        )

    def collective_context(self, op_id: str) -> dict[str, Any]:
        try:
            return copy.deepcopy(self.collective_contexts[op_id])
        except KeyError as exc:
            raise KeyError(f"operator {op_id!r} is not a collective") from exc

    def service_time_s(self, op_id: str, device_id: str) -> float:
        try:
            return self._service_times[(op_id, device_id)]
        except KeyError as exc:
            raise KeyError(
                f"no T_service entry for op_id={op_id!r}, device_id={device_id!r}"
            ) from exc

    def movement_time_s(
        self,
        tensor_id: str,
        source_device_id: str,
        destination_device_id: str,
        bytes_: int,
        layout: str,
    ) -> float:
        key = (tensor_id, source_device_id, destination_device_id, bytes_, layout)
        try:
            return self._movement_times[key]
        except KeyError as exc:
            raise KeyError(
                "no T_move entry for "
                f"tensor_id={tensor_id!r}, source={source_device_id!r}, "
                f"destination={destination_device_id!r}, bytes={bytes_!r}, "
                f"layout={layout!r}"
            ) from exc


def _error(field: str, message: str) -> DOPSPriorValidationError:
    return DOPSPriorValidationError(f"invalid {field}: {message}")


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(field, "expected a JSON object")
    return dict(value)


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise _error(field, "expected a JSON array")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], field: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise _error(field, f"missing fields: {missing}")
    if extra:
        raise _error(field, f"unexpected fields: {extra}")


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(field, "expected a non-empty string")
    return value.strip()


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _error(field, "expected a non-negative integer")
    return value


def _duration(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(field, "expected a finite, non-negative number of seconds")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise _error(field, "expected a finite, non-negative number of seconds")
    return result


def _unique_strings(value: Any, field: str, *, nonempty: bool) -> tuple[str, ...]:
    raw = _array(value, field)
    if nonempty and not raw:
        raise _error(field, "cannot be empty")
    result = tuple(_string(item, f"{field}[{index}]") for index, item in enumerate(raw))
    if len(set(result)) != len(result):
        raise _error(field, "contains duplicates")
    return result


def _missing_keys(
    actual: set[Any], expected: set[Any], field: str, description: str
) -> None:
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise _error(
        field,
        f"must cover every {description} exactly once; "
        f"missing={missing}, unexpected={unexpected}",
    )


def _validate_acyclic(dependencies: Mapping[str, tuple[str, ...]]) -> None:
    state: dict[str, int] = {}

    def visit(op_id: str) -> None:
        marker = state.get(op_id, 0)
        if marker == 1:
            raise _error("operators", f"dependency cycle includes {op_id!r}")
        if marker == 2:
            return
        state[op_id] = 1
        for dependency in dependencies[op_id]:
            visit(dependency)
        state[op_id] = 2

    for op_id in dependencies:
        visit(op_id)


def _canonical_payload(root: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable bytes for arrays whose contract semantics are unordered."""

    payload = copy.deepcopy(dict(root))
    payload["devices"].sort(key=lambda entry: entry["device_id"])
    for operator in payload["operators"]:
        operator["dependencies"] = sorted(operator["dependencies"])
        operator["legal_devices"] = sorted(operator["legal_devices"])
    payload["operators"].sort(key=lambda entry: entry["op_id"])
    for entry in payload["inputs"]:
        entry["source_residencies"].sort(
            key=lambda residency: (residency["device_id"], residency["layout"])
        )
        entry["destination_devices"] = sorted(entry["destination_devices"])
    payload["inputs"].sort(
        key=lambda entry: (
            entry["consumer_op_id"],
            entry["producer_op_id"] or "",
            entry["tensor_id"],
            entry["semantics"],
            entry["bytes"],
            tuple(
                (residency["device_id"], residency["layout"])
                for residency in entry["source_residencies"]
            ),
            tuple(entry["destination_devices"]),
        )
    )
    for context in payload["collective_contexts"]:
        context["participant_device_ids"] = sorted(
            context["participant_device_ids"]
        )
        context["output_device_ids"] = sorted(context["output_device_ids"])
        context["resource_device_ids"] = sorted(
            context["resource_device_ids"]
        )
    payload["collective_contexts"].sort(key=lambda entry: entry["op_id"])
    payload["legal_movement_routes"].sort(
        key=lambda entry: (
            entry["tensor_id"],
            entry["source_device_id"],
            entry["destination_device_id"],
            entry["bytes"],
            entry["layout"],
        )
    )
    payload["expert_placement"].sort(key=lambda entry: entry["op_id"])
    payload["t_service"].sort(key=lambda entry: (entry["op_id"], entry["device_id"]))
    payload["t_move"].sort(
        key=lambda entry: (
            entry["tensor_id"],
            entry["source_device_id"],
            entry["destination_device_id"],
            entry["bytes"],
            entry["layout"],
        )
    )
    return payload


def validate_prior_artifact(payload: Any) -> DOPSPriorArtifact:
    """Validate all structural and cross-table invariants."""

    root = _object(payload, "<root>")
    _exact_fields(root, _ROOT_FIELDS, "<root>")
    if root["schema"] != PRIOR_SCHEMA:
        raise _error("schema", f"must equal {PRIOR_SCHEMA!r}")
    version = root["schema_version"]
    if isinstance(version, bool) or version != PRIOR_SCHEMA_VERSION:
        raise _error("schema_version", f"must equal {PRIOR_SCHEMA_VERSION}")
    _string(root["graph_id"], "graph_id")
    _string(root["workload_id"], "workload_id")
    if root["time_unit"] != TIME_UNIT:
        raise _error("time_unit", f"must equal {TIME_UNIT!r}")

    device_ids: list[str] = []
    for index, raw in enumerate(_array(root["devices"], "devices")):
        field = f"devices[{index}]"
        device = _object(raw, field)
        _exact_fields(device, _DEVICE_FIELDS, field)
        device_ids.append(_string(device["device_id"], f"{field}.device_id"))
    if len(device_ids) < 2:
        raise _error("devices", "requires at least two devices")
    if len(set(device_ids)) != len(device_ids):
        raise _error("devices", "contains duplicate device_id values")
    known_devices = set(device_ids)

    operator_ids: list[str] = []
    dependencies: dict[str, tuple[str, ...]] = {}
    legal_devices: dict[str, tuple[str, ...]] = {}
    for index, raw in enumerate(_array(root["operators"], "operators")):
        field = f"operators[{index}]"
        operator = _object(raw, field)
        _exact_fields(operator, _OPERATOR_FIELDS, field)
        op_id = _string(operator["op_id"], f"{field}.op_id")
        if op_id in legal_devices:
            raise _error("operators", f"duplicate op_id {op_id!r}")
        deps = _unique_strings(
            operator["dependencies"], f"{field}.dependencies", nonempty=False
        )
        legal = _unique_strings(
            operator["legal_devices"], f"{field}.legal_devices", nonempty=True
        )
        unknown_devices = sorted(set(legal) - known_devices)
        if unknown_devices:
            raise _error(
                f"{field}.legal_devices", f"unknown devices: {unknown_devices}"
            )
        operator_ids.append(op_id)
        dependencies[op_id] = deps
        legal_devices[op_id] = legal
    if not operator_ids:
        raise _error("operators", "cannot be empty")
    known_operators = set(operator_ids)
    for op_id, deps in dependencies.items():
        unknown = sorted(set(deps) - known_operators)
        if unknown:
            raise _error(
                f"operators[{op_id!r}].dependencies", f"unknown op_id values: {unknown}"
            )
        if op_id in deps:
            raise _error(
                f"operators[{op_id!r}].dependencies", "cannot depend on itself"
            )
    _validate_acyclic(dependencies)

    collective_contexts: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(
        _array(root["collective_contexts"], "collective_contexts")
    ):
        field = f"collective_contexts[{index}]"
        entry = _object(raw, field)
        _exact_fields(entry, _COLLECTIVE_FIELDS, field)
        op_id = _string(entry["op_id"], f"{field}.op_id")
        if op_id not in known_operators:
            raise _error(field, f"unknown op_id {op_id!r}")
        if op_id in collective_contexts:
            raise _error("collective_contexts", f"duplicate op_id {op_id!r}")
        primitive = _string(entry["primitive"], f"{field}.primitive")
        topology = _string(entry["topology"], f"{field}.topology")
        canonical = _string(
            entry["canonical_device_id"], f"{field}.canonical_device_id"
        )
        if canonical not in known_devices:
            raise _error(field, f"unknown canonical device {canonical!r}")
        if set(legal_devices[op_id]) != {canonical}:
            raise _error(
                field,
                "a fixed collective must expose exactly its canonical device "
                "as legal_devices",
            )
        participants = _unique_strings(
            entry["participant_device_ids"],
            f"{field}.participant_device_ids",
            nonempty=True,
        )
        outputs = _unique_strings(
            entry["output_device_ids"],
            f"{field}.output_device_ids",
            nonempty=True,
        )
        resources = _unique_strings(
            entry["resource_device_ids"],
            f"{field}.resource_device_ids",
            nonempty=True,
        )
        unknown_context_devices = sorted(
            (set(participants) | set(outputs) | set(resources)) - known_devices
        )
        if unknown_context_devices:
            raise _error(
                field, f"unknown collective devices: {unknown_context_devices}"
            )
        if canonical not in outputs:
            raise _error(
                f"{field}.output_device_ids",
                "must include canonical_device_id",
            )
        if not (set(participants) | set(outputs)).issubset(set(resources)):
            raise _error(
                f"{field}.resource_device_ids",
                "must include every participant and output device",
            )
        internal_transport = _string(
            entry["internal_transport"], f"{field}.internal_transport"
        )
        if internal_transport != "included_in_t_service":
            raise _error(
                f"{field}.internal_transport",
                "must equal 'included_in_t_service'",
            )
        collective_contexts[op_id] = {
            "op_id": op_id,
            "primitive": primitive,
            "topology": topology,
            "canonical_device_id": canonical,
            "participant_device_ids": list(participants),
            "output_device_ids": list(outputs),
            "resource_device_ids": list(resources),
            "tensor_bytes": _non_negative_int(
                entry["tensor_bytes"], f"{field}.tensor_bytes"
            ),
            "internal_transport": internal_transport,
        }

    inputs: list[dict[str, Any]] = []
    input_keys: set[tuple[str, str | None, str]] = set()
    dependency_inputs: set[tuple[str, str]] = set()
    tensor_bindings: dict[str, tuple[str | None, int]] = {}
    for index, raw in enumerate(_array(root["inputs"], "inputs")):
        field = f"inputs[{index}]"
        entry = _object(raw, field)
        _exact_fields(entry, _INPUT_FIELDS, field)
        consumer = _string(entry["consumer_op_id"], f"{field}.consumer_op_id")
        if consumer not in known_operators:
            raise _error(field, f"unknown consumer_op_id {consumer!r}")
        producer_raw = entry["producer_op_id"]
        producer = (
            None
            if producer_raw is None
            else _string(producer_raw, f"{field}.producer_op_id")
        )
        if producer is not None:
            if producer not in known_operators:
                raise _error(field, f"unknown producer_op_id {producer!r}")
            if producer not in dependencies[consumer]:
                raise _error(
                    field,
                    f"producer {producer!r} is not a dependency of {consumer!r}",
                )
            dependency_inputs.add((consumer, producer))
        tensor_id = _string(entry["tensor_id"], f"{field}.tensor_id")
        semantics = _string(entry["semantics"], f"{field}.semantics")
        if semantics not in _INPUT_SEMANTICS:
            raise _error(
                f"{field}.semantics",
                f"must be one of {sorted(_INPUT_SEMANTICS)}",
            )
        bytes_ = _non_negative_int(entry["bytes"], f"{field}.bytes")
        residencies: list[dict[str, str]] = []
        residency_devices: set[str] = set()
        for residency_index, raw_residency in enumerate(
            _array(entry["source_residencies"], f"{field}.source_residencies")
        ):
            residency_field = f"{field}.source_residencies[{residency_index}]"
            residency = _object(raw_residency, residency_field)
            _exact_fields(residency, _SOURCE_RESIDENCY_FIELDS, residency_field)
            device = _string(
                residency["device_id"], f"{residency_field}.device_id"
            )
            layout = _string(residency["layout"], f"{residency_field}.layout")
            if device in residency_devices:
                raise _error(
                    f"{field}.source_residencies",
                    f"duplicate device_id {device!r}",
                )
            residency_devices.add(device)
            residencies.append({"device_id": device, "layout": layout})
        unknown_sources = sorted(residency_devices - known_devices)
        if unknown_sources:
            raise _error(
                f"{field}.source_residencies", f"unknown devices: {unknown_sources}"
            )
        destinations = _unique_strings(
            entry["destination_devices"],
            f"{field}.destination_devices",
            nonempty=False,
        )
        unknown_destinations = sorted(set(destinations) - known_devices)
        if unknown_destinations:
            raise _error(
                f"{field}.destination_devices",
                f"unknown devices: {unknown_destinations}",
            )
        if producer is None and semantics != "data":
            raise _error(
                field, "an external input (producer_op_id=null) must use data semantics"
            )
        if semantics == "barrier":
            if producer is None:
                raise _error(field, "barrier semantics requires a producer_op_id")
            if bytes_ != 0 or residencies or destinations:
                raise _error(
                    field,
                    "barrier semantics requires bytes=0, source_residencies=[], "
                    "and destination_devices=[]",
                )
        elif not residencies or not destinations:
            raise _error(
                field,
                f"{semantics} semantics requires non-empty source and destination devices",
            )
        if semantics == "data":
            if set(destinations) != set(legal_devices[consumer]):
                raise _error(
                    f"{field}.destination_devices",
                    "data inputs must target exactly the consumer legal_devices",
                )
            if consumer in collective_contexts:
                raise _error(
                    field,
                    "collective dependencies must use collective_staging semantics",
                )
        elif semantics == "collective_staging":
            if producer is None:
                raise _error(
                    field, "collective_staging semantics requires a producer_op_id"
                )
            context = collective_contexts.get(consumer)
            if context is None:
                raise _error(
                    field,
                    "collective_staging consumer has no collective_context",
                )
            if len(destinations) != 1:
                raise _error(
                    f"{field}.destination_devices",
                    "collective_staging requires exactly one fixed staging device",
                )
            if destinations[0] not in context["participant_device_ids"]:
                raise _error(
                    f"{field}.destination_devices",
                    "staging device is not a collective participant",
                )

        key = (consumer, producer, tensor_id)
        if key in input_keys:
            raise _error("inputs", f"duplicate input key {key!r}")
        input_keys.add(key)
        tensor_binding = (producer, bytes_)
        previous_binding = tensor_bindings.setdefault(tensor_id, tensor_binding)
        if previous_binding != tensor_binding:
            raise _error(
                field,
                "tensor_id must bind one producer_op_id and byte count globally",
            )
        inputs.append(
            {
                "consumer_op_id": consumer,
                "producer_op_id": producer,
                "tensor_id": tensor_id,
                "semantics": semantics,
                "bytes": bytes_,
                "source_residencies": residencies,
                "destination_devices": list(destinations),
            }
        )
    if not inputs:
        raise _error("inputs", "cannot be empty")
    expected_dependencies = {
        (consumer, producer)
        for consumer, producers in dependencies.items()
        for producer in producers
    }
    _missing_keys(
        dependency_inputs,
        expected_dependencies,
        "inputs",
        "declared operator dependency",
    )
    for op_id, context in collective_contexts.items():
        staging_inputs = [
            entry
            for entry in inputs
            if entry["consumer_op_id"] == op_id
            and entry["semantics"] == "collective_staging"
        ]
        if not staging_inputs:
            raise _error(
                f"collective_contexts[{op_id!r}]",
                "must have at least one collective_staging input",
            )
        staged_devices = {
            destination
            for entry in staging_inputs
            for destination in entry["destination_devices"]
        }
        if staged_devices != set(context["participant_device_ids"]):
            raise _error(
                f"collective_contexts[{op_id!r}].participant_device_ids",
                "must exactly equal the staging input destinations; "
                f"staged={sorted(staged_devices)}",
            )

    placement: dict[str, str] = {}
    for index, raw in enumerate(_array(root["expert_placement"], "expert_placement")):
        field = f"expert_placement[{index}]"
        entry = _object(raw, field)
        _exact_fields(entry, _PLACEMENT_FIELDS, field)
        op_id = _string(entry["op_id"], f"{field}.op_id")
        device = _string(entry["device_id"], f"{field}.device_id")
        if op_id not in known_operators:
            raise _error(field, f"unknown op_id {op_id!r}")
        if op_id in placement:
            raise _error("expert_placement", f"duplicate op_id {op_id!r}")
        if device not in legal_devices[op_id]:
            raise _error(field, f"device {device!r} is not legal for {op_id!r}")
        placement[op_id] = device
    _missing_keys(set(placement), known_operators, "expert_placement", "operator")
    for op_id, context in collective_contexts.items():
        if placement[op_id] != context["canonical_device_id"]:
            raise _error(
                f"expert_placement[{op_id!r}]",
                "collective placement must equal canonical_device_id",
            )

    service_times: dict[tuple[str, str], float] = {}
    for index, raw in enumerate(_array(root["t_service"], "t_service")):
        field = f"t_service[{index}]"
        entry = _object(raw, field)
        _exact_fields(entry, _SERVICE_FIELDS, field)
        op_id = _string(entry["op_id"], f"{field}.op_id")
        device = _string(entry["device_id"], f"{field}.device_id")
        if op_id not in known_operators:
            raise _error(field, f"unknown op_id {op_id!r}")
        if device not in legal_devices[op_id]:
            raise _error(field, f"device {device!r} is not legal for {op_id!r}")
        key = (op_id, device)
        if key in service_times:
            raise _error("t_service", f"duplicate key {key!r}")
        service_times[key] = _duration(entry["duration_s"], f"{field}.duration_s")
    expected_service = {
        (op_id, device)
        for op_id, devices in legal_devices.items()
        for device in devices
    }
    _missing_keys(
        set(service_times), expected_service, "t_service", "legal operator-device"
    )

    legal_routes: set[tuple[str, str, str, int, str]] = set()
    for index, raw in enumerate(
        _array(root["legal_movement_routes"], "legal_movement_routes")
    ):
        field = f"legal_movement_routes[{index}]"
        route = _object(raw, field)
        _exact_fields(route, _ROUTE_FIELDS, field)
        tensor_id = _string(route["tensor_id"], f"{field}.tensor_id")
        source = _string(route["source_device_id"], f"{field}.source_device_id")
        destination = _string(
            route["destination_device_id"], f"{field}.destination_device_id"
        )
        bytes_ = _non_negative_int(route["bytes"], f"{field}.bytes")
        layout = _string(route["layout"], f"{field}.layout")
        unknown = sorted({source, destination} - known_devices)
        if unknown:
            raise _error(field, f"unknown devices: {unknown}")
        key = (tensor_id, source, destination, bytes_, layout)
        if key in legal_routes:
            raise _error("legal_movement_routes", f"duplicate key {key!r}")
        legal_routes.add(key)
    if not legal_routes:
        raise _error("legal_movement_routes", "cannot be empty")

    for index, entry in enumerate(inputs):
        if entry["semantics"] == "barrier":
            barrier_routes = sorted(
                route for route in legal_routes if route[0] == entry["tensor_id"]
            )
            if barrier_routes:
                raise _error(
                    f"inputs[{index}]",
                    "barrier input must not have legal movement routes: "
                    f"{barrier_routes}",
                )
            continue
        expected_routes = {
            (
                entry["tensor_id"],
                residency["device_id"],
                destination,
                entry["bytes"],
                residency["layout"],
            )
            for residency in entry["source_residencies"]
            for destination in entry["destination_devices"]
        }
        missing_routes = sorted(expected_routes - legal_routes)
        if missing_routes:
            raise _error(
                f"inputs[{index}]",
                f"{entry['semantics']} input is missing legal movement routes: "
                f"{missing_routes}",
            )

    movement_times: dict[tuple[str, str, str, int, str], float] = {}
    for index, raw in enumerate(_array(root["t_move"], "t_move")):
        field = f"t_move[{index}]"
        entry = _object(raw, field)
        _exact_fields(entry, _MOVE_FIELDS, field)
        key = (
            _string(entry["tensor_id"], f"{field}.tensor_id"),
            _string(entry["source_device_id"], f"{field}.source_device_id"),
            _string(entry["destination_device_id"], f"{field}.destination_device_id"),
            _non_negative_int(entry["bytes"], f"{field}.bytes"),
            _string(entry["layout"], f"{field}.layout"),
        )
        if key in movement_times:
            raise _error("t_move", f"duplicate key {key!r}")
        duration_s = _duration(entry["duration_s"], f"{field}.duration_s")
        if key[1] == key[2] and duration_s != 0.0:
            raise _error(
                f"{field}.duration_s", "resident source/destination must be zero"
            )
        movement_times[key] = duration_s
    _missing_keys(set(movement_times), legal_routes, "t_move", "legal movement route")

    return DOPSPriorArtifact(
        payload=_canonical_payload(root),
        device_ids=tuple(device_ids),
        operator_ids=tuple(operator_ids),
        legal_devices=legal_devices,
        inputs=tuple(inputs),
        collective_contexts=collective_contexts,
        placement=placement,
        service_times=service_times,
        movement_times=movement_times,
    )


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DOPSPriorValidationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    raise DOPSPriorValidationError(
        f"non-finite JSON number is forbidden: {value}"
    )


def load_prior_artifact(path: str | Path) -> DOPSPriorArtifact:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"prior artifact does not exist: {source}")
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_object_keys,
                parse_constant=_reject_nonstandard_number,
            )
    except json.JSONDecodeError as exc:
        raise DOPSPriorValidationError(
            f"invalid JSON in {source}: {exc}"
        ) from exc
    return validate_prior_artifact(payload)


def write_prior_artifact(
    artifact: DOPSPriorArtifact | Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically write the canonical offline handoff file."""

    validated = (
        artifact
        if isinstance(artifact, DOPSPriorArtifact)
        else validate_prior_artifact(artifact)
    )
    destination = Path(path).expanduser()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite prior artifact: {destination}"
        )
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"prior artifact output directory does not exist: {destination.parent}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                validated.payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = [
    "PRIOR_SCHEMA",
    "PRIOR_SCHEMA_VERSION",
    "TIME_UNIT",
    "DOPSPriorArtifact",
    "DOPSPriorValidationError",
    "load_prior_artifact",
    "validate_prior_artifact",
    "write_prior_artifact",
]
