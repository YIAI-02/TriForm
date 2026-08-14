"""Build the static Het-Infer prior from completed DOPS schedule snapshots.

The scheduler owns placement and local cost-model evaluation.  This module is
an offline, fail-closed projection: it never invokes a scheduler, simulator,
Value model, trainer, or online Het-Infer component.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hetinfer_prior import validate_prior_artifact, write_prior_artifact


ATLAS_TIMING_SCHEMA = "dops.hetinfer_atlas_timings.v1"
ATLAS_TIMING_SCHEMA_VERSION = 1
ATLAS_TIMING_REQUEST_SCHEMA = "dops.hetinfer_atlas_timing_request.v1"
ATLAS_TIMING_REQUEST_SCHEMA_VERSION = 1

_SNAPSHOT_FIELDS = frozenset(
    {
        "schedule_call_index",
        "phase",
        "devices",
        "operators",
        "inputs",
        "collective_contexts",
        "routes",
    }
)
_SNAPSHOT_DEVICE_FIELDS = frozenset({"device_id", "device_type"})
_SNAPSHOT_OPERATOR_FIELDS = frozenset(
    {
        "op_id",
        "dependencies",
        "legal_devices",
        "expert_device",
        "service_s",
        "atlas_descriptor",
    }
)
_SNAPSHOT_ROUTE_FIELDS = frozenset(
    {
        "tensor_id",
        "source_device_id",
        "destination_device_id",
        "bytes",
        "layout",
        "duration_s",
        "requires_atlas",
        "atlas_descriptor",
    }
)
_SNAPSHOT_INPUT_FIELDS = frozenset(
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
_SNAPSHOT_SOURCE_RESIDENCY_FIELDS = frozenset({"device_id", "layout"})
_SNAPSHOT_COLLECTIVE_CONTEXT_FIELDS = frozenset(
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
_SNAPSHOT_OPERATOR_DESCRIPTOR_FIELDS = frozenset(
    {
        "op_kind",
        "phase",
        "batch",
        "seq_len",
        "attrs",
        "weight_layout_by_device",
        "collective_primitive",
        "collective_participants",
        "topology",
    }
)
_SNAPSHOT_ROUTE_DESCRIPTOR_FIELDS = frozenset(
    {"topology", "source_device_type", "destination_device_type"}
)
_ATLAS_ROOT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "graph_id",
        "workload_id",
        "timing_context_sha256",
        "service",
        "movement",
    }
)
_ATLAS_SERVICE_FIELDS = frozenset(
    {"op_id", "device_id", "cycles", "frequency_MHz"}
)
_ATLAS_MOVEMENT_FIELDS = frozenset(
    {
        "tensor_id",
        "source_device_id",
        "destination_device_id",
        "bytes",
        "layout",
        "cycles",
        "frequency_MHz",
    }
)
_ATLAS_REQUEST_ROOT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "graph_id",
        "workload_id",
        "timing_context_sha256",
        "timing_context",
        "service",
        "movement",
    }
)
_ATLAS_REQUEST_SERVICE_FIELDS = frozenset(
    {"op_id", "device_id", "descriptor"}
)
_ATLAS_REQUEST_SERVICE_DESCRIPTOR_FIELDS = frozenset(
    {
        "op_kind",
        "phase",
        "batch",
        "seq_len",
        "attrs",
        "weight_layout",
        "collective_primitive",
        "collective_participants",
        "topology",
        "target_device_type",
    }
)
_ATLAS_REQUEST_MOVEMENT_FIELDS = frozenset(
    {
        "tensor_id",
        "source_device_id",
        "destination_device_id",
        "bytes",
        "layout",
        "descriptor",
    }
)
_ATLAS_REQUEST_MOVEMENT_DESCRIPTOR_FIELDS = frozenset(
    {"topology", "source_device_type", "destination_device_type"}
)
_ATLAS_TIMING_CONTEXT_FIELDS = frozenset(
    {"snapshot_sha256", "timing_cfg", "input_files"}
)
_ATLAS_TIMING_INPUT_FILE_FIELDS = frozenset(
    {"config_key", "path", "size_bytes", "sha256"}
)
_ROUTE_KEY_FIELDS = (
    "tensor_id",
    "source_device_id",
    "destination_device_id",
    "bytes",
    "layout",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

# These values control destinations, diagnostics, scheduler selection, or
# human-facing labels.  They must not perturb the timing identity and, in
# particular, graph/workload labels cannot be used as an escape hatch to choose
# the timing digest.  The descriptor snapshot binds the resulting placement,
# legal sets, phases, shapes, layouts, and collective structure directly.
_NON_TIMING_CFG_KEYS = frozenset(
    {
        "_config_path",
        "algo",
        "all_passes_json",
        "baseline_out",
        "best_summary_json",
        "combined_out",
        "debug",
        "decode_plan_refresh_stride",
        "decode_sample_stride",
        "dump_graph_dir",
        "hetinfer_atlas_manifest_out",
        "hetinfer_atlas_timings",
        "hetinfer_graph_id",
        "hetinfer_prior_out",
        "hetinfer_workload_id",
        "result_dir",
        "scheduler_seed",
        "serve_out",
        "simulation_log_file",
        "weight_format_compare_json",
        "weight_format_json",
        "weight_suggest_al_log_path",
    }
)

# Hash the bytes, not merely the path, for every external input that can alter
# graph shapes, hardware/link costs, an NPU LUT, or ATLAS/Ramulator behavior.
_TIMING_INPUT_FILE_CFG_KEYS = (
    "burstgpt_csv",
    "gpu_runtime_model_json",
    "hardware_json",
    "pim_config_path",
    "ramulator_config_path",
    "request_trace_path",
    "shape_file",
    "workload_path",
)


class PriorExportError(ValueError):
    """A completed schedule cannot be projected into a complete static prior."""


def _error(field: str, message: str) -> PriorExportError:
    return PriorExportError(f"invalid {field}: {message}")


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(field, "expected an object")
    return dict(value)


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise _error(field, "expected an array")
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


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _sha256(value: Any, field: str) -> str:
    result = _string(value, field)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise _error(field, "expected a lowercase 64-character SHA-256 digest")
    return result


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _error(field, "expected a non-negative integer")
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _non_negative_int(value, field)
    if result == 0:
        raise _error(field, "expected a positive integer")
    return result


def _duration(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(field, "expected a finite, non-negative duration in seconds")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise _error(field, "expected a finite, non-negative duration in seconds")
    return result


def _unique_strings(value: Any, field: str, *, nonempty: bool) -> tuple[str, ...]:
    raw = _array(value, field)
    if nonempty and not raw:
        raise _error(field, "cannot be empty")
    result = tuple(_string(item, f"{field}[{index}]") for index, item in enumerate(raw))
    if len(set(result)) != len(result):
        raise _error(field, "contains duplicates")
    return result


def _canonical_json_value(value: Any, field: str) -> Any:
    """Return a JSON-only deep copy suitable for deterministic hashing."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error(field, "non-finite numbers are forbidden")
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise _error(field, "object keys must be non-empty strings")
            result[raw_key] = _canonical_json_value(
                raw_item, f"{field}.{raw_key}"
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_value(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise _error(
        field,
        f"expected JSON data, got {type(value).__name__}",
    )


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_timing_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, Mapping):
        raise _error("cfg", "expected an object")
    selected = {
        key: value
        for key, value in cfg.items()
        if key not in _NON_TIMING_CFG_KEYS and not str(key).startswith("_")
    }
    return _canonical_json_value(selected, "cfg")


def _timing_input_files(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in _TIMING_INPUT_FILE_CFG_KEYS:
        raw_path = cfg.get(key)
        if raw_path in (None, ""):
            continue
        if not isinstance(raw_path, (str, os.PathLike)):
            raise _error(f"cfg.{key}", "expected a filesystem path")
        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise _error(f"cfg.{key}", f"cannot resolve input file {path}: {exc}") from exc
        if not resolved.is_file():
            raise _error(f"cfg.{key}", f"input is not a regular file: {resolved}")
        digest = hashlib.sha256()
        size = 0
        try:
            with resolved.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            raise _error(f"cfg.{key}", f"cannot hash input file {resolved}: {exc}") from exc
        result.append(
            {
                "config_key": key,
                "path": str(resolved),
                "size_bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
    return result


def _timing_cfg_for_digest(cfg: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalized_timing_cfg(cfg)
    # File identity is represented by canonical path + size + byte digest in
    # input_files.  Avoid a second, spelling-sensitive path identity here.
    for key in _TIMING_INPUT_FILE_CFG_KEYS:
        normalized.pop(key, None)
    return normalized


def _timing_context(
    *,
    cfg: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_snapshots = _canonical_json_value(list(snapshots), "snapshots")
    return {
        "snapshot_sha256": _canonical_json_sha256(normalized_snapshots),
        "timing_cfg": _timing_cfg_for_digest(cfg),
        "input_files": _timing_input_files(cfg),
    }


def _timing_context_sha256(
    *,
    timing_context: Mapping[str, Any],
    service: Sequence[Mapping[str, Any]],
    movement: Sequence[Mapping[str, Any]],
) -> str:
    # Bind the manifest descriptors/keys as well as the snapshot/config/file
    # context.  A hand-edited request therefore cannot retain a valid digest.
    return _canonical_json_sha256(
        {
            "timing_context": dict(timing_context),
            "service": list(service),
            "movement": list(movement),
        }
    )


def atlas_duration_s(cycles: Any, frequency_mhz: Any) -> float:
    """Convert a precomputed ATLAS cycle count to seconds."""

    cycle_count = _non_negative_int(cycles, "cycles")
    if isinstance(frequency_mhz, bool) or not isinstance(
        frequency_mhz, (int, float)
    ):
        raise _error("frequency_MHz", "expected a finite positive number")
    frequency = float(frequency_mhz)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise _error("frequency_MHz", "expected a finite positive number")
    return float(cycle_count) / (frequency * 1_000_000.0)


@dataclass(frozen=True)
class AtlasTimingTable:
    graph_id: str
    workload_id: str
    timing_context_sha256: str
    service_s: Mapping[tuple[str, str], float]
    movement_s: Mapping[tuple[str, str, str, int, str], float]

    def service_time_s(self, op_id: str, device_id: str) -> float:
        key = (op_id, device_id)
        try:
            return float(self.service_s[key])
        except KeyError as exc:
            raise PriorExportError(f"missing precomputed ATLAS service key: {key!r}") from exc

    def movement_time_s(
        self,
        tensor_id: str,
        source_device_id: str,
        destination_device_id: str,
        bytes_: int,
        layout: str,
    ) -> float:
        key = (
            tensor_id,
            source_device_id,
            destination_device_id,
            bytes_,
            layout,
        )
        try:
            return float(self.movement_s[key])
        except KeyError as exc:
            raise PriorExportError(f"missing precomputed ATLAS movement key: {key!r}") from exc


def validate_atlas_timings(payload: Any) -> AtlasTimingTable:
    root = _object(payload, "atlas.<root>")
    _exact_fields(root, _ATLAS_ROOT_FIELDS, "atlas.<root>")
    if root["schema"] != ATLAS_TIMING_SCHEMA:
        raise _error("atlas.schema", f"must equal {ATLAS_TIMING_SCHEMA!r}")
    version = root["schema_version"]
    if isinstance(version, bool) or version != ATLAS_TIMING_SCHEMA_VERSION:
        raise _error(
            "atlas.schema_version", f"must equal {ATLAS_TIMING_SCHEMA_VERSION}"
        )
    graph_id = _string(root["graph_id"], "atlas.graph_id")
    workload_id = _string(root["workload_id"], "atlas.workload_id")
    timing_context_digest = _sha256(
        root["timing_context_sha256"], "atlas.timing_context_sha256"
    )

    service: dict[tuple[str, str], float] = {}
    for index, raw in enumerate(_array(root["service"], "atlas.service")):
        field = f"atlas.service[{index}]"
        entry = _object(raw, field)
        _exact_fields(entry, _ATLAS_SERVICE_FIELDS, field)
        key = (
            _string(entry["op_id"], f"{field}.op_id"),
            _string(entry["device_id"], f"{field}.device_id"),
        )
        if key in service:
            raise _error("atlas.service", f"duplicate key: {key!r}")
        service[key] = atlas_duration_s(entry["cycles"], entry["frequency_MHz"])

    movement: dict[tuple[str, str, str, int, str], float] = {}
    for index, raw in enumerate(_array(root["movement"], "atlas.movement")):
        field = f"atlas.movement[{index}]"
        entry = _object(raw, field)
        _exact_fields(entry, _ATLAS_MOVEMENT_FIELDS, field)
        key = (
            _string(entry["tensor_id"], f"{field}.tensor_id"),
            _string(entry["source_device_id"], f"{field}.source_device_id"),
            _string(
                entry["destination_device_id"],
                f"{field}.destination_device_id",
            ),
            _non_negative_int(entry["bytes"], f"{field}.bytes"),
            _string(entry["layout"], f"{field}.layout"),
        )
        if key in movement:
            raise _error("atlas.movement", f"duplicate key: {key!r}")
        movement[key] = atlas_duration_s(
            entry["cycles"], entry["frequency_MHz"]
        )
    return AtlasTimingTable(
        graph_id=graph_id,
        workload_id=workload_id,
        timing_context_sha256=timing_context_digest,
        service_s=service,
        movement_s=movement,
    )


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PriorExportError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    raise PriorExportError(f"non-finite JSON number is forbidden: {value}")


def load_atlas_timings(path: str | Path) -> AtlasTimingTable:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"precomputed ATLAS timing file does not exist: {source}")
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_object_keys,
                parse_constant=_reject_nonstandard_number,
            )
    except json.JSONDecodeError as exc:
        raise PriorExportError(f"invalid JSON in {source}: {exc}") from exc
    return validate_atlas_timings(payload)


def _stable_id(prefix: str, payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _derive_identity(
    *,
    cfg: Mapping[str, Any],
    timing_context: Mapping[str, Any],
    schedule_calls: set[int],
) -> tuple[str, str]:
    config = dict(cfg)
    graph_identity = {
        "declared_graph_id": config.get("hetinfer_graph_id"),
        "snapshot_sha256": timing_context["snapshot_sha256"],
    }
    workload_identity = {
        "declared_workload_id": config.get("hetinfer_workload_id"),
        "timing_cfg": timing_context["timing_cfg"],
        "input_files": timing_context["input_files"],
        "schedule_calls": sorted(schedule_calls),
    }
    # Friendly labels are inputs to the identity, never escape hatches that can
    # override the instantiated graph/config/file-byte digest.
    graph_id = _stable_id("dops-graph", graph_identity)
    workload_id = _stable_id("dops-workload", workload_identity)
    _string(graph_id, "graph_id")
    _string(workload_id, "workload_id")
    return graph_id, workload_id


def _timing_table(
    value: AtlasTimingTable | Mapping[str, Any] | None,
) -> AtlasTimingTable | None:
    if value is None or isinstance(value, AtlasTimingTable):
        return value
    return validate_atlas_timings(value)


def _validate_snapshot_operator_descriptor(
    value: Any,
    field: str,
    *,
    snapshot_phase: str,
    legal_devices: Sequence[str],
    device_types: Mapping[str, str],
) -> dict[str, Any]:
    descriptor = _object(value, field)
    _exact_fields(descriptor, _SNAPSHOT_OPERATOR_DESCRIPTOR_FIELDS, field)
    op_kind = _string(descriptor["op_kind"], f"{field}.op_kind")
    descriptor_phase = _string(descriptor["phase"], f"{field}.phase")
    if descriptor_phase != snapshot_phase:
        raise _error(
            f"{field}.phase",
            f"must equal enclosing snapshot phase {snapshot_phase!r}",
        )
    batch = _positive_int(descriptor["batch"], f"{field}.batch")
    seq_len = _non_negative_int(descriptor["seq_len"], f"{field}.seq_len")
    attrs = _object(descriptor["attrs"], f"{field}.attrs")
    attrs = _canonical_json_value(attrs, f"{field}.attrs")
    raw_layouts = _object(
        descriptor["weight_layout_by_device"],
        f"{field}.weight_layout_by_device",
    )
    if set(raw_layouts) != set(legal_devices):
        raise _error(
            f"{field}.weight_layout_by_device",
            "keys must exactly equal legal_devices; "
            f"missing={sorted(set(legal_devices) - set(raw_layouts))}, "
            f"unexpected={sorted(set(raw_layouts) - set(legal_devices))}",
        )
    layouts = {
        device_id: _string(
            raw_layouts[device_id],
            f"{field}.weight_layout_by_device[{device_id!r}]",
        )
        for device_id in legal_devices
    }
    primitive = _optional_string(
        descriptor["collective_primitive"],
        f"{field}.collective_primitive",
    )
    participants = _unique_strings(
        descriptor["collective_participants"],
        f"{field}.collective_participants",
        nonempty=primitive is not None,
    )
    if primitive is None and participants:
        raise _error(
            f"{field}.collective_participants",
            "must be empty when collective_primitive is null",
        )
    unknown_participants = sorted(set(participants) - set(device_types))
    if unknown_participants:
        raise _error(
            f"{field}.collective_participants",
            f"unknown devices: {unknown_participants}",
        )
    topology = _string(descriptor["topology"], f"{field}.topology")
    return {
        "op_kind": op_kind,
        "phase": descriptor_phase,
        "batch": batch,
        "seq_len": seq_len,
        "attrs": attrs,
        "weight_layout_by_device": layouts,
        "collective_primitive": primitive,
        "collective_participants": list(participants),
        "topology": topology,
    }


def _validate_snapshot_route_descriptor(
    value: Any,
    field: str,
    *,
    source: str,
    destination: str,
    device_types: Mapping[str, str],
) -> dict[str, str]:
    descriptor = _object(value, field)
    _exact_fields(descriptor, _SNAPSHOT_ROUTE_DESCRIPTOR_FIELDS, field)
    topology = _string(descriptor["topology"], f"{field}.topology")
    source_type = _string(
        descriptor["source_device_type"],
        f"{field}.source_device_type",
    ).lower()
    destination_type = _string(
        descriptor["destination_device_type"],
        f"{field}.destination_device_type",
    ).lower()
    if source_type != device_types[source]:
        raise _error(
            f"{field}.source_device_type",
            f"must equal device {source!r} type {device_types[source]!r}",
        )
    if destination_type != device_types[destination]:
        raise _error(
            f"{field}.destination_device_type",
            f"must equal device {destination!r} type {device_types[destination]!r}",
        )
    return {
        "topology": topology,
        "source_device_type": source_type,
        "destination_device_type": destination_type,
    }


def _validate_snapshot_execution_manifest(
    snapshot: Mapping[str, Any],
    field: str,
    *,
    device_types: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate the non-timing input and collective binding manifest."""

    inputs: list[dict[str, Any]] = []
    for index, raw_input in enumerate(_array(snapshot["inputs"], f"{field}.inputs")):
        input_field = f"{field}.inputs[{index}]"
        entry = _object(raw_input, input_field)
        _exact_fields(entry, _SNAPSHOT_INPUT_FIELDS, input_field)
        producer_raw = entry["producer_op_id"]
        producer = (
            None
            if producer_raw is None
            else _string(producer_raw, f"{input_field}.producer_op_id")
        )
        residencies: list[dict[str, str]] = []
        for residency_index, raw_residency in enumerate(
            _array(
                entry["source_residencies"],
                f"{input_field}.source_residencies",
            )
        ):
            residency_field = (
                f"{input_field}.source_residencies[{residency_index}]"
            )
            residency = _object(raw_residency, residency_field)
            _exact_fields(
                residency,
                _SNAPSHOT_SOURCE_RESIDENCY_FIELDS,
                residency_field,
            )
            device_id = _string(
                residency["device_id"], f"{residency_field}.device_id"
            )
            if device_id not in device_types:
                raise _error(residency_field, f"unknown device {device_id!r}")
            residencies.append(
                {
                    "device_id": device_id,
                    "layout": _string(
                        residency["layout"], f"{residency_field}.layout"
                    ),
                }
            )
        destinations = list(
            _unique_strings(
                entry["destination_devices"],
                f"{input_field}.destination_devices",
                nonempty=False,
            )
        )
        unknown_destinations = sorted(set(destinations) - set(device_types))
        if unknown_destinations:
            raise _error(
                f"{input_field}.destination_devices",
                f"unknown devices: {unknown_destinations}",
            )
        inputs.append(
            {
                "consumer_op_id": _string(
                    entry["consumer_op_id"], f"{input_field}.consumer_op_id"
                ),
                "producer_op_id": producer,
                "tensor_id": _string(
                    entry["tensor_id"], f"{input_field}.tensor_id"
                ),
                "semantics": _string(
                    entry["semantics"], f"{input_field}.semantics"
                ),
                "bytes": _non_negative_int(
                    entry["bytes"], f"{input_field}.bytes"
                ),
                "source_residencies": residencies,
                "destination_devices": destinations,
            }
        )

    contexts: dict[str, dict[str, Any]] = {}
    for index, raw_context in enumerate(
        _array(snapshot["collective_contexts"], f"{field}.collective_contexts")
    ):
        context_field = f"{field}.collective_contexts[{index}]"
        entry = _object(raw_context, context_field)
        _exact_fields(
            entry, _SNAPSHOT_COLLECTIVE_CONTEXT_FIELDS, context_field
        )
        op_id = _string(entry["op_id"], f"{context_field}.op_id")
        if op_id in contexts:
            raise _error(f"{field}.collective_contexts", f"duplicate op_id {op_id!r}")
        context = {
            "op_id": op_id,
            "primitive": _string(
                entry["primitive"], f"{context_field}.primitive"
            ),
            "topology": _string(
                entry["topology"], f"{context_field}.topology"
            ),
            "canonical_device_id": _string(
                entry["canonical_device_id"],
                f"{context_field}.canonical_device_id",
            ),
            "participant_device_ids": list(
                _unique_strings(
                    entry["participant_device_ids"],
                    f"{context_field}.participant_device_ids",
                    nonempty=True,
                )
            ),
            "output_device_ids": list(
                _unique_strings(
                    entry["output_device_ids"],
                    f"{context_field}.output_device_ids",
                    nonempty=True,
                )
            ),
            "resource_device_ids": list(
                _unique_strings(
                    entry["resource_device_ids"],
                    f"{context_field}.resource_device_ids",
                    nonempty=True,
                )
            ),
            "tensor_bytes": _non_negative_int(
                entry["tensor_bytes"], f"{context_field}.tensor_bytes"
            ),
            "internal_transport": _string(
                entry["internal_transport"],
                f"{context_field}.internal_transport",
            ),
        }
        all_context_devices = {
            context["canonical_device_id"],
            *context["participant_device_ids"],
            *context["output_device_ids"],
            *context["resource_device_ids"],
        }
        unknown_context_devices = sorted(all_context_devices - set(device_types))
        if unknown_context_devices:
            raise _error(context_field, f"unknown devices: {unknown_context_devices}")
        if context["internal_transport"] != "included_in_t_service":
            raise _error(
                f"{context_field}.internal_transport",
                "must equal 'included_in_t_service'",
            )
        contexts[op_id] = context
    return inputs, contexts


def _validate_timing_context(value: Any, field: str) -> dict[str, Any]:
    context = _object(value, field)
    _exact_fields(context, _ATLAS_TIMING_CONTEXT_FIELDS, field)
    snapshot_sha256 = _sha256(
        context["snapshot_sha256"], f"{field}.snapshot_sha256"
    )
    timing_cfg = _object(context["timing_cfg"], f"{field}.timing_cfg")
    timing_cfg = _canonical_json_value(timing_cfg, f"{field}.timing_cfg")
    input_files: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for index, raw_entry in enumerate(
        _array(context["input_files"], f"{field}.input_files")
    ):
        entry_field = f"{field}.input_files[{index}]"
        entry = _object(raw_entry, entry_field)
        _exact_fields(entry, _ATLAS_TIMING_INPUT_FILE_FIELDS, entry_field)
        config_key = _string(entry["config_key"], f"{entry_field}.config_key")
        if config_key in seen_keys:
            raise _error(f"{field}.input_files", f"duplicate config_key: {config_key!r}")
        seen_keys.add(config_key)
        input_files.append(
            {
                "config_key": config_key,
                "path": _string(entry["path"], f"{entry_field}.path"),
                "size_bytes": _non_negative_int(
                    entry["size_bytes"], f"{entry_field}.size_bytes"
                ),
                "sha256": _sha256(entry["sha256"], f"{entry_field}.sha256"),
            }
        )
    input_files.sort(key=lambda item: item["config_key"])
    return {
        "snapshot_sha256": snapshot_sha256,
        "timing_cfg": timing_cfg,
        "input_files": input_files,
    }


def _atlas_request_outline(
    *,
    cfg: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
) -> tuple[
    str,
    str,
    dict[str, Any],
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Discover marked ATLAS keys before the formal builder validates them."""

    if not isinstance(cfg, Mapping):
        raise _error("cfg", "expected an object")
    if isinstance(snapshots, (str, bytes)) or not isinstance(snapshots, Sequence):
        raise _error("snapshots", "expected a non-string sequence")
    if not snapshots:
        raise _error("snapshots", "cannot be empty")

    device_types: dict[str, str] = {}
    operators: list[dict[str, Any]] = []
    service_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    movement_by_key: dict[
        tuple[str, str, str, int, str], dict[str, Any]
    ] = {}
    seen_calls: set[int] = set()
    seen_ops: set[str] = set()
    seen_routes: set[tuple[str, str, str, int, str]] = set()
    previous_call_index = 0

    for snapshot_index, raw_snapshot in enumerate(snapshots):
        snapshot_field = f"snapshots[{snapshot_index}]"
        snapshot = _object(raw_snapshot, snapshot_field)
        _exact_fields(snapshot, _SNAPSHOT_FIELDS, snapshot_field)
        call_index = _positive_int(
            snapshot["schedule_call_index"],
            f"{snapshot_field}.schedule_call_index",
        )
        if call_index in seen_calls:
            raise _error("snapshots", f"duplicate schedule_call_index: {call_index}")
        if call_index != previous_call_index + 1:
            raise _error(
                "snapshots",
                "schedule_call_index values must be contiguous and ordered "
                f"from 1; expected {previous_call_index + 1}, got {call_index}",
            )
        previous_call_index = call_index
        seen_calls.add(call_index)
        snapshot_phase = _string(snapshot["phase"], f"{snapshot_field}.phase")

        for device_index, raw_device in enumerate(
            _array(snapshot["devices"], f"{snapshot_field}.devices")
        ):
            field = f"{snapshot_field}.devices[{device_index}]"
            device = _object(raw_device, field)
            _exact_fields(device, _SNAPSHOT_DEVICE_FIELDS, field)
            device_id = _string(device["device_id"], f"{field}.device_id")
            device_type = _string(
                device["device_type"], f"{field}.device_type"
            ).lower()
            previous = device_types.get(device_id)
            if previous is not None and previous != device_type:
                raise _error(
                    "snapshots.devices",
                    f"device {device_id!r} changes type from {previous!r} to {device_type!r}",
                )
            device_types[device_id] = device_type

        snapshot_inputs, context_by_op = _validate_snapshot_execution_manifest(
            snapshot,
            snapshot_field,
            device_types=device_types,
        )

        for operator_index, raw_operator in enumerate(
            _array(snapshot["operators"], f"{snapshot_field}.operators")
        ):
            field = f"{snapshot_field}.operators[{operator_index}]"
            operator = _object(raw_operator, field)
            _exact_fields(operator, _SNAPSHOT_OPERATOR_FIELDS, field)
            op_id = _string(operator["op_id"], f"{field}.op_id")
            if op_id in seen_ops:
                raise _error("snapshots.operators", f"duplicate op_id: {op_id!r}")
            seen_ops.add(op_id)
            dependencies = _unique_strings(
                operator["dependencies"], f"{field}.dependencies", nonempty=False
            )
            legal_devices = _unique_strings(
                operator["legal_devices"], f"{field}.legal_devices", nonempty=True
            )
            unknown_devices = sorted(set(legal_devices) - set(device_types))
            if unknown_devices:
                raise _error(field, f"unknown legal devices: {unknown_devices}")
            raw_service = _object(operator["service_s"], f"{field}.service_s")
            if set(raw_service) != set(legal_devices):
                raise _error(
                    f"{field}.service_s",
                    "keys must exactly equal legal_devices; "
                    f"missing={sorted(set(legal_devices) - set(raw_service))}, "
                    f"unexpected={sorted(set(raw_service) - set(legal_devices))}",
                )
            descriptor = _validate_snapshot_operator_descriptor(
                operator["atlas_descriptor"],
                f"{field}.atlas_descriptor",
                snapshot_phase=snapshot_phase,
                legal_devices=legal_devices,
                device_types=device_types,
            )
            context = context_by_op.get(op_id)
            if context is None:
                if descriptor["collective_primitive"] is not None:
                    raise _error(
                        f"{field}.atlas_descriptor.collective_primitive",
                        "operator has no collective_context",
                    )
                if descriptor["attrs"].get("collective_context") is not None:
                    raise _error(
                        f"{field}.atlas_descriptor.attrs.collective_context",
                        "must be null for an ordinary operator",
                    )
                if descriptor["attrs"].get("collective_input_bindings") != []:
                    raise _error(
                        f"{field}.atlas_descriptor.attrs.collective_input_bindings",
                        "must be empty for an ordinary operator",
                    )
            else:
                if set(legal_devices) != {context["canonical_device_id"]}:
                    raise _error(
                        f"{field}.legal_devices",
                        "fixed collective must expose only its canonical device",
                    )
                if descriptor["collective_primitive"] != context["primitive"]:
                    raise _error(
                        f"{field}.atlas_descriptor.collective_primitive",
                        "must equal collective_context.primitive",
                    )
                if set(descriptor["collective_participants"]) != set(
                    context["participant_device_ids"]
                ):
                    raise _error(
                        f"{field}.atlas_descriptor.collective_participants",
                        "must equal collective_context participants",
                    )
                if descriptor["topology"] != context["topology"]:
                    raise _error(
                        f"{field}.atlas_descriptor.topology",
                        "must equal collective_context.topology",
                    )
                if descriptor["attrs"].get("collective_context") != context:
                    raise _error(
                        f"{field}.atlas_descriptor.attrs.collective_context",
                        "must embed the complete collective_context",
                    )
                expected_bindings = sorted(
                    [
                        entry
                        for entry in snapshot_inputs
                        if entry["consumer_op_id"] == op_id
                        and entry["semantics"] == "collective_staging"
                    ],
                    key=lambda entry: (
                        entry["producer_op_id"] or "",
                        entry["tensor_id"],
                    ),
                )
                actual_bindings = descriptor["attrs"].get(
                    "collective_input_bindings"
                )
                if actual_bindings != expected_bindings:
                    raise _error(
                        f"{field}.atlas_descriptor.attrs.collective_input_bindings",
                        "must embed every fixed staging binding exactly",
                    )
            pim_collective = bool(
                context is not None
                and any(
                    device_types[device_id] == "pim"
                    for device_id in context["resource_device_ids"]
                )
            )
            for device_id in legal_devices:
                must_use_atlas = (
                    device_types[device_id] == "pim" or pim_collective
                )
                if (raw_service[device_id] is None) != must_use_atlas:
                    raise _error(
                        f"{field}.service_s[{device_id!r}]",
                        "must be null exactly for PIM execution or a collective "
                        "whose fixed resource set includes PIM",
                    )
                if must_use_atlas:
                    key = (op_id, device_id)
                    service_by_key[key] = {
                        "op_id": op_id,
                        "device_id": device_id,
                        "descriptor": {
                            "op_kind": descriptor["op_kind"],
                            "phase": descriptor["phase"],
                            "batch": descriptor["batch"],
                            "seq_len": descriptor["seq_len"],
                            "attrs": descriptor["attrs"],
                            "weight_layout": descriptor[
                                "weight_layout_by_device"
                            ][device_id],
                            "collective_primitive": descriptor[
                                "collective_primitive"
                            ],
                            "collective_participants": descriptor[
                                "collective_participants"
                            ],
                            "topology": descriptor["topology"],
                            "target_device_type": device_types[device_id],
                        },
                    }
            operators.append(
                {
                    "op_id": op_id,
                    "dependencies": list(dependencies),
                    "legal_devices": list(legal_devices),
                }
            )

        for route_index, raw_route in enumerate(
            _array(snapshot["routes"], f"{snapshot_field}.routes")
        ):
            field = f"{snapshot_field}.routes[{route_index}]"
            route = _object(raw_route, field)
            _exact_fields(route, _SNAPSHOT_ROUTE_FIELDS, field)
            key = (
                _string(route["tensor_id"], f"{field}.tensor_id"),
                _string(
                    route["source_device_id"], f"{field}.source_device_id"
                ),
                _string(
                    route["destination_device_id"],
                    f"{field}.destination_device_id",
                ),
                _non_negative_int(route["bytes"], f"{field}.bytes"),
                _string(route["layout"], f"{field}.layout"),
            )
            unknown_route_devices = sorted(
                {key[1], key[2]} - set(device_types)
            )
            if unknown_route_devices:
                raise _error(
                    field,
                    f"unknown route devices: {unknown_route_devices}",
                )
            if key in seen_routes:
                raise _error("snapshots.routes", f"duplicate route key: {key!r}")
            seen_routes.add(key)
            requires_atlas = route["requires_atlas"]
            if not isinstance(requires_atlas, bool):
                raise _error(f"{field}.requires_atlas", "expected a boolean")
            if requires_atlas:
                descriptor = _validate_snapshot_route_descriptor(
                    route["atlas_descriptor"],
                    f"{field}.atlas_descriptor",
                    source=key[1],
                    destination=key[2],
                    device_types=device_types,
                )
                movement_by_key[key] = {
                    "tensor_id": key[0],
                    "source_device_id": key[1],
                    "destination_device_id": key[2],
                    "bytes": key[3],
                    "layout": key[4],
                    "descriptor": descriptor,
                }
            else:
                _validate_snapshot_route_descriptor(
                    route["atlas_descriptor"],
                    f"{field}.atlas_descriptor",
                    source=key[1],
                    destination=key[2],
                    device_types=device_types,
                )

    service = [service_by_key[key] for key in sorted(service_by_key)]
    movement = [movement_by_key[key] for key in sorted(movement_by_key)]
    timing_context = _timing_context(cfg=cfg, snapshots=snapshots)
    graph_id, workload_id = _derive_identity(
        cfg=cfg,
        timing_context=timing_context,
        schedule_calls=seen_calls,
    )
    context_digest = _timing_context_sha256(
        timing_context=timing_context,
        service=service,
        movement=movement,
    )
    return (
        graph_id,
        workload_id,
        timing_context,
        context_digest,
        service,
        movement,
    )


def validate_atlas_timing_request(payload: Any) -> dict[str, Any]:
    """Validate and canonicalize the exact offline ATLAS work manifest."""

    root = _object(payload, "atlas_request.<root>")
    _exact_fields(root, _ATLAS_REQUEST_ROOT_FIELDS, "atlas_request.<root>")
    if root["schema"] != ATLAS_TIMING_REQUEST_SCHEMA:
        raise _error(
            "atlas_request.schema",
            f"must equal {ATLAS_TIMING_REQUEST_SCHEMA!r}",
        )
    version = root["schema_version"]
    if isinstance(version, bool) or version != ATLAS_TIMING_REQUEST_SCHEMA_VERSION:
        raise _error(
            "atlas_request.schema_version",
            f"must equal {ATLAS_TIMING_REQUEST_SCHEMA_VERSION}",
        )
    graph_id = _string(root["graph_id"], "atlas_request.graph_id")
    workload_id = _string(root["workload_id"], "atlas_request.workload_id")
    context_digest = _sha256(
        root["timing_context_sha256"],
        "atlas_request.timing_context_sha256",
    )
    timing_context = _validate_timing_context(
        root["timing_context"], "atlas_request.timing_context"
    )

    service: list[dict[str, Any]] = []
    seen_service: set[tuple[str, str]] = set()
    for index, raw_entry in enumerate(
        _array(root["service"], "atlas_request.service")
    ):
        field = f"atlas_request.service[{index}]"
        entry = _object(raw_entry, field)
        _exact_fields(entry, _ATLAS_REQUEST_SERVICE_FIELDS, field)
        key = (
            _string(entry["op_id"], f"{field}.op_id"),
            _string(entry["device_id"], f"{field}.device_id"),
        )
        if key in seen_service:
            raise _error("atlas_request.service", f"duplicate key: {key!r}")
        seen_service.add(key)
        descriptor_field = f"{field}.descriptor"
        descriptor = _object(entry["descriptor"], descriptor_field)
        _exact_fields(
            descriptor,
            _ATLAS_REQUEST_SERVICE_DESCRIPTOR_FIELDS,
            descriptor_field,
        )
        primitive = _optional_string(
            descriptor["collective_primitive"],
            f"{descriptor_field}.collective_primitive",
        )
        participants = _unique_strings(
            descriptor["collective_participants"],
            f"{descriptor_field}.collective_participants",
            nonempty=primitive is not None,
        )
        if primitive is None and participants:
            raise _error(
                f"{descriptor_field}.collective_participants",
                "must be empty when collective_primitive is null",
            )
        service.append(
            {
                "op_id": key[0],
                "device_id": key[1],
                "descriptor": {
                    "op_kind": _string(
                        descriptor["op_kind"], f"{descriptor_field}.op_kind"
                    ),
                    "phase": _string(
                        descriptor["phase"], f"{descriptor_field}.phase"
                    ),
                    "batch": _positive_int(
                        descriptor["batch"], f"{descriptor_field}.batch"
                    ),
                    "seq_len": _non_negative_int(
                        descriptor["seq_len"], f"{descriptor_field}.seq_len"
                    ),
                    "attrs": _canonical_json_value(
                        _object(
                            descriptor["attrs"], f"{descriptor_field}.attrs"
                        ),
                        f"{descriptor_field}.attrs",
                    ),
                    "weight_layout": _string(
                        descriptor["weight_layout"],
                        f"{descriptor_field}.weight_layout",
                    ),
                    "collective_primitive": primitive,
                    "collective_participants": list(participants),
                    "topology": _string(
                        descriptor["topology"],
                        f"{descriptor_field}.topology",
                    ),
                    "target_device_type": _string(
                        descriptor["target_device_type"],
                        f"{descriptor_field}.target_device_type",
                    ).lower(),
                },
            }
        )

    movement: list[dict[str, Any]] = []
    seen_movement: set[tuple[str, str, str, int, str]] = set()
    for index, raw_entry in enumerate(
        _array(root["movement"], "atlas_request.movement")
    ):
        field = f"atlas_request.movement[{index}]"
        entry = _object(raw_entry, field)
        _exact_fields(entry, _ATLAS_REQUEST_MOVEMENT_FIELDS, field)
        key = (
            _string(entry["tensor_id"], f"{field}.tensor_id"),
            _string(
                entry["source_device_id"], f"{field}.source_device_id"
            ),
            _string(
                entry["destination_device_id"],
                f"{field}.destination_device_id",
            ),
            _non_negative_int(entry["bytes"], f"{field}.bytes"),
            _string(entry["layout"], f"{field}.layout"),
        )
        if key in seen_movement:
            raise _error("atlas_request.movement", f"duplicate key: {key!r}")
        seen_movement.add(key)
        descriptor_field = f"{field}.descriptor"
        descriptor = _object(entry["descriptor"], descriptor_field)
        _exact_fields(
            descriptor,
            _ATLAS_REQUEST_MOVEMENT_DESCRIPTOR_FIELDS,
            descriptor_field,
        )
        movement.append(
            {
                "tensor_id": key[0],
                "source_device_id": key[1],
                "destination_device_id": key[2],
                "bytes": key[3],
                "layout": key[4],
                "descriptor": {
                    "topology": _string(
                        descriptor["topology"],
                        f"{descriptor_field}.topology",
                    ),
                    "source_device_type": _string(
                        descriptor["source_device_type"],
                        f"{descriptor_field}.source_device_type",
                    ).lower(),
                    "destination_device_type": _string(
                        descriptor["destination_device_type"],
                        f"{descriptor_field}.destination_device_type",
                    ).lower(),
                },
            }
        )

    service.sort(key=lambda item: (item["op_id"], item["device_id"]))
    movement.sort(
        key=lambda item: (
            item["tensor_id"],
            item["source_device_id"],
            item["destination_device_id"],
            item["bytes"],
            item["layout"],
        )
    )
    expected_digest = _timing_context_sha256(
        timing_context=timing_context,
        service=service,
        movement=movement,
    )
    if context_digest != expected_digest:
        raise _error(
            "atlas_request.timing_context_sha256",
            f"must equal canonical request digest {expected_digest!r}",
        )
    return {
        "schema": ATLAS_TIMING_REQUEST_SCHEMA,
        "schema_version": ATLAS_TIMING_REQUEST_SCHEMA_VERSION,
        "graph_id": graph_id,
        "workload_id": workload_id,
        "timing_context_sha256": context_digest,
        "timing_context": timing_context,
        "service": service,
        "movement": movement,
    }


def build_prior_artifact(
    *,
    cfg: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    atlas_timings: AtlasTimingTable | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project completed schedule snapshots into the strict static v1 contract."""

    if isinstance(snapshots, (str, bytes)) or not isinstance(snapshots, Sequence):
        raise _error("snapshots", "expected a non-string sequence")
    if not snapshots:
        raise _error("snapshots", "cannot be empty")
    (
        graph_id,
        workload_id,
        _request_context,
        timing_context_digest,
        request_service,
        request_movement,
    ) = _atlas_request_outline(cfg=cfg, snapshots=snapshots)
    request_service_keys = {
        (entry["op_id"], entry["device_id"]) for entry in request_service
    }
    request_movement_keys = {
        tuple(entry[key] for key in _ROUTE_KEY_FIELDS)
        for entry in request_movement
    }
    atlas = _timing_table(atlas_timings)

    device_types: dict[str, str] = {}
    operators: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    collective_contexts: list[dict[str, Any]] = []
    placement: list[dict[str, Any]] = []
    service_table: list[dict[str, Any]] = []
    legal_routes: list[dict[str, Any]] = []
    move_table: list[dict[str, Any]] = []
    seen_calls: set[int] = set()
    seen_ops: set[str] = set()
    seen_routes: set[tuple[str, str, str, int, str]] = set()
    expected_atlas_service: set[tuple[str, str]] = set()
    expected_atlas_movement: set[tuple[str, str, str, int, str]] = set()
    previous_call_index = 0

    for snapshot_index, raw_snapshot in enumerate(snapshots):
        snapshot_field = f"snapshots[{snapshot_index}]"
        snapshot = _object(raw_snapshot, snapshot_field)
        _exact_fields(snapshot, _SNAPSHOT_FIELDS, snapshot_field)
        call_index = _positive_int(
            snapshot["schedule_call_index"],
            f"{snapshot_field}.schedule_call_index",
        )
        if call_index in seen_calls:
            raise _error("snapshots", f"duplicate schedule_call_index: {call_index}")
        if call_index != previous_call_index + 1:
            raise _error(
                "snapshots",
                "schedule_call_index values must be contiguous and ordered "
                f"from 1; expected {previous_call_index + 1}, got {call_index}",
            )
        previous_call_index = call_index
        seen_calls.add(call_index)
        _string(snapshot["phase"], f"{snapshot_field}.phase")

        for device_index, raw_device in enumerate(
            _array(snapshot["devices"], f"{snapshot_field}.devices")
        ):
            field = f"{snapshot_field}.devices[{device_index}]"
            device = _object(raw_device, field)
            _exact_fields(device, _SNAPSHOT_DEVICE_FIELDS, field)
            device_id = _string(device["device_id"], f"{field}.device_id")
            device_type = _string(device["device_type"], f"{field}.device_type").lower()
            previous = device_types.get(device_id)
            if previous is not None and previous != device_type:
                raise _error(
                    "snapshots.devices",
                    f"device {device_id!r} changes type from {previous!r} to {device_type!r}",
                )
            device_types[device_id] = device_type

        for operator_index, raw_operator in enumerate(
            _array(snapshot["operators"], f"{snapshot_field}.operators")
        ):
            field = f"{snapshot_field}.operators[{operator_index}]"
            operator = _object(raw_operator, field)
            _exact_fields(operator, _SNAPSHOT_OPERATOR_FIELDS, field)
            op_id = _string(operator["op_id"], f"{field}.op_id")
            if op_id in seen_ops:
                raise _error("snapshots.operators", f"duplicate op_id: {op_id!r}")
            seen_ops.add(op_id)
            dependencies = _unique_strings(
                operator["dependencies"], f"{field}.dependencies", nonempty=False
            )
            legal_devices = _unique_strings(
                operator["legal_devices"], f"{field}.legal_devices", nonempty=True
            )
            unknown_devices = sorted(set(legal_devices) - set(device_types))
            if unknown_devices:
                raise _error(field, f"unknown legal devices: {unknown_devices}")
            expert_device = _string(
                operator["expert_device"], f"{field}.expert_device"
            )
            if expert_device not in legal_devices:
                raise _error(
                    field,
                    f"expert device {expert_device!r} is not in legal_devices",
                )
            raw_service = _object(operator["service_s"], f"{field}.service_s")
            if set(raw_service) != set(legal_devices):
                raise _error(
                    f"{field}.service_s",
                    "keys must exactly equal legal_devices; "
                    f"missing={sorted(set(legal_devices) - set(raw_service))}, "
                    f"unexpected={sorted(set(raw_service) - set(legal_devices))}",
                )

            operators.append(
                {
                    "op_id": op_id,
                    "dependencies": list(dependencies),
                    "legal_devices": list(legal_devices),
                }
            )
            placement.append({"op_id": op_id, "device_id": expert_device})
            for device_id in legal_devices:
                modeled_duration = raw_service[device_id]
                if modeled_duration is None:
                    expected_atlas_service.add((op_id, device_id))
                    if atlas is None:
                        raise PriorExportError(
                            "precomputed ATLAS timings are required for "
                            f"service {(op_id, device_id)!r}"
                        )
                    duration_s = atlas.service_time_s(op_id, device_id)
                else:
                    duration_s = _duration(
                        modeled_duration,
                        f"{field}.service_s[{device_id!r}]",
                    )
                service_table.append(
                    {
                        "op_id": op_id,
                        "device_id": device_id,
                        "duration_s": duration_s,
                    }
                )

        for input_index, raw_input in enumerate(
            _array(snapshot["inputs"], f"{snapshot_field}.inputs")
        ):
            field = f"{snapshot_field}.inputs[{input_index}]"
            inputs.append(_object(raw_input, field))

        for context_index, raw_context in enumerate(
            _array(
                snapshot["collective_contexts"],
                f"{snapshot_field}.collective_contexts",
            )
        ):
            field = f"{snapshot_field}.collective_contexts[{context_index}]"
            collective_contexts.append(_object(raw_context, field))

        for route_index, raw_route in enumerate(
            _array(snapshot["routes"], f"{snapshot_field}.routes")
        ):
            field = f"{snapshot_field}.routes[{route_index}]"
            route = _object(raw_route, field)
            _exact_fields(route, _SNAPSHOT_ROUTE_FIELDS, field)
            tensor_id = _string(route["tensor_id"], f"{field}.tensor_id")
            source = _string(
                route["source_device_id"], f"{field}.source_device_id"
            )
            destination = _string(
                route["destination_device_id"],
                f"{field}.destination_device_id",
            )
            bytes_ = _non_negative_int(route["bytes"], f"{field}.bytes")
            layout = _string(route["layout"], f"{field}.layout")
            unknown_devices = sorted({source, destination} - set(device_types))
            if unknown_devices:
                raise _error(field, f"unknown route devices: {unknown_devices}")
            key = (tensor_id, source, destination, bytes_, layout)
            if key in seen_routes:
                raise _error("snapshots.routes", f"duplicate route key: {key!r}")
            seen_routes.add(key)
            requires_atlas = route["requires_atlas"]
            if not isinstance(requires_atlas, bool):
                raise _error(f"{field}.requires_atlas", "expected a boolean")

            resident = source == destination
            pim_related = (
                device_types[source] == "pim" or device_types[destination] == "pim"
            )
            if resident:
                if requires_atlas:
                    raise _error(field, "resident routes cannot require ATLAS")
                duration_s = _duration(route["duration_s"], f"{field}.duration_s")
                if duration_s != 0.0:
                    raise _error(field, "resident route duration must be exactly zero")
            elif pim_related and not requires_atlas:
                raise _error(
                    field,
                    "non-resident PIM routes must require ATLAS",
                )
            if not resident and requires_atlas:
                if route["duration_s"] is not None:
                    raise _error(
                        field,
                        "ATLAS routes cannot carry a modeled duration",
                    )
                expected_atlas_movement.add(key)
                if atlas is None:
                    raise PriorExportError(
                        f"precomputed ATLAS timings are required for route {key!r}"
                    )
                duration_s = atlas.movement_time_s(*key)
            elif not resident:
                duration_s = _duration(route["duration_s"], f"{field}.duration_s")

            route_payload = {
                "tensor_id": tensor_id,
                "source_device_id": source,
                "destination_device_id": destination,
                "bytes": bytes_,
                "layout": layout,
            }
            legal_routes.append(route_payload)
            move_table.append({**route_payload, "duration_s": duration_s})

    unknown_dependencies = sorted(
        {
            dependency
            for operator in operators
            for dependency in operator["dependencies"]
            if dependency not in seen_ops
        }
    )
    if unknown_dependencies:
        raise _error("snapshots.operators", f"unknown dependencies: {unknown_dependencies}")
    if len(device_types) < 2:
        raise _error("snapshots.devices", "requires at least two devices")
    if not legal_routes:
        raise _error("snapshots.routes", "cannot be empty")

    if expected_atlas_service != request_service_keys:
        raise _error(
            "snapshots.operators",
            "ATLAS service request keys disagree with null service entries",
        )
    if expected_atlas_movement != request_movement_keys:
        raise _error(
            "snapshots.routes",
            "ATLAS movement request keys disagree with marked routes",
        )
    if atlas is not None and atlas.graph_id != graph_id:
        raise _error(
            "atlas.graph_id",
            f"must equal snapshot graph_id {graph_id!r}",
        )
    if atlas is not None and atlas.workload_id != workload_id:
        raise _error(
            "atlas.workload_id",
            f"must equal snapshot workload_id {workload_id!r}",
        )
    if (
        atlas is not None
        and atlas.timing_context_sha256 != timing_context_digest
    ):
        raise _error(
            "atlas.timing_context_sha256",
            "must equal the digest derived from the complete snapshots, "
            "timing configuration, input-file bytes, and request descriptors; "
            f"expected {timing_context_digest!r}",
        )

    actual_atlas_service = set() if atlas is None else set(atlas.service_s)
    actual_atlas_movement = set() if atlas is None else set(atlas.movement_s)
    if actual_atlas_service != expected_atlas_service:
        raise _error(
            "atlas.service",
            "must exactly cover service_s=null keys; "
            f"missing={sorted(expected_atlas_service - actual_atlas_service)}, "
            f"unexpected={sorted(actual_atlas_service - expected_atlas_service)}",
        )
    if actual_atlas_movement != expected_atlas_movement:
        raise _error(
            "atlas.movement",
            "must exactly cover requires_atlas=true route keys; "
            f"missing={sorted(expected_atlas_movement - actual_atlas_movement)}, "
            f"unexpected={sorted(actual_atlas_movement - expected_atlas_movement)}",
        )

    payload = {
        "schema": "dops.hetinfer_prior.v1",
        "schema_version": 1,
        "graph_id": graph_id,
        "workload_id": workload_id,
        "time_unit": "seconds",
        "devices": [
            {"device_id": device_id} for device_id in sorted(device_types)
        ],
        "operators": operators,
        "inputs": inputs,
        "collective_contexts": collective_contexts,
        "legal_movement_routes": legal_routes,
        "expert_placement": placement,
        "t_service": service_table,
        "t_move": move_table,
    }
    return validate_prior_artifact(payload).payload


def build_atlas_timing_request(
    *,
    cfg: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact ATLAS work manifest for one completed snapshot set.

    The key discovery pass is followed by the normal prior builder with
    zero-cycle timing placeholders.  Consequently a request is never emitted
    for a partial or structurally invalid scheduler snapshot.
    """

    (
        graph_id,
        workload_id,
        timing_context,
        context_digest,
        service,
        movement,
    ) = _atlas_request_outline(cfg=cfg, snapshots=snapshots)
    request = validate_atlas_timing_request(
        {
            "schema": ATLAS_TIMING_REQUEST_SCHEMA,
            "schema_version": ATLAS_TIMING_REQUEST_SCHEMA_VERSION,
            "graph_id": graph_id,
            "workload_id": workload_id,
            "timing_context_sha256": context_digest,
            "timing_context": timing_context,
            "service": service,
            "movement": movement,
        }
    )
    zero_cycle_timings = {
        "schema": ATLAS_TIMING_SCHEMA,
        "schema_version": ATLAS_TIMING_SCHEMA_VERSION,
        "graph_id": graph_id,
        "workload_id": workload_id,
        "timing_context_sha256": context_digest,
        "service": [
            {
                "op_id": entry["op_id"],
                "device_id": entry["device_id"],
                "cycles": 0,
                "frequency_MHz": 1,
            }
            for entry in request["service"]
        ],
        "movement": [
            {
                key: entry[key]
                for key in _ROUTE_KEY_FIELDS
            }
            | {"cycles": 0, "frequency_MHz": 1}
            for entry in request["movement"]
        ],
    }
    build_prior_artifact(
        cfg=cfg,
        snapshots=snapshots,
        atlas_timings=zero_cycle_timings,
    )
    return request


def snapshots_require_atlas(
    snapshots: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether the validated snapshots contain any ATLAS-marked key."""

    request = build_atlas_timing_request(cfg={}, snapshots=snapshots)
    return bool(request["service"] or request["movement"])


def _write_atlas_timing_request(
    payload: Mapping[str, Any],
    destination: Path,
    *,
    overwrite: bool,
) -> Path:
    validated = validate_atlas_timing_request(payload)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite ATLAS timing request: {destination}"
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
                validated,
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


def export_atlas_timing_request(
    *,
    cfg: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    output: str | Path,
    overwrite: bool = False,
) -> Path:
    """Fully validate, then atomically write the offline ATLAS request."""

    request = build_atlas_timing_request(cfg=cfg, snapshots=snapshots)
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    return _write_atlas_timing_request(
        request,
        destination,
        overwrite=overwrite,
    )


def export_prior_artifact(
    *,
    cfg: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    output: str | Path,
    atlas_timings: AtlasTimingTable | Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Build first, then atomically write; coverage failures leave no file."""

    artifact = build_prior_artifact(
        cfg=cfg,
        snapshots=snapshots,
        atlas_timings=atlas_timings,
    )
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    return write_prior_artifact(artifact, destination, overwrite=overwrite)


__all__ = [
    "ATLAS_TIMING_REQUEST_SCHEMA",
    "ATLAS_TIMING_REQUEST_SCHEMA_VERSION",
    "ATLAS_TIMING_SCHEMA",
    "ATLAS_TIMING_SCHEMA_VERSION",
    "AtlasTimingTable",
    "PriorExportError",
    "atlas_duration_s",
    "build_atlas_timing_request",
    "build_prior_artifact",
    "export_atlas_timing_request",
    "export_prior_artifact",
    "load_atlas_timings",
    "snapshots_require_atlas",
    "validate_atlas_timing_request",
    "validate_atlas_timings",
]
