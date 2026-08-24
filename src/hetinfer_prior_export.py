"""Project native DOPS schedule snapshots into the Het-Infer prior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_prior_artifact(
    *,
    cfg: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the prior from timings already produced by DOPS LUT/AiM models."""

    devices: set[str] = set()
    operators: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    collective_contexts: list[dict[str, Any]] = []
    placement: list[dict[str, Any]] = []
    service_table: list[dict[str, Any]] = []
    legal_routes: list[dict[str, Any]] = []
    move_table: list[dict[str, Any]] = []

    for snapshot in snapshots:
        devices.update(device["device_id"] for device in snapshot["devices"])
        for operator in snapshot["operators"]:
            op_id = operator["op_id"]
            operators.append(
                {
                    "op_id": op_id,
                    "dependencies": list(operator["dependencies"]),
                    "legal_devices": list(operator["legal_devices"]),
                }
            )
            placement.append(
                {"op_id": op_id, "device_id": operator["expert_device"]}
            )
            for device_id, duration_s in operator["service_s"].items():
                if duration_s is None:
                    raise RuntimeError(
                        f"native DOPS service timing is missing for {(op_id, device_id)!r}"
                    )
                service_table.append(
                    {
                        "op_id": op_id,
                        "device_id": device_id,
                        "duration_s": float(duration_s),
                    }
                )

        inputs.extend(dict(item) for item in snapshot["inputs"])
        collective_contexts.extend(
            dict(item) for item in snapshot["collective_contexts"]
        )
        for route in snapshot["routes"]:
            route_payload = {
                "tensor_id": route["tensor_id"],
                "source_device_id": route["source_device_id"],
                "destination_device_id": route["destination_device_id"],
                "bytes": int(route["bytes"]),
                "layout": route["layout"],
            }
            if route["duration_s"] is None:
                raise RuntimeError(
                    "native DOPS movement timing is missing for "
                    f"{tuple(route_payload.values())!r}"
                )
            legal_routes.append(route_payload)
            move_table.append(
                {**route_payload, "duration_s": float(route["duration_s"])}
            )

    return {
        "schema": "dops.hetinfer_prior.v1",
        "schema_version": 1,
        "graph_id": str(cfg["hetinfer_graph_id"]),
        "workload_id": str(cfg["hetinfer_workload_id"]),
        "time_unit": "seconds",
        "devices": [{"device_id": device_id} for device_id in sorted(devices)],
        "operators": operators,
        "inputs": inputs,
        "collective_contexts": collective_contexts,
        "legal_movement_routes": legal_routes,
        "expert_placement": placement,
        "t_service": service_table,
        "t_move": move_table,
    }


__all__ = ["build_prior_artifact"]
