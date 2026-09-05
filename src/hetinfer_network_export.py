"""Export the native DOPS schedule as a Het-Infer network manifest."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

NETWORK_SCHEMA = "dops.hetinfer_network.v1"
NETWORK_SCHEMA_VERSION = 1
POLICY_DEVICES = ("Ascend_910B_NPU0", "PIM0", "PIM1")


def _token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").upper()).strip("_")


def _op_role(name: Any, attrs: Mapping[str, Any]) -> str:
    if attrs.get("expert") is not None:
        return "EXPERT"
    token = _token(attrs.get("op_role") or name)
    roles = {
        "LN": "LAYERNORM",
        "LN1": "LAYERNORM",
        "LN2": "LAYERNORM",
        "LAYERNORM": "LAYERNORM",
        "RMSNORM": "LAYERNORM",
        "WQ": "Q_PROJ",
        "Q": "Q_PROJ",
        "Q_PROJ": "Q_PROJ",
        "WK": "K_PROJ",
        "K": "K_PROJ",
        "K_PROJ": "K_PROJ",
        "WV": "V_PROJ",
        "V": "V_PROJ",
        "V_PROJ": "V_PROJ",
        "K_WRITE": "KV_WRITE",
        "V_WRITE": "KV_WRITE",
        "KV_WRITE": "KV_WRITE",
        "QK": "QK",
        "SCORE": "QK",
        "SOFTMAX": "SOFTMAX",
        "SV": "SV",
        "WO": "O_PROJ",
        "O": "O_PROJ",
        "O_PROJ": "O_PROJ",
        "FFN_W1": "FFN_UP",
        "FFN_W3": "FFN_UP",
        "FFN_UP": "FFN_UP",
        "FFN_GATE": "FFN_UP",
        "SWIGLU": "ACTIVATION",
        "SILU": "ACTIVATION",
        "GELU": "ACTIVATION",
        "ACTIVATION": "ACTIVATION",
        "FFN_W2": "FFN_DOWN",
        "FFN_DOWN": "FFN_DOWN",
        "MOE_ROUTER": "ROUTER",
        "ROUTER": "ROUTER",
        "MOE_COMBINE": "COMBINE",
        "COMBINE": "COMBINE",
        "CANDIDATE_TRANSFER": "CANDIDATE_TRANSFER",
        "ALLREDUCE": "COLLECTIVE",
        "ALL_REDUCE": "COLLECTIVE",
        "REDUCE": "COLLECTIVE",
        "GATHER": "COLLECTIVE",
        "SCATTER": "COLLECTIVE",
        "TRANSFER": "COPY",
        "COPY": "COPY",
    }
    return "EXPERT" if "EXPERT" in token else roles.get(token, "OTHER")


def _block_type(attrs: Mapping[str, Any], role: str, layer: int | None) -> str:
    explicit = _token(attrs.get("block_type"))
    if explicit:
        return explicit
    if role in {"ROUTER", "EXPERT", "COMBINE"} or attrs.get("experts") is not None:
        return "MOE_BLOCK"
    if role == "CANDIDATE_TRANSFER" or attrs.get("sd_component") is not None:
        return "SD_COMPONENT"
    if attrs.get("attention_sparsity") is not None:
        return "SPECIAL_ATTENTION_BLOCK"
    return "DENSE_BLOCK" if layer is not None else "OTHER"


def _device_memory(cfg: Mapping[str, Any]) -> dict[str, int]:
    root = json.loads(Path(cfg["hardware_json"]).read_text(encoding="utf-8"))
    devices = root.get("hardware", root)["devices"]
    by_name = {device["name"]: device for device in devices}
    return {
        device_id: int(float(by_name[device_id]["mem_capacity_GB"]) * 1024**3)
        for device_id in POLICY_DEVICES
    }


def _snapshot_network(
    snapshot: Mapping[str, Any],
    *,
    graph_id: str,
    workload_id: str,
    device_memory_bytes: Mapping[str, int],
) -> dict[str, Any]:
    metadata = [operator["network_metadata"] for operator in snapshot["operators"]]
    batch = int(metadata[0]["batch"])
    mean_context = int(metadata[0]["seq_len"])
    layers = [
        item["node_attrs"]["layer"]
        for item in metadata
        if "layer" in item["node_attrs"]
    ]
    inferred_total_repeats = max(layers) + 1 if layers else 1
    operators: list[dict[str, Any]] = []
    for operator_index, (operator, item) in enumerate(
        zip(snapshot["operators"], metadata, strict=True)
    ):
        attrs = item["node_attrs"]
        layer = attrs.get("layer_index", attrs.get("layer"))
        role = _op_role(item["name"], attrs)
        repeat_index = int(attrs.get("repeat_index", layer if layer is not None else 0))
        total_repeats = int(attrs.get("total_repeats", inferred_total_repeats))
        block_id = attrs.get(
            "block_id", f"layer:{layer}" if layer is not None else "global"
        )
        operators.append(
            {
                "op_id": operator["op_id"],
                "dependencies": list(operator["dependencies"]),
                "op_role": role,
                "block_type": _block_type(attrs, role, layer),
                "block_id": str(block_id),
                "layer_index": int(layer) if layer is not None else None,
                "canonical_op_slot": str(attrs["canonical_op_slot"]),
                "operator_index": operator_index,
                "repeat_index": repeat_index,
                "total_repeats": total_repeats,
            }
        )

    phase = snapshot["phase"]
    return {
        "graph_id": graph_id,
        "workload_id": workload_id,
        "schedule_call_index": int(snapshot["schedule_call_index"]),
        "phase": phase,
        "policy_devices": list(POLICY_DEVICES),
        "workload": {
            "batch": batch,
            "sequence_length": mean_context,
            "past_kv_len": 0 if phase == "prefill" else mean_context,
            "query_len": mean_context if phase == "prefill" else 1,
            "scheduled_tokens": batch * (mean_context if phase == "prefill" else 1),
            "mean_context": float(mean_context),
        },
        "device_memory_bytes": dict(device_memory_bytes),
        "operators": operators,
    }


def build_network_manifest(
    *,
    cfg: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    prior_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    memory = _device_memory(cfg)
    return {
        "schema": NETWORK_SCHEMA,
        "schema_version": NETWORK_SCHEMA_VERSION,
        "networks": [
            _snapshot_network(
                snapshot,
                graph_id=prior_artifact["graph_id"],
                workload_id=prior_artifact["workload_id"],
                device_memory_bytes=memory,
            )
            for snapshot in snapshots
        ],
    }


def export_network_manifest(
    *,
    cfg: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    prior_artifact: Mapping[str, Any],
    output: str | Path,
) -> Path:
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            build_network_manifest(
                cfg=cfg,
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


__all__ = [
    "NETWORK_SCHEMA",
    "NETWORK_SCHEMA_VERSION",
    "POLICY_DEVICES",
    "build_network_manifest",
    "export_network_manifest",
]
