"""Versioned DOPS -> Het-Infer placement-prior artifact support.

The online runtime consumes this artifact as a *prior*.  Simulated start/finish
times remain diagnostic data and are deliberately not part of the execution
contract.  Candidate metrics may be unavailable when importing an old
``best_summary``; unavailable values are emitted as JSON ``null`` rather than
being guessed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SCHEMA_NAME = "dops.hetinfer_prior.v1"
SCHEMA_VERSION = 1
SCORE_FIELDS = (
    "dops_score_s",
    "eft_s",
    "window_s",
    "compute_s",
    "reload_s",
    "comm_s",
    "weight_reuse_bias_s",
    "decode_amort_bias_s",
)


class PriorValidationError(ValueError):
    """Raised when an artifact violates the v1 contract."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        return _jsonable(vars(value))
    except Exception:
        return str(value)


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(value)).hexdigest()


def digest_file(path: str | os.PathLike[str] | None) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float_or_none(value: Any, *, field: str) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception as exc:
        raise PriorValidationError(f"{field} must be numeric or null") from exc
    if not math.isfinite(out):
        raise PriorValidationError(f"{field} must be finite or null")
    return out


@dataclass(frozen=True)
class CandidatePrior:
    """Bifocal score terms for one legal device at one scheduling decision."""

    dops_score_s: Optional[float]
    eft_s: Optional[float]
    window_s: Optional[float]
    compute_s: Optional[float]
    reload_s: Optional[float]
    comm_s: Optional[float]
    weight_reuse_bias_s: Optional[float]
    decode_amort_bias_s: Optional[float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidatePrior":
        if not isinstance(value, Mapping):
            raise PriorValidationError("candidate entry must be an object")
        return cls(
            **{
                name: _float_or_none(value.get(name), field=name)
                for name in SCORE_FIELDS
            }
        )

    @classmethod
    def unavailable(cls) -> "CandidatePrior":
        return cls(**{name: None for name in SCORE_FIELDS})

    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


def _require_nonempty_str(value: Any, path: str) -> str:
    out = str(value or "").strip()
    if not out:
        raise PriorValidationError(f"{path} must be a non-empty string")
    return out


def _require_positive_int(value: Any, path: str, *, allow_zero: bool = False) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise PriorValidationError(f"{path} must be an integer") from exc
    if out < 0 or (out == 0 and not allow_zero):
        relation = "non-negative" if allow_zero else "positive"
        raise PriorValidationError(f"{path} must be {relation}")
    return out


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate cross-field invariants not expressible concisely in JSON Schema."""

    if not isinstance(artifact, Mapping):
        raise PriorValidationError("artifact must be an object")
    if artifact.get("schema") != SCHEMA_NAME:
        raise PriorValidationError(f"schema must equal {SCHEMA_NAME!r}")
    try:
        schema_version = int(artifact.get("schema_version", -1))
    except Exception as exc:
        raise PriorValidationError("schema_version must be an integer") from exc
    if schema_version != SCHEMA_VERSION:
        raise PriorValidationError(f"schema_version must equal {SCHEMA_VERSION}")
    _require_nonempty_str(artifact.get("artifact_id"), "artifact_id")
    _require_nonempty_str(artifact.get("created_at"), "created_at")

    semantics = artifact.get("semantics")
    if not isinstance(semantics, Mapping):
        raise PriorValidationError("semantics must be an object")
    expected_semantics = {
        "role": "offline_placement_prior",
        "timeline_is_runtime_contract": False,
        "online_device_selection_required": True,
        "score_units": "seconds",
    }
    for field, expected in expected_semantics.items():
        if semantics.get(field) != expected:
            raise PriorValidationError(
                f"semantics.{field} must equal {expected!r}"
            )

    provenance = artifact.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PriorValidationError("provenance must be an object")
    for section in ("model", "workload", "hardware", "graph", "producer"):
        if not isinstance(provenance.get(section), Mapping):
            raise PriorValidationError(f"provenance.{section} must be an object")
    for field in (
        "model_family",
        "model_revision",
        "graph_sha256",
        "hardware_sha256",
        "dops_revision",
        "source_artifact_sha256",
        "policy",
    ):
        _require_nonempty_str(provenance.get(field), f"provenance.{field}")
    for field in ("graph_sha256", "hardware_sha256", "source_artifact_sha256"):
        value = str(provenance.get(field, ""))
        if not re_full_sha256(value):
            raise PriorValidationError(f"provenance.{field} must be a lowercase 64-hex SHA256")
    if not isinstance(provenance.get("parallelism"), Mapping):
        raise PriorValidationError("provenance.parallelism must be an object")
    for axis in ("tp", "pp", "ep"):
        _require_positive_int(
            provenance["parallelism"].get(axis),
            f"provenance.parallelism.{axis}",
        )

    profiles = artifact.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise PriorValidationError("profiles must be a non-empty array")
    seen_profiles: set[str] = set()
    for pidx, profile in enumerate(profiles):
        if not isinstance(profile, Mapping):
            raise PriorValidationError(f"profiles[{pidx}] must be an object")
        profile_id = _require_nonempty_str(profile.get("profile_id"), f"profiles[{pidx}].profile_id")
        if profile_id in seen_profiles:
            raise PriorValidationError(f"duplicate profile_id {profile_id!r}")
        seen_profiles.add(profile_id)

        workload = profile.get("workload")
        if not isinstance(workload, Mapping):
            raise PriorValidationError(f"profiles[{pidx}].workload must be an object")
        _require_positive_int(workload.get("batch"), f"profiles[{pidx}].workload.batch")
        _require_positive_int(
            workload.get("context_len"),
            f"profiles[{pidx}].workload.context_len",
            allow_zero=True,
        )
        for field in ("prefill_len", "decode_len", "max_seq_len"):
            _require_positive_int(
                workload.get(field), f"profiles[{pidx}].workload.{field}", allow_zero=True
            )
        phases = profile.get("phases")
        if not isinstance(phases, Mapping) or not phases:
            raise PriorValidationError(f"profiles[{pidx}].phases must be a non-empty object")
        if set(str(k) for k in phases) != {"prefill", "decode"}:
            raise PriorValidationError(
                f"profiles[{pidx}].phases must contain exactly prefill and decode"
            )
        for phase, phase_data in phases.items():
            if str(phase) not in {"prefill", "decode"}:
                raise PriorValidationError(f"profiles[{pidx}].phases has invalid key {phase!r}")
            if not isinstance(phase_data, Mapping):
                raise PriorValidationError(f"profiles[{pidx}].phases.{phase} must be an object")
            operators = phase_data.get("operators")
            if not isinstance(operators, list) or not operators:
                raise PriorValidationError(
                    f"profiles[{pidx}].phases.{phase}.operators must be a non-empty array"
                )
            seen_ops: set[str] = set()
            for oidx, op in enumerate(operators):
                base = f"profiles[{pidx}].phases.{phase}.operators[{oidx}]"
                if not isinstance(op, Mapping):
                    raise PriorValidationError(f"{base} must be an object")
                node_id = _require_nonempty_str(op.get("node_id"), f"{base}.node_id")
                if node_id in seen_ops:
                    raise PriorValidationError(f"{base}.node_id duplicates {node_id!r}")
                seen_ops.add(node_id)
                baseline = _require_nonempty_str(op.get("baseline_device"), f"{base}.baseline_device")
                legal = op.get("legal_devices")
                if not isinstance(legal, list) or not legal:
                    raise PriorValidationError(f"{base}.legal_devices must be a non-empty array")
                legal_names = [_require_nonempty_str(v, f"{base}.legal_devices") for v in legal]
                if len(set(legal_names)) != len(legal_names):
                    raise PriorValidationError(f"{base}.legal_devices contains duplicates")
                if baseline not in legal_names:
                    raise PriorValidationError(f"{base}.baseline_device is not legal")
                candidates = op.get("candidates")
                if not isinstance(candidates, Mapping):
                    raise PriorValidationError(f"{base}.candidates must be an object")
                if set(str(k) for k in candidates) != set(legal_names):
                    raise PriorValidationError(f"{base}.candidates keys must equal legal_devices")
                parsed_candidates: Dict[str, CandidatePrior] = {}
                for dev_name, score in candidates.items():
                    parsed = CandidatePrior.from_mapping(score)
                    parsed_candidates[str(dev_name)] = parsed
                    for field in ("eft_s", "window_s", "compute_s", "reload_s", "comm_s"):
                        number = getattr(parsed, field)
                        if number is not None and number < 0.0:
                            raise PriorValidationError(
                                f"{base}.candidates.{dev_name}.{field} must be non-negative"
                            )

                if not isinstance(op.get("dynamic_eligible"), bool):
                    raise PriorValidationError(f"{base}.dynamic_eligible must be boolean")
                if bool(op.get("dynamic_eligible")) and len(legal_names) < 2:
                    raise PriorValidationError(
                        f"{base}.dynamic_eligible requires at least two legal devices"
                    )
                if bool(op.get("dynamic_eligible")) and any(
                    candidate.dops_score_s is None
                    for candidate in parsed_candidates.values()
                ):
                    raise PriorValidationError(
                        f"{base}.dynamic_eligible requires a DOPS score for every legal device"
                    )
                op_phase = _require_nonempty_str(op.get("phase"), f"{base}.phase")
                if op_phase != str(phase):
                    raise PriorValidationError(f"{base}.phase must match enclosing phase")
                weight = op.get("weight")
                if not isinstance(weight, Mapping):
                    raise PriorValidationError(f"{base}.weight must be an object")
                if not isinstance(op.get("constraints"), Mapping):
                    raise PriorValidationError(f"{base}.constraints must be an object")
                _require_positive_int(
                    weight.get("size_bytes", 0), f"{base}.weight.size_bytes", allow_zero=True
                )
                _require_nonempty_str(weight.get("storage_layout"), f"{base}.weight.storage_layout")


def _git_revision(repo_root: str | os.PathLike[str] | None = None) -> str:
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _shape_snapshot(shape: Any) -> Dict[str, Any]:
    if shape is None:
        return {}
    fields = (
        "layer_num",
        "dim",
        "ffn_dim",
        "n_heads",
        "n_kv_heads",
        "experts_per_layer",
        "experts_top_k",
        "active_experts_per_layer",
        "max_seq_len",
    )
    return {name: _jsonable(getattr(shape, name)) for name in fields if hasattr(shape, name)}


def _graph_snapshot(graph: Any) -> Dict[str, Any]:
    if graph is None:
        empty = {"nodes": [], "edges": []}
        return {
            **empty,
            "node_count": 0,
            "edge_count": 0,
            "digest_algorithm": "sha256",
            "digest": digest_json(empty),
        }
    nodes: list[Dict[str, Any]] = []
    edges: list[list[str]] = []
    for nid, node in getattr(graph, "nodes", {}).items():
        node_id = str(getattr(node, "id", nid))
        preds = sorted(str(x) for x in (graph.predecessors(str(nid)) or []))
        succs = sorted(str(x) for x in (graph.successors(str(nid)) or []))
        nodes.append(
            {
                "node_id": node_id,
                "op_type": str(getattr(node, "name", "") or ""),
                "predecessors": preds,
                "successors": succs,
                "flops": float(getattr(node, "flops", 0.0) or 0.0),
                "bytes_read": float(getattr(node, "bytes_read", 0.0) or 0.0),
                "bytes_write": float(getattr(node, "bytes_write", 0.0) or 0.0),
                "weight_id": (
                    None if getattr(node, "weight_id", None) in (None, "") else str(node.weight_id)
                ),
                "weight_size_bytes": int(getattr(node, "weight_size", 0) or 0),
                "allowed_device_types": _jsonable(getattr(node, "allowed", {}) or {}),
                "attrs": _jsonable(getattr(node, "attrs", {}) or {}),
            }
        )
        edges.extend([[node_id, str(v)] for v in succs])
    nodes.sort(key=lambda item: item["node_id"])
    edges.sort()
    core = {"nodes": nodes, "edges": edges}
    return {
        **core,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "digest_algorithm": "sha256",
        "digest": digest_json(core),
    }


def _hardware_snapshot(cluster: Any, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    devices: list[Dict[str, Any]] = []
    links: list[Dict[str, Any]] = []
    if cluster is not None:
        for name, dev in sorted(getattr(cluster, "devices", {}).items()):
            device_snapshot = _jsonable(vars(dev))
            if not isinstance(device_snapshot, dict):
                device_snapshot = {}
            device_snapshot["name"] = str(name)
            device_snapshot["type"] = str(getattr(dev, "type", "") or "")
            devices.append(device_snapshot)
        seen: set[Tuple[str, str]] = set()
        for (src, dst), spec in sorted(getattr(cluster, "link_specs", {}).items()):
            key = tuple(sorted((str(src), str(dst))))
            if key in seen:
                continue
            seen.add(key)
            links.append(
                {
                    "a": key[0],
                    "b": key[1],
                    "bw_GBs": float(getattr(spec, "bw_GBs", 0.0) or 0.0),
                    "latency_s": float(getattr(spec, "latency_s", 0.0) or 0.0),
                    "overhead_s": float(getattr(spec, "overhead_s", 0.0) or 0.0),
                    "flit_size_B": int(getattr(spec, "flit_size_B", 0) or 0),
                    "max_payload_B": int(getattr(spec, "max_payload_B", 0) or 0),
                }
            )

    raw_path = cfg.get("hardware_json")
    raw_digest = digest_file(str(raw_path)) if raw_path else None
    raw_snapshot = None
    if raw_path:
        try:
            raw_snapshot = json.loads(Path(str(raw_path)).read_text(encoding="utf-8"))
        except Exception:
            raw_snapshot = None
    default_link = None
    if cluster is not None and getattr(cluster, "default_link_spec", None) is not None:
        default_link = _jsonable(vars(cluster.default_link_spec))
    normalized = {
        "topology": str(getattr(cluster, "topology", cfg.get("topology", "")) or ""),
        "devices": devices,
        "links": links,
        "default_link": default_link,
        "pim_memory": _jsonable(getattr(cluster, "pim_memory", {}) or {}) if cluster is not None else {},
    }
    return {
        **normalized,
        "source_path": str(raw_path) if raw_path else None,
        "source_digest": raw_digest,
        "source_snapshot": _jsonable(raw_snapshot),
        "digest_algorithm": "sha256",
        "digest": digest_json(normalized),
    }


def build_provenance(
    *,
    cfg: Mapping[str, Any],
    graph: Any = None,
    cluster: Any = None,
    shape: Any = None,
    producer_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an explicit, self-contained provenance snapshot."""

    cfg_snapshot = _jsonable(dict(cfg))
    graph_info = _graph_snapshot(graph)
    hardware_info = _hardware_snapshot(cluster, cfg)
    tp_effective = int(
        cfg.get(
            "tp",
            max(
                int(cfg.get("tp_qkv_effective", cfg.get("tp_qkv", 1)) or 1),
                int(cfg.get("tp_ffn_effective", cfg.get("tp_ffn", 1)) or 1),
                int(cfg.get("tp_moe_effective", cfg.get("tp_moe", 1)) or 1),
            ),
        )
        or 1
    )
    pp_effective = int(cfg.get("pp", cfg.get("pipeline_parallel_size", 1)) or 1)
    ep_effective = int(cfg.get("ep", cfg.get("expert_parallel_size", 1)) or 1)
    shape_info = _shape_snapshot(shape)
    explicit_model_revision = cfg.get(
        "model_revision", cfg.get("model_commit", cfg.get("revision"))
    )
    model_revision = (
        str(explicit_model_revision)
        if explicit_model_revision not in (None, "")
        else ("shape-sha256:" + digest_json(shape_info) if shape_info else "unknown")
    )
    model = {
        "family": cfg.get("model_family", cfg.get("model_type")),
        "variant": cfg.get("model_variant"),
        "revision": model_revision,
        "revision_kind": (
            "explicit" if explicit_model_revision not in (None, "") else "normalized_shape"
        ),
        "shape": shape_info,
        "parallelism": {
            "tp": tp_effective,
            "tp_qkv": cfg.get("tp_qkv"),
            "tp_qkv_effective": cfg.get("tp_qkv_effective"),
            "tp_ffn": cfg.get("tp_ffn"),
            "tp_ffn_effective": cfg.get("tp_ffn_effective"),
            "tp_moe": cfg.get("tp_moe"),
            "tp_moe_effective": cfg.get("tp_moe_effective"),
            "pp": pp_effective,
            "ep": ep_effective,
        },
    }
    workload = {
        "batch": int(cfg.get("batch", 1) or 1),
        "max_batch_size": int(cfg.get("max_batch_size", cfg.get("batch", 1)) or 1),
        "prefill_len": int(cfg.get("prefill_len", 0) or 0),
        "decode_len": int(cfg.get("decode_len", 0) or 0),
        "max_seq_len": int(
            cfg.get(
                "max_seq_len",
                int(cfg.get("prefill_len", 0) or 0) + int(cfg.get("decode_len", 0) or 0),
            )
            or 0
        ),
        "dtype": cfg.get("dtype"),
        "decode_plan_refresh_stride": cfg.get("decode_plan_refresh_stride"),
        "decode_sample_stride": cfg.get("decode_sample_stride"),
    }
    missing: list[str] = []
    for path, value in (
        ("model.family", model["family"]),
        ("model.variant", model["variant"]),
        ("workload.dtype", workload["dtype"]),
    ):
        if value in (None, ""):
            missing.append(path)
    if not hardware_info["devices"]:
        missing.append("hardware.devices")
    if graph_info["node_count"] <= 0:
        missing.append("graph.nodes")
    revision = str(producer_revision or _git_revision())
    return {
        "status": "complete" if not missing else "partial",
        "missing_fields": missing,
        "model_family": str(model["family"] or "unknown"),
        "model_revision": model_revision,
        "graph_sha256": str(graph_info["digest"]),
        "hardware_sha256": str(hardware_info["digest"]),
        "dops_revision": revision,
        "source_artifact_sha256": str(cfg.get("source_artifact_sha256") or "pending"),
        "policy": str(cfg.get("algo", "Bifocal") or "Bifocal"),
        "parallelism": dict(model["parallelism"]),
        "model": model,
        "workload": workload,
        "hardware": hardware_info,
        "graph": graph_info,
        "config": {
            "snapshot": cfg_snapshot,
            "digest_algorithm": "sha256",
            "digest": digest_json(cfg_snapshot),
        },
        "producer": {
            "name": "DOPS",
            "repository": "YIAI-02/DOPS",
            "revision": revision,
            "scheduler": str(cfg.get("algo", "Bifocal")),
        },
    }


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _profile_id(workload: Mapping[str, Any], phase: str, ordinal: int) -> str:
    batch = int(workload.get("batch", 1) or 1)
    context = int(workload.get("context_len", 0) or 0)
    token = workload.get("token_idx")
    token_part = "" if token is None else f"-t{int(token)}"
    short = digest_json({"workload": workload, "phase": phase})[:10]
    return f"{phase}-b{batch}-c{context}{token_part}-p{ordinal}-{short}"


def _normalize_candidate_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    node_id = _require_nonempty_str(record.get("node_id"), "candidate_record.node_id")
    baseline = _require_nonempty_str(record.get("baseline_device"), "candidate_record.baseline_device")
    legal_raw = record.get("legal_devices")
    if not isinstance(legal_raw, Sequence) or isinstance(legal_raw, (str, bytes)) or not legal_raw:
        raise PriorValidationError(f"candidate record {node_id!r} has no legal_devices")
    legal = [
        _require_nonempty_str(x, f"candidate record {node_id!r} legal_devices")
        for x in legal_raw
    ]
    candidates_raw = record.get("candidates")
    if not isinstance(candidates_raw, Mapping):
        raise PriorValidationError(f"candidate record {node_id!r} has no candidates object")
    candidates_by_name: Dict[str, Any] = {}
    for key, value in candidates_raw.items():
        name = _require_nonempty_str(key, f"candidate record {node_id!r} candidate key")
        if name in candidates_by_name:
            raise PriorValidationError(
                f"candidate record {node_id!r} has duplicate normalized candidate {name!r}"
            )
        candidates_by_name[name] = value
    if set(candidates_by_name) != set(legal):
        raise PriorValidationError(
            f"candidate record {node_id!r} candidate keys must equal legal_devices"
        )
    candidates: Dict[str, Dict[str, Optional[float]]] = {}
    for dev in legal:
        raw = candidates_by_name[dev]
        candidates[dev] = CandidatePrior.from_mapping(raw).to_dict()
    weight_raw = record.get("weight") or {}
    if not isinstance(weight_raw, Mapping):
        raise PriorValidationError(f"candidate record {node_id!r} weight must be an object")
    constraints_raw = record.get("constraints") or {}
    if not isinstance(constraints_raw, Mapping):
        raise PriorValidationError(
            f"candidate record {node_id!r} constraints must be an object"
        )
    if "dynamic_eligible" in record and not isinstance(record.get("dynamic_eligible"), bool):
        raise PriorValidationError(
            f"candidate record {node_id!r} dynamic_eligible must be boolean"
        )
    return {
        "node_id": node_id,
        "op_type": str(record.get("op_type", "") or ""),
        "phase": str(record.get("phase", "") or ""),
        "baseline_device": baseline,
        "legal_devices": legal,
        "candidates": candidates,
        "constraints": _jsonable(constraints_raw),
        "dynamic_eligible": bool(
            record.get(
                "dynamic_eligible",
                len(legal) > 1 and str(record.get("op_type", "") or "").upper() not in {
                    "ALLREDUCE",
                    "REDUCE",
                    "GATHER",
                    "SCATTER",
                    "TRANSFER",
                },
            )
        ),
        "weight": {
            "weight_id": (
                None
                if weight_raw.get("weight_id") in (None, "")
                else str(weight_raw.get("weight_id"))
            ),
            "size_bytes": _require_positive_int(
                weight_raw.get("size_bytes", 0),
                f"candidate record {node_id!r} weight.size_bytes",
                allow_zero=True,
            ),
            "storage_layout": str(
                weight_raw.get("storage_layout", "ND") or "ND"
            ),
        },
    }


def profiles_from_candidate_records(
    records: Iterable[Mapping[str, Any]], *, cfg: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    """Group Bifocal decisions into profile-selector points.

    Each decode-context capture is paired with the prefill capture from the
    same offline workload.  Consequently every canonical profile contains
    both ``phases.prefill`` and ``phases.decode``.  Decode fixed-plan replay
    does not create candidate records and therefore cannot be mistaken for a
    newly evaluated DOPS prior.
    """

    grouped: Dict[Tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for rec in records:
        phase = str(rec.get("phase", "") or "")
        key = (
            int(rec.get("schedule_call_index", 0) or 0),
            phase,
            int(rec.get("batch", cfg.get("batch", 1)) or 1),
            int(rec.get("seq_len", cfg.get("prefill_len", 0)) or 0),
            rec.get("token_idx"),
        )
        grouped.setdefault(key, []).append(rec)

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            int(item[0][0]),
            0 if item[0][1] == "prefill" else 1,
            int(item[0][2]),
            int(item[0][3]),
            -1 if item[0][4] is None else int(item[0][4]),
        ),
    )
    prefill_groups = [(key, group) for key, group in ordered_groups if key[1] == "prefill"]
    decode_groups = [(key, group) for key, group in ordered_groups if key[1] == "decode"]
    if not prefill_groups or not decode_groups:
        raise PriorValidationError(
            "canonical workload profiles require both prefill and decode candidate captures"
        )

    profiles: list[Dict[str, Any]] = []
    for ordinal, (decode_key, decode_group) in enumerate(decode_groups):
        decode_call, _, batch, decode_context, token_idx = decode_key
        # A scheduler can be reused for more than one capture session. Pair a
        # decode decision with the latest preceding prefill of the same batch,
        # rather than silently attaching every decode group to the first
        # prefill ever observed.
        matching_prefill = [
            item
            for item in prefill_groups
            if int(item[0][2]) == int(batch) and int(item[0][0]) < int(decode_call)
        ]
        if not matching_prefill:
            raise PriorValidationError(
                f"decode capture call={decode_call} batch={batch} has no preceding matching prefill capture"
            )
        prefill_key, prefill_group = max(
            matching_prefill,
            key=lambda item: int(item[0][0]),
        )
        prefill_call, _, _, prefill_context, _ = prefill_key
        phase_operators = {
            "prefill": [_normalize_candidate_record(rec) for rec in prefill_group],
            "decode": [_normalize_candidate_record(rec) for rec in decode_group],
        }
        available = sorted(
            {
                field
                for operators in phase_operators.values()
                for op in operators
                for score in op["candidates"].values()
                for field in SCORE_FIELDS
                if score.get(field) is not None
            }
        )
        workload = {
            "batch": int(batch),
            "context_len": int(decode_context),
            "token_idx": token_idx,
            "dtype": cfg.get("dtype"),
            "prefill_len": int(cfg.get("prefill_len", prefill_context) or prefill_context),
            "decode_len": int(cfg.get("decode_len", 0) or 0),
            "max_seq_len": int(
                cfg.get(
                    "max_seq_len",
                    int(cfg.get("prefill_len", prefill_context) or prefill_context)
                    + int(cfg.get("decode_len", 0) or 0),
                )
                or 0
            ),
        }
        all_operators = phase_operators["prefill"] + phase_operators["decode"]
        profiles.append(
            {
                "profile_id": _profile_id(workload, "prefill-decode", ordinal),
                "workload": workload,
                "source": {
                    "kind": "bifocal_candidate_capture",
                    "schedule_call_indices": {
                        "prefill": int(prefill_call),
                        "decode": int(decode_call),
                    },
                    "score_formula": "(1-gamma)*eft_s + gamma*window_s + weight_reuse_bias_s + decode_amort_bias_s",
                    "gamma": decode_group[0].get("gamma"),
                    "available_metrics": available,
                    "candidate_scores_complete": all(
                        all(score.get("dops_score_s") is not None for score in op["candidates"].values())
                        for op in all_operators
                    ),
                },
                "phases": {
                    "prefill": {
                        "context_len": int(prefill_context),
                        "token_idx": None,
                        "operators": phase_operators["prefill"],
                    },
                    "decode": {
                        "context_len": int(decode_context),
                        "token_idx": token_idx,
                        "operators": phase_operators["decode"],
                    },
                },
            }
        )
    return profiles


def _legacy_schedule_profiles(summary: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Convert legacy timeline placement into explicitly unscored priors."""

    prefill_schedule: Optional[Sequence[Mapping[str, Any]]] = None
    prefill = summary.get("prefill_schedule")
    if isinstance(prefill, list) and prefill:
        prefill_schedule = prefill
    decode_groups: list[Tuple[int, int, Sequence[Mapping[str, Any]]]] = []
    decode_steps = summary.get("decode_steps")
    if isinstance(decode_steps, list):
        for step in decode_steps:
            if not isinstance(step, Mapping):
                continue
            schedule = step.get("schedule")
            if not isinstance(schedule, list) or not schedule:
                continue
            token = int(step.get("t", 0) or 0)
            context = int(step.get("seq_len", int(cfg.get("prefill_len", 0) or 0) + token) or 0)
            decode_groups.append((token, context, schedule))

    if prefill_schedule is None or not decode_groups:
        raise PriorValidationError(
            "legacy best_summary needs both prefill_schedule and at least one materialized decode schedule"
        )

    def _legacy_operators(
        schedule: Sequence[Mapping[str, Any]], phase: str
    ) -> list[Dict[str, Any]]:
        operators: list[Dict[str, Any]] = []
        for item in schedule:
            if not isinstance(item, Mapping):
                continue
            node_id = str(item.get("node_id", "") or "")
            baseline = str(item.get("device", "") or "")
            if not node_id or not baseline:
                continue
            operators.append(
                {
                    "node_id": node_id,
                    "op_type": str(item.get("op_type", "") or ""),
                    "phase": phase,
                    "baseline_device": baseline,
                    "legal_devices": [baseline],
                    "candidates": {baseline: CandidatePrior.unavailable().to_dict()},
                    "constraints": {"source": "legacy_schedule_only"},
                    "dynamic_eligible": False,
                    "weight": {
                        "weight_id": None,
                        "size_bytes": 0,
                        "storage_layout": "UNKNOWN",
                    },
                }
            )
        return operators

    profiles: list[Dict[str, Any]] = []
    prefill_context = int(cfg.get("prefill_len", 0) or 0)
    prefill_operators = _legacy_operators(prefill_schedule, "prefill")
    for ordinal, (token_idx, context_len, schedule) in enumerate(decode_groups):
        decode_operators = _legacy_operators(schedule, "decode")
        workload = {
            "batch": int(cfg.get("batch", 1) or 1),
            "context_len": context_len,
            "token_idx": token_idx,
            "dtype": cfg.get("dtype"),
            "prefill_len": int(cfg.get("prefill_len", 0) or 0),
            "decode_len": int(cfg.get("decode_len", 0) or 0),
            "max_seq_len": int(
                cfg.get(
                    "max_seq_len",
                    int(cfg.get("prefill_len", 0) or 0) + int(cfg.get("decode_len", 0) or 0),
                )
                or 0
            ),
        }
        if prefill_operators and decode_operators:
            profiles.append(
                {
                    "profile_id": _profile_id(workload, "prefill-decode", ordinal),
                    "workload": workload,
                    "source": {
                        "kind": "legacy_best_summary",
                        "score_formula": None,
                        "gamma": None,
                        "available_metrics": [],
                        "candidate_scores_complete": False,
                        "warning": "Only the historical baseline placement was available; no alternative score was fabricated.",
                    },
                    "phases": {
                        "prefill": {
                            "context_len": prefill_context,
                            "token_idx": None,
                            "operators": prefill_operators,
                        },
                        "decode": {
                            "context_len": int(context_len),
                            "token_idx": token_idx,
                            "operators": decode_operators,
                        }
                    },
                }
            )
    return profiles


def build_artifact(
    *,
    cfg: Mapping[str, Any],
    graph: Any = None,
    cluster: Any = None,
    shape: Any = None,
    candidate_records: Optional[Iterable[Mapping[str, Any]]] = None,
    legacy_best_summary: Optional[Mapping[str, Any]] = None,
    producer_revision: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    provenance = build_provenance(
        cfg=cfg,
        graph=graph,
        cluster=cluster,
        shape=shape,
        producer_revision=producer_revision,
    )
    records = list(candidate_records or [])
    provenance["source_artifact_sha256"] = digest_json(
        records if records else dict(legacy_best_summary or {})
    )
    if records:
        profiles = profiles_from_candidate_records(records, cfg=cfg)
    elif legacy_best_summary is not None:
        profiles = _legacy_schedule_profiles(legacy_best_summary, cfg)
    else:
        raise PriorValidationError("candidate_records or legacy_best_summary is required")
    if not profiles:
        raise PriorValidationError("no executable profiles could be produced")

    identity_core = {
        "config_digest": provenance["config"]["digest"],
        "graph_digest": provenance["graph"]["digest"],
        "hardware_digest": provenance["hardware"]["digest"],
        "profiles": profiles,
    }
    artifact = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "artifact_id": "dops-prior-" + digest_json(identity_core)[:20],
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "semantics": {
            "role": "offline_placement_prior",
            "timeline_is_runtime_contract": False,
            "online_device_selection_required": True,
            "score_units": "seconds",
        },
        "provenance": provenance,
        "profiles": profiles,
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(artifact: Mapping[str, Any], output: str | os.PathLike[str]) -> Path:
    validate_artifact(artifact)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(artifact), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PriorValidationError(f"{path} must contain a JSON object")
    return value


__all__ = [
    "CandidatePrior",
    "PriorValidationError",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SCORE_FIELDS",
    "build_artifact",
    "build_provenance",
    "digest_file",
    "digest_json",
    "load_json",
    "profiles_from_candidate_records",
    "validate_artifact",
    "write_artifact",
]
