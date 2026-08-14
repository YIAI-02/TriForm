from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = (
    TEST_DIR
    if (TEST_DIR / "hetinfer_prior_export.py").is_file()
    else REPO_ROOT / "src"
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hetinfer_prior_export import (  # noqa: E402
    ATLAS_TIMING_REQUEST_SCHEMA,
    PriorExportError,
    atlas_duration_s,
    build_atlas_timing_request,
    build_prior_artifact,
    export_atlas_timing_request,
    export_prior_artifact,
    snapshots_require_atlas,
)


CFG = {
    "hetinfer_graph_id": "fixture-fork-join",
    "hetinfer_workload_id": "fixture-b1-s8",
}


def _operator_descriptor(
    op_kind: str,
    legal_devices: list[str],
    *,
    collective_primitive: str | None = None,
    collective_participants: list[str] | None = None,
) -> dict:
    return {
        "op_kind": op_kind,
        "phase": "prefill",
        "batch": 1,
        "seq_len": 8,
        "attrs": {
            "hidden_dim": 4096,
            "collective_context": None,
            "collective_input_bindings": [],
        },
        "weight_layout_by_device": {
            device_id: ("PIM_BLOCKED" if device_id.startswith("PIM") else "ND")
            for device_id in legal_devices
        },
        "collective_primitive": collective_primitive,
        "collective_participants": collective_participants or [],
        "topology": "fc",
    }


def _route(
    tensor: str,
    source: str,
    destination: str,
    bytes_: int,
    layout: str,
    duration_s: float | None,
    requires_atlas: bool,
) -> dict:
    return {
        "tensor_id": tensor,
        "source_device_id": source,
        "destination_device_id": destination,
        "bytes": bytes_,
        "layout": layout,
        "duration_s": duration_s,
        "requires_atlas": requires_atlas,
        "atlas_descriptor": {
            "topology": "fc",
            "source_device_type": (
                "pim" if source.startswith("PIM") else "npu"
            ),
            "destination_device_type": (
                "pim" if destination.startswith("PIM") else "npu"
            ),
        },
    }


def _sync_collective_descriptor(snapshot: dict) -> None:
    context = copy.deepcopy(snapshot["collective_contexts"][0])
    bindings = sorted(
        [
            copy.deepcopy(entry)
            for entry in snapshot["inputs"]
            if entry["consumer_op_id"] == context["op_id"]
            and entry["semantics"] == "collective_staging"
        ],
        key=lambda entry: (entry["producer_op_id"] or "", entry["tensor_id"]),
    )
    operator = next(
        entry for entry in snapshot["operators"] if entry["op_id"] == context["op_id"]
    )
    attrs = operator["atlas_descriptor"]["attrs"]
    attrs["collective_context"] = context
    attrs["collective_input_bindings"] = bindings


def _snapshot() -> dict:
    snapshot = {
        "schedule_call_index": 1,
        "phase": "prefill",
        "devices": [
            {"device_id": "NPU0", "device_type": "npu"},
            {"device_id": "NPU2", "device_type": "npu"},
            {"device_id": "PIM0", "device_type": "pim"},
        ],
        "operators": [
            {
                "op_id": "source",
                "dependencies": [],
                "legal_devices": ["NPU0"],
                "expert_device": "NPU0",
                "service_s": {"NPU0": 0.001},
                "atlas_descriptor": _operator_descriptor("INPUT", ["NPU0"]),
            },
            {
                "op_id": "left",
                "dependencies": ["source"],
                "legal_devices": ["NPU0", "PIM0"],
                "expert_device": "NPU0",
                "service_s": {"NPU0": 0.002, "PIM0": None},
                "atlas_descriptor": _operator_descriptor(
                    "GEMM", ["NPU0", "PIM0"]
                ),
            },
            {
                "op_id": "right",
                "dependencies": ["source"],
                "legal_devices": ["NPU0", "PIM0"],
                "expert_device": "PIM0",
                "service_s": {"NPU0": 0.004, "PIM0": None},
                "atlas_descriptor": _operator_descriptor(
                    "GEMM", ["NPU0", "PIM0"]
                ),
            },
            {
                "op_id": "join",
                "dependencies": ["left", "right"],
                "legal_devices": ["NPU0"],
                "expert_device": "NPU0",
                "service_s": {"NPU0": None},
                "atlas_descriptor": _operator_descriptor(
                    "ALLREDUCE",
                    ["NPU0"],
                    collective_primitive="ALLREDUCE",
                    collective_participants=["NPU0", "PIM0"],
                ),
            },
        ],
        "inputs": [
            {
                "consumer_op_id": "source",
                "producer_op_id": None,
                "tensor_id": "request_input",
                "semantics": "data",
                "bytes": 4096,
                "source_residencies": [
                    {"device_id": "NPU0", "layout": "ND"}
                ],
                "destination_devices": ["NPU0"],
            },
            {
                "consumer_op_id": "left",
                "producer_op_id": "source",
                "tensor_id": "activation:source",
                "semantics": "data",
                "bytes": 32768,
                "source_residencies": [
                    {"device_id": "NPU0", "layout": "ND"}
                ],
                "destination_devices": ["NPU0", "PIM0"],
            },
            {
                "consumer_op_id": "right",
                "producer_op_id": "source",
                "tensor_id": "activation:source",
                "semantics": "data",
                "bytes": 32768,
                "source_residencies": [
                    {"device_id": "NPU0", "layout": "ND"}
                ],
                "destination_devices": ["NPU0", "PIM0"],
            },
            {
                "consumer_op_id": "join",
                "producer_op_id": "left",
                "tensor_id": "activation:left",
                "semantics": "collective_staging",
                "bytes": 65536,
                "source_residencies": [
                    {"device_id": "NPU0", "layout": "ND"},
                    {"device_id": "PIM0", "layout": "PIM_BLOCKED"},
                ],
                "destination_devices": ["NPU0"],
            },
            {
                "consumer_op_id": "join",
                "producer_op_id": "right",
                "tensor_id": "activation:right",
                "semantics": "collective_staging",
                "bytes": 131072,
                "source_residencies": [
                    {"device_id": "NPU0", "layout": "ND"},
                    {"device_id": "PIM0", "layout": "PIM_BLOCKED"},
                ],
                "destination_devices": ["PIM0"],
            },
        ],
        "collective_contexts": [
            {
                "op_id": "join",
                "primitive": "ALLREDUCE",
                "topology": "fc",
                "canonical_device_id": "NPU0",
                "participant_device_ids": ["NPU0", "PIM0"],
                "output_device_ids": ["NPU0", "PIM0"],
                "resource_device_ids": ["NPU0", "PIM0"],
                "tensor_bytes": 196608,
                "internal_transport": "included_in_t_service",
            }
        ],
        "routes": [
            _route("request_input", "NPU0", "NPU0", 4096, "ND", 0.0, False),
            _route("activation:source", "NPU0", "NPU0", 32768, "ND", 0.0, False),
            _route("activation:source", "NPU0", "PIM0", 32768, "ND", None, True),
            _route("activation:left", "NPU0", "NPU0", 65536, "ND", 0.0, False),
            _route("activation:left", "NPU0", "PIM0", 65536, "ND", None, True),
            _route("activation:left", "PIM0", "NPU0", 65536, "PIM_BLOCKED", None, True),
            _route("activation:left", "PIM0", "PIM0", 65536, "PIM_BLOCKED", 0.0, False),
            _route("activation:right", "NPU0", "NPU0", 131072, "ND", 0.0, False),
            _route("activation:right", "NPU0", "PIM0", 131072, "ND", None, True),
            _route("activation:right", "PIM0", "NPU0", 131072, "PIM_BLOCKED", None, True),
            _route("activation:right", "PIM0", "PIM0", 131072, "PIM_BLOCKED", 0.0, False),
        ],
    }
    _sync_collective_descriptor(snapshot)
    return snapshot


def _atlas() -> dict:
    def service(op_id: str, device_id: str, cycles: int) -> dict:
        return {
            "op_id": op_id,
            "device_id": device_id,
            "cycles": cycles,
            "frequency_MHz": 500,
        }

    def movement(
        tensor: str,
        source: str,
        destination: str,
        bytes_: int,
        layout: str,
        cycles: int,
    ) -> dict:
        return {
            "tensor_id": tensor,
            "source_device_id": source,
            "destination_device_id": destination,
            "bytes": bytes_,
            "layout": layout,
            "cycles": cycles,
            "frequency_MHz": 1000,
        }

    request = build_atlas_timing_request(cfg=CFG, snapshots=[_snapshot()])
    return {
        "schema": "dops.hetinfer_atlas_timings.v1",
        "schema_version": 1,
        "graph_id": request["graph_id"],
        "workload_id": request["workload_id"],
        "timing_context_sha256": request["timing_context_sha256"],
        "service": [
            service("left", "PIM0", 1_500_000),
            service("right", "PIM0", 2_500_000),
            service("join", "NPU0", 3_500_000),
        ],
        "movement": [
            movement("activation:source", "NPU0", "PIM0", 32768, "ND", 34_768),
            movement("activation:left", "NPU0", "PIM0", 65536, "ND", 67_536),
            movement("activation:left", "PIM0", "NPU0", 65536, "PIM_BLOCKED", 73_536),
            movement("activation:right", "NPU0", "PIM0", 131072, "ND", 133_072),
            movement("activation:right", "PIM0", "NPU0", 131072, "PIM_BLOCKED", 139_072),
        ],
    }


def _no_pim_snapshot() -> dict:
    snapshot = copy.deepcopy(_snapshot())
    for index, device in enumerate(snapshot["devices"]):
        if device["device_id"] == "PIM0":
            snapshot["devices"][index] = {
                "device_id": "NPU1",
                "device_type": "npu",
            }
    for operator_index, operator in enumerate(snapshot["operators"]):
        operator["legal_devices"] = [
            "NPU1" if device_id == "PIM0" else device_id
            for device_id in operator["legal_devices"]
        ]
        if operator["expert_device"] == "PIM0":
            operator["expert_device"] = "NPU1"
        if "PIM0" in operator["service_s"]:
            operator["service_s"]["NPU1"] = 0.010 + operator_index * 0.001
            del operator["service_s"]["PIM0"]
        layouts = operator["atlas_descriptor"]["weight_layout_by_device"]
        if "PIM0" in layouts:
            layouts["NPU1"] = "ND"
            del layouts["PIM0"]
        operator["atlas_descriptor"]["collective_participants"] = [
            "NPU1" if device_id == "PIM0" else device_id
            for device_id in operator["atlas_descriptor"][
                "collective_participants"
            ]
        ]
    # The fixture's collective is ATLAS-modeled on an NPU only to exercise the
    # collective descriptor.  The all-NPU variant supplies a local model for it.
    snapshot["operators"][-1]["service_s"]["NPU0"] = 0.006
    for entry in snapshot["inputs"]:
        entry["source_residencies"] = [
            {
                **residency,
                "device_id": (
                    "NPU1"
                    if residency["device_id"] == "PIM0"
                    else residency["device_id"]
                ),
                "layout": (
                    "ND"
                    if residency["device_id"] == "PIM0"
                    else residency["layout"]
                ),
            }
            for residency in entry["source_residencies"]
        ]
        entry["destination_devices"] = [
            "NPU1" if device_id == "PIM0" else device_id
            for device_id in entry["destination_devices"]
        ]
    for context in snapshot["collective_contexts"]:
        for key in (
            "participant_device_ids",
            "output_device_ids",
            "resource_device_ids",
        ):
            context[key] = [
                "NPU1" if device_id == "PIM0" else device_id
                for device_id in context[key]
            ]
    for route in snapshot["routes"]:
        source_was_pim = route["source_device_id"] == "PIM0"
        if source_was_pim:
            route["source_device_id"] = "NPU1"
            route["layout"] = "ND"
        if route["destination_device_id"] == "PIM0":
            route["destination_device_id"] = "NPU1"
        route["atlas_descriptor"]["source_device_type"] = "npu"
        route["atlas_descriptor"]["destination_device_type"] = "npu"
        route["requires_atlas"] = False
        route["duration_s"] = (
            0.0
            if route["source_device_id"] == route["destination_device_id"]
            else route["bytes"] / 1_000_000_000.0
        )
    _sync_collective_descriptor(snapshot)
    return snapshot


EXPECTED_SERVICE = {
    ("source", "NPU0"),
    ("left", "NPU0"),
    ("left", "PIM0"),
    ("right", "NPU0"),
    ("right", "PIM0"),
    ("join", "NPU0"),
}
EXPECTED_ROUTES = {
    ("request_input", "NPU0", "NPU0", 4096, "ND"),
    ("activation:source", "NPU0", "NPU0", 32768, "ND"),
    ("activation:source", "NPU0", "PIM0", 32768, "ND"),
    ("activation:left", "NPU0", "NPU0", 65536, "ND"),
    ("activation:left", "NPU0", "PIM0", 65536, "ND"),
    ("activation:left", "PIM0", "NPU0", 65536, "PIM_BLOCKED"),
    ("activation:left", "PIM0", "PIM0", 65536, "PIM_BLOCKED"),
    ("activation:right", "NPU0", "NPU0", 131072, "ND"),
    ("activation:right", "NPU0", "PIM0", 131072, "ND"),
    ("activation:right", "PIM0", "NPU0", 131072, "PIM_BLOCKED"),
    ("activation:right", "PIM0", "PIM0", 131072, "PIM_BLOCKED"),
}


class HetInferPriorExportTests(unittest.TestCase):
    def _build(self) -> dict:
        return build_prior_artifact(
            cfg=CFG,
            snapshots=[_snapshot()],
            atlas_timings=_atlas(),
        )

    def test_service_is_compute_only_complete_and_uses_atlas_formula(self) -> None:
        payload = self._build()
        service = {
            (entry["op_id"], entry["device_id"]): entry["duration_s"]
            for entry in payload["t_service"]
        }
        self.assertEqual(set(service), EXPECTED_SERVICE)
        self.assertEqual(service[("left", "NPU0")], 0.002)
        self.assertEqual(service[("right", "PIM0")], 0.005)
        self.assertEqual(service[("join", "NPU0")], 0.007)
        self.assertEqual(atlas_duration_s(2_500_000, 500), 0.005)

        polluted = _snapshot()
        polluted["operators"][1]["eft_s"] = 999.0
        with self.assertRaisesRegex(PriorExportError, "unexpected fields"):
            build_prior_artifact(
                cfg=CFG, snapshots=[polluted], atlas_timings=_atlas()
            )

    def test_routes_are_the_exact_closure_and_resident_is_zero(self) -> None:
        payload = self._build()
        move = {
            (
                entry["tensor_id"],
                entry["source_device_id"],
                entry["destination_device_id"],
                entry["bytes"],
                entry["layout"],
            ): entry["duration_s"]
            for entry in payload["t_move"]
        }
        self.assertEqual(set(move), EXPECTED_ROUTES)
        for key, duration in move.items():
            if key[1] == key[2]:
                self.assertEqual(duration, 0.0)
        self.assertAlmostEqual(
            move[("activation:left", "PIM0", "NPU0", 65536, "PIM_BLOCKED")],
            73_536 / (1000 * 1_000_000),
        )

    def test_atlas_coverage_is_exact(self) -> None:
        missing = _atlas()
        missing["service"].pop()
        with self.assertRaisesRegex(PriorExportError, "missing precomputed ATLAS"):
            build_prior_artifact(cfg=CFG, snapshots=[_snapshot()], atlas_timings=missing)

        extra = _atlas()
        extra["service"].append(
            {
                "op_id": "source",
                "device_id": "NPU0",
                "cycles": 1,
                "frequency_MHz": 500,
            }
        )
        with self.assertRaisesRegex(PriorExportError, "unexpected"):
            build_prior_artifact(cfg=CFG, snapshots=[_snapshot()], atlas_timings=extra)

        missing_movement = _atlas()
        missing_movement["movement"].pop()
        with self.assertRaisesRegex(PriorExportError, "missing precomputed ATLAS"):
            build_prior_artifact(
                cfg=CFG,
                snapshots=[_snapshot()],
                atlas_timings=missing_movement,
            )

        extra_movement = _atlas()
        extra_movement["movement"].append(
            {
                "tensor_id": "unexpected",
                "source_device_id": "NPU0",
                "destination_device_id": "PIM0",
                "bytes": 1,
                "layout": "ND",
                "cycles": 1,
                "frequency_MHz": 1000,
            }
        )
        with self.assertRaisesRegex(PriorExportError, "unexpected"):
            build_prior_artifact(
                cfg=CFG,
                snapshots=[_snapshot()],
                atlas_timings=extra_movement,
            )

    def test_prior_strictly_passes_input_and_collective_semantics(self) -> None:
        snapshot = _snapshot()
        payload = self._build()
        inputs = {
            (
                entry["consumer_op_id"],
                entry["producer_op_id"],
                entry["tensor_id"],
            ): entry
            for entry in payload["inputs"]
        }
        expected = {
            (
                entry["consumer_op_id"],
                entry["producer_op_id"],
                entry["tensor_id"],
            ): entry
            for entry in snapshot["inputs"]
        }
        self.assertEqual(inputs, expected)
        self.assertEqual(payload["collective_contexts"], snapshot["collective_contexts"])
        self.assertEqual(
            inputs[("join", "left", "activation:left")]["semantics"],
            "collective_staging",
        )
        self.assertEqual(
            payload["collective_contexts"][0]["internal_transport"],
            "included_in_t_service",
        )

        extra_field = _snapshot()
        extra_field["inputs"][0]["unversioned_hint"] = True
        with self.assertRaisesRegex(PriorExportError, "unexpected fields"):
            build_atlas_timing_request(cfg=CFG, snapshots=[extra_field])

    def test_timing_digest_binds_inputs_collective_and_descriptors(self) -> None:
        base = build_atlas_timing_request(cfg=CFG, snapshots=[_snapshot()])
        variants: list[dict] = []

        input_binding = _snapshot()
        input_binding["inputs"][0]["tensor_id"] = "request_input:v2"
        input_binding["routes"][0]["tensor_id"] = "request_input:v2"
        variants.append(input_binding)

        collective_outputs = _snapshot()
        collective_outputs["collective_contexts"][0]["output_device_ids"] = [
            "NPU0"
        ]
        _sync_collective_descriptor(collective_outputs)
        variants.append(collective_outputs)

        collective_resources = _snapshot()
        collective_resources["collective_contexts"][0][
            "resource_device_ids"
        ].append("NPU2")
        _sync_collective_descriptor(collective_resources)
        variants.append(collective_resources)

        staging = _snapshot()
        staging["inputs"][3]["destination_devices"] = ["PIM0"]
        staging["inputs"][4]["destination_devices"] = ["NPU0"]
        _sync_collective_descriptor(staging)
        variants.append(staging)

        attrs = _snapshot()
        attrs["operators"][1]["atlas_descriptor"]["attrs"]["hidden_dim"] = 8192
        variants.append(attrs)

        layout = _snapshot()
        layout["operators"][1]["atlas_descriptor"][
            "weight_layout_by_device"
        ]["PIM0"] = "PIM_TILED_V2"
        variants.append(layout)

        requests = [
            build_atlas_timing_request(cfg=CFG, snapshots=[snapshot])
            for snapshot in variants
        ]
        self.assertEqual(
            len({base["timing_context_sha256"], *[
                request["timing_context_sha256"] for request in requests
            ]}),
            len(requests) + 1,
        )
        # The declared friendly labels remain identical, but instantiated graph
        # identity cannot be reused after any snapshot/context mutation.
        for request in requests:
            self.assertNotEqual(request["graph_id"], base["graph_id"])

    def test_missing_input_route_or_collective_context_fails_closed(self) -> None:
        missing_route = _snapshot()
        missing_route["routes"] = [
            route
            for route in missing_route["routes"]
            if not (
                route["tensor_id"] == "activation:left"
                and route["source_device_id"] == "PIM0"
                and route["destination_device_id"] == "NPU0"
            )
        ]
        with self.assertRaisesRegex(ValueError, "missing legal movement routes"):
            build_atlas_timing_request(cfg=CFG, snapshots=[missing_route])

        missing_context = _snapshot()
        missing_context["collective_contexts"] = []
        with self.assertRaisesRegex(
            ValueError,
            "no collective_context",
        ):
            build_atlas_timing_request(cfg=CFG, snapshots=[missing_context])

    def test_atlas_timing_request_has_exact_marked_keys(self) -> None:
        request = build_atlas_timing_request(cfg=CFG, snapshots=[_snapshot()])
        self.assertEqual(
            set(request),
            {
                "schema",
                "schema_version",
                "graph_id",
                "workload_id",
                "timing_context_sha256",
                "timing_context",
                "service",
                "movement",
            },
        )
        self.assertEqual(request["schema"], ATLAS_TIMING_REQUEST_SCHEMA)
        self.assertRegex(request["graph_id"], r"^dops-graph:[0-9a-f]{64}$")
        self.assertRegex(
            request["workload_id"], r"^dops-workload:[0-9a-f]{64}$"
        )
        self.assertEqual(
            {(entry["op_id"], entry["device_id"]) for entry in request["service"]},
            {("left", "PIM0"), ("right", "PIM0"), ("join", "NPU0")},
        )
        self.assertEqual(
            {
                (
                    entry["tensor_id"],
                    entry["source_device_id"],
                    entry["destination_device_id"],
                    entry["bytes"],
                    entry["layout"],
                )
                for entry in request["movement"]
            },
            {
                key
                for key in EXPECTED_ROUTES
                if key[1] != key[2] and (key[1] == "PIM0" or key[2] == "PIM0")
            },
        )
        for entry in request["service"] + request["movement"]:
            self.assertNotIn("cycles", entry)
            self.assertNotIn("frequency_MHz", entry)
            self.assertIn("descriptor", entry)
        self.assertRegex(request["timing_context_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(snapshots_require_atlas([_snapshot()]))

    def test_timing_identity_must_match_snapshot_and_cfg(self) -> None:
        missing_identity = _atlas()
        del missing_identity["graph_id"]
        with self.assertRaisesRegex(PriorExportError, "missing fields"):
            build_prior_artifact(
                cfg=CFG,
                snapshots=[_snapshot()],
                atlas_timings=missing_identity,
            )

        wrong_graph = _atlas()
        wrong_graph["graph_id"] = "wrong-graph"
        with self.assertRaisesRegex(PriorExportError, "atlas.graph_id"):
            build_prior_artifact(
                cfg=CFG,
                snapshots=[_snapshot()],
                atlas_timings=wrong_graph,
            )

        wrong_workload = _atlas()
        wrong_workload["workload_id"] = "wrong-workload"
        with self.assertRaisesRegex(PriorExportError, "atlas.workload_id"):
            build_prior_artifact(
                cfg=CFG,
                snapshots=[_snapshot()],
                atlas_timings=wrong_workload,
            )

    def test_no_atlas_markers_need_no_timing_payload(self) -> None:
        snapshot = _no_pim_snapshot()
        request = build_atlas_timing_request(cfg=CFG, snapshots=[snapshot])
        self.assertEqual(request["service"], [])
        self.assertEqual(request["movement"], [])
        self.assertFalse(snapshots_require_atlas([snapshot]))
        artifact = build_prior_artifact(
            cfg=CFG,
            snapshots=[snapshot],
            atlas_timings=None,
        )
        self.assertEqual(len(artifact["t_service"]), len(EXPECTED_SERVICE))

    def test_ordinary_non_pim_null_service_is_rejected(self) -> None:
        snapshot = _no_pim_snapshot()
        snapshot["operators"][1]["service_s"]["NPU0"] = None
        with self.assertRaisesRegex(
            PriorExportError,
            "must be null exactly for PIM execution or a collective",
        ):
            build_atlas_timing_request(cfg=CFG, snapshots=[snapshot])

    def test_atlas_request_export_is_atomic_and_honors_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "atlas-request.json"
            export_atlas_timing_request(
                cfg=CFG,
                snapshots=[_snapshot()],
                output=output,
            )
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                export_atlas_timing_request(
                    cfg=CFG,
                    snapshots=[_snapshot()],
                    output=output,
                )
            self.assertEqual(output.read_bytes(), original)

            broken = _snapshot()
            broken["operators"][0]["expert_device"] = "PIM0"
            with self.assertRaisesRegex(PriorExportError, "expert device"):
                export_atlas_timing_request(
                    cfg=CFG,
                    snapshots=[broken],
                    output=output,
                    overwrite=True,
                )
            self.assertEqual(output.read_bytes(), original)

            new_output = Path(tmp) / "never-created" / "request.json"
            missing_route = _snapshot()
            missing_route["routes"] = missing_route["routes"][:-1]
            with self.assertRaises(ValueError):
                export_atlas_timing_request(
                    cfg=CFG,
                    snapshots=[missing_route],
                    output=new_output,
                )
            self.assertFalse(new_output.exists())
            self.assertFalse(new_output.parent.exists())

            changed_cfg = dict(CFG)
            changed_cfg["hetinfer_workload_id"] = "fixture-b2-s8"
            export_atlas_timing_request(
                cfg=changed_cfg,
                snapshots=[_snapshot()],
                output=output,
                overwrite=True,
            )
            self.assertEqual(
                json.loads(output.read_text())["graph_id"],
                json.loads(original)["graph_id"],
            )
            self.assertNotEqual(
                json.loads(output.read_text())["workload_id"],
                json.loads(original)["workload_id"],
            )
            self.assertEqual(
                json.loads(output.read_text())["timing_context_sha256"],
                json.loads(original)["timing_context_sha256"],
            )
            self.assertNotEqual(output.read_bytes(), original)

    def test_timing_context_change_rejects_old_timings_even_with_ids_overridden(self) -> None:
        snapshot = _snapshot()
        old_request = build_atlas_timing_request(cfg=CFG, snapshots=[snapshot])
        old_timings = _atlas()
        changed = copy.deepcopy(snapshot)
        changed["operators"][1]["atlas_descriptor"]["attrs"]["hidden_dim"] = 8192
        new_request = build_atlas_timing_request(cfg=CFG, snapshots=[changed])
        self.assertNotEqual(
            old_request["timing_context_sha256"],
            new_request["timing_context_sha256"],
        )
        self.assertNotEqual(old_request["graph_id"], new_request["graph_id"])
        old_timings["graph_id"] = new_request["graph_id"]
        old_timings["workload_id"] = new_request["workload_id"]
        # The caller-controlled labels are unchanged; the derived digest still
        # prevents reusing measurements for a different operator descriptor.
        with self.assertRaisesRegex(
            PriorExportError, "atlas.timing_context_sha256"
        ):
            build_prior_artifact(
                cfg=CFG,
                snapshots=[changed],
                atlas_timings=old_timings,
            )

    def test_hardware_file_bytes_and_timing_cfg_bind_the_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hardware = Path(tmp) / "hardware.json"
            hardware.write_text('{"links":[1]}', encoding="utf-8")
            cfg = {**CFG, "hardware_json": str(hardware), "batch": 1}
            first = build_atlas_timing_request(cfg=cfg, snapshots=[_snapshot()])
            hardware.write_text('{"links":[2]}', encoding="utf-8")
            second = build_atlas_timing_request(cfg=cfg, snapshots=[_snapshot()])
            third = build_atlas_timing_request(
                cfg={**cfg, "batch": 2}, snapshots=[_snapshot()]
            )
            self.assertNotEqual(
                first["timing_context_sha256"],
                second["timing_context_sha256"],
            )
            self.assertEqual(first["graph_id"], second["graph_id"])
            self.assertNotEqual(first["workload_id"], second["workload_id"])
            self.assertNotEqual(
                second["timing_context_sha256"],
                third["timing_context_sha256"],
            )
            self.assertEqual(second["graph_id"], third["graph_id"])
            self.assertNotEqual(second["workload_id"], third["workload_id"])
            [hashed] = second["timing_context"]["input_files"]
            self.assertEqual(hashed["config_key"], "hardware_json")
            self.assertRegex(hashed["sha256"], r"^[0-9a-f]{64}$")

    def test_export_fails_closed_without_partial_or_overwrite(self) -> None:
        broken = _snapshot()
        del broken["operators"][1]["service_s"]["PIM0"]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new" / "prior.json"
            with mock.patch("hetinfer_prior_export.write_prior_artifact") as writer:
                with self.assertRaisesRegex(PriorExportError, "service_s"):
                    export_prior_artifact(
                        cfg=CFG,
                        snapshots=[broken],
                        atlas_timings=_atlas(),
                        output=output,
                    )
                writer.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())

            output = Path(tmp) / "prior.json"
            sentinel = b"OLD-COMPLETE-ARTIFACT\n"
            output.write_bytes(sentinel)
            missing_atlas = _atlas()
            missing_atlas["movement"].pop()
            with self.assertRaisesRegex(PriorExportError, "missing precomputed ATLAS"):
                export_prior_artifact(
                    cfg=CFG,
                    snapshots=[_snapshot()],
                    atlas_timings=missing_atlas,
                    output=output,
                    overwrite=True,
                )
            self.assertEqual(output.read_bytes(), sentinel)

    def test_exporter_has_no_runtime_or_training_dependency(self) -> None:
        tree = ast.parse((SRC_ROOT / "hetinfer_prior_export.py").read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
        allowed = {
            "__future__",
            "hashlib",
            "json",
            "math",
            "os",
            "re",
            "tempfile",
            "collections.abc",
            "dataclasses",
            "pathlib",
            "typing",
            "hetinfer_prior",
        }
        self.assertLessEqual(imports, allowed)
        self.assertEqual(
            set(inspect.signature(build_prior_artifact).parameters),
            {"cfg", "snapshots", "atlas_timings"},
        )
        self.assertEqual(
            set(inspect.signature(build_atlas_timing_request).parameters),
            {"cfg", "snapshots"},
        )

    def test_inputs_are_not_mutated_and_canonical_output_is_stable(self) -> None:
        snapshot = _snapshot()
        atlas = _atlas()
        before_snapshot = copy.deepcopy(snapshot)
        before_atlas = copy.deepcopy(atlas)
        first = build_prior_artifact(cfg=CFG, snapshots=[snapshot], atlas_timings=atlas)
        second = build_prior_artifact(
            cfg=CFG,
            snapshots=[copy.deepcopy(snapshot)],
            atlas_timings=copy.deepcopy(atlas),
        )
        self.assertEqual(snapshot, before_snapshot)
        self.assertEqual(atlas, before_atlas)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
