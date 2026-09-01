from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest import mock
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "dops_hetinfer_prior_v1_minimal.json"
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from commands.export_hetinfer_camc_profile import main as command_main  # noqa: E402
from hetinfer_camc_profile_export import (  # noqa: E402
    CAMC_PROFILE_SCHEMA,
    build_camc_profile,
    export_camc_profile,
)


def _artifacts() -> tuple[dict, dict, dict, dict]:
    prior = json.loads(FIXTURE.read_text(encoding="utf-8"))
    order = ["source", "left", "right", "join", "finish"]
    network = {
        "schema": "dops.hetinfer_network.v1",
        "schema_version": 1,
        "networks": [
            {
                "graph_id": prior["graph_id"],
                "workload_id": prior["workload_id"],
                "phase": "prefill",
                "operators": [
                    {
                        "op_id": operator["op_id"],
                        "dependencies": operator["dependencies"],
                        "layer_index": 0,
                    }
                    for operator in prior["operators"]
                ],
            }
        ],
    }
    tensor_bindings = {
        "schema": "dops.hetinfer_tensor_bindings.v1",
        "schema_version": 1,
        "graph_id": prior["graph_id"],
        "workload_id": prior["workload_id"],
        "bindings": [
            {
                "network_index": 0,
                "tensor_id": "request_input",
                "layer_index": None,
                "canonical_tensor_slot": "request_input",
                "persistence": "request_input",
                "size_bytes": 4096,
            }
        ],
    }
    layer_spec = {
        "graph_id": prior["graph_id"],
        "workload_id": prior["workload_id"],
        "device_domains": {"GPU0": "NPU", "PIM0": "PIM"},
        "layers": [
            {
                "network_index": 0,
                "layer_class": "dense_transformer",
                "phase": "prefill",
                "shape_bucket": "b1-s8",
                "capability_basis": "compute",
                "domain_capabilities": {
                    "NPU": {
                        "effective_compute_flops_per_s": 1000.0,
                        "effective_bandwidth_bytes_per_s": 2000.0,
                        "queue_count": 1,
                    },
                    "PIM": {
                        "effective_compute_flops_per_s": 500.0,
                        "effective_bandwidth_bytes_per_s": 4000.0,
                        "queue_count": 2,
                    },
                },
                "default_order": order,
                "nodes": [
                    {
                        "op_id": op_id,
                        "operator_family": "FORK_JOIN",
                        "placement_supernode": op_id,
                        "parallel_group_hint": (
                            "fork-pair" if op_id in {"left", "right"} else None
                        ),
                        "weight_home": None,
                        "kv_home": None,
                        "expert_id": "expert-3" if op_id == "right" else None,
                        "expert_service_buckets": (
                            [
                                {
                                    "min_tokens": 1,
                                    "max_tokens": 4,
                                    "service_time_s": {
                                        "GPU0": 0.004,
                                        "PIM0": 0.006,
                                    },
                                },
                                {
                                    "min_tokens": 5,
                                    "max_tokens": 8,
                                    "service_time_s": {
                                        "GPU0": 0.007,
                                        "PIM0": 0.009,
                                    },
                                },
                            ]
                            if op_id == "right" else []
                        ),
                    }
                    for op_id in order
                ],
            }
        ],
    }
    return prior, network, tensor_bindings, layer_spec


class HetInferCAMCProfileExportTests(unittest.TestCase):
    def test_builds_complete_profile_without_copying_timing_tables(self) -> None:
        prior, network, tensor_bindings, layer_spec = _artifacts()
        payload = build_camc_profile(
            prior_artifact=prior,
            network_manifest=network,
            tensor_bindings=tensor_bindings,
            layer_spec=layer_spec,
        )

        self.assertEqual(
            set(payload),
            {
                "schema",
                "schema_version",
                "graph_id",
                "workload_id",
                "device_domains",
                "layers",
            },
        )
        self.assertEqual(payload["schema"], CAMC_PROFILE_SCHEMA)
        self.assertEqual(payload["graph_id"], prior["graph_id"])
        self.assertNotIn("t_service", payload)
        self.assertNotIn("t_move", payload)

        layer = payload["layers"][0]
        self.assertEqual(layer["default_order"], layer_spec["layers"][0]["default_order"])
        self.assertEqual(
            layer["domain_capabilities"]["PIM"],
            {
                "effective_compute_flops_per_s": 500.0,
                "effective_bandwidth_bytes_per_s": 4000.0,
                "queue_count": 2,
            },
        )
        nodes = {node["op_id"]: node for node in layer["nodes"]}
        self.assertEqual(nodes["right"]["legal_devices"], ["GPU0", "PIM0"])
        self.assertEqual(nodes["right"]["default_device"], "PIM0")
        self.assertEqual(nodes["right"]["expert_id"], "expert-3")
        self.assertEqual(
            nodes["right"]["expert_service_buckets"][1],
            {
                "min_tokens": 5,
                "max_tokens": 8,
                "service_time_s": {"GPU0": 0.007, "PIM0": 0.009},
            },
        )
        self.assertEqual(nodes["source"]["expert_service_buckets"], [])
        self.assertIsNone(nodes["source"]["weight_home"])

    def test_exporter_and_command_write_the_same_profile(self) -> None:
        prior, network, tensor_bindings, layer_spec = _artifacts()
        expected = build_camc_profile(
            prior_artifact=prior,
            network_manifest=network,
            tensor_bindings=tensor_bindings,
            layer_spec=layer_spec,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct_output = root / "direct.json"
            self.assertEqual(
                export_camc_profile(
                    prior_artifact=prior,
                    network_manifest=network,
                    tensor_bindings=tensor_bindings,
                    layer_spec=layer_spec,
                    output=direct_output,
                ),
                direct_output,
            )
            self.assertEqual(
                json.loads(direct_output.read_text(encoding="utf-8")), expected
            )

            paths = {}
            for name, artifact in (
                ("prior", prior),
                ("network", network),
                ("tensor_bindings", tensor_bindings),
                ("layer_spec", layer_spec),
            ):
                path = root / f"{name}.json"
                path.write_text(json.dumps(artifact), encoding="utf-8")
                paths[name] = path
            command_output = root / "command.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "export_hetinfer_camc_profile.py",
                    "--prior",
                    str(paths["prior"]),
                    "--network",
                    str(paths["network"]),
                    "--tensor-bindings",
                    str(paths["tensor_bindings"]),
                    "--layer-spec",
                    str(paths["layer_spec"]),
                    "--output",
                    str(command_output),
                ],
            ):
                self.assertEqual(command_main(), 0)
            self.assertEqual(
                json.loads(command_output.read_text(encoding="utf-8")), expected
            )

    def test_rejects_mismatched_sidecars(self) -> None:
        prior, network, tensor_bindings, layer_spec = _artifacts()
        wrong_network = copy.deepcopy(network)
        wrong_network["networks"][0]["workload_id"] = "other"
        with self.assertRaisesRegex(RuntimeError, "workload_id"):
            build_camc_profile(
                prior_artifact=prior,
                network_manifest=wrong_network,
                tensor_bindings=tensor_bindings,
                layer_spec=layer_spec,
            )

        missing_network_binding = copy.deepcopy(tensor_bindings)
        missing_network_binding["bindings"] = []
        with self.assertRaisesRegex(RuntimeError, "cover every network_index"):
            build_camc_profile(
                prior_artifact=prior,
                network_manifest=network,
                tensor_bindings=missing_network_binding,
                layer_spec=layer_spec,
            )

    def test_maps_four_profile_phases_to_two_network_phases(self) -> None:
        for profile_phase, network_phase in (
            ("prefill", "prefill"),
            ("decode", "decode"),
            ("verify", "decode"),
            ("draft", "decode"),
        ):
            with self.subTest(profile_phase=profile_phase):
                prior, network, tensor_bindings, layer_spec = _artifacts()
                network["networks"][0]["phase"] = network_phase
                layer_spec["layers"][0]["phase"] = profile_phase
                payload = build_camc_profile(
                    prior_artifact=prior,
                    network_manifest=network,
                    tensor_bindings=tensor_bindings,
                    layer_spec=layer_spec,
                )
                self.assertEqual(payload["layers"][0]["phase"], profile_phase)

        prior, network, tensor_bindings, layer_spec = _artifacts()
        layer_spec["layers"][0]["phase"] = "verify"
        with self.assertRaisesRegex(RuntimeError, "does not match network phase"):
            build_camc_profile(
                prior_artifact=prior,
                network_manifest=network,
                tensor_bindings=tensor_bindings,
                layer_spec=layer_spec,
            )

        network["networks"][0]["phase"] = "verify"
        with self.assertRaisesRegex(RuntimeError, "prefill.*decode"):
            build_camc_profile(
                prior_artifact=prior,
                network_manifest=network,
                tensor_bindings=tensor_bindings,
                layer_spec=layer_spec,
            )

    def test_rejects_network_dependency_drift(self) -> None:
        prior, network, tensor_bindings, layer_spec = _artifacts()

        reordered = copy.deepcopy(network)
        next(
            item
            for item in reordered["networks"][0]["operators"]
            if item["op_id"] == "join"
        )["dependencies"] = ["right", "left"]
        build_camc_profile(
            prior_artifact=prior,
            network_manifest=reordered,
            tensor_bindings=tensor_bindings,
            layer_spec=layer_spec,
        )

        duplicate = copy.deepcopy(network)
        next(
            item
            for item in duplicate["networks"][0]["operators"]
            if item["op_id"] == "left"
        )["dependencies"] = ["source", "source"]

        mismatch = copy.deepcopy(network)
        next(
            item
            for item in mismatch["networks"][0]["operators"]
            if item["op_id"] == "right"
        )["dependencies"] = []

        non_string = copy.deepcopy(network)
        next(
            item
            for item in non_string["networks"][0]["operators"]
            if item["op_id"] == "left"
        )["dependencies"] = [1]

        for expected, case_network in (
            ("unique strings", duplicate),
            ("do not match the prior", mismatch),
            ("non-empty string", non_string),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    build_camc_profile(
                        prior_artifact=prior,
                        network_manifest=case_network,
                        tensor_bindings=tensor_bindings,
                        layer_spec=layer_spec,
                    )

    def test_validates_supernodes_and_parallel_hints(self) -> None:
        prior, network, tensor_bindings, layer_spec = _artifacts()

        connected = copy.deepcopy(layer_spec)
        connected_nodes = {
            node["op_id"]: node for node in connected["layers"][0]["nodes"]
        }
        for op_id in ("source", "left"):
            connected_nodes[op_id]["placement_supernode"] = "stem"
            connected_nodes[op_id]["parallel_group_hint"] = None
        payload = build_camc_profile(
            prior_artifact=prior,
            network_manifest=network,
            tensor_bindings=tensor_bindings,
            layer_spec=connected,
        )
        self.assertEqual(
            [
                node["placement_supernode"]
                for node in payload["layers"][0]["nodes"][:2]
            ],
            ["stem", "stem"],
        )

        different_legal = copy.deepcopy(layer_spec)
        nodes = {
            node["op_id"]: node
            for node in different_legal["layers"][0]["nodes"]
        }
        nodes["source"]["placement_supernode"] = "mixed"
        nodes["join"]["placement_supernode"] = "mixed"

        different_default = copy.deepcopy(layer_spec)
        nodes = {
            node["op_id"]: node
            for node in different_default["layers"][0]["nodes"]
        }
        nodes["source"]["placement_supernode"] = "mixed-default"
        nodes["right"]["placement_supernode"] = "mixed-default"
        nodes["right"]["parallel_group_hint"] = None

        different_hint = copy.deepcopy(layer_spec)
        nodes = {
            node["op_id"]: node
            for node in different_hint["layers"][0]["nodes"]
        }
        nodes["source"]["placement_supernode"] = "hint-mismatch"
        nodes["left"]["placement_supernode"] = "hint-mismatch"

        disconnected = copy.deepcopy(layer_spec)
        nodes = {
            node["op_id"]: node
            for node in disconnected["layers"][0]["nodes"]
        }
        nodes["source"]["placement_supernode"] = "disconnected"
        nodes["finish"]["placement_supernode"] = "disconnected"

        serial_hint = copy.deepcopy(layer_spec)
        nodes = {
            node["op_id"]: node
            for node in serial_hint["layers"][0]["nodes"]
        }
        nodes["source"]["parallel_group_hint"] = "serial"
        nodes["left"]["parallel_group_hint"] = "serial"

        for expected, case_spec in (
            ("identical legal_devices", different_legal),
            ("one default_device", different_default),
            ("one parallel_group_hint", different_hint),
            ("dependency-connected", disconnected),
            ("mutually unreachable", serial_hint),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    build_camc_profile(
                        prior_artifact=prior,
                        network_manifest=network,
                        tensor_bindings=tensor_bindings,
                        layer_spec=case_spec,
                    )

    def test_rejects_invalid_expert_service_buckets(self) -> None:
        prior, network, tensor_bindings, layer_spec = _artifacts()
        nodes = {
            node["op_id"]: node for node in layer_spec["layers"][0]["nodes"]
        }
        cases: list[tuple[str, dict]] = []

        missing_expert_cost = copy.deepcopy(layer_spec)
        next(
            node
            for node in missing_expert_cost["layers"][0]["nodes"]
            if node["op_id"] == "right"
        )["expert_service_buckets"] = []
        cases.append(("non-empty for an expert", missing_expert_cost))

        nonexpert_cost = copy.deepcopy(layer_spec)
        nonexpert_nodes = {
            node["op_id"]: node
            for node in nonexpert_cost["layers"][0]["nodes"]
        }
        nonexpert_nodes["source"]["expert_service_buckets"] = copy.deepcopy(
            nodes["right"]["expert_service_buckets"]
        )
        cases.append(("empty for a non-expert", nonexpert_cost))

        overlapping = copy.deepcopy(layer_spec)
        next(
            node
            for node in overlapping["layers"][0]["nodes"]
            if node["op_id"] == "right"
        )["expert_service_buckets"][1]["min_tokens"] = 4
        cases.append(("must not overlap", overlapping))

        wrong_devices = copy.deepcopy(layer_spec)
        del next(
            node
            for node in wrong_devices["layers"][0]["nodes"]
            if node["op_id"] == "right"
        )["expert_service_buckets"][0]["service_time_s"]["PIM0"]
        cases.append(("exactly cover legal devices", wrong_devices))

        nonpositive = copy.deepcopy(layer_spec)
        next(
            node
            for node in nonpositive["layers"][0]["nodes"]
            if node["op_id"] == "right"
        )["expert_service_buckets"][0]["service_time_s"]["PIM0"] = 0.0
        cases.append(("finite number > 0", nonpositive))

        for expected, case_spec in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    build_camc_profile(
                        prior_artifact=prior,
                        network_manifest=network,
                        tensor_bindings=tensor_bindings,
                        layer_spec=case_spec,
                    )

    def test_rejects_missing_or_invalid_explicit_deployment_fields(self) -> None:
        prior, network, tensor_bindings, layer_spec = _artifacts()
        cases: list[tuple[str, dict]] = []

        missing_home = copy.deepcopy(layer_spec)
        missing_home["layers"][0]["nodes"][0].pop("weight_home")
        cases.append(("fields mismatch", missing_home))

        invalid_expert = copy.deepcopy(layer_spec)
        invalid_expert["layers"][0]["nodes"][0]["expert_id"] = 3
        cases.append(("non-empty string", invalid_expert))

        missing_domain = copy.deepcopy(layer_spec)
        missing_domain["device_domains"].pop("PIM0")
        cases.append(("exactly cover schedulable legal devices", missing_domain))

        invalid_basis = copy.deepcopy(layer_spec)
        invalid_basis["layers"][0]["capability_basis"] = "synthetic"
        cases.append(("compute.*bandwidth", invalid_basis))

        non_topological = copy.deepcopy(layer_spec)
        non_topological["layers"][0]["default_order"] = [
            "left",
            "source",
            "right",
            "join",
            "finish",
        ]
        cases.append(("not topological", non_topological))

        for expected, case_spec in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    build_camc_profile(
                        prior_artifact=prior,
                        network_manifest=network,
                        tensor_bindings=tensor_bindings,
                        layer_spec=case_spec,
                    )

    def test_source_only_cpu_is_not_a_camc_domain_device(self) -> None:
        prior, network, tensor_bindings, layer_spec = _artifacts()
        prior["devices"].append({"device_id": "CPU0"})
        request_input = next(
            entry for entry in prior["inputs"]
            if entry["tensor_id"] == "request_input"
        )
        request_input["source_residencies"].append(
            {"device_id": "CPU0", "layout": "row_major"}
        )
        for destination in ("GPU0", "PIM0"):
            route = {
                "tensor_id": "request_input",
                "source_device_id": "CPU0",
                "destination_device_id": destination,
                "bytes": 4096,
                "layout": "row_major",
            }
            prior["legal_movement_routes"].append(route)
            prior["t_move"].append(
                {**route, "duration_s": 0.001 if destination == "GPU0" else 0.002}
            )

        profile = build_camc_profile(
            prior_artifact=prior,
            network_manifest=network,
            tensor_bindings=tensor_bindings,
            layer_spec=layer_spec,
        )

        self.assertEqual(profile["device_domains"], layer_spec["device_domains"])
        self.assertNotIn("CPU0", profile["device_domains"])
        invalid_spec = copy.deepcopy(layer_spec)
        invalid_spec["device_domains"]["CPU0"] = "NPU"
        with self.assertRaisesRegex(
            RuntimeError, "exactly cover schedulable legal devices"
        ):
            build_camc_profile(
                prior_artifact=prior,
                network_manifest=network,
                tensor_bindings=tensor_bindings,
                layer_spec=invalid_spec,
            )

    def test_prior_operator_subset_requires_explicit_closed_selection(self) -> None:
        prior, network, tensor_bindings, layer_spec = _artifacts()
        source_network = copy.deepcopy(network)
        source_network["networks"][0]["operators"] = [
            source_network["networks"][0]["operators"][0]
        ]
        source_spec = copy.deepcopy(layer_spec)
        source_spec["layers"][0]["default_order"] = ["source"]
        source_spec["layers"][0]["nodes"] = [
            node
            for node in source_spec["layers"][0]["nodes"]
            if node["op_id"] == "source"
        ]

        with self.assertRaisesRegex(RuntimeError, "exactly cover prior operators"):
            build_camc_profile(
                prior_artifact=prior,
                network_manifest=source_network,
                tensor_bindings=tensor_bindings,
                layer_spec=source_spec,
            )
        profile = build_camc_profile(
            prior_artifact=prior,
            network_manifest=source_network,
            tensor_bindings=tensor_bindings,
            layer_spec=source_spec,
            allow_prior_operator_subset=True,
        )
        self.assertEqual(profile["layers"][0]["default_order"], ["source"])

        open_network = copy.deepcopy(network)
        open_network["networks"][0]["operators"] = [
            open_network["networks"][0]["operators"][1]
        ]
        open_spec = copy.deepcopy(layer_spec)
        open_spec["layers"][0]["default_order"] = ["left"]
        open_spec["layers"][0]["nodes"] = [
            node for node in open_spec["layers"][0]["nodes"]
            if node["op_id"] == "left"
        ]
        with self.assertRaisesRegex(RuntimeError, "dependencies outside"):
            build_camc_profile(
                prior_artifact=prior,
                network_manifest=open_network,
                tensor_bindings=tensor_bindings,
                layer_spec=open_spec,
                allow_prior_operator_subset=True,
            )

    def test_machine_readable_schema_matches_strict_contract(self) -> None:
        schema = json.loads(
            (
                REPO_ROOT
                / "schemas"
                / "dops.hetinfer_camc_profile.v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["schema"]["const"], CAMC_PROFILE_SCHEMA
        )
        self.assertIn("device_domains", schema["required"])
        layer = schema["$defs"]["layer"]
        self.assertEqual(
            layer["properties"]["phase"]["enum"],
            ["prefill", "decode", "verify", "draft"],
        )
        self.assertEqual(
            layer["properties"]["capability_basis"]["enum"],
            ["compute", "bandwidth"],
        )
        node = schema["$defs"]["node"]
        self.assertFalse(node["additionalProperties"])
        self.assertEqual(
            node["properties"]["expert_id"]["$ref"],
            "#/$defs/nullableIdentifier",
        )
        for field in (
            "operator_family",
            "legal_devices",
            "default_device",
            "placement_supernode",
            "parallel_group_hint",
            "weight_home",
            "kv_home",
            "expert_id",
            "expert_service_buckets",
        ):
            self.assertIn(field, node["required"])
        bucket = schema["$defs"]["expertServiceBucket"]
        self.assertFalse(bucket["additionalProperties"])
        self.assertEqual(
            set(bucket["required"]),
            {"min_tokens", "max_tokens", "service_time_s"},
        )


if __name__ == "__main__":
    unittest.main()
