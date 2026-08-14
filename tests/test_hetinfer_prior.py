from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
if (REPO_ROOT / "src").is_dir():
    SRC_ROOT = REPO_ROOT / "src"
    FIXTURE = REPO_ROOT / "tests" / "fixtures" / "dops_hetinfer_prior_v1_minimal.json"
    SCHEMA = REPO_ROOT / "schemas" / "dops.hetinfer_prior.v1.schema.json"
else:
    SRC_ROOT = Path(__file__).resolve().parent
    FIXTURE = SRC_ROOT / "dops_hetinfer_prior_v1_minimal.json"
    SCHEMA = SRC_ROOT / "dops.hetinfer_prior.v1.schema.json"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hetinfer_prior import (  # noqa: E402
    PRIOR_SCHEMA,
    DOPSPriorValidationError,
    load_prior_artifact,
    validate_prior_artifact,
    write_prior_artifact,
)



class DOPSPriorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_shared_fork_join_fixture_is_complete_and_queryable(self) -> None:
        artifact = load_prior_artifact(FIXTURE)
        self.assertEqual(PRIOR_SCHEMA, "dops.hetinfer_prior.v1")
        self.assertEqual(artifact.device_ids, ("GPU0", "PIM0"))
        self.assertEqual(
            artifact.expert_placement,
            {
                "source": "GPU0",
                "left": "GPU0",
                "right": "PIM0",
                "join": "GPU0",
                "finish": "GPU0",
            },
        )
        operators = {entry["op_id"]: entry for entry in self.payload["operators"]}
        self.assertEqual(operators["left"]["dependencies"], ["source"])
        self.assertEqual(operators["right"]["dependencies"], ["source"])
        self.assertEqual(operators["join"]["dependencies"], ["left", "right"])
        self.assertEqual(operators["finish"]["dependencies"], ["join"])
        self.assertEqual(len(artifact.inputs), 6)
        self.assertEqual(
            {entry["semantics"] for entry in artifact.inputs_for("join")},
            {"collective_staging"},
        )
        self.assertEqual(
            artifact.inputs_for("source")[0]["producer_op_id"], None
        )
        self.assertEqual(
            artifact.inputs_for("source")[0]["source_residencies"],
            [
                {"device_id": "GPU0", "layout": "row_major"},
                {"device_id": "PIM0", "layout": "pim_blocked"},
            ],
        )
        routes = {
            (
                entry["tensor_id"],
                entry["source_device_id"],
                entry["destination_device_id"],
                entry["bytes"],
                entry["layout"],
            )
            for entry in self.payload["legal_movement_routes"]
        }
        self.assertEqual(
            routes,
            {
                ("request_input", "GPU0", "GPU0", 4096, "row_major"),
                ("request_input", "GPU0", "PIM0", 4096, "row_major"),
                ("request_input", "PIM0", "GPU0", 4096, "pim_blocked"),
                ("request_input", "PIM0", "PIM0", 4096, "pim_blocked"),
                ("fork_activation", "GPU0", "GPU0", 8192, "row_major"),
                ("fork_activation", "GPU0", "PIM0", 8192, "row_major"),
                ("left_activation", "GPU0", "GPU0", 8192, "row_major"),
                ("left_activation", "PIM0", "GPU0", 8192, "pim_blocked"),
                ("right_activation", "GPU0", "PIM0", 8192, "row_major"),
                ("right_activation", "PIM0", "PIM0", 8192, "pim_blocked"),
            },
        )
        for operator in self.payload["operators"]:
            for device in operator["legal_devices"]:
                self.assertGreaterEqual(
                    artifact.service_time_s(operator["op_id"], device), 0.0
                )
        self.assertEqual(
            artifact.movement_time_s(
                "request_input", "GPU0", "GPU0", 4096, "row_major"
            ),
            0.0,
        )
        self.assertEqual(
            artifact.movement_time_s(
                "fork_activation", "GPU0", "PIM0", 8192, "row_major"
            ),
            0.0025,
        )

    def test_collective_context_has_fixed_staging_and_atomic_service(self) -> None:
        artifact = validate_prior_artifact(self.payload)
        context = artifact.collective_context("join")
        self.assertEqual(context["primitive"], "ALLREDUCE")
        self.assertEqual(context["topology"], "ring")
        self.assertEqual(context["canonical_device_id"], "GPU0")
        self.assertEqual(
            context["internal_transport"], "included_in_t_service"
        )
        staging = artifact.inputs_for("join")
        self.assertEqual(
            {entry["producer_op_id"]: entry["destination_devices"] for entry in staging},
            {"left": ["GPU0"], "right": ["PIM0"]},
        )
        self.assertEqual(
            {
                device
                for entry in staging
                for device in entry["destination_devices"]
            },
            set(context["participant_device_ids"]),
        )
        route_keys = {
            (
                entry["tensor_id"],
                entry["source_device_id"],
                entry["destination_device_id"],
                entry["bytes"],
                entry["layout"],
            )
            for entry in self.payload["legal_movement_routes"]
        }
        for entry in staging:
            expected_closure = {
                (
                    entry["tensor_id"],
                    source["device_id"],
                    destination,
                    entry["bytes"],
                    source["layout"],
                )
                for source in entry["source_residencies"]
                for destination in entry["destination_devices"]
            }
            self.assertTrue(expected_closure.issubset(route_keys))
        self.assertEqual(artifact.legal_devices["join"], ("GPU0",))
        self.assertEqual(artifact.service_time_s("join", "GPU0"), 0.0041)

    def test_writer_round_trip_preserves_inputs_and_all_three_tables(self) -> None:
        expected = validate_prior_artifact(self.payload).payload
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prior.json"
            write_prior_artifact(self.payload, output)
            reloaded = load_prior_artifact(output)
        self.assertEqual(reloaded.payload["inputs"], expected["inputs"])
        self.assertEqual(
            reloaded.payload["expert_placement"], expected["expert_placement"]
        )
        self.assertEqual(reloaded.payload["t_service"], expected["t_service"])
        self.assertEqual(reloaded.payload["t_move"], expected["t_move"])

    def test_machine_readable_schema_matches_contract_version(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema"]["const"], PRIOR_SCHEMA)
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertIn("inputs", schema["required"])
        self.assertIn("collective_contexts", schema["required"])
        self.assertFalse(schema["$defs"]["input"]["additionalProperties"])
        self.assertIn(
            "destination_devices", schema["$defs"]["input"]["required"]
        )
        self.assertEqual(
            schema["$defs"]["collectiveContext"]["properties"]
            ["internal_transport"]["const"],
            "included_in_t_service",
        )
        self.assertFalse(
            schema["$defs"]["sourceResidency"]["additionalProperties"]
        )
        self.assertFalse(schema["additionalProperties"])

    def test_rejects_invalid_ambiguous_or_unbound_inputs(self) -> None:
        cases: list[tuple[str, dict]] = []

        duplicate = copy.deepcopy(self.payload)
        duplicate["inputs"].append(copy.deepcopy(duplicate["inputs"][0]))
        cases.append(("duplicate input key", duplicate))

        unexpected_field = copy.deepcopy(self.payload)
        unexpected_field["inputs"][0]["input_id"] = "not-in-v1"
        cases.append(("unexpected fields", unexpected_field))

        unexpected_residency_field = copy.deepcopy(self.payload)
        unexpected_residency_field["inputs"][0]["source_residencies"][0][
            "kind"
        ] = "GPU"
        cases.append(("unexpected fields", unexpected_residency_field))

        unknown_consumer = copy.deepcopy(self.payload)
        unknown_consumer["inputs"][0]["consumer_op_id"] = "unknown"
        cases.append(("unknown consumer_op_id", unknown_consumer))

        not_dependency = copy.deepcopy(self.payload)
        not_dependency["inputs"][1]["producer_op_id"] = "right"
        cases.append(("is not a dependency", not_dependency))

        missing_dependency = copy.deepcopy(self.payload)
        missing_dependency["inputs"] = [
            entry
            for entry in missing_dependency["inputs"]
            if not (
                entry["consumer_op_id"] == "join"
                and entry["producer_op_id"] == "left"
            )
        ]
        cases.append(("declared operator dependency", missing_dependency))

        unknown_source = copy.deepcopy(self.payload)
        unknown_source["inputs"][0]["source_residencies"] = [
            {"device_id": "GPU9", "layout": "row_major"}
        ]
        cases.append(("unknown devices", unknown_source))

        empty_data_sources = copy.deepcopy(self.payload)
        empty_data_sources["inputs"][0]["source_residencies"] = []
        cases.append(("data semantics requires non-empty", empty_data_sources))

        incomplete_data_destinations = copy.deepcopy(self.payload)
        incomplete_data_destinations["inputs"][0]["destination_devices"] = [
            "GPU0"
        ]
        cases.append(
            ("data inputs must target exactly", incomplete_data_destinations)
        )

        external_collective = copy.deepcopy(self.payload)
        external_collective["inputs"][0]["semantics"] = "collective_staging"
        cases.append(("external input", external_collective))

        unknown_semantics = copy.deepcopy(self.payload)
        unknown_semantics["inputs"][0]["semantics"] = "control"
        cases.append(("must be one of", unknown_semantics))

        duplicate_source = copy.deepcopy(self.payload)
        duplicate_source["inputs"][0]["source_residencies"].append(
            {"device_id": "GPU0", "layout": "column_major"}
        )
        cases.append(("duplicate device_id", duplicate_source))

        malformed_barrier = copy.deepcopy(self.payload)
        malformed_barrier["inputs"][5]["bytes"] = 1
        cases.append(("barrier semantics requires", malformed_barrier))

        inconsistent_tensor = copy.deepcopy(self.payload)
        inconsistent_tensor["inputs"][2]["bytes"] = 4096
        cases.append(("tensor_id must bind", inconsistent_tensor))

        missing_data_route = copy.deepcopy(self.payload)
        missing_data_route["legal_movement_routes"] = [
            entry
            for entry in missing_data_route["legal_movement_routes"]
            if not (
                entry["tensor_id"] == "request_input"
                and entry["destination_device_id"] == "PIM0"
            )
        ]
        cases.append(("data input is missing legal movement routes", missing_data_route))

        for expected, payload in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(DOPSPriorValidationError, expected):
                    validate_prior_artifact(payload)

    def test_barrier_has_no_route_and_collective_only_exposes_staging_routes(self) -> None:
        route_tensors = {
            entry["tensor_id"] for entry in self.payload["legal_movement_routes"]
        }
        self.assertNotIn("barrier:join:finish", route_tensors)
        self.assertFalse(any(tensor.startswith("collective_internal:") for tensor in route_tensors))
        artifact = validate_prior_artifact(self.payload)
        self.assertEqual(
            [entry["semantics"] for entry in artifact.inputs_for("finish")],
            ["barrier"],
        )

        barrier_route = copy.deepcopy(self.payload)
        route = {
            "tensor_id": "barrier:join:finish",
            "source_device_id": "GPU0",
            "destination_device_id": "GPU0",
            "bytes": 0,
            "layout": "barrier",
        }
        barrier_route["legal_movement_routes"].append(route)
        barrier_route["t_move"].append({**route, "duration_s": 0.0})
        with self.assertRaisesRegex(
            DOPSPriorValidationError, "barrier input must not have"
        ):
            validate_prior_artifact(barrier_route)

    def test_rejects_incomplete_collective_context_and_double_count_marker(self) -> None:
        cases: list[tuple[str, dict]] = []

        missing_staging_route = copy.deepcopy(self.payload)
        missing_staging_route["legal_movement_routes"] = [
            entry
            for entry in missing_staging_route["legal_movement_routes"]
            if not (
                entry["tensor_id"] == "left_activation"
                and entry["source_device_id"] == "PIM0"
                and entry["destination_device_id"] == "GPU0"
            )
        ]
        cases.append(("collective_staging input is missing legal movement routes", missing_staging_route))

        wrong_participants = copy.deepcopy(self.payload)
        wrong_participants["collective_contexts"][0]["participant_device_ids"] = [
            "GPU0"
        ]
        cases.append(("staging device is not a collective participant", wrong_participants))

        double_count_marker = copy.deepcopy(self.payload)
        double_count_marker["collective_contexts"][0]["internal_transport"] = (
            "separate_t_move"
        )
        cases.append(("must equal 'included_in_t_service'", double_count_marker))

        for expected, payload in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(DOPSPriorValidationError, expected):
                    validate_prior_artifact(payload)

    def test_rejects_duplicate_missing_unknown_and_illegal_placement(self) -> None:
        cases: list[tuple[str, dict]] = []
        duplicate = copy.deepcopy(self.payload)
        duplicate["expert_placement"].append(
            copy.deepcopy(duplicate["expert_placement"][0])
        )
        cases.append(("duplicate op_id", duplicate))
        missing = copy.deepcopy(self.payload)
        missing["expert_placement"].pop()
        cases.append(("must cover every operator", missing))
        unknown = copy.deepcopy(self.payload)
        unknown["expert_placement"][0]["op_id"] = "unknown"
        cases.append(("unknown op_id", unknown))
        illegal = copy.deepcopy(self.payload)
        illegal["operators"][0]["legal_devices"] = ["GPU0"]
        illegal["inputs"][0]["destination_devices"] = ["GPU0"]
        illegal["expert_placement"][0]["device_id"] = "PIM0"
        illegal["t_service"] = [
            entry
            for entry in illegal["t_service"]
            if not (entry["op_id"] == "source" and entry["device_id"] == "PIM0")
        ]
        cases.append(("is not legal", illegal))
        for expected, payload in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(DOPSPriorValidationError, expected):
                    validate_prior_artifact(payload)

    def test_rejects_incomplete_duplicate_or_invalid_service_entries(self) -> None:
        cases: list[tuple[str, dict]] = []
        missing = copy.deepcopy(self.payload)
        missing["t_service"].pop()
        cases.append(("must cover every legal operator-device", missing))
        duplicate = copy.deepcopy(self.payload)
        duplicate["t_service"].append(copy.deepcopy(duplicate["t_service"][0]))
        cases.append(("duplicate key", duplicate))
        negative = copy.deepcopy(self.payload)
        negative["t_service"][0]["duration_s"] = -0.1
        cases.append(("finite, non-negative", negative))
        nonfinite = copy.deepcopy(self.payload)
        nonfinite["t_service"][0]["duration_s"] = float("inf")
        cases.append(("finite, non-negative", nonfinite))
        unknown_device = copy.deepcopy(self.payload)
        unknown_device["t_service"][0]["device_id"] = "GPU9"
        cases.append(("is not legal", unknown_device))
        for expected, payload in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(DOPSPriorValidationError, expected):
                    validate_prior_artifact(payload)

    def test_rejects_incomplete_duplicate_or_invalid_movement_entries(self) -> None:
        cases: list[tuple[str, dict]] = []
        missing = copy.deepcopy(self.payload)
        missing["t_move"].pop()
        cases.append(("must cover every legal movement route", missing))
        duplicate = copy.deepcopy(self.payload)
        duplicate["t_move"].append(copy.deepcopy(duplicate["t_move"][0]))
        cases.append(("duplicate key", duplicate))
        nonzero_resident = copy.deepcopy(self.payload)
        nonzero_resident["t_move"][0]["duration_s"] = 0.1
        cases.append(("resident source/destination must be zero", nonzero_resident))
        unknown_device = copy.deepcopy(self.payload)
        unknown_device["legal_movement_routes"][0]["source_device_id"] = "GPU9"
        cases.append(("unknown devices", unknown_device))
        for expected, payload in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(DOPSPriorValidationError, expected):
                    validate_prior_artifact(payload)

    def test_rejects_missing_fields_wrong_units_and_internal_score_terms(self) -> None:
        missing = copy.deepcopy(self.payload)
        del missing["t_move"][0]["layout"]
        with self.assertRaisesRegex(DOPSPriorValidationError, "missing fields"):
            validate_prior_artifact(missing)

        wrong_units = copy.deepcopy(self.payload)
        wrong_units["time_unit"] = "milliseconds"
        with self.assertRaisesRegex(DOPSPriorValidationError, "time_unit"):
            validate_prior_artifact(wrong_units)

        for forbidden in ("dops_score_s", "eft_s", "window_s", "reload_s", "comm_s"):
            polluted = copy.deepcopy(self.payload)
            polluted["t_service"][0][forbidden] = 123.0
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(
                    DOPSPriorValidationError, "unexpected fields"
                ):
                    validate_prior_artifact(polluted)

    def test_rejects_replaced_score_prior_and_retired_v2_discriminator(self) -> None:
        old_score_prior = {
            "schema": "dops.hetinfer_prior.v1",
            "schema_version": 1,
            "profile_id": "legacy-score-profile",
            "placements": [],
            "operator_costs": [],
        }
        with self.assertRaisesRegex(DOPSPriorValidationError, "missing fields"):
            validate_prior_artifact(old_score_prior)

        retired_v2 = copy.deepcopy(self.payload)
        retired_v2["schema"] = "dops.hetinfer_prior.v2"
        retired_v2["schema_version"] = 2
        with self.assertRaisesRegex(DOPSPriorValidationError, "schema"):
            validate_prior_artifact(retired_v2)

    def test_json_loader_rejects_duplicate_keys_and_nonfinite_literals(self) -> None:
        text = FIXTURE.read_text(encoding="utf-8")
        duplicate_key = text.replace(
            '"time_unit": "seconds",',
            '"time_unit": "seconds",\n  "time_unit": "seconds",',
            1,
        )
        nonfinite = text.replace('"duration_s": 0.0010', '"duration_s": NaN', 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate_path = root / "duplicate.json"
            duplicate_path.write_text(duplicate_key, encoding="utf-8")
            with self.assertRaisesRegex(
                DOPSPriorValidationError, "duplicate JSON"
            ):
                load_prior_artifact(duplicate_path)
            nonfinite_path = root / "nonfinite.json"
            nonfinite_path.write_text(nonfinite, encoding="utf-8")
            with self.assertRaisesRegex(
                DOPSPriorValidationError, "non-finite JSON"
            ):
                load_prior_artifact(nonfinite_path)


if __name__ == "__main__":
    unittest.main()
