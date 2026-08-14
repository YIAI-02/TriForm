from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
import tempfile
import typing
import unittest


TEST_FILE = Path(__file__).resolve()
REPO_ROOT = TEST_FILE.parents[1]
if (REPO_ROOT / "src").is_dir():
    SRC_ROOT = REPO_ROOT / "src"
    MAINLIB_ROOT = SRC_ROOT / "mainlib"
else:
    # The Step-2 staging directory keeps production files flattened next to
    # this test.  The fallback makes the pure-exporter tests runnable before
    # the file is copied back to DOPS/tests on the HPC checkout.
    SRC_ROOT = TEST_FILE.parent
    MAINLIB_ROOT = TEST_FILE.parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hetinfer_prior_export import (  # noqa: E402
    PriorExportError,
    export_atlas_timing_request,
    export_prior_artifact,
    snapshots_require_atlas,
)


CFG = {
    "hetinfer_graph_id": "cli-two-stage-graph",
    "hetinfer_workload_id": "cli-two-stage-workload",
}
EXPORT_CONFIG_KEYS = {
    "hetinfer_prior_out",
    "hetinfer_atlas_timings",
    "hetinfer_atlas_manifest_out",
}
EXPORT_FLAGS = {
    "--hetinfer-prior-out",
    "--hetinfer-atlas-timings",
    "--hetinfer-atlas-manifest-out",
}


def _source_path(name: str) -> Path:
    if name == "hetinfer_prior_export.py":
        return SRC_ROOT / name
    return MAINLIB_ROOT / name


def _source(name: str) -> str:
    return _source_path(name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    return ast.parse(_source(name), filename=str(_source_path(name)))


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _snapshot(*, with_pim: bool) -> dict:
    def descriptor(*, devices: list[str]) -> dict:
        return {
            "op_kind": "ROOT",
            "phase": "prefill",
            "batch": 1,
            "seq_len": 8,
            "attrs": {
                "collective_context": None,
                "collective_input_bindings": [],
            },
            "weight_layout_by_device": {device: "NONE" for device in devices},
            "collective_primitive": None,
            "collective_participants": [],
            "topology": "fc",
        }

    def route_descriptor(source_type: str, destination_type: str) -> dict:
        return {
            "topology": "fc",
            "source_device_type": source_type,
            "destination_device_type": destination_type,
        }

    if not with_pim:
        return {
            "schedule_call_index": 1,
            "phase": "prefill",
            "devices": [
                {"device_id": "CPU0", "device_type": "cpu"},
                {"device_id": "NPU0", "device_type": "npu"},
            ],
            "operators": [
                {
                    "op_id": "root",
                    "dependencies": [],
                    "legal_devices": ["CPU0", "NPU0"],
                    "expert_device": "NPU0",
                    "service_s": {"CPU0": 0.004, "NPU0": 0.001},
                    "atlas_descriptor": descriptor(devices=["CPU0", "NPU0"]),
                }
            ],
            "inputs": [
                {
                    "consumer_op_id": "root",
                    "producer_op_id": None,
                    "tensor_id": "request_input",
                    "semantics": "data",
                    "bytes": 4096,
                    "source_residencies": [
                        {"device_id": "CPU0", "layout": "ND"}
                    ],
                    "destination_devices": ["CPU0", "NPU0"],
                }
            ],
            "collective_contexts": [],
            "routes": [
                {
                    "tensor_id": "request_input",
                    "source_device_id": "CPU0",
                    "destination_device_id": destination,
                    "bytes": 4096,
                    "layout": "ND",
                    "duration_s": 0.0 if destination == "CPU0" else 0.00001,
                    "requires_atlas": False,
                    "atlas_descriptor": route_descriptor(
                        "cpu", "cpu" if destination == "CPU0" else "npu"
                    ),
                }
                for destination in ("CPU0", "NPU0")
            ],
        }

    return {
        "schedule_call_index": 1,
        "phase": "prefill",
        "devices": [
            {"device_id": "NPU0", "device_type": "npu"},
            {"device_id": "PIM0", "device_type": "pim"},
        ],
        "operators": [
            {
                "op_id": "root",
                "dependencies": [],
                "legal_devices": ["NPU0", "PIM0"],
                "expert_device": "NPU0",
                "service_s": {"NPU0": 0.001, "PIM0": None},
                "atlas_descriptor": descriptor(devices=["NPU0", "PIM0"]),
            }
        ],
        "inputs": [
            {
                "consumer_op_id": "root",
                "producer_op_id": None,
                "tensor_id": "request_input",
                "semantics": "data",
                "bytes": 4096,
                "source_residencies": [
                    {"device_id": "NPU0", "layout": "ND"}
                ],
                "destination_devices": ["NPU0", "PIM0"],
            }
        ],
        "collective_contexts": [],
        "routes": [
            {
                "tensor_id": "request_input",
                "source_device_id": "NPU0",
                "destination_device_id": destination,
                "bytes": 4096,
                "layout": "ND",
                "duration_s": 0.0 if destination == "NPU0" else None,
                "requires_atlas": destination == "PIM0",
                "atlas_descriptor": route_descriptor(
                    "npu", "npu" if destination == "NPU0" else "pim"
                ),
            }
            for destination in ("NPU0", "PIM0")
        ],
    }


class HetInferPriorCLITests(unittest.TestCase):
    def test_export_paths_are_canonical_and_pairwise_distinct(self) -> None:
        storage_tree = _tree("storage.py")
        function = next(
            node
            for node in storage_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_canonicalize_hetinfer_export_paths"
        )
        isolated = ast.Module(body=[copy.deepcopy(function)], type_ignores=[])
        ast.fix_missing_locations(isolated)
        namespace = {
            "Path": Path,
            "Dict": typing.Dict,
            "Tuple": typing.Tuple,
        }
        exec(compile(isolated, str(_source_path("storage.py")), "exec"), namespace)
        canonicalize = namespace["_canonicalize_hetinfer_export_paths"]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            timing = root / "atlas-timings.json"
            timing.write_text("{}", encoding="utf-8")

            atlas, manifest, prior = canonicalize(
                atlas_timings=timing,
                atlas_manifest_out=root / "atlas-request.json",
                prior_out=root / "prior.json",
            )
            self.assertEqual(atlas, timing)
            self.assertEqual(manifest, root / "atlas-request.json")
            self.assertEqual(prior, root / "prior.json")
            self.assertTrue(all(path.is_absolute() for path in (atlas, manifest, prior)))

            relative_alias = timing.parent / "unused" / ".." / timing.name
            with self.assertRaisesRegex(ValueError, "pairwise distinct"):
                canonicalize(
                    atlas_timings=timing,
                    atlas_manifest_out=relative_alias,
                    prior_out=root / "prior.json",
                )

            with self.assertRaisesRegex(ValueError, "pairwise distinct"):
                canonicalize(
                    atlas_timings=timing,
                    atlas_manifest_out=root / "atlas-request.json",
                    prior_out=timing,
                )

            shared_output = root / "shared-output.json"
            with self.assertRaisesRegex(ValueError, "pairwise distinct"):
                canonicalize(
                    atlas_timings=timing,
                    atlas_manifest_out=shared_output,
                    prior_out=shared_output,
                )

            symlink_alias = root / "timings-link.json"
            symlink_alias.symlink_to(timing)
            with self.assertRaisesRegex(ValueError, "pairwise distinct"):
                canonicalize(
                    atlas_timings=timing,
                    atlas_manifest_out=symlink_alias,
                    prior_out=root / "prior.json",
                )

    def test_evaluate_and_weight_suggest_expose_both_export_stages(self) -> None:
        parse_args = next(
            node
            for node in _tree("cli.py").body
            if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
        )
        flags_by_parser: dict[str, set[str]] = {"sp_eval": set(), "sp_ws": set()}
        for node in ast.walk(parse_args):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_argument" or not isinstance(
                node.func.value, ast.Name
            ):
                continue
            parser_name = node.func.value.id
            if parser_name not in flags_by_parser:
                continue
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str
                ):
                    flags_by_parser[parser_name].add(argument.value)

        self.assertTrue(
            EXPORT_FLAGS.issubset(flags_by_parser["sp_eval"]),
            flags_by_parser["sp_eval"],
        )
        self.assertTrue(
            EXPORT_FLAGS.issubset(flags_by_parser["sp_ws"]),
            flags_by_parser["sp_ws"],
        )

        # app.main must forward parsed CLI values into the normalized config.
        app_literals = {
            node.value
            for node in ast.walk(_tree("app.py"))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue(EXPORT_CONFIG_KEYS.issubset(app_literals), app_literals)

    def test_workflows_use_only_the_new_capture_api(self) -> None:
        for name in ("evaluate.py", "runner.py", "kv_policy.py"):
            with self.subTest(name=name):
                literals = {
                    node.value
                    for node in ast.walk(_tree(name))
                    if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                }
                self.assertIn("enable_hetinfer_prior_capture", literals)
                self.assertNotIn("enable_hetinfer_candidate_capture", literals)

    def test_cpu_npu_only_snapshots_do_not_require_atlas_timings(self) -> None:
        snapshots = [_snapshot(with_pim=False)]
        self.assertFalse(snapshots_require_atlas(snapshots))

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prior.json"
            written = export_prior_artifact(
                cfg=CFG,
                snapshots=snapshots,
                atlas_timings=None,
                output=output,
                overwrite=True,
            )
            self.assertEqual(written, output)
            self.assertTrue(output.is_file())

    def test_pim_missing_timings_fails_closed_after_manifest_write(self) -> None:
        snapshots = [_snapshot(with_pim=True)]
        self.assertTrue(snapshots_require_atlas(snapshots))

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "atlas-request.json"
            prior = Path(tmp) / "prior.json"

            export_atlas_timing_request(
                cfg=CFG,
                snapshots=snapshots,
                output=manifest,
                overwrite=True,
            )
            self.assertTrue(manifest.is_file())

            with self.assertRaisesRegex(
                PriorExportError, "precomputed ATLAS timings are required"
            ):
                export_prior_artifact(
                    cfg=CFG,
                    snapshots=snapshots,
                    atlas_timings=None,
                    output=prior,
                    overwrite=True,
                )

            self.assertTrue(manifest.is_file())
            self.assertFalse(prior.exists())

    def test_workflows_write_manifest_before_timing_gate_and_force_overwrite(
        self,
    ) -> None:
        for name in ("evaluate.py", "runner.py"):
            with self.subTest(name=name):
                calls = [
                    node
                    for node in ast.walk(_tree(name))
                    if isinstance(node, ast.Call)
                ]
                by_name: dict[str, list[ast.Call]] = {}
                for call in calls:
                    call_name = _call_name(call)
                    if call_name is not None:
                        by_name.setdefault(call_name, []).append(call)

                manifest_calls = by_name.get("export_atlas_timing_request", [])
                canonicalize_calls = by_name.get(
                    "_canonicalize_hetinfer_export_paths", []
                )
                timing_load_calls = by_name.get("load_atlas_timings", [])
                gate_calls = by_name.get("snapshots_require_atlas", [])
                prior_calls = by_name.get("export_prior_artifact", [])
                self.assertEqual(len(manifest_calls), 1)
                self.assertEqual(len(canonicalize_calls), 1)
                self.assertEqual(len(timing_load_calls), 1)
                self.assertEqual(len(gate_calls), 1)
                self.assertEqual(len(prior_calls), 1)
                self.assertLess(
                    canonicalize_calls[0].lineno, timing_load_calls[0].lineno
                )
                self.assertLess(timing_load_calls[0].lineno, manifest_calls[0].lineno)
                self.assertLess(manifest_calls[0].lineno, gate_calls[0].lineno)
                self.assertLess(gate_calls[0].lineno, prior_calls[0].lineno)

                for call in manifest_calls + prior_calls:
                    overwrite = next(
                        (kw.value for kw in call.keywords if kw.arg == "overwrite"),
                        None,
                    )
                    self.assertIsInstance(overwrite, ast.Constant)
                    self.assertIs(overwrite.value, True)

    def test_export_config_is_removed_from_placement_comparisons(self) -> None:
        runner_tree = _tree("runner.py")
        function = next(
            node
            for node in runner_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_comparison_cfg_without_prior_export"
        )
        isolated = ast.Module(body=[copy.deepcopy(function)], type_ignores=[])
        ast.fix_missing_locations(isolated)
        namespace = {"Dict": typing.Dict, "Any": typing.Any}
        exec(compile(isolated, str(_source_path("runner.py")), "exec"), namespace)
        strip_export = namespace["_comparison_cfg_without_prior_export"]

        original = {
            "algo": "Bifocal",
            "scheduler_seed": 17,
            "placement_canary": "keep-me",
            "hetinfer_prior_out": "prior.json",
            "hetinfer_atlas_timings": "atlas.json",
            "hetinfer_atlas_manifest_out": "request.json",
        }
        comparison = strip_export(original)

        self.assertEqual(comparison["placement_canary"], "keep-me")
        self.assertEqual(comparison["scheduler_seed"], 17)
        self.assertTrue(EXPORT_CONFIG_KEYS.isdisjoint(comparison))
        self.assertTrue(EXPORT_CONFIG_KEYS.issubset(original))


if __name__ == "__main__":
    unittest.main()
