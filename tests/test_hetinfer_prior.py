from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hardware import Cluster, DeviceSpec  # noqa: E402
from hetinfer_prior import (  # noqa: E402
    PriorValidationError,
    build_artifact,
    load_artifact_bundle,
    validate_artifact,
    write_artifact,
)
from mainlib.storage import _best_summary_config_snapshot  # noqa: E402
from mainlib.runner import _comparison_cfg_without_prior_export  # noqa: E402
from model_parser import build_graph  # noqa: E402
from task_graph import TaskGraph, TaskNode  # noqa: E402


def _cfg() -> dict:
    return {
        "model_family": "llama",
        "model_variant": "7b",
        "model_revision": "test-model-revision",
        "dtype": "fp16",
        "batch": 4,
        "max_batch_size": 8,
        "prefill_len": 128,
        "decode_len": 32,
        "max_seq_len": 160,
        "tp": 2,
        "tp_qkv": 2,
        "tp_ffn": 2,
        "pp": 1,
        "ep": 1,
        "algo": "Bifocal",
    }


def _graph() -> TaskGraph:
    graph = TaskGraph()
    graph.add_node(
        TaskNode(
            "L0_FFN1",
            "FFN_W1",
            weight_id="layers.0.ffn.w1",
            weight_size=4096,
            allowed={"npu": True, "pim": True, "cpu": False},
        )
    )
    return graph


def _cluster() -> Cluster:
    cluster = Cluster()
    cluster.add_device(DeviceSpec("CPU0", "cpu", 1.0, 100.0, 64.0))
    cluster.add_device(DeviceSpec("NPU0", "npu", 100.0, 900.0, 80.0))
    cluster.add_device(DeviceSpec("PIM0", "pim", 10.0, 1200.0, 16.0, pim_type="3d-dram"))
    cluster.connect("CPU0", "NPU0", 64.0)
    cluster.connect("CPU0", "PIM0", 32.0)
    return cluster


def _scores(npu_score: float, pim_score: float) -> dict:
    def one(score: float, compute: float, reload: float) -> dict:
        return {
            "dops_score_s": score,
            "eft_s": score + 0.001,
            "window_s": score + 0.002,
            "compute_s": compute,
            "reload_s": reload,
            "comm_s": 0.0002,
            "weight_reuse_bias_s": -0.0001,
            "decode_amort_bias_s": 0.0,
        }

    return {
        "NPU0": one(npu_score, 0.0012, 0.0004),
        "PIM0": one(pim_score, 0.0009, 0.0008),
    }


def _record(phase: str, call: int, seq_len: int, token_idx: int | None) -> dict:
    return {
        "schedule_call_index": call,
        "node_id": "L0_FFN1",
        "op_type": "FFN_W1",
        "phase": phase,
        "batch": 4,
        "seq_len": seq_len,
        "token_idx": token_idx,
        "baseline_device": "NPU0",
        "legal_devices": ["NPU0", "PIM0"],
        "candidates": _scores(0.003, 0.004),
        "gamma": 0.25,
        "constraints": {"operator_allowed_device_types": {"npu": True, "pim": True}},
        "dynamic_eligible": True,
        "weight": {
            "weight_id": "layers.0.ffn.w1",
            "size_bytes": 4096,
            "storage_layout": "NZ",
        },
    }


class HetInferPriorTests(unittest.TestCase):
    def test_native_evaluate_cli_writes_scored_prior_bundle(self) -> None:
        """Exercise the real DOPS evaluate path, not the legacy exporter."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape = root / "one_layer_shape.json"
            shape.write_text(
                json.dumps(
                    {
                        "hidden_dim": 64,
                        "layer_num": 1,
                        "intermediate_dim": 128,
                        "q_head_num": 4,
                        "kv_head_num": 2,
                    }
                ),
                encoding="utf-8",
            )
            result_dir = root / "results"
            config = root / "evaluate.json"
            config.write_text(
                json.dumps(
                    {
                        "model_family": "qwen",
                        "model_variant": "1.8b",
                        "model_revision": "test:one-layer-shape",
                        "shape_file": str(shape),
                        "dtype": "fp16",
                        "batch": 1,
                        "max_batch_size": 1,
                        "prefill_len": 2,
                        "decode_len": 2,
                        "max_seq_len": 4,
                        "decode_sample_stride": 1,
                        "decode_plan_refresh_stride": 0,
                        "result_dir": str(result_dir),
                        "hardware_json": str(
                            REPO_ROOT / "src" / "examples" / "hardware_1npu_2aim.json"
                        ),
                        "algo": ["Bifocal"],
                        "baselines": [],
                        "tp_qkv": 1,
                        "tp_ffn": 1,
                        "tp_moe": 1,
                        "pp": 1,
                        "ep": 1,
                        "npu_backend": "fast",
                        "pim_fast_mode": True,
                        "scheduler_seed": 0,
                        "dump_graph": False,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "prior.json"
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["TRIFORM_SKIP_PYCACHE_PURGE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SRC_ROOT / "main.py"),
                    "evaluate",
                    "--config",
                    str(config),
                    "--algo",
                    "Bifocal",
                    "--hetinfer-prior-out",
                    str(output),
                ],
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file(), completed.stdout)

            artifact, source_path = load_artifact_bundle(output)
            source = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["schema"], "dops.hetinfer_prior.v1")
            self.assertEqual(source["schema"], "dops.hetinfer_prior_source.v1")
            self.assertEqual(source["kind"], "bifocal_candidate_records")
            self.assertTrue(source["candidate_records"])

            dynamic = []
            for profile in artifact["profiles"]:
                self.assertEqual(set(profile["phases"]), {"prefill", "decode"})
                self.assertTrue(profile["source"]["candidate_scores_complete"])
                for phase in ("prefill", "decode"):
                    for operator in profile["phases"][phase]["operators"]:
                        self.assertEqual(
                            set(operator["legal_devices"]),
                            set(operator["candidates"]),
                        )
                        if operator["dynamic_eligible"]:
                            dynamic.append(operator)
                            self.assertGreater(len(operator["legal_devices"]), 1)
                            self.assertTrue(
                                all(
                                    score["dops_score_s"] is not None
                                    for score in operator["candidates"].values()
                                )
                            )
            self.assertTrue(dynamic)

            summaries = list(result_dir.rglob("best_summary_*.json"))
            self.assertEqual(len(summaries), 1, summaries)
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertEqual(summary["hetinfer_prior_path"], str(output))

    def test_scored_capture_builds_dual_phase_profiles(self) -> None:
        records = [
            _record("prefill", 1, 128, None),
            _record("decode", 2, 128, 0),
            _record("decode", 3, 144, 16),
        ]
        artifact = build_artifact(
            cfg=_cfg(),
            graph=_graph(),
            cluster=_cluster(),
            candidate_records=records,
            producer_revision="0123456789abcdef0123456789abcdef01234567",
            created_at="2026-08-09T00:00:00+00:00",
        )
        validate_artifact(artifact)
        self.assertEqual(artifact["schema"], "dops.hetinfer_prior.v1")
        self.assertEqual(len(artifact["profiles"]), 2)
        for profile in artifact["profiles"]:
            self.assertEqual(set(profile["phases"]), {"prefill", "decode"})
            self.assertTrue(profile["phases"]["decode"]["operators"][0]["dynamic_eligible"])
        provenance = artifact["provenance"]
        self.assertEqual(len(provenance["graph_sha256"]), 64)
        self.assertEqual(len(provenance["hardware_sha256"]), 64)
        self.assertEqual(len(provenance["source_artifact_sha256"]), 64)

        with tempfile.TemporaryDirectory() as tmp:
            path = write_artifact(
                artifact,
                Path(tmp) / "prior.json",
                candidate_records=records,
            )
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(loaded["semantics"]["timeline_is_runtime_contract"])
            verified, source_path = load_artifact_bundle(path)
            self.assertEqual(
                verified["provenance"]["source_artifact_path"],
                source_path.name,
            )
            self.assertEqual(
                verified["provenance"]["source_artifact_sha256"],
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )

    def test_bundle_validation_rejects_tampered_source(self) -> None:
        records = [
            _record("prefill", 1, 128, None),
            _record("decode", 2, 128, 0),
        ]
        artifact = build_artifact(
            cfg=_cfg(),
            graph=_graph(),
            cluster=_cluster(),
            candidate_records=records,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_artifact(
                artifact,
                Path(tmp) / "prior.json",
                candidate_records=records,
            )
            _, source_path = load_artifact_bundle(path)
            source_path.write_text('{"tampered":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(PriorValidationError, "SHA256 mismatch"):
                load_artifact_bundle(path)

    def test_grid_manifest_lists_the_complete_training_bundle(self) -> None:
        records = [
            _record("prefill", 1, 128, None),
            _record("decode", 2, 128, 0),
        ]
        artifact = build_artifact(
            cfg=_cfg(),
            graph=_graph(),
            cluster=_cluster(),
            candidate_records=records,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            case_root = root / "b4_p128_d32"
            prior = write_artifact(
                artifact,
                case_root / "prior.json",
                candidate_records=records,
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(SRC_ROOT)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(
                        REPO_ROOT
                        / "commands"
                        / "hetinfer_gpu_proxy"
                        / "build_prior_grid_manifest.py"
                    ),
                    "--grid-root",
                    str(root),
                    "--require-count",
                    "1",
                ],
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads(
                (root / "prior_grid_manifest.json").read_text(encoding="utf-8")
            )
            entry = manifest["artifacts"][0]
            _, source_path = load_artifact_bundle(prior)
            self.assertEqual(
                entry["bundle_files"],
                [
                    str(prior.relative_to(root)),
                    str(source_path.relative_to(root)),
                ],
            )
            self.assertEqual(
                entry["source_artifact_sha256"],
                artifact["provenance"]["source_artifact_sha256"],
            )

    def test_validator_rejects_unmasked_baseline(self) -> None:
        artifact = build_artifact(
            cfg=_cfg(),
            graph=_graph(),
            cluster=_cluster(),
            candidate_records=[
                _record("prefill", 1, 128, None),
                _record("decode", 2, 128, 0),
            ],
        )
        broken = copy.deepcopy(artifact)
        broken["profiles"][0]["phases"]["decode"]["operators"][0][
            "baseline_device"
        ] = "PIM9"
        with self.assertRaises(PriorValidationError):
            validate_artifact(broken)

    def test_validator_rejects_changed_contract_semantics(self) -> None:
        artifact = build_artifact(
            cfg=_cfg(),
            graph=_graph(),
            cluster=_cluster(),
            candidate_records=[
                _record("prefill", 1, 128, None),
                _record("decode", 2, 128, 0),
            ],
        )
        broken = copy.deepcopy(artifact)
        broken["semantics"]["timeline_is_runtime_contract"] = True
        with self.assertRaises(PriorValidationError):
            validate_artifact(broken)

    def test_candidate_capture_must_cover_exact_legal_set(self) -> None:
        prefill = _record("prefill", 1, 128, None)
        del prefill["candidates"]["PIM0"]
        with self.assertRaisesRegex(PriorValidationError, "candidate keys"):
            build_artifact(
                cfg=_cfg(),
                graph=_graph(),
                cluster=_cluster(),
                candidate_records=[
                    prefill,
                    _record("decode", 2, 128, 0),
                ],
            )

    def test_dynamic_candidate_requires_scores_for_all_legal_devices(self) -> None:
        prefill = _record("prefill", 1, 128, None)
        prefill["candidates"]["PIM0"]["dops_score_s"] = None
        with self.assertRaisesRegex(PriorValidationError, "requires a DOPS score"):
            build_artifact(
                cfg=_cfg(),
                graph=_graph(),
                cluster=_cluster(),
                candidate_records=[
                    prefill,
                    _record("decode", 2, 128, 0),
                ],
            )

    def test_reused_capture_pairs_latest_preceding_prefill(self) -> None:
        prefill_first = _record("prefill", 1, 128, None)
        prefill_second = _record("prefill", 3, 256, None)
        prefill_second["baseline_device"] = "PIM0"
        artifact = build_artifact(
            cfg=_cfg(),
            graph=_graph(),
            cluster=_cluster(),
            candidate_records=[
                prefill_first,
                _record("decode", 2, 128, 0),
                prefill_second,
                _record("decode", 4, 256, 1),
            ],
        )
        self.assertEqual(len(artifact["profiles"]), 2)
        first, second = artifact["profiles"]
        self.assertEqual(
            first["phases"]["prefill"]["operators"][0]["baseline_device"],
            "NPU0",
        )
        self.assertEqual(
            second["phases"]["prefill"]["operators"][0]["baseline_device"],
            "PIM0",
        )

    def test_legacy_summary_is_explicitly_unscored(self) -> None:
        legacy = {
            "config": {"batch": 4, "prefill_len": 128, "decode_len": 1, "dtype": "fp16"},
            "prefill_schedule": [{"node_id": "L0_FFN1", "device": "NPU0"}],
            "decode_steps": [
                {
                    "t": 0,
                    "seq_len": 128,
                    "schedule": [{"node_id": "L0_FFN1", "device": "PIM0"}],
                }
            ],
        }
        artifact = build_artifact(cfg=_cfg(), legacy_best_summary=legacy)
        profile = artifact["profiles"][0]
        self.assertEqual(set(profile["phases"]), {"prefill", "decode"})
        op = profile["phases"]["decode"]["operators"][0]
        self.assertFalse(op["dynamic_eligible"])
        self.assertIsNone(op["candidates"]["PIM0"]["dops_score_s"])
        self.assertFalse(profile["source"]["candidate_scores_complete"])

    def test_schema_file_is_valid_json(self) -> None:
        schema = json.loads(
            (REPO_ROOT / "schemas" / "dops.hetinfer_prior.v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(schema["properties"]["schema"]["const"], "dops.hetinfer_prior.v1")

    def test_best_summary_snapshot_preserves_default_shape_recovery(self) -> None:
        cfg = {
            "model_family": "qwen",
            "model_variant": "7b",
            "shape_file": None,
            "dtype": "fp16",
            "batch": 1,
            "prefill_len": 8,
            "decode_len": 1,
        }
        snapshot = _best_summary_config_snapshot(cfg)
        self.assertNotIn("shape_file", snapshot)
        graph, _ = build_graph(snapshot)
        self.assertGreater(len(graph.nodes), 0)

    def test_weight_suggest_comparisons_cannot_overwrite_selected_prior(self) -> None:
        original = {
            "hetinfer_prior_out": "/tmp/selected-layout-prior.json",
            "algo": "Bifocal",
        }
        comparison = _comparison_cfg_without_prior_export(original)
        self.assertNotIn("hetinfer_prior_out", comparison)
        self.assertEqual(
            original["hetinfer_prior_out"],
            "/tmp/selected-layout-prior.json",
        )

    def test_legacy_export_resolves_paths_relative_to_explicit_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hardware = root / "hardware.json"
            shutil.copyfile(
                REPO_ROOT / "src" / "examples" / "hardware_1npu_2aim.json",
                hardware,
            )
            shape = root / "shape.json"
            shape.write_text(
                json.dumps(
                    {
                        "hidden_dim": 64,
                        "layer_num": 1,
                        "intermediate_dim": 128,
                        "q_head_num": 4,
                        "kv_head_num": 2,
                    }
                ),
                encoding="utf-8",
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "model_family": "qwen",
                        "model_variant": "7b",
                        "dtype": "fp16",
                        "batch": 1,
                        "prefill_len": 8,
                        "decode_len": 1,
                        "hardware_json": "hardware.json",
                        "shape_file": "shape.json",
                        "algo": "Bifocal",
                    }
                ),
                encoding="utf-8",
            )
            summary = root / "best_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "config": {
                            "batch": 1,
                            "prefill_len": 8,
                            "decode_len": 1,
                            "dtype": "fp16",
                            "shape_file": None,
                        },
                        "prefill_schedule": [
                            {"node_id": "L0_FFN1", "device": "NPU0"}
                        ],
                        "decode_steps": [
                            {
                                "t": 0,
                                "seq_len": 8,
                                "schedule": [
                                    {"node_id": "L0_FFN1", "device": "NPU0"}
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "prior.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SRC_ROOT / "export_hetinfer_prior.py"),
                    "--best-summary",
                    str(summary),
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                ],
                # Run outside the config directory without assuming a macOS
                # /private/tmp alias exists on Linux compute nodes.
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            artifact = json.loads(output.read_text(encoding="utf-8"))
            verified, source_path = load_artifact_bundle(output)
            self.assertEqual(verified["artifact_id"], artifact["artifact_id"])
            self.assertEqual(
                artifact["provenance"]["source_artifact_path"],
                source_path.name,
            )
            self.assertEqual(artifact["provenance"]["status"], "complete")
            self.assertEqual(
                artifact["provenance"]["hardware"]["source_path"],
                str(hardware.resolve()),
            )
            self.assertEqual(
                artifact["provenance"]["config"]["snapshot"]["shape_file"],
                str(shape.resolve()),
            )
            self.assertEqual(
                artifact["provenance"]["model"]["shape"]["dim"],
                64,
            )

    def test_cross_repo_golden_artifact(self) -> None:
        golden = json.loads(
            (REPO_ROOT / "tests" / "fixtures" / "dops_hetinfer_prior_v1_golden.json").read_text(
                encoding="utf-8"
            )
        )
        validate_artifact(golden)
        self.assertEqual(set(golden["profiles"][0]["phases"]), {"prefill", "decode"})

    def test_het_infer_loader_verifies_written_bundle_when_available(self) -> None:
        het_infer_root_raw = os.environ.get("HET_INFER_ROOT")
        if not het_infer_root_raw:
            self.skipTest("set HET_INFER_ROOT for the cross-repository loader check")
        het_infer_root = Path(het_infer_root_raw).expanduser().resolve()
        het_infer_src = het_infer_root / "src"
        if not (het_infer_src / "het_infer" / "planning" / "prior.py").is_file():
            self.fail(f"HET_INFER_ROOT has no Het-Infer loader: {het_infer_root}")

        records = [
            _record("prefill", 1, 128, None),
            _record("decode", 2, 128, 0),
        ]
        artifact = build_artifact(
            cfg=_cfg(),
            graph=_graph(),
            cluster=_cluster(),
            candidate_records=records,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_artifact(
                artifact,
                Path(tmp) / "prior.json",
                candidate_records=records,
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(het_infer_src)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from het_infer.planning import load_dops_prior_library; "
                        "p=load_dops_prior_library(__import__('sys').argv[1]).profiles()[0]; "
                        "assert p.provenance.source_artifact_verified; "
                        "print('HET_INFER_DOPS_SOURCE_VERIFIED', p.profile_id)"
                    ),
                    str(path),
                ],
                cwd=het_infer_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("HET_INFER_DOPS_SOURCE_VERIFIED", completed.stdout)


if __name__ == "__main__":
    unittest.main()
