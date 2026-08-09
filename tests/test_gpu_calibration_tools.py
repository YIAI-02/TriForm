from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "measurements" / "gpu_calibration"
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(SRC))

from calibration_common import RAW_SCHEMA, ordinary_least_squares, quantile
from gpu_microbench import _auto_inner_iterations

from hardware import demo_cluster
from mainlib.cli import _apply_runtime_config_overrides, _load_cfg_from_json


def _synthetic_raw() -> dict:
    matmul = []
    for ordinal, (m, n, k) in enumerate(
        ((1, 1536, 1536), (8, 8960, 1536), (128, 1536, 1536), (512, 8960, 1536))
    ):
        flops = 2 * m * n * k
        center_ms = (12e-6 + flops / 100e12) * 1e3
        samples = [center_ms * factor for factor in (0.98, 1.0, 1.02)]
        matmul.append(
            {
                "label": f"shape_{ordinal}",
                "m": m,
                "n": n,
                "k": k,
                "flops_per_op": flops,
                "samples_ms_per_op": samples,
            }
        )

    copies = []
    for size in (32 * 1024**2, 128 * 1024**2, 512 * 1024**2):
        for direction, bandwidth_gib, overhead_s, multiplier in (
            ("d2d", 1200.0, 5e-6, 2),
            ("h2d", 24.0, 10e-6, 1),
            ("d2h", 22.0, 12e-6, 1),
        ):
            center_ms = (
                overhead_s + (size * multiplier) / (bandwidth_gib * 1024**3)
            ) * 1e3
            copies.append(
                {
                    "direction": direction,
                    "size_bytes": size,
                    "samples_ms_per_copy": [
                        center_ms * factor for factor in (0.99, 1.0, 1.01)
                    ],
                }
            )
    return {
        "schema": RAW_SCHEMA,
        "status": "complete",
        "machine": {"hostname": "synthetic-test", "slurm": {"job_id": "fixture"}},
        "software": {"torch_version": "fixture", "torch_cuda_version": "fixture"},
        "device": {
            "name": "Synthetic CUDA GPU",
            "total_memory_bytes": 80 * 1024**3,
            "compute_capability": [8, 0],
        },
        "benchmark": {"dtype": "float16", "warmup": 1, "repeats": 3},
        "matmul": matmul,
        "copy": copies,
        "errors": [],
    }


class GPUCalibrationToolTests(unittest.TestCase):
    def test_dependency_free_math_helpers(self) -> None:
        self.assertEqual(quantile([1, 2, 3], 0.5), 2.0)
        fit = ordinary_least_squares([(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)])
        self.assertAlmostEqual(float(fit["intercept"]), 1.0)
        self.assertAlmostEqual(float(fit["slope"]), 2.0)
        self.assertEqual(_auto_inner_iterations(100, 1000, 8), 8)

    def test_synthetic_fit_export_and_runtime_load(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="dops-gpu-calibration-test-"
        ) as temporary:
            root = Path(temporary)
            raw = root / "raw.json"
            fit = root / "fit.json"
            hardware = root / "hardware.json"
            config = root / "evaluate.json"
            runtime = root / "runtime.json"
            raw.write_text(json.dumps(_synthetic_raw()), encoding="utf-8")
            environment = {**os.environ, "PYTHONPATH": f"{TOOLS}:{SRC}"}

            subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "fit_gpu_calibration.py"),
                    "--input",
                    str(raw),
                    "--output",
                    str(fit),
                    "--device-name-prefix",
                    "GPU0",
                ],
                check=True,
                env=environment,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "export_dops_calibration.py"),
                    "--fit",
                    str(fit),
                    "--base-hardware",
                    str(
                        REPO_ROOT
                        / "configs/hetinfer_gpu_proxy/hardware_gpu0_pim0_a100_llmcompass_proxy.json"
                    ),
                    "--base-config",
                    str(
                        REPO_ROOT
                        / "configs/hetinfer_gpu_proxy/evaluate_qwen1p8b_gpu_proxy_pim_fast.json"
                    ),
                    "--hardware-out",
                    str(hardware),
                    "--config-out",
                    str(config),
                    "--runtime-model-out",
                    str(runtime),
                    "--expect-device-regex",
                    "Synthetic CUDA",
                ],
                check=True,
                env=environment,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "validate_calibrated_export.py"),
                    "--config",
                    str(config),
                ],
                check=True,
                env=environment,
            )

            exported = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(exported["npu_backend"], "fast")
            self.assertIn("gpu_runtime_model_json", exported)
            hardware_raw = json.loads(hardware.read_text(encoding="utf-8"))["hardware"]
            self.assertEqual(hardware_raw["topology"], "star")
            gpu = next(
                device for device in hardware_raw["devices"] if device["name"] == "GPU0"
            )
            self.assertNotIn("llmcompass_device", gpu)
            self.assertGreater(float(gpu["tflops"]), 0.0)

            import config as runtime_config

            old_compute = runtime_config.COMPUTE_UTILIZATION
            old_launch = runtime_config.KERNEL_LAUNCH_OVERHEAD
            try:
                loaded = _load_cfg_from_json(str(config))
                rejected = dict(loaded)
                rejected["npu_backend"] = "llmcompass"
                with self.assertRaisesRegex(
                    ValueError, "only valid with npu_backend=fast"
                ):
                    _apply_runtime_config_overrides(rejected)
                applied = _apply_runtime_config_overrides(loaded)
                self.assertEqual(
                    applied["GPU_RUNTIME_MODEL"]["device_name_prefix"], "GPU0"
                )
                cluster = demo_cluster(loaded)
                self.assertGreater(cluster.devices["GPU0"].tflops, 0.0)
                self.assertIn(
                    "GPU0", runtime_config.COMPUTE_UTILIZATION["by_device_name"]
                )
                self.assertIn(
                    "GPU0", runtime_config.KERNEL_LAUNCH_OVERHEAD["by_device_name"]
                )
                self.assertTrue(Path(loaded["shape_file"]).is_file())
            finally:
                runtime_config.COMPUTE_UTILIZATION = old_compute
                runtime_config.KERNEL_LAUNCH_OVERHEAD = old_launch


if __name__ == "__main__":
    unittest.main()
