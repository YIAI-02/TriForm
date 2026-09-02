from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mainlib.kv_policy import _ensure_weight_suggest_supported


class WeightSuggestBackendTests(unittest.TestCase):
    def test_fast_npu_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Fast mode is evaluate-only"):
            _ensure_weight_suggest_supported(
                {"npu_backend": "fast", "pim_fast_mode": False}
            )

    def test_pim_fast_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Fast mode is evaluate-only"):
            _ensure_weight_suggest_supported(
                {"npu_backend": "lut", "pim_fast_mode": True}
            )

    def test_non_fast_backends_are_accepted(self) -> None:
        _ensure_weight_suggest_supported(
            {"npu_backend": "lut", "pim_fast_mode": False}
        )

    def test_cli_rejects_fast_mode_before_creating_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir) / "must-not-exist"
            proc = subprocess.run(
                [
                    sys.executable,
                    "src/main.py",
                    "weight-suggest",
                    "--config",
                    "src/examples/weight_suggest_test_config.json",
                    "--npu_backend",
                    "fast",
                    "--no-pim_fast_mode",
                    "--result_dir",
                    str(result_dir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(
                "Fast mode is evaluate-only",
                proc.stdout + proc.stderr,
            )
            self.assertFalse(result_dir.exists())


if __name__ == "__main__":
    unittest.main()
