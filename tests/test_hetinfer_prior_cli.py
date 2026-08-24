from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mainlib.cli import _load_cfg_from_json, parse_args  # noqa: E402


class HetInferNativeCliTests(unittest.TestCase):
    def test_fixed_config_has_native_backends_and_readable_ids(self) -> None:
        path = (
            REPO_ROOT
            / "configs"
            / "hetinfer_native"
            / "evaluate_qwen1layer_b1_s8_d1_1npu2aim.json"
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw["hetinfer_graph_id"], "qwen-1.8b-1layer-1npu2pim")
        self.assertEqual(raw["hetinfer_workload_id"], "b1-prefill8-decode1")
        self.assertEqual(raw["npu_backend"], "lut")
        self.assertIs(raw["npu_lut_strict"], True)
        self.assertIs(raw["pim_fast_mode"], False)
        self.assertIs(raw["pim_trace_strict"], True)

        cfg = _load_cfg_from_json(str(path))
        self.assertTrue(Path(cfg["hardware_json"]).is_file())
        self.assertTrue(Path(cfg["pim_config_path"]).is_file())
        self.assertTrue(Path(cfg["ramulator_config_path"]).is_file())

    def test_evaluate_cli_accepts_the_paired_outputs(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "main.py",
                "evaluate",
                "--config",
                "config.json",
                "--hetinfer-prior-out",
                "prior.json",
                "--hetinfer-network-out",
                "network.json",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.hetinfer_prior_out, "prior.json")
        self.assertEqual(args.hetinfer_network_out, "network.json")


if __name__ == "__main__":
    unittest.main()
