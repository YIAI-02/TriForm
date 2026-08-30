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
from mainlib.storage import _resolve_hetinfer_tensor_bindings_output  # noqa: E402


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
        self.assertEqual(
            Path(raw["hetinfer_tensor_bindings_out"]).name,
            "dops_hetinfer_tensor_bindings.json",
        )

        cfg = _load_cfg_from_json(str(path))
        self.assertTrue(Path(cfg["hardware_json"]).is_file())
        self.assertTrue(Path(cfg["pim_config_path"]).is_file())
        self.assertTrue(Path(cfg["ramulator_config_path"]).is_file())

    def test_evaluate_cli_accepts_all_three_outputs(self) -> None:
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
                "--hetinfer-tensor-bindings-out",
                "tensor_bindings.json",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.hetinfer_prior_out, "prior.json")
        self.assertEqual(args.hetinfer_network_out, "network.json")
        self.assertEqual(
            args.hetinfer_tensor_bindings_out, "tensor_bindings.json"
        )

    def test_tensor_bindings_directory_gets_an_automatic_filename(self) -> None:
        self.assertEqual(
            _resolve_hetinfer_tensor_bindings_output("artifacts", tag="8x1"),
            Path("artifacts/dops_hetinfer_tensor_bindings_8x1.json"),
        )


if __name__ == "__main__":
    unittest.main()
