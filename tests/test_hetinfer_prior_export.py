from __future__ import annotations

from pathlib import Path
import sys
import unittest


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hetinfer_prior import validate_prior_artifact  # noqa: E402
from hetinfer_prior_export import build_prior_artifact  # noqa: E402


def _snapshot(*, service_s: float | None = 0.001) -> dict:
    return {
        "schedule_call_index": 1,
        "phase": "prefill",
        "devices": [
            {"device_id": "CPU0", "device_type": "cpu"},
            {"device_id": "Ascend_910B_NPU0", "device_type": "npu"},
        ],
        "operators": [
            {
                "op_id": "prefill:1:q",
                "dependencies": [],
                "legal_devices": ["Ascend_910B_NPU0"],
                "expert_device": "Ascend_910B_NPU0",
                "service_s": {"Ascend_910B_NPU0": service_s},
                "network_metadata": {
                    "name": "Q",
                    "phase": "prefill",
                    "batch": 1,
                    "seq_len": 8,
                    "node_attrs": {"layer": 0},
                },
            }
        ],
        "inputs": [
            {
                "consumer_op_id": "prefill:1:q",
                "producer_op_id": None,
                "tensor_id": "prefill:1:input",
                "semantics": "data",
                "bytes": 16,
                "source_residencies": [{"device_id": "CPU0", "layout": "ND"}],
                "destination_devices": ["Ascend_910B_NPU0"],
            }
        ],
        "collective_contexts": [],
        "routes": [
            {
                "tensor_id": "prefill:1:input",
                "source_device_id": "CPU0",
                "destination_device_id": "Ascend_910B_NPU0",
                "bytes": 16,
                "layout": "ND",
                "duration_s": 0.0001,
            }
        ],
    }


class HetInferPriorExportTests(unittest.TestCase):
    def test_projects_native_timings_with_readable_ids(self) -> None:
        payload = build_prior_artifact(
            cfg={
                "hetinfer_graph_id": "qwen-1layer-1npu2pim",
                "hetinfer_workload_id": "b1-prefill8-decode1",
            },
            snapshots=[_snapshot()],
        )
        validated = validate_prior_artifact(payload)
        self.assertEqual(validated.payload["graph_id"], "qwen-1layer-1npu2pim")
        self.assertEqual(
            validated.payload["workload_id"], "b1-prefill8-decode1"
        )
        self.assertEqual(
            validated.service_time_s("prefill:1:q", "Ascend_910B_NPU0"),
            0.001,
        )

    def test_native_export_rejects_missing_service_timing(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "service timing is missing"):
            build_prior_artifact(
                cfg={"hetinfer_graph_id": "g", "hetinfer_workload_id": "w"},
                snapshots=[_snapshot(service_s=None)],
            )


if __name__ == "__main__":
    unittest.main()
