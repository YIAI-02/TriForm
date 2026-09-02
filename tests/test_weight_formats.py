from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mainlib.weight_formats import _build_weight_blocks


class WeightBlockGroupingTests(unittest.TestCase):
    def test_layer_block_grouping_is_case_insensitive(self) -> None:
        self.assertEqual(
            _build_weight_blocks(
                ["L0_WQ", "L7_WQ", "L8_WQ", "l9_WQ_s0"],
                layer_span=8,
            ),
            {
                "L0000-0007_WQ": ["L0_WQ", "L7_WQ"],
                "L0008-0015_WQ": ["L8_WQ", "l9_WQ_s0"],
            },
        )

    def test_non_layer_weight_id_remains_usable(self) -> None:
        self.assertEqual(
            _build_weight_blocks(["embedding_s0"], layer_span=8),
            {"embedding": ["embedding_s0"]},
        )


if __name__ == "__main__":
    unittest.main()
