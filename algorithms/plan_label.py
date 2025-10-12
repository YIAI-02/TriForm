from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


@dataclass
class PlanLabel:
    """Memory planning decision shared across modules."""

    pim_mode: str  # "small" | "medium" | "large"
    kv_in_pim: bool
    pim_weight_capacity_bytes: int = 0
    pinned_fc_on_pim: Set[str] = field(default_factory=set)

    def print_debug(self) -> None:
        """Print all PlanLabel settings for debugging."""

        print("=" * 50)
        print("PIM MEMORY PLAN DEBUG INFO")
        print("=" * 50)
        print(f"PIM Mode: {self.pim_mode}")
        print(f"KV Cache in PIM: {self.kv_in_pim}")
        print(f"Number of Pinned FC Weights: {len(self.pinned_fc_on_pim)}")
        if self.pinned_fc_on_pim:
            print("Pinned FC Weights:")
            for weight_id in sorted(self.pinned_fc_on_pim):
                print(f"  - {weight_id}")
        else:
            print("Pinned FC Weights: None")
        print("=" * 50)