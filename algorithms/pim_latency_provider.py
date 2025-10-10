from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class PIMLatencyProvider:
    # per-op GFLOPS (or per-flop throughput) for matmul-like ops
    gflops: Dict[str, float] = field(default_factory=dict)
    # fallback matmul gflops
    default_gflops: float = 20.0  # example

    @classmethod
    def from_config(cls, cfg: dict) -> "PIMLatencyProvider":
        prov = cls()
        table = cfg.get("pim_latency_table", {})
        prov.gflops = {k: float(v) for k, v in table.items()}
        prov.default_gflops = float(cfg.get("pim_default_gflops", 20.0))
        return prov

    def matmul_time(self, op_name: str, flops: float) -> float:
        gflops = self.gflops.get(op_name, self.default_gflops)
        if gflops <= 0: gflops = self.default_gflops
        return flops / (gflops * 1e9)
