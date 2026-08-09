"""Statically validate an exported calibrated DOPS config/hardware/runtime trio."""

from __future__ import annotations

import argparse
from pathlib import Path

from hardware import demo_cluster
from mainlib.cli import _apply_runtime_config_overrides, _load_cfg_from_json
from model_parser import build_graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = _load_cfg_from_json(str(args.config))
    if cfg.get("npu_backend") != "fast":
        raise ValueError("calibrated config must select npu_backend=fast")
    applied = _apply_runtime_config_overrides(cfg)
    if "GPU_RUNTIME_MODEL" not in applied:
        raise ValueError("calibrated config did not apply a GPU runtime model")
    cluster = demo_cluster(cfg)
    gpu = cluster.devices.get("GPU0")
    if gpu is None or gpu.type != "npu":
        raise ValueError("calibrated hardware must contain GPU0 with DOPS type=npu")
    if gpu.tflops <= 0.0 or gpu.mem_bw_GBs <= 0.0 or gpu.mem_capacity_GB <= 0.0:
        raise ValueError("calibrated GPU0 hardware parameters must be positive")
    graph, _ = build_graph(cfg)
    print(
        "[GPU-EXPORT-VALID] "
        f"config={args.config} nodes={len(graph.nodes)} "
        f"tflops={gpu.tflops:.6g} mem_bw_GiB_s={gpu.mem_bw_GBs:.6g} "
        f"runtime_prefix={applied['GPU_RUNTIME_MODEL']['device_name_prefix']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
