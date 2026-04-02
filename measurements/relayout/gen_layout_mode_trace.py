#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

PRESETS = {
    "demo": (512, 512),
    "llama3_proj": (4096, 4096),
    "llama3_ffn_up": (4096, 14336),
    "llama3_ffn_down": (14336, 4096),
}


def channel_to_mask(channel: int, total_channels: int = 32) -> str:
    if not (0 <= channel < total_channels):
        raise ValueError(f"channel out of range: {channel}")
    return hex(1 << (total_channels - 1 - channel))


class TraceConfig:
    def __init__(
        self,
        rows: int,
        cols: int,
        dtype_bytes: int = 2,
        burst_bytes: int = 32,
        channels: int = 32,
        banks: int = 16,
        gpr_count: int = 16,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.dtype_bytes = dtype_bytes
        self.burst_bytes = burst_bytes
        self.channels = channels
        self.banks = banks
        self.gpr_count = gpr_count
        self.matrix_bytes = rows * cols * dtype_bytes
        if self.matrix_bytes % self.burst_bytes != 0:
            raise ValueError("matrix size must be divisible by burst size")
        self.num_bursts = self.matrix_bytes // self.burst_bytes


def linear_order(cfg: TraceConfig):
    for burst_idx in range(cfg.num_bursts):
        ch = burst_idx % cfg.channels
        bank = (burst_idx // cfg.channels) % cfg.banks
        row = burst_idx // (cfg.channels * cfg.banks)
        yield burst_idx, ch, bank, row


def pimopt_order(cfg: TraceConfig):
    # Approximation of a bank-striped PIM-friendly order.
    # The paper's PIM-friendly layout wants chunks contiguous inside a bank and
    # banks of a channel aligned for lock-step operation. This trace format
    # cannot express intra-row column bits for host MEM requests, so we model
    # PIM-OPT at the bank/channel traversal level by making bank the fastest-
    # changing dimension and then channel. This is intentionally approximate.
    for burst_idx in range(cfg.num_bursts):
        bank = burst_idx % cfg.banks
        ch = (burst_idx // cfg.banks) % cfg.channels
        row = burst_idx // (cfg.channels * cfg.banks)
        yield burst_idx, ch, bank, row


def generate_trace(cfg: TraceConfig, out_path: Path, mode: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        w = f.write
        w(f"# mode = {mode}\n")
        w(f"# matrix = {cfg.rows} x {cfg.cols}, dtype_bytes = {cfg.dtype_bytes}\n")
        w(f"# matrix_bytes = {cfg.matrix_bytes}, bursts = {cfg.num_bursts}, burst_bytes = {cfg.burst_bytes}\n")

        for gpr in range(cfg.gpr_count):
            w(f"W GPR {gpr}\n")

        if mode == "linear_to_pimopt":
            w("\n# Phase 1: linear read via host MEM\n")
            for _, ch, bank, row in linear_order(cfg):
                w(f"R MEM {ch} {bank} {row}\n")

            w("\n# Phase 2: PIM-OPT write via all-bank AiM write\n")
            for burst_idx, _, _, _ in linear_order(cfg):
                ch = burst_idx % cfg.channels
                row = burst_idx // cfg.channels
                gpr = burst_idx % cfg.gpr_count
                w(f"AiM WR_ABK {gpr} {channel_to_mask(ch, cfg.channels)} {row}\n")

        elif mode == "pimopt_rw_host":
            w("\n# Phase 1: PIM-OPT write via host MEM in bank-striped order\n")
            for _, ch, bank, row in pimopt_order(cfg):
                w(f"W MEM {ch} {bank} {row}\n")

            w("\n# Phase 2: PIM-OPT read via host MEM in the same bank-striped order\n")
            for _, ch, bank, row in pimopt_order(cfg):
                w(f"R MEM {ch} {bank} {row}\n")

        elif mode == "pimopt_rw_dma":
            w("\n# Phase 1: PIM-OPT write via WR_ABK\n")
            for burst_idx in range(cfg.num_bursts):
                ch = burst_idx % cfg.channels
                row = burst_idx // cfg.channels
                gpr = burst_idx % cfg.gpr_count
                w(f"AiM WR_ABK {gpr} {channel_to_mask(ch, cfg.channels)} {row}\n")

            w("\n# Phase 2: PIM-OPT read via RD_SBK in bank-striped order\n")
            for burst_idx, ch, bank, row in pimopt_order(cfg):
                gpr = burst_idx % cfg.gpr_count
                w(f"AiM RD_SBK {gpr} {channel_to_mask(ch, cfg.channels)} {bank} {row}\n")

        else:
            raise ValueError(f"unsupported mode: {mode}")

        w("\nAiM EOC\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["linear_to_pimopt", "pimopt_rw_host", "pimopt_rw_dma"], required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), default="demo")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--burst-bytes", type=int, default=32)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--banks", type=int, default=16)
    ap.add_argument("--gpr-count", type=int, default=16)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows, cols = PRESETS[args.preset]
    if args.rows is not None and args.cols is not None:
        rows, cols = args.rows, args.cols

    cfg = TraceConfig(
        rows=rows,
        cols=cols,
        dtype_bytes=args.dtype_bytes,
        burst_bytes=args.burst_bytes,
        channels=args.channels,
        banks=args.banks,
        gpr_count=args.gpr_count,
    )
    generate_trace(cfg, args.output, args.mode)
    print(args.output)


if __name__ == "__main__":
    main()
