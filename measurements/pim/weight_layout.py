from __future__ import annotations
from dataclasses import dataclass
from typing import List
from dialect import TraceDialect

@dataclass
class WeightSeg:
    """A contiguous K-slice placed at a single (channel, bank, row)."""
    channel: int
    bank:    int
    row:     int      # physical row index inside the bank
    cols:    int      # number of K elements stored in this segment

@dataclass
class WeightRowLayout:
    """One logical N-row (in transposed storage) mapped to a bank."""
    bank: int                 # which bank this logical N-row belongs to
    row:  int                 # base physical row index (for ABK alignment)
    base_channel: int         # base channel index where this N-row starts
    segs: List[WeightSeg]     # row-first, then channel

@dataclass
class WeightLayout:
    """Layout for the whole weight matrix, with capacity meta for scheduling."""
    rows: List[WeightRowLayout]
    banks_per_channel: int
    channels_per_die: int
    row_per_bank: int
    k_elems_per_row_per_ch: int
    rows_per_n_total: int      # total single-channel rows per logical N-row (across channels)

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def plan_weight_layout_for_linear(shape_k_n: list[int], pim: dict, bpe: int) -> WeightLayout:
    """
    Transposed storage: N across banks; K along rows in a **row-first, then channel** manner.

    Rules for one logical N-row (fixed bank):
      1) per-row capacity per channel:
           k_per_row_per_ch = floor(column_per_bank * dq_width / bits_per_elem)
      2) rows needed (in single-channel rows):
           rows_per_n_total = ceil(K / k_per_row_per_ch)
      3) ABK alignment across groups g = floor(n / banks_per_channel):
           base_abs     = g * rows_per_n_total        # base offset in single-channel rows
           base_channel = base_abs // row_per_bank
           base_row     = base_abs %  row_per_bank
         For seg_id = 0..rows_per_n_total-1:
           abs_r = base_abs + seg_id
           ch    = abs_r // row_per_bank
           row   = abs_r %  row_per_bank
    """
    K, N = int(shape_k_n[0]), int(shape_k_n[1])

    cap = pim.get("pim", {}).get("capacity", {})
    banks_per_channel    = int(cap.get("banks_per_channel", 16))
    channels_per_die  = int(cap.get("channels_per_die", 1))
    row_per_bank      = int(cap.get("row_per_bank", 16384))
    column_per_bank   = int(cap.get("column_per_bank", 1024))
    dq_bits           = int(cap.get("dq_width", 16))

    bits_per_elem           = int(bpe) * 8
    k_per_row_per_ch        = max(1, (column_per_bank * dq_bits) // max(1, bits_per_elem))
    rows_per_n_total        = _ceil_div(K, k_per_row_per_ch)

    rows: List[WeightRowLayout] = []

    for n in range(N):
        bank       = n % banks_per_channel
        group_idx  = n // banks_per_channel         # ABK group index (across banks)
        base_abs   = group_idx * rows_per_n_total
        base_ch    = base_abs // row_per_bank
        base_row   = base_abs %  row_per_bank

        if base_ch >= channels_per_die:
            raise ValueError(
                f"[weight_layout] Not enough channels: base_channel={base_ch} "
                f"for group={group_idx}, rows_per_n_total={rows_per_n_total}, "
                f"row_per_bank={row_per_bank}, channels={channels_per_die}"
            )

        segs: List[WeightSeg] = []
        k_left, seg_id = K, 0
        while k_left > 0:
            take = min(k_per_row_per_ch, k_left)
            abs_r = base_abs + seg_id
            ch    = abs_r // row_per_bank
            row   = abs_r %  row_per_bank
            if ch >= channels_per_die:
                raise ValueError(
                    f"[weight_layout] Channel overflow while laying N={n}: ch={ch} "
                    f"(rows_per_n_total={rows_per_n_total}, seg_id={seg_id})"
                )
            segs.append(WeightSeg(channel=ch, bank=bank, row=row, cols=take))
            k_left -= take
            seg_id += 1

        rows.append(WeightRowLayout(bank=bank, row=base_row, base_channel=base_ch, segs=segs))

    return WeightLayout(
        rows=rows,
        banks_per_channel=banks_per_channel,
        channels_per_die=channels_per_die,
        row_per_bank=row_per_bank,
        k_elems_per_row_per_ch=k_per_row_per_ch,
        rows_per_n_total=rows_per_n_total,
    )

def emit_weight_write_trace(layout: WeightLayout, dialect: TraceDialect) -> List[str]:
    """
    Convert the layout into a write trace:
      - each seg -> `W MEM [channel] [bank] [row]`
    """
    lines: List[str] = [
        "# --- weight write trace (row-first then channel) ---",
        f"# banks={layout.banks_per_channel}, channels={layout.channels_per_die}, "
        f"rows/bank={layout.row_per_bank}, k_per_row_per_ch={layout.k_elems_per_row_per_ch}, "
        f"rows_per_n_total={layout.rows_per_n_total}",
    ]
    for n, row in enumerate(layout.rows):
        for s, seg in enumerate(row.segs):
            lines.append(f"# n={n} seg={s}: ch={seg.channel}, bk={seg.bank}, row={seg.row}, K_elems={seg.cols}")
            if 'W MEM' in dialect.rw_ops:
                lines.append(f"W MEM {seg.channel} {seg.bank} {seg.row}")
            else:
                lines.append("# TODO: W MEM not available in dialect")
    return lines
