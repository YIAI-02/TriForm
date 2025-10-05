from __future__ import annotations
from typing import Dict, List
from dialect import TraceDialect, ARG_DEFAULTS, CFR_BROADCAST, CFR_AFM
from weight_layout import WeightLayout, plan_weight_layout_for_linear

def _mask_for_channel(ch: int) -> int:
    return 1 << max(0, ch)

def _fmt_mac_abk(mask: int, row: int) -> str:
    a = ARG_DEFAULTS["MAC_ABK"]
    return f"AiM MAC_ABK {a['opsize']} 0x{mask:08x} {row}"

def _fmt_mac_sbk(mask: int, bank: int, row: int) -> str:
    a = ARG_DEFAULTS["MAC_SBK"]
    return f"AiM MAC_SBK {a['opsize']} 0x{mask:08x} {bank} {row}"

def _fmt_rd_mac(mask: int) -> str:
    a = ARG_DEFAULTS["RD_MAC"]
    return f"AiM RD_MAC {a['arg0']} 0x{mask:08x}"

def _fmt_wr_bias(mask: int) -> str:
    a = ARG_DEFAULTS["WR_BIAS"]
    return f"AiM WR_BIAS {a['burst']} 0x{mask:08x}"

def _fmt_wr_gb(mask: int) -> str:
    a = ARG_DEFAULTS["WR_GB"]
    return f"AiM WR_GB {a['burst']} {a['banks']} 0x{mask:08x}"

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def emit_mvm(op: Dict, pim: Dict, dialect: TraceDialect, bpe: int,
             weight_layout: WeightLayout | None,
             defer_final_rdmac_to_activation: bool = False) -> List[str]:
    """
    inputs : [B, S, K]
    weights: [K, N]  (physically: N rows × K cols; N across banks; K is row-first then channel)
    outputs: [B, S, N]

    K scheduling:
      - chunk by GB capacity
      - tile by PU throughput, but DO NOT cross a single-channel physical row capacity
    Per tile:
      - first ABK (full groups across banks), then SBK (leftover banks)
      - RD_MAC at tile end; if K not done -> WR_BIAS; if last tile over K -> WR_GB
    Channel selection:
      - derive (channel, row) from (group_idx, rseg) consistent with the weight layout
      - set channel_mask = (1 << channel) for that tile
    """
    if ("AiM MAC_SBK" not in dialect.aim_ops) and ("AiM MAC_ABK" not in dialect.aim_ops):
        return ["# TODO: neither MAC_SBK nor MAC_ABK available for MVM"]

    in_shape  = op["inputs"][0]["shape"]
    out_shape = op["outputs"][0]["shape"]
    B, S, K = int(in_shape[0]), int(in_shape[1]), int(in_shape[2])
    N       = int(out_shape[2])

    cap = pim.get("pim", {}).get("capacity", {})
    pu  = pim.get("pim", {}).get("pu", {})

    banks_per_channel    = int(cap.get("banks_per_channel", 16))
    channels_per_die  = int(cap.get("channels_per_die", 1))
    row_per_bank      = int(cap.get("row_per_bank", 16384))
    column_per_bank   = int(cap.get("column_per_bank", 1024))
    dq_bits           = int(cap.get("dq_width", 16))
    gb_kb             = int(cap.get("global_buffer_KB_per_die", 64))
    mul_per_pu        = int(pu.get("mul_per_pu", 2))
    pus_per_bank      = int(pu.get("pus_per_bank", 2))

    bits_per_elem     = int(bpe) * 8
    k_per_row_per_ch  = max(1, (column_per_bank * dq_bits) // max(1, bits_per_elem))
    rows_per_n_total  = _ceil_div(K, k_per_row_per_ch)

    # chunk/tile along K
    gb_bytes    = max(1, gb_kb) * 1024
    k_per_chunk = max(1, min(K, gb_bytes // max(1, bpe)))
    tile_nom    = max(1, min(K, mul_per_pu * pus_per_bank))

    # ABK groups vs leftover
    full_groups = N // banks_per_channel
    leftover    = N  % banks_per_channel

    if weight_layout is None:
        weight_layout = plan_weight_layout_for_linear([K, N], pim, bpe)

    # Chunk-level WR_GB for the input vector -> make it visible to all channels
    all_channels_mask = (1 << channels_per_die) - 1

    lines: List[str] = []
    if "W CFR" in dialect.rw_ops:
        lines.append(f"W CFR {CFR_BROADCAST} 1")

    for bidx in range(B):
        lines.append(f"# -- batch {bidx} --")
        for sidx in range(S):
            lines.append(f"# -- sequence {sidx} --")

            k_global = 0
            while k_global < K:
                k_chunk_end = min(K, k_global + k_per_chunk)
                k_pos = k_global

                # write current chunk of the input vector
                lines.append(f"# write v chunk (K from {k_global} to {k_chunk_end}) for (b={bidx}, s={sidx})")
                lines.append(_fmt_wr_gb(all_channels_mask))

                while k_pos < k_chunk_end:
                    # rseg: which single-channel physical row this tile is on
                    rseg     = (k_pos // k_per_row_per_ch)
                    row_rem  = k_per_row_per_ch - (k_pos % k_per_row_per_ch)
                    tile_len = min(tile_nom, k_chunk_end - k_pos, row_rem)

                    last_tile_all_k = (k_chunk_end == K) and (k_pos + tile_len == K)

                    # ---------- ABK (full groups) ----------
                    for g in range(full_groups):
                        base_abs = g * rows_per_n_total
                        abs_r    = base_abs + rseg
                        ch       = abs_r // row_per_bank
                        row      = abs_r %  row_per_bank
                        if ch >= channels_per_die:
                            lines.append(f"# ERROR: ABK channel overflow g={g} rseg={rseg} -> ch={ch} (channels={channels_per_die})")
                            continue
                        mask = _mask_for_channel(ch)
                        lines.append(f"# [ABK] g={g}, rseg={rseg}, ch={ch}, row={row}, tile_len={tile_len}")
                        lines.append(_fmt_mac_abk(mask, row))
                        lines.append(_fmt_rd_mac(mask))
                        lines.append(_fmt_wr_gb(all_channels_mask) if last_tile_all_k else _fmt_wr_bias(mask))

                    # ---------- SBK (leftover rows) ----------
                    if leftover > 0:
                        g = full_groups
                        base_abs = g * rows_per_n_total
                        abs_r    = base_abs + rseg
                        ch       = abs_r // row_per_bank
                        row      = abs_r %  row_per_bank
                        if ch >= channels_per_die:
                            lines.append(f"# ERROR: SBK channel overflow rseg={rseg} -> ch={ch} (channels={channels_per_die})")
                        else:
                            mask = _mask_for_channel(ch)
                            for bk in range(leftover):
                                lines.append(f"# [SBK] leftover bank={bk}, rseg={rseg}, ch={ch}, row={row}, tile_len={tile_len}")
                                lines.append(_fmt_mac_sbk(mask, bk, row))
                                lines.append(_fmt_rd_mac(mask))
                                lines.append(_fmt_wr_gb(all_channels_mask) if last_tile_all_k else _fmt_wr_bias(mask))

                    k_pos += tile_len

                k_global = k_chunk_end

    if "AiM EOC" in dialect.aim_ops:
        lines.append("AiM EOC")
    return lines

# Activation kept unchanged (RD_MAC is handled per tile above).
def emit_activation_op(_: Dict, dialect: TraceDialect) -> List[str]:
    lines: List[str] = []
    if "W CFR" in dialect.rw_ops:
        lines.append(f"W CFR {CFR_AFM} {ARG_DEFAULTS['AF']['af_id_fixed']}")
    lines.append(f"AiM AF {ARG_DEFAULTS['AF']['channel_mask']}")
    a_rdaf = ARG_DEFAULTS['RD_AF']
    lines.append(f"AiM RD_AF {a_rdaf['arg0']} {a_rdaf['channel_mask']}")
    return lines
