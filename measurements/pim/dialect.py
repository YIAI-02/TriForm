from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass
class TraceDialect:
    aim_ops: set
    rw_ops: set

def load_trace_dialect(trace_path: str | Path) -> TraceDialect:
    txt = Path(trace_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    aim_ops, rw_ops = set(), set()
    for s in (ln.strip() for ln in txt):
        if not s or s.startswith("#"):
            continue
        m_ctrl = re.match(r'^(W|R)\s+([A-Z]+)\b', s)
        if m_ctrl:
            rw_ops.add(m_ctrl.group(0).split()[0] + " " + m_ctrl.group(0).split()[1])
        m_aim = re.match(r'^AiM\s+([A-Za-z0-9_]+)', s)
        if m_aim:
            aim_ops.add("AiM " + m_aim.group(1))
    return TraceDialect(aim_ops=aim_ops, rw_ops=rw_ops)

CFR_BROADCAST = 0  
CFR_EWMUL_BG  = 1  
CFR_AFM       = 2  

ARG_DEFAULTS = {
    "EWMUL":   {"opsize": 2, "channel_mask": 0xffffffff, "row": 0}, 
    "MAC_ABK": {"opsize": 2, "channel_mask": 0xffffffff, "row": 0},       
    "MAC_SBK": {"opsize": 2, "channel_mask": 0xffffffff, "bank": 0, "row": 0},
    "RD_MAC":  {"arg0": 8,  "channel_mask": 0xffffffff},                  
    "WR_BIAS": {"burst": 4, "channel_mask": 0xffffffff},                  
    "WR_GB":   {"burst": 2, "banks": 2, "channel_mask": 0xffffffff},      
    "AF":      {"channel_mask": 0xffffffff, "af_id_fixed": 1},            
    "RD_AF":   {"arg0": 12, "channel_mask": 0xffffffff},                  
}

def bytes_per_elem_from_pim(pim: dict, fallback: int = 2) -> int:
    fmt = pim.get("pim", {}).get("pu", {}).get("support_data_format", "").upper()
    if fmt in ("FP16", "BF16"): return 2
    if fmt in ("FP32",):        return 4
    if fmt in ("FP8", "E4M3", "E5M2"): return 1
    return fallback

def emit_wr_gb(dialect: TraceDialect) -> str:
    a = ARG_DEFAULTS["WR_GB"]
    return f"AiM WR_GB {a['burst']} {a['banks']} 0x{a['channel_mask']:08x}"

def emit_rd_mac(dialect: TraceDialect) -> str:
    a = ARG_DEFAULTS["RD_MAC"]
    return f"AiM RD_MAC {a['arg0']} 0x{a['channel_mask']:08x}"

def emit_wr_bias(dialect: TraceDialect) -> str:
    a = ARG_DEFAULTS["WR_BIAS"]
    return f"AiM WR_BIAS {a['burst']} 0x{a['channel_mask']:08x}"
