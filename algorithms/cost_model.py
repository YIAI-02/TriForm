# cost_model.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from hardware import Cluster, DeviceSpec
from config import (
    HOST_NAME, DEVICE_PREFERRED_FORMAT,
    FORMAT_SIZE_MULTIPLIER, FORMAT_CONV_BW_GBs,
    PIM_FORMULA_PATHS, PIM_FREQ_GHZ,
)

DTYPE_BYTES: Dict[str, int] = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "fp8": 1,
}

# =========================
# PIM formula support
# =========================

@dataclass
class OpFormula:
    basis: List[str]
    coeffs: List[float]

    def eval_cycles(
        self,
        *,
        seqlen: int = 0,
        vector_dim: int = 0,
        matrix_col: int = 0,
        n_heads: int = 0,
    ) -> float:
        def val(name: str) -> float:
            if name == "1":    return 1.0
            if name == "L":    return float(seqlen)
            if name == "L2":   return float(seqlen) ** 2
            if name == "H":    return float(n_heads)
            if name == "LxH":  return float(seqlen) * float(n_heads)
            if name == "V":    return float(vector_dim)
            if name == "N":    return float(matrix_col)
            if name == "VxN":  return float(vector_dim) * float(matrix_col)
            return 0.0
        return sum(float(c) * val(b) for b, c in zip(self.basis, self.coeffs))


@dataclass
class PIMFormulaLatency:
    per_op: Dict[str, OpFormula]  # key: lower-case op key

    @staticmethod
    def _from_json_obj(obj: dict) -> "PIMFormulaLatency":
        if "per_op_formula" in obj:
            per = obj["per_op_formula"]
        elif "per_op" in obj:
            per = obj["per_op"]
        else:
            raise ValueError("Unrecognized PIM formula JSON structure")
        per_op: Dict[str, OpFormula] = {}
        for k, v in per.items():
            if isinstance(v, dict) and ("basis" in v) and ("coeffs" in v):
                per_op[k.lower()] = OpFormula(
                    basis=list(v["basis"]),
                    coeffs=[float(x) for x in v["coeffs"]],
                )
        return PIMFormulaLatency(per_op=per_op)

    @classmethod
    def try_load_from_paths(cls, paths: List[str]) -> Optional["PIMFormulaLatency"]:
        for p in paths:
            path = Path(p)
            if path.is_file():
                try:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    return cls._from_json_obj(obj)
                except Exception:
                    continue
        return None

    def get(self, key: str) -> Optional[OpFormula]:
        return self.per_op.get(key.lower())


class CostModel:
    def __init__(self, cluster: Cluster, dtype: str = "fp16"):
        self.cluster = cluster
        self.dtype = dtype
        # 尝试加载 PIM 公式
        self._pim_formula: Optional[PIMFormulaLatency] = PIMFormulaLatency.try_load_from_paths(PIM_FORMULA_PATHS)

    # --------------------------
    # Basic times
    # --------------------------
    def flop_time(self, flops: float, dev: DeviceSpec) -> float:
        if dev.tflops <= 0:
            return 0.0
        return flops / (dev.tflops * 1e12)

    def mem_time(self, bytes_amount: int, dev: DeviceSpec) -> float:
        bw = dev.mem_bw_GBs * 1e9
        return 0.0 if bw <= 0 else bytes_amount / bw

    def link_time(self, bytes_amount: int, src: DeviceSpec, dst: DeviceSpec) -> float:
        bw = self.cluster.get_link_bw(src.name, dst.name) * 1e9
        return 0.0 if bw <= 0 else bytes_amount / bw

    def comm_cost(self, src: DeviceSpec, dst: DeviceSpec, bytes_amount: int) -> float:
        if src.name == dst.name:
            return 0.0
        return self.link_time(bytes_amount, src, dst)

    # --------------------------
    # Format helpers
    # --------------------------
    def get_host_device(self) -> DeviceSpec:
        if HOST_NAME in self.cluster.devices:
            return self.cluster.devices[HOST_NAME]
        cpus = self.cluster.devices_by_type("cpu")
        return cpus[0] if cpus else next(iter(self.cluster.devices.values()))

    def device_preferred_fmt(self, dev: DeviceSpec) -> str:
        return DEVICE_PREFERRED_FORMAT.get(dev.type, "ND")

    def format_size(self, size_bytes: int, fmt: str) -> int:
        m = float(FORMAT_SIZE_MULTIPLIER.get(fmt, 1.0))
        return int(size_bytes * m)

    def format_conversion_time(self, size_src_bytes: int, src_fmt: str, dst_fmt: str, dev: DeviceSpec) -> float:
        if src_fmt == dst_fmt:
            return 0.0
        bw_gbs = float(FORMAT_CONV_BW_GBs.get(dev.type, FORMAT_CONV_BW_GBs.get("default", 50.0)))
        bw = bw_gbs * 1e9
        return 0.0 if bw <= 0 else size_src_bytes / bw

    def gb_move_and_format(self, dev: DeviceSpec, size_src_bytes: int, src_fmt: str, dst_fmt: str) -> float:
        host = self.get_host_device()
        t_move = self.link_time(size_src_bytes, host, dev)
        t_conv = self.format_conversion_time(size_src_bytes, src_fmt, dst_fmt, dev)
        return t_move + t_conv

    # --------------------------
    # PIM op -> formula key / args
    # --------------------------
    def _resolve_pim_key(self, node) -> List[str]:
        """
        - score: QK matmul operations  
        - output: SV matmul operations
        - weight: Linear layers (Q/K/V/O/FFN)
        - weight_af: Linear layers with activation fusion
        """
        keys: List[str] = []
        name = (node.name or "").upper()
        if ("QK" in name) or ("QK_MATMUL" in name) or ("ATTN_QK" in name) or ("SCORE" in name):
            keys.append("score")
        elif ("SV" in name) or ("SV_MATMUL" in name) or ("ATTN_SV" in name) or ("OUTPUT" in name):
            keys.append("output")
        elif name in ("Q", "K", "V", "O", "FFN_W1", "FFN_W2", "FFN_W3"):
            attrs = getattr(node, "attrs", {}) or {}
            has_activation = any(act in str(attrs).upper() for act in ["GELU", "RELU", "SILU", "SWISH"])
            if has_activation:
                keys.append("weight_af")
            else:
                keys.append("weight")
        if not keys:
            keys = ["weight", "score", "output", "weight_af"]
        return keys

    def _infer_vnH_for_node(self, node, seq_len: int) -> Tuple[int, int, int]:
        """
        推断 (V, N, H) 参数：
        V=vector_dim，N=matrix_col（线性层右侧维度），H=多头数（注意力相关）。
        若 attrs 信息不足，尽量使用合理近似；不足则返回 0。
        """
        attrs = getattr(node, "attrs", {}) or {}
        dim = int(attrs.get("dim", 0) or 0)
        ffn_dim = int(attrs.get("ffn_dim", 0) or 0)
        head_dim = int(attrs.get("head_dim", 0) or 0)
        n_heads = int(attrs.get("n_heads", attrs.get("kv_heads", attrs.get("n_kv_heads", 0))) or 0)

        # 常见线性层
        nm = (node.name or "").upper()
        if nm == "Q":
            q_dim = max(0, n_heads * head_dim)
            return dim, q_dim, n_heads
        if nm in ("K", "V"):
            kvh = int(attrs.get("n_kv_heads", attrs.get("kv_heads", n_heads)) or n_heads)
            kv_dim = max(0, kvh * head_dim)
            return dim, kv_dim, kvh
        if nm == "O":
            o_in = max(0, n_heads * head_dim)
            return o_in, dim, n_heads
        if nm in ("FFN_W1", "FFN_W3"):
            return dim, ffn_dim, n_heads
        if nm == "FFN_W2":
            return ffn_dim, dim, n_heads

        # Attention Matmuls / Softmax
        if ("QK" in nm) or ("QK_MATMUL" in nm) or ("ATTN_QK" in nm):
            return head_dim, head_dim, n_heads
        if ("SV" in nm) or ("SV_MATMUL" in nm) or ("ATTN_SV" in nm):
            return head_dim, head_dim, n_heads
        if ("SOFTMAX" in nm):
            # Softmax 的公式一般依赖 L 和 H，V/N 可置为 0 或 head_dim
            return 0, 0, n_heads

        # 其他（LN 等）
        return 0, 0, n_heads

    # --------------------------
    # Dynamic flop estimation
    # --------------------------
    def estimate_flops(self, node, batch: int, seq_len: int, phase: str) -> float:
        attrs = getattr(node, "attrs", {}) or {}
        default = float(getattr(node, "flops", 0.0) or 0.0)

        b = int(batch or 0)
        if b <= 0:
            b = int(attrs.get("batch", 0) or 0)

        dim = int(attrs.get("dim", 0) or 0)
        ffn = int(attrs.get("ffn_dim", 0) or 0)
        qh = int(attrs.get("q_heads", attrs.get("kv_heads", 0)) or 0)
        kvh = int(attrs.get("kv_heads", attrs.get("n_kv_heads", 0)) or 0)
        hd = int(attrs.get("head_dim", 0) or 0)
        q_dim = int(attrs.get("q_dim", qh * hd) or 0)
        kv_dim = int(attrs.get("kv_dim", kvh * hd) or 0)
        o_dim = int(attrs.get("o_dim", qh * hd) or 0)

        if b <= 0:
            return default

        token_len = seq_len if phase == "prefill" else 1
        kv_len = seq_len

        name = (node.name or "").upper()

        if name == "LN" and dim > 0:
            return float(b * token_len * dim)

        if name in ("Q", "K", "V") and dim > 0:
            out_dim = q_dim if name == "Q" else kv_dim
            if out_dim <= 0:
                return default
            return float(2.0 * dim * out_dim * b * token_len)

        if name == "QK" and qh > 0 and hd > 0:
            if phase == "prefill":
                return float(2.0 * b * qh * hd * token_len * token_len)
            return float(2.0 * b * qh * hd * kv_len)

        if name == "SOFTMAX" and qh > 0:
            if phase == "prefill":
                return float(b * qh * token_len * token_len)
            return float(b * qh * kv_len)

        if name == "SV" and qh > 0 and hd > 0:
            if phase == "prefill":
                return float(2.0 * b * qh * hd * token_len * token_len)
            return float(2.0 * b * qh * hd * kv_len)

        if name == "O" and dim > 0 and o_dim > 0:
            return float(2.0 * o_dim * dim * b * token_len)

        if name in ("FFN_W1", "FFN_W3") and dim > 0 and ffn > 0:
            return float(2.0 * dim * ffn * b * token_len)

        if name == "FFN_W2" and dim > 0 and ffn > 0:
            return float(2.0 * ffn * dim * b * token_len)

        if name in ("SWIGLU", "GELU") and ffn > 0:
            return float(b * ffn * token_len)

        if name == "ADD" and dim > 0:
            return float(b * dim * token_len)

        if name == "IDENTITY" and dim > 0:
            return float(b * dim * token_len)

        if name in ("KV_READ", "KV_WRITE"):
            return 0.0

        return default

    # --------------------------
    # Node device cost
    # --------------------------
    def node_device_cost(self, node, dev: DeviceSpec, batch: int, seq_len: int, phase: str) -> float:
        """
        Compute time on a device. For PIM, if a formula JSON is provided, use it;
        otherwise fall back to (flops + mem) model. NPU uses (flops + mem).
        """
        # Special KV ops (decode)
        if phase == "decode" and node.name in ("KV_read", "KV_write"):
            r, w = self.kv_rw_bytes_decode(node, batch, seq_len)
            return self.mem_time(r + w, dev)

        # NPU: baseline model
        if dev.type == "npu":
            flops = self.estimate_flops(node, batch, seq_len, phase)
            rd = int(getattr(node, "bytes_read", 0) or 0)
            wr = int(getattr(node, "bytes_write", 0) or 0)
            return self.flop_time(flops, dev) + self.mem_time(rd + wr, dev)

        # PIM: try formula-based estimation
        if dev.type == "pim" and self._pim_formula is not None:
            # 解析 op key
            keys = self._resolve_pim_key(node)
            opf: Optional[OpFormula] = None
            matched_key = None
            for k in keys:
                opf = self._pim_formula.get(k)
                if opf is not None:
                    matched_key = k
                    break
                    
            if opf is not None:
                V, N, H = self._infer_vnH_for_node(node, seq_len)
                cycles = opf.eval_cycles(seqlen=seq_len, vector_dim=V, matrix_col=N, n_heads=H)
                if cycles > 0.0 and PIM_FREQ_GHZ > 0.0:
                    return cycles / (PIM_FREQ_GHZ * 1e9)


        # Other / fallback: baseline (flops + mem)
        flops = self.estimate_flops(node, batch, seq_len, phase)
        rd = int(getattr(node, "bytes_read", 0) or 0)
        wr = int(getattr(node, "bytes_write", 0) or 0)
        return self.flop_time(flops, dev) + self.mem_time(rd + wr, dev)

    # --------------------------
    # KV read/write for decode
    # --------------------------
    def kv_rw_bytes_decode(self, node, batch: int, seq_len: int) -> Tuple[int, int]:
        """
        Estimate KV read/write bytes in decode.
        Prefer node.attrs if provided: expect keys 'n_kv_heads', 'head_dim'.
        """
        attrs = getattr(node, "attrs", {}) or {}
        n_kv = int(attrs.get("n_kv_heads", attrs.get("kv_heads", 0)) or 0)
        head_dim = int(attrs.get("head_dim", 0) or 0)
        dtype_b = int(DTYPE_BYTES.get(self.dtype, 2))

        if n_kv <= 0 or head_dim <= 0 or batch <= 0:
            return 0, 0

        # K and V per token vector size: n_kv * head_dim
        # Read: full seq_len history; Write: current token
        read_elems = 2 * batch * n_kv * head_dim * seq_len  # K+V
        write_elems = 2 * batch * n_kv * head_dim           # append 1 token
        return read_elems * dtype_b, write_elems * dtype_b

    