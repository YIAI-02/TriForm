# cost_model.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from task_graph import TaskGraph, TaskNode
from hardware import Cluster, DeviceSpec
from plan_label import PlanLabel
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
        bw_gbs = float(FORMAT_CONV_BW_GBs.get(dev.type, FORMAT_CONV_BW_GBs.get("default", 50.0))) #优先查找dev.type的，如果也没有就查找default，如果都没有就用默认
        bw = bw_gbs * 1e9
        return 0.0 if bw <= 0 else size_src_bytes / bw

    def gb_move_and_format(self, dev: DeviceSpec, size_src_bytes: int, src_fmt: str, dst_fmt: str) -> float:
        host = self.get_host_device()
        t_move = self.link_time(size_src_bytes, host, dev)
        t_conv = self.format_conversion_time(size_src_bytes, src_fmt, dst_fmt, dev)
        return max(t_move,t_conv)

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
        V=vector_dim, N=matrix_col（线性层右侧维度），H=多头数（注意力相关）。
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

        b = int(batch or attrs.get("batch", 0) or 0)
        if b <= 0:
            return default

        # 维度
        D   = int(attrs.get("dim", 0) or 0)                       # model dim
        Hf  = int(attrs.get("ffn_dim", attrs.get("hidden_dim", 0)) or 0)
        qh  = int(attrs.get("q_heads", attrs.get("n_head", attrs.get("kv_heads", 0))) or 0)
        kvh = int(attrs.get("kv_heads", attrs.get("n_kv_heads", qh)) or 0)
        hd  = int(attrs.get("head_dim", D // max(qh, 1)) or 0)

        q_dim = int(attrs.get("q_dim", qh * hd) or 0)
        kv_dim = int(attrs.get("kv_dim", kvh * hd) or 0)
        o_dim  = int(attrs.get("o_dim", qh * hd) or 0)
        q_len  = seq_len if phase == "prefill" else 1
        kv_len = int(attrs.get("kv_len", attrs.get("past_kv_len", seq_len)) or seq_len)
        causal = bool(attrs.get("causal", True))  # 因果注意力
        def tri(n: int) -> int:
            return n * (n + 1) // 2

        C_MATMUL  = 2.0   # matmul: MAC -> 2 FLOPs
        C_LN      = 5.0   # LN: 均值、方差、归一化、仿射
        C_SOFTMAX = 5.0   # softmax: max/sub/exp/sum/div（不细算exp代价）
        C_GELU    = 6.0   # GELU 近似
        C_SILU    = 5.0   # SiLU(x)=x*sigmoid(x) 近似

        name = (getattr(node, "name", "") or "").upper()

        if name in ("LN") and D > 0:
            return float(b * q_len * D * C_LN)

        if name in ("Q", "K", "V") and D > 0:
            out_dim = q_dim if name == "Q" else kv_dim
            if out_dim <= 0:
                return default
            return float(C_MATMUL * D * out_dim * b * q_len)

        if name in ("QK") and qh > 0 and hd > 0:
            if phase == "prefill":
                pairs = tri(q_len) if causal else q_len * q_len
            else:
                pairs = kv_len  # q_len==1
            return float(C_MATMUL * b * qh * hd * pairs)

        if name in ("SOFTMAX") and qh > 0:
            if phase == "prefill":
                elems = tri(q_len) if causal else q_len * q_len
            else:
                elems = kv_len
            return float(b * qh * elems * C_SOFTMAX)

        if name in ("SV") and qh > 0 and hd > 0:
            if phase == "prefill":
                pairs = tri(q_len) if causal else q_len * q_len
            else:
                pairs = kv_len
            return float(C_MATMUL * b * qh * hd * pairs)

        if name in ("O") and D > 0 and o_dim > 0:
            return float(C_MATMUL * o_dim * D * b * q_len)

        if name in ("FFN_W1", "FFN_W3", "FFN_UP", "FFN_GATE") and D > 0 and Hf > 0:
            # SwiGLU 的 W1/W3（或门控/上投影）
            return float(C_MATMUL * D * Hf * b * q_len)

        if name in ("FFN_W2", "FFN_DOWN") and D > 0 and Hf > 0:
            return float(C_MATMUL * Hf * D * b * q_len)

        if name in ("SWIGLU", "SILU_GLU") and Hf > 0:
            # SiLU(Hf) + 与另一条支路做逐元素乘（门控）
            return float(b * q_len * Hf * (C_SILU + 1.0))

        if name in ("GELU",) and Hf > 0:
            return float(b * q_len * Hf * C_GELU)

        if name == "ADD" and D > 0:
            return float(b * q_len * D)  # 残差加

        if name in ("IDENTITY", "RESIDUAL", "DROPOUT") and D > 0:
            return float(b * q_len * D)

        if name in ("KV_READ", "KV_WRITE", "ROPE", "ALIBI"):
            return 0.0  # 仅内存/索引或轻量计算，这里不计入 FLOPs

        return default

    def estimate_activation_bytes(self, node, batch: int, seq_len: int, phase: str):

        attrs = getattr(node, "attrs", {}) or {}
        dtype_bytes = int(DTYPE_BYTES.get(self.dtype, 2))

        def to_bytes(elems: float) -> int:
            return int(max(0.0, float(elems))) * dtype_bytes

        b = int(batch or attrs.get("batch", 0) or 1)

        T = int(seq_len or 0)
        if T <= 0:
            return 0, 0

        kv_len = int(attrs.get("kv_len", attrs.get("past_kv_len", T)) or T)
        active_tokens  = T if phase == "prefill" else 1

        causal = bool(attrs.get("causal", True))
        def tri(n: int) -> int:
            return n * (n + 1) // 2

        D       = int(attrs.get("dim", attrs.get("hidden_size", 0)) or 0)
        Hf      = int(attrs.get("ffn_dim", attrs.get("mlp_dim", 0)) or 0)
        hd      = int(attrs.get("head_dim", 0) or 0)
        qh      = int(attrs.get("q_heads", attrs.get("n_heads", 0)) or 0)
        kvh     = int(attrs.get("n_kv_heads", attrs.get("kv_heads", qh)) or 0)
        q_dim   = int(attrs.get("q_dim", qh * hd) or 0)
        kv_dim  = int(attrs.get("kv_dim", kvh * hd) or 0)
        o_dim   = int(attrs.get("o_dim", qh * hd) or 0)

        name = (getattr(node, "name", attrs.get("op", "")) or "").upper()

        if phase == "prefill":
            attn_pairs = tri(T) if causal else T * T #是否有mask
        else:
            attn_pairs = kv_len

        if name in ("LN") and D > 0:
            elems = b * active_tokens * D
            return to_bytes(elems), to_bytes(elems)

        if name == "Q" and D > 0:
            out_dim = q_dim if q_dim > 0 else D
            return to_bytes(b * active_tokens * D), to_bytes(b * active_tokens * out_dim)

        if name in ("K", "V") and D > 0:
            out_dim = kv_dim if kv_dim > 0 else D
            write_tokens = active_tokens  # prefill=T, decode=1
            return to_bytes(b * active_tokens * D), to_bytes(b * write_tokens * out_dim)

        if name in ("O") and D > 0:
            inp_dim = o_dim if o_dim > 0 else D
            return to_bytes(b * active_tokens * inp_dim), to_bytes(b * active_tokens * D)

        if name in ("FFN_W1", "FFN_W3") and D > 0 and Hf > 0:
            return to_bytes(b * active_tokens * D), to_bytes(b * active_tokens * Hf)

        if name in ("FFN_W2",) and D > 0 and Hf > 0:
            return to_bytes(b * active_tokens * Hf), to_bytes(b * active_tokens * D)

        if name in ("SWIGLU", "SILU_GLU") and Hf > 0:
            # 读 gate(Hf) 与 up(Hf)，写 Hf
            return to_bytes(b * active_tokens * (2 * Hf)), to_bytes(b * active_tokens * Hf)

        if name in ("GELU", "RELU"):
            width = Hf if Hf > 0 else D
            return to_bytes(b * active_tokens * width), to_bytes(b * active_tokens * width)

        if name == "ADD" and D > 0:
            # 读两路，写一路
            read_elems = b * active_tokens * D * 2
            write_elems = b * active_tokens * D
            return to_bytes(read_elems), to_bytes(write_elems)

        if name in ("IDENTITY",):
            elems = b * active_tokens * D
            return to_bytes(elems), to_bytes(elems)

        if name in ("QK") and qh > 0 and hd > 0:
            q_read = b * active_tokens * q_dim
            k_read = b * (T if phase == "prefill" else kv_len) * kv_dim
            # 写出：每个 head 上的有效元素（考虑三角）
            write_elems = b * qh * attn_pairs
            return to_bytes(q_read + k_read), to_bytes(write_elems)

        if name in ("SOFTMAX", "ATTN_SOFTMAX") and qh > 0:
            # 读/写形状相同（常见实现原地）
            elems = b * qh * attn_pairs
            return to_bytes(elems), to_bytes(elems)

        if name in ("SV") and qh > 0 and hd > 0:
            # 读：注意力权重 + V；写：b * qh * q_len * hd
            attn_read = b * qh * attn_pairs
            v_read    = b * (T if phase == "prefill" else kv_len) * kv_dim
            out_elems = b * qh * q_dim * hd
            # 注：这里按每个 group 共享 V，不乘 qh/kvh，作为一次性读取的上界
            return to_bytes(attn_read + v_read), to_bytes(out_elems)

        if name in ("KV_READ", "KV_WRITE"):
            read = 2 * batch * kvh * hd * kv_len
            write = 2 * batch * kvh * hd * active_tokens
            return read, write

        if D > 0:
            elems = b * active_tokens * D
            return to_bytes(elems), to_bytes(elems)

        return 0, 0


    # --------------------------
    # Node device cost
    # --------------------------
    def node_device_cost(self, node:TaskNode, dev: DeviceSpec, label:PlanLabel,batch: int, seq_len: int, phase: str) -> float:
        # NPU:
        if dev.type == "npu":
            flops = self.estimate_flops(node, batch, seq_len, phase)
            rd, wr = self.estimate_activation_bytes(node,batch,seq_len,phase)
            return max(self.flop_time(flops, dev), self.mem_time(rd + wr, dev))

        # PIM:
        if dev.type == "pim" and self._pim_formula is not None:
            keys = self._resolve_pim_key(node)
            opf: Optional[OpFormula] = None
            for k in keys:
                opf = self._pim_formula.get(k)
                if opf is not None:
                    V, N, H = self._infer_vnH_for_node(node, seq_len)
                    cycles = opf.eval_cycles(seqlen=seq_len, vector_dim=V, matrix_col=N, n_heads=H)
                    if cycles > 0.0 and PIM_FREQ_GHZ > 0.0:
                        compute_time = cycles / (PIM_FREQ_GHZ * 1e9)
        
            kv_in_pim = getattr(label, "kv_in_lable", False)
            if kv_in_pim:
                mem_time = 0.0
            else:
                rd, wr = self.estimate_activation_bytes(node,batch,seq_len,phase)
                mem_time = self.mem_time((rd + wr),dev)
            return compute_time + mem_time

    