from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from task_graph import TaskGraph, TaskNode
from hardware import HardwareManager, Device
from buffer_manager import BufferManager
from cost_model import CostModel
from config import DEVICE_PREFERRED_FORMAT, RANK_COST_POLICY, ALLOW_HYBRID, HYBRID_SPEEDUP, ENABLE_WEIGHT_PREFETCH, OBJECTIVE, BALANCED_ALPHA, HYBRID_GATE_BY_DIFF, HYBRID_RELATIVE_DIFF, HYBRID_ABSOLUTE_MARGIN, HYSTERESIS_ENABLE, HYST_REL_ENTER, HYST_ABS_ENTER, HYST_REL_EXIT, HYST_ABS_EXIT, OP_PLACEMENT_CONSTRAINTS, NODE_PLACEMENT_CONSTRAINTS

@dataclass
class Assignment:
    node_id: str
    state: str                # "NPU"|"PIM"|"HYBRID"
    devices: Tuple[Optional[int], Optional[int]]  # (npu_id or None, pim_id or None)
    start_time: float
    finish_time: float
    output_format: str
    output_device: str

class HeftScheduler:

    def _resolve_placement_constraints(self, node: TaskNode) -> Tuple[bool, bool, bool]:
        allow_npu, allow_pim = node.allowed
        op_name = node.attrs.get("op") or getattr(node, "name", None) or ""
        nid = node.id

        # by op
        cfg_op = OP_PLACEMENT_CONSTRAINTS.get(op_name, {})
        if "allow_npu" in cfg_op: allow_npu = bool(cfg_op["allow_npu"])
        if "allow_pim" in cfg_op: allow_pim = bool(cfg_op["allow_pim"])
        allow_hybrid = bool(cfg_op.get("allow_hybrid", True))

        # by node id (override)
        cfg_node = NODE_PLACEMENT_CONSTRAINTS.get(nid, {})
        if "allow_npu" in cfg_node: allow_npu = bool(cfg_node["allow_npu"])
        if "allow_pim" in cfg_node: allow_pim = bool(cfg_node["allow_pim"])
        if "allow_hybrid" in cfg_node: allow_hybrid = bool(cfg_node["allow_hybrid"])

        if not allow_npu or not allow_pim:
            allow_hybrid = False
        return allow_npu, allow_pim, allow_hybrid
    
    def __init__(self, cm: CostModel, hw: HardwareManager, buf: BufferManager) -> None:
        self.cm = cm
        self.hw = hw
        self.buf = buf
        self.assignments: Dict[str, Assignment] = {}
        self.mode_mem: Dict[str, str] = {}

    # ---------- rank-u ----------
    def compute_rank_u(self, g: TaskGraph) -> Dict[str, float]:
        memo: Dict[str, float] = {}

        def avg_comp(n: TaskNode) -> float:
            ts = []
            if n.allowed[0]: ts.append(self.cm.compute_time(n.work, "NPU", n.attrs.get("op")))
            if n.allowed[1]: ts.append(self.cm.compute_time(n.work, "PIM", n.attrs.get("op")))
            if not ts: raise ValueError(f"Node {n.id} has no allowed devices")
            if RANK_COST_POLICY == "avg": return sum(ts)/len(ts)
            if RANK_COST_POLICY == "best": return min(ts)
            if RANK_COST_POLICY == "npu_only": return self.cm.compute_time(n.work, "NPU", n.attrs.get("op"))
            if RANK_COST_POLICY == "pim_only": return self.cm.compute_time(n.work, "PIM", n.attrs.get("op"))
            return sum(ts)/len(ts)

        def rec(u: str) -> float:
            if u in memo: return memo[u]
            n = g.nodes[u]
            if len(g.successors(u)) == 0:
                val = avg_comp(n)
            else:
                val = avg_comp(n) + max(rec(v) for v in g.successors(u))
            memo[u] = val
            return val

        return {nid: rec(nid) for nid in g.nodes}

    # ---------- earliest finish on single device ----------
    def _earliest_finish_on_device(self, g: TaskGraph, node_id: str, dev_type: str):
        node = g.nodes[node_id]
        dev, ready = self.hw.earliest_available(dev_type)

        # inputs ready?
        t_ready_inputs = 0.0
        for pid in g.predecessors(node_id):
            p = g.nodes[pid]
            pa = self.assignments[pid]
            t_avail = pa.finish_time
            src_fmt = pa.output_format
            dst_fmt = DEVICE_PREFERRED_FORMAT[dev_type]
            if pa.output_device == "HYBRID":
                t_avail += self.cm.gb_move_and_format(dev_type, p.out_size, src_fmt, dst_fmt)
            elif pa.output_device != dev_type:
                t_avail += self.cm.inter_device_comm(pa.output_device, dev_type, p.out_size, src_fmt, dst_fmt)
            else:
                t_avail += self.cm.local_pass_time(dev_type, p.out_size)
                t_avail += self.cm.format_conversion_time(src_fmt, dst_fmt, p.out_size)
            t_ready_inputs = max(t_ready_inputs, t_avail)

        # weight ready?
        t_ready_weight = 0.0
        if node.weight_id:
            w = node.weight_id
            if not self.buf.has_weight(w):
                # 首轮/无预加载时默认全为ND，下一轮用 --init-plan 预加载
                self.buf.place_weight(w, node.weight_size, initial_format="ND")
            stored_formats = self.buf.weight_formats(w)
            preferred = DEVICE_PREFERRED_FORMAT[dev_type]
            if ENABLE_WEIGHT_PREFETCH and preferred not in stored_formats:
                self.buf.ensure_format(w, preferred)
                stored_formats = self.buf.weight_formats(w)
            best_time = float("inf")
            for fmt in stored_formats or {"ND"}:
                size_fmt = self.cm.format_size(node.weight_size, fmt)
                t = self.cm.gb_to_device_time(dev_type, size_fmt) + self.cm.format_conversion_time(fmt, preferred, node.weight_size)
                if t < best_time:
                    best_time = t
            t_ready_weight = best_time

        start = max(ready, t_ready_inputs, (ready + t_ready_weight))
        comp = self.cm.compute_time(node.work, dev_type, node.attrs.get("op"))
        end = start + comp
        out_fmt = DEVICE_PREFERRED_FORMAT[dev_type]
        return start, end, dev, out_fmt

    # ---------- earliest finish hybrid ----------
    def _earliest_finish_hybrid(self, g: TaskGraph, node_id: str):
        node = g.nodes[node_id]
        # npu & pim free time
        npu_dev, t_npu_free = self.hw.earliest_available("NPU")
        pim_dev, t_pim_free = self.hw.earliest_available("PIM")

        # ready inputs?
        t_ready_inputs_npu = 0.0
        t_ready_inputs_pim = 0.0
        for pid in g.predecessors(node_id):
            p = g.nodes[pid]
            pa = self.assignments[pid]
            # NPU
            t_n = pa.finish_time
            if pa.output_device == "HYBRID":
                t_n += self.cm.gb_move_and_format("NPU", p.out_size, pa.output_format, "NPU_OPT")
            elif pa.output_device != "NPU":
                t_n += self.cm.inter_device_comm(pa.output_device, "NPU", p.out_size, pa.output_format, "NPU_OPT")
            else:
                t_n += self.cm.local_pass_time("NPU", p.out_size)
                t_n += self.cm.format_conversion_time(pa.output_format, "NPU_OPT", p.out_size)
            t_ready_inputs_npu = max(t_ready_inputs_npu, t_n)

            # PIM
            t_p = pa.finish_time
            if pa.output_device == "HYBRID":
                t_p += self.cm.gb_move_and_format("PIM", p.out_size, pa.output_format, "PIM_OPT")
            elif pa.output_device != "PIM":
                t_p += self.cm.inter_device_comm(pa.output_device, "PIM", p.out_size, pa.output_format, "PIM_OPT")
            else:
                t_p += self.cm.local_pass_time("PIM", p.out_size)
                t_p += self.cm.format_conversion_time(pa.output_format, "PIM_OPT", p.out_size)
            t_ready_inputs_pim = max(t_ready_inputs_pim, t_p)

        # weight ready?
        t_ready_weight_npu = 0.0
        t_ready_weight_pim = 0.0
        if node.weight_id:
            w = node.weight_id
            if not self.buf.has_weight(w):
                self.buf.place_weight(w, node.weight_size, initial_format="ND")
            stored_formats = self.buf.weight_formats(w) or {"ND"}

            # 最快转化的格式
            best_n = float("inf")
            for fmt in stored_formats:
                size_fmt = self.cm.format_size(node.weight_size, fmt)
                t = self.cm.gb_move_and_format("NPU", size_fmt, fmt, "NPU_OPT")
                if t < best_n: best_n = t
            t_ready_weight_npu = best_n

            best_p = float("inf")
            for fmt in stored_formats:
                size_fmt = self.cm.format_size(node.weight_size, fmt)
                t = self.cm.gb_move_and_format("PIM", size_fmt, fmt, "PIM_OPT")
                if t < best_p: best_p = t
            t_ready_weight_pim = best_p

        # AST
        start_npu = max(t_npu_free, t_ready_inputs_npu, (t_npu_free + t_ready_weight_npu))
        start_pim = max(t_pim_free, t_ready_inputs_pim, (t_pim_free + t_ready_weight_pim))

        # compute time & rate
        tN = self.cm.compute_time(node.work, "NPU", node.attrs.get("op"))
        tP = self.cm.compute_time(node.work, "PIM", node.attrs.get("op"))
        rN = (1.0 / tN) if tN > 0 else float("inf")
        rP = (1.0 / tP) if tP > 0 else float("inf")

        # 先进行npu/pim，join后并行
        if start_npu <= start_pim:
            lead = "NPU"; lead_start = start_npu; join = start_pim; rate_lead = rN; rate_tail = rP
        else:
            lead = "PIM"; lead_start = start_pim; join = start_npu; rate_lead = rP; rate_tail = rN

        # 先进行的硬件完成度
        lead_time = max(0.0, join - lead_start)
        work_done = (0.0 if (rate_lead == float('inf')) else rate_lead * lead_time)

        out_fmt = "ND"  # Hybrid 输出默认 ND（保持中立；如需可改为最后完成侧格式）
        if work_done >= 1.0 - 1e-12:
            # 退化为单设备
            finish = join
            if lead == "NPU":
                return {"mode":"NPU", "start_npu": start_npu, "start_pim": None, "finish": finish, "npu": npu_dev, "pim": pim_dev, "out_fmt": DEVICE_PREFERRED_FORMAT["NPU"]}
            else:
                return {"mode":"PIM", "start_npu": None, "start_pim": start_pim, "finish": finish, "npu": npu_dev, "pim": pim_dev, "out_fmt": DEVICE_PREFERRED_FORMAT["PIM"]}
        else:
            # 并行
            agg = rate_lead + rate_tail
            finish = join + (1.0 - work_done) / agg
            return {"mode":"HYBRID", "start_npu": start_npu, "start_pim": start_pim, "finish": finish, "npu": npu_dev, "pim": pim_dev, "out_fmt": out_fmt}

    # ---------- public ----------
    def schedule(self, g: TaskGraph) -> Dict[str, Assignment]:
        import math
        ranks = self.compute_rank_u(g)
        order = sorted(g.nodes.keys(), key=lambda nid: -ranks[nid])

        for nid in order:
            node = g.nodes[nid]
            options: List[tuple] = []

            # 应用 placement 约束
            allow_npu, allow_pim, allow_hybrid_cfg = self._resolve_placement_constraints(node)

            # 单设备
            npu_opt = None; pim_opt = None
            if allow_npu:
                s,e,dev,fmt = self._earliest_finish_on_device(g, nid, "NPU")
                npu_opt = ("NPU", s, e, dev, fmt)
                options.append(npu_opt)
            if allow_pim:
                s,e,dev,fmt = self._earliest_finish_on_device(g, nid, "PIM")
                pim_opt = ("PIM", s, e, dev, fmt)
                options.append(pim_opt)

            # Hybrid 候选（先看配置，再看 gating + 滞回）
            consider_hybrid = ALLOW_HYBRID and allow_hybrid_cfg and allow_npu and allow_pim
            if consider_hybrid:
                # 差异门控
                pass_gating = True
                if HYBRID_GATE_BY_DIFF and (npu_opt is not None) and (pim_opt is not None):
                    f1, f2 = npu_opt[2], pim_opt[2]  # finish times
                    rel_diff = abs(f1 - f2) / max(1e-9, min(f1, f2))
                    abs_diff = abs(f1 - f2)
                    pass_gating = (rel_diff <= HYBRID_RELATIVE_DIFF) or (abs_diff <= HYBRID_ABSOLUTE_MARGIN)

                # 滞回窗
                if HYSTERESIS_ENABLE and (npu_opt is not None) and (pim_opt is not None):
                    op_name = node.attrs.get("op") or getattr(node, "name", None) or ""
                    last_mode = self.mode_mem.get(op_name, None)
                    f1, f2 = npu_opt[2], pim_opt[2]
                    rel_diff = abs(f1 - f2) / max(1e-9, min(f1, f2))
                    abs_diff = abs(f1 - f2)
                    if last_mode == "HYBRID":
                        # 只有当差距变得“显著更大”时，才退出 Hybrid
                        if (rel_diff > HYST_REL_EXIT) and (abs_diff > HYST_ABS_EXIT):
                            pass_gating = False
                        else:
                            pass_gating = True
                    else:
                        # 进入 Hybrid 需要更严格（差距足够小）
                        if (rel_diff < HYST_REL_ENTER) or (abs_diff < HYST_ABS_ENTER):
                            pass_gating = pass_gating and True
                        else:
                            pass_gating = False

                if pass_gating:
                    hy = self._earliest_finish_hybrid(g, nid)
                    if hy["mode"] == "HYBRID":
                        # devinfo 携带 start_npu/start_pim 以便 reserve
                        options.append(("HYBRID",
                                        min(hy["start_npu"], hy["start_pim"]),
                                        hy["finish"],
                                        (hy["npu"], hy["pim"], hy["start_npu"], hy["start_pim"]),
                                        hy["out_fmt"]))
                    elif hy["mode"] == "NPU":
                        options.append(("NPU", hy["start_npu"], hy["finish"], hy["npu"], hy["out_fmt"]))
                    elif hy["mode"] == "PIM":
                        options.append(("PIM", hy["start_pim"], hy["finish"], hy["pim"], hy["out_fmt"]))

            if not options: raise RuntimeError(f"No scheduling options for node {nid}")

            from config import OBJECTIVE, BALANCED_ALPHA
            if OBJECTIVE == "latency":
                best = min(options, key=lambda x: x[2])
            elif OBJECTIVE == "memory":
                order_pref = {"PIM":0,"NPU":1,"HYBRID":2}
                best = min(options, key=lambda x: (order_pref[x[0]], x[2]))
            else:
                best = min(options, key=lambda x: BALANCED_ALPHA*x[2] + (1-BALANCED_ALPHA)*x[1])

            state, s, e, devinfo, outfmt = best
            if state == "HYBRID":
                # devinfo may be (npu, pim, start_npu, start_pim)
                if isinstance(devinfo, tuple) and len(devinfo) >= 4:
                    npu, pim, start_npu, start_pim = devinfo[:4]
                else:
                    npu, pim = devinfo
                    start_npu, start_pim = s, s
                # 允许一侧为 None（退化情形）
                self.hw.reserve_hybrid(npu, pim, nid, (start_npu, start_pim), e)
                self.assignments[nid] = Assignment(
                    nid, state,
                    (npu.id if start_npu is not None else None,
                     pim.id if start_pim is not None else None),
                    min([t for t in (start_npu, start_pim) if t is not None]),
                    e, outfmt, "HYBRID"
                )
            else:
                dev: Device = devinfo
                self.hw.reserve(dev, nid, s, e)
                self.assignments[nid] = Assignment(
                    nid, state,
                    (dev.id if dev.type=="NPU" else None, dev.id if dev.type=="PIM" else None),
                    s, e, outfmt, dev.type
                )
            # 记录滞回的选择模式（按 op 聚合）
            op_name = node.attrs.get("op") or getattr(node, "name", None) or ""
            self.mode_mem[op_name] = state
        return self.assignments
