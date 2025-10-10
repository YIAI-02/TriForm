# scheduler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from hardware import Cluster, DeviceSpec
from task_graph import TaskGraph, TaskNode
from cost_model import CostModel
from config import (
    ALLOW_HYBRID, HYBRID_GATE_BY_DIFF,
    HYBRID_RELATIVE_DIFF, HYBRID_ABSOLUTE_MARGIN,
    HYSTERESIS_ENABLE, HYST_REL_ENTER, HYST_ABS_ENTER,
    HYST_REL_EXIT, HYST_ABS_EXIT,
    RANKU_INCLUDE_AVG_WEIGHT_LOAD,
)

DEBUG_SCHEDULER = True

# ------------------------------
# Simple Gantt record
# ------------------------------
@dataclass
class ScheduledTask:
    node_id: str
    device: str
    start: float
    finish: float


# ------------------------------
# Communication manager
# ------------------------------
class CommManager:
    """
    Maintain independent timelines per (src, dst) channel.
    """
    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.timeline_end: Dict[Tuple[str, str], float] = {}

    def reserve(
        self,
        src: str,
        dst: str,
        bytes_amount: int,
        earliest: float,
        commit: bool = True,
    ) -> Tuple[float, float]:
        key = (src, dst)
        bw = self.cluster.get_link_bw(src, dst) * 1e9  # bytes/s
        ch_end = self.timeline_end.get(key, 0.0)
        start = max(ch_end, earliest)
        dt = 0.0 if bw <= 0 else bytes_amount / bw
        end = start + dt
        if commit:
            self.timeline_end[key] = end
        return start, end


# ------------------------------
# HEFT Scheduler with Hybrid + two-pass format tuning
# ------------------------------
class HEFTScheduler:
    def __init__(self, cluster: Cluster, cost: CostModel, label, batch: int, seq_len: int):
        self.cluster = cluster
        self.cost = cost
        self.label = label
        self.batch = batch
        self.seq_len = seq_len

        self.comm = CommManager(cluster)
        self.avail: Dict[str, float] = {name: 0.0 for name in self.cluster.devices}

        # per-node results
        self._node_finish_time: Dict[str, float] = {}
        self._node_placement: Dict[str, str] = {}
        self._node_out_fmt: Dict[str, str] = {}

        # weights
        self.weight_cached: Dict[Tuple[str, str], bool] = {}  # (dev_name, weight_id) -> cached
        self.storage_fmt_map: Dict[str, str] = {}  # host storage fmt by weight_id
        self._weight_load_count: Dict[Tuple[str, str], int] = defaultdict(int)  # (wid, dev.type) -> cnt
        self._weight_sizes: Dict[str, int] = {}

        # hybrid hysteresis memory
        self.mode_mem: Dict[str, str] = {}

    # Public: allow progressive decode to update seq_len
    def set_seq_len(self, seq_len: int) -> None:
        self.seq_len = int(seq_len)

    # --------------------------
    # HEFT: upward rank helpers
    # --------------------------
    def _avg_compute_cost(self, node: TaskNode, phase: str) -> float:
        devs = list(self.cluster.devices.values())
        total_compute = 0.0
        total_w = 0.0
        k = 0
        
        for d in devs:
            if not node.allowed.get(d.type, True):
                continue
            k += 1
            device_compute = self.cost.node_device_cost(node, d, self.batch, self.seq_len, phase=phase)
            total_compute += device_compute
 
            if RANKU_INCLUDE_AVG_WEIGHT_LOAD and node.weight_id and node.weight_size > 0:
                wid = node.weight_id
                stored_fmt = self.storage_fmt_map.get(wid, "ND")
                size_src = self.cost.format_size(node.weight_size, stored_fmt)
                weight_cost = self.cost.gb_move_and_format(d, size_src, stored_fmt, self.cost.device_preferred_fmt(d))
                total_w += weight_cost
        avg_compute = (total_compute / k) if k else 0.0
        avg_w = (total_w / k) if (k and RANKU_INCLUDE_AVG_WEIGHT_LOAD and node.weight_id) else 0.0
        total_avg = avg_compute + avg_w
        return total_avg

    def _avg_comm_cost(self, u: TaskNode, v: TaskNode) -> float:
        devs = list(self.cluster.devices.values())
        total = 0.0
        k = 0
        bytes_nd = max(u.bytes_write, v.bytes_read, 16 * 1024)
        for i in range(len(devs)):
            for j in range(len(devs)):
                di, dj = devs[i], devs[j]
                if not (u.allowed.get(di.type, True) and v.allowed.get(dj.type, True)):
                    continue
                # payload size by source's output fmt
                src_fmt = self.cost.device_preferred_fmt(di)
                dst_fmt = self.cost.device_preferred_fmt(dj)
                payload_src = self.cost.format_size(bytes_nd, src_fmt)
                t_link = self.cost.comm_cost(di, dj, payload_src)
                t_conv = 0.0
                if di.type != dj.type:
                    t_conv = self.cost.format_conversion_time(payload_src, src_fmt, dst_fmt, dj)
                total += (t_link + t_conv)
                k += 1
        return total / k if k else 0.0

    def _upward_rank(self, g: TaskGraph, phase: str) -> List[str]:
        # compute rank_u bottom-up
        succ = {nid: list(g.successors(nid)) for nid in g.nodes}
        order = list(reversed(g.topological()))
        rank_u: Dict[str, float] = {}
        for nid in order:
            node = g.nodes[nid]
            if not succ[nid]:
                # Leaf node - only compute cost
                compute_cost = self._avg_compute_cost(node, phase=phase)
                rank_u[nid] = compute_cost
            else:
                # Non-leaf node - compute + max path to successors
                compute_cost = self._avg_compute_cost(node, phase=phase)
                best = 0.0
               
                for v in succ[nid]:
                    comm_cost = self._avg_comm_cost(node, g.nodes[v])
                    path_cost = comm_cost + rank_u[v]
                    if path_cost > best:
                        best = path_cost
                
                rank_u[nid] = compute_cost + best
        # schedule order: descending rank_u
        sorted_nodes = sorted(g.nodes.keys(), key=lambda x: -rank_u[x])
        return sorted_nodes

    # --------------------------
    # Streaming rule on PIM
    # --------------------------
    def _needs_streaming_on_pim(self, weight_id: Optional[str]) -> bool:
        if not weight_id:
            return False
        mode = getattr(self.label, "pim_mode", "small")
        if mode == "large":
            return False
        if mode == "medium":
            pinned = getattr(self.label, "pinned_fc_on_pim", set())
            return weight_id not in pinned
        # small
        return True

    # --------------------------
    # Detailed timing helpers
    # --------------------------
    def _weight_load_time(self, node: TaskNode, dev: DeviceSpec, t0: float, commit: bool) -> float:
        """Host->dev load + format conversion; overlappable with compute."""
        if not node.weight_id or node.weight_size <= 0:
            return 0.0
        wid = node.weight_id

        # cached and not streaming case
        if self.weight_cached.get((dev.name, wid), False) and not (dev.type == "pim" and self._needs_streaming_on_pim(wid)):
            return 0.0

        must_stream = (dev.type == "pim" and self._needs_streaming_on_pim(wid))
        host = self.cost.get_host_device().name
        stored_fmt = self.storage_fmt_map.get(wid, "ND")
        size_src = self.cost.format_size(node.weight_size, stored_fmt)

        # 1) transfer
        _, link_end = self.comm.reserve(host, dev.name, size_src, earliest=t0, commit=commit)
        # 2) convert on device
        conv_t = self.cost.format_conversion_time(size_src, stored_fmt, self.cost.device_preferred_fmt(dev), dev)
        end = link_end + conv_t

        if commit:
            self._weight_load_count[(wid, dev.type)] += 1
            self._weight_sizes[wid] = node.weight_size
            if not must_stream:
                self.weight_cached[(dev.name, wid)] = True

        return max(0.0, end - t0)

    def _kv_transfer_time_if_needed(self, node: TaskNode, dev: DeviceSpec, phase: str, t0: float, commit: bool) -> float:
        """Only for decode on PIM in small mode: host<->PIM KV movement."""
        if phase != "decode":
            return 0.0
        if node.name not in ("KV_read", "KV_write"):
            return 0.0
        if dev.type != "pim":
            return 0.0
        if getattr(self.label, "kv_in_pim", False):
            return 0.0

        r, w = self.cost.kv_rw_bytes_decode(node, self.batch, self.seq_len)
        host = self.cost.get_host_device().name
        if node.name == "KV_read":
            _, end = self.comm.reserve(host, dev.name, r, earliest=t0, commit=commit)
        else:
            _, end = self.comm.reserve(dev.name, host, w, earliest=t0, commit=commit)
        return max(0.0, end - t0)

    def _earliest_finish_on_device(self, g: TaskGraph, nid: str, dev: DeviceSpec, phase: str, commit: bool) -> Tuple[float, float]:
        node = g.nodes[nid]

        # 1) inputs ready (consider cross-device comm + dst format conversion)
        ready_time = 0.0
        for u in g.predecessors(nid):
            pred_finish = self._node_finish_time.get(u, 0.0)
            pred_dev_name = self._node_placement.get(u, dev.name)
            pred_dev = self.cluster.devices[pred_dev_name]
            if pred_dev.name == dev.name:
                ready_time = max(ready_time, pred_finish)
            else:
                payload_nd = max(g.nodes[u].bytes_write, node.bytes_read, 16 * 1024)
                src_fmt = self._node_out_fmt.get(u, self.cost.device_preferred_fmt(pred_dev))
                dst_fmt = self.cost.device_preferred_fmt(dev)
                payload_src = self.cost.format_size(payload_nd, src_fmt)
                _, link_end = self.comm.reserve(pred_dev.name, dev.name, payload_src, earliest=pred_finish, commit=commit)
                dep_end = link_end + self.cost.format_conversion_time(payload_src, src_fmt, dst_fmt, dev)
                ready_time = max(ready_time, dep_end)

        # 2) device available
        t0 = max(self.avail[dev.name], ready_time)

        # 3) compute
        compute_t = self.cost.node_device_cost(node, dev, self.batch, self.seq_len, phase)

        # 4) overlappable transfers
        wload_t = self._weight_load_time(node, dev, t0, commit)
        kv_t = self._kv_transfer_time_if_needed(node, dev, phase, t0, commit)

        # 5) finish
        finish = t0 + max(compute_t, wload_t, kv_t)
        if commit:
            self._node_out_fmt[nid] = self.cost.device_preferred_fmt(dev)
        return t0, finish

    # --------------------------
    # Hybrid helpers
    # --------------------------
    def _earliest_free_device(self, dev_type: str) -> Tuple[Optional[DeviceSpec], float]:
        devs = self.cluster.devices_by_type(dev_type)
        if not devs:
            return None, float("inf")
        best = None
        best_t = float("inf")
        for d in devs:
            t = self.avail.get(d.name, 0.0)
            if t < best_t:
                best, best_t = d, t
        return best, best_t

    def _ready_time_for_device(self, g: TaskGraph, nid: str, dev: DeviceSpec, phase: str, commit: bool) -> float:
        node = g.nodes[nid]
        ready = 0.0
        for u in g.predecessors(nid):
            pred_finish = self._node_finish_time.get(u, 0.0) #前驱节点完成时间
            pred_dev_name = self._node_placement.get(u, dev.name) #前驱节点被分配的设备名称
            pred_dev = self.cluster.devices[pred_dev_name] #前驱节点被分配的设备
            if pred_dev.name == dev.name: #同设备
                ready = max(ready, pred_finish)
            else:
                payload_nd = max(g.nodes[u].bytes_write, node.bytes_read, 16 * 1024)
                src_fmt = self._node_out_fmt.get(u, self.cost.device_preferred_fmt(pred_dev))
                dst_fmt = self.cost.device_preferred_fmt(dev)
                payload_src = self.cost.format_size(payload_nd, src_fmt)
                _, link_end = self.comm.reserve(pred_dev.name, dev.name, payload_src, earliest=pred_finish, commit=commit)
                dep_end = link_end + self.cost.format_conversion_time(payload_src, src_fmt, dst_fmt, dev)
                ready = max(ready, dep_end)
        return ready

    def _earliest_finish_hybrid(self, g: TaskGraph, nid: str, phase: str, commit: bool) -> Optional[Dict[str, object]]:
        node = g.nodes[nid]
        npu_dev, t_npu_free = self._earliest_free_device("npu")
        pim_dev, t_pim_free = self._earliest_free_device("pim")
        if (npu_dev is None) or (pim_dev is None):
            return None

        t_ready_npu = self._ready_time_for_device(g, nid, npu_dev, phase, commit)
        t_ready_pim = self._ready_time_for_device(g, nid, pim_dev, phase, commit)

        t_w_npu = self._weight_load_time(node, npu_dev, t_npu_free, commit)
        t_w_pim = self._weight_load_time(node, pim_dev, t_pim_free, commit)
        t_kv_npu = self._kv_transfer_time_if_needed(node, npu_dev, phase, t_npu_free, commit)
        t_kv_pim = self._kv_transfer_time_if_needed(node, pim_dev, phase, t_pim_free, commit)

        start_npu = max(t_npu_free, t_ready_npu, t_npu_free + t_w_npu, t_npu_free + t_kv_npu)
        start_pim = max(t_pim_free, t_ready_pim, t_pim_free + t_w_pim, t_pim_free + t_kv_pim)

        tN = self.cost.node_device_cost(node, npu_dev, self.batch, self.seq_len, phase)
        tP = self.cost.node_device_cost(node, pim_dev, self.batch, self.seq_len, phase)
        rN = (0.0 if tN <= 0.0 else 1.0 / tN)
        rP = (0.0 if tP <= 0.0 else 1.0 / tP)

        if start_npu <= start_pim:
            lead = "npu"; lead_start = start_npu; tail_start = start_pim; r_lead = rN; r_tail = rP
        else:
            lead = "pim"; lead_start = start_pim; tail_start = start_npu; r_lead = rP; r_tail = rN

        lead_interval = max(0.0, tail_start - lead_start)
        work_done = r_lead * lead_interval
        eps = 1e-12
        if work_done >= 1.0 - eps:
            # degenerate single-device
            finish = tail_start
            mode = "NPU" if lead == "npu" else "PIM"
            return {
                "mode": mode,
                "start_npu": (start_npu if mode == "NPU" else None),
                "start_pim": (start_pim if mode == "PIM" else None),
                "finish": finish,
                "npu": npu_dev,
                "pim": pim_dev,
                "out_dev": (npu_dev if mode == "NPU" else pim_dev),
            }
        else:
            agg = rN + rP #合起来处理的计算速率
            finish = tail_start if agg <= 0.0 else tail_start + (1.0 - work_done) / agg
            # choose output ownership by contribution
            contrib_n = max(0.0, finish - start_npu) * rN
            contrib_p = max(0.0, finish - start_pim) * rP
            out_dev = npu_dev if contrib_n >= contrib_p else pim_dev
            return {
                "mode": "HYBRID",
                "start_npu": start_npu,
                "start_pim": start_pim,
                "finish": finish,
                "npu": npu_dev,
                "pim": pim_dev,
                "out_dev": out_dev,
            }

    # --------------------------
    # Public API
    # --------------------------
    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        order = self._upward_rank(g, phase=phase)
        schedule: List[ScheduledTask] = []

        for nid in order:
            node = g.nodes[nid]

            # Try Hybrid first (if allowed)
            used_hybrid = False
            allow_npu = node.allowed.get("npu", True)
            allow_pim = node.allowed.get("pim", True)

            if ALLOW_HYBRID and allow_npu and allow_pim:
                hy_est = self._earliest_finish_hybrid(g, nid, phase, commit=False)
                if hy_est is not None:
                    # single-device baselines for gating
                    npu_dev, _ = self._earliest_free_device("npu")
                    pim_dev, _ = self._earliest_free_device("pim")
                    _, f_npu = self._earliest_finish_on_device(g, nid, npu_dev, phase, commit=False)
                    _, f_pim = self._earliest_finish_on_device(g, nid, pim_dev, phase, commit=False)
                    abs_diff = abs(f_npu - f_pim)
                    rel_diff = abs_diff / max(1e-9, min(f_npu, f_pim))

                    pass_gating = True
                    if HYBRID_GATE_BY_DIFF:
                        pass_gating = (rel_diff <= HYBRID_RELATIVE_DIFF) or (abs_diff <= HYBRID_ABSOLUTE_MARGIN)

                    if HYSTERESIS_ENABLE:
                        op_name = node.attrs.get("op") or node.name
                        last_mode = self.mode_mem.get(op_name)
                        if last_mode == "HYBRID":
                            if (rel_diff > HYST_REL_EXIT) and (abs_diff > HYST_ABS_EXIT):
                                pass_gating = False
                            else:
                                pass_gating = True
                        else:
                            if (rel_diff < HYST_REL_ENTER) or (abs_diff < HYST_ABS_ENTER):
                                pass_gating = pass_gating and True
                            else:
                                pass_gating = False

                    # both nearly free?
                    if pass_gating:
                        sN = hy_est["start_npu"] if hy_est["start_npu"] is not None else float("inf")
                        sP = hy_est["start_pim"] if hy_est["start_pim"] is not None else float("inf")
                        if abs(sN - sP) > HYBRID_ABSOLUTE_MARGIN:
                            pass_gating = False

                    if pass_gating:
                        hy = self._earliest_finish_hybrid(g, nid, phase, commit=True)
                        mode = hy["mode"]
                        if mode == "HYBRID":
                            npu = hy["npu"]; pim = hy["pim"]
                            start = min(hy["start_npu"], hy["start_pim"])
                            finish = hy["finish"]
                            self.avail[npu.name] = finish
                            self.avail[pim.name] = finish
                            self._node_finish_time[nid] = finish
                            self._node_placement[nid] = hy["out_dev"].name
                            self._node_out_fmt[nid] = "ND"  # hybrid outputs ND by convention
                            schedule.append(ScheduledTask(nid, f"HYBRID({npu.name}+{pim.name})", start, finish))
                            op_name = node.attrs.get("op") or node.name
                            self.mode_mem[op_name] = "HYBRID"
                            used_hybrid = True
                        elif mode in ("NPU", "PIM"):
                            dev = hy["npu"] if mode == "NPU" else hy["pim"]
                            start = hy["start_npu"] if mode == "NPU" else hy["start_pim"]
                            finish = hy["finish"]
                            self.avail[dev.name] = finish
                            self._node_finish_time[nid] = finish
                            self._node_placement[nid] = dev.name
                            self._node_out_fmt[nid] = self.cost.device_preferred_fmt(dev)
                            schedule.append(ScheduledTask(nid, dev.name, start, finish))
                            op_name = node.attrs.get("op") or node.name
                            self.mode_mem[op_name] = mode
                            used_hybrid = True

            if used_hybrid:
                continue

            # HEFT device enumeration
            best_dev: Optional[DeviceSpec] = None
            best_finish = float("inf")
            for dev in self.cluster.devices.values():
                if not node.allowed.get(dev.type, True):
                    continue
                start, finish = self._earliest_finish_on_device(g, nid, dev, phase, commit=False)
                if finish < best_finish:
                    best_finish = finish
                    best_dev = dev
            if best_dev is None:
                raise RuntimeError(f"No feasible device for node {nid}")

            start, finish = self._earliest_finish_on_device(g, nid, best_dev, phase, commit=True)
            schedule.append(ScheduledTask(nid, best_dev.name, start, finish))
            self.avail[best_dev.name] = finish
            self._node_finish_time[nid] = finish
            self._node_placement[nid] = best_dev.name
            self._node_out_fmt[nid] = self.cost.device_preferred_fmt(best_dev)

        return schedule

    def makespan(self, schedule: List[ScheduledTask]) -> float:
        return max((t.finish for t in schedule), default=0.0)


    def set_storage_format_map(self, fmt_map: Dict[str, str]):
        self.storage_fmt_map = dict(fmt_map or {})

    def suggest_weight_storage_formats(self) -> Dict[str, str]:
        candidates = ["ND", "NPU_OPT", "PIM_OPT"]
        sugg: Dict[str, str] = {}
        by_wid: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for (wid, dev_type), cnt in self._weight_load_count.items():
            by_wid[wid][dev_type] += cnt

        for wid, counts in by_wid.items():
            w_bytes_nd = self._weight_sizes.get(wid, 0)
            best_fmt = "ND"
            best_t = float("inf")
            for fmt in candidates:
                size_src = self.cost.format_size(w_bytes_nd, fmt)
                total = 0.0
                for dev_type, cnt in counts.items():
                    devs = self.cluster.devices_by_type(dev_type)
                    if not devs:
                        continue
                    d = devs[0]
                    total += cnt * self.cost.gb_move_and_format(d, size_src, fmt, self.cost.device_preferred_fmt(d))
                if total < best_t:
                    best_t, best_fmt = total, fmt
            sugg[wid] = best_fmt
        return sugg

    def reset_state(self):
        """Reset mutable scheduling state for a fresh pass (keep stats and storage_fmt_map)."""
        self.comm.timeline_end.clear()
        self.avail = {name: 0.0 for name in self.cluster.devices}
        self._node_finish_time.clear()
        self._node_placement.clear()
        self._node_out_fmt.clear()
        self.weight_cached.clear()
        self.mode_mem.clear()
