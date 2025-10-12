# scheduler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from main import PlanLabel
from hardware import Cluster, DeviceSpec
from task_graph import TaskGraph, TaskNode
from cost_model import CostModel
from buffer_manager import BufferManager, LRUCache
from config import (
    ALLOW_HYBRID,
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
    def __init__(self, cluster: Cluster, cost: CostModel, label:PlanLabel, batch: int, seq_len: int, buffer: BufferManager):
        self.cluster = cluster
        self.cost = cost
        self.label = label
        self.batch = batch
        self.seq_len = seq_len
        # external buffer manager (host + device weight caching / replacement)
        self.buffer = buffer or BufferManager() #所有的pim一个buffermanager
        self._pim_cache_capacity: Dict[str, int] = {}
        total_budget = int(getattr(self.label, "pim_weight_capacity_bytes", 0) or 0)
        pim_devs = self.cluster.devices_by_type("pim")
        if pim_devs: 
            n_dev = len(pim_devs)
            share = total_budget // n_dev
            remainder = total_budget % n_dev
            for idx, d in enumerate(pim_devs):
                cap = share + (1 if idx < remainder else 0)
                max_dev_bytes = int(d.mem_capacity_GB * 1e9)
                self._pim_cache_capacity[d.name] = min(max_dev_bytes, cap)
            for d in pim_devs:
                desired = max(0, self._pim_cache_capacity.get(d.name, int(d.mem_capacity_GB * 1e9)))
                cache = self.buffer.device_cache.get(d.name) #buffer manager里面的cache
                if cache is not None: #已经有这个设备的cache
                    cache.capacity = desired
                    while cache.used > cache.capacity and cache.order:
                        ev = cache.order.pop(0)
                        cache.pinned.discard(ev)
                        cache.used -= cache.items.pop(ev, 0)
                else: #没有就创建
                    self.buffer.ensure_device_cache(d.name, desired)


        self.comm = CommManager(cluster)
        self.avail: Dict[str, float] = {name: 0.0 for name in self.cluster.devices}

        # per-node results
        self._node_finish_time: Dict[str, float] = {}
        self._node_placement: Dict[str, str] = {}
        self._node_out_fmt: Dict[str, str] = {}

        # weights
        self.weight_cached: Dict[Tuple[str, str], bool] = {}  # legacy simple flag (keep for compatibility)
        self.storage_fmt_map: Dict[str, str] = {}  # host storage fmt by weight_id (overrides buffer.host_format)
        self._weight_load_count: Dict[Tuple[str, str], int] = defaultdict(int)  # (wid, dev.type) -> cnt
        self._weight_sizes: Dict[str, int] = {}

        # hybrid hysteresis memory
        self.mode_mem: Dict[str, str] = {}

    # Public: allow progressive decode to update seq_len
    def set_seq_len(self, seq_len: int) -> None:
        self.seq_len = int(seq_len)

    def _pim_cache_capacity_for(self, dev: DeviceSpec) -> int:
        cap = self._pim_cache_capacity.get(dev.name)
        if cap is None:
            return int(dev.mem_capacity_GB * 1e9)
        return max(0, cap)
    
    def _estimate_node_io_bytes(
        self,
        node: TaskNode,
        phase: str,
        *,
        batch: Optional[int] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[int, int]:
        batch = int(batch if batch is not None else getattr(self, "batch", 0) or 0)
        seq_len = int(seq_len if seq_len is not None else getattr(self, "seq_len", 0) or 0)
    # --------------------------
    # HEFT: upward rank helpers
    # --------------------------
    def _avg_compute_cost(self, node: TaskNode, phase: str) -> float:
        devs = list(self.cluster.devices.values())
        total_compute = 0.0
        total_w = 0.0
        k = 0
        seq_len = int(getattr(self, "seq_len", 0) or 0)
        batch = int(getattr(self, "batch", 0) or 0)
        node_flops = self.cost.estimate_flops(node, batch, seq_len, phase)
        node_weight_size = node.weight_size
        for d in devs:
            if not node.allowed.get(d.type, True):
                continue
            k += 1
            device_compute = self.cost.flop_time(node_flops, d)
            total_compute += device_compute
            if RANKU_INCLUDE_AVG_WEIGHT_LOAD and node.weight_id and node_weight_size > 0:
                wid = node.weight_id
                stored_fmt = self.storage_fmt_map.get(wid, "ND")
                size_src = self.cost.format_size(int(node_weight_size), stored_fmt)
                weight_cost = self.cost.gb_move_and_format(d, size_src, stored_fmt, self.cost.device_preferred_fmt(d))
                total_w += weight_cost
        avg_compute = (total_compute / k) if k else 0.0
        avg_w = (total_w / k) if (k and RANKU_INCLUDE_AVG_WEIGHT_LOAD and node.weight_id) else 0.0
        total_avg = avg_compute + avg_w
        return total_avg

    def _avg_comm_cost(self, u: TaskNode, v: TaskNode, phase:str) -> float:
        devs = list(self.cluster.devices.values())
        total = 0.0
        k = 0
        batch = int(getattr(self, "batch", 0) or 0)
        seq_len = int(getattr(self, "seq_len", 0) or 0)
        u_read,u_write = self.cost.estimate_activation_bytes(u, batch, seq_len, phase)
        v_read,_ = self.cost.estimate_activation_bytes(v, batch, seq_len, phase)
        payload_bytes = max(u_write, v_read, 16 * 1024)
        for i in range(len(devs)):
            for j in range(len(devs)):
                di, dj = devs[i],devs[j]
                if not (u.allowed.get(di.type, True) and v.allowed.get(dj.type, True)):
                    continue
                src_fmt = self.cost.device_preferred_fmt(di)
                dst_fmt = self.cost.device_preferred_fmt(dj)
                payload_src = self.cost.format_size(int(payload_bytes), src_fmt)
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
    # Detailed timing helpers
    # --------------------------
    def _weight_load_time(self, node: TaskNode, dev: DeviceSpec, t0: float, commit: bool) -> float:
        """Host->dev load + format conversion; overlappable with compute."""
        if not node.weight_id or node.weight_size <= 0:
            return 0.0
        wid = node.weight_id

        # 判断是否已缓存
        if dev.type == "pim" and self.buffer.is_cached(dev.name, wid):
            if commit:
                self.buffer.device_cache[dev.name].touch(wid)
            return 0.0

        host = self.cost.get_host_device().name
        stored_fmt = self.storage_fmt_map.get(wid, self.buffer.get_host_fmt(wid) or "ND")
        size_src = self.cost.format_size(node.weight_size, stored_fmt)
        _, link_end = self.comm.reserve(host, dev.name, size_src, earliest=t0, commit=commit)
        conv_t = self.cost.format_conversion_time(size_src, stored_fmt, self.cost.device_preferred_fmt(dev), dev)
        end = link_end + conv_t

        if commit:
            self._weight_load_count[(wid, dev.type)] += 1
            self._weight_sizes[wid] = node.weight_size
            if dev.type == "pim":
                # 加载后缓存并更新顺序
                self.weight_cached[(dev.name, wid)] = True
                self.buffer.mark_cached(dev.name, wid, node.weight_size, pinned=False)
            if wid not in self.buffer.host_format and wid in self.storage_fmt_map:
                self.buffer.set_host_fmt(wid, self.storage_fmt_map[wid])

        return max(0.0, end - t0)

    def _earliest_finish_on_device(self, g: TaskGraph, nid: str, dev: DeviceSpec, label:PlanLabel, phase: str, commit: bool) -> Tuple[float, float]:
        node = g.nodes[nid]

        # 1) inputs ready (consider cross-device comm + dst format conversion)
        ready_time = self._ready_time_for_device(g,nid,dev,phase,commit)

        # 2）device available
        t0 = max(self.avail[dev.name],ready_time)

        # 3) compute
        compute_t = self.cost.node_device_cost(node,dev,label,self.batch,self.seq_len,phase)
        wload_t = self._weight_load_time(node,dev,t0,commit)
        
        # 5) npu overlap
        if dev.type == "npu":
            total = max(compute_t,wload_t)
            finish = t0 + total
        else:
            cursor = t0
            cursor += wload_t
            cursor += compute_t
            finish = cursor

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

        inbound_start_times: List[float] = []
        inbound_end_times: List[float] = []
        batch = int(getattr(self, "batch", 0) or 0)
        seq_len = int(getattr(self, "seq_len", 0) or 0)
        node_read, _ = self.cost.estimate_activation_bytes(u, batch, seq_len, phase)

        for u in g.predecessors(nid):
            pred_finish = self._node_finish_time.get(u, 0.0) #前驱节点完成时间
            pred_dev_name = self._node_placement.get(u, dev.name) #前驱节点被分配的设备名称
            pred_dev = self.cluster.devices[pred_dev_name] #前驱节点被分配的设备
            if pred_dev.name == dev.name: #同设备
                inbound_start_times.append(pred_finish)
                inbound_end_times.append(pred_finish)
                continue

            else:
                src_fmt = self._node_out_fmt.get(u, self.cost.device_preferred_fmt(pred_dev))
                dst_fmt = self.cost.device_preferred_fmt(dev)
                pred_node = g.nodes[u]
                _, pred_write = self.cost.estimate_activation_bytes(u, batch, seq_len, phase)
                payload_nd = max(pred_write, node_read)
                payload_src = self.cost.format_size(payload_nd, src_fmt)
                link_start, link_end = self.comm.reserve(pred_dev.name, dev.name, payload_src, earliest=pred_finish, commit=commit)
                conv_t = self.cost.format_conversion_time(payload_src, src_fmt, dst_fmt, dev)
                dep_end = link_end + conv_t
                inbound_start_times.append(link_start)
                inbound_end_times.append(dep_end)
        if dev.type == "npu":
            return max(inbound_start_times, default=0.0)
        return max(inbound_end_times, default=0.0)

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

        start_npu = max(t_npu_free, t_ready_npu, t_npu_free + t_w_npu)
        start_pim = max(t_pim_free, t_ready_pim, t_pim_free + t_w_pim)

        tN = self.cost.node_device_cost(node, npu_dev, self.batch, self.seq_len, phase)
        tP = self.cost.node_device_cost(node, pim_dev, self.batch, self.seq_len, phase)
        rN = (0.0 if tN <= 0.0 else 1.0 / tN)
        rP = (0.0 if tP <= 0.0 else 1.0 / tP)

        if start_npu <= start_pim:
            lead_start = start_npu; tail_start = start_pim; r_lead = rN
        else:
            lead_start = start_pim; tail_start = start_npu; r_lead = rP

        lead_interval = max(0.0, tail_start - lead_start)
        work_done = r_lead * lead_interval
        
        print(f"tN: {tN}, tP: {tP}")
        agg = rN + rP #合起来处理的计算速率
        finish = tail_start + (1.0 - work_done) / agg
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
            allow_npu = node.allowed.get("npu", False)
            allow_pim = node.allowed.get("pim", False)

            candidates = []

            #npu
            if allow_npu:
                best_npu_dev = None
                best_npu_finish = float("inf")
                for dev in self.cluster.devices_by_type("npu"):
                    _, finish = self._earliest_finish_on_device(g, nid, dev, phase, commit=False)
                    if finish < best_npu_finish:
                        best_npu_finish = finish
                        best_npu_dev = dev  
                if best_npu_dev is not None:
                    candidates.append(("NPU", best_npu_finish, best_npu_dev))

            #pim
            if allow_pim:
                best_pim_dev = None
                best_pim_finish = float("inf")  
                for dev in self.cluster.devices_by_type("pim"):
                    _, finish = self._earliest_finish_on_device(g, nid, dev, phase, commit=False)
                    if finish < best_pim_finish:
                        best_pim_finish = finish
                        best_pim_dev = dev  
                if best_pim_dev is not None:
                    candidates.append(("PIM", best_pim_finish, best_pim_dev))

            #hybrid
            if ALLOW_HYBRID and allow_npu and allow_pim:
                best_result = self._earliest_finish_hybrid(g, nid, phase, commit=False)
                if best_result is not None:
                    candidates.append(("HYBRID", best_result["finish"], None))

            chosen_mode, chosen_finish, chosen_data = min(candidates, key=lambda x: x[1])

            if chosen_mode == "HYBRID":
                # Re-compute with commit=True
                hy = self._earliest_finish_hybrid(g, nid, phase, commit=True)
                npu = hy["npu"]
                pim = hy["pim"]
                start = min(hy["start_npu"], hy["start_pim"])
                finish = hy["finish"]
                
                self.avail[npu.name] = finish
                self.avail[pim.name] = finish
                self._node_finish_time[nid] = finish
                self._node_placement[nid] = hy["out_dev"].name
                self._node_out_fmt[nid] = "ND"##
                schedule.append(ScheduledTask(nid, f"HYBRID({npu.name}+{pim.name})", start, finish))
                
                op_name = node.attrs.get("op") or node.name
                self.mode_mem[op_name] = "HYBRID"
            else:
                # Single device execution (NPU or PIM)
                dev = chosen_data
                start, finish = self._earliest_finish_on_device(g, nid, dev, phase, commit=True)
                
                self.avail[dev.name] = finish
                self._node_finish_time[nid] = finish
                self._node_placement[nid] = dev.name
                self._node_out_fmt[nid] = self.cost.device_preferred_fmt(dev)
                schedule.append(ScheduledTask(nid, dev.name, start, finish))
                
                op_name = node.attrs.get("op") or node.name
                self.mode_mem[op_name] = chosen_mode
            print(f"[Schedule] Node {nid} on {self._node_placement[nid]} from {start:.3f} to {finish:.3f} ({chosen_mode})")

        return schedule

    def makespan(self, schedule: List[ScheduledTask]) -> float:
        return max((t.finish for t in schedule), default=0.0)


    def set_storage_format_map(self, fmt_map: Dict[str, str]):
        self.storage_fmt_map = dict(fmt_map or {})
        # sync into buffer manager host_format
        for k, v in self.storage_fmt_map.items():
            self.buffer.set_host_fmt(k, v)

    def suggest_weight_storage_formats(self) -> Dict[str, str]:
        candidates = ["ND", "NPU_OPT", "PIM_OPT"]
        sugg: Dict[str, str] = {}
        by_wid: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int)) #按权重ID和设备类型统计使用次数
        for (wid, dev_type), cnt in self._weight_load_count.items():
            by_wid[wid][dev_type] += cnt

        for wid, counts in by_wid.items(): #遍历dict的key和value对，将权重id（key）赋值给wid，value赋值给counts（是一个嵌套字典）
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
        # Keep buffer statistics but clear PIM cache states for deterministic reruns
        # (only PIM devices have LRU cache for storage management)
        for cache in self.buffer.device_cache.values():
            cache.items.clear(); cache.order.clear(); cache.used = 0; cache.pinned.intersection_update(cache.pinned)
        self._node_out_fmt.clear()
        self.weight_cached.clear()
        self.mode_mem.clear()
        # Keep buffer statistics but clear PIM cache states for deterministic reruns
        # (only PIM devices have LRU cache for storage management)
        for cache in self.buffer.device_cache.values():
            cache.items.clear(); cache.order.clear(); cache.used = 0; cache.pinned.intersection_update(cache.pinned)
