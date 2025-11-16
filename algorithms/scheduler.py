from __future__ import annotations
from config import attach_local_debug_filter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterable
from collections import defaultdict
from plan_label import PlanLabel
from hardware import Cluster, DeviceSpec
from task_graph import TaskGraph, TaskNode
from cost_model import CostModel
from buffer_manager import GlobalMemoryManager, LRUCache
from config import ALLOW_HYBRID, RANKU_INCLUDE_AVG_WEIGHT_LOAD
import logging
import math
import random
import copy

logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: DEBUG_SCHEDULER)
DEBUG_SCHEDULER = True

@dataclass
class ScheduledTask:
    node_id: str
    device: str
    start: float
    finish: float

class CommManager:
    """
    Maintain independent timelines per (src, dst) channel.
    """

    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.timeline_end: Dict[Tuple[str, str], float] = {}

    def reserve(self, src: str, dst: str, bytes_amount: int, earliest: float, commit: bool=True) -> Tuple[float, float]:
        key = (src, dst)
        bw = self.cluster.get_link_bw(src, dst) * 1000000000.0
        ch_end = self.timeline_end.get(key, 0.0)
        start = max(ch_end, earliest)
        dt = 0.0 if bw <= 0 else bytes_amount / bw
        end = start + dt
        if commit:
            self.timeline_end[key] = end
            logger.debug(
                f"[COMM] {src}->{dst} bytes={bytes_amount} bw={bw/1e9:.2f}GB/s start={start:.6f} end={end:.6f} dt={end-start:.6f} commit={commit}"
            )
            assert not ((src.startswith('NPU') and dst.startswith('PIM')) or
            (src.startswith('PIM') and dst.startswith('NPU'))), f'Forbidden direct NPU<->PIM transfer: {src}->{dst}'
        return (start, end)

class HEFTScheduler:

    def __init__(self, cluster: Cluster, cost: CostModel, label: PlanLabel, batch: int, seq_len: int, buffer: GlobalMemoryManager):
        self.cluster = cluster
        self.cost = cost
        self.label = label
        self.batch = batch
        self.seq_len = seq_len
        self.buffer = buffer or GlobalMemoryManager()
        self._pim_cache_capacity: Dict[str, int] = {}
        self._node_host_store_end: Dict[str, float] = {}  #节点输出在host上的可用时间
        self._act_cap: Dict[str, int] = {}                 # 每个设备用于激活的容量上限（字节），约 90% 可用
        self._act_used: Dict[str, int] = defaultdict(int)  # 当前每个设备已被激活占用的字节数
        self._act_resident: Dict[Tuple[str, str], int] = {}# (dev.name, node_id) -> bytes，表示该结点输出是否仍驻留在该设备
        self._act_refcnt: Dict[str, int] = {}              # node_id -> 还剩多少个下游消费者未消费
        self._max_util: float = 0.90                       # 最多只能占满 90%
        self._kv_reserved_per_dev: Dict[str, int] = {}     # PIM 每块设备为 KV 预留的容量（字节）
        self.comm = CommManager(cluster)
        self.avail: Dict[str, float] = {name: 0.0 for name in self.cluster.devices}
        self._node_finish_time: Dict[str, float] = {}
        self._node_placement: Dict[str, str] = {}
        self._node_out_fmt: Dict[str, str] = {}
        self.weight_cached: Dict[Tuple[str, str], bool] = {}
        self.storage_fmt_map: Dict[str, str] = {}
        self._weight_load_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self._weight_sizes: Dict[str, int] = {}
        self.mode_mem: Dict[str, str] = {}
        self._kv_reserved_per_dev: Dict[str, int] = {}

        total_budget = int(getattr(self.label, 'pim_weight_capacity_bytes', 0) or 0)
        pim_devs = self.cluster.devices_by_type('pim')

        if pim_devs:
            n_dev = len(pim_devs)
            share = total_budget // n_dev
            remainder = total_budget % n_dev
            for idx, d in enumerate(pim_devs):
                cap = share + (1 if idx < remainder else 0)
                max_dev_bytes = int(d.mem_capacity_GB * 1000000000.0)
                self._pim_cache_capacity[d.name] = min(max_dev_bytes, cap)
            for d in pim_devs:
                desired = max(0, self._pim_cache_capacity.get(d.name, int(d.mem_capacity_GB * 1000000000.0)))
                cache = self.buffer.device_cache.get(d.name)
                if cache is not None:
                    cache.capacity = desired
                    while cache.used > cache.capacity and cache.order:
                        ev = cache.order.pop(0)
                        cache.pinned.discard(ev)
                        cache.used -= cache.items.pop(ev, 0)
                else:
                    self.buffer.ensure_device_cache(d.name, desired)

        if pim_devs and getattr(self.label, 'kv_in_pim', False):
            kv_total_bytes = int(getattr(self.label, 'kv_total_bytes', 0) or 0)
            caps = [int(d.mem_capacity_GB * 1e9) for d in pim_devs]
            cap_sum = max(1, sum(caps))
            alloc = [ (caps[i] * kv_total_bytes) // cap_sum for i in range(len(pim_devs)) ]
            remainder = kv_total_bytes - sum(alloc)
            order = sorted(range(len(pim_devs)), key=lambda i: caps[i], reverse=True)
            for i in range(remainder):
                alloc[order[i % len(order)]] += 1
            for i, d in enumerate(pim_devs):
                self._kv_reserved_per_dev[d.name] = alloc[i]
                logger.debug(f"[INIT] KV_RESERVE dev={d.name} = {alloc[i]}")
        else:
            for d in pim_devs:
                self._kv_reserved_per_dev[d.name] = 0

        # --- Compute activation residency budget for ALL devices (90%) ---
        for name, d in self.cluster.devices.items():
            phy = int(d.mem_capacity_GB * 1e9)
            kv_reserve = int(self._kv_reserved_per_dev.get(name, 0)) if d.type == 'pim' else 0
            weight_budget = int(self._pim_cache_capacity.get(name, 0)) if d.type == 'pim' else 0
            cap = max(0, int(self._max_util * max(0, phy - kv_reserve - weight_budget)))
            self._act_cap[name] = cap
            logger.debug(f"[INIT] ACT_CAP dev={name} phy={phy} kv={kv_reserve} weight_budget={weight_budget} act_cap(90%)={cap}")

    def _kv_reserved_for(self, dev: DeviceSpec) -> int:
        return int(self._kv_reserved_per_dev.get(dev.name, 0))


    def set_seq_len(self, seq_len: int) -> None:
        self.seq_len = int(seq_len)

    def _pim_cache_capacity_for(self, dev: DeviceSpec) -> int:
        cap = self._pim_cache_capacity.get(dev.name)
        if cap is None:
            return int(dev.mem_capacity_GB * 1000000000.0)
        return max(0, cap)

    def _estimate_node_io_bytes(self, node: TaskNode, phase: str, *, batch: Optional[int]=None, seq_len: Optional[int]=None) -> Tuple[int, int]:
        batch = int(batch if batch is not None else getattr(self, 'batch', 0) or 0)
        seq_len = int(seq_len if seq_len is not None else getattr(self, 'seq_len', 0) or 0)

    def _avg_compute_cost(self, node: TaskNode, phase: str) -> float:
        devs = list(self.cluster.devices.values())
        total_compute = 0.0
        total_w = 0.0
        k = 0
        seq_len = int(getattr(self, 'seq_len', 0) or 0)
        batch = int(getattr(self, 'batch', 0) or 0)
        node_flops = self.cost.estimate_flops(node, batch, seq_len, phase)
        node_weight_size = node.weight_size
        for d in devs:
            if not node.allowed.get(d.type, True):
                continue
            k += 1
            device_compute = self.cost.flop_time(node_flops, d)
            total_compute += device_compute
            if RANKU_INCLUDE_AVG_WEIGHT_LOAD and node.weight_id and (node_weight_size > 0):
                wid = node.weight_id
                stored_fmt = self.storage_fmt_map.get(wid, 'ND')
                size_src = self.cost.format_size(int(node_weight_size), stored_fmt)
                weight_cost = self.cost.gb_move_and_format(d, size_src, stored_fmt, self.cost.device_preferred_fmt(d))
                total_w += weight_cost
        avg_compute = total_compute / k if k else 0.0
        avg_w = total_w / k if k and RANKU_INCLUDE_AVG_WEIGHT_LOAD and node.weight_id else 0.0
        total_avg = avg_compute + avg_w
        return total_avg

    def _avg_comm_cost(self, u: TaskNode, v: TaskNode, phase: str) -> float:
        devs = list(self.cluster.devices.values())
        total = 0.0
        k = 0
        batch = int(getattr(self, 'batch', 0) or 0)
        seq_len = int(getattr(self, 'seq_len', 0) or 0)
        u_read, u_write = self.cost.estimate_activation_bytes(u, batch, seq_len, phase)
        v_read, _ = self.cost.estimate_activation_bytes(v, batch, seq_len, phase)
        payload_bytes = max(u_write, v_read, 16 * 1024)
        for i in range(len(devs)):
            for j in range(len(devs)):
                di, dj = (devs[i], devs[j])
                if not (u.allowed.get(di.type, True) and v.allowed.get(dj.type, True)):
                    continue
                src_fmt = self.cost.device_preferred_fmt(di)
                dst_fmt = self.cost.device_preferred_fmt(dj)
                payload_src = self.cost.format_size(int(payload_bytes), src_fmt)
                t_link = self.cost.comm_cost(di, dj, payload_src)
                t_conv = 0.0
                if di.type != dj.type:
                    t_conv = self.cost.format_conversion_time(payload_src, src_fmt, dst_fmt, dj)
                total += max(t_link, t_conv)
                k += 1
        return total / k if k else 0.0

    def _upward_rank(self, g: TaskGraph, phase: str) -> List[str]:
        succ = {nid: list(g.successors(nid)) for nid in g.nodes}
        order = list(reversed(g.topological()))
        rank_u: Dict[str, float] = {}
        for nid in order:
            node = g.nodes[nid]
            if not succ[nid]:
                compute_cost = self._avg_compute_cost(node, phase=phase)
                rank_u[nid] = compute_cost
            else:
                compute_cost = self._avg_compute_cost(node, phase=phase)
                best = 0.0
                for v in succ[nid]:
                    comm_cost = self._avg_comm_cost(node, g.nodes[v], phase)
                    path_cost = comm_cost + rank_u[v]
                    if path_cost > best:
                        best = path_cost
                rank_u[nid] = compute_cost + best
        sorted_nodes = sorted(g.nodes.keys(), key=lambda x: -rank_u[x])
        return sorted_nodes

    def _weight_load_time(self, node: TaskNode, dev: DeviceSpec, t0: float, commit: bool) -> float:
        """Host->dev load + format conversion; overlappable with compute."""
        if not node.weight_id or node.weight_size <= 0:
            return 0.0
        wid = node.weight_id
        if dev.type == 'pim' and self.buffer.is_cached(dev.name, wid):
            logger.debug(f"[WEIGHT] cache-hit wid={wid} dev={dev.name}")
            if commit:
                self.buffer.device_cache[dev.name].touch(wid)
            return 0.0
        if dev.type == 'pim':
            logger.debug(f"[WEIGHT] host->pim load wid={wid} size={node.weight_size}")
            load_time = self.cost.weight_load_time_pim(node.weight_size)
            if commit:
                self._weight_load_count[wid, dev.type] += 1
                self._weight_sizes[wid] = node.weight_size
                self.weight_cached[dev.name, wid] = True
                pinned_flag = bool(getattr(self.label, 'pinned_fc_on_pim', set()) and (wid in self.label.pinned_fc_on_pim))
                self.buffer.mark_cached(dev.name, wid, node.weight_size, pinned=pinned_flag)                
                if wid not in self.buffer.host_format and wid in self.storage_fmt_map:
                    self.buffer.set_host_fmt(wid, self.storage_fmt_map[wid])
            return load_time
        host = self.cost.get_host_device().name
        stored_fmt = self.storage_fmt_map.get(wid, self.buffer.get_host_fmt(wid) or 'ND')
        size_src = self.cost.format_size(node.weight_size, stored_fmt)
        logger.debug(f"[WEIGHT] host->{dev.name} load wid={wid} stored_fmt={stored_fmt} size={node.weight_size}")
        _, link_end = self.comm.reserve(host, dev.name, size_src, earliest=t0, commit=commit)
        conv_t = self.cost.format_conversion_time(size_src, stored_fmt, self.cost.device_preferred_fmt(dev), dev)
        end = link_end + conv_t
        if commit:
            self._weight_load_count[wid, dev.type] += 1
            self._weight_sizes[wid] = node.weight_size
            if wid not in self.buffer.host_format and wid in self.storage_fmt_map:
                self.buffer.set_host_fmt(wid, self.storage_fmt_map[wid])
        return max(0.0, end - t0)

    def _earliest_finish_on_device(self, g: TaskGraph, nid: str, dev: DeviceSpec, label: PlanLabel, phase: str, commit: bool) -> Tuple[float, float]:
        node = g.nodes[nid]
        ready_time = self._ready_time_for_device(g, nid, dev, phase, commit)
        t0 = max(self.avail[dev.name], ready_time)
        compute_t = self.cost.node_device_cost(node, dev, label, self.batch, self.seq_len, phase)
        wload_t = self._weight_load_time(node, dev, t0, commit)
        if dev.type == 'npu':
            total = max(compute_t, wload_t)
            finish = t0 + total
        else:
            cursor = t0
            cursor += wload_t
            cursor += compute_t
            finish = cursor
        if commit:
            self._node_out_fmt[nid] = self.cost.device_preferred_fmt(dev)
            # ——尝试为当前结点输出做“就地驻留”，否则立即回写 Host（只写一次的早触发）——
            out_read_nd, out_write_nd = self.cost.estimate_activation_bytes(node, self.batch, self.seq_len, phase)
            out_nd = max(out_write_nd, out_read_nd)
            cap = int(self._act_cap.get(dev.name, 0))
            used = int(self._act_used.get(dev.name, 0))
            if used + out_nd <= cap:
                self._act_used[dev.name] = used + out_nd
                self._act_resident[(dev.name, nid)] = out_nd
                logger.debug(f"[LOCAL] retain out nid={nid}@{dev.name} bytes_nd={out_nd} used={self._act_used[dev.name]}/{cap}")
            else:
                # 空间不足：立即把该输出写回 Host（供后续消费者复用）
                logger.debug(f"[LOCAL->HOST] evict out nid={nid}@{dev.name} need={out_nd} used={used}/{cap}")
                src_fmt = self.cost.device_preferred_fmt(dev)
                self._ensure_host_store(nid, dev, out_nd, src_fmt, finish, commit=True)

            if nid not in self._act_refcnt:
                self._act_refcnt[nid] = len(g.successors(nid))
        return (t0, finish)

    def _earliest_free_device(self, dev_type: str) -> Tuple[Optional[DeviceSpec], float]:
        devs = self.cluster.devices_by_type(dev_type)
        if not devs:
            return (None, float('inf'))
        best = None
        best_t = float('inf')
        for d in devs:
            t = self.avail.get(d.name, 0.0)
            if t < best_t:
                best, best_t = (d, t)
        return (best, best_t)

    def _ready_time_for_device(self, g: TaskGraph, nid: str, dev: DeviceSpec, phase: str, commit: bool) -> float:
        node = g.nodes[nid]
        inbound_start_times: List[float] = []
        inbound_end_times: List[float] = []
        batch = int(getattr(self, 'batch', 0) or 0)
        seq_len = int(getattr(self, 'seq_len', 0) or 0)
        node_read, _ = self.cost.estimate_activation_bytes(node, batch, seq_len, phase)
        for u in g.predecessors(nid):
            pred_finish = self._node_finish_time.get(u, 0.0)
            pred_dev_name = self._node_placement.get(u, dev.name)
            # Fallback for legacy 'HYBRID' marker: treat as host-written ND payload
            if pred_dev_name == 'HYBRID':
                host = self.cost.get_host_device()
                pred_finish = self._node_finish_time.get(u, 0.0)
                pred_node = g.nodes[u]
                _, pred_write = self.cost.estimate_activation_bytes(pred_node, batch, seq_len, phase)
                payload_nd = max(pred_write, node_read)
                size_nd = self.cost.format_size(payload_nd, 'ND')
                l2s, l2e = self.comm.reserve(host.name, dev.name, size_nd,
                                             earliest=pred_finish, commit=commit)
                inbound_start_times.append(l2s)
                inbound_end_times.append(l2e)
                continue

            pred_dev = self.cluster.devices[pred_dev_name]
            if pred_dev.name == dev.name:
                src_fmt = self._node_out_fmt.get(u, self.cost.device_preferred_fmt(pred_dev))
                pred_node = g.nodes[u]
                _, pred_write = self.cost.estimate_activation_bytes(pred_node, batch, seq_len, phase)
                payload_nd = max(pred_write, node_read)
                size_nd = self.cost.format_size(payload_nd, 'ND')
                if dev.type == 'pim':
                    avail = self._pim_avail_for_activation(pred_dev)
                    if avail >= payload_nd:
                        if commit:
                            self._act_used[pred_dev.name] = self._act_used.get(pred_dev.name, 0) + payload_nd
                        logger.debug(f"[LOCAL] PIM in-place u={u}@{pred_dev.name} -> {nid}@{dev.name} bytes_nd={payload_nd} (use_inplace, avail={avail})")
                        inbound_start_times.append(pred_finish)
                        inbound_end_times.append(pred_finish)
                    else:
                        logger.debug(f"[LOCAL->HOST] PIM fallback (no space) u={u}@{pred_dev.name} -> {nid}@{dev.name} need={payload_nd} avail={avail}")
                        host_ready = self._ensure_host_store(u, pred_dev, payload_nd, src_fmt, pred_finish, commit)
                        l2s, l2e = self.comm.reserve(self.cost.get_host_device().name, dev.name, size_nd, earliest=host_ready, commit=commit)
                        conv2 = self.cost.format_conversion_time(size_nd, 'ND', self.cost.device_preferred_fmt(dev), dev)
                        inbound_start_times.append(l2s)
                        inbound_end_times.append(l2e + conv2)
                else:
                    logger.debug(f"[LOCAL] NPU reuse u={u}@{pred_dev.name} -> {nid}@{dev.name} (no xfer) pred_finish={pred_finish:.6f}")
                    inbound_start_times.append(pred_finish)
                    inbound_end_times.append(pred_finish)
                continue
            else:
                src_fmt = self._node_out_fmt.get(u, self.cost.device_preferred_fmt(pred_dev))
                dst_fmt = self.cost.device_preferred_fmt(dev)
                pred_node = g.nodes[u]
                _, pred_write = self.cost.estimate_activation_bytes(pred_node, batch, seq_len, phase)
                payload_nd = max(pred_write, node_read)
                payload_src = self.cost.format_size(payload_nd, src_fmt)
                host = self.cost.get_host_device()
                dst_fmt = self.cost.device_preferred_fmt(dev)
                pred_node = g.nodes[u]
                _, pred_write = self.cost.estimate_activation_bytes(pred_node, batch, seq_len, phase)
                payload_nd = max(pred_write, node_read)  # 以 ND 为“逻辑尺寸”
                src_fmt = self._node_out_fmt.get(u, self.cost.device_preferred_fmt(pred_dev))

                # 1) 若源是 NPU：无论目标是谁，都必须把 u 的输出先回写 Host（NPU_OPT->ND->Host）
                if pred_dev.type == 'npu':
                    host_ready = self._ensure_host_store(u, pred_dev, payload_nd, src_fmt, pred_finish, commit)
                else:
                    host_ready = None

                if dev.type == 'npu':
                    # 2) 目标是 NPU：一律从 Host 读取（Host->NPU + ND->NPU_OPT）
                    if host_ready is None:
                        host_ready = self._ensure_host_store(u, pred_dev, payload_nd, src_fmt, pred_finish, commit)
                    size_nd = self.cost.format_size(payload_nd, 'ND')
                    l2s, l2e = self.comm.reserve(host.name, dev.name, size_nd, earliest=host_ready, commit=commit)
                    conv2 = self.cost.format_conversion_time(size_nd, 'ND', dst_fmt, dev)
                    logger.debug(f"[CROSS] HOST->NPU u={u}@{pred_dev.name} -> {nid}@{dev.name} bytes_nd={payload_nd}")
                    inbound_start_times.append(l2s)
                    inbound_end_times.append(l2e + conv2)
                    continue

                if dev.type == 'pim':
                    # 3) 目标是 PIM：
                    if pred_dev.type == 'pim' and (self._pim_avail_for_activation(pred_dev) >= payload_nd):
                        # (a) PIM->PIM 就地：容量足够且允许就地，直接复用；不走 Host
                        if commit:
                            self._act_used[pred_dev.name] = self._act_used.get(pred_dev.name, 0) + payload_nd
                        logger.debug(f"[LOCAL] PIM in-place (cross device branch) u={u}@{pred_dev.name} -> {nid}@{dev.name} bytes_nd={payload_nd}")
                        inbound_start_times.append(pred_finish)
                        inbound_end_times.append(pred_finish)
                    else:
                        # (b) 其他情况：一律 pred->Host，再 Host->PIM
                        if host_ready is None:
                            host_ready = self._ensure_host_store(u, pred_dev, payload_nd, src_fmt, pred_finish, commit)
                        size_nd = self.cost.format_size(payload_nd, 'ND')
                        l2s, l2e = self.comm.reserve(host.name, dev.name, size_nd, earliest=host_ready, commit=commit)
                        logger.debug(f"[CROSS] HOST->PIM u={u}@{pred_dev.name} -> {nid}@{dev.name} bytes_nd={payload_nd}")
                        conv2 = self.cost.format_conversion_time(size_nd, 'ND', dst_fmt, dev)
                        inbound_start_times.append(l2s)
                        inbound_end_times.append(l2e + conv2)
                    continue

                # 4) 目标是 CPU/其他：一律确保回写 Host 即可
                if host_ready is None:
                    host_ready = self._ensure_host_store(u, pred_dev, payload_nd, src_fmt, pred_finish, commit)
                inbound_start_times.append(host_ready)
                inbound_end_times.append(host_ready)

                # NPU 不再“边传边算”，统一用依赖完成时间
                return max(inbound_end_times, default=0.0)

        if dev.type == 'npu':
            return max(inbound_start_times, default=0.0)
        return max(inbound_end_times, default=0.0)

    def _earliest_finish_hybrid(self, g, nid: str, phase: str, commit: bool):
        """
        计算把结点 nid 以 HYBRID（NPU+PIM 协同）方式执行时的最早完成时间，并在 commit=True 时真正预约链路。
        约束与行为：
        - NPU 与 PIM 不直连通信；仅使用 Host 作为交换点。
        - HYBRID：分别把权重加载到 NPU 与 PIM，然后两边各自计算。
        - 结尾阶段：把该算子“整体输出”统一写回 Host（主存），
            其中 NPU 源侧做 NPU_OPT->ND 转换后再写；PIM 源侧用 trace 仿真近存读出 + PIM_OPT->ND 转换，再写 Host。
        - 返回一个 dict，包含 start/finish 及分设备的细节；调度器上层只需要 'start' 和 'finish'。
        依赖：
        """
        node = g.nodes[nid]
        batch = int(getattr(self, 'batch', 0) or 0)
        seq_len = int(getattr(self, 'seq_len', 0) or 0)

        # 设备选取：默认取第一块 NPU 与第一块 PIM
        npu_list = getattr(self.cluster, 'devices_by_type')('npu')
        pim_list = getattr(self.cluster, 'devices_by_type')('pim')
        assert npu_list and pim_list, "HYBRID 需要至少一块 NPU 和一块 PIM"

        npu_dev = npu_list[0]
        pim_dev = pim_list[0]
        host = self.cost.get_host_device()

        # HYBRID 切分比例（0~1）：alpha 给 NPU，其余给 PIM
        alpha = getattr(self.label, 'hybrid_ratio', 0.5)
        try:
            alpha = float(alpha)
        except Exception:
            alpha = 0.5
        alpha = max(0.0, min(1.0, alpha))

        # 估算该结点的输入/输出（ND 逻辑字节数）
        node_read_nd, node_write_nd = self.cost.estimate_activation_bytes(node, batch, seq_len, phase)
        out_total_nd = max(node_write_nd, node_read_nd)  # 以“需要对下游可见”的 ND 逻辑尺寸为准

        # —— 1) 依赖就绪时间（分别对 NPU 与 PIM 侧）——
        # 注意：_ready_time_for_device 内部已确保跨设备走 Host 两跳，不会出现 NPU<->PIM 直连
        t_ready_npu = self._ready_time_for_device(g, nid, npu_dev, phase, commit=False)
        t_ready_pim = self._ready_time_for_device(g, nid, pim_dev, phase, commit=False)

        # —— 2) 权重加载（分别加载到 NPU 与 PIM）——
        # 2.1 NPU：Host->NPU 传输 + ND->NPU_OPT 转换
        # 说明：若你的工程里已有 self._weight_load_time(node, dev, earliest, phase, commit)
        #       可直接用该函数替代下述 2.1 这段“内联实现”。
        weight_total_nd = int(getattr(node, 'weight_size', 0) or 0)
        weight_npu_nd = int(round(weight_total_nd * alpha))
        weight_pim_nd = max(0, weight_total_nd - weight_npu_nd)

        # NPU 权重加载
        t_w_npu_end = t_ready_npu
        if weight_npu_nd > 0:
            w_nd_bytes = self.cost.format_size(weight_npu_nd, 'ND')
            # Host -> NPU 传
            _, link_end = self.comm.reserve(host.name, npu_dev.name, w_nd_bytes,
                                            earliest=t_ready_npu, commit=commit)
            # ND -> NPU_OPT 转换（在 NPU 上）
            conv_w = self.cost.format_conversion_time(w_nd_bytes, 'ND',
                                                    self.cost.device_preferred_fmt(npu_dev), npu_dev)
            t_w_npu_end = link_end + conv_w

        # 2.2 PIM：PIM 侧从自身内存加载权重（trace/带宽近似）
        # 若你已有 cost.weight_load_time_pim(bytes)，可直接使用；否则回退到 mem_time。
        t_w_pim_end = t_ready_pim
        if weight_pim_nd > 0:
            try:
                t_load_pim = self.cost.weight_load_time_pim(weight_pim_nd)
            except Exception:
                # 回退：按 PIM 内存带宽估
                t_load_pim = self.cost.mem_time(self.cost.format_size(weight_pim_nd, 'ND'), pim_dev)
            t_w_pim_end = t_ready_pim + t_load_pim

        # —— 3) 计算（两侧并行），用比例 alpha 拆分算力占用 —— 
        # 若你的 cost.compute_time 已支持 frac 参数，请改成 compute_time(..., frac=alpha)
        # 否则我们按“完整算子时长 × 比例”近似：
        t_comp_full_on_npu = self.cost.node_device_cost(node, npu_dev, self.label, self.batch, seq_len, phase)
        t_comp_full_on_pim = self.cost.node_device_cost(node, pim_dev, self.label, self.batch, seq_len, phase)
        t_comp_npu = t_comp_full_on_npu * alpha
        t_comp_pim = t_comp_full_on_pim * (1.0 - alpha)

        start_npu = max(t_ready_npu, t_w_npu_end)
        finish_npu = start_npu + (t_comp_npu if alpha > 0 else 0.0)

        start_pim = max(t_ready_pim, t_w_pim_end)
        finish_pim = start_pim + (t_comp_pim if (1.0 - alpha) > 0 else 0.0)

        # —— 4) 统一写回主存（Host）——
        # 输出切分：NPU 写回 alpha 部分，PIM 写回 (1-alpha) 部分；两边都在“自身设备”完成源侧处理后，经 Host 链路写回。
        npu_out_nd = int(round(out_total_nd * alpha))
        pim_out_nd = max(0, out_total_nd - npu_out_nd)

        # 4.1 NPU -> Host：源侧 NPU_OPT->ND 转换，再写入 Host
        # 注意：conversion_time 传入的是“源格式的字节数”
        if npu_out_nd > 0:
            npu_src_bytes = self.cost.format_size(npu_out_nd, self.cost.device_preferred_fmt(npu_dev))
            npu_conv_time = self.cost.format_conversion_time(npu_src_bytes,
                                                            self.cost.device_preferred_fmt(npu_dev), 'ND', npu_dev)
            npu_write_nd_bytes = self.cost.format_size(npu_out_nd, 'ND')
            # earliest 放在 NPU 计算完成 + 源侧转换之后
            _, npu_w_end = self.comm.reserve(npu_dev.name, host.name,
                                            npu_write_nd_bytes,
                                            earliest=finish_npu + npu_conv_time,
                                            commit=commit)
        else:
            npu_w_end = finish_npu

        # 4.2 PIM -> Host：先用 trace 仿真从 PIM 近存读出，再做 PIM_OPT->ND 转换，最后写入 Host
        if pim_out_nd > 0:
            # 近存读出（trace）
            t_read_pim = self.cost.activation_read_time_pim(pim_out_nd)
            # 源侧格式转换
            pim_src_bytes = self.cost.format_size(pim_out_nd, self.cost.device_preferred_fmt(pim_dev))
            pim_conv_time = self.cost.format_conversion_time(pim_src_bytes,
                                                            self.cost.device_preferred_fmt(pim_dev), 'ND', pim_dev)
            pim_write_nd_bytes = self.cost.format_size(pim_out_nd, 'ND')
            _, pim_w_end = self.comm.reserve(pim_dev.name, host.name,
                                            pim_write_nd_bytes,
                                            earliest=finish_pim + t_read_pim + pim_conv_time,
                                            commit=commit)
        else:
            pim_w_end = finish_pim

        # 统一完成时间 = 两侧都把各自份额写回 Host 后
        finish = max(npu_w_end, pim_w_end)
        # 定义开始时间为两侧“真正开算”的最早时刻（供上层记录/展示）
        start = min(start_npu if alpha > 0 else float('inf'),
                    start_pim if (1.0 - alpha) > 0 else float('inf'))
        if start == float('inf'):
            start = min(t_ready_npu, t_ready_pim)

        # 在 HYBRID 下，这个结点的对外可见输出统一是 Host/ND
        if commit:
            self._node_out_fmt[nid] = 'ND'
            self._node_host_store_end[nid] = finish   # 供后续消费者复用 Host 版本，避免重复写回


        logger.debug(f"[HYBRID] nid={nid} alpha={alpha} "
                     f"start_npu={start_npu:.6f} finish_npu={finish_npu:.6f} "
                     f"start_pim={start_pim:.6f} finish_pim={finish_pim:.6f} finish={finish:.6f}")
        return {
            'device': 'HYBRID',
            'start': start,
            'finish': finish,
            'npu': npu_dev,                     # DeviceSpec（供上层 .name）
            'pim': pim_dev,                     # DeviceSpec
            'out_dev': host,                    # 输出对外驻留在 Host
            'out_fmt': 'ND',
            'start_npu': start_npu,
            'start_pim': start_pim,
            'npu_detail': {
                'ready': t_ready_npu,
                'w_end': t_w_npu_end,
                'finish_compute': finish_npu,
            },
            'pim_detail': {
                'ready': t_ready_pim,
                'w_end': t_w_pim_end,
                'finish_compute': finish_pim,
            },
            'unified_writeback_to_host': True
        }
        return result

        
    def _ensure_host_store(self, u: str, pred_dev: DeviceSpec,
                    bytes_nd: int, src_fmt: str,
                    pred_finish: float, commit: bool) -> float:
        '''
        u 的输出已经在 Host 上以 ND 格式可用
        返回该输出在 Host 上的可用时间
        '''
        t_done = self._node_host_store_end.get(u)
        if t_done is not None:
            return t_done

        host = self.cost.get_host_device()
        size_src = self.cost.format_size(bytes_nd, src_fmt)
        t_conv_src = 0.0
        if src_fmt != 'ND':
            t_conv_src = self.cost.format_conversion_time(size_src, src_fmt, 'ND', pred_dev)

        size_nd = self.cost.format_size(bytes_nd, 'ND')

        if pred_dev.type == 'pim':
            t_mem = self.cost.activation_read_time_pim(size_nd)
            earliest = pred_finish + t_conv_src + t_mem
        else:
            earliest = pred_finish + t_conv_src

        _, t_link_end = self.comm.reserve(pred_dev.name, host.name, size_nd,
                                        earliest=earliest, commit=commit)
        t_done = t_link_end
        if commit:
            self._node_host_store_end[u] = t_done
        return t_done

    def _pim_avail_for_activation(self, dev: DeviceSpec) -> int:
        total = int(dev.mem_capacity_GB * 1e9)

        # 1) KV 预留（若 kv_in_pim=True）
        kv_reserved = self._kv_reserved_for(dev) if getattr(self.label, 'kv_in_pim', False) else 0

        # 2) 权重缓存预算（PlanLabel 分摊到本 PIM 的预算容量）
        weight_budget = self._pim_cache_capacity_for(dev)

        # 3) 已被“激活就地”占用的字节（我们自己维护，用于防止过量就地缓存）
        act_used = int(self._act_used.get(dev.name, 0))

        # 4) 计算可用
        avail = total - kv_reserved - weight_budget - act_used
        return max(0, avail)


    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        order = self._upward_rank(g, phase=phase)
        schedule: List[ScheduledTask] = []
        for nid in order:
            node = g.nodes[nid]
            allow_npu = node.allowed.get('npu', False)
            allow_pim = node.allowed.get('pim', False)
            candidates = []
            if allow_npu:
                best_npu_dev = None
                best_npu_finish = float('inf')
                for dev in self.cluster.devices_by_type('npu'):
                    _, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=False)
                    if finish < best_npu_finish:
                        best_npu_finish = finish
                        best_npu_dev = dev
                if best_npu_dev is not None:
                    candidates.append(('NPU', best_npu_finish, best_npu_dev))
            if allow_pim:
                best_pim_dev = None
                best_pim_finish = float('inf')
                for dev in self.cluster.devices_by_type('pim'):
                    _, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=False)
                    if finish < best_pim_finish:
                        best_pim_finish = finish
                        best_pim_dev = dev
                if best_pim_dev is not None:
                    candidates.append(('PIM', best_pim_finish, best_pim_dev))
            if ALLOW_HYBRID and allow_npu and allow_pim:
                best_result = self._earliest_finish_hybrid(g, nid, phase, commit=False)
                if best_result is not None:
                    candidates.append(('HYBRID', best_result['finish'], None))
            chosen_mode, chosen_finish, chosen_data = min(candidates, key=lambda x: x[1])
            if chosen_mode == 'HYBRID':
                hy = self._earliest_finish_hybrid(g, nid, phase, commit=True)
                npu = hy['npu']
                pim = hy['pim']
                start = min(hy['start_npu'], hy['start_pim'])
                finish = hy['finish']
                self.avail[npu.name] = finish
                self.avail[pim.name] = finish
                self._node_finish_time[nid] = finish
                self._node_placement[nid] = hy['out_dev'].name
                self._node_out_fmt[nid] = 'ND'
                schedule.append(ScheduledTask(nid, f'HYBRID({npu.name}+{pim.name})', start, finish))
                op_name = node.attrs.get('op') or node.name
                self.mode_mem[op_name] = 'HYBRID'
                self._after_commit_consume_predecessors(g, nid)
            else:
                dev = chosen_data
                start, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=True)
                self.avail[dev.name] = finish
                self._node_finish_time[nid] = finish
                self._node_placement[nid] = dev.name
                self._node_out_fmt[nid] = self.cost.device_preferred_fmt(dev)
                schedule.append(ScheduledTask(nid, dev.name, start, finish))
                op_name = node.attrs.get('op') or node.name
                self.mode_mem[op_name] = chosen_mode
                self._after_commit_consume_predecessors(g, nid)
            logger.debug(str(f'[Schedule] Node {nid} on {self._node_placement[nid]} from {start:.3f} to {finish:.3f} ({chosen_mode})'))
        return schedule

    def makespan(self, schedule: List[ScheduledTask]) -> float:
        return max((t.finish for t in schedule), default=0.0)

    def export_weight_stats(self):
        """
        Export weight statistics collected during scheduling passes.
        Returns a JSON-serializable dict with:
          - weight_sizes: {wid: bytes}
          - weight_load_counts: {wid: {dev_type: cnt}}
          - storage_fmt_map: the host-side storage format map used in this pass
          - host_format: buffer manager's current host_format (after syncing)
        """
        from collections import defaultdict
        by_wid = defaultdict(lambda: defaultdict(int))
        for (wid, dev_type), cnt in self._weight_load_count.items():
            by_wid[wid][dev_type] += cnt
        return {'weight_sizes': dict(self._weight_sizes), 'weight_load_counts': {wid: dict(cnts) for wid, cnts in by_wid.items()}, 'storage_fmt_map': dict(self.storage_fmt_map or {}), 'host_format': dict(self.buffer.host_format or {})}

    def set_storage_format_map(self, fmt_map: Dict[str, str]):
        self.storage_fmt_map = dict(fmt_map or {})
        for k, v in self.storage_fmt_map.items():
            self.buffer.set_host_fmt(k, v)

    # 用尽即释放：在一个结点 commit 后，视为完成对所有pre的一次“消费”
    def _after_commit_consume_predecessors(self, g: TaskGraph, nid: str) -> None:
        for u in g.predecessors(nid):
            if u not in self._act_refcnt:
                self._act_refcnt[u] = len(g.successors(u))
            self._act_refcnt[u] = max(0, self._act_refcnt[u] - 1)
            udev = self._node_placement.get(u)
            if udev and (udev, u) in self._act_resident and self._act_refcnt[u] == 0:
                bytes_kept = self._act_resident.pop((udev, u), 0)
                self._act_used[udev] = max(0, self._act_used[udev] - bytes_kept)
                logger.debug(f"[ACT] release u={u}@{udev} freed={bytes_kept} used={self._act_used[udev]}/{self._act_cap.get(udev,0)}")

    def suggest_weight_storage_formats(self) -> Dict[str, str]:
        """
        Propose a host-side weight storage format for each weight_id based on
        how often that weight is consumed by each device type during the last
        scheduling pass. We evaluate candidate formats by estimating the
        (host->device move + format-conversion) time aggregated across all
        loads observed for that weight.
        """
        from collections import defaultdict
        Base = ['ND', 'NPU_OPT', 'PIM_OPT']
        sugg: Dict[str, str] = {}

        # 统计：{wid: {dev_type: load_count}}
        by_wid = defaultdict(lambda: defaultdict(int))
        for (wid, dev_type), cnt in self._weight_load_count.items():
            by_wid[wid][dev_type] += cnt

        EPS = 1e-6
        for wid, counts in by_wid.items():
            w_bytes_nd = self._weight_sizes.get(wid, 0)
            dominant = max(counts.items(), key=lambda x: x[1])[0] if counts else 'pim'
            if dominant == 'npu':
                candidates = ['NPU_OPT', 'ND', 'PIM_OPT']
                native = 'NPU_OPT'
            elif dominant == 'pim':
                candidates = ['PIM_OPT', 'ND', 'NPU_OPT']
                native = 'PIM_OPT'
            else:
                candidates = Base
                native = 'ND'

            best_t, best_fmt = float('inf'), candidates[0]
            for fmt in candidates:
                size_src = self.cost.format_size(w_bytes_nd, fmt)
                total = 0.0
                for dev_type, cnt in counts.items():
                    devs = self.cluster.devices_by_type(dev_type)
                    if not devs:
                        continue
                    d = devs[0]
                    total += cnt * self.cost.gb_move_and_format(
                        d, size_src, fmt, self.cost.device_preferred_fmt(d)
                    )
                if total + EPS < best_t or (abs(total - best_t) < EPS and fmt == native):
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
        for cache in self.buffer.device_cache.values():
            cache.items.clear()
            cache.order.clear()
            cache.used = 0
            cache.pinned.intersection_update(cache.pinned)
        self._node_out_fmt.clear()
        self.weight_cached.clear()
        self.mode_mem.clear()
        self._act_used.clear()
        self._act_resident.clear()
        self._act_refcnt.clear()
        self._node_host_store_end.clear()

# =====================
# Additional Schedulers
# =====================
def _dev_type(name: str) -> str:
    if name is None:
        return 'cpu'
    name = str(name).lower()
    if 'npu' in name: return 'npu'
    if 'pim' in name: return 'pim'
    if 'cpu' in name: return 'cpu'
    if 'hybrid' in name: return 'hybrid'
    return 'cpu'

class _SearchSchedulerMixin(HEFTScheduler):
    """
    Mixin that provides helpers to build schedules from a fixed per-node action map.
    Each action is one of: 'npu' | 'pim' | 'cpu' | 'hybrid' (if both 'npu' and 'pim' allowed).
    """
    def _topo_order(self, g: TaskGraph) -> List[str]:
        try:
            return list(g.topological())
        except Exception:
            # fallback to HEFT order if needed
            return self._upward_rank(g, phase="prefill")

    def _allowed_actions(self, node: TaskNode) -> List[str]:
        acts = []
        for t in ['npu','pim']:
            if node.allowed.get(t, False):
                acts.append(t)
        if ALLOW_HYBRID and node.allowed.get('npu', False) and node.allowed.get('pim', False):
            acts.append('hybrid')
        return acts or ['cpu']

    def _make_initial_action_map(self, g: TaskGraph, phase: str) -> Dict[str, str]:
        """
        Run greedy HEFT once; derive an action map from its placements.
        """
        # Build a seed action map WITHOUT touching the caller's state.
        snap = self._snapshot()
        try:
            self.reset_state()
            base_sched = super().schedule(g, phase=phase)
        finally:
            # Restore whatever state the caller already had (prefill / previous tokens)
            self._restore(snap)
        actions: Dict[str, str] = {}
        for t in base_sched:
            devs = str(t.device)
            if 'HYBRID' in devs:
                actions[t.node_id] = 'hybrid'
            else:
                actions[t.node_id] = _dev_type(devs)
        return actions

    def _snapshot(self):
        return {
            'avail': copy.deepcopy(self.avail),
            'comm': copy.deepcopy(self.comm.timeline_end),
            'node_finish': copy.deepcopy(self._node_finish_time),
            'node_place': copy.deepcopy(self._node_placement),
            'node_fmt': copy.deepcopy(self._node_out_fmt),
            'weight_cached': copy.deepcopy(self.weight_cached),
            'weight_load_count': copy.deepcopy(self._weight_load_count),
            'weight_sizes': copy.deepcopy(self._weight_sizes),
            'mode_mem': copy.deepcopy(self.mode_mem),
            'act_used': copy.deepcopy(self._act_used),
            'act_resident': copy.deepcopy(self._act_resident),
            'act_refcnt': copy.deepcopy(self._act_refcnt),
            'host_store_end': copy.deepcopy(self._node_host_store_end),
            'buffer_device_cache': copy.deepcopy(self.buffer.device_cache),
            'buffer_host_fmt': copy.deepcopy(self.buffer.host_format),
        }

    def _restore(self, snap):
        self.avail = copy.deepcopy(snap['avail'])
        self.comm.timeline_end = copy.deepcopy(snap['comm'])
        self._node_finish_time = copy.deepcopy(snap['node_finish'])
        self._node_placement = copy.deepcopy(snap['node_place'])
        self._node_out_fmt = copy.deepcopy(snap['node_fmt'])
        self.weight_cached = copy.deepcopy(snap['weight_cached'])
        self._weight_load_count = copy.deepcopy(snap['weight_load_count'])
        self._weight_sizes = copy.deepcopy(snap['weight_sizes'])
        self.mode_mem = copy.deepcopy(snap['mode_mem'])
        self._act_used = copy.deepcopy(snap['act_used'])
        self._act_resident = copy.deepcopy(snap['act_resident'])
        self._act_refcnt = copy.deepcopy(snap['act_refcnt'])
        self._node_host_store_end = copy.deepcopy(snap['host_store_end'])
        self.buffer.device_cache = copy.deepcopy(snap['buffer_device_cache'])
        self.buffer.host_format = copy.deepcopy(snap['buffer_host_fmt'])


    def _evaluate_action_map(self, g: TaskGraph, phase: str, actions: Dict[str, str], dry_run: bool) -> Tuple[float, List[ScheduledTask]]:
        """
        Build a full schedule according to the provided per-node action map.
        Returns (makespan, schedule). This call resets internal state.
        """
        snap = self._snapshot() if dry_run else None
        order = self._topo_order(g)
        ready_pred = {nid: set(g.predecessors(nid)) for nid in g.nodes}
        ready = [nid for nid in order if not ready_pred[nid]]
        scheduled = []
        placed: set[str] = set()

        while ready:
            best_tuple = None  # (finish, start, nid, mode, dev)
            for nid in list(ready):
                node = g.nodes[nid]
                act = actions.get(nid)
                if act is None or act == 'auto':
                    # default to greedy (like HEFT)
                    # try both types that are allowed
                    cands = []
                    if node.allowed.get('npu', False):
                        for dev in self.cluster.devices_by_type('npu'):
                            start, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=False)
                            cands.append( (finish, start, 'npu', dev) )
                    if node.allowed.get('pim', False):
                        for dev in self.cluster.devices_by_type('pim'):
                            start, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=False)
                            cands.append( (finish, start, 'pim', dev) )
                    if node.allowed.get('cpu', False):
                        for dev in self.cluster.devices_by_type('cpu'):
                            start, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=False)
                            cands.append( (finish, start, 'cpu', dev) )
                    if ALLOW_HYBRID and node.allowed.get('npu', False) and node.allowed.get('pim', False):
                        hy = self._earliest_finish_hybrid(g, nid, phase, commit=False)
                        if hy is not None:
                            cands.append( (hy['finish'], min(hy['start_npu'], hy['start_pim']), 'hybrid', None) )
                    if not cands:
                        continue
                    f, s, mode, dev = min(cands, key=lambda x: x[0])
                else:
                    mode = act
                    if mode == 'hybrid':
                        hy = self._earliest_finish_hybrid(g, nid, phase, commit=False)
                        if hy is None:
                            # fall back to greedy among allowed single devices
                            mode = 'auto'
                            continue
                        f = hy['finish']
                        s = min(hy['start_npu'], hy['start_pim'])
                        dev = None
                    else:
                        # pick the best physical device of this type
                        devs = self.cluster.devices_by_type(mode)
                        if not devs:
                            # fallback to greedy
                            mode = 'auto'
                            continue
                        best = None
                        for d in devs:
                            st, ft = self._earliest_finish_on_device(g, nid, d, self.label, phase, commit=False)
                            if (best is None) or (ft < best[0]):
                                best = (ft, st, d)
                        if best is None:
                            continue
                        f, s, dev = best[0], best[1], best[2]
                # choose best across ready set
                cand = (f, s, nid, mode, dev)
                if (best_tuple is None) or (cand[0] < best_tuple[0]):
                    best_tuple = cand

            if best_tuple is None:
                # no schedulable node (shouldn't happen in DAG), break to avoid infinite loop
                break

            f, s, nid, mode, dev = best_tuple
            node = g.nodes[nid]
            # commit
            if mode == 'hybrid':
                hy = self._earliest_finish_hybrid(g, nid, phase, commit=True)
                start = min(hy['start_npu'], hy['start_pim'])
                finish = hy['finish']
                self.avail[hy['npu'].name] = finish
                self.avail[hy['pim'].name] = finish
                self._node_finish_time[nid] = finish
                self._node_placement[nid] = hy['out_dev'].name
                self._node_out_fmt[nid] = 'ND'
                scheduled.append(ScheduledTask(nid, f"HYBRID({hy['npu'].name}+{hy['pim'].name})", start, finish))
                op_name = node.attrs.get('op') or node.name
                self.mode_mem[op_name] = 'HYBRID'
                self._after_commit_consume_predecessors(g, nid)
            else:
                start, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=True)
                self.avail[dev.name] = finish
                self._node_finish_time[nid] = finish
                self._node_placement[nid] = dev.name
                self._node_out_fmt[nid] = self.cost.device_preferred_fmt(dev)
                scheduled.append(ScheduledTask(nid, dev.name, start, finish))
                op_name = node.attrs.get('op') or node.name
                self.mode_mem[op_name] = dev.type
                self._after_commit_consume_predecessors(g, nid)

            placed.add(nid)
            ready.remove(nid)
            # release new ready nodes
            for v in g.successors(nid):
                if v in placed:
                    continue
                deps = ready_pred[v]
                deps.discard(nid)
                if not deps and (v not in ready):
                    ready.append(v)

            cost = self.makespan(scheduled)
            if snap is not None:
                self._restore(snap)
        return cost, scheduled


class SimulatedAnnealingScheduler(_SearchSchedulerMixin):
    """
    SA over device-type assignment per node; schedule built with list scheduling.
    """
    def __init__(self, *args, sa_iters: int = 200, T0: float = 1.0, alpha: float = 0.9, flip_prob: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.sa_iters = int(sa_iters)
        self.T0 = float(T0)
        self.alpha = float(alpha)
        self.flip_prob = float(flip_prob)

    def _neighbors(self, g: TaskGraph, base: Dict[str, str]) -> Dict[str, str]:
        out = dict(base)
        for nid, node in g.nodes.items():
            if random.random() < self.flip_prob:
                acts = self._allowed_actions(node)
                if not acts:
                    continue
                old = out.get(nid, None)
                choices = [a for a in acts if a != old] or acts
                out[nid] = random.choice(choices)
        return out

    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        # initial mapping from greedy HEFT
        cur_map = self._make_initial_action_map(g, phase)
        cur_cost, cur_sched = self._evaluate_action_map(g, phase, cur_map, dry_run=True)
        best_map, best_cost, best_sched = dict(cur_map), float(cur_cost), list(cur_sched)

        T = self.T0
        for _ in range(max(1, self.sa_iters)):
            nb = self._neighbors(g, cur_map)
            nb_cost, nb_sched = self._evaluate_action_map(g, phase, nb, dry_run=True)
            delta = nb_cost - cur_cost
            accept = (delta < 0) or (random.random() < math.exp(-max(0.0, delta) / max(1e-9, T)))
            if accept:
                cur_map, cur_cost, cur_sched = nb, nb_cost, nb_sched
                if cur_cost + 1e-9 < best_cost:
                    best_map, best_cost, best_sched = dict(cur_map), float(cur_cost), list(cur_sched)
            T *= self.alpha

        # rebuild best to ensure internal state reflects it
        _, sched = self._evaluate_action_map(g, phase, best_map, dry_run=False)
        return sched


class GeneticScheduler(_SearchSchedulerMixin):
    """
    Simple GA over device-type assignment per node.
    """
    def __init__(self, *args, pop: int = 24, gens: int = 40, elite: int = 2, mut_prob: float = 0.08, cross_prob: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.pop = int(pop)
        self.gens = int(gens)
        self.elite = int(elite)
        self.mut_prob = float(mut_prob)
        self.cross_prob = float(cross_prob)

    def _random_map(self, g: TaskGraph, phase: str) -> Dict[str, str]:
        m = {}
        for nid, node in g.nodes.items():
            acts = self._allowed_actions(node)
            m[nid] = random.choice(acts)
        return m

    def _crossover(self, g: TaskGraph, a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
        out = {}
        for nid, node in g.nodes.items():
            if random.random() < self.cross_prob:
                cand = a.get(nid, None)
            else:
                cand = b.get(nid, None)
            if cand not in self._allowed_actions(node):
                acts = self._allowed_actions(node)
                cand = random.choice(acts)
            out[nid] = cand
        return out

    def _mutate(self, g: TaskGraph, m: Dict[str, str]) -> Dict[str, str]:
        out = dict(m)
        for nid, node in g.nodes.items():
            if random.random() < self.mut_prob:
                acts = self._allowed_actions(node)
                out[nid] = random.choice(acts)
        return out

    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        # seed population: greedy + randoms
        seed = self._make_initial_action_map(g, phase)
        pop: List[Tuple[float, Dict[str, str]]] = []
        # evaluate helper
        def eval_map(m: Dict[str, str]) -> float:
            cost, _ = self._evaluate_action_map(g, phase, m, dry_run=True)
            return cost

        # initial population
        pop_maps = [seed] + [self._random_map(g, phase) for _ in range(max(0, self.pop - 1))]
        for m in pop_maps:
            pop.append( (eval_map(m), m) )
        pop.sort(key=lambda x: x[0])
        best_cost, best_map = pop[0][0], dict(pop[0][1])

        for _ in range(max(1, self.gens)):
            new_pop: List[Tuple[float, Dict[str, str]]] = []
            # elitism
            elites = pop[:self.elite]
            new_pop.extend(elites)

            # tournament selection
            def select_one() -> Dict[str, str]:
                k = min(3, len(pop))
                cand = random.sample(pop, k)
                cand.sort(key=lambda x: x[0])
                return cand[0][1]

            # generate offspring
            while len(new_pop) < self.pop:
                p1 = select_one()
                p2 = select_one()
                child = self._crossover(g, p1, p2)
                child = self._mutate(g, child)
                score = eval_map(child)
                new_pop.append( (score, child) )

            new_pop.sort(key=lambda x: x[0])
            pop = new_pop[:self.pop]
            if pop[0][0] + 1e-9 < best_cost:
                best_cost, best_map = pop[0][0], dict(pop[0][1])

        # rebuild best
        _, sched = self._evaluate_action_map(g, phase, best_map, dry_run=False)
        return sched


class RLScheduler(_SearchSchedulerMixin):
    """
    Simple epsilon-greedy bandit RL by operator type.
    State = op kind; Action = device-type; Reward = negative task duration (finish - start).
    Trains for a small number of episodes, then schedules greedily.
    """
    def __init__(self, *args, episodes: int = 30, epsilon_start: float = 0.3, epsilon_end: float = 0.05, alpha: float = 0.3, gamma: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.episodes = int(episodes)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.Q: Dict[Tuple[str, str], float] = {}

    def _op_key(self, node: TaskNode) -> str:
        return str(node.attrs.get('op') or node.name)

    def _eps(self, step: int) -> float:
        if self.episodes <= 1:
            return self.epsilon_end
        r = step / float(max(1, self.episodes - 1))
        return self.epsilon_start * (1 - r) + self.epsilon_end * r

    def _pick_action(self, node: TaskNode, eps: float) -> str:
        acts = self._allowed_actions(node)
        if (random.random() < eps) or all(self.Q.get((self._op_key(node), a), 0.0) == 0.0 for a in acts):
            return random.choice(acts)
        # greedy by Q
        best_a, best_q = None, -1e30
        for a in acts:
            q = self.Q.get((self._op_key(node), a), 0.0)
            if q > best_q:
                best_q, best_a = q, a
        return best_a or random.choice(acts)

    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        order = self._upward_rank(g, phase=phase)
        # training episodes
        base_snap = self._snapshot()
        for ep in range(max(1, self.episodes)):
            eps = self._eps(ep)
            self._restore(base_snap)
            prev_op = None
            prev_a = None
            prev_q = None
            for nid in order:
                node = g.nodes[nid]
                a = self._pick_action(node, eps)
                # evaluate choice to get (start, finish) without committing
                cand_s_f = []
                if a == 'hybrid':
                    hy = self._earliest_finish_hybrid(g, nid, phase, commit=False)
                    if hy is not None:
                        cand_s_f.append( (min(hy['start_npu'], hy['start_pim']), hy['finish'], None, 'hybrid') )
                else:
                    for d in self.cluster.devices_by_type(a):
                        st, ft = self._earliest_finish_on_device(g, nid, d, self.label, phase, commit=False)
                        cand_s_f.append( (st, ft, d, a) )
                if not cand_s_f:
                    continue
                # choose the specific device with min finish
                st, ft, d, a = min(cand_s_f, key=lambda x: x[1])
                reward = -(ft - st)
                # TD(0) update for previous state-action with max over current state's actions
                if prev_op is not None and prev_a is not None:
                    key_prev = (prev_op, prev_a)
                    # bootstrap target
                    q_next_best = max((self.Q.get((self._op_key(node), act), 0.0) for act in self._allowed_actions(node)), default=0.0)
                    target = prev_q + self.alpha * ((reward + self.gamma * q_next_best) - prev_q)
                    self.Q[key_prev] = target
                # commit the chosen execution
                if a == 'hybrid':
                    hy = self._earliest_finish_hybrid(g, nid, phase, commit=True)
                    self.avail[hy['npu'].name] = hy['finish']
                    self.avail[hy['pim'].name] = hy['finish']
                else:
                    st, ft = self._earliest_finish_on_device(g, nid, d, self.label, phase, commit=True)
                    self.avail[d.name] = ft
                prev_op = self._op_key(node)
                prev_a = a
                prev_q = self.Q.get((prev_op, prev_a), 0.0)
        # exploitation: derive deterministic action map and build final schedule
        action_map: Dict[str, str] = {}
        for nid, node in g.nodes.items():
            best_a = None
            best_q = -1e30
            for a in self._allowed_actions(node):
                q = self.Q.get((self._op_key(node), a), 0.0)
                if q > best_q:
                    best_q, best_a = q, a
            action_map[nid] = best_a or random.choice(self._allowed_actions(node))
        self._restore(base_snap)
        _, sched = self._evaluate_action_map(g, phase, action_map, dry_run=False)
        return sched


class AStarBeamScheduler(_SearchSchedulerMixin):
    """
    Beam-A* over partial schedules.
    Each node expansion chooses an action among allowed device-types (including 'hybrid' when allowed).
    We maintain a small beam of partial states ordered by g + h, where:
      g = current max finish time (makespan so far),
      h = optimistic lower bound: max remaining node rank_u (avg compute + avg comm).
    """
    @dataclass
    class _State:
        idx: int
        order: List[str]
        g_end: float
        actions: Dict[str, str]
        scheduled: List[ScheduledTask]
        snapshot: Any  # opaque snapshot of internal state

    def __init__(self, *args, beam: int = 5, max_expansions: int = 200, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam = int(beam)
        self.max_expansions = int(max_expansions)

    def _snapshot(self):
        # Deep snapshot of key mutable state so we can branch.
        return {
            'avail': copy.deepcopy(self.avail),
            'comm': copy.deepcopy(self.comm.timeline_end),
            'node_finish': copy.deepcopy(self._node_finish_time),
            'node_place': copy.deepcopy(self._node_placement),
            'node_fmt': copy.deepcopy(self._node_out_fmt),
            'weight_cached': copy.deepcopy(self.weight_cached),
            'weight_load_count': copy.deepcopy(self._weight_load_count),
            'weight_sizes': copy.deepcopy(self._weight_sizes),
            'mode_mem': copy.deepcopy(self.mode_mem),
            'act_used': copy.deepcopy(self._act_used),
            'act_resident': copy.deepcopy(self._act_resident),
            'act_refcnt': copy.deepcopy(self._act_refcnt),
            'host_store_end': copy.deepcopy(self._node_host_store_end),
            'buffer_device_cache': copy.deepcopy(self.buffer.device_cache),
            'buffer_host_fmt': copy.deepcopy(self.buffer.host_format),
        }

    def _restore(self, snap):
        self.avail = copy.deepcopy(snap['avail'])
        self.comm.timeline_end = copy.deepcopy(snap['comm'])
        self._node_finish_time = copy.deepcopy(snap['node_finish'])
        self._node_placement = copy.deepcopy(snap['node_place'])
        self._node_out_fmt = copy.deepcopy(snap['node_fmt'])
        self.weight_cached = copy.deepcopy(snap['weight_cached'])
        self._weight_load_count = copy.deepcopy(snap['weight_load_count'])
        self._weight_sizes = copy.deepcopy(snap['weight_sizes'])
        self.mode_mem = copy.deepcopy(snap['mode_mem'])
        self._act_used = copy.deepcopy(snap['act_used'])
        self._act_resident = copy.deepcopy(snap['act_resident'])
        self._act_refcnt = copy.deepcopy(snap['act_refcnt'])
        self._node_host_store_end = copy.deepcopy(snap['host_store_end'])
        self.buffer.device_cache = copy.deepcopy(snap['buffer_device_cache'])
        self.buffer.host_format = copy.deepcopy(snap['buffer_host_fmt'])

    def _heuristic_remaining(self, g: TaskGraph, phase: str, remaining: Iterable[str]) -> float:
        # optimistic: take max of upward ranks among remaining nodes
        ru_order = self._upward_rank(g, phase=phase)
        rank_u = {nid: (len(ru_order)-i) for i, nid in enumerate(ru_order)}  # fallback monotonic priority if needed
        try:
            # better: recompute precise rank_u subset using helper already in HEFT
            ru_sorted = self._upward_rank(g, phase=phase)
            # Create a numeric map where earlier-in-ru_sorted means larger remaining work.
            rank_u = {nid: (len(ru_sorted) - idx) for idx, nid in enumerate(ru_sorted)}
        except Exception:
            pass
        if not remaining:
            return 0.0
        # scale by average per-node cost proxy using cost model
        avg_cost = 0.0
        k = 0
        for nid in remaining:
            node = g.nodes[nid]
            avg_cost += self._avg_compute_cost(node, phase)
            k += 1
        avg_cost = (avg_cost / k) if k else 0.0
        return max((rank_u.get(nid, 1) for nid in remaining)) * (avg_cost * 0.5)

    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        order = self._topo_order(g)
        init = AStarBeamScheduler._State(
            idx=0,
            order=order,
            g_end=0.0,
            actions={},
            scheduled=[],
            snapshot=self._snapshot()
        )
        beam: List[AStarBeamScheduler._State] = [init]
        expansions = 0

        while beam and expansions < self.max_expansions:
            # expand current beam by one node decision
            new_beam: List[AStarBeamScheduler._State] = []
            for st in beam:
                if st.idx >= len(st.order):
                    new_beam.append(st)
                    continue
                nid = st.order[st.idx]
                node = g.nodes[nid]
                acts = self._allowed_actions(node)
                # branch per action
                for a in acts:
                    # restore snapshot
                    self._restore(st.snapshot)
                    # choose specific device (for non-hybrid)
                    if a == 'hybrid':
                        hy = self._earliest_finish_hybrid(g, nid, phase, commit=True)
                        if hy is None:
                            continue
                        finish = hy['finish']
                        self.avail[hy['npu'].name] = finish
                        self.avail[hy['pim'].name] = finish
                        sched_item = ScheduledTask(nid, f"HYBRID({hy['npu'].name}+{hy['pim'].name})",
                                                   min(hy['start_npu'], hy['start_pim']), finish)
                    else:
                        # pick best device of type a
                        best = None
                        for d in self.cluster.devices_by_type(a):
                            st0, ft0 = self._earliest_finish_on_device(g, nid, d, self.label, phase, commit=False)
                            if (best is None) or (ft0 < best[0]):
                                best = (ft0, st0, d)
                        if best is None:
                            continue
                        _, _, d = best
                        st0, ft0 = self._earliest_finish_on_device(g, nid, d, self.label, phase, commit=True)
                        self.avail[d.name] = ft0
                        sched_item = ScheduledTask(nid, d.name, st0, ft0)

                    # update deps and snapshot
                    self._after_commit_consume_predecessors(g, nid)
                    snap2 = self._snapshot()
                    actions2 = dict(st.actions)
                    actions2[nid] = a
                    scheduled2 = list(st.scheduled) + [sched_item]
                    g_end2 = self.makespan(scheduled2)
                    # optimistic h over remaining nodes
                    remaining = st.order[st.idx+1:]
                    h = self._heuristic_remaining(g, phase, remaining)
                    fscore = g_end2 + h
                    new_beam.append(AStarBeamScheduler._State(
                        idx=st.idx+1, order=st.order, g_end=g_end2,
                        actions=actions2, scheduled=scheduled2, snapshot=snap2
                    ))
                    expansions += 1
                    if expansions >= self.max_expansions:
                        break
                if expansions >= self.max_expansions:
                    break
            # keep top beam by g_end + heuristic (recompute to avoid carrying 'h' around)
            new_beam.sort(key=lambda s0: s0.g_end + self._heuristic_remaining(g, phase, s0.order[s0.idx:]))
            beam = new_beam[:self.beam]

        # pick best finished or most advanced state and finalize (ensure internal state matches)
        best = min(beam, key=lambda s0: (s0.g_end + self._heuristic_remaining(g, phase, s0.order[s0.idx:]))) if beam else init
        # If partial, complete greedily with our helper using chosen actions (fallback to auto for missing)
        actions = best.actions
        for nid in g.nodes:
            actions.setdefault(nid, 'auto')
        _, sched = self._evaluate_action_map(g, phase, actions, dry_run=False)
        return sched
