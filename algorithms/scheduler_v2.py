from __future__ import annotations
import os
from config import attach_local_debug_filter
from dataclasses import dataclass,field
from typing import Dict, List, Tuple, Optional, Any, Iterable, OrderedDict, Hashable, Mapping
from collections import defaultdict, deque
from collections import ChainMap
from collections.abc import Hashable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from plan_label import PlanLabel
from hardware import Cluster, DeviceSpec
from task_graph import TaskGraph, TaskNode
from cost_model import CostModel
from buffer_manager import GlobalMemoryManager, LRUCache
from config import ALLOW_HYBRID, RANKU_INCLUDE_AVG_WEIGHT_LOAD, PIM_RUNTIME_LRU_THRESHOLD
import logging
import math
import random
import copy
from stats_recorder import StatsRecorder

_MISSING = object()
DEBUG_SCHEDULER = True
logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: DEBUG_SCHEDULER)
@dataclass
class _GraphIndex:
    nodes: tuple
    nodes_set: frozenset
    preds: dict
    succs: dict
    topo: tuple
    rev_topo: tuple
    order_by_phase: dict = field(default_factory=dict)        # phase -> tuple[nid...]
    allowed_actions: dict = field(default_factory=dict)       # nid -> tuple[action...]
class SchedulerBase:
    def _get_graph_index(self, g):
        cache = getattr(self, "_graph_index_cache", None)
        if cache is None:
            cache = self._graph_index_cache = {}
        gid = id(g)
        idx = cache.get(gid)
        if idx is not None:
            return idx
        nodes_iter = g.nodes.keys() if hasattr(g.nodes, "keys") else g.nodes
        nodes = tuple(nodes_iter)
        preds = {nid: tuple(g.predecessors(nid)) for nid in nodes}
        succs = {nid: tuple(g.successors(nid)) for nid in nodes}
        topo = tuple(g.topological())
        idx = _GraphIndex(
            nodes=nodes,
            nodes_set=frozenset(nodes),
            preds=preds,
            succs=succs,
            topo=topo,
            rev_topo=tuple(reversed(topo)),
        )
        cache[gid] = idx
        return idx

    def _allowed_actions_by_id(self, g, nid):
        idx = self._get_graph_index(g)
        acts = idx.allowed_actions.get(nid)
        if acts is not None:
            return acts
        out = self._allowed_actions(g.nodes[nid])
        acts = tuple(out) if out else tuple()
        idx.allowed_actions[nid] = acts
        return acts

    def _avg_compute_cost_cached(self, g, nid, phase: str) -> float:
        cache = getattr(self, "_avg_compute_cache", None)
        if cache is None:
            cache = self._avg_compute_cache = {}
        key = (id(g), phase, nid, int(getattr(self, 'seq_len', 0) or 0))
        v = cache.get(key)
        if v is not None:
            return v
        v = float(self._avg_compute_cost(g.nodes[nid], phase=phase))
        cache[key] = v
        return v

    def _avg_comm_cost_cached(self, g, u, v, phase: str) -> float:
        cache = getattr(self, "_avg_comm_cache", None)
        if cache is None:
            cache = self._avg_comm_cache = {}
        key = (id(g), phase, u, v, int(getattr(self, 'seq_len', 0) or 0))
        val = cache.get(key)
        if val is not None:
            return val
        val = float(self._avg_comm_cost(g.nodes[u], g.nodes[v], phase))
        cache[key] = val
        return val

@dataclass
class ScheduledTask:
    node_id: str
    device: str
    start: float
    finish: float


@dataclass(frozen=True)
class JointNodeMeta:
    """Metadata for nodes in a joint (prefill + multi-step decode) graph."""
    base_id: str
    phase: str          # 'prefill' or 'decode'
    step: int           # 0 for prefill, 1..decode_steps
    seq_len: int        # seq_len used for cost model at this step


class JointTaskGraph:
    """A minimal TaskGraph-like wrapper used by joint scheduling.

    It provides the subset of the TaskGraph API that the schedulers rely on:
      - .nodes (dict-like)
      - predecessors(nid)
      - successors(nid)
      - topological()

    Note: We intentionally keep it lightweight so it can be created without
    depending on TaskGraph internals.
    """

    def __init__(
        self,
        nodes: Dict[str, TaskNode],
        preds: Dict[str, Tuple[str, ...]],
        succs: Dict[str, Tuple[str, ...]],
        topo: Tuple[str, ...],
        meta: Dict[str, JointNodeMeta],
    ):
        self.nodes = nodes
        self._preds = preds
        self._succs = succs
        self._topo = topo
        self.meta = meta

    def predecessors(self, nid: str) -> Tuple[str, ...]:
        return self._preds.get(nid, ())

    def successors(self, nid: str) -> Tuple[str, ...]:
        return self._succs.get(nid, ())

    def topological(self) -> List[str]:
        return list(self._topo)
class CommManager:
    """
    Maintain independent timelines per (src, dst) channel.
    """

    def __init__(self, cluster: Cluster, stats: StatsRecorder | None = None):
        self.cluster = cluster
        self.timeline_end: Dict[Tuple[str, str], float] = {}
        self.stats = stats

    def reserve(self, src: str, dst: str, bytes_amount: int, earliest: float, commit: bool=True, tag: str | None = None):
        key = (src, dst)
        bw = self.cluster.get_link_bw(src, dst) * 1024**3
        ch_end = self.timeline_end.get(key, 0.0)
        start = max(ch_end, earliest)
        dt = 0.0 if bw <= 0 else bytes_amount / bw
        end = start + dt
        if commit:
            self.timeline_end[key] = end
            # logger = logging.getLogger(__name__)
            # logger.debug(
            #     f"[COMM] {src}->{dst} bytes={bytes_amount} bw={bw/1e9:.2f}GB/s "
            #     f"start={start:.6f} end={end:.6f} dt={end-start:.6f} commit={commit}"
            # )

            src_dev = self.cluster.devices.get(src)
            dst_dev = self.cluster.devices.get(dst)

            if self.stats is not None:
                try:
                    self.stats.log_comm(
                        src=src, dst=dst, bytes=bytes_amount,
                        start=float(start), end=float(end),
                        tag=tag or 'comm'
                    )
                except AttributeError:
                    pass
        return (start, end)


class HEFTScheduler(SchedulerBase):

    def __init__(self, cluster: Cluster, cost: CostModel, label: PlanLabel, batch: int, seq_len: int, buffer: GlobalMemoryManager):
        self.cluster = cluster
        self.cost = cost
        self.label = label
        self.batch = batch
        self.seq_len = seq_len
        self.buffer = buffer or GlobalMemoryManager()
        self._pim_cache_capacity: Dict[str, int] = {}
        self._pim_lru_threshold_bytes: Dict[str, int] = {}
        self._node_host_store_end: Dict[str, float] = {}  #节点输出在host上的可用时间
        self._act_cap: Dict[str, int] = {}                 # 每个设备用于激活的容量上限（字节），约 90% 可用
        self._act_used: Dict[str, int] = defaultdict(int)  # 当前每个设备已被激活占用的字节数
        self._act_resident: Dict[Tuple[str, str], int] = {}# (dev.name, node_id) -> bytes，表示该结点输出是否仍驻留在该设备
        self._act_refcnt: Dict[str, int] = {}              # node_id -> 还剩多少个下游消费者未消费
        self._max_util: float = 0.90                       # 最多只能占满 90%
        self._kv_reserved_per_dev: Dict[str, int] = {}     # PIM 每块设备为 KV 预留的容量（字节）
        self._rank_cache: Dict[Tuple[int, str], List[str]] = {} #parallel 新增
        self._topo_cache: Dict[int, List[str]] = {} #parallel 新增
        self.stats = StatsRecorder()
        self.comm = CommManager(cluster, stats=self.stats)
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
        self._kv_blocks_lru: Dict[str, "OrderedDict[Hashable, int]"] = defaultdict(OrderedDict) # 每个 PIM device 上的 KV 块 LRU：dev_name -> OrderedDict[block_key, bytes]
        self._kv_used_bytes: Dict[str, int] = defaultdict(int) # 每个 PIM device 上 KV 块总占用字节数
        self._last_weight_cache_event: Dict[str, Dict[str, Any]] = {}  # dev_name -> event dict
        self._npu_attached_pimds: Dict[str, List[DeviceSpec]] = defaultdict(list)
        self._pim_trace: List[Dict[str, Any]] = []
        total_budget = int(getattr(self.label, 'pim_weight_capacity_bytes', 0) or 0)
        pim_devs = self.cluster.devices_by_type('pim')

        if pim_devs:
            n_dev = len(pim_devs)
            share = total_budget // n_dev
            remainder = total_budget % n_dev
            for idx, d in enumerate(pim_devs):
                cap = share + (1 if idx < remainder else 0)
                max_dev_bytes = int(d.mem_capacity_GB * 1000000000.0)
                cap_bytes = min(max_dev_bytes, cap)
                self._pim_cache_capacity[d.name] = cap_bytes
                self._pim_lru_threshold_bytes[d.name] = int(cap_bytes * float(PIM_RUNTIME_LRU_THRESHOLD))
                # logger.debug(
                #     "[PIM-INIT] dev=%s weight_cache_cap=%dB lru_threshold=%dB",
                #     d.name, cap_bytes, self._pim_lru_threshold_bytes[d.name],
                # )
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
            caps = [int(d.mem_capacity_GB * 1024**3) for d in pim_devs]
            cap_sum = max(1, sum(caps))
            alloc = [ (caps[i] * kv_total_bytes) // cap_sum for i in range(len(pim_devs)) ]
            remainder = kv_total_bytes - sum(alloc)
            order = sorted(range(len(pim_devs)), key=lambda i: caps[i], reverse=True)
            for i in range(remainder):
                alloc[order[i % len(order)]] += 1
            for i, d in enumerate(pim_devs):
                self._kv_reserved_per_dev[d.name] = alloc[i]
                # logger.debug(f"[INIT] KV_RESERVE dev={d.name} = {alloc[i]}")
        else:
            for d in pim_devs:
                self._kv_reserved_per_dev[d.name] = 0

        # --- Compute activation residency budget for ALL devices (90%) ---
        for name, d in self.cluster.devices.items():
            phy = int(d.mem_capacity_GB * 1024**3)
            kv_reserve = int(self._kv_reserved_per_dev.get(name, 0)) if d.type == 'pim' else 0
            weight_budget = int(self._pim_cache_capacity.get(name, 0)) if d.type == 'pim' else 0
            cap = max(0, int(self._max_util * max(0, phy - kv_reserve - weight_budget)))
            self._act_cap[name] = cap
            # logger.debug(f"[INIT] ACT_CAP dev={name} phy={phy} kv={kv_reserve} weight_budget={weight_budget} act_cap(90%)={cap}")

        for d in self.cluster.devices_by_type('pim'):
            ptype = (getattr(d, 'pim_type', None) or 'accel').lower()
            attached = getattr(d, 'attached_npu', None)
            if ptype in ('dram', 'hbm') and attached:
                self._npu_attached_pimds[attached].append(d)
                
    @property
    def pim_trace(self) -> List[Dict[str, Any]]:
        return self._pim_trace

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
        devs = [d for d in self.cluster.devices.values() if d.type in ('npu', 'pim')]  # 只考虑真正会运行算子的设备
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
                
                if di.type == dj.type:
                    # 同类型设备（比如两个 NPU 或两个 PIM），直接 comm_cost
                    t_link = self.cost.comm_cost(di, dj, payload_src)
                else:
                    # 不同类型，允许2种类型传输
                    host = self.cost.get_host_device()
                    t_host = (
                        self.cost.comm_cost(di, host, payload_src) +
                        self.cost.comm_cost(host, dj, payload_src)
                    )
                    try:
                        t_direct = self.cost.link_time(payload_src, di, dj)
                        if t_direct <= 0:
                            t_direct = float('inf')
                    except Exception:
                        t_direct = float('inf')

                    t_link = min(t_host, t_direct)

                t_conv = 0.0
                if di.type != dj.type:
                    t_conv = self.cost.format_conversion_time(payload_src, src_fmt, dst_fmt, dj)
                total += max(t_link, t_conv)
                k += 1
        return total / k if k else 0.0

    def _upward_rank(self, g: TaskGraph, phase: str):
        idx = self._get_graph_index(g)
        cached = idx.order_by_phase.get(phase)
        if cached is not None:
            return list(cached)

        succ = idx.succs
        order = idx.rev_topo
        rank_u = {}
        for nid in order:
            if not succ[nid]:
                compute_cost = self._avg_compute_cost_cached(g, nid, phase)
                rank_u[nid] = compute_cost
            else:
                compute_cost = self._avg_compute_cost_cached(g, nid, phase)
                best = 0.0
                for v in succ[nid]:
                    comm_cost = self._avg_comm_cost_cached(g, nid, v, phase)
                    path_cost = comm_cost + rank_u[v]
                    if path_cost > best:
                        best = path_cost
                rank_u[nid] = compute_cost + best
        sorted_nodes = tuple(sorted(idx.nodes, key=lambda x: -rank_u[x]))
        idx.order_by_phase[phase] = sorted_nodes
        return list(sorted_nodes)
    def _pim_used_bytes(self, dev: DeviceSpec):
        """返回 (phy_bytes, kv_used, act_used, weight_used, total_used)."""
        dev_name = dev.name
        phy_bytes = int(getattr(dev, "mem_capacity_GB", 0.0) * 1024**3)

        kv_used = int(self._kv_used_bytes.get(dev_name, 0))
        act_used = int(self._act_used.get(dev_name, 0))

        cache = self.buffer.device_cache.get(dev_name)
        weight_used = int(cache.used) if cache is not None else 0

        total_used = kv_used + act_used + weight_used
        return phy_bytes, kv_used, act_used, weight_used, total_used
    
    def _record_pim_trace(self, dev:DeviceSpec, *, phase:str, finish: float, event:str, node: Optional[TaskNode]=None, extra: Optional[Dict[str,Any]]=None):
        if dev.type != 'pim':
            return
        phy, kv_used, act_used, weight_used, total_used = self._pim_used_bytes(dev)
        seq_len = int(getattr(self, "seq_len", 0) or 0)
        rec: Dict[str, Any] = {
            "device": dev.name,
            "phase": phase,
            "finish": float(finish),
            "seq_len": seq_len,
            "kv_used_bytes": int(kv_used),
            "act_used_bytes": int(act_used),
            "weight_used_bytes": int(weight_used),
            "total_used_bytes": int(total_used),
            "event": str(event),
        }
        if node is not None:
            nid = getattr(node, "id", None) or getattr(node, "name", None)
            if nid is not None:
                rec["node_id"] = str(nid)

        if extra:
            for k, v in extra.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    rec[k] = v
                else:
                    rec[k] = v  # 比如 list / tuple，json 也能直接吃

        self._pim_trace.append(rec)


    def _kv_block_lru_touch(
        self,
        dev: DeviceSpec,
        block_key: Hashable,
        block_bytes: int,
        *,
        touch_only: bool = False,
    )  -> Dict[str, Any]:
        """
        在 PIM 上按块管理 KV cache：
        - touch_only=True 只更新 LRU 顺序，不改变占用大小；
        - 否则视为写入一个大小为 block_bytes 的新块，
          如有需要并且总占用超过 PIM_RUNTIME_LRU_THRESHOLD，则淘汰最老的 KV 块。
        """
        if dev.type != 'pim':
            return

        dev_name = dev.name
        lru = self._kv_blocks_lru[dev_name]
        kv_used = int(self._kv_used_bytes.get(dev_name, 0))
        info: Dict[str, Any] = {
            "kv_block_key": block_key,
            "kv_block_bytes": int(block_bytes),
            "kv_before_bytes": kv_used,
            "kv_evicted": [],
        }

        #case 1: kv cache hit 
        if block_key in lru:
            size0 = lru.pop(block_key)
            lru[block_key] = size0  # 挪到队尾代表最近使用
            # if logger.isEnabledFor(logging.DEBUG):
            #     logger.debug("[KV-LRU] dev=%s block=%r hit size=%d kv_used=%dB",dev_name, block_key, size0, kv_used)
            info["kind"] = "hit"
            info["kv_after_bytes"] = kv_used
            info["kv_delta_bytes"] = 0
            return info
        
        #case 2: touch 不分配也不驱逐
        if touch_only:   
            # if logger.isEnabledFor(logging.DEBUG):
            #     logger.debug("[KV-LRU] dev=%s block=%r touch_only(no_write) kv_used=%dB",
            #                 dev_name, block_key, kv_used)
            info["kind"] = "touch_only"
            info["kv_block_bytes"] = 0
            info["kv_after_bytes"] = kv_used
            info["kv_delta_bytes"] = 0
            return info

        #case 3: 写miss 但size<=0
        if block_bytes <= 0:
            info["kind"] = "skip"
            info["kv_after_bytes"] = kv_used
            info["kv_delta_bytes"] = 0
            return info        
        
        # case 4:写miss 可能触发驱逐
        victims: List[Tuple[Hashable, int]] = []
        phy_bytes, _, act_used, weight_used, total_used = self._pim_used_bytes(dev)
        if phy_bytes > 0 and PIM_RUNTIME_LRU_THRESHOLD > 0.0:
            threshold = int(PIM_RUNTIME_LRU_THRESHOLD * phy_bytes)
            target_total = max(0, threshold - block_bytes)
            max_kv_after = max(0, target_total - act_used - weight_used)
            while lru and kv_used > max_kv_after:
                victim, v_size = lru.popitem(last=False)
                kv_used -= int(v_size)
                victims.append((victim, int(v_size)))
                # if logger.isEnabledFor(logging.DEBUG):
                #     logger.debug("[KV-LRU] dev=%s evict block=%r size=%d new_kv_used=%dB",
                #                 dev_name, victim, v_size, kv_used)

        lru[block_key] = int(block_bytes)
        new_used = kv_used + int(block_bytes)
        self._kv_used_bytes[dev_name] = new_used
        info["kind"] = "insert"
        info["kv_evicted"] = victims
        info["kv_after_bytes"] = new_used
        info["kv_delta_bytes"] = new_used - info["kv_before_bytes"]
        return info        



    def _update_kv_cache_for_node(self, node: TaskNode, dev: DeviceSpec, phase: str, commit: bool, finish: float | None = None) -> None:
        if not commit:
            return

        # Determine target PIM device
        target_dev = dev
        if dev.type != 'pim':
            kv_home = getattr(self.label, 'kv_home', 'host').lower()
            if kv_home == 'pim':
                pim_devs = self.cluster.devices_by_type('pim')
                if not pim_devs:
                    return
                
                found_pim = None
                if dev.type == 'npu':
                    attached_list = self._npu_attached_pimds.get(dev.name, [])
                    if not attached_list:
                        for pd in pim_devs:
                            ptype = (getattr(pd, 'pim_type', None) or 'accel').lower()
                            if ptype in ('dram', 'hbm') and getattr(pd, 'attached_npu', None) == dev.name:
                                attached_list.append(pd)
                        if attached_list:
                            self._npu_attached_pimds[dev.name].extend(attached_list)
                    
                    if attached_list:
                        found_pim = attached_list[0]
                
                if found_pim is None:
                    found_pim = pim_devs[0]
                target_dev = found_pim
            else:
                return

        if target_dev.type != 'pim':
            return
        
        # Use target_dev as the PIM device to update
        dev = target_dev

        name = (getattr(node, 'name', '') or '').upper()
        if name not in ('K_READ', 'V_READ', 'K_WRITE', 'V_WRITE', 'KV_READ', 'KV_WRITE'):
            return

        attrs = getattr(node, 'attrs', {}) or {}
        if 'kv_block_id' in attrs:
            block_key: Hashable = attrs['kv_block_id']
        else:
            layer = attrs.get('layer', None)
            # decode 阶段用当前 seq_len 作为一个近似的 block index；
            # prefill 阶段统一记为 0。
            step_idx = self.seq_len if phase == 'decode' else 0
            block_key = (layer, step_idx, name)
        
        # if logger.isEnabledFor(logging.DEBUG):
        #     logger.debug(
        #         "[KV-BLOCK] phase=%s node=%s dev=%s op=%s block_key=%r seq_len=%d",
        #         phase, getattr(node, 'name', '?'), dev.name, name, block_key, self.seq_len
        #     )

        # 当前算子的 KV 读写字节，用于估算块大小
        rd_bytes, wr_bytes = self.cost.estimate_activation_bytes(
            node,
            batch=int(getattr(self, 'batch', 1) or 1),
            seq_len=int(getattr(self, 'seq_len', 1) or 1),
            phase=phase,
        )
        if name in ('K_WRITE', 'V_WRITE', 'KV_WRITE'):
            block_bytes = max(int(wr_bytes), 0)
            kv_info = self._kv_block_lru_touch(dev, block_key, block_bytes, touch_only=False)
        else:  # READ 类
            kv_info = self._kv_block_lru_touch(dev, block_key, 0, touch_only=True)

        if finish is not None:
            if name in ('K_WRITE', 'V_WRITE', 'KV_WRITE'):
                event = 'KV_BLOCK_WRITE'
            else:
                event = 'KV_BLOCK_READ'
            extra = dict(kv_info or {})
            if "kv_block_key" in extra:
                extra["kv_block_key"] = repr(extra["kv_block_key"]) #repr把tuple变成str
            if "kv_evicted" in extra:
                extra["kv_evicted"] = [
                    (repr(k), int(sz)) for (k, sz) in extra["kv_evicted"]
                ]
            self._record_pim_trace(dev, phase=phase, finish=float(finish), event=event, node=node, extra=extra)

    def _weight_load_time(self, node: TaskNode, dev: DeviceSpec, t0: float, commit: bool) -> float:
        """Host->dev load + format conversion; overlappable with compute."""
        if not node.weight_id or node.weight_size <= 0:
            return 0.0
        wid = node.weight_id
        lru_threshold = int(self._pim_lru_threshold_bytes.get(dev.name, 0))
        cache_cap     = int(self._pim_cache_capacity.get(dev.name, 0))
        if dev.type == 'pim' and self.buffer.is_cached(dev.name, wid):
            # logger.debug(f"[WEIGHT] cache-hit wid={wid} dev={dev.name}")
            if commit:
                self.buffer.device_cache[dev.name].touch(wid)
            return 0.0
        if dev.type == 'pim':
            load_time = self.cost.weight_load_time_pim(node.weight_size)
            if commit:
                self._weight_load_count[wid, dev.type] += 1
                self._weight_sizes[wid] = node.weight_size
                self.weight_cached[dev.name, wid] = True
                pinned_flag = bool(getattr(self.label, 'pinned_fc_on_pim', set())
                                and (wid in self.label.pinned_fc_on_pim))

                cache = self.buffer.device_cache.get(dev.name, None)
                before_used = int(getattr(cache, "used", 0))
                before_items = set(getattr(cache, "items", {}).keys()) if cache is not None else set()

                if cache_cap > 0:
                    used_bytes = int(self.buffer.device_cache[dev.name].used)
                    if lru_threshold <= 0 or used_bytes > lru_threshold:
                        # 当前权重存入，触发驱逐逻辑
                        self.buffer.mark_cached(dev.name, wid, node.weight_size, pinned=pinned_flag)
                    else:
                        # 未超过阈值，只占用空间
                        self.buffer.mark_cached(dev.name, wid, node.weight_size, pinned=pinned_flag)

                cache = self.buffer.device_cache.get(dev.name, None)
                after_used = int(getattr(cache, "used", 0))
                after_items = set(getattr(cache, "items", {}).keys()) if cache is not None else set()

                evicted = list(before_items - after_items)
                inserted = list(after_items - before_items)

                self._last_weight_cache_event[dev.name] = {
                    "weight_id": wid,
                    "weight_bytes": int(node.weight_size),
                    "weight_cache_before": before_used,
                    "weight_cache_after": after_used,
                    "weight_cache_delta": after_used - before_used,
                    "weight_evicted": [str(x) for x in evicted],
                    "weight_inserted": [str(x) for x in inserted],
                }

                if wid not in self.buffer.host_format and wid in self.storage_fmt_map:
                    self.buffer.set_host_fmt(wid, self.storage_fmt_map[wid])
            return load_time

        host = self.cost.get_host_device().name
        stored_fmt = self.storage_fmt_map.get(wid, self.buffer.get_host_fmt(wid) or 'ND')
        size_src = self.cost.format_size(node.weight_size, stored_fmt)
        # logger.debug(f"[WEIGHT] host->{dev.name} load wid={wid} stored_fmt={stored_fmt} size={node.weight_size}")
        _, link_end = self.comm.reserve(host, dev.name, size_src, earliest=t0, commit=commit)
        conv_t = self.cost.format_conversion_time(size_src, stored_fmt, self.cost.device_preferred_fmt(dev), dev)
        end = link_end + conv_t
        if commit:
            self._weight_load_count[wid, dev.type] += 1
            self._weight_sizes[wid] = node.weight_size
            if wid not in self.buffer.host_format and wid in self.storage_fmt_map:
                self.buffer.set_host_fmt(wid, self.storage_fmt_map[wid])
        return max(0.0, end - t0)
    
    def _kv_load_time(self, node: TaskNode, dev: DeviceSpec, t0: float, phase: str, commit: bool) -> float:
        name = (getattr(node, 'name', '') or '').upper()
        if name not in ("K_READ","V_READ","KV_READ"):
            return 0.0
        
        kv_home = getattr(self.label, 'kv_home', 'host').lower()
        dst_dev = dev
        src_dev: Optional[DeviceSpec] = None
        src_name: Optional[str] = None
        
        # ========= Case 1: KV in PIM =========
        if kv_home == 'pim':
            pim_devs = self.cluster.devices_by_type('pim')
            if not pim_devs:
                return 0.0
            
            if dev.type == 'pim':
                return 0.0
            
            if dev.type == 'npu':
                attached_list: List[DeviceSpec] = []
                if hasattr(self, 'npu_attached_pimds'):
                    attached_list = self._npu_attached_pimds.get(dev.name, [])

                    if not attached_list:
                        for pd in pim_devs:
                            ptype = (getattr(pd, 'pim_type', None) or 'accel').lower()
                            if ptype in ('dram', 'hbm') and getattr(pd, 'attached_npu', None) == dev.name:
                                attached_list.append(pd)
                        
                        if attached_list and hasattr(self, '_npu_attached_pimds'):
                            self._npu_attached_pimds[dev.name].extend(attached_list)

                    if attached_list:
                        src_dev = attached_list[0]
                if src_dev is None:
                   src_dev = pim_devs[0]
                
                src_name = src_dev.name

        # ========= Case 2: KV in Host =========
        else:
            host_dev = self.cost.get_host_device()

            if dev.type == 'pim':
                # 在 PIM 上做KV 相关计算：先查 LRU 是否命中
                attrs = getattr(node, 'attrs', {}) or {}
                if 'kv_block_id' in attrs:
                    block_key = attrs['kv_block_id']
                else:
                    layer = attrs.get('layer', None)
                    step_idx = self.seq_len if phase == 'decode' else 0
                    block_key = (layer, step_idx, name)

                lru = self._kv_blocks_lru[dev.name]
                if block_key in lru:
                    return 0.0  # cache hit

                # miss：从 Host 读到 PIM
                src_dev = host_dev
                src_name = host_dev.name
            else:
                # NPU / CPU 做KV 相关计算：直接从 Host 读
                src_dev = host_dev
                src_name = host_dev.name

        if not src_name:
            return 0.0
        
        
        # 估算需要搬运的字节数（按 activation 大小）
        rd_bytes, _ = self.cost.estimate_activation_bytes(
            node,
            batch=int(getattr(self, 'batch', 1) or 1),
            seq_len=int(getattr(self, 'seq_len', 1) or 1),
            phase=phase,
        )
        if rd_bytes <= 0:
            return 0.0

        size_nd = self.cost.format_size(rd_bytes, 'ND')
        _, link_end = self.comm.reserve(src_name, dst_dev.name, size_nd, earliest=t0, commit=commit, tag='kv_load')
        
        # 目的设备上的格式转换时间
        conv_t = self.cost.format_conversion_time(size_nd, 'ND', self.cost.device_preferred_fmt(dst_dev), dst_dev,)
        finish = link_end + conv_t

        # Host 模式下，PIM 侧要更新 KV LRU（作为 KV cache）
        if commit and dev.type == 'pim' and kv_home == 'host':
            attrs = getattr(node, 'attrs', {}) or {}
            if 'kv_block_id' in attrs:
                block_key = attrs['kv_block_id']
            else:
                layer = attrs.get('layer', None)
                step_idx = self.seq_len if phase == 'decode' else 0
                block_key = (layer, step_idx, name)
            self._kv_block_lru_touch(dev, block_key, rd_bytes, touch_only=False)

        # if logger.isEnabledFor(logging.DEBUG):
        #     logger.debug(
        #         "[KV] home=%s phase=%s node=%s dst=%s src=%s bytes=%d",
        #         kv_home,
        #         phase,
        #         getattr(node, "name", ""),
        #         dst_dev.name,
        #         src_dev.name if src_dev else None,
        #         rd_bytes,
        #     )

        return max(0.0, finish - t0)

    def _earliest_finish_on_device(self, g: TaskGraph, nid: str, dev: DeviceSpec, label: PlanLabel, phase: str, commit: bool) -> Tuple[float, float]:
        node = g.nodes[nid]
        ready_time = self._ready_time_for_device(g, nid, dev, phase, commit)
        t0 = max(self.avail[dev.name], ready_time)
        compute_t = self.cost.node_device_cost(node, dev, label, self.batch, self.seq_len, phase)
        wload_t = self._weight_load_time(node, dev, t0, commit)
        kvload_t = self._kv_load_time(node, dev, t0, phase, commit)
        start_exec = t0 + max(wload_t, kvload_t)
        finish = start_exec + compute_t

        if commit:
            self._node_out_fmt[nid] = self.cost.device_preferred_fmt(dev)
            self._update_kv_cache_for_node(node, dev, phase, commit=True, finish=finish)
            if dev.type == 'pim':
                phy_bytes, kv_used, act_used, weight_used, total_used = self._pim_used_bytes(dev)
                extra = getattr(self, "_last_weight_cache_event", {}).get(dev.name)
                if extra:
                    try:
                        self._record_pim_trace(
                            dev,
                            phase=phase,
                            finish=float(finish),
                            event='WEIGHT_CACHE_UPDATE',
                            node=node,
                            extra=extra)
                    except Exception:
                        pass
                
                self._pim_trace.append({
                    "phase": phase,
                    "node_id": nid,
                    "op": getattr(node, "name", "") or "",
                    "device": dev.name,
                    "start": float(t0),
                    "finish": float(finish),
                    "seq_len": int(self.seq_len),
                    "kv_used_bytes": int(kv_used),
                    "act_used_bytes": int(act_used),
                    "weight_used_bytes": int(weight_used),
                    "total_used_bytes": int(total_used),
                    "phy_bytes": int(phy_bytes),
                })            
            out_read_nd, out_write_nd = self.cost.estimate_activation_bytes(node, self.batch, self.seq_len, phase)
            out_nd = max(out_write_nd, out_read_nd)
            cap = int(self._act_cap.get(dev.name, 0))
            used = int(self._act_used.get(dev.name, 0))
            if used + out_nd <= cap:
                self._act_used[dev.name] = used + out_nd
                self._act_resident[(dev.name, nid)] = out_nd
                # logger.debug(f"[LOCAL] retain out nid={nid}@{dev.name} bytes_nd={out_nd} used={self._act_used[dev.name]}/{cap}")
            else:
                # 空间不足：立即把该输出写回 Host（供后续消费者复用）
                # logger.debug(f"[LOCAL->HOST] evict out nid={nid}@{dev.name} need={out_nd} used={used}/{cap}")
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

    def _reserve_activation_transfer_best_path(
        self,
        prod_nid: str,
        src_dev: DeviceSpec,
        dst_dev: DeviceSpec,
        bytes_nd: int,
        src_fmt: str,
        pred_finish: float,
        commit: bool,
    ) -> Tuple[float, float]:

        """
        Reserve an activation transfer from src_dev -> dst_dev.

        For NPU<->PIM we support two routes:
          - src -> Host -> dst
          - src -> dst direct (if topology provides this link)

        During real scheduling, we check the current communication occupancy (per-link timelines)
        and pick the route that completes earlier.

        Returns:
          (start_time_on_last_hop, ready_time_on_dst_after_format_conversion)
        """

        dst_fmt = self.cost.device_preferred_fmt(dst_dev)
        size_nd = self.cost.format_size(bytes_nd, 'ND')
        host = self.cost.get_host_device()

        def _via_host(commit_flag:bool) -> Tuple[float, float]:
            host_ready = self._ensure_host_store(
                prod_nid, src_dev, bytes_nd, src_fmt, pred_finish, commit_flag)
            l2s, l2e = self.comm.reserve(host.name, dst_dev.name, size_nd, earliest=host_ready, commit=commit_flag,tag='act_move')
            conv2 = self.cost.format_conversion_time(size_nd, 'ND', dst_fmt, dst_dev)
            return (l2s, l2e + conv2)
        
        def _direct(commit_flag: bool) -> Tuple[float, float]:
            # Only consider direct transfer if link exists.
            try:
                t_direct = self.cost.link_time(size_nd, src_dev, dst_dev)
            except Exception:
                t_direct = 0.0
            if t_direct <= 0:
                return (float('inf'), float('inf'))

            # If source is PIM, require activation still resident on that PIM.
            if src_dev.type == 'pim' and (src_dev.name, prod_nid) not in self._act_resident:
                return (float('inf'), float('inf'))
            # Convert src_fmt -> ND on source if needed; then send ND bytes.
            size_src = self.cost.format_size(int(bytes_nd), src_fmt)
            t_conv_src = 0.0
            if src_fmt != 'ND':
                t_conv_src = self.cost.format_conversion_time(size_src, src_fmt, 'ND', src_dev)

            earliest = pred_finish + t_conv_src
            if src_dev.type == 'pim':
                earliest += self.cost.activation_read_time_pim(size_nd)
            l2s, l2e = self.comm.reserve(
                src_dev.name, dst_dev.name, size_nd,
                earliest=earliest, commit=commit_flag, tag='act_move'
            )
            conv2 = self.cost.format_conversion_time(size_nd, 'ND', dst_fmt, dst_dev)
            return l2s, l2e + conv2

        # Decide best route by completion time.
        s_direct, e_direct = _direct(False)
        s_host, e_host = _via_host(False)

        use_direct = e_direct < e_host
        if commit:
            return _direct(True) if use_direct else _via_host(True)
        else:
            return (s_direct, e_direct) if use_direct else (s_host, e_host)

    def _ready_time_for_device(self, g: TaskGraph, nid: str, dev: DeviceSpec, phase: str, commit: bool) -> float:
        '''
        calculate the earliest time on `dev` for node `nid`
        '''
        node = g.nodes[nid]
        inbound_start_times: List[float] = []
        inbound_end_times: List[float] = []
        batch = int(getattr(self, 'batch', 1) or 1)
        seq_len = int(getattr(self, 'seq_len', 128) or 128)
        node_read, _ = self.cost.estimate_activation_bytes(node, batch, seq_len, phase)

        for u in g.predecessors(nid): #对当前节点的所有前驱节点u进行评估
            pred_finish = self._node_finish_time.get(u, 0.0)
            pred_dev_name = self._node_placement.get(u, dev.name)
            
            if pred_dev_name == 'HYBRID': # Fallback for legacy 'HYBRID' marker: treat as host-written ND payload
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
            if pred_dev.name == dev.name: #前驱节点和当前节点在同一设备上
                src_fmt = self._node_out_fmt.get(u, self.cost.device_preferred_fmt(pred_dev))
                pred_node = g.nodes[u]
                _, pred_write = self.cost.estimate_activation_bytes(pred_node, batch, seq_len, phase)
                payload_nd = max(pred_write, node_read)
                size_nd = self.cost.format_size(payload_nd, 'ND')
                if dev.type == 'pim': #如果是pim，要检查pim的空间是否够存储act
                    avail = self._pim_avail_for_activation(pred_dev)
                    if avail >= payload_nd:
                        if commit:
                            self._act_used[pred_dev.name] = self._act_used.get(pred_dev.name, 0) + payload_nd
                        inbound_end_times.append(pred_finish)
                    else:
                        # logger.debug(f"[LOCAL->HOST] PIM fallback (no space) u={u}@{pred_dev.name} -> {nid}@{dev.name} need={payload_nd} avail={avail}")
                        host_ready = self._ensure_host_store(u, pred_dev, payload_nd, src_fmt, pred_finish, commit)
                        l2s, l2e = self.comm.reserve(self.cost.get_host_device().name, dev.name, size_nd, earliest=host_ready, commit=commit)
                        conv2 = self.cost.format_conversion_time(size_nd, 'ND', self.cost.device_preferred_fmt(dev), dev)
                        inbound_start_times.append(l2s)
                        inbound_end_times.append(l2e + conv2)
                else:
                    inbound_start_times.append(pred_finish)
                    inbound_end_times.append(pred_finish)
                continue
            else: #不同设备
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

                # 1) 检查两条传输链，选择最快的路径
                l2s, ready = self._reserve_activation_transfer_best_path(
                    prod_nid=u,
                    src_dev=pred_dev,
                    dst_dev=dev,
                    bytes_nd=payload_nd,
                    src_fmt=src_fmt,
                    pred_finish=pred_finish,
                    commit=commit,
                )
                inbound_start_times.append(l2s)
                inbound_end_times.append(ready)
                continue    

        return max(inbound_end_times, default=0.0)

    def _earliest_finish_hybrid(self, g, nid: str, phase: str, commit: bool):
        """
        计算把结点 nid 以 HYBRID（NPU+PIM 协同）方式执行时的最早完成时间，并在 commit=True 时真正预约链路。
        行为说明：
        - 依赖传输阶段：已允许 NPU<->PIM 直连或经 Host，调用 _ready_time_for_device 时会自动选更快路径。
        - HYBRID：分别把权重加载到 NPU 与 PIM，然后两边各自计算。
        - 结尾阶段：把该算子“整体输出”统一写回 Host（主存），
            其中 NPU 源侧做 NPU_OPT->ND 转换后再写；PIM 源侧用 trace 仿真近存读出 + PIM_OPT->ND 转换，再写 Host。
        - 返回一个 dict，包含 start/finish 及分设备的细节；调度器上层只需要 'start' 和 'finish'。
        依赖：
        """
        node = g.nodes[nid]
        batch = int(getattr(self, 'batch', 0) or 0)
        seq_len = int(getattr(self, 'seq_len', 0) or 0)

        npu_list = self.cluster.devices_by_type('npu')
        pim_list_all = self.cluster.devices_by_type('pim')
        if not npu_list or not pim_list_all:
            return None
        npu_dev = npu_list[0]

        # HYBRID 优先使用 PIM-Accelerator；若没有，再退化用任意 PIM
        pim_dev = None
        for d in pim_list_all:
            ptype = (getattr(d, 'pim_type', None) or 'accel').lower()
            if ptype not in ('dram', 'hbm'):
                pim_dev = d
                break
        if pim_dev is None:
            pim_dev = pim_list_all[0]

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

        # 2.2 PIM：PIM 侧从自身内存加载权重（统一走 cost.weight_load_time_pim，内部自带带宽回退）
        t_w_pim_end = t_ready_pim
        if weight_pim_nd > 0:
            t_load_pim = self.cost.weight_load_time_pim(weight_pim_nd)
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


        # logger.debug(f"[HYBRID] nid={nid} alpha={alpha} "
        #              f"start_npu={start_npu:.6f} finish_npu={finish_npu:.6f} "
        #              f"start_pim={start_pim:.6f} finish_pim={finish_pim:.6f} finish={finish:.6f}")
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
                                        earliest=earliest, commit=commit, tag='act_move')
        t_done = t_link_end
        if commit:
            self._node_host_store_end[u] = t_done
        return t_done

    def _pim_avail_for_activation(self, dev: DeviceSpec) -> int:
        total = int(dev.mem_capacity_GB * 1024**3)

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
        if getattr(self, 'stats', None):
            self.stats.set_phase(phase)
        order = self._upward_rank(g, phase=phase)
        schedule: List[ScheduledTask] = []
        for nid in order:
            node = g.nodes[nid]
            allow_npu = node.allowed.get('npu', True)
            allow_pim = (
                node.allowed.get('pim', True)
                or node.allowed.get('pima', False)
                or node.allowed.get('pimd', False)
            )
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
                    # Check if node is allowed on this specific PIM device (pima/pimd) according to config
                    if not self.cost._op_allowed_on(node, dev):
                        continue
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
            
            # if logger.isEnabledFor(logging.DEBUG):
            #     cand_str = ", ".join([f"{m}={f:.6f}" for m, f, _ in candidates])
            #     logger.debug(f"[HEFT-DECISION] Node {nid} Candidates: {cand_str}")

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
                if getattr(self, 'stats', None):
                    op_name = node.attrs.get('op') or node.name
                    try:
                        self.stats.log_op_device(
                            nid=nid, op=op_name,
                            device=npu.name, device_type=npu.type,
                            start=float(hy['start_npu']),
                            end=float(hy['npu_detail']['finish_compute']),
                            mode='HYBRID'
                        )
                        self.stats.log_op_device(
                            nid=nid, op=op_name,
                            device=pim.name, device_type=pim.type,
                            start=float(hy['start_pim']),
                            end=float(hy['pim_detail']['finish_compute']),
                            mode='HYBRID'
                        )
                    except Exception:
                        pass
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
                if getattr(self, 'stats', None):
                    op_name = node.attrs.get('op') or node.name
                    try:
                        self.stats.log_op_device(
                            nid=nid, op=op_name,
                            device=dev.name, device_type=dev.type,
                            start=float(start), end=float(finish),
                            mode=chosen_mode
                        )
                    except Exception:
                        pass
            # logger.debug(str(f'[Schedule] Node {nid} on {self._node_placement[nid]} from {start:.3f} to {finish:.3f} ({chosen_mode})'))
        return schedule


    # =========================
    # Joint (Prefill + Decode) HEFT with OCT lookahead
    # =========================

    def _joint_allowed_on_device(self, node: TaskNode, dev: DeviceSpec) -> bool:
        """Check whether a node is allowed to run on a given device."""
        # Default permissive if not specified.
        allowed = True
        if hasattr(node, "allowed") and isinstance(node.allowed, Mapping):
            if dev.type == "pim":
                # Some graphs may annotate PIM subtypes (pima/pimd). Treat them as PIM-enabled.
                allowed = bool(
                    node.allowed.get("pim", True)
                    or node.allowed.get("pima", False)
                    or node.allowed.get("pimd", False)
                )
            else:
                allowed = bool(node.allowed.get(dev.type, True))
        return allowed

    def _joint_candidate_devices(self, node: TaskNode) -> List[DeviceSpec]:
        """Return all physical devices that can run this node (NPU/PIM)."""
        devs: List[DeviceSpec] = []
        allow_npu = True
        allow_pim = True
        if hasattr(node, "allowed") and isinstance(node.allowed, Mapping):
            allow_npu = bool(node.allowed.get("npu", True))
            allow_pim = bool(
                node.allowed.get("pim", True)
                or node.allowed.get("pima", False)
                or node.allowed.get("pimd", False)
            )
        if allow_npu:
            devs.extend(self.cluster.devices_by_type("npu"))
        if allow_pim:
            devs.extend(self.cluster.devices_by_type("pim"))
        # Filter with per-device allow if needed.
        devs = [d for d in devs if self._joint_allowed_on_device(node, d)]
        return devs

    def _joint_static_comm_time(
        self,
        bytes_nd: int,
        src_dev: DeviceSpec,
        dst_dev: DeviceSpec,
        src_fmt: str,
    ) -> float:
        """Static (no-occupancy) estimate of src->dst activation transfer time for ND payload."""
        if bytes_nd <= 0 or src_dev.name == dst_dev.name:
            return 0.0

        dst_fmt = self.cost.device_preferred_fmt(dst_dev)
        size_nd = self.cost.format_size(int(bytes_nd), "ND")

        # Convert src_fmt -> ND on src if needed (approx).
        size_src = self.cost.format_size(int(bytes_nd), src_fmt)
        t_conv_src = 0.0
        if src_fmt != "ND":
            t_conv_src = self.cost.format_conversion_time(size_src, src_fmt, "ND", src_dev)

        # Optional PIM activation read overhead.
        t_read = 0.0
        if src_dev.type == "pim":
            try:
                t_read = float(self.cost.activation_read_time_pim(size_nd))
            except Exception:
                t_read = 0.0

        # Convert ND -> dst_fmt on dst.
        t_conv_dst = self.cost.format_conversion_time(size_nd, "ND", dst_fmt, dst_dev)

        # Two possible routes: direct or via host. We ignore per-link occupancy here.
        host = self.cost.get_host_device()

        # Direct
        try:
            t_link_direct = float(self.cost.link_time(size_nd, src_dev, dst_dev))
        except Exception:
            t_link_direct = float("inf")
        direct = t_conv_src + t_read + t_link_direct + t_conv_dst

        # Via host
        try:
            t_link_1 = float(self.cost.link_time(size_nd, src_dev, host))
            t_link_2 = float(self.cost.link_time(size_nd, host, dst_dev))
        except Exception:
            t_link_1 = t_link_2 = float("inf")
        via_host = t_conv_src + t_read + t_link_1 + t_link_2 + t_conv_dst

        return min(direct, via_host)

    def _build_joint_graph(
        self,
        g_prefill: TaskGraph,
        g_decode: TaskGraph,
        prefill_len: int,
        decode_steps: int,
        decode_seq_mode: str = "context",
    ) -> JointTaskGraph:
        """
        Build a joint graph that schedules:
          - one prefill pass
          - `decode_steps` autoregressive decode steps, connected in series

        Node ids are prefixed as:
          - P::<base_id>
          - D{step}::<base_id>    (step starts from 1)

        decode_seq_mode:
          - "context": seq_len for decode step r is (prefill_len + r - 1)
          - "one":     seq_len for decode step r is 1
        """

        def _find_source_sink(g: TaskGraph) -> Tuple[str, str]:
            nodes_iter = g.nodes.keys() if hasattr(g.nodes, "keys") else g.nodes
            nodes = list(nodes_iter)
            preds = {nid: tuple(g.predecessors(nid)) for nid in nodes}
            succs = {nid: tuple(g.successors(nid)) for nid in nodes}
            srcs = [nid for nid in nodes if len(preds[nid]) == 0]
            snks = [nid for nid in nodes if len(succs[nid]) == 0]
            if not srcs or not snks:
                raise ValueError("Graph must have at least one source and one sink.")
            # Assume unique source/sink in this project.
            return srcs[0], snks[0]

        srcP, snkP = _find_source_sink(g_prefill)
        srcD, snkD = _find_source_sink(g_decode)

        nodes: Dict[str, TaskNode] = {}
        preds: Dict[str, List[str]] = defaultdict(list)
        succs: Dict[str, List[str]] = defaultdict(list)
        meta: Dict[str, JointNodeMeta] = {}

        def add_edge(u: str, v: str):
            succs[u].append(v)
            preds[v].append(u)

        # ---- Prefill copy ----
        for base_id, node in g_prefill.nodes.items():
            jid = f"P::{base_id}"
            nodes[jid] = node
            meta[jid] = JointNodeMeta(base_id=base_id, phase="prefill", step=0, seq_len=int(prefill_len))

        for u in g_prefill.nodes.keys():
            for v in g_prefill.successors(u):
                add_edge(f"P::{u}", f"P::{v}")

        # ---- Decode steps ----
        decode_steps = max(0, int(decode_steps))
        for step in range(1, decode_steps + 1):
            if decode_seq_mode == "one":
                seq_len = 1
            else:
                seq_len = int(prefill_len + step - 1)

            for base_id, node in g_decode.nodes.items():
                jid = f"D{step}::{base_id}"
                nodes[jid] = node
                meta[jid] = JointNodeMeta(base_id=base_id, phase="decode", step=step, seq_len=seq_len)

            for u in g_decode.nodes.keys():
                for v in g_decode.successors(u):
                    add_edge(f"D{step}::{u}", f"D{step}::{v}")

            # barrier edges between phases/steps
            if step == 1:
                add_edge(f"P::{snkP}", f"D{step}::{srcD}")
            else:
                add_edge(f"D{step-1}::{snkD}", f"D{step}::{srcD}")

        # finalize empty preds/succs
        preds_t: Dict[str, Tuple[str, ...]] = {}
        succs_t: Dict[str, Tuple[str, ...]] = {}
        for nid in nodes.keys():
            preds_t[nid] = tuple(preds.get(nid, []))
            succs_t[nid] = tuple(succs.get(nid, []))

        # Kahn topo sort
        indeg = {nid: len(preds_t[nid]) for nid in nodes.keys()}
        q = deque([nid for nid, d in indeg.items() if d == 0])
        topo: List[str] = []
        while q:
            n = q.popleft()
            topo.append(n)
            for s in succs_t.get(n, ()):
                indeg[s] -= 1
                if indeg[s] == 0:
                    q.append(s)
        if len(topo) != len(nodes):
            raise ValueError("Joint graph is cyclic or disconnected in an unexpected way.")

        return JointTaskGraph(nodes=nodes, preds=preds_t, succs=succs_t, topo=tuple(topo), meta=meta)

    def _compute_joint_rank_u(self, g: JointTaskGraph) -> Tuple[Dict[str, float], List[str]]:
        """
        Compute HEFT upward rank on the joint graph using average costs.
        We intentionally make the cache key include (phase, seq_len) to avoid
        mixing different decode steps.
        """
        # Memoize compute + avg comm
        p_avg_cache: Dict[str, float] = {}
        q_avg_cache: Dict[Tuple[str, str], float] = {}

        batch = int(getattr(self, "batch", 1) or 1)

        def p_avg(nid: str) -> float:
            if nid in p_avg_cache:
                return p_avg_cache[nid]
            meta = g.meta[nid]
            node = g.nodes[nid]
            devs = self._joint_candidate_devices(node)
            if not devs:
                p_avg_cache[nid] = 0.0
                return 0.0
            total = 0.0
            cnt = 0
            for dev in devs:
                try:
                    t = float(self.cost.node_device_cost(node, dev, self.label, batch, meta.seq_len, meta.phase))
                except Exception:
                    t = 0.0
                total += t
                cnt += 1
            out = total / max(1, cnt)
            p_avg_cache[nid] = out
            return out

        def q_avg(u: str, v: str) -> float:
            key = (u, v)
            if key in q_avg_cache:
                return q_avg_cache[key]
            mu = g.meta[u]
            mv = g.meta[v]
            # Cross-phase/step barrier edges: treat as zero comm.
            if (mu.phase != mv.phase) or (mu.step != mv.step):
                q_avg_cache[key] = 0.0
                return 0.0
            phase = mu.phase
            seq_len = mu.seq_len
            u_node = g.nodes[u]
            v_node = g.nodes[v]
            try:
                u_read, u_write = self.cost.estimate_activation_bytes(u_node, batch, seq_len, phase)
                v_read, _ = self.cost.estimate_activation_bytes(v_node, batch, seq_len, phase)
                payload = int(max(u_write, v_read, 16 * 1024))
            except Exception:
                payload = 16 * 1024

            u_devs = self._joint_candidate_devices(u_node)
            v_devs = self._joint_candidate_devices(v_node)
            if not u_devs or not v_devs:
                q_avg_cache[key] = 0.0
                return 0.0

            total = 0.0
            cnt = 0
            for du in u_devs:
                src_fmt = self.cost.device_preferred_fmt(du)
                for dv in v_devs:
                    total += float(self._joint_static_comm_time(payload, du, dv, src_fmt))
                    cnt += 1
            out = total / max(1, cnt)
            q_avg_cache[key] = out
            return out

        rank_u: Dict[str, float] = {}
        for nid in reversed(g.topological()):
            succs = g.successors(nid)
            if not succs:
                rank_u[nid] = p_avg(nid)
            else:
                best = 0.0
                for s in succs:
                    path = q_avg(nid, s) + rank_u.get(s, 0.0)
                    if path > best:
                        best = path
                rank_u[nid] = p_avg(nid) + best

        order = sorted(g.nodes.keys(), key=lambda x: -rank_u.get(x, 0.0))
        return rank_u, order

    def _compute_joint_oct(self, g: JointTaskGraph) -> Dict[Tuple[str, str], float]:
        """
        Compute PEFT-style OCT(n,dev) table on the joint graph:
          OCT(n,k) = max_{s in succ(n)} min_{l} [ q(n,s,k,l) + p(s,l) + OCT(s,l) ]
        Communication is estimated statically (no occupancy).
        """
        batch = int(getattr(self, "batch", 1) or 1)

        # Cache p(n,dev) and q(u,v,dev_u,dev_v)
        p_cache: Dict[Tuple[str, str], float] = {}
        payload_cache: Dict[Tuple[str, str], int] = {}
        q_cache: Dict[Tuple[str, str, str, str], float] = {}

        def p_time(nid: str, dev: DeviceSpec) -> float:
            key = (nid, dev.name)
            if key in p_cache:
                return p_cache[key]
            meta = g.meta[nid]
            node = g.nodes[nid]
            try:
                t = float(self.cost.node_device_cost(node, dev, self.label, batch, meta.seq_len, meta.phase))
            except Exception:
                t = 0.0
            p_cache[key] = t
            return t

        def payload_bytes(u: str, v: str) -> int:
            key = (u, v)
            if key in payload_cache:
                return payload_cache[key]
            mu = g.meta[u]
            mv = g.meta[v]
            if (mu.phase != mv.phase) or (mu.step != mv.step):
                payload_cache[key] = 0
                return 0
            phase = mu.phase
            seq_len = mu.seq_len
            u_node = g.nodes[u]
            v_node = g.nodes[v]
            try:
                _, u_write = self.cost.estimate_activation_bytes(u_node, batch, seq_len, phase)
                v_read, _ = self.cost.estimate_activation_bytes(v_node, batch, seq_len, phase)
                payload = int(max(u_write, v_read, 16 * 1024))
            except Exception:
                payload = 16 * 1024
            payload_cache[key] = payload
            return payload

        def q_time(u: str, v: str, du: DeviceSpec, dv: DeviceSpec) -> float:
            key = (u, v, du.name, dv.name)
            if key in q_cache:
                return q_cache[key]
            mu = g.meta[u]
            mv = g.meta[v]
            if (mu.phase != mv.phase) or (mu.step != mv.step):
                q_cache[key] = 0.0
                return 0.0
            payload = payload_bytes(u, v)
            src_fmt = self.cost.device_preferred_fmt(du)
            t = float(self._joint_static_comm_time(payload, du, dv, src_fmt))
            q_cache[key] = t
            return t

        # Precompute candidate devices per node (allowed set)
        cand_devs: Dict[str, List[DeviceSpec]] = {}
        for nid, node in g.nodes.items():
            cand_devs[nid] = self._joint_candidate_devices(node)

        oct_tbl: Dict[Tuple[str, str], float] = {}

        # Reverse topo DP
        for nid in reversed(g.topological()):
            succs = g.successors(nid)
            for du in cand_devs.get(nid, []):
                if not succs:
                    oct_tbl[(nid, du.name)] = 0.0
                    continue
                worst = 0.0
                for s in succs:
                    # For each successor, choose best dv
                    best = float("inf")
                    for dv in cand_devs.get(s, []):
                        t = q_time(nid, s, du, dv) + p_time(s, dv) + oct_tbl.get((s, dv.name), 0.0)
                        if t < best:
                            best = t
                    worst = max(worst, best if best < float("inf") else 0.0)
                oct_tbl[(nid, du.name)] = worst

        return oct_tbl

    def schedule_joint(
        self,
        g_prefill: TaskGraph,
        prefill_len: int,
        decode_steps: int,
        g_decode: TaskGraph | None = None,
        decode_seq_mode: str = "context",
    ) -> List[ScheduledTask]:
        """
        Jointly schedule one prefill pass and `decode_steps` decode passes *in one DAG*,
        and select devices using HEFT+OCT score:

            Score(n, dev) = EFT(n, dev) + OCT(n, dev)

        This makes prefill placement aware of long-horizon decode costs, instead of
        "schedule prefill first, then per-token decode" greedy behavior.
        """
        if g_decode is None:
            g_decode = g_prefill

        # Build joint graph
        joint_g = self._build_joint_graph(
            g_prefill=g_prefill,
            g_decode=g_decode,
            prefill_len=int(prefill_len),
            decode_steps=int(decode_steps),
            decode_seq_mode=str(decode_seq_mode),
        )

        # Compute rank_u order and OCT table
        _, order = self._compute_joint_rank_u(joint_g)
        oct_tbl = self._compute_joint_oct(joint_g)

        sched: List[ScheduledTask] = []

        # Preserve caller seq_len, but override per joint node
        seq_len_saved = int(getattr(self, "seq_len", 0) or 0)

        for nid in order:
            meta = joint_g.meta[nid]
            phase = meta.phase
            self.seq_len = int(meta.seq_len)

            if getattr(self, "stats", None):
                try:
                    self.stats.set_phase(phase)
                except Exception:
                    pass

            node = joint_g.nodes[nid]
            allow_npu = True
            allow_pim = True
            if hasattr(node, "allowed") and isinstance(node.allowed, Mapping):
                allow_npu = bool(node.allowed.get("npu", True))
                allow_pim = bool(
                    node.allowed.get("pim", True)
                    or node.allowed.get("pima", False)
                    or node.allowed.get("pimd", False)
                )

            candidates: List[Tuple[str, float, float, DeviceSpec | None]] = []
            # (mode, score, finish, dev)

            if allow_npu:
                for dev in self.cluster.devices_by_type("npu"):
                    if not self._joint_allowed_on_device(node, dev):
                        continue
                    _, finish = self._earliest_finish_on_device(joint_g, nid, dev, self.label, phase, commit=False)
                    score = float(finish) + float(oct_tbl.get((nid, dev.name), 0.0))
                    candidates.append(("NPU", score, float(finish), dev))

            if allow_pim:
                for dev in self.cluster.devices_by_type("pim"):
                    if not self._joint_allowed_on_device(node, dev):
                        continue
                    _, finish = self._earliest_finish_on_device(joint_g, nid, dev, self.label, phase, commit=False)
                    score = float(finish) + float(oct_tbl.get((nid, dev.name), 0.0))
                    candidates.append(("PIM", score, float(finish), dev))

            if ALLOW_HYBRID and allow_npu and allow_pim:
                best_result = self._earliest_finish_hybrid(joint_g, nid, phase, commit=False)
                if best_result is not None:
                    out_dev = best_result.get("out_dev", None)
                    if out_dev is not None:
                        score = float(best_result["finish"]) + float(oct_tbl.get((nid, out_dev.name), 0.0))
                    else:
                        score = float(best_result["finish"])
                    candidates.append(("HYBRID", score, float(best_result["finish"]), None))

            if not candidates:
                raise RuntimeError(f"No feasible device candidates for node={nid} phase={phase}")

            chosen_mode, _, _, chosen_dev = min(candidates, key=lambda x: x[1])

            if chosen_mode == "HYBRID":
                hy = self._earliest_finish_hybrid(joint_g, nid, phase, commit=True)
                npu = hy["npu"]
                pim = hy["pim"]
                start = min(hy["start_npu"], hy["start_pim"])
                finish = hy["finish"]
                self.avail[npu.name] = finish
                self.avail[pim.name] = finish
                self._node_finish_time[nid] = finish
                self._node_placement[nid] = hy["out_dev"].name
                self._node_out_fmt[nid] = "ND"
                sched.append(ScheduledTask(nid, f"HYBRID({npu.name}+{pim.name})", start, finish))
                op_name = node.attrs.get("op") or node.name
                self.mode_mem[op_name] = "HYBRID"
                self._after_commit_consume_predecessors(joint_g, nid)
            else:
                assert chosen_dev is not None
                dev = chosen_dev
                start, finish = self._earliest_finish_on_device(joint_g, nid, dev, self.label, phase, commit=True)
                self.avail[dev.name] = finish
                self._node_finish_time[nid] = finish
                self._node_placement[nid] = dev.name
                self._node_out_fmt[nid] = self.cost.device_preferred_fmt(dev)
                sched.append(ScheduledTask(nid, dev.name, start, finish))
                op_name = node.attrs.get("op") or node.name
                self.mode_mem[op_name] = dev.type
                self._after_commit_consume_predecessors(joint_g, nid)

        # Restore caller seq_len
        self.seq_len = seq_len_saved
        return sched
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
                # logger.debug(f"[ACT] release u={u}@{udev} freed={bytes_kept} used={self._act_used[udev]}/{self._act_cap.get(udev,0)}")

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
        self._kv_blocks_lru.clear()
        self._kv_used_bytes.clear()
        self._pim_trace.clear()

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
    
    def _topo_order(self, g: TaskGraph, phase: str) -> List[str]: #parallel
        return self._upward_rank(g, phase=phase)


    def _pim_kind(self, dev: DeviceSpec) -> str:
        ptype = (getattr(dev, 'pim_type', None) or 'accel').lower()
        return 'pimd' if ptype in ('dram', 'hbm') else 'pima'

    def _devices_for_mode(self, mode: str) -> List[DeviceSpec]:
        if mode in ('cpu', 'npu'):
            return self.cluster.devices_by_type(mode)
        if mode == 'pim':
            return self.cluster.devices_by_type('pim')
        if mode in ('pima', 'pimd'):
            return [d for d in self.cluster.devices_by_type('pim') if self._pim_kind(d) == mode]
        return []

    def _mode_matches_device(self, mode: str, dev: DeviceSpec | None) -> bool:
        if dev is None:
            return False
        if mode in ('cpu', 'npu'):
            return dev.type == mode
        if mode == 'pim':
            return dev.type == 'pim'
        if mode in ('pima', 'pimd'):
            return dev.type == 'pim' and self._pim_kind(dev) == mode
        return False

    def _has_pim_mode(self, acts: Iterable[str]) -> bool:
        return any(m in ('pim', 'pima', 'pimd') for m in acts)


    def _allowed_actions(self, node: TaskNode) -> List[str]:
        acts: List[str] = []
        seen: set[str] = set()

        def add_mode(mode: str) -> None:
            if mode in seen:
                return
            devs = self._devices_for_mode(mode)
            if not devs:
                return
            if not any(self.cost._op_allowed_on(node, d) for d in devs):
                return
            acts.append(mode)
            seen.add(mode)

        if node.allowed.get('npu', False):
            add_mode('npu')

        for mode in ('pim', 'pima', 'pimd'):
            if node.allowed.get(mode, False):
                add_mode(mode)

        if ALLOW_HYBRID and ('npu' in seen) and self._has_pim_mode(seen):
            acts.append('hybrid')
            seen.add('hybrid')

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
            try:
                self._heft_seed_device = {t.node_id: str(t.device) for t in base_sched}
                self._heft_seed_actions = dict(actions)
            except Exception:
                pass
        return actions
    

    def _ensure_heft_seed(self, g: TaskGraph, phase: str) -> None:
        """
        Lazily compute and cache HEFT seed device/actions if not present.
        """
        if getattr(self, "_heft_seed_device", None) is not None and getattr(self, "_heft_seed_actions", None) is not None:
            return
        snap = self._snapshot() #记录当前状态
        try:
            self.reset_state()
            base_sched = super().schedule(g, phase=phase) #调用最基本的heftscheduler
            self._heft_seed_device = {t.node_id: str(t.device) for t in base_sched} #得到heft算法中每个节点的设备
            actions: Dict[str, str] = {}
            for t in base_sched:
                devs = str(t.device)
                actions[t.node_id] = 'hybrid' if 'HYBRID' in devs else _dev_type(devs)
            self._heft_seed_actions = actions
        finally:
            self._restore(snap) #不管try是否执行成功，都会恢复之前的状态

    def _restore(self, snap):
        """Restore state from a snapshot produced by _snapshot."""
        # 一律用 .copy()，既避免 alias，又保持原来的 dict/defaultdict 类型
        self.avail = snap['avail'].copy()
        self.comm.timeline_end = snap['comm'].copy()
        self._node_finish_time = snap['node_finish'].copy()
        self._node_placement = snap['node_place'].copy()
        self._node_out_fmt = snap['node_fmt'].copy()
        self.weight_cached = snap['weight_cached'].copy()
        self._weight_load_count = snap['weight_load_count'].copy()
        self._weight_sizes = snap['weight_sizes'].copy()
        self.mode_mem = snap['mode_mem'].copy()
        self._act_used = snap['act_used'].copy()
        self._act_resident = snap['act_resident'].copy()
        self._act_refcnt = snap['act_refcnt'].copy()
        self._node_host_store_end = snap['host_store_end'].copy()

        # buffer_device_cache 本身在 snapshot 阶段 deep copy 过一次，
        # 这里直接复用就行；如果你担心 alias，也可以再 .copy() / deepcopy 一层。
        self.buffer.device_cache = snap['buffer_device_cache']
        self.buffer.host_format = snap['buffer_host_fmt'].copy()
        self._kv_blocks_lru = copy.deepcopy(snap['kv_blocks_lru'])
        self._kv_used_bytes = snap['kv_used_bytes'].copy()
    
    def _snapshot(self):
        """Lightweight snapshot of mutable scheduling state for search algorithms.

        """
        return {
            # 普通 dict / defaultdict 都有 copy()，能保持原类型（包括 default_factory）
            'avail': self.avail.copy(),
            'comm': self.comm.timeline_end.copy(),
            'node_finish': self._node_finish_time.copy(),
            'node_place': self._node_placement.copy(),
            'node_fmt': self._node_out_fmt.copy(),
            'weight_cached': self.weight_cached.copy(),
            'weight_load_count': self._weight_load_count.copy(),   # 保留 defaultdict(int)
            'weight_sizes': self._weight_sizes.copy(),
            'mode_mem': self.mode_mem.copy(),
            'act_used': self._act_used.copy(),
            'act_resident': self._act_resident.copy(),
            'act_refcnt': self._act_refcnt.copy(),
            'host_store_end': self._node_host_store_end.copy(),
            'buffer_device_cache': copy.deepcopy(self.buffer.device_cache),
            'buffer_host_fmt': self.buffer.host_format.copy(),
            'kv_blocks_lru': copy.deepcopy(self._kv_blocks_lru),
            'kv_used_bytes': self._kv_used_bytes.copy(),
        }


    def _step_node(self, g: TaskGraph, nid: str, action: str, phase: str) -> None:
        """
        Execute a single node with a specific action and update the scheduler state.
        Used for incremental scheduling in Lookahead/MCTS.
        """
        node = g.nodes[nid]
        mode = action
        
        if mode == 'hybrid':
            hy = self._earliest_finish_hybrid(g, nid, phase, commit=True)
            start = min(hy['start_npu'], hy['start_pim'])
            finish = hy['finish']
            self.avail[hy['npu'].name] = finish #更新设备可用时间（在此之前被占用）
            self.avail[hy['pim'].name] = finish
            self._node_finish_time[nid] = finish
            self._node_placement[nid] = hy['out_dev'].name
            self._node_out_fmt[nid] = 'ND'
            
            op_name = node.attrs.get('op') or node.name
            self.mode_mem[op_name] = 'HYBRID'
            self._after_commit_consume_predecessors(g, nid) #检查前驱节点，如果前驱节点数据不再被需要，则释放其占用的内存
            if getattr(self, "_log_decision_trace", False):
                logger.debug(
                    "[HEFT-LA] commit node=%s action=hybrid device=HYBRID(%s+%s) start=%.6f finish=%.6f",
                    nid,
                    hy['npu'].name,
                    hy['pim'].name,
                    start,
                    finish,
                )
            
            if getattr(self, 'stats', None):
                try:
                    self.stats.log_op_device(
                        nid=nid, op=op_name,
                        device=hy['npu'].name, device_type=hy['npu'].type,
                        start=float(hy['start_npu']),
                        end=float(hy['npu_detail']['finish_compute']),
                        mode='HYBRID'
                    )
                    self.stats.log_op_device(
                        nid=nid, op=op_name,
                        device=hy['pim'].name, device_type=hy['pim'].type,
                        start=float(hy['start_pim']),
                        end=float(hy['pim_detail']['finish_compute']),
                        mode='HYBRID'
                    )
                except Exception:
                    pass
        else:
            devs = self._devices_for_mode(mode)
            if not devs:
                return

            best = None
            for d in devs:
                st, ft = self._earliest_finish_on_device(g, nid, d, self.label, phase, commit=False)
                if (best is None) or (ft < best[0]):
                    best = (ft, st, d)
            
            try:
                seed_name = getattr(self, '_heft_seed_device', {}).get(nid)
                if seed_name:
                    seed_dev = self.cluster.devices.get(seed_name)
                    if seed_dev and self._mode_matches_device(mode, seed_dev):
                        for dd in devs:
                            if dd.name == seed_name:
                                st0, ft0 = self._earliest_finish_on_device(g, nid, dd, self.label, phase, commit=False)
                                if best and (ft0 <= best[0] * 1.01):
                                    best = (ft0, st0, dd)
                                break
            except Exception:
                pass

            if best is None:
                return

            f, s, dev = best
            
            start, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=True)
            self.avail[dev.name] = finish
            self._node_finish_time[nid] = finish
            self._node_placement[nid] = dev.name
            self._node_out_fmt[nid] = self.cost.device_preferred_fmt(dev)
            
            op_name = node.attrs.get('op') or node.name
            self.mode_mem[op_name] = dev.type
            self._after_commit_consume_predecessors(g, nid)
            if getattr(self, "_log_decision_trace", False):
                logger.debug(
                    "[HEFT-LA] commit node=%s action=%s device=%s start=%.6f finish=%.6f",
                    nid,
                    mode,
                    dev.name,
                    start,
                    finish,
                )
            
            if getattr(self, 'stats', None):
                try:
                    self.stats.log_op_device(
                        nid=nid, op=op_name,
                        device=dev.name, device_type=dev.type,
                        start=float(start), end=float(finish),
                        mode=mode
                    )
                except Exception:
                    pass

    def _evaluate_action_map(self, g, phase, actions: Mapping | None, dry_run=True, order=None):
        snap = self._snapshot() if dry_run else None
        scheduled: List[ScheduledTask] = []

        idx = self._get_graph_index(g)
        if order is None:
            cached_order = idx.order_by_phase.get(phase)
            order = cached_order if cached_order is not None else tuple(self._topo_order(g, phase))
        else:
            # caller may pass list; keep it as-is for stable iteration
            order = tuple(order)

        preds = idx.preds
        succs = idx.succs

        actions_get = actions.get if actions is not None else None
        done_nodes = set(self._node_finish_time.keys())
        done_nodes -= idx.nodes_set
        placed = set(done_nodes)
        # int-count is much faster than set() diff + discard()
        pred_left = {}
        for nid in order:
            if nid in placed:
                continue
            cnt = 0
            for p in preds[nid]:
                if p not in placed:
                    cnt += 1
            pred_left[nid] = cnt

        ready = [nid for nid in order if nid not in placed and pred_left.get(nid, 0) == 0]
        in_ready = set(ready)


        while ready:
            best_tuple = None  # (finish, start, nid, mode, dev)
            best_i = -1
            for i, nid in enumerate(ready):
                node = g.nodes[nid]
                act = actions_get(nid) if actions_get is not None else None
                if act is None or act == 'auto':
                    # default to greedy (like HEFT)
                    cands = []
                    node_modes = self._allowed_actions(node)
                    for mode in node_modes:
                        if mode in ('hybrid', 'cpu'):
                            continue
                        devs = self._devices_for_mode(mode)
                        for dev in devs:
                            if not self.cost._op_allowed_on(node, dev):
                                continue
                            start, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=False)
                            cands.append((finish, start, mode, dev))
                    if 'hybrid' in node_modes:
                        hy = self._earliest_finish_hybrid(g, nid, phase, commit=False)
                        if hy is not None:
                            cands.append((hy['finish'], min(hy['start_npu'], hy['start_pim']), 'hybrid', None))
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
                        devs = self._devices_for_mode(mode)
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
                        # Prefer HEFT's seed device if comparable (<=1% slower)
                        try:
                            seed_name = getattr(self, '_heft_seed_device', {}).get(nid)
                            if seed_name:
                                seed_dev = self.cluster.devices.get(seed_name)
                                if seed_dev and self._mode_matches_device(mode, seed_dev):
                                    for dd in devs:
                                        if dd.name == seed_name:
                                            st0, ft0 = self._earliest_finish_on_device(g, nid, dd, self.label, phase, commit=False)
                                            if (ft0 <= f * 1.01):
                                                f, s, dev = ft0, st0, dd
                                            break
                        except Exception:
                            pass
                # choose best across ready set
                cand = (f, s, nid, mode, dev)
                if (best_tuple is None) or (cand[0] < best_tuple[0]):
                    best_tuple = cand
                    best_i = i

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
                if not dry_run and getattr(self, 'stats', None):
                    try:
                        self.stats.log_op_device(
                            nid=nid, op=op_name,
                            device=hy['npu'].name, device_type=hy['npu'].type,
                            start=float(hy['start_npu']),
                            end=float(hy['npu_detail']['finish_compute']),
                            mode='HYBRID'
                        )
                        self.stats.log_op_device(
                            nid=nid, op=op_name,
                            device=hy['pim'].name, device_type=hy['pim'].type,
                            start=float(hy['start_pim']),
                            end=float(hy['pim_detail']['finish_compute']),
                            mode='HYBRID'
                        )
                    except Exception:
                        pass
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
                if not dry_run and getattr(self, 'stats', None):
                    try:
                        self.stats.log_op_device(
                            nid=nid, op=op_name,
                            device=dev.name, device_type=dev.type,
                            start=float(start), end=float(finish),
                            mode=mode
                        )
                    except Exception:
                        pass

            placed.add(nid)
            in_ready.remove(nid)
            last = ready.pop()
            if best_i != len(ready):
                ready[best_i] = last
            for v in succs[nid]:
                if v in placed:
                    continue
                c = pred_left.get(v)
                if c is None:
                    continue
                c -= 1
                pred_left[v] = c
                if c == 0 and v not in in_ready:
                    ready.append(v)
                    in_ready.add(v)

        cost = self.makespan(scheduled)
        # If we did incremental scheduling, the true makespan is max(new_tasks, old_tasks)
        if self._node_finish_time:
            cost = max(cost, max(self._node_finish_time.values()))
        if snap is not None:
            self._restore(snap)
        return cost, scheduled


class SimulatedAnnealingScheduler(_SearchSchedulerMixin):
    """
    SA over device-type assignment per node; schedule built with list scheduling.
    """
    def __init__(self, *args, sa_iters: int = 200, T0: float = -1.0, alpha: float = 0.95, flip_prob: float = 0.05, sa_k: int = 2, critical_frac: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.sa_iters = int(sa_iters)
        self.T0 = float(T0)
        self.alpha = float(alpha)
        self.flip_prob = float(flip_prob)
        self.sa_k = int(sa_k)
        self.critical_frac = float(critical_frac)

    def _neighbors(self, g: TaskGraph, phase: str, base: Dict[str, str]) -> Dict[str, str]:
        out = dict(base)
        # Guided small-step neighborhood: flip K nodes, biasing to critical (high rank_u) nodes
        if getattr(self, 'sa_k', 0) and self.sa_k > 0:
            order = self._upward_rank(g, phase=phase)
            hot_n = max(1, int(max(0.05, self.critical_frac) * len(order)))
            pool = order[:hot_n] or order
            k = min(self.sa_k, len(pool))
            flip_nodes = set(random.sample(pool, k))
            for nid in flip_nodes:
                node = g.nodes[nid]
                acts = self._allowed_actions(node)
                if not acts:
                    continue
                old = out.get(nid, None)
                choices = [a for a in acts if a != old] or acts
                out[nid] = random.choice(choices)
        else:
            # fallback: independent flips by probability
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
        if getattr(self, 'stats', None):
            self.stats.set_phase(phase)
        cur_map = self._make_initial_action_map(g, phase)
        cur_cost, cur_sched = self._evaluate_action_map(g, phase, cur_map, dry_run=True)
        best_map, best_cost, best_sched = dict(cur_map), float(cur_cost), list(cur_sched)

        T = self.T0 if self.T0 > 0 else 0.05 * cur_cost
        for _ in range(max(1, self.sa_iters)):
            nb = self._neighbors(g, phase, cur_map)
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
    def __init__(self, *args, pop: int = 100, gens: int = 100, elite: int = 10, mut_prob: float = 0.05, cross_prob: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.pop = int(pop) # population size
        self.gens = int(gens) # generations
        self.elite = int(elite) # number of elites to keep
        self.mut_prob = float(mut_prob) # per-gene mutation probability
        self.cross_prob = float(cross_prob) # per-gene crossover probability(from parent A)

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

    def _mutate(self, g: TaskGraph, phase: str, m: Dict[str, str]) -> Dict[str, str]:
        out = dict(m)
        try:
            order = self._upward_rank(g, phase=phase)
            hot = set(order[:max(1, int(0.3 * len(order)))])
        except Exception:
            hot = set()
        for nid, node in g.nodes.items():
            p = self.mut_prob * (2.0 if nid in hot else 1.0)
            if random.random() < p:
                acts = self._allowed_actions(node)
                out[nid] = random.choice(acts)
        return out

    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        # seed population: greedy + randoms
        if getattr(self, 'stats', None):
            self.stats.set_phase(phase)
        seed = self._make_initial_action_map(g, phase)
        pop: List[Tuple[float, Dict[str, str]]] = []
        # evaluate helper
        def eval_map(m: Dict[str, str]) -> float:
            cost, _ = self._evaluate_action_map(g, phase, m, dry_run=True)
            return cost

        # initial population
        pop_maps = [seed]
        # extra heuristic seeds to diversify population
        def _pref_seed(pref: str) -> Dict[str, str]:
            m = {}
            for nid, node in g.nodes.items():
                if node.allowed.get(pref, False):
                    m[nid] = pref
                else:
                    m[nid] = 'auto'
            return m
        pop_maps += [_pref_seed('npu'), _pref_seed('pim')]
        if ALLOW_HYBRID:
            m_h = {}
            for nid, node in g.nodes.items():
                if node.allowed.get('npu', False) and (
                    node.allowed.get('pim', False)
                    or node.allowed.get('pima', False)
                    or node.allowed.get('pimd', False)
                ):
                    m_h[nid] = 'hybrid'
                else:
                    m_h[nid] = 'auto'
            pop_maps.append(m_h)
        # fill rest with randoms
        pop_maps += [self._random_map(g, phase) for _ in range(max(0, self.pop - len(pop_maps)))]
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
                child = self._mutate(g, phase, child)
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
    def __init__(self, *args, episodes: int = 120, epsilon_start: float = 0.5, epsilon_end: float = 0.05, alpha: float = 0.3, gamma: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.episodes = int(episodes) # training episodes
        self.epsilon_start = float(epsilon_start) # initial exploration probability
        self.epsilon_end = float(epsilon_end) # final exploration probability
        self.alpha = float(alpha) # learning rate
        self.gamma = float(gamma) # discount factor
        self.Q: Dict[Tuple[str, str], float] = {} # (算子类型, 动作类型) -> 价值

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
            return random.choice(acts) #以概率 eps 进行随机探索，如果 Q 值全为 0 也随机选择
        # greedy by Q
        best_a, best_q = None, -1e30
        for a in acts:
            q = self.Q.get((self._op_key(node), a), 0.0)
            if q > best_q:
                best_q, best_a = q, a
        return best_a or random.choice(acts)

    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        if getattr(self, 'stats', None):
            self.stats.set_phase(phase)
        order = self._upward_rank(g, phase=phase)
        # training episodes
        base_snap = self._snapshot()
        for ep in range(max(1, self.episodes)):
            eps = self._eps(ep)
            self._restore(base_snap)
            prev_op = None
            prev_a = None
            prev_q = None
            g_end = 0.0
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
                reward = -(max(g_end, ft) - g_end)
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
                    g_end = max(g_end, hy['finish'])
                else:
                    st, ft = self._earliest_finish_on_device(g, nid, d, self.label, phase, commit=True)
                    self.avail[d.name] = ft
                    g_end = max(g_end, ft)
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

    def __init__(self, *args, beam: int = 24, max_expansions: int = 3000, **kwargs):
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
        """
        Time-dimension lower bound: max of HEFT-style rank_u times over remaining nodes
        where rank_u includes avg compute + avg comm to successor.
        """
        if not remaining:
            return 0.0
        succ = {nid: list(g.successors(nid)) for nid in g.nodes}
        # compute rank_u times in reverse topological order
        try:
            topo = list(reversed(g.topological()))
        except Exception:
            topo = list(reversed(self._upward_rank(g, phase=phase)))
        rank_u: Dict[str, float] = {}
        for nid in topo:
            node = g.nodes[nid]
            if not succ[nid]:
                rank_u[nid] = self._avg_compute_cost(node, phase=phase)
            else:
                comp = self._avg_compute_cost(node, phase=phase)
                best = 0.0
                for v in succ[nid]:
                    comm = self._avg_comm_cost(node, g.nodes[v], phase=phase)
                    path = comm + rank_u.get(v, 0.0)
                    if path > best:
                        best = path
                rank_u[nid] = comp + best
        return max((rank_u.get(nid, 0.0) for nid in remaining), default=0.0)

    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        if getattr(self, 'stats', None):
            self.stats.set_phase(phase)
        order = self._topo_order(g, phase)
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

class HeftLookaheadScheduler(_SearchSchedulerMixin):
    """
    确定性的 HEFT 多步前瞻。
    """

    def __init__(self, *args, lookahead_depth: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookahead_depth = max(1, int(lookahead_depth))
        self._log_decision_trace = True

    def _lookahead_value(self, g, phase, base_actions: dict, order, start_idx, depth):
        """
        从 order[start_idx] 开始，最多往前看 depth 个“可决策节点”，
        返回在这些节点上最优动作组合下的整体 makespan。
        """
        if depth <= 0 or start_idx >= len(order):
            cost, _ = self._evaluate_action_map(g, phase, base_actions, dry_run=True,order=order)
            return float(cost)

        # 跳过那些没有可选动作的节点
        idx = start_idx
        while idx < len(order):
            nid = order[idx]
            node = g.nodes[nid]
            acts = self._allowed_actions_by_id(g,nid)
            if acts:
                break
            idx += 1

        if idx >= len(order):
            cost, _ = self._evaluate_action_map(g, phase, base_actions, dry_run=True,order=order)
            return float(cost)

        nid = order[idx]
        node = g.nodes[nid]
        acts = self._allowed_actions_by_id(g,nid)

        best_cost = math.inf
        for a in acts:
            prev = base_actions.get(nid, _MISSING)
            base_actions[nid] = a
            cost = self._lookahead_value(
                g=g,
                phase=phase,
                base_actions=base_actions,
                order=order,
                start_idx=idx + 1,
                depth=depth - 1,
            )
            if cost < best_cost:
                best_cost = cost
            if prev is _MISSING:
                del base_actions[nid]
            else:
                base_actions[nid] = prev

        return best_cost

    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        if getattr(self, "stats", None): self.stats.set_phase(phase)

        order = self._upward_rank(g, phase=phase)
        fixed_actions: Dict[str, str] = {}
        
        # Capture state at start of phase (e.g. prefill results) to pass to workers
        start_snap = self._snapshot()
        current_snap = self._snapshot()
        saved_stats = getattr(self, 'stats', None)
        saved_comm_stats = getattr(getattr(self, "comm", None), "stats", None)

        self.stats = None
        if getattr(self, "comm", None) is not None:
            self.comm.stats = None

        try:
            for idx, nid in enumerate(order):
                node = g.nodes[nid]
                op_name = node.attrs.get("op") or node.name
                acts = self._allowed_actions_by_id(g, nid)
                if not acts:
                    continue

                best_a: Optional[str] = None
                best_cost: float = float("inf")

                if len(acts) == 1 or (os.cpu_count() or 1) <= 1:
                    # 只有一个 action 或者 CPU 核太少就退化成原来的串行逻辑
                    for a in acts:
                        tmp_actions = dict(fixed_actions)
                        tmp_actions[nid] = a

                        cost = self._lookahead_value(
                            g=g, phase=phase,
                            base_actions=tmp_actions,
                            order=order,
                            start_idx=idx + 1,
                            depth=self.lookahead_depth - 1,
                        )
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("[HEFT-LA]   try action=%s cost=%.6f (serial)", a, cost)
                        if cost < best_cost:
                            best_cost = cost
                            best_a = a
                else:
                    def _eval_one_action(a: str, snap_state: Any) -> tuple[str, float]:
                        # 为每个线程构造一个独立的 Scheduler + Buffer，避免共享内部状态
                        worker = HeftLookaheadScheduler(
                            self.cluster,
                            self.cost,
                            self.label,
                            batch=self.batch,
                            seq_len=self.seq_len,
                            buffer=GlobalMemoryManager(),
                            lookahead_depth=self.lookahead_depth,
                        )
                        # Restore worker to the state at start of phase (including prefill cache/avail)
                        worker._restore(copy.deepcopy(snap_state))

                        tmp_actions = dict(fixed_actions)
                        tmp_actions[nid] = a
                        cost = worker._lookahead_value(
                            g=g, phase=phase,
                            base_actions=tmp_actions,
                            order=order,
                            start_idx=idx + 1,
                            depth=self.lookahead_depth - 1,
                        )
                        return a, float(cost)

                    max_workers = min(len(acts), os.cpu_count() or 16)
                    if max_workers < 1:
                        max_workers = 1

                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        futures = {ex.submit(_eval_one_action, a, current_snap): a for a in acts}
                        for fut in as_completed(futures):
                            a, cost = fut.result()
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug("[HEFT-LA]   try action=%s cost=%.6f (parallel)", a, cost)
                            if cost < best_cost:
                                best_cost = cost
                                best_a = a

                chosen_action = best_a or acts[0]
                if logger.isEnabledFor(logging.DEBUG):
                    cost_val = best_cost if math.isfinite(best_cost) else float("nan")
                    logger.debug("[HEFT-LA] => choose action=%s cost=%.6f", chosen_action, cost_val)
                fixed_actions[nid] = chosen_action

                # Incrementally update self and current_snap (NO TRACE during this block)
                self._step_node(g, nid, chosen_action, phase)
                current_snap = self._snapshot()
        finally:
            # 无论中途是否异常，都要恢复 stats 指针，避免把 scheduler 留在“无 stats”状态
            self.stats = saved_stats
            if getattr(self, "comm", None) is not None:
                self.comm.stats = saved_comm_stats if saved_comm_stats is not None else saved_stats

        self._restore(start_snap)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[HEFT-LA] final action map: %s", fixed_actions)
        _, sched = self._evaluate_action_map(g, phase, fixed_actions, dry_run=False)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[HEFT-LA] schedule done phase=%s tasks=%d", phase, len(sched))
        return sched


