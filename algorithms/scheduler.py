from __future__ import annotations
import os
import re
from config import attach_local_debug_filter
from dataclasses import dataclass,field
from typing import Dict, List, Tuple, Optional, Any, Iterable, OrderedDict, Hashable
try:  # pragma: no cover
    from typing import override  # type: ignore
except Exception:  # pragma: no cover
    from typing_extensions import override  # type: ignore
from collections import defaultdict
from collections import ChainMap
from collections.abc import Hashable, Mapping
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from plan_label import PlanLabel
from hardware import Cluster, DeviceSpec
from task_graph import TaskGraph, TaskNode, JointTaskGraph, JointNodeMeta
from cost_model import CostModel
from buffer_manager import GlobalMemoryManager, LRUCache
from config import (
    RANKU_INCLUDE_AVG_WEIGHT_LOAD,
    PIM_RUNTIME_LRU_THRESHOLD,
    SCHED_JOINT_LK_ENABLE,
    SCHED_JOINT_LK_H,
    SCHED_JOINT_LK_GAMMA,
    SCHED_JOINT_LK_CONSIST_LAMBDA,
    SCHED_JOINT_LK_PLAN_HINT_MAX,
    SCHED_WEIGHT_BIAS_ETA,
)
from types import SimpleNamespace
import logging
import math
import random
import copy
import heapq
import itertools
from stats_recorder import StatsRecorder
from comm_primitives import (
    normalize_topology,
    ring_allreduce,
    reduce_to_host,
    gather_to_host,
    scatter_from_host,
    transfer_p2p,
)

_MISSING = object()
DEBUG_SCHEDULER = False
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
    rank_u_by_phase: dict = field(default_factory=dict)      # phase -> {nid: upward_rank}
    allowed_actions: dict = field(default_factory=dict)       # nid -> tuple[action...]

@dataclass
class ScheduledTask:
    node_id: str
    device: str
    start: float
    finish: float

class CommManager:

    def __init__(self, cluster: Cluster, cost: CostModel, stats: StatsRecorder | None = None):
        self.cluster = cluster
        self.cost = cost
        self.timeline_end: Dict[Tuple[str, str], float] = {}
        self.stats = stats

    def reserve(
        self,
        src: str,
        dst: str,
        bytes_amount: int,
        earliest: float,
        commit: bool = True,
        tag: str | None = None,
        extra: dict | None = None,
    ):
        """Reserve a single directed transfer (src->dst) on the comm timeline."""
        key = (str(src), str(dst))
        bytes_amount = int(bytes_amount or 0)
        earliest = float(earliest or 0.0)

        ch_end = float(self.timeline_end.get(key, 0.0))
        start = max(ch_end, earliest)

        # Intra-device and empty transfers: do not consume the link.
        if src == dst or bytes_amount <= 0:
            end = float(start)
            if commit:
                self.timeline_end[key] = float(end)
            return (float(start), float(end))

        src_dev = self.cluster.devices.get(str(src))
        dst_dev = self.cluster.devices.get(str(dst))
        if src_dev is None or dst_dev is None:
            dt = float("inf")
        else:
            dt = float(self.cost.comm_cost(src_dev, dst_dev, int(bytes_amount)))

        end = float(start + dt)

        if commit:
            self.timeline_end[key] = float(end)
            if self.stats is not None:
                try:
                    self.stats.log_comm(
                        src=str(src),
                        dst=str(dst),
                        bytes=int(bytes_amount),
                        start=float(start),
                        end=float(end),
                        tag=tag or "comm",
                        extra=extra,
                    )
                except AttributeError:
                    pass

        return (float(start), float(end))


class SchedulerBase:
    def __init__(self, cluster: Cluster, cost: CostModel, label: PlanLabel, batch: int, seq_len: int, buffer: GlobalMemoryManager):
        self.cluster = cluster
        self.cost = cost
        self.label = label
        self.batch = batch
        self.seq_len = seq_len
        self.buffer = buffer or GlobalMemoryManager()
        self.stats = StatsRecorder()
        self.comm = CommManager(cluster, cost, stats=self.stats)

        # runtime state (PIM runtime is managed by buffer manager)
        self._node_host_store_end: Dict[str, float] = {}
        self._runtime_cap: Dict[str, int] = {} 
        self._act_used: Dict[str, int] = defaultdict(int)
        self._act_resident: Dict[Tuple[str, str], int] = {}# (dev_name, nid) -> bytes retained on device
        self._act_refcnt: Dict[str, int] = {}

        # Register PIM runtime budgets to buffer manager (unify all PIM-space checks there)
        pim_devs = [d for d in self.cluster.devices.values() if d.type == 'pim']
        if pim_devs:
            kv_in_pim = bool(getattr(self.label, 'kv_in_pim', getattr(self.label, 'kv_in_pim', False)))
            kv_total_bytes = int(getattr(self.label, 'kv_total_bytes', 0) or 0) if kv_in_pim else 0
            # Optional: per-PIM KV reservation (e.g., round-robin KV placement by layer).
            kv_bytes_by_pim = getattr(self.label, 'kv_bytes_by_pim', None) if kv_in_pim else None

            runtime_caps: List[int] = []
            phy_caps: List[int] = []
            for d in pim_devs:
                phy_bytes = int(d.mem_capacity_GB * 1024**3)
                runtime_cap = max(0, int(PIM_RUNTIME_LRU_THRESHOLD * phy_bytes))
                phy_caps.append(phy_bytes)
                runtime_caps.append(runtime_cap)

            total_runtime = sum(runtime_caps) or 1
            for d, phy_bytes, runtime_cap in zip(pim_devs, phy_caps, runtime_caps):
                kv_reserve = 0
                if kv_in_pim:
                    # Prefer explicit per-device KV budget if provided by memory planner.
                    if isinstance(kv_bytes_by_pim, Mapping) and kv_bytes_by_pim:
                        try:
                            need = kv_bytes_by_pim.get(d.name, kv_bytes_by_pim.get(str(d.name), 0))
                            kv_reserve = max(0, int(need or 0))
                        except Exception:
                            kv_reserve = 0
                        # Runtime budget is a modeling limit; clamp reservations to avoid negative weight cap.
                        kv_reserve = min(kv_reserve, int(runtime_cap))
                    elif kv_total_bytes > 0:
                        # Fallback: proportional split by runtime budgets.
                        kv_reserve = int(kv_total_bytes * (runtime_cap / total_runtime))
                weight_cap = max(0, int(runtime_cap - kv_reserve))
                self.buffer.register_pim_device(
                    d.name,
                    phy_bytes=phy_bytes,
                    runtime_limit_bytes=runtime_cap,
                    kv_reserved_bytes=kv_reserve,
                    kv_in_pim=kv_in_pim,
                    weight_cache_capacity_bytes=weight_cap,
                )

        # non-PIM activation budget
        for name, d in self.cluster.devices.items():
            if d.type == 'pim':
                continue
            phy = int(d.mem_capacity_GB * 1024**3)
            cap = max(0, int(PIM_RUNTIME_LRU_THRESHOLD * phy))
            self._runtime_cap[name] = cap
            self.buffer.register_runtime_device(
                str(name),phy_bytes=int(phy),runtime_limit_bytes=int(cap),weight_cache_capacity_bytes=int(cap),
            )

        #node schedule
        self._node_finish_time: Dict[str, float] = {}
        self._node_placement: Dict[str, str] = {}
        self._node_out_fmt: Dict[str, str] = {}
        self.avail: Dict[str, float] = {name: 0.0 for name in self.cluster.devices}

        #weight format and caching
        self.weight_cached: Dict[Tuple[str, str], bool] = {}
        self.storage_fmt_map: Dict[str, str] = {}
        self._weight_load_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self._weight_sizes: Dict[str, int] = {}

    def set_seq_len(self, seq_len: int) -> None:
        self.seq_len = int(seq_len)

    def _executor_device_types(self) -> Tuple[str, ...]:
        try:
            has_npu = bool(self.cluster.devices_by_type('npu'))
        except Exception:
            has_npu = False

        types: List[str] = []
        if has_npu:
            types.append('npu')
        # Always keep PIM as an executor if present.
        try:
            if self.cluster.devices_by_type('pim'):
                types.append('pim')
        except Exception:
            pass
        # Only treat CPU as an executor when NPU is absent.
        if (not has_npu):
            try:
                if self.cluster.devices_by_type('cpu'):
                    types.append('cpu')
            except Exception:
                pass

        # Fallback: never return empty.
        if not types:
            try:
                types = sorted({str(getattr(d, 'type', '')) for d in self.cluster.devices.values() if getattr(d, 'type', None)})
            except Exception:
                types = ['cpu']
        return tuple(types)

    def reset_state(self, *, clear_caches: bool = True) -> None:
        """Reset scheduler runtime state."""

        # Per-node / per-schedule bookkeeping (always cleared)
        self._node_finish_time.clear()
        self._node_placement.clear()
        self._node_out_fmt.clear()

        self._node_host_store_end.clear()
        self._act_used.clear()
        self._act_resident.clear()
        self._act_refcnt.clear()
        self._collective_output_devs = {}
        # KV/activation runtime states on PIM are centrally managed in buffer manager
        try:
            self.buffer.reset_runtime_state()
        except Exception:
            pass

        if clear_caches:
            # Clear weight caches on all devices (PIM + NPU/CPU) and scheduler-side
            # cache bookkeeping.
            for cache in self.buffer.device_cache.values():
                cache.items.clear()
                cache.order.clear()
                cache.used = 0
                cache.pinned.clear()
            self.weight_cached.clear()
            self.storage_fmt_map.clear()
            self._weight_load_count.clear()
            self._weight_sizes.clear()

    # ------------------------------------------------------------------
    # Weight residency policy
    # ------------------------------------------------------------------
    def _weights_preloaded_on_pim(self) -> bool:

        lb = getattr(self, 'label', None)
        if lb is None:
            return False
        for a in (
            'weights_preloaded_on_pim',
            'weights_in_pim',
            'all_weights_in_pim',
            'weights_on_pim',
            'weights_resident_on_pim',
        ):
            try:
                if bool(getattr(lb, a, False)):
                    return True
            except Exception:
                pass
        return False

    # -------- Node/graph abstraction hooks --------

    def _node_phase(self, g: Any, nid: str, default_phase: str) -> str:
        """Effective phase: prefer per-node meta.phase if present, else fallback."""
        return str(default_phase)
    
    def _node_batch(self, g: Any, nid: str, phase: str) -> int:
        """Effective batch for a node."""
        return int(getattr(self, "batch", 1) or 1)

    def _node_seq_len(self, g: Any, nid: str, phase: str) -> int:
        """Effective seq_len for a node. return meta.seq_len if exists else self.seq_len
        """
        try:
            return int(getattr(self, "seq_len", 0) or 0)
        except Exception:
            return 0

    # -------- KV placement helpers (multi-PIM partitioning) --------
    def _node_layer_id(self, node: Any) -> Optional[int]:
        """Best-effort layer id extraction.

        We primarily use node.attrs['layer'] (as produced by model_definition.py).
        """
        try:
            attrs = getattr(node, "attrs", None)
            if isinstance(attrs, Mapping):
                if "layer" in attrs:
                    return int(attrs.get("layer"))
                if "layer_id" in attrs:
                    return int(attrs.get("layer_id"))
        except Exception:
            pass
        return None

    def _node_kv_head_range(self, node: Any) -> Optional[Tuple[int, int]]:
        """Return [kv_head_start, kv_head_end) for head-sharded nodes."""
        try:
            attrs = getattr(node, "attrs", None)
            if not isinstance(attrs, Mapping):
                return None
            if "kv_head_start" in attrs and "kv_head_end" in attrs:
                return (int(attrs["kv_head_start"]), int(attrs["kv_head_end"]))
        except Exception:
            return None
        return None
         
    def _kv_mapped_pim_name_for_layer(self, layer: int) -> Optional[str]:
        """Return mapped pim device name for a layer, if label provides it."""
        try:
            m = getattr(self.label, "kv_layer_to_pim", None)
        except Exception:
            m = None
        if not isinstance(m, Mapping) or not m:
            return None
        # Allow int or str keys.
        if layer in m:
            try:
                return str(m[layer])
            except Exception:
                return None
        s = str(int(layer))
        if s in m:
            try:
                return str(m[s])
            except Exception:
                return None
        return None

    def _kv_pim_for_node(self, node: Any) -> Optional[DeviceSpec]:

        # KV not on PIM -> no mapped PIM
        try:
            if not bool(getattr(self.label, "kv_in_pim", False)):
                return None
        except Exception:
            return None

        # KV-head based mapping
        head_map = getattr(self.label, "kv_head_to_pim", None)
        if isinstance(head_map, Mapping) and head_map:
            r = self._node_kv_head_range(node)
            if r is not None:
                hs, he = int(r[0]), int(r[1])
                if hs < he:
                    # Find the mapped PIM name for the first head in the range.
                    pim_name = head_map.get(hs, head_map.get(str(hs)))
                    if pim_name is not None:
                        # Sanity: the entire head range should map to the same PIM.
                        for h in range(hs, he):
                            if head_map.get(h, head_map.get(str(h))) != pim_name:
                                return None
                        dev = self.cluster.devices.get(str(pim_name))
                        if dev is not None and str(getattr(dev, "type", "")).lower() == "pim":
                            return dev

            # Node has no head range: if all KV heads map to ONE PIM, infer that PIM.
            try:
                uniq = {str(v) for v in head_map.values() if v is not None}
            except Exception:
                uniq = set()
            if len(uniq) == 1:
                pn = next(iter(uniq))
                pim_devs = {
                    str(d.name): d
                    for d in (self.cluster.devices_by_type("pim") or [])
                    if str(getattr(d, "type", "")).lower() == "pim"
                }
                dev = pim_devs.get(str(pn))
                if dev is not None:
                    return dev
        return None


    def _preferred_kv_write_device(self, g: Any, nid: str) -> Optional[DeviceSpec]:
        try:
            node = g.nodes[nid]
        except Exception:
            return None

        name = str(getattr(node, "name", "")).upper()
        if name not in ("K_WRITE", "V_WRITE", "KV_WRITE"):
            return None


        if bool(getattr(self.label, "kv_in_pim", False)):
            mapped = self._kv_pim_for_node(node)
            return mapped
        return self.cost.get_host_device()

    def _node_allowed_on(self, node: TaskNode, dev: DeviceSpec) -> bool:

        # ---- 0) KV-write hard rule (override operator/baseline allow-list) ----
        try:
            kv_in_pim = bool(getattr(self.label, "kv_in_pim", False))
        except Exception:
            kv_in_pim = False

        name_up = str(getattr(node, "name", "") or "").upper()
        dev_type = str(getattr(dev, "type", "") or "").lower()
        dev_name = str(getattr(dev, "name", "") or "")
    
        # KV write ops under KV-on-PIM: must execute on the mapped PIM.
        if kv_in_pim and name_up in ("K_WRITE", "V_WRITE", "KV_WRITE"):
            if dev_type != "pim":
                return False
            mapped = self._kv_pim_for_node(node)
            if mapped is None:
                return False  # mapping unavailable -> conservative
            return dev_name == str(getattr(mapped, "name", ""))

        # ---- 1) Communication primitives (collectives / transfers) ----
        try:
            attrs = getattr(node, "attrs", {}) or {}
        except Exception:
            attrs = {}
        prim = attrs.get("primitive", None)
        prim_up = str(prim or name_up or "").upper()
        if prim_up in ("ALLREDUCE", "ALL_REDUCE", "ALL-REDUCE", "REDUCE", "GATHER", "SCATTER", "TRANSFER"):
            host = self.cost.get_host_device()
            host_name = str(getattr(host, "name", "")) if host is not None else ""
            return dev_name == host_name

        # ---- 2) KV locality restriction (multi-PIM KV placement) ----
        if kv_in_pim:
            kv_local_ops = {
                "K", "V",
                "QK", "SOFTMAX", "SV",
            }
            if name_up in kv_local_ops and dev_type == "pim":
                mapped = self._kv_pim_for_node(node)
                if mapped is None:
                    return False
                if dev_name != str(getattr(mapped, "name", "")):
                    return False

        # ---- 3) operator-level allow-list ----
        allowed = getattr(node, "allowed", None)
        if isinstance(allowed, Mapping):
            key = getattr(dev, "type", None) or str(dev)
            if not bool(allowed.get(key, True)):
                return False

        return True

    # -----------------------------
    # Communication primitive helpers
    # -----------------------------
    def _comm_primitive(self, node: TaskNode) -> Optional[str]:
        """Return the normalized communication primitive kind for `node`, or None.

        We treat these nodes as communication-only operators:
          - ALLREDUCE (or decomposed REDUCE/SCATTER)
          - REDUCE / GATHER / SCATTER / TRANSFER
        """
        try:
            attrs = getattr(node, "attrs", {}) or {}
        except Exception:
            attrs = {}
        prim = attrs.get("primitive", None)
        name = getattr(node, "name", None)
        prim_up = str(prim or name or "").upper()
        if prim_up in ("ALLREDUCE", "ALL_REDUCE", "ALL-REDUCE", "REDUCE", "GATHER", "SCATTER", "TRANSFER"):
            return prim_up
        return None

    def _is_comm_node(self, node: TaskNode) -> bool:
        return self._comm_primitive(node) is not None

    def _node_weight_id(self, node: TaskNode) -> Optional[str]:
        wid = getattr(node, "weight_id", None)
        if wid in (None, "", 0):
            return None
        try:
            return str(wid)
        except Exception:
            return None

    def _node_weight_size(self, node: TaskNode) -> int:
        try:
            return int(getattr(node, "weight_size", 0) or 0)
        except Exception:
            return 0

    def _earliest_finish_on_device(
        self,
        g: TaskGraph,
        nid: str,
        dev: DeviceSpec,
        label: PlanLabel,
        phase: str,
        commit: bool,
    ) -> Tuple[float, float]:
        node = g.nodes[nid]
        phase_eff = self._node_phase(g, nid, phase)
        batch = self._node_batch(g, nid, phase_eff)
        seq_len = self._node_seq_len(g, nid, phase_eff)
        
        #---------------------1. KV write specially (write KV cache back)
        if node.name.upper() in ("K_WRITE", "V_WRITE", "KV_WRITE"):
            kv_in_pim = bool(getattr(self.label, "kv_in_pim", False))
            _, out_write_nd = self.cost.estimate_activation_bytes(node, batch, seq_len, phase_eff)
            size_nd = self.cost.format_size(out_write_nd, "ND")
            if commit:
                logger.debug("[kv-write] node=%s target=%s kv_in_pim=%s bytes_nd=%d", nid, getattr(dev, 'name', dev), bool(kv_in_pim), int(out_write_nd or 0))

            if kv_in_pim:
                pim_devs = self.cluster.devices_by_type("pim")
                if not pim_devs:
                    raise RuntimeError("kv_in_pim is True but no PIM device exists")

                target_pim = self._kv_pim_for_node(node) or pim_devs[0]

                # Ensure the K/V activation is available on the target PIM before the store.
                ready_kv = self._ready_time_for_device(g, nid, target_pim, phase_eff, commit)
                start = max(float(self.avail.get(target_pim.name, 0.0)), float(ready_kv))
                finish = start + float(self.cost.pim_write_time(int(out_write_nd), target_pim))
                if commit:
                    logger.debug(
                        "[kv-write] node=%s -> %s dur=%.4f", nid, target_pim.name,
                        float(max(0.0, finish - start)),
                    )
                if commit:
                    self.avail[target_pim.name] = float(finish)
                    self._node_finish_time[nid] = float(finish)
                    self._node_placement[nid] = str(target_pim.name)
                    self._node_out_fmt[nid] = "ND"
                    self._act_resident[(str(target_pim.name), str(nid))] = 0
                return float(start), float(finish)
            else:
                # kv in host: convert on source device then send to host
                host = self.cost.get_host_device()
                ready_kv = self._ready_time_for_device(g, nid, dev, phase_eff, commit)
                conv_start = max(float(self.avail.get(dev.name, 0.0)), float(ready_kv))
                conv_cost = self.cost.format_conversion_time(size_nd, self.cost.device_preferred_fmt(dev), "ND", dev)
                _, l2e = self.comm.reserve(dev.name, host.name, size_nd, earliest=conv_start + conv_cost, commit=commit, tag="kv_write")
                link_end = float(l2e)
                host_wr_t = float(self.cost.cpu_write_time(int(size_nd), host))
                wr_start = max(float(self.avail.get(host.name, 0.0)), float(link_end))
                finish = float(wr_start + host_wr_t)
                if commit:
                    logger.debug(
                        "[kv-write] node=%s %s->host dur=%.4f", nid, dev.name, float(max(0.0, finish - conv_start)))
                if commit:
                    self.avail[dev.name] = max(self.avail.get(dev.name, 0.0), link_end)
                    self.avail[host.name] = max(self.avail.get(host.name, 0.0), finish)
                    self._node_finish_time[nid] = finish
                    self._node_placement[nid] = host.name
                    self._node_out_fmt[nid] = "ND"
                return float(conv_start), float(finish)
                
        #---------------------1b. Communication primitives -------------------
        attrs = getattr(node, "attrs", {}) or {}
        prim = attrs.get("primitive", None)
        prim_up = str(prim or getattr(node, "name", "") or "").upper()

        if prim_up in ("ALLREDUCE", "ALL_REDUCE", "ALL-REDUCE"):
            return self._earliest_finish_allreduce(g, nid, label, phase_eff, commit)

        if prim_up in ("REDUCE", "GATHER", "SCATTER", "TRANSFER"):
            host = self.cost.get_host_device()
            # These primitives are modeled as host-centric control ops; enforce placement on host.
            if str(getattr(dev, "name", "")) != str(getattr(host, "name", "")):
                return (float("inf"), float("inf"))
            if prim_up == "REDUCE":
                return self._earliest_finish_reduce(g, nid, label, phase_eff, commit)
            if prim_up == "GATHER":
                return self._earliest_finish_gather(g, nid, label, phase_eff, commit)
            if prim_up == "SCATTER":
                return self._earliest_finish_scatter(g, nid, label, phase_eff, commit)
            if prim_up == "TRANSFER":
                return self._earliest_finish_transfer(g, nid, label, phase_eff, commit)

        attrs = getattr(node, "attrs", {}) or {}
        ready = self._ready_time_for_device(g, nid, dev, phase_eff, commit)

        #---------------------2.  If this node consumes cached KV (QK/SV), add KV load before compute
        kv_ready = float(ready)
        if node.name.upper() in ("QK", "SV"):
            kv_bytes = int(self.cost.estimate_kv_cache_read_bytes(node, batch, seq_len, phase_eff))
            if kv_bytes > 0:
                dev_fmt = str(self.cost.device_preferred_fmt(dev))
                size_nd = int(self.cost.format_size(int(kv_bytes), "ND"))

                if bool(getattr(label, "kv_in_pim", False)):
                    # KV cache is sharded on PIMs by KV head. For a shard, KV should come from a single mapped PIM.
                    src_pim = self._kv_pim_for_node(node)
                    if src_pim is None:
                        # Fallback: treat as host-resident KV
                        host = self.cost.get_host_device()
                        if str(dev.name) == str(host.name):
                            mem_t = float(self.cost.cpu_read_time(int(kv_bytes), host))
                            rd_start = max(float(self.avail.get(host.name, 0.0)), float(ready))
                            rd_end = rd_start + mem_t
                            if commit:
                                self.avail[host.name] = rd_end
                            kv_ready = max(kv_ready, rd_end)
                        else:
                            host_rd_t = float(self.cost.cpu_read_time(int(size_nd), host))
                            rd_start = max(float(self.avail.get(host.name, 0.0)), float(ready))
                            rd_end = float(rd_start + host_rd_t)
                            if commit:
                                self.avail[host.name] = max(float(self.avail.get(host.name, 0.0)), float(rd_end))
                            _, xfer_end = self.comm.reserve(host.name, dev.name, size_nd, earliest=float(rd_end), commit=commit, tag="kv_load")
                            conv_t = float(self.cost.format_conversion_time(int(size_nd), "ND", dev_fmt, dev))
                            kv_ready = max(kv_ready, float(xfer_end) + float(NONOVERLAP_TIME) * conv_t)
                    else:
                        # If we execute on the same PIM and are in trace mode, the trace already models KV traffic.
                        if (
                            str(getattr(dev, "type", "")).lower() == "pim"
                            and str(dev.name) == str(src_pim.name)
                            and (not getattr(self.cost, "pim_fast_mode", True))
                        ):
                            pass
                        else:
                            # Read from src PIM memory (serialize by PIM availability)
                            rd_t = float(self.cost.activation_read_time_pim(int(kv_bytes)))
                            rd_start = max(float(self.avail.get(src_pim.name, 0.0)), float(ready))
                            rd_end = rd_start + rd_t
                            if commit:
                                self.avail[src_pim.name] = rd_end

                            if str(dev.name) == str(src_pim.name):
                                kv_ready = max(kv_ready, rd_end)
                            else:
                                # Transfer KV to target device (prefer direct, otherwise via host)
                                host = self.cost.get_host_device()
                                t_direct = float(self.cost.comm_cost(src_pim, dev, int(size_nd)))
                                if math.isfinite(t_direct):
                                    _, xfer_end = self.comm.reserve(
                                        src_pim.name, dev.name, int(size_nd),
                                        earliest=float(rd_end), commit=commit, tag="kv_load",
                                    )
                                else:
                                    _, t1_end = self.comm.reserve(
                                        src_pim.name, host.name, int(size_nd),
                                        earliest=float(rd_end), commit=commit, tag="kv_load",
                                    )
                                    _, xfer_end = self.comm.reserve(
                                        host.name, dev.name, int(size_nd),
                                        earliest=float(t1_end), commit=commit, tag="kv_load",
                                    )
                                conv_t = float(self.cost.format_conversion_time(int(size_nd), "ND", dev_fmt, dev))
                                kv_ready = max(kv_ready, float(xfer_end) + conv_t)

                else:
                    host = self.cost.get_host_device()
                    host_rd_t = float(self.cost.cpu_read_time(int(size_nd), host))
                    rd_start = max(float(self.avail.get(host.name, 0.0)), float(ready))
                    rd_end = float(rd_start + host_rd_t)
                    if commit:
                        self.avail[host.name] = max(float(self.avail.get(host.name, 0.0)), float(rd_end))

                    l2s, l2e = self.comm.reserve(host.name, dev.name, size_nd, earliest=float(rd_end), commit=commit, tag="kv_load")
                    kv_ready = max(kv_ready, float(l2s)
                        + float(self.cost.combine_transfer_and_convert(host, dev, int(size_nd), "ND", str(self.cost.device_preferred_fmt(dev)),)
                        ),
                    )
                if commit:
                    logger.debug("[kv-load] node=%s target=%s kv_ready=%.4f", nid, dev.name, kv_ready)

        #---------------------3. normal weight load + compute + activation handling
        start = max(float(self.avail.get(dev.name, 0.0)), kv_ready)
        compute = self.cost.node_device_cost(node, dev, label, batch, seq_len, phase_eff)
        wload = self._weight_load_time(node, dev, start, commit)
        finish = start + wload + compute

        if commit:
            self.avail[dev.name] = finish
            self._node_finish_time[nid] = finish
            self._node_placement[nid] = dev.name
            self._node_out_fmt[nid] = self.cost.device_preferred_fmt(dev)
            out_read_nd, out_write_nd = self.cost.estimate_activation_bytes(node, self.batch, self.seq_len, phase)
            out_nd = max(int(out_write_nd), int(out_read_nd))

            #TODO：check activation residency or host store
            if self.buffer.pim_reserve_activation(dev.name, out_nd, commit=True):
                # Activation stays resident on this device.
                self._act_resident[(dev.name, nid)] = out_nd
            else:
                # Unified policy (same as PIM): evict weights first; if still
                # cannot fit, spill activation to host.
                src_fmt = self.cost.device_preferred_fmt(dev)
                self._ensure_host_store(nid, dev, out_nd, src_fmt, finish, commit=True)
        return float(start), float(finish)

    def _earliest_finish_allreduce(
        self,
        g: TaskGraph,
        nid: str,
        label: PlanLabel,
        phase: str,
        commit: bool,
    ) -> Tuple[float, float]:
        node = g.nodes[nid]
        phase_v = self._node_phase(g, nid, phase)
        batch_v = self._node_batch(g, nid, phase_v)
        seq_v = self._node_seq_len(g, nid, phase_v)

        rd_b, wr_b = self.cost.estimate_activation_bytes(node, batch_v, seq_v, phase_v)
        tensor_bytes = int(max(int(rd_b), int(wr_b), 0))

        # Participants inferred from predecessor placements.
        dev_ready: Dict[str, float] = {}
        participants: List[str] = []
        for u in g.predecessors(nid):
            dname = self._node_placement.get(u)
            if not dname:
                dname = str(next(iter(self.cluster.devices.keys()), "CPU0"))
            dname = str(dname)
            t_ready = float(self._node_finish_time.get(u, 0.0))
            if dname not in dev_ready:
                dev_ready[dname] = t_ready
                participants.append(dname)
            else:
                dev_ready[dname] = max(float(dev_ready[dname]), t_ready)

        if not participants or tensor_bytes <= 0:
            start = float(max(dev_ready.values(), default=0.0))
            end = float(start)
            if commit:
                canon = participants[0] if participants else str(next(iter(self.cluster.devices.keys()), "CPU0"))
                self._node_finish_time[nid] = float(end)
                self._node_placement[nid] = str(canon)
                self._node_out_fmt[nid] = "ND"
                self._collective_output_devs[nid] = set(participants or [canon])
            return (float(start), float(end))

        ring = sorted(set(participants))
        start = float(max(dev_ready.get(d, 0.0) for d in ring))
        start = float(max(start, max(float(self.avail.get(d, 0.0)) for d in ring)))

        topo = normalize_topology(getattr(self.cluster, "topology", None))
        if len(ring) <= 1:
            end = float(start)
        elif topo == "fc":
            # No contention: analytic ring allreduce.
            end = float(ring_allreduce(cost=self.cost, cluster=self.cluster, ring=ring, tensor_bytes=int(tensor_bytes), start=float(start)))
        else:
            # STAR fallback: reduce->host + broadcast scatter.
            host = self.cost.get_host_device()
            red_end = float(reduce_to_host(comm=self.comm, cost=self.cost, cluster=self.cluster, participants=ring, tensor_bytes=int(tensor_bytes), start=float(start), commit=commit, tag='reduce', host_name=str(getattr(host, 'name', ''))))
            # host accumulation time (adds).
            try:
                dtype_b = float(self.cost._act_dtype_bytes(node, phase_v))
            except Exception:
                dtype_b = 2.0
            elems = float(tensor_bytes) / max(1.0, float(dtype_b))
            flops = float(max(0, len(ring) - 1)) * float(elems)
            acc_t = float(self.cost.flop_time(flops, host))
            red_end2 = float(red_end + acc_t)
            end = float(scatter_from_host(comm=self.comm, cost=self.cost, cluster=self.cluster, targets=ring, bytes_per_target=int(tensor_bytes), start=float(red_end2), commit=commit, tag='scatter', host_name=str(getattr(host, 'name', ''))))

        if commit:
            canon = ring[0]
            self._node_finish_time[nid] = float(end)
            self._node_placement[nid] = str(canon)
            self._node_out_fmt[nid] = "ND"
            self._collective_output_devs[nid] = set(ring)
            for d in ring:
                self.avail[d] = max(float(self.avail.get(d, 0.0)), float(end))

        return (float(start), float(end))


    def _earliest_finish_reduce(
        self,
        g: TaskGraph,
        nid: str,
        label: PlanLabel,
        phase: str,
        commit: bool,
    ) -> Tuple[float, float]:
        """Host-centric REDUCE: all predecessors send tensor to host; host sums.
        Output is resident on host only.
        """
        node = g.nodes[nid]
        phase_v = self._node_phase(g, nid, phase)
        batch_v = self._node_batch(g, nid, phase_v)
        seq_v = self._node_seq_len(g, nid, phase_v)
        rd_b, wr_b = self.cost.estimate_activation_bytes(node, batch_v, seq_v, phase_v)
        tensor_bytes = int(max(int(rd_b), int(wr_b), 0))
        host = self.cost.get_host_device()
        host_name = str(getattr(host, "name", "CPU0") or "CPU0")

        # Participants from predecessor placements.
        dev_ready: Dict[str, float] = {}
        sources: List[str] = []
        for u in g.predecessors(nid):
            dname = self._node_placement.get(u)
            if not dname:
                dname = str(next(iter(self.cluster.devices.keys()), host_name))
            dname = str(dname)
            t_ready = float(self._node_finish_time.get(u, 0.0))
            if dname not in dev_ready:
                dev_ready[dname] = t_ready
                sources.append(dname)
            else:
                dev_ready[dname] = max(float(dev_ready[dname]), t_ready)

        start = float(max([float(self.avail.get(host_name, 0.0))] + [max(float(self.avail.get(d,0.0)), float(dev_ready.get(d,0.0))) for d in sources] or [0.0]))
        if not sources or tensor_bytes <= 0:
            end = float(start)
            if commit:
                self._node_finish_time[nid] = float(end)
                self._node_placement[nid] = host_name
                self._node_out_fmt[nid] = "ND"
                self._collective_output_devs[nid] = {host_name}
            return (float(start), float(end))

        red_end = float(reduce_to_host(comm=self.comm, cost=self.cost, cluster=self.cluster, participants=sources, tensor_bytes=int(tensor_bytes), start=float(start), commit=commit, tag='reduce', host_name=str(getattr(host, 'name', ''))))
        # Accumulation at host (adds).
        try:
            dtype_b = float(self.cost._act_dtype_bytes(node, phase_v))
        except Exception:
            dtype_b = 2.0
        elems = float(tensor_bytes) / max(1.0, float(dtype_b))
        flops = float(max(0, len(sources) - 1)) * float(elems)
        acc_t = float(self.cost.flop_time(flops, host))
        end = float(red_end + acc_t)

        if commit:
            self._node_finish_time[nid] = float(end)
            self._node_placement[nid] = host_name
            self._node_out_fmt[nid] = "ND"
            self._collective_output_devs[nid] = {host_name}
            self.avail[host_name] = max(float(self.avail.get(host_name, 0.0)), float(end))
            for d in sources:
                self.avail[d] = max(float(self.avail.get(d, 0.0)), float(red_end))
        return (float(start), float(end))


    def _earliest_finish_gather(
        self,
        g: TaskGraph,
        nid: str,
        label: PlanLabel,
        phase: str,
        commit: bool,
    ) -> Tuple[float, float]:
        """Host-centric GATHER: all predecessors send tensor to host; concatenate/collect.
        Output is resident on host only.
        """
        node = g.nodes[nid]
        phase_v = self._node_phase(g, nid, phase)
        batch_v = self._node_batch(g, nid, phase_v)
        seq_v = self._node_seq_len(g, nid, phase_v)
        rd_b, wr_b = self.cost.estimate_activation_bytes(node, batch_v, seq_v, phase_v)
        tensor_bytes = int(max(int(rd_b), int(wr_b), 0))
        host = self.cost.get_host_device()
        host_name = str(getattr(host, "name", "CPU0") or "CPU0")
        dev_ready: Dict[str, float] = {}
        sources: List[str] = []
        for u in g.predecessors(nid):
            dname = self._node_placement.get(u)
            if not dname:
                dname = str(next(iter(self.cluster.devices.keys()), host_name))
            dname = str(dname)
            t_ready = float(self._node_finish_time.get(u, 0.0))
            if dname not in dev_ready:
                dev_ready[dname] = t_ready
                sources.append(dname)
            else:
                dev_ready[dname] = max(float(dev_ready[dname]), t_ready)
        start = float(max([float(self.avail.get(host_name, 0.0))] + [max(float(self.avail.get(d,0.0)), float(dev_ready.get(d,0.0))) for d in sources] or [0.0]))
        if not sources or tensor_bytes <= 0:
            end = float(start)
            if commit:
                self._node_finish_time[nid] = float(end)
                self._node_placement[nid] = host_name
                self._node_out_fmt[nid] = "ND"
                self._collective_output_devs[nid] = {host_name}
            return (float(start), float(end))
        end = float(gather_to_host(comm=self.comm, cost=self.cost, cluster=self.cluster, participants=sources, tensor_bytes=int(tensor_bytes), start=float(start), commit=commit, tag='gather', host_name=str(getattr(host, 'name', ''))))
        if commit:
            self._node_finish_time[nid] = float(end)
            self._node_placement[nid] = host_name
            self._node_out_fmt[nid] = "ND"
            self._collective_output_devs[nid] = {host_name}
            self.avail[host_name] = max(float(self.avail.get(host_name, 0.0)), float(end))
            for d in sources:
                self.avail[d] = max(float(self.avail.get(d, 0.0)), float(end))
        return (float(start), float(end))


    def _earliest_finish_scatter(
        self,
        g: TaskGraph,
        nid: str,
        label: PlanLabel,
        phase: str,
        commit: bool,
    ) -> Tuple[float, float]:
        """Host-centric SCATTER: host sends tensor to target devices.

        scatter_mode:
          - broadcast: each target receives full tensor_bytes
          - partition: each target receives ceil(tensor_bytes/num_targets)
        """
        node = g.nodes[nid]
        attrs = getattr(node, "attrs", {}) or {}
        phase_v = self._node_phase(g, nid, phase)
        batch_v = self._node_batch(g, nid, phase_v)
        seq_v = self._node_seq_len(g, nid, phase_v)
        rd_b, wr_b = self.cost.estimate_activation_bytes(node, batch_v, seq_v, phase_v)
        tensor_bytes = int(max(int(rd_b), int(wr_b), 0))
        host = self.cost.get_host_device()
        host_name = str(getattr(host, "name", "CPU0") or "CPU0")
        # Target selection
        targets = []
        t_attr = attrs.get("targets", None)
        if isinstance(t_attr, (list, tuple)):
            targets = [str(x) for x in t_attr if x is not None]
        else:
            ttype = str(attrs.get("target_type", "pim") or "pim").lower()
            if ttype in ("all", "*", "any"):
                targets = [d.name for d in self.cluster.devices.values() if d.name != host_name]
            else:
                targets = [d.name for d in self.cluster.devices_by_type(ttype) if d.name != host_name]
        # Ensure deterministic order
        targets = sorted(set(targets))

        # Ready time on host from predecessors
        ready_host = self._ready_time_for_device(g, nid, host, phase_v, commit)
        start = float(max(float(self.avail.get(host_name, 0.0)), float(ready_host)))
        if not targets or tensor_bytes <= 0:
            end = float(start)
            if commit:
                self._node_finish_time[nid] = float(end)
                self._node_placement[nid] = host_name
                self._node_out_fmt[nid] = "ND"
                self._collective_output_devs[nid] = set(targets + [host_name])
            return (float(start), float(end))

        mode = str(attrs.get("scatter_mode", "broadcast") or "broadcast").lower()
        if mode in ("partition", "shard", "split"):
            import math
            per = int(math.ceil(float(tensor_bytes) / float(max(1, len(targets)))))
        else:
            per = int(tensor_bytes)
        end = float(scatter_from_host(comm=self.comm, cost=self.cost, cluster=self.cluster, targets=targets, bytes_per_target=int(per), start=float(start), commit=commit, tag='scatter', host_name=str(getattr(host, 'name', ''))))

        if commit:
            self._node_finish_time[nid] = float(end)
            self._node_placement[nid] = host_name
            self._node_out_fmt[nid] = "ND"
            self._collective_output_devs[nid] = set(targets + [host_name])
            self.avail[host_name] = max(float(self.avail.get(host_name, 0.0)), float(end))
            for d in targets:
                self.avail[d] = max(float(self.avail.get(d, 0.0)), float(end))
        return (float(start), float(end))


    def _earliest_finish_transfer(
        self,
        g: TaskGraph,
        nid: str,
        label: PlanLabel,
        phase: str,
        commit: bool,
    ) -> Tuple[float, float]:
        """Explicit point-to-point TRANSFER.

        Expects node.attrs to optionally include:
          - src: device name
          - dst: device name
          - bytes: override bytes to transfer
        If not provided, we infer bytes from activation size of the node and
        use predecessors placements as src and host as dst.
        """
        node = g.nodes[nid]
        attrs = getattr(node, "attrs", {}) or {}
        phase_v = self._node_phase(g, nid, phase)
        batch_v = self._node_batch(g, nid, phase_v)
        seq_v = self._node_seq_len(g, nid, phase_v)
        rd_b, wr_b = self.cost.estimate_activation_bytes(node, batch_v, seq_v, phase_v)
        tensor_bytes = int(max(int(rd_b), int(wr_b), 0))
        override = attrs.get("bytes", attrs.get("bytes_nd", None))
        if override is not None:
            try:
                tensor_bytes = int(override)
            except Exception:
                pass
        host = self.cost.get_host_device()
        host_name = str(getattr(host, "name", "CPU0") or "CPU0")
        src = attrs.get("src", None)
        dst = attrs.get("dst", None)
        if src is None:
            # use first predecessor placement
            preds = list(g.predecessors(nid))
            if preds:
                src = self._node_placement.get(preds[0], host_name)
            else:
                src = host_name
        if dst is None:
            dst = host_name
        src = str(src); dst = str(dst)
        # Earliest start based on predecessor completion and device availability
        pred_finish = float(max((float(self._node_finish_time.get(u, 0.0)) for u in g.predecessors(nid)), default=0.0))
        start = float(max(pred_finish, float(self.avail.get(src, 0.0)), float(self.avail.get(dst, 0.0))))
        end = float(transfer_p2p(comm=self.comm, cost=self.cost, cluster=self.cluster, src=src, dst=dst, bytes_amount=int(tensor_bytes), start=float(start), commit=commit, tag='transfer'))
        if commit:
            self._node_finish_time[nid] = float(end)
            self._node_placement[nid] = str(dst)
            self._node_out_fmt[nid] = "ND"
            self._collective_output_devs[nid] = {str(dst)}
            self.avail[src] = max(float(self.avail.get(src, 0.0)), float(end))
            self.avail[dst] = max(float(self.avail.get(dst, 0.0)), float(end))
        return (float(start), float(end))

    def _after_commit_consume_predecessors(self, g: TaskGraph, nid: str) -> None:
        for u in g.predecessors(nid):
            if u not in self._act_refcnt:
                self._act_refcnt[u] = len(g.successors(u))
            self._act_refcnt[u] = max(0, self._act_refcnt[u] - 1)
            udev = self._node_placement.get(u)
            if udev and (udev, u) in self._act_resident and self._act_refcnt[u] == 0:
                bytes_kept = int(self._act_resident.pop((udev, u), 0))
                self.buffer.pim_release_activation(udev, bytes_kept, commit=True)


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

        dst_fmt = self.cost.device_preferred_fmt(dst_dev)
        size_nd = self.cost.format_size(bytes_nd, 'ND')
        host = self.cost.get_host_device()

        def _via_host(commit_flag:bool) -> Tuple[float, float]:
            host_ready = self._ensure_host_store(
                prod_nid, src_dev, bytes_nd, src_fmt, pred_finish, commit_flag)
            l2s, l2e = self.comm.reserve(
                host.name, dst_dev.name, size_nd,
                earliest=host_ready,
                commit=commit_flag,
                tag='act_load',
                extra={
                    'payload': 'activation',
                    'action': 'load',
                    'route': 'host',
                    'prod_node': str(prod_nid),
                    'bytes_nd': int(bytes_nd),
                    'src_fmt': 'ND',
                    'dst_fmt': str(dst_fmt),
                },
            )
            ready = float(l2s) + float(
                self.cost.combine_transfer_and_convert(host, dst_dev, int(size_nd), "ND", str(dst_fmt))
            )
            return (float(l2s), float(ready))
        
        def _direct(commit_flag: bool) -> Tuple[float, float]:
            # Only consider direct transfer if link exists.
            t_direct = float(self.cost.comm_cost(src_dev, dst_dev, int(size_nd)))
            if (not math.isfinite(float(t_direct))) or float(t_direct) <= 0.0:
                return (float("inf"), float("inf"))
            if (src_dev.name, prod_nid) not in self._act_resident:
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
                earliest=earliest,
                commit=commit_flag,
                tag='act_move',
                extra={
                    'payload': 'activation',
                    'action': 'move',
                    'route': 'direct',
                    'prod_node': str(prod_nid),
                    'bytes_nd': int(bytes_nd),
                    'src_fmt': str(src_fmt),
                    'wire_fmt': 'ND',
                    'dst_fmt': str(dst_fmt),
                },
            )
            ready = float(l2s) + float(
                self.cost.combine_transfer_and_convert(src_dev, dst_dev, int(size_nd), "ND", str(dst_fmt))
            )
            return float(l2s), float(ready)

        topo = normalize_topology(getattr(self.cluster, "topology", None))

        if topo == "fc":
            # FC: direct device-to-device transfers are always supported; do not route through host
            s_direct, e_direct = _direct(False)
            if math.isfinite(float(e_direct)):
                return _direct(True) if commit else (float(s_direct), float(e_direct))
            s_host, e_host = _via_host(False)
            return _via_host(True) if commit else (float(s_host), float(e_host))

        s_direct, e_direct = _direct(False)
        s_host, e_host = _via_host(False)

        use_direct = e_direct < e_host
        if commit:
            return _direct(True) if use_direct else _via_host(True)
        else:
            return (s_direct, e_direct) if use_direct else (s_host, e_host)

    def _ready_time_for_device(self, g: TaskGraph, nid: str, dev: DeviceSpec, phase: str, commit: bool) -> float:
        node = g.nodes[nid]
        phase_v = self._node_phase(g, nid, phase)
        batch_v = self._node_batch(g, nid, phase_v)
        seq_v = self._node_seq_len(g, nid, phase_v)
        node_read, _ = self.cost.estimate_activation_bytes(node, batch_v, seq_v, phase_v)

        inbound_start_times: List[float] = []
        inbound_end_times: List[float] = []
        for u in g.predecessors(nid):
            pred_finish = float(self._node_finish_time.get(u, 0.0))
            
            pred_node = g.nodes[u]
            pred_dev_name = self._node_placement.get(u, dev.name)
            phase_u = self._node_phase(g, u, phase_v)
            batch_u = self._node_batch(g, u, phase_u)
            seq_u = self._node_seq_len(g, u, phase_u)
            _, pred_write = self.cost.estimate_activation_bytes(pred_node, batch_u, seq_u, phase_u)
            
            # Collective predecessor (e.g., tensor-parallel all-reduce)
            if u in getattr(self, '_collective_output_devs', {}) and dev.name in self._collective_output_devs[u]:
                inbound_start_times.append(pred_finish)
                inbound_end_times.append(pred_finish)
                continue

            pred_dev = self.cluster.devices[pred_dev_name]
            src_fmt = self._node_out_fmt.get(u, self.cost.device_preferred_fmt(pred_dev))

            if pred_dev.name == dev.name:
                # Same device
                # NOTE: NO transfer unless the producer activation was spilled.
                pred_name_up = str(getattr(g.nodes.get(u), 'name', '')).upper() if hasattr(g, 'nodes') else ''
                if pred_name_up in ('K_WRITE', 'V_WRITE', 'KV_WRITE'):
                    inbound_start_times.append(pred_finish)
                    inbound_end_times.append(pred_finish)
                    continue

                if (pred_dev.name, u) in self._act_resident:
                    inbound_start_times.append(pred_finish)
                    inbound_end_times.append(pred_finish)
                else:
                    # Need host round-trip
                    host_ready = self._ensure_host_store(u, pred_dev, pred_write, src_fmt, pred_finish, commit)
                    size_nd = self.cost.format_size(pred_write, 'ND')
                    host_dev = self.cost.get_host_device()
                    l2s, l2e = self.comm.reserve(
                        host_dev.name,
                        dev.name,
                        size_nd,
                        earliest=host_ready,
                        commit=commit,
                        tag='act_reload',
                        extra={
                            'payload': 'activation',
                            'action': 'reload',
                            'route': 'host',
                            'prod_node': str(u),
                            'cons_node': str(nid),
                            'bytes_nd': int(pred_write),
                            'src_fmt': 'ND',
                            'dst_fmt': str(self.cost.device_preferred_fmt(dev)),
                            'reason': 'evicted',
                        },
                    )
                    inbound_start_times.append(float(l2s))
                    ready = float(l2s) + float(
                        self.cost.combine_transfer_and_convert(host_dev, dev, int(size_nd), "ND", str(self.cost.device_preferred_fmt(dev)),)
                    )
                    inbound_end_times.append(float(ready))
            else:
                # Different devices: choose best path (direct vs via host), also handles format conversion.
                l2s, ready = self._reserve_activation_transfer_best_path(
                    prod_nid=u,
                    src_dev=pred_dev,
                    dst_dev=dev,
                    bytes_nd=pred_write,
                    src_fmt=src_fmt,
                    pred_finish=pred_finish,
                    commit=commit,
                )
                inbound_start_times.append(float(l2s))
                inbound_end_times.append(float(ready))

        # The node can start after all required inbound transfers are done.
        ready_t = float(max(inbound_end_times, default=0.0))
        return ready_t

    def _weight_load_time(self, node: TaskNode, dev: DeviceSpec, earliest: float, commit: bool) -> float:
        wid = self._node_weight_id(node)
        if not wid:
            return 0.0
        wsize_nd = int(self._node_weight_size(node) or 0)
        if wsize_nd <= 0:
            return 0.0

        # skip any host->PIM link load and skip PIM-side weight-loading latency.
        if str(getattr(dev, 'type', '')).lower() == 'pim' and self._weights_preloaded_on_pim():
            if commit:
                try:
                    if wid not in self._weight_sizes:
                        self._weight_sizes[wid] = int(wsize_nd)
                except Exception:
                    pass
            return 0.0

        # --- size-aware cache lookup (ND bytes) ---
        cached_nd = 0
        cache = None
        try:
            cache = getattr(self.buffer, "device_cache", {}).get(dev.name, None)
            items = getattr(cache, "items", None)
            if isinstance(items, dict):
                v = items.get(wid, 0)
                if isinstance(v, (int, float)):
                    cached_nd = int(v)
        except Exception:
            cached_nd = 0

        # Fallback: if we cannot inspect size, rely on old boolean check.
        if cached_nd <= 0:
            try:
                if self.buffer.is_cached(dev.name, wid):
                    if commit and cache is not None:
                        try:
                            cache.touch(wid)
                        except Exception:
                            pass
                    return 0.0
            except Exception:
                pass

        # Full hit
        if cached_nd >= wsize_nd:
            if commit and cache is not None:
                try:
                    cache.touch(wid)
                except Exception:
                    pass
            return 0.0

        # Need to fetch only the missing bytes
        need_nd = int(wsize_nd - max(0, cached_nd))
        if need_nd <= 0:
            return 0.0

        host = self.cost.get_host_device()

        # Host-side stored format (defaults to ND).
        try:
            from_fmt = self.buffer.get_host_fmt(wid) or "ND"
        except Exception:
            from_fmt = "ND"
        to_fmt = self.cost.device_preferred_fmt(dev)

        rd_bytes = int(self.cost.format_size(need_nd, from_fmt))

        # Communication annotation: this is a WEIGHT load (host -> device).
        l2s, l2e = self.comm.reserve(
            host.name,
            dev.name,
            rd_bytes,
            earliest=earliest,
            commit=commit,
            tag='weight_load',
            extra={
                'payload': 'weight',
                'action': 'load',
                'weight_id': str(wid),
                'node_id': str(getattr(node, 'id', getattr(node, 'nid', '')) or ''),
                'op': str(getattr(node, 'name', '') or ''),
                'bytes_nd': int(need_nd),
                'bytes_full_nd': int(wsize_nd),
                'cached_before_nd': int(cached_nd),
                'from_fmt': str(from_fmt),
                'to_fmt': str(to_fmt),
                'cache_capacity_bytes': int(getattr(getattr(self.buffer, 'device_cache', {}).get(dev.name, None), 'capacity', 0) or 0),
            },
        )

        ready = float(l2s) + float(
            self.cost.combine_transfer_and_convert(host, dev, int(rd_bytes), str(from_fmt), str(to_fmt))
        )
        end = max(float(ready), float(earliest))

        if commit:
            try:
                self._weight_load_count[wid, dev.type] += 1
                self._weight_sizes[wid] = int(wsize_nd)
            except Exception:
                pass
            # Mark cached as the *required full size* (not only the delta).
            try:
                self.buffer.mark_cached(dev.name, wid, int(wsize_nd), pinned=False)
            except Exception:
                pass

        if dev.type == "pim":
            load_time = float(self.cost.weight_load_time_pim(int(need_nd)))
            return float(end - float(earliest) + load_time)

        return float(end - float(earliest))

    
    #NOTE: need check if there is any repetition
    def _ensure_host_store(self, u: str, pred_dev: DeviceSpec,bytes_nd: int, src_fmt: str, pred_finish: float, commit: bool) -> float:
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

        # Communication annotation: activation STORE (producer device -> host).
        _, t_link_end = self.comm.reserve(
            pred_dev.name,
            host.name,
            size_nd,
            earliest=earliest,
            commit=commit,
            tag='act_store',
            extra={
                'payload': 'activation',
                'action': 'store',
                'route': 'host',
                'prod_node': str(u),
                'bytes_nd': int(bytes_nd),
                'src_fmt': str(src_fmt),
                'wire_fmt': 'ND',
                'dst_fmt': 'ND',
                'reason': 'evict_or_route',
            },
        )
        t_done = t_link_end
        if commit:
            self._node_host_store_end[u] = t_done
        return t_done

    def makespan(self, schedule: List[ScheduledTask]) -> float:
        return max((t.finish for t in schedule), default=0.0)

class NaiveTopoScheduler(SchedulerBase):
    """A minimal scheduler that walks ready nodes in topo order without uprank."""

    def __init__(self, cluster: Cluster, cost: CostModel, label: PlanLabel, batch: int, seq_len: int, buffer: GlobalMemoryManager):
        super().__init__(cluster, cost, label, batch, seq_len, buffer)

    def _pick_first_allowed_device(self, node: TaskNode) -> DeviceSpec:
        """Return the first available device by simple priority (npu -> pim -> cpu)."""
        kv_in_pim = bool(getattr(self.label, "kv_in_pim", False))
        is_kv_write = node.name.upper() in ("K_WRITE", "V_WRITE", "KV_WRITE")

        # If KV must reside in PIM, force KV writes onto a PIM device to avoid inf schedules.
        if is_kv_write and kv_in_pim:
            # Only pick PIM devices that the node is allowed to run on.
            pim_devs = [d for d in self.cluster.devices_by_type('pim') if self._node_allowed_on(node, d)]
            if not pim_devs:
                raise RuntimeError("kv_in_pim=True but no PIM device available for KV write")
            return pim_devs[0]

        # Executor types depend on whether NPU exists in the cluster.
        for dev_type in self._executor_device_types():
            devs = self.cluster.devices_by_type(dev_type)
            devs = [d for d in devs if self._node_allowed_on(node, d)]
            if devs:
                return devs[0]
        raise RuntimeError(f"No available device for node {getattr(node, 'id', None)}")
    
    def _pick_best_allowed_device(self, g: TaskGraph, node: TaskNode, phase: str) -> DeviceSpec:
        """Pick the device that yields the earliest finish time (EFT) among allowed devices.
        """
        kv_in_pim = bool(getattr(self.label, "kv_in_pim", False))
        is_kv_write = node.name.upper() in ("K_WRITE", "V_WRITE", "KV_WRITE")

        if is_kv_write and kv_in_pim:
            candidates = [d for d in self.cluster.devices_by_type('pim') if self._node_allowed_on(node, d)]
        else:
            candidates = []
            for dev_type in self._executor_device_types():
                candidates.extend(d for d in self.cluster.devices_by_type(dev_type) if self._node_allowed_on(node, d))
        
        if not candidates:
            raise RuntimeError(f"No available device for node {node.id}")

        best_dev: Optional[DeviceSpec] = None
        best_dev = min(candidates, key=lambda d: float(self.avail.get(d.name, 0.0)))
        return best_dev

    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        if getattr(self, 'stats', None):
            self.stats.set_phase(phase)
        self.reset_state(clear_caches=False)
        idx = self._get_graph_index(g)
        topo_pos = {nid: i for i, nid in enumerate(idx.topo)}
        remaining_preds = {nid: len(idx.preds[nid]) for nid in idx.nodes}

        ready: List[str] = [nid for nid in idx.nodes if remaining_preds[nid] == 0]
        ready.sort(key=lambda n: topo_pos.get(n, 0))

        schedule: List[ScheduledTask] = []
        scheduled = set()

        while ready:
            nid = ready.pop(0)
            if nid in scheduled:
                continue
            scheduled.add(nid)
            node = g.nodes[nid]

            # Communication primitives
            if self._is_comm_node(node):
                host = self.cost.get_host_device()
                start, finish = self._earliest_finish_on_device(g, nid, host, self.label, phase, commit=True)
                trace_dev = "COMM"
                schedule.append(ScheduledTask(nid, trace_dev, start, finish))
                self._after_commit_consume_predecessors(g, nid)

                if getattr(self, 'stats', None):
                    op_name = node.attrs.get('op') or node.name
                    try:
                        self.stats.log_op_device(
                            nid=nid, op=op_name,
                            device=trace_dev, device_type='comm',
                            start=float(start), end=float(finish),
                            mode='COMM',
                        )
                    except Exception:
                        pass

                newly_ready: List[str] = []
                for v in idx.succs.get(nid, ()):  # unlock successors
                    remaining_preds[v] -= 1
                    if remaining_preds[v] == 0:
                        newly_ready.append(v)
                newly_ready.sort(key=lambda n: topo_pos.get(n, 0))
                ready.extend(newly_ready)
                continue

            name_up = str(getattr(node, "name", "")).upper()
            is_kv_write = name_up in ("K_WRITE", "V_WRITE", "KV_WRITE")

            dev: DeviceSpec
            if is_kv_write:
                pinned = self._preferred_kv_write_device(g, nid)
                if pinned is not None and self._node_allowed_on(node, pinned):
                    dev = pinned
                else:
                    dev = self._pick_best_allowed_device(g, node, phase)
                    # dev = self._pick_first_allowed_device(node)
            else:
                dev = self._pick_best_allowed_device(g, node, phase)
                # dev = self._pick_first_allowed_device(node)

            start, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=True)
            schedule.append(ScheduledTask(nid, dev.name, start, finish))
            self._after_commit_consume_predecessors(g, nid)

            if getattr(self, 'stats', None):
                op_name = node.attrs.get('op') or node.name
                try:
                    self.stats.log_op_device(
                        nid=nid, op=op_name,
                        device=dev.name, device_type=dev.type,
                        start=float(start), end=float(finish),
                        mode='NAIVE'
                    )
                except Exception:
                    pass

            newly_ready: List[str] = []
            for v in idx.succs.get(nid, ()):  # unlock successors
                remaining_preds[v] -= 1
                if remaining_preds[v] == 0:
                    newly_ready.append(v)
            newly_ready.sort(key=lambda n: topo_pos.get(n, 0))
            ready.extend(newly_ready)

        if len(scheduled) != len(idx.nodes):
            missing = [n for n in idx.nodes if n not in scheduled]
            raise RuntimeError(
                f"Schedule failed: graph may have cycles or missing deps; unscheduled nodes: {missing[:16]}"
            )

        return schedule

class HEFTScheduler(SchedulerBase):
    def __init__(self, cluster: Cluster, cost: CostModel, label: PlanLabel, batch: int, seq_len: int, buffer: GlobalMemoryManager):
        super().__init__(cluster, cost, label, batch, seq_len, buffer)

    def _avg_compute_cost(self, g: TaskGraph, nid: str, phase: str) -> float:
        node = g.nodes[nid]
        phase_eff = self._node_phase(g, nid, phase)
        batch_eff = self._node_batch(g, nid, phase_eff)
        seq_len_eff = self._node_seq_len(g, nid, phase_eff)

        exec_types = set(self._executor_device_types())
        devs = [d for d in self.cluster.devices.values() if str(getattr(d, 'type', '')).lower() in exec_types]
        total_compute = 0.0
        total_w = 0.0
        k = 0
        node_flops = self.cost.estimate_flops(node, batch_eff, seq_len_eff, phase_eff)
        node_weight_size = self._node_weight_size(node)
        wid = self._node_weight_id(node)
        for d in devs:
            if not self._node_allowed_on(node, d):
                continue
            k += 1
            device_compute = self.cost.flop_time(node_flops, d)
            total_compute += device_compute

            if RANKU_INCLUDE_AVG_WEIGHT_LOAD and wid and (node_weight_size > 0):
                if str(getattr(d, 'type', '')).lower() == 'pim' and self._weights_preloaded_on_pim():
                    weight_cost = 0.0
                else:
                    stored_fmt = self.storage_fmt_map.get(wid, 'ND')
                    host = self.cost.get_host_device()
                    size_src = int(self.cost.format_size(int(node_weight_size), str(stored_fmt)))
                    weight_cost = float(
                        self.cost.combine_transfer_and_convert(
                            host,
                            d,
                            int(size_src),
                            str(stored_fmt),
                            str(self.cost.device_preferred_fmt(d)),
                        )
                    )
                    if str(getattr(d, 'type', '')).lower() == 'pim':
                        try:
                            weight_cost += float(self.cost.weight_load_time_pim(int(node_weight_size)))
                        except Exception:
                            pass
                total_w += float(weight_cost)

        avg_compute = total_compute / k if k else 0.0
        avg_w = total_w / k if k and RANKU_INCLUDE_AVG_WEIGHT_LOAD and wid else 0.0
        return avg_compute + avg_w
    
    def _avg_comm_cost(self, g: TaskGraph, u: str, v: str, phase: str) -> float:
        # Barrier edges carry ordering only; treat comm cost as zero.
        node_u = g.nodes[u]
        node_v = g.nodes[v]

        phase_u = self._node_phase(g, u, phase)
        phase_v = self._node_phase(g, v, phase)
        batch_u = self._node_batch(g, u, phase_u)
        batch_v = self._node_batch(g, v, phase_v)
        seq_u = self._node_seq_len(g, u, phase_u)
        seq_v = self._node_seq_len(g, v, phase_v)

        u_read, u_write = self.cost.estimate_activation_bytes(node_u, batch_u, seq_u, phase_u)
        v_read, _ = self.cost.estimate_activation_bytes(node_v, batch_v, seq_v, phase_v)
        payload_bytes = max(u_write, v_read, 16 * 1024)

        exec_types = set(self._executor_device_types())
        devs = [d for d in self.cluster.devices.values() if str(getattr(d, 'type', '')).lower() in exec_types]
        topo = normalize_topology(getattr(self.cluster, "topology", None))
        total = 0.0
        k = 0
        for di in devs:
            for dj in devs:
                if not (self._node_allowed_on(node_u, di) and self._node_allowed_on(node_v, dj)):
                    continue
                src_fmt = str(self.cost.device_preferred_fmt(di))
                dst_fmt = str(self.cost.device_preferred_fmt(dj))
                if di.name == dj.name:
                    total += 0.0
                    k += 1
                    continue

                host = self.cost.get_host_device()
                size_nd = int(self.cost.format_size(int(payload_bytes), 'ND'))

                # Source-side conversion (preferred -> ND) is serialized before transfer.
                t_conv_src = 0.0
                if src_fmt != 'ND':
                    size_src = int(self.cost.format_size(int(payload_bytes), str(src_fmt)))
                    t_conv_src = float(self.cost.format_conversion_time(size_src, str(src_fmt), 'ND', di))

                # Direct: transfer ND and convert ND -> dst_fmt on destination.
                t_direct = float(self.cost.combine_transfer_and_convert(di, dj, int(size_nd), 'ND', str(dst_fmt)))

                # Via host: di -> host (ND) then host -> dj (ND + dst conversion).
                t_to_host = float(self.cost.comm_cost(di, host, int(size_nd)))
                t_from_host = float(self.cost.combine_transfer_and_convert(host, dj, int(size_nd), 'ND', str(dst_fmt)))
                if topo == "fc":
                    t_path = float(t_direct)
                else:
                    t_path = float(min(t_direct, float(t_to_host + t_from_host)))

                total += float(t_conv_src + t_path)
                k += 1
        return total / k if k else 0.0

    def _upward_rank(self, g: TaskGraph, phase: str):
        idx = self._get_graph_index(g)

        succ = idx.succs
        order = idx.rev_topo
        rank_u: Dict[str, float] = {}
        
        for nid in order:
            compute_cost = self._avg_compute_cost(g, nid, phase)
            if not succ[nid]:
                rank_u[nid] = compute_cost
            else:
                best = 0.0
                for v in succ[nid]:
                    comm_cost = self._avg_comm_cost(g, nid, v, phase)
                    path_cost = comm_cost + rank_u[v]
                    if path_cost > best:
                        best = path_cost
                rank_u[nid] = compute_cost + best

        idx.rank_u_by_phase[phase] = rank_u
        sorted_nodes = tuple(sorted(idx.nodes, key=lambda x: -rank_u[x]))
        return list(sorted_nodes)

    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        if getattr(self, 'stats', None):
            self.stats.set_phase(phase)

        self.reset_state(clear_caches=False)
        idx = self._get_graph_index(g)
       
        # Pre-compute uprank scores (HEFT priority) for this phase.
        self._upward_rank(g, phase=phase)  # fills idx.rank_u_by_phase[phase]
        rank_u = idx.rank_u_by_phase.get(phase, {})

        # Ready queue: always pick a *ready* task with the highest uprank.
        topo_pos = {nid: i for i, nid in enumerate(idx.topo)}
        remaining_preds = {nid: len(idx.preds[nid]) for nid in idx.nodes}
        heap: List[Tuple[float, int, str]] = []  # (-uprank, topo_pos, nid)
        for nid in idx.nodes:
            if remaining_preds[nid] == 0:
                heapq.heappush(heap, (-rank_u.get(nid, 0.0), topo_pos.get(nid, 0), nid))

        schedule: List[ScheduledTask] = []
        scheduled = set()
        while heap:
            _, _, nid = heapq.heappop(heap)
            if nid in scheduled:
                continue
            scheduled.add(nid)
            node = g.nodes[nid]

            # Communication primitives: record them on a virtual 'COMM' lane.
            if self._is_comm_node(node):
                host = self.cost.get_host_device()
                start, finish = self._earliest_finish_on_device(g, nid, host, self.label, phase, commit=True)
                trace_dev = "COMM"
                schedule.append(ScheduledTask(nid, trace_dev, start, finish))
                self._after_commit_consume_predecessors(g, nid)

                if getattr(self, 'stats', None):
                    op_name = node.attrs.get('op') or node.name
                    try:
                        self.stats.log_op_device(
                            nid=nid, op=op_name,
                            device=trace_dev, device_type='comm',
                            start=float(start), end=float(finish),
                            mode='COMM',
                        )
                    except Exception:
                        pass

                # Unlock successors whose predecessors are now all scheduled (ready-queue).
                for v in idx.succs.get(nid, ()):
                    remaining_preds[v] -= 1
                    if remaining_preds[v] == 0:
                        heapq.heappush(heap, (-rank_u.get(v, 0.0), topo_pos.get(v, 0), v))
                continue

            kv_in_pim = getattr(self.label, 'kv_in_pim', False)
            is_kv_write = node.name.upper() in ('K_WRITE', 'V_WRITE', 'KV_WRITE')

            exec_types = tuple(self._executor_device_types())

            pinned_dev: Optional[DeviceSpec] = None
            if is_kv_write:
                pinned_dev = self._preferred_kv_write_device(g, nid)
                if pinned_dev is not None and not self._node_allowed_on(node, pinned_dev):
                    pinned_dev = None

            candidates: List[Tuple[str, float, Any]] = []

            if pinned_dev is not None:
                # 只有一个候选：与源 K/V 同设备
                _, finish = self._earliest_finish_on_device(
                    g, nid, pinned_dev, self.label, phase, commit=False
                )
                mode = str(getattr(pinned_dev, "type", "") or "").upper() or "DEV"
                candidates.append((mode, float(finish), pinned_dev))
            else:
                # Pick the best device within each executor type.
                for t in exec_types:
                    best_dev = None
                    best_finish = float('inf')
                    for dev in self.cluster.devices_by_type(t):
                        if not self._node_allowed_on(node, dev):
                            continue
                        _, finish = self._earliest_finish_on_device(
                            g, nid, dev, self.label, phase, commit=False
                        )
                        if float(finish) < float(best_finish):
                            best_finish = float(finish)
                            best_dev = dev
                    if best_dev is not None:
                        candidates.append((str(t).upper(), float(best_finish), best_dev))


            if not candidates:
                raise RuntimeError("No available device for node %s" % nid)

            chosen_mode, chosen_finish, chosen_data = min(candidates, key=lambda x: x[1])
            dev = chosen_data
            # Keep this log compact: candidate lists can be very long.
            logger.debug(
                "[sched] choose node=%s dev=%s mode=%s finish=%.4f (cands=%d)", nid, getattr(dev, 'name', dev), chosen_mode, float(chosen_finish), int(len(candidates)))
            start, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=True)
            schedule.append(ScheduledTask(nid, dev.name, start, finish))
            op_name = node.attrs.get('op') or node.name
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
            # Unlock successors whose predecessors are now all scheduled (ready-queue).
            for v in idx.succs.get(nid, ()):
                remaining_preds[v] -= 1
                if remaining_preds[v] == 0:
                    heapq.heappush(heap, (-rank_u.get(v, 0.0), topo_pos.get(v, 0), v))
        if len(scheduled) != len(idx.nodes):
            missing = [n for n in idx.nodes if n not in scheduled]
            raise RuntimeError(
                f"Schedule failed: graph may have cycles or missing deps; unscheduled nodes: {missing[:16]}"
            )
        return schedule

    def export_weight_stats(self):
        from collections import defaultdict
        by_wid = defaultdict(lambda: defaultdict(int))
        for (wid, dev_type), cnt in self._weight_load_count.items():
            by_wid[wid][dev_type] += cnt
        chain_hits = {}

        try:
            chain_hits = dict(getattr(self, "_weight_chain_hits", {}) or {})
        except Exception:
            chain_hits = {}

        return {
            'weight_sizes': dict(self._weight_sizes),
            'weight_load_counts': {wid: dict(cnts) for wid, cnts in by_wid.items()},
            'storage_fmt_map': dict(self.storage_fmt_map or {}),
            'host_format': dict(self.buffer.host_format or {}),
            'weight_chain_hits': chain_hits,
        }


    def set_storage_format_map(self, fmt_map: Dict[str, str]):
        self.storage_fmt_map = dict(fmt_map or {})
        try:
            self.buffer.host_format.clear()
        except Exception:
            pass
        for k, v in self.storage_fmt_map.items():
            self.buffer.set_host_fmt(str(k), str(v))

    def suggest_weight_storage_formats(self) -> Dict[str, str]:
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
                total = 0.0
                for dev_type, cnt in counts.items():
                    devs = self.cluster.devices_by_type(dev_type)
                    if not devs:
                        continue
                    d = devs[0]
                    host = self.cost.get_host_device()
                    size_src = int(self.cost.format_size(int(w_bytes_nd), str(fmt)))
                    w_cost = float(self.cost.combine_transfer_and_convert( host,d,int(size_src),str(fmt),str(self.cost.device_preferred_fmt(d)),))
                    if str(getattr(d, 'type', '')).lower() == 'pim':
                        try:
                            w_cost += float(self.cost.weight_load_time_pim(int(w_bytes_nd)))
                        except Exception:
                            pass
                    total += float(cnt) * float(w_cost)
                if total + EPS < best_t or (abs(total - best_t) < EPS and fmt == native):
                    best_t, best_fmt = total, fmt
            sugg[wid] = best_fmt

        return sugg

    # ------------------------------------------------------------------
    # Block Coordinate Descent (BCD) weight-format suggestion
    # ------------------------------------------------------------------
    def _strip_layer_prefix(self, wid: str) -> str:
        """Strip leading layer tag from weight_id.

        Examples:
            L12_WQ      -> WQ
            L3_WQ_S0    -> WQ_S0
            L7_E2_W1    -> E2_W1
        """
        if not wid:
            return ""
        try:
            m = re.match(r"^L\d+_(.*)$", str(wid))
            return m.group(1) if m else str(wid)
        except Exception:
            return str(wid)

    def _weight_block_key(self, wid: str, *, mode: str = "coupled") -> str:
        """Return a block key for `wid`.

        mode:
          - 'none'   : only strip layer prefix, no coupling.
          - 'coupled': additionally couple (WQ,WK,WV) as one block, and (W1,W3) as one block,
                       while keeping shard/expert suffixes.
        """
        base = self._strip_layer_prefix(wid)
        if mode in ("none", "strip_only", ""):
            return base

        parts = [p for p in str(base).split("_") if p]
        if not parts:
            return base

        # Common (non-MoE) weights: WQ/WK/WV, WO, W1/W2/W3, possibly with _S{sid}.
        head = parts[0]
        tail = "_".join(parts[1:]) if len(parts) > 1 else ""

        def _join(prefix: str, rest: str) -> str:
            return f"{prefix}_{rest}" if rest else prefix

        if head in ("WQ", "WK", "WV"):
            return _join("ATTN_QKV", tail)
        if head in ("W1", "W3"):
            return _join("FFN_W13", tail)

        # MoE style: E{e}_W1 / E{e}_W3 etc. Keep expert id as part of the key.
        if head.startswith("E") and len(parts) >= 2:
            wname = parts[1]
            rest = "_".join(parts[2:]) if len(parts) > 2 else ""
            if wname in ("W1", "W3"):
                return _join(f"{head}_FFN_W13", rest)
            if wname in ("WQ", "WK", "WV"):
                return _join(f"{head}_ATTN_QKV", rest)

        return base

    def _estimate_weight_host_to_device_cost(
        self,
        wid: str,
        counts: Mapping[str, int],
        fmt: str,
        *,
        lookahead_beta: float = 0.0,
        max_chain_hits: float = 0.0,
        chain_hits: Mapping[str, float] | None = None,
    ) -> float:
        """Aggregate (host->device move + conversion) cost for a weight under a host format.

        This is a *surrogate* objective computed from the last scheduling pass'
        observed cache-miss load counts. It is *exact* if placements & cache-miss
        patterns stay unchanged, and is used to guide format updates between passes.
        """
        w_bytes_nd = int(self._weight_sizes.get(wid, 0) or 0)
        if w_bytes_nd <= 0:
            return 0.0

        # Lookahead weighting: emphasize weights that appear in selected lookahead chains.
        factor = 1.0
        try:
            if lookahead_beta and chain_hits and max_chain_hits and max_chain_hits > 0:
                h = float(chain_hits.get(wid, 0.0) or 0.0)
                factor = float(1.0 + float(lookahead_beta) * (h / float(max_chain_hits)))
        except Exception:
            factor = 1.0

        total = 0.0
        for dev_type, cnt in counts.items():
            if not cnt:
                continue
            devs = self.cluster.devices_by_type(str(dev_type))
            if not devs:
                continue
            d = devs[0]
            host = self.cost.get_host_device()
            size_src = int(self.cost.format_size(int(w_bytes_nd), str(fmt)))
            w_cost = float(self.cost.combine_transfer_and_convert(host, d, int(size_src), str(fmt), str(self.cost.device_preferred_fmt(d)), ))
            if str(getattr(d, 'type', '')).lower() == 'pim':
                try:
                    w_cost += float(self.cost.weight_load_time_pim(int(w_bytes_nd)))
                except Exception:
                    pass
            total += float(cnt) * float(w_cost)
        return float(factor * total)

    def suggest_weight_storage_formats_bcd(
        self,
        current_map: Dict[str, str] | None = None,
        *,
        max_block_changes: int = 1,
        min_gain_ratio: float = 0.005,
        block_mode: str = "coupled",
        candidates: Tuple[str, ...] = ("ND", "NPU_OPT", "PIM_OPT"),
        lookahead_beta: float = 0.25,
    ) -> Dict[str, str]:
        """Suggest next host weight formats via *block* coordinate descent (BCD).

        Returns:
            A new map (wid -> fmt).
        """
        cur = dict(current_map or {})

        # Aggregate cache-miss load counts observed in the last scheduling pass.
        by_wid: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for (wid, dev_type), cnt in self._weight_load_count.items():
            by_wid[str(wid)][str(dev_type)] += int(cnt)

        # Include weights already in the map so blocks stay consistent even if
        # a few weights did not show up in the stats (e.g., sampling).
        for wid in list(cur.keys()):
            by_wid.setdefault(str(wid), defaultdict(int))

        # Lookahead-chain hits (optional, only available in HEFTCOMMAWARE).
        chain_hits = {}
        try:
            chain_hits = dict(getattr(self, "_weight_chain_hits", {}) or {})
        except Exception:
            chain_hits = {}
        max_hits = float(max(chain_hits.values(), default=0.0)) if chain_hits else 0.0

        # Build blocks.
        blocks: Dict[str, List[str]] = defaultdict(list)
        for wid in by_wid.keys():
            key = self._weight_block_key(wid, mode=block_mode)
            blocks[str(key)].append(str(wid))

        # Helper: block-level dominant device type for tie-breaking.
        def _block_native_fmt(wids: List[str]) -> str:
            npu = 0
            pim = 0
            for w in wids:
                c = by_wid.get(w, {})
                npu += int(c.get("npu", 0) or 0)
                pim += int(c.get("pim", 0) or 0)
            if npu > pim:
                return "NPU_OPT"
            if pim > npu:
                return "PIM_OPT"
            return "ND"

        # Evaluate each block's best format and improvement.
        cand = tuple(str(x) for x in candidates if x)
        scored: List[Tuple[float, float, str, str]] = []  # (improve, ratio, block_key, best_fmt)
        eps = 1e-12
        for bkey, wids in blocks.items():
            if not wids:
                continue

            native = _block_native_fmt(wids)

            cur_cost = 0.0
            for w in wids:
                fmt0 = cur.get(w, "ND")
                cur_cost += self._estimate_weight_host_to_device_cost(
                    w, by_wid.get(w, {}), fmt0,
                    lookahead_beta=lookahead_beta, max_chain_hits=max_hits, chain_hits=chain_hits
                )

            best_cost = float("inf")
            best_fmt = cur.get(wids[0], "ND")
            for fmt in cand:
                total = 0.0
                for w in wids:
                    total += self._estimate_weight_host_to_device_cost(
                        w, by_wid.get(w, {}), fmt,
                        lookahead_beta=lookahead_beta, max_chain_hits=max_hits, chain_hits=chain_hits
                    )
                # Tie-break to native.
                if total + eps < best_cost or (abs(total - best_cost) <= eps and fmt == native):
                    best_cost = float(total)
                    best_fmt = str(fmt)

            improve = float(cur_cost - best_cost)
            ratio = float(improve / (cur_cost + eps)) if cur_cost > 0 else 0.0
            if improve > 0.0 and ratio >= float(min_gain_ratio):
                scored.append((improve, ratio, str(bkey), str(best_fmt)))

        if not scored:
            return cur

        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))

        # Apply top-K block updates (BCD step).
        out = dict(cur)
        k = max(1, int(max_block_changes))
        for _, _, bkey, best_fmt in scored[:k]:
            for w in blocks.get(bkey, []):
                out[str(w)] = str(best_fmt)

        return out

class HEFTCOMMAWAREScheduler(HEFTScheduler):
    """Joint-graph scheduler that reuses SchedulerBase timing + buffer/comm modeling."""

    def __init__(
        self, 
        cluster: Cluster, 
        cost: CostModel, 
        label: PlanLabel, 
        batch: int, 
        seq_len: int, 
        buffer: GlobalMemoryManager, 
        *, 
        rand_seed: int | None = None):
        super().__init__(cluster, cost, label, batch, seq_len, buffer)

        # Near-future plan hints used by the lookahead consistency penalty.
        # the oldest hints when exceeding SCHED_JOINT_LK_PLAN_HINT_MAX.
        self._plan_hint: Dict[str, str] = {}
        self._rng = random.Random(rand_seed)
        self._weight_plan_hint: Dict[str, str] = {}
        self._model_total_weight_bytes: Optional[int] = None
        self._weight_chain_hits: Dict[str, float] = defaultdict(float)

    def reset_state(self, *, clear_caches: bool = True) -> None:
        """Reset runtime state + clear COMMAWARE hints."""
        super().reset_state(clear_caches=clear_caches)
        self._plan_hint.clear()
        self._weight_plan_hint.clear()
        self._model_total_weight_bytes = None
        try:
            self._weight_chain_hits.clear()
        except Exception:
            pass
    # -----------------------------
    # Helpers for COMMAWARE HEFT
    # -----------------------------
    def _edge_data_bytes(self, g: TaskGraph, u: str, v: str, phase: str) -> int:
        """Best-effort edge payload estimation.

        Prefer explicit edge metadata if present; otherwise fall back to an
        activation-based heuristic similar to `_avg_comm_cost`.
        """

        node_u = g.nodes[u]
        node_v = g.nodes[v]
        phase_u = self._node_phase(g, u, phase)
        phase_v = self._node_phase(g, v, phase)
        batch_u = self._node_batch(g, u, phase_u)
        batch_v = self._node_batch(g, v, phase_v)
        seq_u = self._node_seq_len(g, u, phase_u)
        seq_v = self._node_seq_len(g, v, phase_v)

        _, u_write = self.cost.estimate_activation_bytes(node_u, batch_u, seq_u, phase_u)
        v_read, _ = self.cost.estimate_activation_bytes(node_v, batch_v, seq_v, phase_v)
        return int(max(u_write, v_read, 16 * 1024))

    def _estimate_transfer_time(self, src: DeviceSpec, dst: DeviceSpec, bytes_nd: int) -> float:
        """Contention-free transfer time estimate used by lookahead & penalties."""
        if src.name == dst.name:
            return 0.0
        size_nd = int(self.cost.format_size(int(bytes_nd), "ND"))

        topo = normalize_topology(getattr(self.cluster, "topology", None))
        # Direct route (FC fabrics always allow direct device-to-device transfers).
        t_direct = float(self.cost.comm_cost(src, dst, size_nd))
        if topo == "fc":
            return float(t_direct)

        host = self.cost.get_host_device()
        # Via-host route (STAR / host-centric fabrics)
        t_via_host = float(self.cost.comm_cost(src, host, size_nd) + self.cost.comm_cost(host, dst, size_nd))
        return float(min(t_via_host, t_direct))

    def _rep_device_by_type(self, dev_type: str) -> Optional[DeviceSpec]:
        devs = self.cluster.devices_by_type(dev_type)
        if not devs:
            return None
        return min(devs, key=lambda d: float(self.avail.get(d.name, 0.0)))

    def _snapshot_cached_weights(self, dev_name: str) -> set[str]:
        """Return a best-effort snapshot of cached weight_ids for a device."""
        cache = getattr(self.buffer, "device_cache", {}).get(dev_name, None)  # type: ignore[attr-defined]
        if cache is not None:
            for attr in ("items", "data", "cache"):
                try:
                    items = getattr(cache, attr, None)
                    if isinstance(items, Mapping):
                        return set(str(k) for k in items.keys())
                except Exception:
                    pass
        # Fallback: unknown cache internals; return empty.
        return set()

    def _estimate_weight_reload_time(self, wid: str, wsize_nd: int, dev: DeviceSpec) -> float:
        """Contention-free estimate of loading (and converting) a weight to `dev`."""
        if not wid or wsize_nd <= 0:
            return 0.0
        # If weights are preloaded on PIM, there is no reload cost for PIM.
        if str(getattr(dev, 'type', '')).lower() == 'pim' and self._weights_preloaded_on_pim():
            return 0.0
        if self.buffer.is_cached(dev.name, wid):
            return 0.0

        from_fmt = self.buffer.get_host_fmt(wid) or "ND"
        to_fmt = self.cost.device_preferred_fmt(dev)
        host = self.cost.get_host_device()
        size_src = int(self.cost.format_size(int(wsize_nd), str(from_fmt)))
        t = float(self.cost.combine_transfer_and_convert(host, dev, size_src, str(from_fmt), str(to_fmt)))
        if str(getattr(dev, "type", "")).lower() == "pim":
            try:
                t += float(self.cost.weight_load_time_pim(int(wsize_nd)))
            except Exception:
                pass
        return float(t)

    def _representative_activation_bytes(self, g: TaskGraph, nid: str, phase: str) -> int:
        """Representative activation footprint for `nid` used in consistency penalty."""
        try:
            node = g.nodes[nid]
            phase_eff = self._node_phase(g, nid, phase)
            batch_eff = self._node_batch(g, nid, phase_eff)
            seq_eff = self._node_seq_len(g, nid, phase_eff)
            r, w = self.cost.estimate_activation_bytes(node, batch_eff, seq_eff, phase_eff)
            return int(max(r, w, 16 * 1024))
        except Exception:
            return int(16 * 1024)

    def _build_lookahead_chain(
        self,
        g: TaskGraph,
        start: str,
        idx: _GraphIndex,
        rank_u: Mapping[str, float],
        phase: str,
        H: int,
    ) -> List[str]:
        """Heaviest-edge successor chain with a small rank_u tie-breaker."""
        H = max(1, int(H))
        chain: List[str] = [start]
        eps = 1e-6
        cur = start
        while len(chain) < H:
            succs = idx.succs.get(cur, ())
            if not succs:
                break
            best_x = None
            best_val = -float("inf")
            for x in succs:
                try:
                    d = float(self._edge_data_bytes(g, cur, x, phase))
                except Exception:
                    d = 0.0
                val = d + eps * float(rank_u.get(x, 0.0))
                if val > best_val:
                    best_val = val
                    best_x = x
            if best_x is None or best_x in chain:
                break
            chain.append(best_x)
            cur = best_x
        return chain

    def _simulate_chain_finish(
        self,
        g: TaskGraph,
        chain: List[str],
        devs: List[DeviceSpec],
        first_finish: float,
        phase: str,
        *,
        apply_consistency_penalty: bool,
    ) -> Tuple[float, Dict[str, str]]:
        """Simulate the completion time of `chain` under a fixed device assignment.

        This is a *lightweight*, contention-free forward estimate used by the lookahead
        window. It deliberately ignores non-chain dependencies to keep it cheap.

        Returns:
            (estimated_finish_time_including_penalty, assignment_map_for_future_nodes)
        """
        if not chain:
            return float(first_finish), {}
        if len(chain) != len(devs):
            raise ValueError("chain/devs length mismatch")

        # Shadow device availability.
        avail_shadow: Dict[str, float] = dict(self.avail)
        # Shadow weight cache (best effort).
        cached_shadow: Dict[str, set[str]] = {}
        for d in {d.name for d in devs}:
            cached_shadow[d] = self._snapshot_cached_weights(d)

        # First node: finish time is taken from EFT(v,k) estimation; assume its weight
        # becomes cached on its device if it uses weights.
        v0 = chain[0]
        d0 = devs[0]
        finish = float(first_finish)
        avail_shadow[d0.name] = max(float(avail_shadow.get(d0.name, 0.0)), finish)
        try:
            n0 = g.nodes[v0]
            wid0 = self._node_weight_id(n0)
            if wid0:
                cached_shadow.setdefault(d0.name, set()).add(wid0)
        except Exception:
            pass

        # Assignment map for future nodes in this chain.
        assign_map: Dict[str, str] = {}
        for nid, d in zip(chain[1:], devs[1:]):
            assign_map[nid] = d.name

        # Sequential simulation along the chain.
        prev_nid = v0
        prev_dev = d0
        for nid, dev in zip(chain[1:], devs[1:]):
            bytes_edge = int(self._edge_data_bytes(g, prev_nid, nid, phase))
            comm_t = float(self._estimate_transfer_time(prev_dev, dev, bytes_edge))

            ready_t = float(finish + comm_t)
            start_t = float(max(float(avail_shadow.get(dev.name, 0.0)), ready_t))

            node = g.nodes[nid]
            phase_eff = self._node_phase(g, nid, phase)
            batch_eff = self._node_batch(g, nid, phase_eff)
            seq_eff = self._node_seq_len(g, nid, phase_eff)
            compute_t = float(self.cost.node_device_cost(node, dev, self.label, batch_eff, seq_eff, phase_eff))

            wid = self._node_weight_id(node)
            wsize = int(self._node_weight_size(node))
            wload_t = 0.0
            if wid and wsize > 0:
                dev_cached = cached_shadow.setdefault(dev.name, set())
                if wid not in dev_cached:
                    wload_t = float(self._estimate_weight_reload_time(wid, wsize, dev))
                    dev_cached.add(wid)

            finish = float(start_t + wload_t + compute_t)
            avail_shadow[dev.name] = finish
            prev_nid, prev_dev = nid, dev

        # Consistency penalty: discourage oscillation in overlapping windows.
        penalty = 0.0
        if apply_consistency_penalty and self._plan_hint and len(chain) > 1:
            lam = float(SCHED_JOINT_LK_CONSIST_LAMBDA)
            if lam > 0:
                for nid, dev in zip(chain[1:], devs[1:]):
                    hinted = self._plan_hint.get(nid)
                    if hinted and hinted != dev.name:
                        src = self.cluster.devices.get(hinted)
                        if src is None:
                            continue
                        bytes_act = int(self._representative_activation_bytes(g, nid, phase))
                        penalty += lam * float(self._estimate_transfer_time(src, dev, bytes_act))

        return float(finish + penalty), assign_map

    def _lookahead_window_estimate(
        self,
        g: TaskGraph,
        chain: List[str],
        first_dev: DeviceSpec,
        first_eft: float,
        phase: str,
    ) -> Tuple[float, Dict[str, str]]:
        """Enumerate a small number of device-type patterns to estimate window completion."""
        if len(chain) <= 1:
            return float(first_eft), {}

        rep_npu = self._rep_device_by_type("npu")
        rep_pim = self._rep_device_by_type("pim")

        # Representative pool used for *future* nodes.
        rep_by_type: Dict[str, Optional[DeviceSpec]] = {"npu": rep_npu, "pim": rep_pim}

        # Collect per-node allowed type options for the (h-1) future nodes.
        type_options: List[List[str]] = []
        for nid in chain[1:]:
            node = g.nodes[nid]
            opts: List[str] = []
            for t in ("npu", "pim"):
                rep = rep_by_type.get(t)
                if rep is None:
                    continue
                try:
                    if self._node_allowed_on(node, rep):
                        opts.append(t)
                except Exception:
                    # If allowed() check fails, be conservative and allow.
                    opts.append(t)
            if not opts:
                # No representative device can run this node; fall back to no-lookahead.
                return float(first_eft), {}
            type_options.append(opts)

        best_finish = float("inf")
        best_assign: Dict[str, str] = {}

        # Enumerate a constant number of patterns (<= 2^(H-1)).
        for types_combo in itertools.product(*type_options):
            devs: List[DeviceSpec] = [first_dev]

            valid = True
            for t in types_combo:
                rep = rep_by_type.get(t)
                if rep is None:
                    valid = False
                    break
                devs.append(rep)
            if not valid:
                continue

            finish, assign = self._simulate_chain_finish(
                g,
                chain=chain,
                devs=devs,
                first_finish=float(first_eft),
                phase=phase,
                apply_consistency_penalty=True,
            )
            if finish < best_finish:
                best_finish = float(finish)
                best_assign = dict(assign)

        if not math.isfinite(best_finish):
            return float(first_eft), {}
        return float(best_finish), best_assign




    def _interval_nodes_for_weight_bias(self, idx: _GraphIndex, g: TaskGraph, nid: str) -> List[str]:
        """Approximate nodes in interval I(v) for weight-bias.

        - 对 decode 的“跨 token 权重复用”，v 到下一次 v 的使用之间会遍历一整轮模型权重；
          因而 interval 的 working-set 近似为整张图的权重集合。

        - 直接返回 topo 全部节点。

        该函数保留是为了兼容旧接口；新 bias 逻辑不会再依赖 repeat/meta。
        """
        try:
            return list(idx.topo)
        except Exception:
            try:
                return list(idx.nodes)
            except Exception:
                return [nid]

    def _pim_weight_cache_remaining_bytes(self, dev_name: str) -> int:
        cache = getattr(self.buffer, "device_cache", {}).get(dev_name, None)  # type: ignore[attr-defined]
        if cache is None:
            return 0
        cap = None
        used = None
        for a in ("capacity", "cap", "max_bytes", "limit"):
            try:
                v = getattr(cache, a, None)
                if isinstance(v, (int, float)):
                    cap = int(v)
                    break
            except Exception:
                continue
        try:
            v = getattr(cache, "used", None)
            if isinstance(v, (int, float)):
                used = int(v)
        except Exception:
            used = None
        if cap is None:
            return 0
        if used is None:
            used = 0
        return max(0, int(cap - used))

    def _weight_reuse_bias(self, g: TaskGraph, idx: _GraphIndex, nid: str, dev: DeviceSpec, phase: str) -> float:
        """Weight-bias w/o node.meta: use label-level total-weight estimate.

        - 若节点有 weight：从本 token 的 decode 使用该 weight 到下一次使用，
          近似等价于把“整模权重”完整走一遍。
        - 因而只要某设备的可用权重空间 >= 全部权重，就认为该 weight 在下一次仍可复用；
          在 prefill 阶段给该设备一个 gain(bias)。
        - 同时维护 weight_id 级别的 hint：下一次遇到相同 weight_id，
          若候选 device_type 与 hint 不一致，则施加惩罚。

        返回值：加入到 score 里的 bias（负数=鼓励，正数=惩罚）。
        """

        node = g.nodes[nid]
        wid = self._node_weight_id(node)
        wsize = int(self._node_weight_size(node))
        if not wid or wsize <= 0:
            return 0.0

        # ---- 1) penalty if deviating from stored hint (跨 token/重复算子一致性) ----
        penalty = 0.0
        hinted_type = self._weight_plan_hint.get(wid)
        if hinted_type is not None and str(hinted_type) != str(getattr(dev, "type", "")):
            lam = float(SCHED_JOINT_LK_CONSIST_LAMBDA) if SCHED_JOINT_LK_CONSIST_LAMBDA is not None else 1.0
            t_reload = float(self._estimate_weight_reload_time(wid, wsize, dev))
            penalty = float(lam * t_reload)

        # ---- 2) gain in prefill if device can keep whole-model weights ----
        gain = 0.0
        phase_l = str(phase or "").lower()
        if phase_l == "prefill":
            total_w = int(self._get_total_model_weight_bytes(g))
            if total_w > 0 and self._device_can_hold_all_weights(dev, total_w):
                eta = float(SCHED_WEIGHT_BIAS_ETA) if SCHED_WEIGHT_BIAS_ETA is not None else 1.0
                t_reload = float(self._estimate_weight_reload_time(wid, wsize, dev))
                gain = float(-eta * t_reload)

        return float(penalty + gain)

    def _get_total_model_weight_bytes(self, g: TaskGraph) -> int:
        """Return total model weight bytes (ND) from label if possible, else from graph."""
        if isinstance(self._model_total_weight_bytes, int) and self._model_total_weight_bytes > 0:
            return int(self._model_total_weight_bytes)

        # 1) Prefer label-provided totals (main：plan label).
        for attr in (
            "total_weight_bytes",
            "model_weight_total_bytes",
            "model_weight_bytes",
            "weights_total_bytes",
            "weight_total_bytes",
            "fc_total_bytes",
            "fc_weight_bytes",
        ):
            try:
                v = getattr(self.label, attr, None)
                if isinstance(v, (int, float)) and float(v) > 0:
                    self._model_total_weight_bytes = int(v)
                    return int(self._model_total_weight_bytes)
            except Exception:
                continue

        # 2) Fallback: sum unique weight_id sizes from graph nodes.
        seen: set[str] = set()
        total = 0
        try:
            for n in g.nodes.values():
                wid = self._node_weight_id(n)
                if not wid or wid in seen:
                    continue
                w = int(self._node_weight_size(n))
                if w <= 0:
                    continue
                seen.add(wid)
                total += w
        except Exception:
            total = 0

        self._model_total_weight_bytes = int(total)
        return int(self._model_total_weight_bytes)

    def _device_weight_capacity_bytes(self, dev: DeviceSpec) -> int:
        """Best-effort estimate of a device's available space for *weights* (bytes)."""
        # Prefer buffer cache capacity if available (尤其是 PIM 的 weight cache 约束).
        try:
            cache = getattr(self.buffer, "device_cache", {}).get(dev.name, None)  # type: ignore[attr-defined]
        except Exception:
            cache = None
        if cache is not None:
            for a in ("capacity", "cap", "max_bytes", "limit"):
                try:
                    v = getattr(cache, a, None)
                    if isinstance(v, (int, float)) and float(v) > 0:
                        return int(v)
                except Exception:
                    continue

        # Label-level budget (常见：pim_weight_capacity_bytes).
        if str(getattr(dev, "type", "")) == "pim":
            for a in ("pim_weight_capacity_bytes", "pim_weight_budget_bytes", "pim_static_weight_bytes"):
                try:
                    v = getattr(self.label, a, None)
                    if isinstance(v, (int, float)) and float(v) > 0:
                        return int(v)
                except Exception:
                    continue
        if str(getattr(dev, "type", "")) == "npu":
            for a in ("npu_weight_capacity_bytes", "npu_weight_budget_bytes", "npu_static_weight_bytes"):
                try:
                    v = getattr(self.label, a, None)
                    if isinstance(v, (int, float)) and float(v) > 0:
                        return int(v)
                except Exception:
                    continue

        # Fallback: use physical memory as an upper bound.
        try:
            phy = int(float(getattr(dev, "mem_capacity_GB", 0.0)) * 1024**3)
            return max(0, phy)
        except Exception:
            return 0

    def _device_can_hold_all_weights(self, dev, total_weight_bytes):
        cap = self._device_weight_capacity_bytes(dev)
        if dev.type == "pim":
            pim_cnt = max(1, len(self.cluster.devices_by_type("pim")))
            return cap >= math.ceil(total_weight_bytes / pim_cnt)
        return cap >= total_weight_bytes


    def _update_plan_hints(self, assignments: Mapping[str, str], scheduled: set[str]) -> None:
        """Update bounded plan-hint mapping σ'(·) for near-future nodes."""
        max_keep = int(SCHED_JOINT_LK_PLAN_HINT_MAX) if SCHED_JOINT_LK_PLAN_HINT_MAX is not None else 0
        if max_keep <= 0:
            return

        # Drop scheduled entries first.
        for n in list(self._plan_hint.keys()):
            if n in scheduled:
                self._plan_hint.pop(n, None)

        for nid, dev_name in assignments.items():
            if nid in scheduled:
                continue
            # Refresh insertion order.
            if nid in self._plan_hint:
                self._plan_hint.pop(nid, None)
            self._plan_hint[nid] = str(dev_name)

        while len(self._plan_hint) > max_keep:
            # pop oldest
            oldest = next(iter(self._plan_hint))
            self._plan_hint.pop(oldest, None)

    def _update_weight_hint_after_commit(self, g: TaskGraph, nid: str, hinted_type: str) -> None:
        """
        Record: weight_id -> last chosen device type.
        """
        try:
            node = g.nodes[nid]
        except Exception:
            return
        wid = self._node_weight_id(node)
        wsize = int(self._node_weight_size(node))
        if not wid or wsize <= 0:
            return
        self._weight_plan_hint[str(wid)] = str(hinted_type).lower()
    # -----------------------------
    # Scheduling
    # -----------------------------
    @override
    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        if getattr(self, "stats", None):
            self.stats.set_phase(phase)

        self.reset_state(clear_caches=False)
        idx = self._get_graph_index(g)

        # Step 1: HEFT priority (upward ranks).
        self._upward_rank(g, phase=phase)
        rank_u = idx.rank_u_by_phase.get(phase, {})

        topo_pos = {nid: i for i, nid in enumerate(idx.topo)}
        remaining_preds = {nid: len(idx.preds[nid]) for nid in idx.nodes}

        # Ready set.
        ready: set[str] = {nid for nid in idx.nodes if remaining_preds[nid] == 0}

        schedule: List[ScheduledTask] = []
        scheduled: set[str] = set()

        # Config knobs (match paper notations).
        H = int(SCHED_JOINT_LK_H)
        gamma = float(SCHED_JOINT_LK_GAMMA)
        use_lookahead = bool(SCHED_JOINT_LK_ENABLE) and H > 1 and gamma > 0.0

        # Multi-device selection: when there are multiple executors, don't rely on rank ordering.
        # Instead, pick from the whole READY set by the lookahead chain score.
        exec_types = tuple(self._executor_device_types())
        num_exec = int(sum(len(self.cluster.devices_by_type(t)) for t in exec_types))
        use_chain_select = bool(use_lookahead) and num_exec > 1

        def _candidates_for(nid: str, node: TaskNode) -> Tuple[List[DeviceSpec], bool]:
            """Return (device_candidates, is_kv_write)."""

            # Communication primitives: always schedule on host (control/comm-only op).
            if self._is_comm_node(node):
                return [self.cost.get_host_device()], False

            name_up = str(getattr(node, "name", "")).upper()
            is_kv_write = name_up in ("K_WRITE", "V_WRITE", "KV_WRITE")

            pinned_dev: Optional[DeviceSpec] = None
            if is_kv_write:
                pinned_dev = self._preferred_kv_write_device(g, nid)
                if pinned_dev is not None and (not self._node_allowed_on(node, pinned_dev)):
                    pinned_dev = None

            if pinned_dev is not None:
                return [pinned_dev], is_kv_write

            cands: List[DeviceSpec] = []
            for dev_type in exec_types:
                for dev in self.cluster.devices_by_type(dev_type):
                    try:
                        if self._node_allowed_on(node, dev):
                            cands.append(dev)
                    except Exception:
                        cands.append(dev)
            return cands, is_kv_write

        def _best_assignment_for(nid: str) -> Tuple[float, float, str, Optional[DeviceSpec], Optional[dict], Dict[str, str]]:
            """Return (best_score, best_eft, best_mode, best_dev, best_hy_est, best_hint_assign)."""
            node = g.nodes[nid]
            cands, is_kv_write = _candidates_for(nid, node)
            if not cands:
                return (float("inf"), float("inf"), "DEV", None, None, {})

            # Lookahead chain for this ready node.
            chain = [nid]
            if use_lookahead:
                chain = self._build_lookahead_chain(g, nid, idx, rank_u, phase=phase, H=H)

            allow_npu = any(d.type == "npu" for d in cands)
            allow_pim = any(d.type == "pim" for d in cands)

            hy_est: Optional[dict] = None
            best_score = float("inf")
            best_eft = float("inf")
            best_mode: str = "DEV"
            best_dev: Optional[DeviceSpec] = None
            best_hy: Optional[dict] = None
            best_hint: Dict[str, str] = {}

            for dev in cands:
                _, eft = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=False)
                eft = float(eft)

                # Strategy 1: lookahead window estimate with consistency penalty.
                window_est = eft
                hint_assign: Dict[str, str] = {}
                if use_lookahead and len(chain) > 1:
                    window_est, hint_assign = self._lookahead_window_estimate(
                        g, chain=chain, first_dev=dev, first_eft=eft, phase=phase
                    )

                # Strategy 2: weight-reuse bias (PIM only).
                bias = float(self._weight_reuse_bias(g, idx, nid, dev, phase=phase))

                score = float((1.0 - gamma) * eft + gamma * float(window_est) + bias)

                # Tie-breakers: prefer lower EFT, then random.
                if (score < best_score) or (
                    abs(score - best_score) < 1e-9
                    and (eft < best_eft or (abs(eft - best_eft) < 1e-9 and self._rng.random() < 0.5))
                ):
                    best_score = score
                    best_eft = eft
                    best_mode = "DEV"
                    best_dev = dev
                    best_hy = None
                    best_hint = dict(hint_assign)

            return best_score, best_eft, best_mode, best_dev, best_hy, best_hint

        while ready:
            # Pick a ready node.
            if use_chain_select:
                best_tuple = (float("inf"), float("inf"), float("inf"), float("inf"))
                pick: Optional[str] = None
                pick_res: Optional[Tuple[float, float, str, Optional[DeviceSpec], Optional[dict], Dict[str, str]]] = None
                for nid in list(ready):
                    score, eft, mode, dev_obj, hy_est, hint = _best_assignment_for(nid)
                    # Tie-break: prefer higher rank_u when scores equal.
                    key = (float(score), float(eft), -float(rank_u.get(nid, 0.0)), float(topo_pos.get(nid, 0)))
                    if key < best_tuple:
                        best_tuple = key
                        pick = nid
                        pick_res = (score, eft, mode, dev_obj, hy_est, hint)
                if pick is None or pick_res is None:
                    raise RuntimeError("No schedulable ready node (all placements infeasible)")
                nid = pick
                _, _, best_mode, best_choice, best_hy, best_hint_assignments = pick_res
            else:
                # Classic HEFT: pick by rank_u.
                nid = max(ready, key=lambda n: (float(rank_u.get(n, 0.0)), -int(topo_pos.get(n, 0))))
                node = g.nodes[nid]
                _, _, best_mode, best_choice, best_hy, best_hint_assignments = _best_assignment_for(nid)

            ready.discard(nid)
            if nid in scheduled:
                continue

            node = g.nodes[nid]

            # Record lookahead-chain weight importance for this *selected* node only.
            if use_lookahead:
                try:
                    chain_sel = self._build_lookahead_chain(g, nid, idx, rank_u, phase=phase, H=H)
                    for cnid in chain_sel:
                        try:
                            cw = self._node_weight_id(g.nodes[cnid])
                            if cw:
                                self._weight_chain_hits[str(cw)] += 1.0
                        except Exception:
                            continue
                except Exception:
                    pass
            # Commit
            scheduled.add(nid)

            if best_choice is None:
                raise RuntimeError(f"No feasible placement found for node {nid}")
            dev = best_choice
            start, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=True)
            is_comm = self._is_comm_node(node)
            trace_dev = "COMM" if is_comm else dev.name
            trace_dev_type = "comm" if is_comm else dev.type
            schedule.append(ScheduledTask(nid, trace_dev, float(start), float(finish)))
            self._after_commit_consume_predecessors(g, nid)

            # 记录 weight-level hint
            self._update_weight_hint_after_commit(g, nid, str(getattr(dev, "type", "")))

            # Stats
            if getattr(self, "stats", None):
                op_name = getattr(node, "attrs", {}).get("op") if hasattr(node, "attrs") else None
                op_name = op_name or getattr(node, "name", "")
                try:
                    self.stats.log_op_device(
                        nid=nid,
                        op=op_name,
                        device=trace_dev,
                        device_type=trace_dev_type,
                        start=float(start),
                        end=float(finish),
                        mode=("COMM" if is_comm else "COMMAWARE"),
                    )
                except Exception:
                    pass

            # Update near-future plan hints from best lookahead pattern.
            if use_lookahead and best_hint_assignments:
                self._update_plan_hints(best_hint_assignments, scheduled)
            else:
                self._update_plan_hints({}, scheduled)

            # Unlock successors.
            for v in idx.succs.get(nid, ()):  # type: ignore[assignment]
                remaining_preds[v] -= 1
                if remaining_preds[v] == 0 and v not in scheduled:
                    ready.add(v)

        if len(scheduled) != len(idx.nodes):
            missing = [n for n in idx.nodes if n not in scheduled]
            raise RuntimeError(
                f"Schedule failed: graph may have cycles or missing deps; unscheduled nodes: {missing[:16]}"
            )
        return schedule

