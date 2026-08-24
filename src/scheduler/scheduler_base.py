"""Common scheduler functionality shared by concrete strategy classes."""

from __future__ import annotations

from .scheduler_common import *
from .scheduler_types import _GraphIndex, ScheduledTask
from .scheduler_comm import CommManager


class SchedulerBaseCoreMixin:
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
        self._weight_proto_node: Dict[str, TaskNode] = {}
        self._pim_weight_desc_cache: Dict[Tuple[str, str], Any] = {}
        self._last_op_trace_extra: Dict[str, Any] = {}
        # Side-channel populated by commit=False candidate evaluations for
        # existing scheduler diagnostics.  Static-prior export never reads it.
        self._last_candidate_component_extra: Dict[str, Any] = {}

        # Optional Step-2 export capture.  This state is observational only:
        # the complete snapshot is published after a successful schedule.
        self._hetinfer_prior_capture_enabled: bool = False
        self._hetinfer_prior_snapshots: List[Dict[str, Any]] = []
        self._hetinfer_schedule_call_index: int = 0

    def set_seq_len(self, seq_len: int) -> None:
        self.seq_len = int(seq_len)
    
    def set_storage_format_map(self, fmt_map: Dict[str, str]) -> None:
        """Install host-side weight storage formats for the next simulation run.

        This is scheduler-agnostic configuration, so it belongs on the common
        base class rather than only on HEFTScheduler. That keeps NaiveTopo and
        any other scheduler variants from silently falling back to ND.
        """
        self.storage_fmt_map = {}
        try:
            self.buffer.host_format.clear()
        except Exception:
            pass
        for k, v in dict(fmt_map or {}).items():
            try:
                canon_v = str(self.cost.weight_storage_format(v))
            except Exception:
                try:
                    canon_v = str(_sched_normalize_weight_format_token(v, allow_compute=False))
                except Exception:
                    canon_v = str(v)
            self.storage_fmt_map[str(k)] = str(canon_v)
            self.buffer.set_host_fmt(str(k), str(canon_v))

    def _log_scheduled_op_trace(self, **kwargs: Any) -> None:
        if not getattr(self, 'stats', None):
            return
        self.stats.log_op_device(**kwargs)


    def export_fixed_plan(self, schedule: List["ScheduledTask"]) -> Dict[str, Any]:

        node_order: List[str] = []
        device_by_node: Dict[str, str] = {}
        for t in list(schedule or []):
            try:
                nid = str(getattr(t, "node_id"))
            except Exception:
                continue
            if not nid:
                continue
            node_order.append(nid)
            try:
                dname = str(getattr(t, "device", "") or "")
            except Exception:
                dname = ""
            if dname in self.cluster.devices:
                device_by_node[nid] = dname
        return {
            "order": tuple(node_order),
            "device_by_node": dict(device_by_node),
        }

    def enable_hetinfer_prior_capture(self, enabled: bool = True) -> None:
        """Enable a post-placement, clean static-prior snapshot."""

        self._hetinfer_prior_capture_enabled = bool(enabled)

    def clear_hetinfer_prior_snapshots(self) -> None:
        self._hetinfer_prior_snapshots.clear()

    def export_hetinfer_prior_snapshots(self) -> List[Dict[str, Any]]:
        """Return isolated completed snapshots; partial schedules are absent."""

        return copy.deepcopy(self._hetinfer_prior_snapshots)

    def schedule_with_plan(
        self,
        g: TaskGraph,
        phase: str,
        plan: Optional[Mapping[str, Any]] = None,
    ) -> List["ScheduledTask"]:

        if getattr(self, "stats", None):
            self.stats.set_phase(phase)

        self.reset_state(clear_caches=False)
        # Artifact instance ids count only successfully published snapshots.
        # Disabled capture and failed schedules therefore cannot perturb the
        # ids used by the Het-Infer artifact export.
        capture_call_index = int(len(self._hetinfer_prior_snapshots) + 1)
        idx = self._get_graph_index(g)
        remaining_preds = {nid: len(idx.preds[nid]) for nid in idx.nodes}

        if not isinstance(plan, Mapping):
            raise RuntimeError("schedule_with_plan requires a non-empty fixed plan mapping")
        plan_map = dict(plan)

        raw_order = plan_map.get("order", None)
        if raw_order is None or isinstance(raw_order, (str, bytes)):
            raise RuntimeError("Fixed plan missing iterable 'order'")
        try:
            order_iter = tuple(str(x) for x in raw_order)
        except Exception as e:
            raise RuntimeError("Fixed plan 'order' is not iterable") from e
        if not order_iter:
            raise RuntimeError("Fixed plan has empty 'order'")
        if len(set(order_iter)) != len(order_iter):
            raise RuntimeError("Fixed plan 'order' contains duplicate node ids")
        unknown_in_order = [nid for nid in order_iter if nid not in idx.nodes_set]
        if unknown_in_order:
            raise RuntimeError(
                f"Fixed plan order contains unknown nodes for this graph: {unknown_in_order[:16]}"
            )
        missing_in_order = [nid for nid in idx.nodes if nid not in set(order_iter)]
        if missing_in_order:
            raise RuntimeError(
                f"Fixed plan order does not cover all graph nodes: missing {missing_in_order[:16]}"
            )

        raw_dev_map = plan_map.get("device_by_node", None)
        if raw_dev_map is None:
            raise RuntimeError("Fixed plan missing mapping 'device_by_node'")
        if not isinstance(raw_dev_map, Mapping):
            raise RuntimeError("Fixed plan field 'device_by_node' must be a mapping")
        device_by_node: Dict[str, str] = {}
        for k, v in raw_dev_map.items():
            ks = str(k)
            vs = str(v)
            if not ks or not vs:
                raise RuntimeError("Fixed plan 'device_by_node' contains empty key/value")
            device_by_node[ks] = vs

        missing_device_nodes: List[str] = []
        for nid in idx.nodes:
            node = g.nodes[nid]
            if self._is_comm_node(node):
                    continue
            if nid not in device_by_node:
                missing_device_nodes.append(str(nid))
        if missing_device_nodes:
            raise RuntimeError(
                f"Fixed plan is missing concrete devices for non-comm nodes: {missing_device_nodes[:16]}"
            )

        schedule: List[ScheduledTask] = []
        scheduled: set[str] = set()

        for order_idx, nid in enumerate(order_iter):
            if nid in scheduled:
                raise RuntimeError(f"Fixed plan order repeats node {nid}")
            if nid not in idx.nodes_set:
                raise RuntimeError(f"Fixed plan referenced unknown node {nid}")
            if remaining_preds[nid] != 0:
                blocked_by = [p for p in idx.preds.get(nid, ()) if p not in scheduled]
                raise RuntimeError(
                    f"Fixed plan order is not executable at position {order_idx}: node {nid} still waits for predecessors {blocked_by[:16]}"
                )

            node = g.nodes[nid]

            if self._is_comm_node(node):
                host = self.cost.get_host_device()
                start, finish = self._earliest_finish_on_device(
                    g, nid, host, self.label, phase, commit=True
                )
                schedule.append(ScheduledTask(nid, "COMM", float(start), float(finish)))
                self._after_commit_consume_predecessors(g, nid)

                if getattr(self, "stats", None):
                    op_name = node.attrs.get("op") or node.name
                    try:
                        self._log_scheduled_op_trace(
                            nid=nid,
                            op=op_name,
                            device="COMM",
                            device_type="comm",
                            start=float(start),
                            end=float(finish),
                            mode="FIXED_COMM",
                        )
                    except Exception:
                        pass
            else:
                dname = device_by_node.get(str(nid), "")
                if not dname:
                    raise RuntimeError(f"Fixed plan has no device assignment for node {nid}")
                dev = self.cluster.devices.get(str(dname))
                if dev is None:
                    raise RuntimeError(
                        f"Fixed plan assigned unknown device '{dname}' to node {nid}"
                    )

                pinned = self._preferred_kv_write_device(g, nid)
                if pinned is not None and str(getattr(pinned, "name", "")) != str(getattr(dev, "name", "")):
                    raise RuntimeError(
                        f"Fixed plan device mismatch for KV-pinned node {nid}: plan={dev.name}, required={pinned.name}"
                    )
                if not self._node_allowed_on(node, dev):
                    raise RuntimeError(
                        f"Fixed plan assigned illegal device '{dev.name}' to node {nid}"
                    )

                start, finish = self._earliest_finish_on_device(
                    g, nid, dev, self.label, phase, commit=True
                )
                schedule.append(ScheduledTask(nid, dev.name, float(start), float(finish)))
                self._after_commit_consume_predecessors(g, nid)

                if getattr(self, "stats", None):
                    op_name = node.attrs.get("op") or node.name
                    try:
                        self._log_scheduled_op_trace(
                            nid=nid,
                            op=op_name,
                            device=dev.name,
                            device_type=dev.type,
                            start=float(start),
                            end=float(finish),
                            mode="FIXED_PLAN",
                        )
                    except Exception:
                        pass

            scheduled.add(nid)
            for v in idx.succs.get(nid, ()):
                remaining_preds[v] -= 1

        if len(scheduled) != len(idx.nodes):
            missing = [n for n in idx.nodes if n not in scheduled]
            raise RuntimeError(
                f"Schedule failed: graph may have cycles or missing deps; unscheduled nodes: {missing[:16]}"
            )

        self._finalize_hetinfer_prior_snapshot(
            g,
            phase,
            schedule_call_index=capture_call_index,
        )

        return schedule

from .scheduler_common import *
from .scheduler_types import _GraphIndex, ScheduledTask
from .scheduler_comm import CommManager


class SchedulerBaseHelperMixin:
    def _hetinfer_legal_devices(
        self, g: Any, nid: str, node: TaskNode
    ) -> Tuple[DeviceSpec, ...]:
        """Return the capability/legal candidate set without timing filters."""

        if self._is_comm_node(node):
            # Communication primitives are not ordinary placement choices.
            # Their committed canonical output device is the scheduler's
            # concrete placement; REDUCE/GATHER/SCATTER may additionally leave
            # copies on devices recorded in _collective_output_devs.
            committed = self._node_placement.get(str(nid))
            if committed in self.cluster.devices:
                return (self.cluster.devices[str(committed)],)
            return (self.cost.get_host_device(),)
        name_up = str(getattr(node, "name", "") or "").upper()
        if name_up in ("K_WRITE", "V_WRITE", "KV_WRITE"):
            pinned = self._preferred_kv_write_device(g, nid)
            if pinned is not None and self._node_allowed_on(node, pinned):
                return (pinned,)
            # Match Bifocal's long-standing fallback when the KV pin cannot be
            # resolved: fall through to the ordinary executor candidates.
        candidates: List[DeviceSpec] = []
        for device_type in self._executor_device_types():
            for device in self.cluster.devices_by_type(device_type):
                try:
                    allowed = self._node_allowed_on(node, device)
                except Exception:
                    # Preserve the scheduler's existing conservative behavior:
                    # an unavailable optional legality hint must not remove an
                    # otherwise capability-compatible executor.
                    allowed = True
                if allowed:
                    candidates.append(device)
        return tuple(candidates)

    def _hetinfer_kv_source_device(self, node: TaskNode) -> DeviceSpec:
        kv_place = str(self._kv_place(self.label) or "host").lower()
        if kv_place == "pim":
            source = self._kv_pim_for_node(node)
            if source is not None:
                return source
        elif kv_place == "npu":
            source = self._kv_npu_device(self.label)
            if source is not None:
                return source
        return self.cost.get_host_device()

    def _hetinfer_comm_devices(self, g: Any, nid: str) -> Tuple[DeviceSpec, ...]:
        node = g.nodes[nid]
        prim = str(self._comm_primitive(node) or "").upper()
        names: set[str] = set()
        for predecessor in g.predecessors(nid):
            placed = self._node_placement.get(str(predecessor))
            if placed in self.cluster.devices:
                names.add(str(placed))
        if prim == "SCATTER":
            names.update(self._collective_output_devs.get(str(nid), set()))
        elif prim == "TRANSFER":
            attrs = getattr(node, "attrs", {}) or {}
            for key in ("src", "dst"):
                name = attrs.get(key)
                if name in self.cluster.devices:
                    names.add(str(name))
        committed = self._node_placement.get(str(nid))
        if committed in self.cluster.devices:
            names.add(str(committed))
        return tuple(
            self.cluster.devices[name]
            for name in sorted(names)
            if name in self.cluster.devices
        )

    def _hetinfer_comm_service_s(
        self, g: Any, nid: str, phase: str
    ) -> Optional[float]:
        """Return an availability-free communication primitive duration.

        The duration comes from the same DOPS topology and communication model
        used by scheduling, including collectives whose participants include
        PIM devices.
        """

        node = g.nodes[nid]
        devices = self._hetinfer_comm_devices(g, nid)
        phase_eff = self._node_phase(g, nid, phase)
        batch = self._node_batch(g, nid, phase_eff)
        seq_len = self._node_seq_len(g, nid, phase_eff)
        read_bytes, write_bytes = self.cost.estimate_activation_bytes(
            node, batch, seq_len, phase_eff
        )
        tensor_bytes = int(max(int(read_bytes), int(write_bytes), 0))
        prim = str(self._comm_primitive(node) or "").upper()
        host = self.cost.get_host_device()
        host_name = str(host.name)

        def transfer(source_name: str, destination_name: str, bytes_: int) -> float:
            if source_name == destination_name or int(bytes_) == 0:
                return 0.0
            source = self.cluster.devices[str(source_name)]
            destination = self.cluster.devices[str(destination_name)]
            value = float(self.cost.comm_cost(source, destination, int(bytes_)))
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError(
                    "no finite communication primitive for "
                    f"{source_name!r}->{destination_name!r}"
                )
            return value

        participant_names = sorted(
            {
                str(self._node_placement[pred])
                for pred in g.predecessors(nid)
                if self._node_placement.get(pred) in self.cluster.devices
            }
        )
        duration = 0.0
        if prim in ("ALLREDUCE", "ALL_REDUCE", "ALL-REDUCE"):
            topology = normalize_topology(getattr(self.cluster, "topology", None))
            if topology == "fc":
                duration = float(
                    ring_allreduce(
                        cost=self.cost,
                        cluster=self.cluster,
                        ring=participant_names,
                        tensor_bytes=tensor_bytes,
                        start=0.0,
                    )
                )
            else:
                reduce_s = max(
                    (
                        transfer(name, host_name, tensor_bytes)
                        for name in participant_names
                    ),
                    default=0.0,
                )
                try:
                    dtype_bytes = float(self.cost._act_dtype_bytes(node, phase_eff))
                except Exception:
                    dtype_bytes = 2.0
                elements = float(tensor_bytes) / max(1.0, dtype_bytes)
                accumulation_s = float(
                    self.cost.flop_time(
                        max(0, len(participant_names) - 1) * elements, host
                    )
                )
                scatter_s = max(
                    (
                        transfer(host_name, name, tensor_bytes)
                        for name in participant_names
                    ),
                    default=0.0,
                )
                duration = reduce_s + accumulation_s + scatter_s
        elif prim in ("REDUCE", "GATHER"):
            duration = max(
                (
                    transfer(name, host_name, tensor_bytes)
                    for name in participant_names
                ),
                default=0.0,
            )
            if prim == "REDUCE":
                try:
                    dtype_bytes = float(self.cost._act_dtype_bytes(node, phase_eff))
                except Exception:
                    dtype_bytes = 2.0
                elements = float(tensor_bytes) / max(1.0, dtype_bytes)
                duration += float(
                    self.cost.flop_time(
                        max(0, len(participant_names) - 1) * elements, host
                    )
                )
        elif prim == "SCATTER":
            attrs = getattr(node, "attrs", {}) or {}
            targets = sorted(
                set(self._collective_output_devs.get(str(nid), set()))
                - {host_name}
            )
            per_target = tensor_bytes
            if str(attrs.get("scatter_mode", "broadcast")).lower() in (
                "partition",
                "shard",
                "split",
            ):
                per_target = int(
                    math.ceil(float(tensor_bytes) / float(max(1, len(targets))))
                )
            duration = max(
                (transfer(host_name, name, per_target) for name in targets),
                default=0.0,
            )
        elif prim == "TRANSFER":
            attrs = getattr(node, "attrs", {}) or {}
            source_name = str(attrs.get("src") or (participant_names or [host_name])[0])
            destination_name = str(attrs.get("dst") or host_name)
            override = attrs.get("bytes", attrs.get("bytes_nd", tensor_bytes))
            duration = transfer(source_name, destination_name, int(override))
        if not math.isfinite(duration) or duration < 0.0:
            raise RuntimeError(f"invalid communication service for {nid!r}: {duration!r}")
        return float(duration)

    def _hetinfer_compute_service_s(
        self,
        g: Any,
        nid: str,
        device: DeviceSpec,
        phase: str,
    ) -> Optional[float]:
        """Return movement-, reload-, and queue-free local execution seconds.

        NPU and PIM values come from the CostModel selected for this DOPS run.
        """

        node = g.nodes[nid]
        if self._is_comm_node(node):
            return self._hetinfer_comm_service_s(g, nid, phase)
        phase_eff = self._node_phase(g, nid, phase)
        batch = self._node_batch(g, nid, phase_eff)
        seq_len = self._node_seq_len(g, nid, phase_eff)
        name_up = str(getattr(node, "name", "") or "").upper()
        if name_up in ("K_WRITE", "V_WRITE", "KV_WRITE"):
            _, write_bytes = self.cost.estimate_activation_bytes(
                node, batch, seq_len, phase_eff
            )
            bytes_nd = int(self.cost.format_size(int(write_bytes), "ND"))
            if str(device.name) == str(self.cost.get_host_device().name):
                # The existing DOPS host-KV path models the predecessor-to-host
                # copy as movement and commits a zero-duration host-local
                # primitive.  Preserve that exact separation in the static
                # tables; adding host mem_time here would double count it.
                return 0.0
            if str(getattr(device, "type", "") or "").lower() == "pim":
                return float(self.cost.pim_write_time(bytes_nd, device))
            source_layout = str(self.cost.device_preferred_fmt(device))
            conversion_s = 0.0
            if source_layout != "ND":
                conversion_s = float(
                    self.cost.format_conversion_time(
                        bytes_nd, source_layout, "ND", device
                    )
                )
            return conversion_s + float(self.cost.mem_time(bytes_nd, device))
        result = float(
            self._weighted_compute_time(
                node,
                device,
                self.label,
                int(batch),
                int(seq_len),
                str(phase_eff),
            )
        )
        if name_up in ("QK", "SV"):
            source = self._hetinfer_kv_source_device(node)
            if str(source.name) == str(device.name):
                kv_bytes = int(
                    self.cost.estimate_kv_cache_read_bytes(
                        node, batch, seq_len, phase_eff
                    )
                )
                if kv_bytes > 0:
                    size_nd = int(self.cost.format_size(kv_bytes, "ND"))
                    result += float(self.cost.mem_time(size_nd, device))
                    destination_layout = str(
                        self.cost.device_preferred_fmt(device)
                    )
                    if destination_layout != "ND":
                        result += float(
                            self.cost.format_conversion_time(
                                size_nd,
                                "ND",
                                destination_layout,
                                device,
                            )
                        )
        if not math.isfinite(result) or result < 0.0:
            raise RuntimeError(
                f"invalid local service for {(nid, device.name)!r}: {result!r}"
            )
        return result

    def _hetinfer_edge_data_bytes(
        self, g: Any, source_id: str, destination_id: str, phase: str
    ) -> int:
        edge_tensor = getattr(g, "edge_tensor", None)
        if callable(edge_tensor):
            metadata = edge_tensor(source_id, destination_id)
            if isinstance(metadata, Mapping) and metadata.get("bytes") is not None:
                return max(0, int(metadata["bytes"]))
        source = g.nodes[source_id]
        source_phase = self._node_phase(g, source_id, phase)
        source_batch = self._node_batch(g, source_id, source_phase)
        source_seq = self._node_seq_len(g, source_id, source_phase)
        _, source_write = self.cost.estimate_activation_bytes(
            source, source_batch, source_seq, source_phase
        )
        return max(0, int(source_write))

    def _hetinfer_edge_tensor_id(
        self,
        g: Any,
        source_id: str,
        destination_id: str,
        phase: str,
        schedule_call_index: int,
    ) -> str:
        edge_tensor = getattr(g, "edge_tensor", None)
        if callable(edge_tensor):
            metadata = edge_tensor(source_id, destination_id)
            if isinstance(metadata, Mapping) and metadata.get("tensor_id"):
                return (
                    f"{phase}:{int(schedule_call_index)}:"
                    f"{str(metadata['tensor_id'])}"
                )
        attrs = getattr(g.nodes[source_id], "attrs", {}) or {}
        output = attrs.get("hetinfer_output")
        if isinstance(output, Mapping) and output.get("tensor_id"):
            return (
                f"{phase}:{int(schedule_call_index)}:"
                f"{str(output['tensor_id'])}"
            )
        if attrs.get("hetinfer_tensor_id"):
            return (
                f"{phase}:{int(schedule_call_index)}:"
                f"{str(attrs['hetinfer_tensor_id'])}"
            )
        # Tensor identity belongs to the producer output, not an edge.  Forked
        # consumers therefore share one tensor and one residency domain.
        return f"{phase}:{int(schedule_call_index)}:tensor:{source_id}"

    def _hetinfer_output_layout(
        self,
        g: Any,
        source_id: str,
        source: DeviceSpec,
        destination_id: Optional[str] = None,
    ) -> str:
        if str(source.name) == str(self.cost.get_host_device().name):
            return "ND"
        edge_tensor = getattr(g, "edge_tensor", None)
        if destination_id is not None and callable(edge_tensor):
            metadata = edge_tensor(source_id, destination_id)
            if isinstance(metadata, Mapping) and metadata.get("layout"):
                return str(metadata["layout"])
        node = g.nodes[source_id]
        attrs = getattr(node, "attrs", {}) or {}
        output = attrs.get("hetinfer_output")
        if isinstance(output, Mapping) and output.get("layout"):
            return str(output["layout"])
        if self._is_comm_node(node) or str(getattr(node, "name", "")).upper() in (
            "K_WRITE",
            "V_WRITE",
            "KV_WRITE",
        ):
            return "ND"
        return str(self.cost.device_preferred_fmt(source))

    def _hetinfer_route_time_s(
        self,
        source: DeviceSpec,
        destination: DeviceSpec,
        bytes_nd: int,
        *,
        source_layout: str,
        include_source_read: bool = False,
    ) -> float:
        if source.name == destination.name:
            return 0.0
        size_nd = int(self.cost.format_size(int(bytes_nd), "ND"))
        source_size = int(self.cost.format_size(int(bytes_nd), source_layout))
        source_conversion = 0.0
        if source_layout != "ND":
            source_conversion = float(
                self.cost.format_conversion_time(
                    source_size, source_layout, "ND", source
                )
            )
        source_read = 0.0
        if str(getattr(source, "type", "") or "").lower() == "pim":
            source_read = float(self.cost.activation_read_time_pim(size_nd))
        elif include_source_read:
            source_read = float(self.cost.mem_time(size_nd, source))
        base = source_conversion + source_read
        destination_layout = str(self.cost.device_preferred_fmt(destination))
        direct_probe = float(self.cost.comm_cost(source, destination, size_nd))
        direct = float("inf")
        if math.isfinite(direct_probe):
            direct = base + float(
                self.cost.combine_transfer_and_convert(
                    source,
                    destination,
                    size_nd,
                    "ND",
                    destination_layout,
                )
            )
        host = self.cost.get_host_device()
        via_host = float("inf")
        to_host = float(self.cost.comm_cost(source, host, size_nd))
        if math.isfinite(to_host):
            via_host = base + to_host + float(
                self.cost.combine_transfer_and_convert(
                    host,
                    destination,
                    size_nd,
                    "ND",
                    destination_layout,
                )
            )
        topology = normalize_topology(getattr(self.cluster, "topology", None))
        # The execution path uses the direct FC route when it exists, but it
        # still falls back through the host when that particular pair has no
        # finite direct link.  Keep the offline primitive identical.
        if topology == "fc" and math.isfinite(direct):
            result = direct
        else:
            result = min(direct, via_host)
        if not math.isfinite(result) or result < 0.0:
            raise RuntimeError(
                "no finite movement route for "
                f"{source.name!r}->{destination.name!r}"
            )
        return float(result)

    def _finalize_hetinfer_prior_snapshot(
        self,
        g: Any,
        phase: str,
        *,
        schedule_call_index: int,
    ) -> None:
        """Publish one complete clean snapshot after all placements commit."""

        if not self._hetinfer_prior_capture_enabled:
            return
        graph_nodes = tuple(str(nid) for nid in self._get_graph_index(g).nodes)
        if set(self._node_placement) != set(graph_nodes):
            raise RuntimeError(
                "cannot export an incomplete final placement: "
                f"missing={sorted(set(graph_nodes) - set(self._node_placement))}, "
                f"unexpected={sorted(set(self._node_placement) - set(graph_nodes))}"
            )

        devices = [
            {
                "device_id": str(device.name),
                "device_type": str(device.type).lower(),
            }
            for device in sorted(
                self.cluster.devices.values(), key=lambda item: str(item.name)
            )
        ]
        host = self.cost.get_host_device()
        host_name = str(host.name)
        topology = str(normalize_topology(getattr(self.cluster, "topology", None)))
        barrier_edges = {
            (str(source), str(destination))
            for source, destination in (getattr(g, "barrier_edges", ()) or ())
        }
        op_id_by_node = {
            nid: f"{phase}:{int(schedule_call_index)}:{nid}"
            for nid in graph_nodes
        }
        legal_by_node: Dict[str, Tuple[DeviceSpec, ...]] = {}
        for nid in graph_nodes:
            node = g.nodes[nid]
            legal = self._hetinfer_legal_devices(g, nid, node)
            if not legal:
                raise RuntimeError(f"operator {nid!r} has no legal export devices")
            legal_by_node[nid] = legal
            expert = str(self._node_placement[nid])
            if expert not in {str(device.name) for device in legal}:
                raise RuntimeError(
                    f"final placement {(nid, expert)!r} is outside legal candidates"
                )

        # Communication nodes expose one fixed, physical DOPS expert context.
        # Inputs may still originate from every legal upstream residency, but
        # they first stage to these fixed devices.  T_service starts only after
        # staging and atomically includes the primitive's internal transport.
        collective_context_by_node: Dict[str, Dict[str, Any]] = {}
        collective_staging_by_edge: Dict[Tuple[str, str], str] = {}
        for nid in graph_nodes:
            node = g.nodes[nid]
            if not self._is_comm_node(node):
                continue
            primitive = str(self._comm_primitive(node) or "").upper()
            if primitive in ("ALL_REDUCE", "ALL-REDUCE"):
                primitive = "ALLREDUCE"
            data_predecessors = [
                str(pred)
                for pred in g.predecessors(nid)
                if (str(pred), nid) not in barrier_edges
            ]
            if not data_predecessors:
                raise RuntimeError(
                    f"collective {nid!r} has no data input to bind to fixed staging"
                )
            attrs = getattr(node, "attrs", {}) or {}
            if primitive == "SCATTER":
                staging_devices = {pred: host_name for pred in data_predecessors}
            elif primitive == "TRANSFER":
                if len(data_predecessors) != 1:
                    raise RuntimeError(
                        "TRANSFER export requires exactly one data predecessor: "
                        f"{nid!r} has {data_predecessors!r}"
                    )
                effective_source = str(
                    attrs.get("src")
                    or self._node_placement.get(data_predecessors[0], host_name)
                )
                if effective_source not in self.cluster.devices:
                    raise RuntimeError(
                        f"TRANSFER {nid!r} references unknown source {effective_source!r}"
                    )
                staging_devices = {
                    data_predecessors[0]: effective_source
                }
            else:
                staging_devices = {
                    pred: str(self._node_placement[pred])
                    for pred in data_predecessors
                }
            participants = sorted(set(staging_devices.values()))
            if any(name not in self.cluster.devices for name in participants):
                raise RuntimeError(
                    f"collective {nid!r} has an unknown participant: {participants!r}"
                )
            recorded_outputs = {
                str(name)
                for name in self._collective_output_devs.get(nid, set())
            }
            if primitive == "ALLREDUCE":
                outputs = recorded_outputs or set(participants)
            elif primitive in ("REDUCE", "GATHER"):
                outputs = recorded_outputs or {host_name}
            elif primitive == "SCATTER":
                outputs = recorded_outputs or {host_name}
            elif primitive == "TRANSFER":
                destination = str(
                    attrs.get("dst") or self._node_placement.get(nid, host_name)
                )
                outputs = recorded_outputs or {destination}
            else:
                raise RuntimeError(
                    f"unsupported communication primitive for export: {primitive!r}"
                )
            canonical = str(self._node_placement[nid])
            if canonical not in outputs:
                raise RuntimeError(
                    f"collective {nid!r} canonical device {canonical!r} is not an output"
                )
            unknown_outputs = sorted(outputs - set(self.cluster.devices))
            if unknown_outputs:
                raise RuntimeError(
                    f"collective {nid!r} has unknown outputs: {unknown_outputs!r}"
                )
            node_phase = self._node_phase(g, nid, phase)
            node_batch = self._node_batch(g, nid, node_phase)
            node_seq = self._node_seq_len(g, nid, node_phase)
            read_bytes, write_bytes = self.cost.estimate_activation_bytes(
                node, node_batch, node_seq, node_phase
            )
            resources = set(participants) | set(outputs)
            if primitive == "ALLREDUCE" and topology != "fc":
                resources.add(host_name)
            context = {
                "op_id": op_id_by_node[nid],
                "primitive": primitive,
                "topology": topology,
                "canonical_device_id": canonical,
                "participant_device_ids": participants,
                "output_device_ids": sorted(outputs),
                "resource_device_ids": sorted(resources),
                "tensor_bytes": max(0, int(read_bytes), int(write_bytes)),
                "internal_transport": "included_in_t_service",
            }
            collective_context_by_node[nid] = context
            for pred, staging_device in staging_devices.items():
                collective_staging_by_edge[(pred, nid)] = staging_device

        routes_by_key: Dict[Tuple[str, str, str, int, str], Dict[str, Any]] = {}
        tensor_source_descriptions: Dict[Tuple[str, str], Tuple[int, str]] = {}
        inputs: List[Dict[str, Any]] = []
        input_keys: set[Tuple[str, Optional[str], str]] = set()
        tensor_bindings: Dict[str, Tuple[Optional[str], int]] = {}

        def add_input(
            *,
            consumer_id: str,
            producer_id: Optional[str],
            tensor_id: str,
            semantics: str,
            bytes_nd: int,
            source_residencies: List[Dict[str, str]],
            destination_devices: List[str],
        ) -> None:
            consumer_op_id = op_id_by_node[consumer_id]
            producer_op_id = (
                None if producer_id is None else op_id_by_node[producer_id]
            )
            key = (consumer_op_id, producer_op_id, str(tensor_id))
            if key in input_keys:
                raise RuntimeError(f"duplicate static-prior input binding: {key!r}")
            input_keys.add(key)
            binding = (producer_op_id, int(bytes_nd))
            previous = tensor_bindings.setdefault(str(tensor_id), binding)
            if previous != binding:
                raise RuntimeError(
                    "one tensor_id cannot change producer or bytes across inputs: "
                    f"{tensor_id!r}, previous={previous!r}, new={binding!r}"
                )
            residency_devices = [
                str(entry["device_id"]) for entry in source_residencies
            ]
            if len(residency_devices) != len(set(residency_devices)):
                raise RuntimeError(
                    f"input {key!r} has duplicate source residency devices"
                )
            inputs.append(
                {
                    "consumer_op_id": consumer_op_id,
                    "producer_op_id": producer_op_id,
                    "tensor_id": str(tensor_id),
                    "semantics": str(semantics),
                    "bytes": int(bytes_nd),
                    "source_residencies": copy.deepcopy(source_residencies),
                    "destination_devices": sorted(
                        {str(name) for name in destination_devices}
                    ),
                }
            )

        def add_route(
            *,
            tensor_id: str,
            source: DeviceSpec,
            destination: DeviceSpec,
            bytes_nd: int,
            layout: str,
            include_source_read: bool = False,
        ) -> None:
            source_description_key = (str(tensor_id), str(source.name))
            source_description = (int(bytes_nd), str(layout))
            previous_description = tensor_source_descriptions.get(
                source_description_key
            )
            if (
                previous_description is not None
                and previous_description != source_description
            ):
                raise RuntimeError(
                    "one tensor residency cannot change bytes/layout across "
                    f"consumers: {source_description_key!r}, "
                    f"previous={previous_description!r}, "
                    f"new={source_description!r}"
                )
            tensor_source_descriptions[source_description_key] = source_description
            resident = str(source.name) == str(destination.name)
            if resident:
                duration_s = 0.0
            else:
                duration_s = self._hetinfer_route_time_s(
                    source,
                    destination,
                    int(bytes_nd),
                    source_layout=str(layout),
                    include_source_read=bool(include_source_read),
                )
            key = (
                str(tensor_id),
                str(source.name),
                str(destination.name),
                int(bytes_nd),
                str(layout),
            )
            entry = {
                "tensor_id": key[0],
                "source_device_id": key[1],
                "destination_device_id": key[2],
                "bytes": key[3],
                "layout": key[4],
                "duration_s": duration_s,
            }
            previous = routes_by_key.get(key)
            if previous is not None and previous != entry:
                raise RuntimeError(f"inconsistent duplicate route {key!r}")
            routes_by_key[key] = entry

        # Root operators consume explicit external inputs when declared.  The
        # compatibility fallback is one ND tensor resident on the host.
        for destination_id in graph_nodes:
            data_predecessors = tuple(
                str(item)
                for item in g.predecessors(destination_id)
                if (str(item), destination_id) not in barrier_edges
            )
            if not data_predecessors:
                raw_inputs: Any = None
                external_inputs_for = getattr(g, "external_inputs_for", None)
                if callable(external_inputs_for):
                    raw_inputs = external_inputs_for(destination_id)
                if raw_inputs in (None, (), []):
                    attrs = getattr(g.nodes[destination_id], "attrs", {}) or {}
                    raw_inputs = attrs.get("hetinfer_external_inputs")
                if raw_inputs in (None, (), []):
                    destination_node = g.nodes[destination_id]
                    destination_phase = self._node_phase(
                        g, destination_id, phase
                    )
                    destination_batch = self._node_batch(
                        g, destination_id, destination_phase
                    )
                    destination_seq = self._node_seq_len(
                        g, destination_id, destination_phase
                    )
                    read_bytes, _ = self.cost.estimate_activation_bytes(
                        destination_node,
                        destination_batch,
                        destination_seq,
                        destination_phase,
                    )
                    raw_inputs = [
                        {
                            "tensor_id": f"input:{destination_id}",
                            "source_devices": [str(host.name)],
                            "bytes": max(0, int(read_bytes)),
                            "layout": "ND",
                        }
                    ]
                if isinstance(raw_inputs, Mapping) or isinstance(
                    raw_inputs, (str, bytes)
                ):
                    raise RuntimeError(
                        f"external inputs for {destination_id!r} must be an array"
                    )
                for input_index, raw_input in enumerate(raw_inputs):
                    if not isinstance(raw_input, Mapping):
                        raise RuntimeError(
                            "external input must be an object: "
                            f"{destination_id!r}[{input_index}]"
                        )
                    raw_tensor_id = str(
                        raw_input.get("tensor_id")
                        or f"input:{destination_id}:{input_index}"
                    )
                    tensor_id = (
                        f"{phase}:{int(schedule_call_index)}:{raw_tensor_id}"
                    )
                    source_names = raw_input.get(
                        "source_devices", [str(host.name)]
                    )
                    if isinstance(source_names, (str, bytes)):
                        source_names = [str(source_names)]
                    try:
                        source_names = [str(item) for item in source_names]
                    except TypeError as exc:
                        raise RuntimeError(
                            "external input source_devices must be an array: "
                            f"{destination_id!r}[{input_index}]"
                        ) from exc
                    if not source_names:
                        raise RuntimeError(
                            "external input source_devices cannot be empty: "
                            f"{destination_id!r}[{input_index}]"
                        )
                    bytes_nd = int(raw_input.get("bytes", 0))
                    if bytes_nd < 0:
                        raise RuntimeError(
                            "external input bytes cannot be negative: "
                            f"{destination_id!r}[{input_index}]"
                        )
                    layout = str(raw_input.get("layout", "ND") or "ND")
                    source_residencies: List[Dict[str, str]] = []
                    destination_names = [
                        str(device.name) for device in legal_by_node[destination_id]
                    ]
                    for source_name in source_names:
                        if str(source_name) not in self.cluster.devices:
                            raise RuntimeError(
                                "external input references unknown source device "
                                f"{source_name!r}"
                            )
                        source = self.cluster.devices[str(source_name)]
                        source_residencies.append(
                            {"device_id": str(source.name), "layout": layout}
                        )
                        for destination in legal_by_node[destination_id]:
                            add_route(
                                tensor_id=tensor_id,
                                source=source,
                                destination=destination,
                                bytes_nd=bytes_nd,
                                layout=layout,
                            )
                    add_input(
                        consumer_id=destination_id,
                        producer_id=None,
                        tensor_id=tensor_id,
                        semantics="data",
                        bytes_nd=bytes_nd,
                        source_residencies=source_residencies,
                        destination_devices=destination_names,
                    )

            # QK/SV have an additional cache tensor whose residency is fixed by
            # the DOPS KV plan.  It is independent of ordinary graph edges.
            node = g.nodes[destination_id]
            role = str(getattr(node, "name", "") or "").upper()
            if role in ("QK", "SV"):
                destination_phase = self._node_phase(g, destination_id, phase)
                destination_batch = self._node_batch(
                    g, destination_id, destination_phase
                )
                destination_seq = self._node_seq_len(
                    g, destination_id, destination_phase
                )
                kv_bytes = max(
                    0,
                    int(
                        self.cost.estimate_kv_cache_read_bytes(
                            node,
                            destination_batch,
                            destination_seq,
                            destination_phase,
                        )
                    ),
                )
                if kv_bytes > 0:
                    source = self._hetinfer_kv_source_device(node)
                    tensor_id = (
                        f"{phase}:{int(schedule_call_index)}:"
                        f"kv:{'K' if role == 'QK' else 'V'}:{destination_id}"
                    )
                    for destination in legal_by_node[destination_id]:
                        add_route(
                            tensor_id=tensor_id,
                            source=source,
                            destination=destination,
                            bytes_nd=kv_bytes,
                            layout="ND",
                            include_source_read=True,
                        )
                    add_input(
                        consumer_id=destination_id,
                        producer_id=None,
                        tensor_id=tensor_id,
                        semantics="data",
                        bytes_nd=kv_bytes,
                        source_residencies=[
                            {"device_id": str(source.name), "layout": "ND"}
                        ],
                        destination_devices=[
                            str(device.name)
                            for device in legal_by_node[destination_id]
                        ],
                    )

        # Each producer output is one tensor even when it fans out.  Its source
        # domain is every legal producer placement (or every fixed collective
        # output) plus a possible host spill.  Ordinary consumers target all of
        # their legal devices; collective inputs target exactly one fixed
        # staging device.  Internal collective hops never appear as T_move.
        for source_id in graph_nodes:
            successors = tuple(str(item) for item in g.successors(source_id))
            if not successors:
                continue
            source_context = collective_context_by_node.get(source_id)
            if source_context is None:
                source_devices: Dict[str, DeviceSpec] = {
                    str(device.name): device
                    for device in legal_by_node[source_id]
                }
            else:
                source_devices = {
                    str(name): self.cluster.devices[str(name)]
                    for name in source_context["output_device_ids"]
                }
            # Ordinary DOPS execution may spill an activation to host.  Host is
            # therefore a legal source residency even when it is not an
            # operator execution candidate.
            source_devices[host_name] = host
            for destination_id in successors:
                if (source_id, destination_id) in barrier_edges:
                    add_input(
                        consumer_id=destination_id,
                        producer_id=source_id,
                        tensor_id=(
                            f"{phase}:{int(schedule_call_index)}:"
                            f"barrier:{source_id}->{destination_id}"
                        ),
                        semantics="barrier",
                        bytes_nd=0,
                        source_residencies=[],
                        destination_devices=[],
                    )
                    continue
                tensor_id = self._hetinfer_edge_tensor_id(
                    g,
                    source_id,
                    destination_id,
                    phase,
                    schedule_call_index,
                )
                bytes_nd = self._hetinfer_edge_data_bytes(
                    g, source_id, destination_id, phase
                )
                staging_device = collective_staging_by_edge.get(
                    (source_id, destination_id)
                )
                if staging_device is None:
                    semantics = "data"
                    consumer_destinations = {
                        str(device.name): device
                        for device in legal_by_node[destination_id]
                    }
                else:
                    semantics = "collective_staging"
                    consumer_destinations = {
                        str(staging_device): self.cluster.devices[str(staging_device)]
                    }
                # DOPS may materialize an activation in host memory before a
                # later consumer reloads it.  Export both halves of that
                # residency transition: producer->host store and
                # host->consumer reload (including host->host resident zero).
                route_destinations = dict(consumer_destinations)
                route_destinations[host_name] = host
                source_residencies: List[Dict[str, str]] = []
                for source in source_devices.values():
                    layout = self._hetinfer_output_layout(
                        g, source_id, source, destination_id
                    )
                    source_residencies.append(
                        {"device_id": str(source.name), "layout": str(layout)}
                    )
                    for destination in route_destinations.values():
                        add_route(
                            tensor_id=tensor_id,
                            source=source,
                            destination=destination,
                            bytes_nd=bytes_nd,
                            layout=layout,
                        )
                add_input(
                    consumer_id=destination_id,
                    producer_id=source_id,
                    tensor_id=tensor_id,
                    semantics=semantics,
                    bytes_nd=bytes_nd,
                    source_residencies=source_residencies,
                    destination_devices=list(consumer_destinations),
                )

        routes = [routes_by_key[key] for key in sorted(routes_by_key)]

        operators: List[Dict[str, Any]] = []
        for nid in graph_nodes:
            node = g.nodes[nid]
            legal = legal_by_node[nid]
            node_phase = self._node_phase(g, nid, phase)
            node_batch = self._node_batch(g, nid, node_phase)
            node_seq = self._node_seq_len(g, nid, node_phase)
            operators.append(
                {
                    "op_id": op_id_by_node[nid],
                    "dependencies": [
                        op_id_by_node[str(pred)] for pred in g.predecessors(nid)
                    ],
                    "legal_devices": [str(device.name) for device in legal],
                    "expert_device": str(self._node_placement[nid]),
                    "service_s": {
                        str(device.name): self._hetinfer_compute_service_s(
                            g, nid, device, phase
                        )
                        for device in legal
                    },
                    "network_metadata": {
                        "name": str(getattr(node, "name", "") or "UNKNOWN"),
                        "phase": str(node_phase),
                        "batch": int(node_batch),
                        "seq_len": int(node_seq),
                        "node_attrs": dict(getattr(node, "attrs", {}) or {}),
                    },
                }
            )

        snapshot = {
            "schedule_call_index": int(schedule_call_index),
            "phase": str(phase),
            "devices": devices,
            "operators": operators,
            "inputs": sorted(
                inputs,
                key=lambda entry: (
                    entry["consumer_op_id"],
                    entry["producer_op_id"] or "",
                    entry["tensor_id"],
                    entry["semantics"],
                ),
            ),
            "collective_contexts": [
                copy.deepcopy(collective_context_by_node[nid])
                for nid in graph_nodes
                if nid in collective_context_by_node
            ],
            "routes": routes,
        }
        self._hetinfer_prior_snapshots.append(copy.deepcopy(snapshot))
        self._hetinfer_schedule_call_index = int(schedule_call_index)

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
        self._last_op_trace_extra = {}
        self._last_candidate_component_extra = {}
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
                cache.meta.clear()
            self.weight_cached.clear()
            self.storage_fmt_map.clear()
            self._weight_load_count.clear()
            self._weight_sizes.clear()
            self._weight_proto_node.clear()

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
         
    def _kv_place(self, label: Optional[PlanLabel] = None) -> str:
        """Normalize KV placement tag."""
        lb = label if label is not None else getattr(self, 'label', None)
        if lb is None:
            return 'host'

        # 1) Explicit tag
        try:
            kp = getattr(lb, 'kv_place', None)
            if kp is not None:
                s = str(kp).strip().lower()
                if s in ('pim', 'aim'):
                    return 'pim'
                if s in ('npu', 'gpu', 'device'):
                    return 'npu'
                if s in ('host', 'cpu', 'dram'):
                    return 'host'
        except Exception:
            pass

        # 2) Backward compat flags
        try:
            if bool(getattr(lb, 'kv_in_pim', False)):
                return 'pim'
        except Exception:
            pass
        try:
            if bool(getattr(lb, 'kv_in_npu', False)):
                return 'npu'
        except Exception:
            pass

        # 3) Heuristic: if a target NPU is set, treat as NPU KV
        try:
            if getattr(lb, 'kv_npu_device', None):
                return 'npu'
        except Exception:
            pass

        return 'host'

    def _kv_npu_device(self, label: Optional[PlanLabel] = None) -> Optional[DeviceSpec]:
        """Return the designated KV-storage NPU device (single-device KV), or None."""
        lb = label if label is not None else getattr(self, 'label', None)
        if lb is None:
            return None
        if self._kv_place(lb) != 'npu':
            return None

        # Prefer explicit pinned device name
        try:
            name = getattr(lb, 'kv_npu_device', None)
        except Exception:
            name = None
        if name:
            dev = self.cluster.devices.get(str(name))
            if dev is not None and str(getattr(dev, 'type', '')).lower() == 'npu':
                return dev
        # Fallback: if not specified, choose the first available NPU
        try:
            npus = self.cluster.devices_by_type("npu") or []
            if npus:
                return npus[0]
        except Exception:
            pass

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
        kv_place = self._kv_place(self.label)

        # KV on PIM: write must go to the mapped PIM shard.
        if kv_place == 'pim':
            return self._kv_pim_for_node(node)

        if kv_place == 'npu':
            return self._kv_npu_device(self.label)
        return self.cost.get_host_device()

    def _node_allowed_on(self, node: TaskNode, dev: DeviceSpec) -> bool:

        # ---- 0) KV-write hard rule (override operator/baseline allow-list) ----
        kv_place = self._kv_place(self.label)

        name_up = str(getattr(node, "name", "") or "").upper()
        dev_type = str(getattr(dev, "type", "") or "").lower()
        dev_name = str(getattr(dev, "name", "") or "")
    
        # KV write ops must execute on the KV-storage device.
        if name_up in ("K_WRITE", "V_WRITE", "KV_WRITE"):
            # KV stored on PIM: write must go to the mapped PIM shard.
            if kv_place == "pim":
                if dev_type != "pim":
                    return False
                mapped = self._kv_pim_for_node(node)
                if mapped is None:
                    # If mapping is unavailable, fall back to first PIM (same as _earliest_finish_on_device)
                    pim_devs = self.cluster.devices_by_type("pim") or []
                    if not pim_devs:
                        return False
                    return dev_name == str(getattr(pim_devs[0], "name", ""))
                return dev_name == str(getattr(mapped, "name", ""))

            # KV stored on NPU: write must go to the designated KV NPU.
            if kv_place == "npu":
                if dev_type != "npu":
                    return False
                tgt = self._kv_npu_device(self.label)
                if tgt is None:
                    return False
                return dev_name == str(getattr(tgt, "name", ""))

            # KV stored on host: write must execute on the host (CPU) device.
            host = self.cost.get_host_device()
            host_name = str(getattr(host, "name", "")) if host is not None else ""
            return dev_name == host_name

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
        if kv_place == 'pim':
            kv_local_ops = {"K","V","QK","SOFTMAX","SV"}
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
# ?
    def _record_weight_proto_node(self, node: TaskNode) -> None:
        wid = self._node_weight_id(node)
        if not wid:
            return
        self._weight_proto_node.setdefault(str(wid), node)

    def _representative_weight_node(self, wid: str) -> TaskNode:
        node = self._weight_proto_node.get(str(wid))
        if node is None:
            raise RuntimeError(
                f"No representative node recorded for weight_id='{wid}'. "
                "Run at least one scheduling pass before asking for weight-format suggestions."
            )
        return node

    def _weight_storage_format_for_wid(self, wid: str) -> str:
        try:
            raw = self.buffer.get_host_fmt(str(wid))
        except Exception:
            raw = 'ND'
        try:
            return str(self.cost.weight_storage_format(raw or 'ND'))
        except Exception:
            s = str(raw or 'ND').strip().upper().replace('_', '-')
            if s == 'NPU-OPT':
                return 'NZ'
            if s == 'PIM-OPT':
                return 'PIM-OPT'
            return 'ND' if not s else s

    def _cached_weight_state(self, dev: DeviceSpec, wid: str, wsize_nd: int) -> Tuple[int, Optional[str], Optional[Any]]:
        cached_nd = 0
        cache = None
        try:
            cache = getattr(self.buffer, 'device_cache', {}).get(dev.name, None)
            items = getattr(cache, 'items', None)
            if isinstance(items, dict):
                v = items.get(wid, 0)
                if isinstance(v, (int, float)):
                    cached_nd = int(v)
        except Exception:
            cached_nd = 0
            cache = None

        if cached_nd <= 0:
            try:
                if self.buffer.is_cached(dev.name, wid):
                    cached_nd = int(wsize_nd)
            except Exception:
                cached_nd = 0

        if 0 < int(cached_nd) < int(wsize_nd):
            raise RuntimeError(
                f"Partial weight caching is not modeled for weight_id='{wid}' on device='{dev.name}'. "
                f"cached_nd={cached_nd} full_nd={wsize_nd}"
            )

        cache_fmt: Optional[str] = None
        if int(cached_nd) >= int(wsize_nd) and int(wsize_nd) > 0:
            try:
                cache_fmt = self.buffer.get_cached_weight_format(dev.name, wid)
            except Exception:
                cache_fmt = None
            if cache_fmt in (None, ''):
                cache_fmt = self._weight_storage_format_for_wid(wid)
            try:
                cache_fmt = str(self.cost.weight_storage_format(cache_fmt))
            except Exception:
                cache_fmt = str(cache_fmt)

        return int(max(0, cached_nd)), cache_fmt, cache


    def _pim_weight_load_overlap_ratio(self) -> float:
        try:
            import config as _cfg
            ratio = getattr(_cfg, 'PIM_WEIGHT_LOAD_OVERLAP_RATIO', 0.0)
        except Exception:
            ratio = 0.0
        return float(clamp_overlap_ratio(ratio, default=0.0))

    def _weight_load_compute_overlap_ratio(self) -> float:
        try:
            import config as _cfg
            ratio = getattr(_cfg, 'WEIGHT_LOAD_COMPUTE_OVERLAP_RATIO', 0.0)
        except Exception:
            ratio = 0.0
        return float(clamp_overlap_ratio(ratio, default=0.0))

    def _weight_overlap_ratio(self) -> float:
        return float(self._weight_load_compute_overlap_ratio())

    def _weight_host_source_format(self, dev: DeviceSpec, src_storage_fmt: str) -> str:
        return str(self.cost.weight_host_source_format(str(src_storage_fmt), dev))

    def _npu_resident_weight_format(self, src_storage_fmt: str) -> str:
        pseudo_dev = SimpleNamespace(type='npu')
        host_src_fmt = self.cost.weight_host_source_format(str(src_storage_fmt), pseudo_dev)
        return str(self.cost.weight_resident_format(str(host_src_fmt), pseudo_dev))

    def _weight_resident_format(self, dev: DeviceSpec, src_storage_fmt: str) -> str:
        host_src_fmt = self._weight_host_source_format(dev, str(src_storage_fmt))
        return str(self.cost.weight_resident_format(str(host_src_fmt), dev))

    def _weight_compute_stage_profile(
        self,
        node: TaskNode,
        dev: DeviceSpec,
        label: PlanLabel,
        batch: int,
        seq_len: int,
        phase: str,
        *,
        resident_fmt: str,
    ) -> Dict[str, Any]:
        stage = self.cost.weighted_compute_stage(
            node,
            dev,
            label,
            int(batch),
            int(seq_len),
            str(phase),
            resident_weight_fmt=str(resident_fmt),
        )
        return {
            'compute_fmt': str(stage.compute_fmt),
            'compute_total_s': float(stage.total_s),
            'compute_backend': str(stage.backend),
            'compute_rule': str(stage.combine_rule),
            'b1_s': float(stage.b1_s),
            'b2_s': float(stage.b2_s),
            'launch_overhead_s': float(stage.launch_overhead_s),
        }

    def _weighted_compute_time(
        self,
        node: TaskNode,
        dev: DeviceSpec,
        label: PlanLabel,
        batch: int,
        seq_len: int,
        phase: str,
        *,
        resident_fmt: Optional[str] = None,
    ) -> float:
        if not self._node_weight_id(node) or int(self._node_weight_size(node) or 0) <= 0:
            return float(self.cost.node_device_cost(node, dev, label, batch, seq_len, phase))
        src_storage_fmt = self._weight_storage_format_for_wid(self._node_weight_id(node))
        resident = str(resident_fmt or self._weight_resident_format(dev, str(src_storage_fmt)))
        prof = self._weight_compute_stage_profile(
            node,
            dev,
            label,
            int(batch),
            int(seq_len),
            str(phase),
            resident_fmt=str(resident),
        )
        return float(prof.get('compute_total_s', 0.0) or 0.0)

    def _zero_weight_service_profile(self, node: TaskNode, dev: DeviceSpec, *, src_storage_fmt: str = 'ND') -> Dict[str, Any]:
        resident_fmt = self._weight_resident_format(dev, str(src_storage_fmt))
        return {
            'wid': str(self._node_weight_id(node) or ''),
            'weight_size_nd': int(self._node_weight_size(node) or 0),
            'host_storage_fmt': str(src_storage_fmt),
            'host_src_fmt': str(self._weight_host_source_format(dev, str(src_storage_fmt))),
            'resident_fmt': str(resident_fmt),
            'compute_fmt': str(self.cost.device_preferred_fmt(dev)),
            'cache_state': '',
            'queue_wait_s': 0.0,
            'load_active_s': 0.0,
            'load_total_s': 0.0,
            'load_comm_s': 0.0,
            'load_l1_s': 0.0,
            'load_l2_s': 0.0,
            'load_l2_write_only_s': 0.0,
            'load_l2_pack_only_est_s': 0.0,
            'load_l1_l2_overlap_ratio': float(self._pim_weight_load_overlap_ratio()),
            'compute_total_s': 0.0,
            'compute_backend': '',
            'compute_rule': '',
            'b1_s': 0.0,
            'b2_s': 0.0,
            'launch_overhead_s': 0.0,
            'lc_overlap_ratio': float(self._weight_load_compute_overlap_ratio()),
            'lc_overlap_saved_s': 0.0,
            'total_s': 0.0,
        }

    def _weight_service_profile_no_contention(
        self,
        node: TaskNode,
        dev: DeviceSpec,
        *,
        src_storage_fmt: Optional[str] = None,
        cached: bool,
        cached_fmt: Optional[str] = None,
        label: Optional[PlanLabel] = None,
        batch: Optional[int] = None,
        seq_len: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        wid = self._node_weight_id(node)
        wsize_nd = int(self._node_weight_size(node) or 0)
        src_storage_fmt = str(src_storage_fmt or self._weight_storage_format_for_wid(wid))
        if not wid or wsize_nd <= 0:
            return self._zero_weight_service_profile(node, dev, src_storage_fmt=str(src_storage_fmt))

        phase_eff = str(phase or 'prefill')
        batch_eff = int(batch or getattr(self, 'batch', 1) or 1)
        seq_eff = int(seq_len or getattr(self, 'seq_len', 1) or 1)
        label_eff = label if label is not None else self.label
        dev_type = str(getattr(dev, 'type', '') or '').lower()
        host_src_fmt = self._weight_host_source_format(dev, str(src_storage_fmt))
        resident_fmt = str(cached_fmt or self._weight_resident_format(dev, str(src_storage_fmt)))

        if dev_type == 'pim' and self._weights_preloaded_on_pim():
            compute_prof = self._weight_compute_stage_profile(
                node, dev, label_eff, batch_eff, seq_eff, phase_eff, resident_fmt='PIM-OPT'
            )
            lc = overlap_time(0.0, float(compute_prof['compute_total_s']), self._weight_load_compute_overlap_ratio())
            return {
                'wid': str(wid),
                'weight_size_nd': int(wsize_nd),
                'host_storage_fmt': str(src_storage_fmt),
                'host_src_fmt': str(host_src_fmt),
                'resident_fmt': 'PIM-OPT',
                'compute_fmt': str(compute_prof['compute_fmt']),
                'cache_state': 'preloaded',
                'queue_wait_s': 0.0,
                'load_active_s': 0.0,
                'load_total_s': 0.0,
                'load_comm_s': 0.0,
                'load_l1_s': 0.0,
                'load_l2_s': 0.0,
                'load_l2_write_only_s': 0.0,
                'load_l2_pack_only_est_s': 0.0,
                'load_l1_l2_overlap_ratio': float(self._pim_weight_load_overlap_ratio()),
                'compute_total_s': float(compute_prof['compute_total_s']),
                'compute_backend': str(compute_prof['compute_backend']),
                'compute_rule': str(compute_prof['compute_rule']),
                'b1_s': float(compute_prof['b1_s']),
                'b2_s': float(compute_prof['b2_s']),
                'launch_overhead_s': float(compute_prof['launch_overhead_s']),
                'lc_overlap_ratio': float(lc.overlap_ratio),
                'lc_overlap_saved_s': float(lc.saved_s),
                'total_s': float(lc.total_s),
            }

        if bool(cached):
            compute_prof = self._weight_compute_stage_profile(
                node, dev, label_eff, batch_eff, seq_eff, phase_eff, resident_fmt=str(resident_fmt)
            )
            lc = overlap_time(0.0, float(compute_prof['compute_total_s']), self._weight_load_compute_overlap_ratio())
            return {
                'wid': str(wid),
                'weight_size_nd': int(wsize_nd),
                'host_storage_fmt': str(src_storage_fmt),
                'host_src_fmt': str(host_src_fmt),
                'resident_fmt': str(resident_fmt),
                'compute_fmt': str(compute_prof['compute_fmt']),
                'cache_state': 'cached',
                'queue_wait_s': 0.0,
                'load_active_s': 0.0,
                'load_total_s': 0.0,
                'load_comm_s': 0.0,
                'load_l1_s': 0.0,
                'load_l2_s': 0.0,
                'load_l2_write_only_s': 0.0,
                'load_l2_pack_only_est_s': 0.0,
                'load_l1_l2_overlap_ratio': float(self._pim_weight_load_overlap_ratio()),
                'compute_total_s': float(compute_prof['compute_total_s']),
                'compute_backend': str(compute_prof['compute_backend']),
                'compute_rule': str(compute_prof['compute_rule']),
                'b1_s': float(compute_prof['b1_s']),
                'b2_s': float(compute_prof['b2_s']),
                'launch_overhead_s': float(compute_prof['launch_overhead_s']),
                'lc_overlap_ratio': float(lc.overlap_ratio),
                'lc_overlap_saved_s': float(lc.saved_s),
                'total_s': float(lc.total_s),
            }

        host = self.cost.get_host_device()
        l1_bytes = int(self.cost.weight_transfer_comm_bytes(int(wsize_nd), str(src_storage_fmt), dev_or_type=dev))
        try:
            l1_s = float(self.cost.comm_cost(host, dev, int(l1_bytes)))
        except Exception:
            l1_s = float('inf')

        if dev_type == 'pim':
            l2_s = float(self.cost.pim_local_weight_load_time(int(wsize_nd), str(host_src_fmt), dev=dev))
            l2_write_only_s = float(self.cost.pim_local_weight_write_only_time(int(wsize_nd), dev=dev))
            l2_pack_only_est_s = float(self.cost.pim_local_weight_pack_only_est_time(int(wsize_nd), str(host_src_fmt), dev=dev))
            load_join = overlap_time(float(l1_s), float(l2_s), self._pim_weight_load_overlap_ratio())
        elif dev_type == 'cpu':
            compute_fmt = str(self.cost.device_preferred_fmt(dev))
            l2_s = float(self.cost.format_conversion_time(int(l1_bytes), str(host_src_fmt), str(compute_fmt), dev)) if str(host_src_fmt) != str(compute_fmt) else 0.0
            l2_write_only_s = 0.0
            l2_pack_only_est_s = 0.0
            load_join = overlap_time(float(l1_s), float(l2_s), 0.0)
            resident_fmt = str(host_src_fmt)
        else:
            l2_s = 0.0
            l2_write_only_s = 0.0
            l2_pack_only_est_s = 0.0
            load_join = overlap_time(float(l1_s), 0.0, 0.0)

        compute_prof = self._weight_compute_stage_profile(
            node, dev, label_eff, batch_eff, seq_eff, phase_eff, resident_fmt=str(resident_fmt)
        )
        lc = overlap_time(float(load_join.total_s), float(compute_prof['compute_total_s']), self._weight_load_compute_overlap_ratio())
        return {
            'wid': str(wid),
            'weight_size_nd': int(wsize_nd),
            'host_storage_fmt': str(src_storage_fmt),
            'host_src_fmt': str(host_src_fmt),
            'resident_fmt': str(resident_fmt),
            'compute_fmt': str(compute_prof['compute_fmt']),
            'cache_state': 'miss',
            'queue_wait_s': 0.0,
            'load_active_s': float(load_join.total_s),
            'load_total_s': float(load_join.total_s),
            'load_comm_s': float(l1_s),
            'load_l1_s': float(l1_s),
            'load_l2_s': float(l2_s),
            'load_l2_write_only_s': float(l2_write_only_s),
            'load_l2_pack_only_est_s': float(l2_pack_only_est_s),
            'load_l1_l2_overlap_ratio': float(load_join.overlap_ratio if dev_type == 'pim' else 0.0),
            'compute_total_s': float(compute_prof['compute_total_s']),
            'compute_backend': str(compute_prof['compute_backend']),
            'compute_rule': str(compute_prof['compute_rule']),
            'b1_s': float(compute_prof['b1_s']),
            'b2_s': float(compute_prof['b2_s']),
            'launch_overhead_s': float(compute_prof['launch_overhead_s']),
            'lc_overlap_ratio': float(lc.overlap_ratio),
            'lc_overlap_saved_s': float(lc.saved_s),
            'total_s': float(lc.total_s),
        }

from .scheduler_common import *
from .scheduler_types import _GraphIndex, ScheduledTask
from .scheduler_comm import CommManager


class SchedulerBaseTimingMixin:
    def _earliest_finish_on_device(
        self,
        g: TaskGraph,
        nid: str,
        dev: DeviceSpec,
        label: PlanLabel,
        phase: str,
        commit: bool,
    ) -> Tuple[float, float]:
        if not commit:
            self._last_candidate_component_extra = {}
        node = g.nodes[nid]
        phase_eff = self._node_phase(g, nid, phase)
        batch = self._node_batch(g, nid, phase_eff)
        seq_len = self._node_seq_len(g, nid, phase_eff)
        
        #---------------------1. KV write specially (write KV cache back)
        if node.name.upper() in ("K_WRITE", "V_WRITE", "KV_WRITE"):
            kv_place = self._kv_place(label)
            _, out_write_nd = self.cost.estimate_activation_bytes(node, batch, seq_len, phase_eff)
            size_nd = self.cost.format_size(out_write_nd, "ND")
            logger.debug(
                "[kv-write] node=%s target=%s kv_place=%s bytes_nd=%s", nid, dev.name, kv_place, out_write_nd
            )

            if kv_place == 'pim':
                pim_devs = self.cluster.devices_by_type("pim")
                if not pim_devs:
                    raise RuntimeError("kv_in_pim is True but no PIM device exists")

                target_pim = self._kv_pim_for_node(node) or pim_devs[0]

                # Ensure the K/V activation is available on the target PIM before the store.
                ready_kv = self._ready_time_for_device(g, nid, target_pim, phase_eff, commit)
                start = max(float(self.avail.get(target_pim.name, 0.0)), float(ready_kv))
                finish = start + float(self.cost.pim_write_time(int(out_write_nd), target_pim))
                logger.debug(
                    "[kv-write] node=%s target_pim=%s start=%.4f finish=%.4f", nid, target_pim.name, start, finish
                )
                if commit:
                    self.avail[target_pim.name] = float(finish)
                    self._node_finish_time[nid] = float(finish)
                    self._node_placement[nid] = str(target_pim.name)
                    self._node_out_fmt[nid] = "ND"
                    self._act_resident[(str(target_pim.name), str(nid))] = 0
                return float(start), float(finish)

            if kv_place == 'npu':
                target_npu = self._kv_npu_device(label)
                if target_npu is None:
                    # No NPU exists (or label inconsistent) -> fall back to host.
                    kv_place = 'host'
                else:
                    # Ensure the K/V activation is available on the target NPU before the store.
                    ready_kv = self._ready_time_for_device(g, nid, target_npu, phase_eff, commit)
                    start = max(float(self.avail.get(target_npu.name, 0.0)), float(ready_kv))

                    # Convert to ND for KV cache storage if needed.
                    src_fmt = str(self.cost.device_preferred_fmt(target_npu))
                    conv_cost = 0.0
                    if src_fmt != 'ND':
                        conv_cost = float(self.cost.format_conversion_time(int(size_nd), str(src_fmt), 'ND', target_npu))

                    # Model a bandwidth-limited store into the KV-cache region.
                    write_t = float(self.cost.mem_time(int(size_nd), target_npu))
                    finish = float(start + conv_cost + write_t)

                    logger.debug(
                        "[kv-write] node=%s target_npu=%s start=%.4f finish=%.4f", nid, target_npu.name, start, finish
                    )
                    if commit:
                        self.avail[target_npu.name] = float(finish)
                        self._node_finish_time[nid] = float(finish)
                        self._node_placement[nid] = str(target_npu.name)
                        self._node_out_fmt[nid] = "ND"
                        self._act_resident[(str(target_npu.name), str(nid))] = 0
                    return float(start), float(finish)

            # KV on host: convert on source device then send to host
            host = self.cost.get_host_device()
            ready_kv = self._ready_time_for_device(g, nid, dev, phase_eff, commit)
            conv_start = max(float(self.avail.get(dev.name, 0.0)), float(ready_kv))
            conv_cost = self.cost.format_conversion_time(size_nd, self.cost.device_preferred_fmt(dev), "ND", dev)
            _, l2e = self.comm.reserve(dev.name, host.name, size_nd, earliest=conv_start + conv_cost, commit=commit, tag="kv_write")
            finish = float(l2e)
            logger.debug(
                "[kv-write] node=%s dev=%s -> host start=%.4f finish=%.4f", nid, dev.name, conv_start, finish
            )
            if commit:
                self.avail[dev.name] = max(self.avail.get(dev.name, 0.0), finish)
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

                kv_node_id = str(getattr(node, 'id', getattr(node, 'nid', nid)) or nid)
                kv_layer = self._node_layer_id(node)
                kv_hr = self._node_kv_head_range(node)
                kv_h0 = kv_h1 = None
                if kv_hr is not None:
                    try:
                        kv_h0, kv_h1 = int(kv_hr[0]), int(kv_hr[1])
                    except Exception:
                        kv_h0 = kv_h1 = None

                def _kv_extra(*, kv_place_used: str, route: str, hop: str) -> dict:
                    ex = {
                        'payload': 'kv_cache',
                        'action': 'load',
                        'node_id': kv_node_id,
                        'op': str(getattr(node, 'name', '') or ''),
                        # QK consumes K-cache; SV consumes V-cache (for attention).
                        'kv_role': ('K' if str(getattr(node, 'name', '')).upper() == 'QK' else ('V' if str(getattr(node, 'name', '')).upper() == 'SV' else '')),
                        'kv_place': str(kv_place_used),
                        'bytes_nd': int(kv_bytes),
                        # Keep both naming conventions so downstream parsers can use either.
                        'from_fmt': 'ND',
                        'to_fmt': str(dev_fmt),
                        'src_fmt': 'ND',
                        'wire_fmt': 'ND',
                        'dst_fmt': str(dev_fmt),
                        'route': str(route),
                        'hop': str(hop),
                    }
                    if kv_layer is not None:
                        ex['layer'] = int(kv_layer)
                    if kv_h0 is not None:
                        ex['kv_head_start'] = int(kv_h0)
                    if kv_h1 is not None:
                        ex['kv_head_end'] = int(kv_h1)
                    return ex

                kv_place = self._kv_place(label)

                # KV in PIM (sharded by KV head). For a shard, KV should come from a single mapped PIM.
                if kv_place == 'pim':
                    src_pim = self._kv_pim_for_node(node)
                    if src_pim is None:
                        kv_place = 'host'
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
                                        src_pim.name,
                                        dev.name,
                                        int(size_nd),
                                        earliest=float(rd_end),
                                        commit=commit,
                                        tag="kv_load",
                                        extra=_kv_extra(kv_place_used='pim', route='direct', hop='pim_to_dst'),
                                    )
                                else:
                                    _, t1_end = self.comm.reserve(
                                        src_pim.name,
                                        host.name,
                                        int(size_nd),
                                        earliest=float(rd_end),
                                        commit=commit,
                                        tag="kv_load",
                                        extra=_kv_extra(kv_place_used='pim', route='host', hop='pim_to_host'),
                                    )
                                    _, xfer_end = self.comm.reserve(
                                        host.name,
                                        dev.name,
                                        int(size_nd),
                                        earliest=float(t1_end),
                                        commit=commit,
                                        tag="kv_load",
                                        extra=_kv_extra(kv_place_used='pim', route='host', hop='host_to_dst'),
                                    )
                                conv_t = float(self.cost.format_conversion_time(int(size_nd), "ND", dev_fmt, dev))
                                kv_ready = max(kv_ready, float(xfer_end) + conv_t)

                # KV fixed on an NPU (kv_place='npu'): load KV from that NPU.
                if kv_place == 'npu':
                    src_npu = self._kv_npu_device(label)
                    if src_npu is None:
                        kv_place = 'host'
                    else:
                        # Read from src NPU memory (serialize by NPU availability)
                        rd_t = float(self.cost.mem_time(int(size_nd), src_npu))
                        rd_start = max(float(self.avail.get(src_npu.name, 0.0)), float(ready))
                        rd_end = rd_start + rd_t
                        if commit:
                            self.avail[src_npu.name] = rd_end

                        if str(dev.name) == str(src_npu.name):
                            # Local read; include optional ND->dev_fmt conversion.
                            conv_t = float(self.cost.format_conversion_time(int(size_nd), "ND", dev_fmt, dev))
                            kv_ready = max(kv_ready, float(rd_end) + conv_t)
                        else:
                            # Transfer KV to target device (prefer direct, otherwise via host)
                            host = self.cost.get_host_device()
                            t_direct = float(self.cost.comm_cost(src_npu, dev, int(size_nd)))
                            if math.isfinite(t_direct):
                                _, xfer_end = self.comm.reserve(
                                    src_npu.name,
                                    dev.name,
                                    int(size_nd),
                                    earliest=float(rd_end),
                                    commit=commit,
                                    tag="kv_load",
                                    extra=_kv_extra(kv_place_used='npu', route='direct', hop='npu_to_dst'),
                                )
                            else:
                                _, t1_end = self.comm.reserve(
                                    src_npu.name,
                                    host.name,
                                    int(size_nd),
                                    earliest=float(rd_end),
                                    commit=commit,
                                    tag="kv_load",
                                    extra=_kv_extra(kv_place_used='npu', route='host', hop='npu_to_host'),
                                )
                                _, xfer_end = self.comm.reserve(
                                    host.name,
                                    dev.name,
                                    int(size_nd),
                                    earliest=float(t1_end),
                                    commit=commit,
                                    tag="kv_load",
                                    extra=_kv_extra(kv_place_used='npu', route='host', hop='host_to_dst'),
                                )
                            conv_t = float(self.cost.format_conversion_time(int(size_nd), "ND", dev_fmt, dev))
                            kv_ready = max(kv_ready, float(xfer_end) + conv_t)

                # KV on host (explicit or fallback)
                if kv_place == 'host':
                    host = self.cost.get_host_device()
                    l2s, _ = self.comm.reserve(
                        host.name,
                        dev.name,
                        size_nd,
                        earliest=float(ready),
                        commit=commit,
                        tag="kv_load",
                        extra=_kv_extra(kv_place_used='host', route='host', hop='host_to_dst'),
                    )
                    kv_ready = max(
                        kv_ready,
                        float(l2s)
                        + float(
                            self.cost.combine_transfer_and_convert(
                                host,
                                dev,
                                int(size_nd),
                                "ND",
                                str(self.cost.device_preferred_fmt(dev)),
                            )
                        ),
                    )

        #---------------------3. normal weight load + compute + activation handling
        start = max(float(self.avail.get(dev.name, 0.0)), kv_ready)
        weight_extra: Dict[str, Any] = {}
        if self._node_weight_id(node) and int(self._node_weight_size(node) or 0) > 0:
            _, _, weight_extra = self._weight_load_time(
                node,
                dev,
                start,
                commit,
                label=label,
                batch=int(batch),
                seq_len=int(seq_len),
                phase=str(phase_eff),
            )
            finish = start + float(weight_extra.get('total_s', 0.0) or 0.0)
        else:
            compute = self.cost.node_device_cost(node, dev, label, batch, seq_len, phase_eff)
            finish = start + float(compute)

        if not commit:
            if weight_extra:
                self._last_candidate_component_extra = {
                    "compute_s": float(weight_extra.get("compute_total_s", 0.0) or 0.0),
                    # End-to-end weight reload service.  This already accounts
                    # for configured load/load and load/compute overlap.
                    "reload_s": float(weight_extra.get("load_total_s", 0.0) or 0.0),
                    "weight_load_comm_s": float(weight_extra.get("load_comm_s", 0.0) or 0.0),
                    "cache_state": str(weight_extra.get("cache_state", "") or ""),
                }
            else:
                self._last_candidate_component_extra = {
                    "compute_s": float(max(0.0, float(finish) - float(start))),
                    "reload_s": 0.0,
                    "weight_load_comm_s": 0.0,
                    "cache_state": "not_applicable",
                }

        if commit:
            self.avail[dev.name] = finish
            self._node_finish_time[nid] = finish
            self._node_placement[nid] = dev.name
            self._node_out_fmt[nid] = self.cost.device_preferred_fmt(dev)
            out_read_nd, out_write_nd = self.cost.estimate_activation_bytes(node, self.batch, self.seq_len, phase)
            out_nd = max(int(out_write_nd), int(out_read_nd))

            # ---- Activation residency / spill policy ----
            # Pure 1-NPU (no PIM) prefill can otherwise retain too many activations
            # on the sole NPU, leaving no room for the next layer's weight load.
            # In that topology, prefer host spill over local activation residency.
            force_host_spill = False
            try:
                force_host_spill = (
                    str(getattr(dev, "type", "") or "").lower() == "npu"
                    and str(phase_eff or "").lower() == "prefill"
                    and len(self.cluster.devices_by_type("npu") or []) == 1
                    and len(self.cluster.devices_by_type("pim") or []) == 0
                )
            except Exception:
                force_host_spill = False

            keep_local = False
            if not force_host_spill:
                try:
                    keep_local = bool(
                        self.buffer.pim_reserve_activation(dev.name, out_nd, commit=False)
                    )
                except Exception:
                    keep_local = False

            if keep_local:
                ok = False
                try:
                    ok = bool(
                        self.buffer.pim_reserve_activation(dev.name, out_nd, commit=True)
                    )
                except Exception:
                    ok = False

                if ok:
                    self._act_resident[(dev.name, nid)] = out_nd
                else:
                    src_fmt = self.cost.device_preferred_fmt(dev)
                    self._ensure_host_store(nid, dev, out_nd, src_fmt, finish, commit=True)
            else:
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
                self.buffer.pim_release_activation(str(udev), bytes_kept, commit=True)

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
    def _weight_load_time(
        self,
        node: TaskNode,
        dev: DeviceSpec,
        earliest: float,
        commit: bool,
        *,
        label: PlanLabel,
        batch: int,
        seq_len: int,
        phase: str,
    ) -> Tuple[float, float, Dict[str, Any]]:
        wid = self._node_weight_id(node)
        if not wid:
            return (0.0, 0.0, {})
        wsize_nd = int(self._node_weight_size(node) or 0)
        if wsize_nd <= 0:
            return (0.0, 0.0, {})

        self._record_weight_proto_node(node)
        src_storage_fmt = self._weight_storage_format_for_wid(wid)
        resident_fmt = self._weight_resident_format(dev, str(src_storage_fmt))
        dev_type = str(getattr(dev, 'type', '') or '').lower()

        if dev_type == 'pim' and self._weights_preloaded_on_pim():
            if commit:
                try:
                    self._weight_sizes[wid] = int(wsize_nd)
                except Exception:
                    pass
            prof = self._weight_service_profile_no_contention(
                node,
                dev,
                src_storage_fmt=str(src_storage_fmt),
                cached=True,
                cached_fmt='PIM-OPT',
                label=label,
                batch=int(batch),
                seq_len=int(seq_len),
                phase=str(phase),
            )
            prof['cache_state'] = 'preloaded'
            return (0.0, 0.0, prof)

        cached_nd, cached_fmt, cache = self._cached_weight_state(dev, wid, wsize_nd)

        if cached_nd >= wsize_nd:
            if commit and cache is not None:
                try:
                    cache.touch(wid)
                except Exception:
                    pass
            prof = self._weight_service_profile_no_contention(
                node,
                dev,
                src_storage_fmt=str(src_storage_fmt),
                cached=True,
                cached_fmt=str(cached_fmt or resident_fmt),
                label=label,
                batch=int(batch),
                seq_len=int(seq_len),
                phase=str(phase),
            )
            return (0.0, 0.0, prof)

        if cached_nd > 0:
            raise RuntimeError(
                f"Partial weight caching is not modeled for weight_id='{wid}' on device='{dev.name}'. "
                f"cached_nd={cached_nd} full_nd={wsize_nd}"
            )

        try:
            cache_feasible = bool(self.buffer.can_cache_weight(
                dev.name, wid, int(wsize_nd), pinned=False, fmt=str(resident_fmt)
            ))
        except Exception as e:
            raise RuntimeError(
                f"can_cache_weight crashed for weight_id='{wid}' on device='{dev.name}'"
            ) from e

        if not cache_feasible:
            cache_obj = getattr(getattr(self, 'buffer', None), 'device_cache', {}).get(dev.name, None)
            cap_bytes = int(getattr(cache_obj, 'capacity', 0) or 0) if cache_obj is not None else 0
            phy_bytes, kv_used_bytes, act_used_bytes, weight_used_bytes, total_used_bytes = (0, 0, 0, 0, 0)
            try:
                phy_bytes, kv_used_bytes, act_used_bytes, weight_used_bytes, total_used_bytes = self.buffer.pim_used_bytes(dev.name)
            except Exception:
                pass
            prof = {
                'wid': str(wid),
                'weight_size_nd': int(wsize_nd),
                'host_storage_fmt': str(src_storage_fmt),
                'host_src_fmt': str(self._weight_host_source_format(dev, str(src_storage_fmt))),
                'resident_fmt': str(resident_fmt),
                'cache_state': 'infeasible',
                'cache_capacity_bytes': int(cap_bytes),
                'runtime_phy_bytes': int(phy_bytes),
                'runtime_kv_used_bytes': int(kv_used_bytes),
                'runtime_act_used_bytes': int(act_used_bytes),
                'runtime_weight_used_bytes': int(weight_used_bytes),
                'runtime_total_used_bytes': int(total_used_bytes),
                'total_s': float('inf'),
            }
            if commit:
                raise RuntimeError(
                    f"Infeasible to cache weight_id='{wid}' on device='{dev.name}': "
                    f"weight_size_nd={int(wsize_nd)}, cache_capacity_bytes={int(cap_bytes)}, "
                    f"kv_used_bytes={int(kv_used_bytes)}, act_used_bytes={int(act_used_bytes)}, "
                    f"weight_used_bytes={int(weight_used_bytes)}, runtime_total_used_bytes={int(total_used_bytes)}."
                )
            return (float('inf'), float('inf'), prof)

        host = self.cost.get_host_device()
        host_src_fmt = self._weight_host_source_format(dev, str(src_storage_fmt))
        rd_bytes = int(self.cost.weight_transfer_comm_bytes(int(wsize_nd), str(src_storage_fmt), dev_or_type=dev))
        l1s, l1e = self.comm.reserve(
            host.name,
            dev.name,
            rd_bytes,
            earliest=float(earliest),
            commit=commit,
            tag='weight_load',
            extra={
                'payload': 'weight',
                'action': 'load',
                'weight_id': str(wid),
                'node_id': str(getattr(node, 'id', getattr(node, 'nid', '')) or ''),
                'op': str(getattr(node, 'name', '') or ''),
                'bytes_nd': int(wsize_nd),
                'bytes_full_nd': int(wsize_nd),
                'cached_before_nd': 0,
                'from_fmt': str(host_src_fmt),
                'to_fmt': str(resident_fmt),
                'cache_capacity_bytes': int(getattr(getattr(self.buffer, 'device_cache', {}).get(dev.name, None), 'capacity', 0) or 0),
            },
        )
        queue_wait_s = max(0.0, float(l1s) - float(earliest))
        l1_s = max(0.0, float(l1e) - float(l1s))

        if dev_type == 'pim':
            l2_s = float(self.cost.pim_local_weight_load_time(int(wsize_nd), str(host_src_fmt), dev=dev))
            l2_write_only_s = float(self.cost.pim_local_weight_write_only_time(int(wsize_nd), dev=dev))
            l2_pack_only_est_s = float(self.cost.pim_local_weight_pack_only_est_time(int(wsize_nd), str(host_src_fmt), dev=dev))
            load_join = overlap_time(float(l1_s), float(l2_s), self._pim_weight_load_overlap_ratio())
        elif dev_type == 'cpu':
            compute_fmt = str(self.cost.device_preferred_fmt(dev))
            l2_s = float(self.cost.format_conversion_time(int(rd_bytes), str(host_src_fmt), str(compute_fmt), dev)) if str(host_src_fmt) != str(compute_fmt) else 0.0
            l2_write_only_s = 0.0
            l2_pack_only_est_s = 0.0
            load_join = overlap_time(float(l1_s), float(l2_s), 0.0)
        else:
            l2_s = 0.0
            l2_write_only_s = 0.0
            l2_pack_only_est_s = 0.0
            load_join = overlap_time(float(l1_s), 0.0, 0.0)

        compute_prof = self._weight_compute_stage_profile(
            node, dev, label, int(batch), int(seq_len), str(phase), resident_fmt=str(resident_fmt)
        )
        lc = overlap_time(float(load_join.total_s), float(compute_prof['compute_total_s']), self._weight_load_compute_overlap_ratio())
        prof = {
            'wid': str(wid),
            'weight_size_nd': int(wsize_nd),
            'host_storage_fmt': str(src_storage_fmt),
            'host_src_fmt': str(host_src_fmt),
            'resident_fmt': str(resident_fmt),
            'compute_fmt': str(compute_prof['compute_fmt']),
            'cache_state': 'miss',
            'queue_wait_s': float(queue_wait_s),
            'load_active_s': float(load_join.total_s),
            'load_total_s': float(queue_wait_s + load_join.total_s),
            'load_comm_s': float(l1_s),
            'load_l1_s': float(l1_s),
            'load_l2_s': float(l2_s),
            'load_l2_write_only_s': float(l2_write_only_s),
            'load_l2_pack_only_est_s': float(l2_pack_only_est_s),
            'load_l1_l2_overlap_ratio': float(load_join.overlap_ratio if dev_type == 'pim' else 0.0),
            'compute_total_s': float(compute_prof['compute_total_s']),
            'compute_backend': str(compute_prof['compute_backend']),
            'compute_rule': str(compute_prof['compute_rule']),
            'b1_s': float(compute_prof['b1_s']),
            'b2_s': float(compute_prof['b2_s']),
            'launch_overhead_s': float(compute_prof['launch_overhead_s']),
            'lc_overlap_ratio': float(lc.overlap_ratio),
            'lc_overlap_saved_s': float(lc.saved_s),
            'total_s': float(queue_wait_s + lc.total_s),
        }

        if commit:
            try:
                self._weight_load_count[wid, dev.type] += 1
                self._weight_sizes[wid] = int(wsize_nd)
            except Exception:
                pass
            ok = bool(self.buffer.mark_cached(dev.name, wid, int(wsize_nd), pinned=False, fmt=str(resident_fmt)))
            if not ok:
                raise RuntimeError(
                    f"Failed to cache weight_id='{wid}' on device='{dev.name}' after an explicit miss load."
                )

        return (float(queue_wait_s), float(load_join.total_s), prof)


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

class SchedulerBase(
    SchedulerBaseCoreMixin,
    SchedulerBaseHelperMixin,
    SchedulerBaseTimingMixin,
):
    """Common scheduler functionality shared by concrete strategy classes."""


__all__ = ["SchedulerBase"]
