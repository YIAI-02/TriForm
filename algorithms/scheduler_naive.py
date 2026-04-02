from __future__ import annotations
from scheduler_common import *
from scheduler_types import _GraphIndex, ScheduledTask
from scheduler_base import SchedulerBase

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
                        self._log_scheduled_op_trace(
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
                    self._log_scheduled_op_trace(
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

