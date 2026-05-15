from __future__ import annotations

from .scheduler_common import *
from .scheduler_types import ScheduledTask
from .scheduler_base import SchedulerBase


@dataclass(frozen=True)
class MCTSAction:
    node_id: str
    device_name: str


@dataclass(frozen=True)
class MCTSState:
    schedule: tuple[ScheduledTask, ...]
    scheduled: frozenset[str]
    ready: frozenset[str]
    remaining_preds: Mapping[str, int]
    avail: Mapping[str, float]
    weight_cache_snapshot: Mapping[str, LRUCache]
    comm_timeline_snapshot: Mapping[Tuple[str, str], float]
    runtime_snapshot: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class MCTSNode:
    state: MCTSState
    parent: Optional["MCTSNode"] = None
    action: Optional[MCTSAction] = None
    children: Dict[MCTSAction, "MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count <= 0:
            return 0.0
        return float(self.value_sum) / float(self.visit_count)


class MCTSScheduler(SchedulerBase):
    """Monte-Carlo tree-search scheduler using SchedulerBase timing semantics."""

    def __init__(
        self,
        cluster: Cluster,
        cost: CostModel,
        label: PlanLabel,
        batch: int,
        seq_len: int,
        buffer: GlobalMemoryManager,
        *,
        mcts_iterations: Optional[int] = None,
        rollout_depth: Optional[int] = None,
        exploration_c: float = 1.41421356237,
        seed: int = 0,
    ):
        super().__init__(cluster, cost, label, batch, seq_len, buffer)
        self.mcts_iterations = int(mcts_iterations if mcts_iterations is not None else os.getenv("SCHED_MCTS_ITERATIONS", "64"))
        self.rollout_depth = int(rollout_depth if rollout_depth is not None else os.getenv("SCHED_MCTS_ROLLOUT_DEPTH", "8"))
        self.exploration_c = float(exploration_c)
        self._rng = random.Random(int(seed))
        self._mcts_topo_pos: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------
    def _capture_runtime_snapshot(self) -> Dict[str, Any]:
        return {
            "node_finish_time": copy.deepcopy(self._node_finish_time),
            "node_placement": copy.deepcopy(self._node_placement),
            "node_out_fmt": copy.deepcopy(self._node_out_fmt),
            "node_host_store_end": copy.deepcopy(self._node_host_store_end),
            "runtime_cap": copy.deepcopy(self._runtime_cap),
            "act_used": copy.deepcopy(self._act_used),
            "act_resident": copy.deepcopy(self._act_resident),
            "act_refcnt": copy.deepcopy(self._act_refcnt),
            "weight_cached": copy.deepcopy(self.weight_cached),
            "weight_load_count": copy.deepcopy(self._weight_load_count),
            "weight_sizes": copy.deepcopy(self._weight_sizes),
            "weight_proto_node": copy.deepcopy(self._weight_proto_node),
            "pim_weight_desc_cache": copy.deepcopy(self._pim_weight_desc_cache),
            "last_op_trace_extra": copy.deepcopy(self._last_op_trace_extra),
            "collective_output_devs": copy.deepcopy(getattr(self, "_collective_output_devs", {})),
            "pim_state": copy.deepcopy(getattr(self.buffer, "pim_state", {})),
        }

    def _capture_state(
        self,
        schedule: Iterable[ScheduledTask],
        scheduled: Iterable[str],
        ready: Iterable[str],
        remaining_preds: Mapping[str, int],
    ) -> MCTSState:
        return MCTSState(
            schedule=tuple(schedule),
            scheduled=frozenset(str(n) for n in scheduled),
            ready=frozenset(str(n) for n in ready),
            remaining_preds=dict(remaining_preds),
            avail=dict(self.avail),
            weight_cache_snapshot=copy.deepcopy(getattr(self.buffer, "device_cache", {})),
            comm_timeline_snapshot=copy.deepcopy(getattr(self.comm, "timeline_end", {})),
            runtime_snapshot=self._capture_runtime_snapshot(),
        )

    def _restore_from_state(self, state: MCTSState) -> None:
        self.avail = dict(state.avail)
        self.buffer.device_cache = copy.deepcopy(dict(state.weight_cache_snapshot))
        self.comm.timeline_end = copy.deepcopy(dict(state.comm_timeline_snapshot))

        runtime = dict(state.runtime_snapshot or {})
        self._node_finish_time = copy.deepcopy(runtime.get("node_finish_time", {}))
        self._node_placement = copy.deepcopy(runtime.get("node_placement", {}))
        self._node_out_fmt = copy.deepcopy(runtime.get("node_out_fmt", {}))
        self._node_host_store_end = copy.deepcopy(runtime.get("node_host_store_end", {}))
        self._runtime_cap = copy.deepcopy(runtime.get("runtime_cap", self._runtime_cap))
        self._act_used = defaultdict(int, copy.deepcopy(runtime.get("act_used", {})))
        self._act_resident = copy.deepcopy(runtime.get("act_resident", {}))
        self._act_refcnt = copy.deepcopy(runtime.get("act_refcnt", {}))
        self.weight_cached = copy.deepcopy(runtime.get("weight_cached", {}))
        self._weight_load_count = defaultdict(int, copy.deepcopy(runtime.get("weight_load_count", {})))
        self._weight_sizes = copy.deepcopy(runtime.get("weight_sizes", {}))
        self._weight_proto_node = copy.deepcopy(runtime.get("weight_proto_node", {}))
        self._pim_weight_desc_cache = copy.deepcopy(runtime.get("pim_weight_desc_cache", {}))
        self._last_op_trace_extra = copy.deepcopy(runtime.get("last_op_trace_extra", {}))
        self._collective_output_devs = copy.deepcopy(runtime.get("collective_output_devs", {}))
        if "pim_state" in runtime:
            self.buffer.pim_state = copy.deepcopy(runtime.get("pim_state", {}))

    # ------------------------------------------------------------------
    # Action generation and state transitions
    # ------------------------------------------------------------------
    def _legal_actions(self, g: TaskGraph, state: MCTSState) -> List[MCTSAction]:
        actions: List[MCTSAction] = []
        for nid in sorted(state.ready, key=lambda n: self._mcts_topo_pos.get(n, 0)):
            node = g.nodes[nid]
            if self._is_comm_node(node):
                actions.append(MCTSAction(nid, "COMM"))
                continue

            name_up = str(getattr(node, "name", "")).upper()
            is_kv_write = name_up in ("K_WRITE", "V_WRITE", "KV_WRITE")
            pinned_dev: Optional[DeviceSpec] = None
            if is_kv_write:
                pinned_dev = self._preferred_kv_write_device(g, nid)
                if pinned_dev is not None and not self._node_allowed_on(node, pinned_dev):
                    pinned_dev = None

            if pinned_dev is not None:
                actions.append(MCTSAction(nid, str(pinned_dev.name)))
                continue

            for dev_type in self._executor_device_types():
                for dev in self.cluster.devices_by_type(dev_type):
                    if self._node_allowed_on(node, dev):
                        actions.append(MCTSAction(nid, str(dev.name)))
        return actions

    def _action_device(self, action: MCTSAction, node: TaskNode) -> DeviceSpec:
        if action.device_name == "COMM" or self._is_comm_node(node):
            return self.cost.get_host_device()
        dev = self.cluster.devices.get(str(action.device_name))
        if dev is None:
            raise RuntimeError(f"Unknown device for MCTS action: {action.device_name}")
        return dev

    def _apply_action_to_state(
        self,
        g: TaskGraph,
        phase: str,
        state: MCTSState,
        action: MCTSAction,
    ) -> MCTSState:
        self._restore_from_state(state)
        node = g.nodes[action.node_id]
        dev = self._action_device(action, node)
        start, finish = self._earliest_finish_on_device(g, action.node_id, dev, self.label, phase, commit=True)

        trace_dev = "COMM" if self._is_comm_node(node) else str(dev.name)
        new_schedule = tuple(state.schedule) + (ScheduledTask(action.node_id, trace_dev, start, finish),)
        new_scheduled = set(state.scheduled)
        new_scheduled.add(action.node_id)

        self._after_commit_consume_predecessors(g, action.node_id)

        remaining = dict(state.remaining_preds)
        ready = set(state.ready)
        ready.discard(action.node_id)
        idx = self._get_graph_index(g)
        for succ in idx.succs.get(action.node_id, ()):
            remaining[succ] = int(remaining.get(succ, 0)) - 1
            if remaining[succ] == 0 and succ not in new_scheduled:
                ready.add(succ)

        return self._capture_state(new_schedule, new_scheduled, ready, remaining)

    # ------------------------------------------------------------------
    # MCTS core
    # ------------------------------------------------------------------
    def _is_terminal(self, idx: _GraphIndex, state: MCTSState) -> bool:
        return len(state.scheduled) >= len(idx.nodes) or not state.ready

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        log_parent = math.log(max(1, node.visit_count))

        def score(child: MCTSNode) -> float:
            if child.visit_count <= 0:
                return float("inf")
            exploitation = child.q_value
            exploration = self.exploration_c * float(child.prior or 1.0) * math.sqrt(log_parent / child.visit_count)
            return float(exploitation + exploration)

        return max(node.children.values(), key=score)

    def _rollout(self, g: TaskGraph, phase: str, idx: _GraphIndex, state: MCTSState) -> float:
        cur = state
        depth = 0
        while not self._is_terminal(idx, cur) and depth < self.rollout_depth:
            actions = self._legal_actions(g, cur)
            if not actions:
                break
            if self._rng.random() < 0.85:
                action = min(actions, key=lambda a: self._one_step_finish(g, phase, cur, a))
            else:
                action = self._rng.choice(actions)
            cur = self._apply_action_to_state(g, phase, cur, action)
            depth += 1

        makespan = max((float(t.finish) for t in cur.schedule), default=0.0)
        if cur.avail:
            makespan = max(makespan, max(float(v) for v in cur.avail.values()))
        remaining_penalty = max(0, len(idx.nodes) - len(cur.scheduled)) * 1e-9
        return -float(makespan + remaining_penalty)

    def _one_step_finish(self, g: TaskGraph, phase: str, state: MCTSState, action: MCTSAction) -> float:
        nxt = self._apply_action_to_state(g, phase, state, action)
        if not nxt.schedule:
            return float("inf")
        return float(nxt.schedule[-1].finish)

    def _search(self, g: TaskGraph, phase: str, root_state: MCTSState, idx: _GraphIndex) -> MCTSNode:
        root = MCTSNode(state=root_state)
        iterations = max(1, int(self.mcts_iterations))
        for _ in range(iterations):
            node = root
            path = [node]

            while not self._is_terminal(idx, node.state):
                legal = self._legal_actions(g, node.state)
                unexpanded = [a for a in legal if a not in node.children]
                if unexpanded:
                    action = self._rng.choice(unexpanded)
                    child_state = self._apply_action_to_state(g, phase, node.state, action)
                    prior = 1.0 / max(1, len(legal))
                    child = MCTSNode(state=child_state, parent=node, action=action, prior=prior)
                    node.children[action] = child
                    node = child
                    path.append(node)
                    break
                if not node.children:
                    break
                node = self._select_child(node)
                path.append(node)

            reward = self._rollout(g, phase, idx, node.state)
            for item in path:
                item.visit_count += 1
                item.value_sum += reward
        return root

    def _choose_root_action(self, root: MCTSNode) -> MCTSAction:
        if not root.children:
            raise RuntimeError("MCTS search produced no legal root actions")
        best_child = max(
            root.children.values(),
            key=lambda c: (int(c.visit_count), float(c.q_value), -self._mcts_topo_pos.get(c.action.node_id if c.action else "", 0)),
        )
        if best_child.action is None:
            raise RuntimeError("MCTS best child has no action")
        return best_child.action

    # ------------------------------------------------------------------
    # Public schedule loop
    # ------------------------------------------------------------------
    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        if getattr(self, "stats", None):
            self.stats.set_phase(phase)

        self.reset_state(clear_caches=False)
        idx = self._get_graph_index(g)
        self._mcts_topo_pos = {nid: i for i, nid in enumerate(idx.topo)}

        remaining_preds: Dict[str, int] = {nid: len(idx.preds[nid]) for nid in idx.nodes}
        ready: set[str] = {nid for nid in idx.nodes if remaining_preds[nid] == 0}
        scheduled: set[str] = set()
        schedule: List[ScheduledTask] = []

        while ready:
            root_state = self._capture_state(schedule, scheduled, ready, remaining_preds)

            saved_stats = self.stats
            saved_comm_stats = self.comm.stats
            try:
                self.stats = None
                self.comm.stats = None
                root = self._search(g, phase, root_state, idx)
            finally:
                self.stats = saved_stats
                self.comm.stats = saved_comm_stats
                self._restore_from_state(root_state)

            action = self._choose_root_action(root)
            node = g.nodes[action.node_id]
            dev = self._action_device(action, node)
            start, finish = self._earliest_finish_on_device(g, action.node_id, dev, self.label, phase, commit=True)

            trace_dev = "COMM" if self._is_comm_node(node) else str(dev.name)
            schedule.append(ScheduledTask(action.node_id, trace_dev, start, finish))
            scheduled.add(action.node_id)
            self._after_commit_consume_predecessors(g, action.node_id)

            if getattr(self, "stats", None):
                op_name = node.attrs.get("op") or node.name
                try:
                    self._log_scheduled_op_trace(
                        nid=action.node_id,
                        op=op_name,
                        device=trace_dev,
                        device_type="comm" if trace_dev == "COMM" else dev.type,
                        start=float(start),
                        end=float(finish),
                        mode="COMM" if trace_dev == "COMM" else str(dev.type).upper(),
                    )
                except Exception:
                    pass

            ready.discard(action.node_id)
            for succ in idx.succs.get(action.node_id, ()):
                remaining_preds[succ] -= 1
                if remaining_preds[succ] == 0:
                    ready.add(succ)

        if len(scheduled) != len(idx.nodes):
            missing = [n for n in idx.nodes if n not in scheduled]
            raise RuntimeError(
                f"Schedule failed: graph may have cycles or missing deps; unscheduled nodes: {missing[:16]}"
            )
        return schedule


__all__ = ["MCTSAction", "MCTSState", "MCTSNode", "MCTSScheduler"]
