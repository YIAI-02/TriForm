from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Any, Iterable, Tuple

@dataclass
class TaskNode:
    id: str
    name: str
    flops: float = 0.0
    bytes_read: float = 0.0
    bytes_write: float = 0.0
    weight_id: Optional[str] = None
    weight_size: int = 0
    allowed: Dict[str, bool] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)

class TaskGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, TaskNode] = {}
        self.succ: Dict[str, List[str]] = {}
        self.pred: Dict[str, List[str]] = {}

    def add_node(self, node: TaskNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        self.nodes[node.id] = node
        self.succ.setdefault(node.id, [])
        self.pred.setdefault(node.id, [])

    def add_edge(self, u: str, v: str) -> None:
        if u not in self.nodes or v not in self.nodes:
            raise KeyError(f"Cannot add edge {u}->{v}, node missing")
        self.succ[u].append(v)
        self.pred[v].append(u)

    def predecessors(self, nid: str) -> List[str]:
        return self.pred.get(nid, [])

    def successors(self, nid: str) -> List[str]:
        return self.succ.get(nid, [])

    def topological(self):
        indeg = {nid: 0 for nid in self.nodes}
        for v in self.nodes:
            for u in self.pred.get(v, []):
                indeg[v] += 1
        q = [nid for nid,d in indeg.items() if d == 0]
        out = []
        while q:
            nid = q.pop(0)
            out.append(nid)
            for w in self.succ.get(nid, []):
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)
        if len(out) != len(self.nodes):
            raise RuntimeError(f"Cycle detected: produced {len(out)} / {len(self.nodes)} nodes in topo order")
        return out

# =========================
# Joint Graph Scheduler
# =========================
@dataclass(frozen=True)
class JointNodeMeta:
    """Metadata for a node in the joint expanded DAG."""
    base_id: str
    phase: str                  # 'prefill' or 'decode'
    step: int                   # decode token index (0 for prefill)
    seq_len: int                # effective seq_len for cost model
    token_idx: int | None = None


class JointTaskGraph:
    """
    A lightweight DAG wrapper that mimics TaskGraph's interface (nodes/preds/succs/topological),
    but allows us to "stack" multiple passes (prefill + sampled decode steps) into one joint graph.
    """

    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}
        self._preds: Dict[str, List[str]] = defaultdict(list)
        self._succs: Dict[str, List[str]] = defaultdict(list)
        self.meta: Dict[str, JointNodeMeta] = {}
        self.barrier_edges: set[Tuple[str, str]] = set()
        self._topo_cache: Optional[Tuple[str, ...]] = None

    def add_node(self, nid: str, node: TaskNode, meta: JointNodeMeta) -> None:
        self.nodes[nid] = node
        self.meta[nid] = meta
        self._preds.setdefault(nid, [])
        self._succs.setdefault(nid, [])
        self._topo_cache = None

    def add_edge(self, u: str, v: str, *, barrier: bool = False) -> None:
        self._succs[u].append(v)
        self._preds[v].append(u)
        if barrier:
            self.barrier_edges.add((u, v))
        self._topo_cache = None

    def predecessors(self, nid: str) -> Iterable[str]:
        return self._preds.get(nid, ())

    def successors(self, nid: str) -> Iterable[str]:
        return self._succs.get(nid, ())

    def topological(self) -> List[str]:
        if self._topo_cache is not None:
            return list(self._topo_cache)

        indeg: Dict[str, int] = {n: 0 for n in self.nodes}
        for u, vs in self._succs.items():
            for v in vs:
                indeg[v] = indeg.get(v, 0) + 1

        q: List[str] = [n for n, d in indeg.items() if d == 0]
        out: List[str] = []
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            out.append(u)
            for v in self._succs.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if len(out) != len(self.nodes):
            raise ValueError("JointTaskGraph contains a cycle or disconnected nodes")
        self._topo_cache = tuple(out)
        return out
    
    def build_joint_graph_positions(
        self,
        g_prefill: TaskGraph,
        g_decode: Optional[TaskGraph],
        prefill_len: int,
        decode_positions: Iterable[int],
        decode_seq_mode: str = "context",
    ) -> JointTaskGraph:
        """
        Build a joint graph with explicit sampled decode positions (token indices).
        Node IDs:
          - Prefill:  P::<base_id>
          - Decode t: D{t}::<base_id>
        """
        def _find_sources_sinks(g: TaskGraph) -> Tuple[List[str], List[str]]:
            nodes_iter = g.nodes.keys() if hasattr(g.nodes, "keys") else g.nodes
            nodes = list(nodes_iter)
            preds = {nid: tuple(g.predecessors(nid)) for nid in nodes}
            succs = {nid: tuple(g.successors(nid)) for nid in nodes}
            srcs = [nid for nid in nodes if len(preds[nid]) == 0]
            snks = [nid for nid in nodes if len(succs[nid]) == 0]
            return srcs, snks

        joint = JointTaskGraph()

        # 1) Copy prefill graph
        srcsP, snksP = _find_sources_sinks(g_prefill)
        for base_id, node in g_prefill.nodes.items():
            meta = JointNodeMeta(
                base_id=str(base_id),
                phase="prefill",
                step=0,
                seq_len=int(prefill_len),
                token_idx=None,
            )
            joint.add_node(f"P::{base_id}", node, meta)
        for u in g_prefill.nodes.keys():
            for v in g_prefill.successors(u):
                joint.add_edge(f"P::{u}", f"P::{v}", barrier=False)

        if g_decode is None:
            g_decode = g_prefill

        srcsD, snksD = _find_sources_sinks(g_decode)

        positions = sorted(set(int(t) for t in decode_positions if int(t) >= 0))
        if not positions:
            return joint

        first = positions[0]

        # Prefill -> first decode cross edges.
        if decode_seq_mode == "context":
            # treat as real activation dependency
            for p_snk in snksP:
                for d_src in srcsD:
                    joint.add_edge(f"P::{p_snk}", f"D{first}::{d_src}", barrier=False)
        else:
            # purely ordering (e.g., KV cache already modeled), no activation payload
            for p_snk in snksP:
                for d_src in srcsD:
                    joint.add_edge(f"P::{p_snk}", f"D{first}::{d_src}", barrier=True)

        # 2) Copy decode graph for each sampled position.
        for idx, t in enumerate(positions):
            if decode_seq_mode == "one":
                seq_len = 1
            elif decode_seq_mode == "full":
                seq_len = int(prefill_len + t)
            else:
                seq_len = int(prefill_len)

            for base_id, node in g_decode.nodes.items():
                jid = f"D{t}::{base_id}"
                meta = JointNodeMeta(
                    base_id=str(base_id),
                    phase="decode",
                    step=int(t),
                    seq_len=int(seq_len),
                    token_idx=int(t),
                )
                joint.add_node(jid, node, meta)

            for u in g_decode.nodes.keys():
                for v in g_decode.successors(u):
                    joint.add_edge(f"D{t}::{u}", f"D{t}::{v}", barrier=False)

            # Cross-step ordering edges (barrier).
            if idx > 0:
                prev_t = positions[idx - 1]
                for d_snk in snksD:
                    for d_src in srcsD:
                        joint.add_edge(f"D{prev_t}::{d_snk}", f"D{t}::{d_src}", barrier=True)

        return joint

    def _build_joint_graph_positions(
        self,
        g_prefill: TaskGraph,
        g_decode: Optional[TaskGraph],
        prefill_len: int,
        decode_positions: Iterable[int],
        decode_seq_mode: str = "context",
    ) -> JointTaskGraph:
        """
        Build a joint graph with explicit sampled decode positions (token indices).
        Node IDs:
          - Prefill:  P::<base_id>
          - Decode t: D{t}::<base_id>
        """
        def _find_sources_sinks(g: TaskGraph) -> Tuple[List[str], List[str]]:
            nodes_iter = g.nodes.keys() if hasattr(g.nodes, "keys") else g.nodes
            nodes = list(nodes_iter)
            preds = {nid: tuple(g.predecessors(nid)) for nid in nodes}
            succs = {nid: tuple(g.successors(nid)) for nid in nodes}
            srcs = [nid for nid in nodes if len(preds[nid]) == 0]
            snks = [nid for nid in nodes if len(succs[nid]) == 0]
            return srcs, snks

        joint = JointTaskGraph()

        # 1) 复制 prefill 图
        srcsP, snksP = _find_sources_sinks(g_prefill)
        for base_id, node in g_prefill.nodes.items():
            meta = JointNodeMeta(
                base_id=str(base_id),
                phase="prefill",
                step=0,
                seq_len=int(prefill_len),
                token_idx=None,
            )
            joint.add_node(f"P::{base_id}", node, meta)
        for u in g_prefill.nodes.keys():
            for v in g_prefill.successors(u):
                joint.add_edge(f"P::{u}", f"P::{v}", barrier=False)

        # 2) decode 图（缺省就复用 prefill 图）
        if g_decode is None:
            g_decode = g_prefill
        srcsD, snksD = _find_sources_sinks(g_decode)

        positions = sorted({int(t) for t in decode_positions if int(t) >= 0})
        if not positions:
            return joint

        first = positions[0]
        if decode_seq_mode == "context":
            for p_snk in snksP:
                for d_src in srcsD:
                    joint.add_edge(f"P::{p_snk}", f"D{first}::{d_src}", barrier=False)
        else:
            for p_snk in snksP:
                for d_src in srcsD:
                    joint.add_edge(f"P::{p_snk}", f"D{first}::{d_src}", barrier=True)

        for idx, t in enumerate(positions):
            if decode_seq_mode == "one":
                seq_len = 1
            elif decode_seq_mode == "full":
                seq_len = int(prefill_len + t)
            else:
                seq_len = int(prefill_len)

            # 节点
            for base_id, node in g_decode.nodes.items():
                jid = f"D{t}::{base_id}"
                meta = JointNodeMeta(
                    base_id=str(base_id),
                    phase="decode",
                    step=int(t),
                    seq_len=int(seq_len),
                    token_idx=int(t),
                )
                joint.add_node(jid, node, meta)

            # 同一 step 内的边
            for u in g_decode.nodes.keys():
                for v in g_decode.successors(u):
                    joint.add_edge(f"D{t}::{u}", f"D{t}::{v}", barrier=False)

            # 不同步 token step 之间的顺序边（barrier=True，不算通信）
            if idx > 0:
                prev_t = positions[idx - 1]
                for d_snk in snksD:
                    for d_src in srcsD:
                        joint.add_edge(f"D{prev_t}::{d_snk}", f"D{t}::{d_src}", barrier=True)

        return joint