from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class TaskNode:
    id: str
    name: str
    flops: float = 0.0
    bytes_read: float = 0.0
    bytes_write: float = 0.0
    weight_id: Optional[str] = None
    weight_size: int = 0
    allowed: Dict[str, bool] = field(default_factory=lambda: {"cpu": True, "npu": True, "pim": True})
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
