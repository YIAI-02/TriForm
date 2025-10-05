from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class TaskNode:
    id: str
    name: str
    work: float
    out_size: int
    weight_id: Optional[str]
    weight_size: int = 0
    allowed: Tuple[bool, bool] = (True, True)
    attrs: Dict = field(default_factory=dict)

class TaskGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, TaskNode] = {}
        self.succ: Dict[str, List[str]] = {}
        self.pred: Dict[str, List[str]] = {}

    def add_node(self, node: TaskNode) -> None:
        self.nodes[node.id] = node
        self.succ.setdefault(node.id, [])
        self.pred.setdefault(node.id, [])

    def add_edge(self, u: str, v: str) -> None:
        assert u in self.nodes and v in self.nodes
        self.succ[u].append(v)
        self.pred[v].append(u)

    def predecessors(self, node_id: str) -> List[str]:
        return self.pred.get(node_id, [])

    def successors(self, node_id: str) -> List[str]:
        return self.succ.get(node_id, [])

    def weights(self) -> Dict[str, int]:
        w: Dict[str, int] = {}
        for n in self.nodes.values():
            if n.weight_id:
                w[n.weight_id] = max(w.get(n.weight_id, 0), n.weight_size)
        return w
