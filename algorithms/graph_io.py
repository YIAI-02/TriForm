from __future__ import annotations
from typing import Dict, Any
from task_graph import TaskGraph, TaskNode

def load_graph_from_json(obj: Dict[str, Any]) -> TaskGraph:
    g = TaskGraph()
    for nd in obj["nodes"]:
        n = TaskNode(
            id=nd["id"],
            name=nd.get("name", nd["id"]),
            work=float(nd["work"]),
            out_size=int(nd.get("out_size", 0)),
            weight_id=nd.get("weight_id"),
            weight_size=int(nd.get("weight_size", 0)),
            allowed=tuple(nd.get("allowed", [True, True])),
            attrs=nd.get("attrs", {})
        )
        g.add_node(n)
    for u, v in obj["edges"]:
        g.add_edge(u, v)
    return g
