from __future__ import annotations
import logging, heapq
from typing import Iterable, List, Dict

def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("heft")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger

logger = setup_logger()

def topological_sort(nodes: Iterable[str], edges: Dict[str, List[str]]) -> List[str]:
    indeg = {n: 0 for n in nodes}
    for u, outs in edges.items():
        for v in outs:
            indeg[v] += 1
    q = [n for n, d in indeg.items() if d == 0]
    heapq.heapify(q)
    order = []
    while q:
        u = heapq.heappop(q)
        order.append(u)
        for v in edges.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(q, v)
    if len(order) != len(list(nodes)):
        raise ValueError("Graph has cycles or disconnected nodes.")
    return order
