from __future__ import annotations
from scheduler_common import *

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


