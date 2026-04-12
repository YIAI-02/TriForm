from __future__ import annotations
from .scheduler_common import *
from .scheduler_types import _GraphIndex, ScheduledTask
from .scheduler_base import SchedulerBase

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
        total_service = 0.0
        total_compute = 0.0
        k = 0
        node_weight_size = self._node_weight_size(node)
        wid = self._node_weight_id(node)
        weighted = bool(wid and node_weight_size > 0)
        for d in devs:
            if not self._node_allowed_on(node, d):
                continue
            k += 1
            if weighted and RANKU_INCLUDE_AVG_WEIGHT_LOAD:
                stored_fmt = self._weight_storage_format_for_wid(wid)
                prof = self._weight_service_profile_no_contention(
                    node,
                    d,
                    src_storage_fmt=str(stored_fmt),
                    cached=False,
                    label=self.label,
                    batch=int(batch_eff),
                    seq_len=int(seq_len_eff),
                    phase=str(phase_eff),
                )
                total_service += float(prof.get('total_s', 0.0) or 0.0)
            else:
                total_compute += float(self._weighted_compute_time(node, d, self.label, batch_eff, seq_len_eff, phase_eff))

        if k == 0:
            return 0.0
        if weighted and RANKU_INCLUDE_AVG_WEIGHT_LOAD:
            return float(total_service / k)
        return float(total_compute / k)
    
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
                        self._log_scheduled_op_trace(
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
                # Only one candidate: keep the same device as the source K/V.
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
                    self._log_scheduled_op_trace(
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

    def suggest_weight_storage_formats(self) -> Dict[str, str]:
        Base = ['ND', 'NZ', 'PIM-OPT']
        sugg: Dict[str, str] = {}

        # Aggregate as {weight_id: {device_type: load_count}}.
        by_wid = defaultdict(lambda: defaultdict(int))
        for (wid, dev_type), cnt in self._weight_load_count.items():
            by_wid[wid][dev_type] += cnt

        EPS = 1e-6
        for wid, counts in by_wid.items():
            counts_eff = self._normalize_counts_by_device_count(counts)
            dominant = max(counts_eff.items(), key=lambda x: x[1])[0] if counts_eff else 'pim'
            if dominant == 'npu':
                candidates = ['NZ', 'ND', 'PIM-OPT']
                native = 'NZ'
            elif dominant == 'pim':
                candidates = ['PIM-OPT', 'ND', 'NZ']
                native = 'PIM-OPT'
            else:
                candidates = Base
                native = 'ND'

            best_t, best_fmt = float('inf'), candidates[0]
            for fmt in candidates:
                total = 0.0
                for dev_type, cnt in counts_eff.items():
                    devs = self.cluster.devices_by_type(dev_type)
                    if not devs:
                        continue
                    d = devs[0]
                    proto = self._representative_weight_node(wid)
                    prof = self._weight_service_profile_no_contention(
                        proto,
                        d,
                        src_storage_fmt=str(fmt),
                        cached=False,
                    )
                    w_cost = float(prof.get('total_s', 0.0) or 0.0)
                    total += float(cnt) * float(w_cost)
                if total + EPS < best_t or (abs(total - best_t) < EPS and fmt == native):
                    best_t, best_fmt = total, fmt
            sugg[wid] = best_fmt

        return sugg

    # ------------------------------------------------------------------
    # Block Coordinate Descent (BCD) weight-format suggestion
    # ------------------------------------------------------------------
    def _split_layer_prefixed_weight_id(self, wid: str) -> Tuple[int | None, str]:
        """Return (layer_idx, rest_name) for layer-scoped weight ids."""
        if not wid:
            return (None, "")
        s = str(wid)
        try:
            m = re.match(r"^L(?P<layer>\d+)_(?P<rest>.*)$", s)
            if not m:
                return (None, s)
            layer_idx = int(m.group('layer'))
            return (layer_idx, m.group('rest') or "")
        except Exception:
            return (None, s)

    def _strip_layer_prefix(self, wid: str) -> str:
        """Strip leading layer tag from weight_id."""
        _layer_idx, rest = self._split_layer_prefixed_weight_id(wid)
        return rest if rest else str(wid or "")

    def _normalize_counts_by_device_count(self, counts: Mapping[str, int | float]) -> Dict[str, float]:
        """Normalize load counts by the number of devices of each type.

        This avoids biasing block decisions toward a device class simply because the
        topology contains more devices of that type (for example 1 NPU vs 2 PIMs).
        """
        out: Dict[str, float] = {}
        for dev_type, cnt in dict(counts or {}).items():
            try:
                denom = max(1, int(len(self.cluster.devices_by_type(str(dev_type))) or 0))
            except Exception:
                denom = 1
            out[str(dev_type)] = float(cnt or 0.0) / float(denom)
        return out

    def _weight_block_key(self, wid: str, *, mode: str = "coupled", layer_span: int = 0) -> str:
        """Return a block key for `wid`.

        mode:
          - 'none'   : only strip layer prefix, no coupling.
          - 'coupled': additionally couple (WQ,WK,WV) as one block, and (W1,W3) as one block,
                       while keeping shard/expert suffixes.

        layer_span:
          - <= 0 : merge all layers together (legacy behavior).
          - 4/8  : keep one block per consecutive 4/8 layers.
        """
        layer_idx, base = self._split_layer_prefixed_weight_id(wid)
        base = base if base else str(wid or "")
        key = base
        if mode not in ("none", "strip_only", ""):
            parts = [p for p in str(base).split("_") if p]
            if parts:
                # Common (non-MoE) weights: WQ/WK/WV, WO, W1/W2/W3, possibly with _S{sid}.
                head = parts[0]
                tail = "_".join(parts[1:]) if len(parts) > 1 else ""

                def _join(prefix: str, rest: str) -> str:
                    return f"{prefix}_{rest}" if rest else prefix

                if head in ("WQ", "WK", "WV"):
                    key = _join("ATTN_QKV", tail)
                elif head in ("W1", "W3"):
                    key = _join("FFN_W13", tail)
                # MoE style: E{e}_W1 / E{e}_W3 etc. Keep expert id as part of the key.
                elif head.startswith("E") and len(parts) >= 2:
                    wname = parts[1]
                    rest = "_".join(parts[2:]) if len(parts) > 2 else ""
                    if wname in ("W1", "W3"):
                        key = _join(f"{head}_FFN_W13", rest)
                    elif wname in ("WQ", "WK", "WV"):
                        key = _join(f"{head}_ATTN_QKV", rest)

        if layer_idx is None or int(layer_span or 0) <= 0:
            return key
        span = max(1, int(layer_span))
        lo = (int(layer_idx) // span) * span
        hi = lo + span - 1
        if span == 1:
            return f"L{int(layer_idx)}_{key}"
        return f"L{lo}-{hi}_{key}"

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
        proto = self._representative_weight_node(wid)
        for dev_type, cnt in counts.items():
            if not cnt:
                continue
            devs = self.cluster.devices_by_type(str(dev_type))
            if not devs:
                continue
            d = devs[0]
            prof = self._weight_service_profile_no_contention(
                proto,
                d,
                src_storage_fmt=str(fmt),
                cached=False,
            )
            w_cost = float(prof.get('total_s', 0.0) or 0.0)
            total += float(cnt) * float(w_cost)
        return float(factor * total)

    def suggest_weight_storage_formats_bcd(
        self,
        current_map: Dict[str, str] | None = None,
        *,
        max_block_changes: int = 1,
        min_gain_ratio: float = 0.005,
        block_mode: str = "coupled",
        candidates: Tuple[str, ...] = ("ND", "NZ", "PIM-OPT"),
        lookahead_beta: float = 0.25,
        layer_span: int = 0,
        normalize_reload_by_device_count: bool = True,
    ) -> Dict[str, str]:
        """Suggest next host weight formats via *block* coordinate descent (BCD).

        Returns:
            A new map (wid -> fmt).
        """
        cur = {}
        for _wk, _wv in dict(current_map or {}).items():
            try:
                cur[str(_wk)] = str(self.cost.weight_storage_format(_wv))
            except Exception:
                cur[str(_wk)] = str(_wv)

        # Aggregate cache-miss load counts observed in the last scheduling pass.
        by_wid: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for (wid, dev_type), cnt in self._weight_load_count.items():
            by_wid[str(wid)][str(dev_type)] += int(cnt)

        # Include weights already in the map so blocks stay consistent even if
        # a few weights did not show up in the stats (e.g., sampling).
        for wid in list(cur.keys()):
            by_wid.setdefault(str(wid), defaultdict(int))

        by_wid_eff: Dict[str, Dict[str, float]] = {}
        for wid, counts in by_wid.items():
            if normalize_reload_by_device_count:
                by_wid_eff[str(wid)] = self._normalize_counts_by_device_count(counts)
            else:
                by_wid_eff[str(wid)] = {str(k): float(v) for k, v in dict(counts or {}).items()}

        # Lookahead-chain hits (optional, only available in Bifocal).
        chain_hits = {}
        try:
            chain_hits = dict(getattr(self, "_weight_chain_hits", {}) or {})
        except Exception:
            chain_hits = {}
        max_hits = float(max(chain_hits.values(), default=0.0)) if chain_hits else 0.0

        # Build blocks.
        blocks: Dict[str, List[str]] = defaultdict(list)
        for wid in by_wid.keys():
            key = self._weight_block_key(wid, mode=block_mode, layer_span=layer_span)
            blocks[str(key)].append(str(wid))

        # Helper: block-level dominant device type for tie-breaking.
        def _block_native_fmt(wids: List[str]) -> str:
            npu = 0.0
            pim = 0.0
            for w in wids:
                c = by_wid_eff.get(w, {})
                npu += float(c.get("npu", 0.0) or 0.0)
                pim += float(c.get("pim", 0.0) or 0.0)
            if npu > pim:
                return "NZ"
            if pim > npu:
                return "PIM-OPT"
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
                    w, by_wid_eff.get(w, {}), fmt0,
                    lookahead_beta=lookahead_beta, max_chain_hits=max_hits, chain_hits=chain_hits
                )

            best_cost = float("inf")
            best_fmt = cur.get(wids[0], "ND")
            for fmt in cand:
                total = 0.0
                for w in wids:
                    total += self._estimate_weight_host_to_device_cost(
                        w, by_wid_eff.get(w, {}), fmt,
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

