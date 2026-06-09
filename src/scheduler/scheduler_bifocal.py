from __future__ import annotations
from .scheduler_common import *
from .scheduler_types import _GraphIndex, ScheduledTask
from .scheduler_heft import HEFTScheduler

class BifocalScheduler(HEFTScheduler):
    """Bifocal scheduler that reuses SchedulerBase timing and communication modeling."""

    def __init__(
        self, 
        cluster: Cluster, 
        cost: CostModel, 
        label: PlanLabel, 
        batch: int, 
        seq_len: int, 
        buffer: GlobalMemoryManager, 
        *, 
        rand_seed: int | None = None):
        super().__init__(cluster, cost, label, batch, seq_len, buffer)

        # Near-future plan hints used by the lookahead consistency penalty.
        # the oldest hints when exceeding SCHED_JOINT_LK_PLAN_HINT_MAX.
        self._plan_hint: Dict[str, str] = {}
        self._rng = random.Random(rand_seed)
        self._weight_plan_hint: Dict[str, str] = {}
        self._model_total_weight_bytes: Optional[int] = None
        self._weight_chain_hits: Dict[str, float] = defaultdict(float)
        self._decode_cur_token_idx: Optional[int] = None
        self._decode_total_tokens: Optional[int] = None
        self._decode_cfg_map: Dict[str, Any] = {}
        self._decode_amort_eval_cache: Dict[Tuple[str, str, str, int, int, int], float] = {}

    def reset_state(self, *, clear_caches: bool = True) -> None:
        """Reset runtime state and clear Bifocal lookahead hints."""
        super().reset_state(clear_caches=clear_caches)
        self._plan_hint.clear()
        self._weight_plan_hint.clear()
        self._model_total_weight_bytes = None
        try:
            self._weight_chain_hits.clear()
        except Exception:
            pass
        try:
            self._decode_amort_eval_cache.clear()
        except Exception:
            pass

    # -----------------------------
    # Long-horizon decode context
    # -----------------------------
    def set_decode_context(
        self,
        *,
        cur_token_idx: Optional[int] = None,
        total_decode_tokens: Optional[int] = None,
        cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        changed = False
        nxt_cur = None if cur_token_idx is None else max(0, int(cur_token_idx))
        nxt_total = None if total_decode_tokens is None else max(0, int(total_decode_tokens))
        if nxt_cur != self._decode_cur_token_idx or nxt_total != self._decode_total_tokens:
            changed = True
        self._decode_cur_token_idx = nxt_cur
        self._decode_total_tokens = nxt_total
        if isinstance(cfg, Mapping):
            new_cfg = dict(cfg)
            if new_cfg != self._decode_cfg_map:
                changed = True
            self._decode_cfg_map = new_cfg
        if changed:
            try:
                self._decode_amort_eval_cache.clear()
            except Exception:
                pass

    def clear_decode_context(self) -> None:
        self._decode_cur_token_idx = None
        self._decode_total_tokens = None
        self._decode_cfg_map = {}
        try:
            self._decode_amort_eval_cache.clear()
        except Exception:
            pass

    def _decode_cfg_value(self, *names: str, default: Any = None) -> Any:
        keys: List[str] = []
        for name in names:
            if not name:
                continue
            s = str(name)
            for cand in (s, s.lower(), s.upper()):
                if cand not in keys:
                    keys.append(cand)

        cfg_map = self._decode_cfg_map if isinstance(self._decode_cfg_map, Mapping) else {}
        for k in keys:
            try:
                if k in cfg_map:
                    return cfg_map[k]
            except Exception:
                continue
            
        mod = sys.modules.get('config', None)
        if mod is not None:
            for k in keys:
                try:
                    if hasattr(mod, k):
                        return getattr(mod, k)
                except Exception:
                    continue
        return default

    @staticmethod
    def _decode_cfg_bool(v: Any, default: bool = False) -> bool:
        if v is None:
            return bool(default)
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, (int, float)):
            return bool(v)
        s = str(v).strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on", "enable", "enabled"):
            return True
        if s in ("0", "false", "f", "no", "n", "off", "disable", "disabled"):
            return False
        return bool(default)

    def _decode_amort_enabled(self, phase: str) -> bool:
        if str(phase or "").lower() != "decode":
            return False
        if self._decode_cur_token_idx is None or self._decode_total_tokens is None:
            return False
        raw = self._decode_cfg_value(
            "decode_amort_enable",
            "sched_decode_amort_enable",
            "SCHED_DECODE_AMORT_ENABLE",
            default=True,
        )
        return bool(self._decode_cfg_bool(raw, default=True))

    def _remaining_decode_tokens(self) -> int:
        total = 0 if self._decode_total_tokens is None else int(self._decode_total_tokens)
        cur = 0 if self._decode_cur_token_idx is None else int(self._decode_cur_token_idx)
        return max(1, int(total - cur))
    def _decode_weight_service_profile(
        self,
        node: TaskNode,
        dev: DeviceSpec,
        *,
        label: Optional[PlanLabel] = None,
        batch: Optional[int] = None,
        seq_len: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> Dict[str, float | int | str]:
        """Contention-free weight service estimate for decode amortization.
        """
        wid = self._node_weight_id(node)
        wsize_nd = int(self._node_weight_size(node) or 0)
        if not wid or wsize_nd <= 0:
            return {
                'wid': str(wid or ''),
                'weight_size_nd': int(wsize_nd),
                'cached_nd': 0,
                'need_nd': 0,
                'warm_weight': 0.0,
                'current_weight': 0.0,
            }

        phase_eff = str(phase or 'decode')
        batch_eff = int(batch or getattr(self, 'batch', 1) or 1)
        seq_eff = int(seq_len or getattr(self, 'seq_len', 1) or 1)
        label_eff = label if label is not None else self.label

        src_storage_fmt = self._weight_storage_format_for_wid(wid)
        dev_type = str(getattr(dev, 'type', '')).lower()
        if dev_type == 'npu':
            resident_fmt = self._npu_resident_weight_format(src_storage_fmt)
        elif dev_type == 'pim':
            resident_fmt = 'PIM-OPT'
        else:
            resident_fmt = str(src_storage_fmt)

        warm_prof = self._weight_service_profile_no_contention(
            node,
            dev,
            src_storage_fmt=str(src_storage_fmt),
            cached=True,
            cached_fmt=str(resident_fmt),
            label=label_eff,
            batch=int(batch_eff),
            seq_len=int(seq_eff),
            phase=str(phase_eff),
        )

        if dev_type == 'pim' and self._weights_preloaded_on_pim():
            return {
                'wid': str(wid),
                'weight_size_nd': int(wsize_nd),
                'cached_nd': int(wsize_nd),
                'need_nd': 0,
                'warm_weight': float(warm_prof.get('total_s', 0.0) or 0.0),
                'current_weight': float(warm_prof.get('total_s', 0.0) or 0.0),
            }

        cached_nd = 0
        try:
            cache = getattr(self.buffer, 'device_cache', {}).get(dev.name, None)
            items = getattr(cache, 'items', None)
            if isinstance(items, dict):
                v = items.get(wid, 0)
                if isinstance(v, (int, float)):
                    cached_nd = int(v)
        except Exception:
            cached_nd = 0

        if cached_nd <= 0:
            try:
                if self.buffer.is_cached(dev.name, wid):
                    cached_nd = int(wsize_nd)
            except Exception:
                cached_nd = 0

        if cached_nd >= wsize_nd:
            return {
                'wid': str(wid),
                'weight_size_nd': int(wsize_nd),
                'cached_nd': int(wsize_nd),
                'need_nd': 0,
                'warm_weight': float(warm_prof.get('total_s', 0.0) or 0.0),
                'current_weight': float(warm_prof.get('total_s', 0.0) or 0.0),
            }

        need_nd = int(max(0, wsize_nd - max(0, cached_nd)))
        miss_prof = self._weight_service_profile_no_contention(
            node,
            dev,
            src_storage_fmt=str(src_storage_fmt),
            cached=False,
            label=label_eff,
            batch=int(batch_eff),
            seq_len=int(seq_eff),
            phase=str(phase_eff),
        )
        return {
            'wid': str(wid),
            'weight_size_nd': int(wsize_nd),
            'cached_nd': int(max(0, cached_nd)),
            'need_nd': int(max(0, need_nd)),
            'warm_weight': float(warm_prof.get('total_s', 0.0) or 0.0),
            'current_weight': float(miss_prof.get('total_s', 0.0) or 0.0),
        }

    def _decode_phase_amortization_bias(self, g: TaskGraph, nid: str, dev: DeviceSpec, phase: str) -> float:
        """Cross-token decode bias: amortize one-time migration over remaining decode steps.        """
        if not self._decode_amort_enabled(phase):
            return 0.0

        try:
            node = g.nodes[nid]
        except Exception:
            return 0.0

        wid = self._node_weight_id(node)
        wsize_nd = int(self._node_weight_size(node) or 0)
        if not wid or wsize_nd <= 0:
            return 0.0

        rem = int(self._remaining_decode_tokens())
        key = (
            str(nid),
            str(getattr(dev, 'name', '')),
            str(phase or '').lower(),
            int(rem),
            int(getattr(self, 'seq_len', 0) or 0),
            int(self._decode_cur_token_idx or 0),
        )
        cached_bias = self._decode_amort_eval_cache.get(key, None)
        if cached_bias is not None:
            return float(cached_bias)

        phase_eff = self._node_phase(g, nid, phase)
        batch_eff = self._node_batch(g, nid, phase_eff)
        seq_eff = self._node_seq_len(g, nid, phase_eff)

        try:
            compute_t = float(self._weighted_compute_time(node, dev, self.label, batch_eff, seq_eff, phase_eff))
        except Exception:
            compute_t = float('inf')
        if not math.isfinite(compute_t):
            self._decode_amort_eval_cache[key] = 0.0
            return 0.0

        prof = self._decode_weight_service_profile(
            node,
            dev,
            label=self.label,
            batch=int(batch_eff),
            seq_len=int(seq_eff),
            phase=str(phase_eff),
        )
        warm_w = float(prof.get('warm_weight', 0.0) or 0.0)
        cur_w = float(prof.get('current_weight', 0.0) or 0.0)

        s_warm = float(warm_w if warm_w > 0.0 else compute_t)
        s_cur = float(cur_w if cur_w > 0.0 else compute_t)
        migrate_extra = float(max(0.0, s_cur - s_warm))
        if migrate_extra <= 0.0:
            self._decode_amort_eval_cache[key] = 0.0
            return 0.0

        alpha = float(self._decode_cfg_value(
            'decode_amort_alpha',
            'sched_decode_amort_alpha',
            'SCHED_DECODE_AMORT_ALPHA',
            default=1.0,
        ) or 0.0)
        if alpha == 0.0:
            self._decode_amort_eval_cache[key] = 0.0
            return 0.0

        rmin = float(self._decode_cfg_value(
            'decode_amort_rmin',
            'sched_decode_amort_rmin',
            'SCHED_DECODE_AMORT_RMIN',
            default=1.0,
        ) or 1.0)
        reuse_prob = float(self._decode_cfg_value(
            'decode_amort_reuse_prob',
            'sched_decode_amort_reuse_prob',
            'SCHED_DECODE_AMORT_REUSE_PROB',
            default=1.0,
        ) or 1.0)
        reuse_prob = min(1.0, max(0.0, reuse_prob))

        r_eff = max(float(rmin), float(rem) * float(reuse_prob))
        avg_decode_service = float(s_warm + migrate_extra / max(1.0, r_eff))

        bias = float(alpha * (avg_decode_service - s_cur))
        if bias > 0.0:
            bias = 0.0

        self._decode_amort_eval_cache[key] = float(bias)
        return float(bias)

    # -----------------------------
    # Helpers for Bifocal scheduling
    # -----------------------------
    def _edge_data_bytes(self, g: TaskGraph, u: str, v: str, phase: str) -> int:
        """Best-effort edge payload estimation.

        Prefer explicit edge metadata if present; otherwise fall back to an
        activation-based heuristic similar to `_avg_comm_cost`.
        """

        node_u = g.nodes[u]
        node_v = g.nodes[v]
        phase_u = self._node_phase(g, u, phase)
        phase_v = self._node_phase(g, v, phase)
        batch_u = self._node_batch(g, u, phase_u)
        batch_v = self._node_batch(g, v, phase_v)
        seq_u = self._node_seq_len(g, u, phase_u)
        seq_v = self._node_seq_len(g, v, phase_v)

        _, u_write = self.cost.estimate_activation_bytes(node_u, batch_u, seq_u, phase_u)
        v_read, _ = self.cost.estimate_activation_bytes(node_v, batch_v, seq_v, phase_v)
        return int(max(u_write, v_read, 16 * 1024))

    def _estimate_transfer_time(self, src: DeviceSpec, dst: DeviceSpec, bytes_nd: int, *, src_fmt: Optional[str] = None) -> float:
        """Contention-free activation-move estimate aligned with evaluate path."""
        if src.name == dst.name:
            return 0.0

        size_nd = int(self.cost.format_size(int(bytes_nd), 'ND'))
        src_fmt_eff = str(src_fmt or self.cost.device_preferred_fmt(src))
        dst_fmt = str(self.cost.device_preferred_fmt(dst))

        size_src = int(self.cost.format_size(int(bytes_nd), str(src_fmt_eff)))
        t_conv_src = 0.0
        if str(src_fmt_eff) != 'ND':
            t_conv_src = float(self.cost.format_conversion_time(int(size_src), str(src_fmt_eff), 'ND', src))

        t_read_src = 0.0
        if str(getattr(src, 'type', '') or '').lower() == 'pim':
            t_read_src = float(self.cost.activation_read_time_pim(int(size_nd)))

        base = float(t_conv_src + t_read_src)
        topo = normalize_topology(getattr(self.cluster, 'topology', None))
        host = self.cost.get_host_device()

        t_direct = float(self.cost.comm_cost(src, dst, int(size_nd)))
        if math.isfinite(t_direct):
            t_direct = float(base + self.cost.combine_transfer_and_convert(src, dst, int(size_nd), 'ND', str(dst_fmt)))
        else:
            t_direct = float('inf')

        t_via_host = float('inf')
        t_to_host = float(self.cost.comm_cost(src, host, int(size_nd)))
        if math.isfinite(t_to_host):
            t_via_host = float(base + t_to_host + self.cost.combine_transfer_and_convert(host, dst, int(size_nd), 'ND', str(dst_fmt)))

        if topo == 'fc':
            return float(t_direct if math.isfinite(t_direct) else t_via_host)
        return float(min(t_direct, t_via_host))

    def _rep_device_by_type(self, dev_type: str) -> Optional[DeviceSpec]:
        devs = self.cluster.devices_by_type(dev_type)
        if not devs:
            return None
        return min(devs, key=lambda d: float(self.avail.get(d.name, 0.0)))

    def _snapshot_cached_weights(self, dev_name: str) -> set[str]:
        """Return a best-effort snapshot of cached weight_ids for a device."""
        cache = getattr(self.buffer, "device_cache", {}).get(dev_name, None)  # type: ignore[attr-defined]
        if cache is not None:
            for attr in ("items", "data", "cache"):
                try:
                    items = getattr(cache, attr, None)
                    if isinstance(items, Mapping):
                        return set(str(k) for k in items.keys())
                except Exception:
                    pass
        # Fallback: unknown cache internals; return empty.
        return set()

    def _estimate_weight_reload_time(
        self,
        node: TaskNode,
        dev: DeviceSpec,
        *,
        label: Optional[PlanLabel] = None,
        batch: Optional[int] = None,
        seq_len: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> float:
        """Contention-free estimate of loading (and converting) a weight to `dev`.

        Score heuristics must use the same runtime context as evaluate; otherwise they can
        penalize/reward a placement with stale prefill defaults.
        """
        wid = self._node_weight_id(node)
        wsize_nd = int(self._node_weight_size(node) or 0)
        if not wid or wsize_nd <= 0:
            return 0.0
        if str(getattr(dev, 'type', '')).lower() == 'pim' and self._weights_preloaded_on_pim():
            return 0.0
        if self.buffer.is_cached(dev.name, wid):
            return 0.0
        from_fmt = self._weight_storage_format_for_wid(wid)
        prof = self._weight_service_profile_no_contention(
            node,
            dev,
            src_storage_fmt=str(from_fmt),
            cached=False,
            label=(label if label is not None else self.label),
            batch=(int(batch) if batch is not None else int(getattr(self, 'batch', 1) or 1)),
            seq_len=(int(seq_len) if seq_len is not None else int(getattr(self, 'seq_len', 1) or 1)),
            phase=(str(phase) if phase is not None else 'prefill'),
        )
        return float(prof.get('total_s', 0.0) or 0.0)

    def _representative_activation_bytes(self, g: TaskGraph, nid: str, phase: str) -> int:
        """Representative activation footprint for `nid` used in consistency penalty."""
        try:
            node = g.nodes[nid]
            phase_eff = self._node_phase(g, nid, phase)
            batch_eff = self._node_batch(g, nid, phase_eff)
            seq_eff = self._node_seq_len(g, nid, phase_eff)
            r, w = self.cost.estimate_activation_bytes(node, batch_eff, seq_eff, phase_eff)
            return int(max(r, w, 16 * 1024))
        except Exception:
            return int(16 * 1024)

    def _build_lookahead_chain(
        self,
        g: TaskGraph,
        start: str,
        idx: _GraphIndex,
        rank_u: Mapping[str, float],
        phase: str,
        H: int,
    ) -> List[str]:
        """Heaviest-edge successor chain with a small rank_u tie-breaker."""
        H = max(1, int(H))
        chain: List[str] = [start]
        eps = 1e-6
        cur = start
        while len(chain) < H:
            succs = idx.succs.get(cur, ())
            if not succs:
                break
            best_x = None
            best_val = -float("inf")
            for x in succs:
                try:
                    d = float(self._edge_data_bytes(g, cur, x, phase))
                except Exception:
                    d = 0.0
                val = d + eps * float(rank_u.get(x, 0.0))
                if val > best_val:
                    best_val = val
                    best_x = x
            if best_x is None or best_x in chain:
                break
            chain.append(best_x)
            cur = best_x
        return chain

    def _simulate_chain_finish(
        self,
        g: TaskGraph,
        chain: List[str],
        devs: List[DeviceSpec],
        first_finish: float,
        phase: str,
        *,
        apply_consistency_penalty: bool,
    ) -> Tuple[float, Dict[str, str]]:
        """Simulate the completion time of `chain` under a fixed device assignment.

        This is a *lightweight*, contention-free forward estimate used by the lookahead
        window. It deliberately ignores non-chain dependencies to keep it cheap.

        Returns:
            (estimated_finish_time_including_penalty, assignment_map_for_future_nodes)
        """
        if not chain:
            return float(first_finish), {}
        if len(chain) != len(devs):
            raise ValueError("chain/devs length mismatch")

        # Shadow device availability.
        avail_shadow: Dict[str, float] = dict(self.avail)
        # Shadow weight cache (best effort).
        cached_shadow: Dict[str, set[str]] = {}
        for d in {d.name for d in devs}:
            cached_shadow[d] = self._snapshot_cached_weights(d)

        # First node: finish time is taken from EFT(v,k) estimation; assume its weight
        # becomes cached on its device if it uses weights.
        v0 = chain[0]
        d0 = devs[0]
        finish = float(first_finish)
        avail_shadow[d0.name] = max(float(avail_shadow.get(d0.name, 0.0)), finish)
        try:
            n0 = g.nodes[v0]
            wid0 = self._node_weight_id(n0)
            if wid0:
                cached_shadow.setdefault(d0.name, set()).add(wid0)
        except Exception:
            pass

        # Assignment map for future nodes in this chain.
        assign_map: Dict[str, str] = {}
        for nid, d in zip(chain[1:], devs[1:]):
            assign_map[nid] = d.name

        # Sequential simulation along the chain.
        prev_nid = v0
        prev_dev = d0
        for nid, dev in zip(chain[1:], devs[1:]):
            bytes_edge = int(self._edge_data_bytes(g, prev_nid, nid, phase))
            src_fmt_prev = str(self.cost.device_preferred_fmt(prev_dev))
            comm_t = float(self._estimate_transfer_time(prev_dev, dev, bytes_edge, src_fmt=src_fmt_prev))

            ready_t = float(finish + comm_t)
            start_t = float(max(float(avail_shadow.get(dev.name, 0.0)), ready_t))

            node = g.nodes[nid]
            phase_eff = self._node_phase(g, nid, phase)
            batch_eff = self._node_batch(g, nid, phase_eff)
            seq_eff = self._node_seq_len(g, nid, phase_eff)
            compute_t = float(self._weighted_compute_time(node, dev, self.label, batch_eff, seq_eff, phase_eff))

            wid = self._node_weight_id(node)
            wsize = int(self._node_weight_size(node))
            total_weight_t = 0.0
            if wid and wsize > 0:
                dev_cached = cached_shadow.setdefault(dev.name, set())
                stored_fmt = self._weight_storage_format_for_wid(wid)
                if wid in dev_cached:
                    resident_fmt = self._weight_resident_format(dev, str(stored_fmt))
                    prof = self._weight_service_profile_no_contention(
                        node,
                        dev,
                        src_storage_fmt=str(stored_fmt),
                        cached=True,
                        cached_fmt=str(resident_fmt),
                        label=self.label,
                        batch=int(batch_eff),
                        seq_len=int(seq_eff),
                        phase=str(phase_eff),
                    )
                else:
                    prof = self._weight_service_profile_no_contention(
                        node,
                        dev,
                        src_storage_fmt=str(stored_fmt),
                        cached=False,
                        label=self.label,
                        batch=int(batch_eff),
                        seq_len=int(seq_eff),
                        phase=str(phase_eff),
                    )
                    dev_cached.add(wid)
                total_weight_t = float(prof.get('total_s', 0.0) or 0.0)
            else:
                total_weight_t = float(compute_t)

            finish = start_t + float(total_weight_t)
            avail_shadow[dev.name] = finish
            prev_nid, prev_dev = nid, dev

        # Consistency penalty: discourage oscillation in overlapping windows.
        penalty = 0.0
        if apply_consistency_penalty and self._plan_hint and len(chain) > 1:
            lam = float(SCHED_JOINT_LK_CONSIST_LAMBDA)
            if lam > 0:
                for nid, dev in zip(chain[1:], devs[1:]):
                    hinted = self._plan_hint.get(nid)
                    if hinted and hinted != dev.name:
                        src = self.cluster.devices.get(hinted)
                        if src is None:
                            continue
                        bytes_act = int(self._representative_activation_bytes(g, nid, phase))
                        src_fmt_hint = str(self.cost.device_preferred_fmt(src))
                        penalty += lam * float(self._estimate_transfer_time(src, dev, bytes_act, src_fmt=src_fmt_hint))

        return float(finish + penalty), assign_map

    def _lookahead_window_estimate(
        self,
        g: TaskGraph,
        chain: List[str],
        first_dev: DeviceSpec,
        first_eft: float,
        phase: str,
    ) -> Tuple[float, Dict[str, str]]:

        if len(chain) <= 1:
            return float(first_eft), {}

        exec_types = tuple(self._executor_device_types())

        # Collect per-node concrete device options for the (h-1) future nodes.
        device_options: List[List[DeviceSpec]] = []
        for nid in chain[1:]:
            node = g.nodes[nid]
            opts: List[DeviceSpec] = []

            if self._is_comm_node(node):
                host = self.cost.get_host_device()
                if host is not None:
                    opts.append(host)
            else:
                for dev_type in exec_types:
                    for dev in self.cluster.devices_by_type(dev_type):
                        try:
                            if self._node_allowed_on(node, dev):
                                opts.append(dev)
                        except Exception:
                            # If allowed() check fails, be conservative and keep
                            # the actual device as an option rather than falling
                            # back to a representative device.
                            opts.append(dev)

            # Deduplicate by device name while preserving a deterministic order.
            by_name: Dict[str, DeviceSpec] = {}
            for dev in opts:
                try:
                    by_name[str(dev.name)] = dev
                except Exception:
                    continue
            opts = [by_name[k] for k in sorted(by_name.keys(), key=lambda name: (float(self.avail.get(name, 0.0)), name))]

            if not opts:
                # No concrete device can run this node; fall back to no-lookahead.
                return float(first_eft), {}
            device_options.append(opts)

        best_finish = float("inf")
        best_assign: Dict[str, str] = {}

        for dev_combo in itertools.product(*device_options):
            devs: List[DeviceSpec] = [first_dev] + list(dev_combo)
            finish, assign = self._simulate_chain_finish(
                g,
                chain=chain,
                devs=devs,
                first_finish=float(first_eft),
                phase=phase,
                apply_consistency_penalty=True,
            )
            if finish < best_finish:
                best_finish = float(finish)
                best_assign = dict(assign)

        if not math.isfinite(best_finish):
            return float(first_eft), {}
        return float(best_finish), best_assign




    def _interval_nodes_for_weight_bias(self, idx: _GraphIndex, g: TaskGraph, nid: str) -> List[str]:
        """Approximate nodes in interval I(v) for weight-bias.

        - For decode-time cross-token weight reuse, the interval from one use of v to the next walks through roughly one full model pass.
          Approximate the interval working set as the weight set of the full graph.

        - Return all nodes in topological order directly.

        This helper remains only for backward compatibility; the new bias logic no longer depends on repeat/meta.
        """
        try:
            return list(idx.topo)
        except Exception:
            try:
                return list(idx.nodes)
            except Exception:
                return [nid]

    def _pim_weight_cache_remaining_bytes(self, dev_name: str) -> int:
        cache = getattr(self.buffer, "device_cache", {}).get(dev_name, None)  # type: ignore[attr-defined]
        if cache is None:
            return 0
        cap = None
        used = None
        for a in ("capacity", "cap", "max_bytes", "limit"):
            try:
                v = getattr(cache, a, None)
                if isinstance(v, (int, float)):
                    cap = int(v)
                    break
            except Exception:
                continue
        try:
            v = getattr(cache, "used", None)
            if isinstance(v, (int, float)):
                used = int(v)
        except Exception:
            used = None
        if cap is None:
            return 0
        if used is None:
            used = 0
        return max(0, int(cap - used))

    def _weight_reuse_bias(self, g: TaskGraph, idx: _GraphIndex, nid: str, dev: DeviceSpec, phase: str) -> float:
        """Weight-bias w/o node.meta: use label-level total-weight estimate.

        - If the node has a weight, approximate the interval from this decode use to the next use as one full pass over model weights.
          This is equivalent to traversing the full model weight set once.
        - If a device has enough free weight-cache space for the full model weights, treat the weight as reusable on the next encounter.
          Apply a gain (negative bias) to that device during prefill.
        - Also maintain a per-weight hint so the next occurrence of the same weight_id can follow it.
          Penalize candidates whose device_type does not match the stored hint.

        Return the bias added to the score (negative encourages, positive penalizes).
        """

        node = g.nodes[nid]
        wid = self._node_weight_id(node)
        wsize = int(self._node_weight_size(node))
        if not wid or wsize <= 0:
            return 0.0

        phase_eff = self._node_phase(g, nid, phase)
        batch_eff = self._node_batch(g, nid, phase_eff)
        seq_eff = self._node_seq_len(g, nid, phase_eff)

        # ---- 1) penalty if deviating from the stored hint (cross-token / repeated-op consistency) ----
        penalty = 0.0
        hinted_type = self._weight_plan_hint.get(wid)
        if hinted_type is not None and str(hinted_type) != str(getattr(dev, "type", "")):
            lam = float(SCHED_JOINT_LK_CONSIST_LAMBDA) if SCHED_JOINT_LK_CONSIST_LAMBDA is not None else 1.0
            t_reload = float(self._estimate_weight_reload_time(
                node,
                dev,
                label=self.label,
                batch=int(batch_eff),
                seq_len=int(seq_eff),
                phase=str(phase_eff),
            ))
            penalty = float(lam * t_reload)

        # ---- 2) gain in prefill if device can keep whole-model weights ----
        gain = 0.0
        phase_l = str(phase_eff or "").lower()
        if phase_l == "prefill":
            total_w = int(self._get_total_model_weight_bytes(g))
            if total_w > 0 and self._device_can_hold_all_weights(dev, total_w):
                eta = float(SCHED_WEIGHT_BIAS_ETA) if SCHED_WEIGHT_BIAS_ETA is not None else 1.0
                t_reload = float(self._estimate_weight_reload_time(
                    node,
                    dev,
                    label=self.label,
                    batch=int(batch_eff),
                    seq_len=int(seq_eff),
                    phase=str(phase_eff),
                ))
                gain = float(-eta * t_reload)

        return float(penalty + gain)

    def _get_total_model_weight_bytes(self, g: TaskGraph) -> int:
        """Return total model weight bytes (ND) from label if possible, else from graph."""
        if isinstance(self._model_total_weight_bytes, int) and self._model_total_weight_bytes > 0:
            return int(self._model_total_weight_bytes)

        # 1) Prefer label-provided totals (main：plan label).
        for attr in (
            "total_weight_bytes",
            "model_weight_total_bytes",
            "model_weight_bytes",
            "weights_total_bytes",
            "weight_total_bytes",
            "fc_total_bytes",
            "fc_weight_bytes",
        ):
            try:
                v = getattr(self.label, attr, None)
                if isinstance(v, (int, float)) and float(v) > 0:
                    self._model_total_weight_bytes = int(v)
                    return int(self._model_total_weight_bytes)
            except Exception:
                continue

        # 2) Fallback: sum unique weight_id sizes from graph nodes.
        seen: set[str] = set()
        total = 0
        try:
            for n in g.nodes.values():
                wid = self._node_weight_id(n)
                if not wid or wid in seen:
                    continue
                w = int(self._node_weight_size(n))
                if w <= 0:
                    continue
                seen.add(wid)
                total += w
        except Exception:
            total = 0

        self._model_total_weight_bytes = int(total)
        return int(self._model_total_weight_bytes)

    def _device_weight_capacity_bytes(self, dev: DeviceSpec) -> int:
        """Best-effort estimate of a device's available space for *weights* (bytes)."""
        # Prefer buffer cache capacity when available, especially for PIM weight-cache limits.
        try:
            cache = getattr(self.buffer, "device_cache", {}).get(dev.name, None)  # type: ignore[attr-defined]
        except Exception:
            cache = None
        if cache is not None:
            for a in ("capacity", "cap", "max_bytes", "limit"):
                try:
                    v = getattr(cache, a, None)
                    if isinstance(v, (int, float)) and float(v) > 0:
                        return int(v)
                except Exception:
                    continue

        # Label-level budget, commonly pim_weight_capacity_bytes.
        if str(getattr(dev, "type", "")) == "pim":
            for a in ("pim_weight_capacity_bytes", "pim_weight_budget_bytes", "pim_static_weight_bytes"):
                try:
                    v = getattr(self.label, a, None)
                    if isinstance(v, (int, float)) and float(v) > 0:
                        return int(v)
                except Exception:
                    continue
        if str(getattr(dev, "type", "")) == "npu":
            for a in ("npu_weight_capacity_bytes", "npu_weight_budget_bytes", "npu_static_weight_bytes"):
                try:
                    v = getattr(self.label, a, None)
                    if isinstance(v, (int, float)) and float(v) > 0:
                        return int(v)
                except Exception:
                    continue

        # Fallback: use physical memory as an upper bound.
        try:
            phy = int(float(getattr(dev, "mem_capacity_GB", 0.0)) * 1024**3)
            return max(0, phy)
        except Exception:
            return 0

    def _device_can_hold_all_weights(self, dev, total_weight_bytes):
        cap = self._device_weight_capacity_bytes(dev)
        if dev.type == "pim":
            pim_cnt = max(1, len(self.cluster.devices_by_type("pim")))
            return cap >= math.ceil(total_weight_bytes / pim_cnt)
        return cap >= total_weight_bytes


    def _update_plan_hints(self, assignments: Mapping[str, str], scheduled: set[str]) -> None:
        """Update bounded plan-hint mapping σ'(·) for near-future nodes."""
        max_keep = int(SCHED_JOINT_LK_PLAN_HINT_MAX) if SCHED_JOINT_LK_PLAN_HINT_MAX is not None else 0
        if max_keep <= 0:
            return

        # Drop scheduled entries first.
        for n in list(self._plan_hint.keys()):
            if n in scheduled:
                self._plan_hint.pop(n, None)

        for nid, dev_name in assignments.items():
            if nid in scheduled:
                continue
            # Refresh insertion order.
            if nid in self._plan_hint:
                self._plan_hint.pop(nid, None)
            self._plan_hint[nid] = str(dev_name)

        while len(self._plan_hint) > max_keep:
            # pop oldest
            oldest = next(iter(self._plan_hint))
            self._plan_hint.pop(oldest, None)

    def _update_weight_hint_after_commit(self, g: TaskGraph, nid: str, hinted_type: str) -> None:
        """
        Record: weight_id -> last chosen device type.
        """
        try:
            node = g.nodes[nid]
        except Exception:
            return
        wid = self._node_weight_id(node)
        wsize = int(self._node_weight_size(node))
        if not wid or wsize <= 0:
            return
        self._weight_plan_hint[str(wid)] = str(hinted_type).lower()
    # -----------------------------
    # Scheduling
    # -----------------------------
    @override
    def schedule(self, g: TaskGraph, phase: str) -> List[ScheduledTask]:
        if getattr(self, "stats", None):
            self.stats.set_phase(phase)

        self.reset_state(clear_caches=False)
        idx = self._get_graph_index(g)

        # Step 1: HEFT priority (upward ranks).
        self._upward_rank(g, phase=phase)
        rank_u = idx.rank_u_by_phase.get(phase, {})

        topo_pos = {nid: i for i, nid in enumerate(idx.topo)}
        remaining_preds = {nid: len(idx.preds[nid]) for nid in idx.nodes}

        # Ready set.
        ready: set[str] = {nid for nid in idx.nodes if remaining_preds[nid] == 0}

        schedule: List[ScheduledTask] = []
        scheduled: set[str] = set()

        # Config knobs (match paper notations).
        H = int(SCHED_JOINT_LK_H)
        gamma = float(SCHED_JOINT_LK_GAMMA)
        use_lookahead = bool(SCHED_JOINT_LK_ENABLE) and H > 1 and gamma > 0.0

        # Multi-device selection: when there are multiple executors, don't rely on rank ordering.
        # Instead, pick from the whole READY set by the lookahead chain score.
        exec_types = tuple(self._executor_device_types())
        num_exec = int(sum(len(self.cluster.devices_by_type(t)) for t in exec_types))
        use_chain_select = bool(use_lookahead) and num_exec > 1

        def _candidates_for(nid: str, node: TaskNode) -> Tuple[List[DeviceSpec], bool]:
            """Return (device_candidates, is_kv_write)."""

            # Communication primitives: always schedule on host (control/comm-only op).
            if self._is_comm_node(node):
                return [self.cost.get_host_device()], False

            name_up = str(getattr(node, "name", "")).upper()
            is_kv_write = name_up in ("K_WRITE", "V_WRITE", "KV_WRITE")

            pinned_dev: Optional[DeviceSpec] = None
            if is_kv_write:
                pinned_dev = self._preferred_kv_write_device(g, nid)
                if pinned_dev is not None and (not self._node_allowed_on(node, pinned_dev)):
                    pinned_dev = None

            if pinned_dev is not None:
                return [pinned_dev], is_kv_write

            cands: List[DeviceSpec] = []
            for dev_type in exec_types:
                for dev in self.cluster.devices_by_type(dev_type):
                    try:
                        if self._node_allowed_on(node, dev):
                            cands.append(dev)
                    except Exception:
                        cands.append(dev)
            return cands, is_kv_write

        def _best_assignment_for(nid: str) -> Tuple[float, float, str, Optional[DeviceSpec], Optional[dict], Dict[str, str]]:
            """Return (best_score, best_eft, best_mode, best_dev, best_hy_est, best_hint_assign)."""
            node = g.nodes[nid]
            cands, is_kv_write = _candidates_for(nid, node)
            if not cands:
                return (float("inf"), float("inf"), "DEV", None, None, {})

            # Lookahead chain for this ready node.
            chain = [nid]
            if use_lookahead:
                chain = self._build_lookahead_chain(g, nid, idx, rank_u, phase=phase, H=H)

            allow_npu = any(d.type == "npu" for d in cands)
            allow_pim = any(d.type == "pim" for d in cands)

            hy_est: Optional[dict] = None
            best_score = float("inf")
            best_eft = float("inf")
            best_mode: str = "DEV"
            best_dev: Optional[DeviceSpec] = None
            best_hy: Optional[dict] = None
            best_hint: Dict[str, str] = {}

            for dev in cands:
                _, eft = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=False)
                eft = float(eft)
                if not math.isfinite(eft):
                    continue

                # Strategy 1: lookahead window estimate with consistency penalty.
                window_est = eft
                hint_assign: Dict[str, str] = {}
                if use_lookahead and len(chain) > 1:
                    window_est, hint_assign = self._lookahead_window_estimate(
                        g, chain=chain, first_dev=dev, first_eft=eft, phase=phase
                    )

                # Strategy 2: short-horizon weight/device consistency bias.
                bias = float(self._weight_reuse_bias(g, idx, nid, dev, phase=phase))

                # Strategy 3: long-horizon decode amortization bias (cross-token only).
                decode_phase_bias = float(self._decode_phase_amortization_bias(g, nid, dev, phase=phase))

                score = float((1.0 - gamma) * eft + gamma * float(window_est) + bias + decode_phase_bias)

                # Tie-breakers: prefer lower EFT, then random.
                if (score < best_score) or (
                    abs(score - best_score) < 1e-9
                    and (eft < best_eft or (abs(eft - best_eft) < 1e-9 and self._rng.random() < 0.5))
                ):
                    best_score = score
                    best_eft = eft
                    best_mode = "DEV"
                    best_dev = dev
                    best_hy = None
                    best_hint = dict(hint_assign)

            return best_score, best_eft, best_mode, best_dev, best_hy, best_hint

        while ready:
            # Pick a ready node.
            if use_chain_select:
                best_tuple = (float("inf"), float("inf"), float("inf"), float("inf"))
                pick: Optional[str] = None
                pick_res: Optional[Tuple[float, float, str, Optional[DeviceSpec], Optional[dict], Dict[str, str]]] = None
                for nid in list(ready):
                    score, eft, mode, dev_obj, hy_est, hint = _best_assignment_for(nid)
                    # Tie-break: prefer higher rank_u when scores equal.
                    key = (float(score), float(eft), -float(rank_u.get(nid, 0.0)), float(topo_pos.get(nid, 0)))
                    if key < best_tuple:
                        best_tuple = key
                        pick = nid
                        pick_res = (score, eft, mode, dev_obj, hy_est, hint)
                if pick is None or pick_res is None:
                    raise RuntimeError("No schedulable ready node (all placements infeasible)")
                nid = pick
                _, _, best_mode, best_choice, best_hy, best_hint_assignments = pick_res
            else:
                # Classic HEFT: pick by rank_u.
                nid = max(ready, key=lambda n: (float(rank_u.get(n, 0.0)), -int(topo_pos.get(n, 0))))
                node = g.nodes[nid]
                _, _, best_mode, best_choice, best_hy, best_hint_assignments = _best_assignment_for(nid)

            ready.discard(nid)
            if nid in scheduled:
                continue

            node = g.nodes[nid]

            # Record lookahead-chain weight importance for this *selected* node only.
            if use_lookahead:
                try:
                    chain_sel = self._build_lookahead_chain(g, nid, idx, rank_u, phase=phase, H=H)
                    for cnid in chain_sel:
                        try:
                            cw = self._node_weight_id(g.nodes[cnid])
                            if cw:
                                self._weight_chain_hits[str(cw)] += 1.0
                        except Exception:
                            continue
                except Exception:
                    pass
            # Commit
            scheduled.add(nid)

            if best_choice is None:
                raise RuntimeError(f"No feasible placement found for node {nid}")
            dev = best_choice
            start, finish = self._earliest_finish_on_device(g, nid, dev, self.label, phase, commit=True)
            is_comm = self._is_comm_node(node)
            trace_dev = "COMM" if is_comm else dev.name
            trace_dev_type = "comm" if is_comm else dev.type
            schedule.append(ScheduledTask(nid, trace_dev, float(start), float(finish)))
            self._after_commit_consume_predecessors(g, nid)

            # Record the per-weight hint.
            self._update_weight_hint_after_commit(g, nid, str(getattr(dev, "type", "")))

            # Stats
            if getattr(self, "stats", None):
                op_name = getattr(node, "attrs", {}).get("op") if hasattr(node, "attrs") else None
                op_name = op_name or getattr(node, "name", "")
                try:
                    self._log_scheduled_op_trace(
                        nid=nid,
                        op=op_name,
                        device=trace_dev,
                        device_type=trace_dev_type,
                        start=float(start),
                        end=float(finish),
                        mode=("COMM" if is_comm else "COMMAWARE"),
                    )
                except Exception:
                    pass

            # Update near-future plan hints from best lookahead pattern.
            if use_lookahead and best_hint_assignments:
                self._update_plan_hints(best_hint_assignments, scheduled)
            else:
                self._update_plan_hints({}, scheduled)

            # Unlock successors.
            for v in idx.succs.get(nid, ()):  # type: ignore[assignment]
                remaining_preds[v] -= 1
                if remaining_preds[v] == 0 and v not in scheduled:
                    ready.add(v)

        if len(scheduled) != len(idx.nodes):
            missing = [n for n in idx.nodes if n not in scheduled]
            raise RuntimeError(
                f"Schedule failed: graph may have cycles or missing deps; unscheduled nodes: {missing[:16]}"
            )
        return schedule



__all__ = ["BifocalScheduler"]
