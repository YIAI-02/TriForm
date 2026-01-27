from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Generic byte-based LRU cache (used for weight caches on all devices)
# ---------------------------------------------------------------------------


@dataclass
class LRUCache:
    """Simple byte-capacity LRU with optional pinning.
    
    - `items`   : key -> size (bytes)
    - `order`   : LRU order, index 0 is least-recently-used
    - `pinned`  : keys that are never evicted by the cache itself
    - `used`    : sum of all item sizes (bytes)
    """
    capacity: int
    items: Dict[Hashable, int] = field(default_factory=dict)
    order: List[Hashable] = field(default_factory=list)
    pinned: Set[Hashable] = field(default_factory=set)
    used: int = 0

    # ---- basic helpers ----------------------------------------------------
    def has(self, key: Hashable) -> bool:
        return key in self.items

    def touch(self, key: Hashable) -> None:
        """Mark a key as most-recently-used if it exists."""
        if key not in self.items:
            return
        self.order.remove(key)
        self.order.append(key)

    def pin(self, key: Hashable) -> None:
        if key in self.items:
            self.pinned.add(key)

    def unpin(self, key: Hashable) -> None:
        self.pinned.discard(key)

    # ---- eviction core ----------------------------------------------------
    def _evict_one_lru(self) -> int:
        """Evict a single least-recently-used *unpinned* item.

        Returns freed bytes, or 0 if nothing can be evicted.
        """
        safety = len(self.order) + 2
        while self.order and safety > 0:
            safety -= 1
            victim = self.order.pop(0)
            if victim in self.pinned:
                # move pinned entry to the back and continue
                self.order.append(victim)
                # if every entry is pinned we will exit shortly
                if self.order and all(k in self.pinned for k in self.order):
                    break
                continue
            size = int(self.items.pop(victim, 0))
            if size > 0:
                self.used = max(0, self.used - size)
                return size
        return 0

    def evict_bytes(self, bytes_needed: int) -> int:
        """Evict unpinned LRU entries until at least `bytes_needed` bytes freed.

        Returns the number of bytes actually freed (<= bytes_needed).
        """
        need = max(0, int(bytes_needed))
        if need <= 0:
            return 0
        freed = 0
        while freed < need:
            got = self._evict_one_lru()
            if got <= 0:
                break
            freed += got
        return freed

    # ---- public mutation API ----------------------------------------------

    def add(self, key: Hashable, size: int, *, pinned: bool = False) -> bool:
        """Insert/update an item with `size` bytes.
        
        If necessary, evict LRU *unpinned* items to respect `capacity`.
        Returns True on success, False if the item cannot be stored at all
        (e.g. it is larger than total capacity and everything else is pinned).
        """
        size = max(0, int(size))
        # Fast path: update existing item
        if key in self.items:
            old = int(self.items[key])
            delta = size - old
            self.items[key] = size
            self.used = max(0, self.used + delta)
            self.touch(key)
            if pinned:
                self.pinned.add(key)
            return True

        if size == 0:
            # nothing to store
            return True

        # If the single item is larger than capacity we cannot place it.
        if self.capacity > 0 and size > self.capacity:
            # best-effort: clear all evictable items first
            self.evict_bytes(self.capacity)
            if size > self.capacity:
                return False

        # Ensure space: evict until fits
        while self.capacity > 0 and self.used + size > self.capacity:
            freed = self._evict_one_lru()
            if freed <= 0:
                break

        if self.capacity == 0 or self.used + size > self.capacity:
            # still cannot fit
            return False

        self.items[key] = size
        self.order.append(key)
        self.used += size
        if pinned:
            self.pinned.add(key)
        return True

    def set_capacity(self, capacity: int) -> int:
        """Change capacity and evict items if we are above the new limit.

        Returns the number of bytes freed.
        """
        self.capacity = max(0, int(capacity or 0))
        over = max(0, self.used - self.capacity)
        return self.evict_bytes(over)


# ---------------------------------------------------------------------------
# PIM runtime bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class PIMRuntimeState:
    """Per-PIM-device runtime state.
        PIM runtime budget only tracks activation + weight cache.
    """
    phy_bytes: int                          # physical device capacity (bytes)
    limit_bytes: int                        # runtime budget for kv + act + weights (bytes)
    kv_in_pim: bool = False                 # True => KV logically on PIM
    kv_reserved_bytes: int = 0              # bytes reserved for KV on this device
    kv_used_bytes: int = 0                  # bytes currently used for KV on this device (>= reserved if modeled)
    act_used_bytes: int = 0                 # bytes currently held for activations


# ---------------------------------------------------------------------------
# Global memory / cache manager
# ---------------------------------------------------------------------------
@dataclass
class GlobalMemoryManager:
    """Host / device / PIM buffer manager.
    """
    # Host-side storage format per weight_id: 'ND' | 'npu-opt' | 'pim-opt'
    host_format: Dict[str, str] = field(default_factory=dict)

    # Conversion throughputs (GB/s) for format conversion
    conv_bw_GBs: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Per-device *weight* cache (LRU by bytes). For PIM devices this is the
    # portion of runtime memory that is allowed to be occupied by weights.
    device_cache: Dict[str, LRUCache] = field(default_factory=dict)

    # Optional statistics hooks
    stats_convert_bytes: Dict[Tuple[str, str], int] = field(default_factory=dict)
    stats_weight_loads: Dict[Tuple[str, str, str], int] = field(default_factory=dict)

    # Per-PIM runtime accounting
    pim_state: Dict[str, PIMRuntimeState] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Generic (format) helpers
    # ------------------------------------------------------------------
    def get_host_fmt(self, wid: str) -> str:
        return self.host_format.get(wid, "ND")

    def set_host_fmt(self, wid: str, fmt: str) -> None:
        self.host_format[wid] = fmt

    # ------------------------------------------------------------------
    # Weight cache helpers
    # ------------------------------------------------------------------
    def ensure_device_cache(self, dev_name: str, capacity_bytes: int) -> None:
        """Ensure a weight-cache exists for `dev_name` with given capacity."""
        cap = max(0, int(capacity_bytes or 0))
        cache = self.device_cache.get(dev_name)
        if cache is None:
            self.device_cache[dev_name] = LRUCache(capacity=cap)
        else:
            cache.set_capacity(cap)

    def is_cached(self, dev_name: str, wid: str) -> bool:
        cache = self.device_cache.get(dev_name)
        return cache.has(wid) if cache else False

    # ------------------------------------------------------------------
    # PIM registration / query
    # ------------------------------------------------------------------
    def register_pim_device(
        self,
        dev_name: str,
        *,
        phy_bytes: int,
        runtime_limit_bytes: int,
        kv_reserved_bytes: int = 0,
        kv_in_pim: bool = False,
        weight_cache_capacity_bytes: Optional[int] = None,
    ) -> None:

        phy_bytes = max(0, int(phy_bytes))
        runtime_limit_bytes = max(0, int(runtime_limit_bytes))
        kv_reserved_bytes = max(0, int(kv_reserved_bytes or 0))
        st = self.pim_state.get(dev_name)
        if st is None:
            st = PIMRuntimeState(
                phy_bytes=phy_bytes,
                limit_bytes=runtime_limit_bytes,
                kv_in_pim=bool(kv_in_pim),
                kv_reserved_bytes=kv_reserved_bytes,
                kv_used_bytes=(kv_reserved_bytes if kv_in_pim else 0),
            )
            self.pim_state[dev_name] = st
        else:
            st.phy_bytes = phy_bytes
            st.limit_bytes = runtime_limit_bytes
            st.kv_in_pim = bool(kv_in_pim)

            st.kv_reserved_bytes = kv_reserved_bytes
            st.kv_used_bytes = (kv_reserved_bytes if kv_in_pim else 0)

        if weight_cache_capacity_bytes is None:
            weight_cache_capacity_bytes = max(0, int(runtime_limit_bytes - (kv_reserved_bytes if kv_in_pim else 0)))

        self.ensure_device_cache(dev_name, int(weight_cache_capacity_bytes))

    def pim_used_bytes(self, dev_name: str) -> Tuple[int, int, int, int, int]:
        """Return (phy_bytes, kv_used, act_used, weight_used, total_used)."""
        st = self.pim_state.get(dev_name)
        if st is None:
            return (0, 0, 0, 0, 0)
        cache = self.device_cache.get(dev_name)
        weight_used = int(cache.used) if cache else 0
        act_used = int(st.act_used_bytes)
        kv_used = int(st.kv_used_bytes) if bool(st.kv_in_pim) else 0
        total = kv_used + act_used + weight_used
        return (int(st.phy_bytes), kv_used, act_used, weight_used, total)

    def reset_runtime_state(self) -> None:
        """Clear dynamic PIM runtime state (activations only). Does not change KV reservation/budgets."""
        for st in self.pim_state.values():
            st.act_used_bytes = 0

    # ------------------------------------------------------------------
    # PIM activation management
    # ------------------------------------------------------------------

    def _pim_weight_cache(self, dev_name: str) -> Optional[LRUCache]:
        return self.device_cache.get(dev_name)

    def _pim_weight_used(self, dev_name: str) -> int:
        cache = self._pim_weight_cache(dev_name)
        return int(cache.used) if cache else 0

    def _pim_evict_weights_bytes(self, dev_name: str, bytes_needed: int, *, commit: bool) -> int:
        """Evict weight-cache entries on a PIM device.
        """
        cache = self._pim_weight_cache(dev_name)
        if cache is None:
            return 0

        need = max(0, int(bytes_needed))
        if need <= 0:
            return 0

        if not commit:
            # dry-run: sum of all evictable (unpinned) bytes
            evictable = 0
            for key, sz in cache.items.items():
                if key in cache.pinned:
                    continue
                evictable += int(sz)
            return min(evictable, need)

        return cache.evict_bytes(need)

    def pim_reserve_activation(self, dev_name: str, bytes_needed: int, *, commit: bool) -> bool:
        """Reserve activation bytes on a PIM device.

                Policy: only activation + weight cache counted; evict weights if needed, never activations.
        """
        st = self.pim_state.get(dev_name)
        if st is None:
            return False

        bytes_needed = max(0, int(bytes_needed))
        if bytes_needed == 0:
            return True

        limit = int(st.limit_bytes)
        if limit <= 0:
            return False

        kv_used = int(st.kv_used_bytes) if bool(st.kv_in_pim) else 0
        act_before = int(st.act_used_bytes)
        act_after = act_before + bytes_needed
        weight_used = self._pim_weight_used(dev_name)

        # Simple fast-path: enough room without evictions
        if kv_used + act_after + weight_used <= limit:
            if commit:
                st.act_used_bytes = act_after
            return True

        # Need to evict some weights
        need_free = kv_used + act_after + weight_used - limit
        freed = self._pim_evict_weights_bytes(dev_name, need_free, commit=commit)

        if not commit:
            # dry-run: succeed iff in principle we can free enough weight bytes
            return freed >= need_free

        # Commit path: recompute and decide
        weight_after = self._pim_weight_used(dev_name)
        if kv_used + act_after + weight_after <= limit:
            st.act_used_bytes = act_after
            return True

        # Even after evicting all weights we still cannot fit this activation
        return False

    def pim_release_activation(self, dev_name: str, bytes_to_release: int, *, commit: bool = True) -> None:
        """Release activation bytes for `dev_name` (PIM only)."""
        if not commit:
            return
        st = self.pim_state.get(dev_name)
        if st is None:
            return
        delta = max(0, int(bytes_to_release))
        st.act_used_bytes = max(0, int(st.act_used_bytes) - delta)

    # ------------------------------------------------------------------
    # Weight caching (all devices)
    # ------------------------------------------------------------------

    def pim_cache_weight(self, dev_name: str, wid: str, size: int, *, pinned: bool, commit: bool) -> bool:
        st = self.pim_state.get(dev_name)
        if st is None:
            # Non-PIM path (or unregistered PIM): treat as a generic weight LRU.
            cache = self.device_cache.get(dev_name)
            if cache is None:
                # Best-effort default: at least large enough to hold this single weight.
                self.ensure_device_cache(dev_name, capacity_bytes=max(0, int(size)))
                cache = self.device_cache[dev_name]
            ok = cache.add(wid, int(size), pinned=bool(pinned))
            if ok:
                cache.touch(wid)
                if pinned:
                    cache.pin(wid)
            return bool(ok)

        size = max(0, int(size))
        if size == 0:
            return True

        limit = int(st.limit_bytes)
        if limit <= 0:
            return False
        if size > limit:
            # Single weight larger than total runtime budget: impossible.
            return False

        cache = self.device_cache.get(dev_name)
        if cache is None:
            self.ensure_device_cache(dev_name, capacity_bytes=limit)
            cache = self.device_cache[dev_name]

        kv_used = int(st.kv_used_bytes) if bool(st.kv_in_pim) else 0
        act_used = int(st.act_used_bytes)
        weight_used = int(cache.used)

        # Quick check: is there enough room *without* extra evictions?
        if kv_used + act_used + weight_used + size > limit:
            need_free = kv_used + act_used + weight_used + size - limit
            freed = self._pim_evict_weights_bytes(dev_name, need_free, commit=commit)
            if not commit:
                # dry-run: success iff we *could* free enough
                if freed < need_free:
                    return False
            else:
                # commit path: recompute used weight after eviction
                weight_used = self._pim_weight_used(dev_name)
                if kv_used + act_used + weight_used + size > limit:
                    # Still cannot fit even after evicting everything.
                    return False

        # At this point we know we can respect runtime limit; rely on
        # the inner LRU cache to enforce its own capacity.
        ok = cache.add(wid, size, pinned=pinned)
        if ok:
            cache.touch(wid)
            if pinned:
                cache.pin(wid)
        return ok

    def mark_cached(self, dev_name: str, wid: str, size: int, pinned: bool = False) -> None:
        """Notify the buffer manager that a weight has been cached.
        """
        if dev_name in self.pim_state:
            # PIM-aware path
            self.pim_cache_weight(dev_name, wid, int(size), pinned=bool(pinned), commit=True)
            return

        # Non-PIM path
        cache = self.device_cache.get(dev_name)
        if cache is None:
            # Best-effort default: at least large enough to hold this weight.
            self.ensure_device_cache(dev_name, capacity_bytes=max(0, int(size)))
            cache = self.device_cache[dev_name]
        cache.add(wid, int(size), pinned=pinned)
        cache.touch(wid)
        if pinned:
            cache.pin(wid)


