from __future__ import annotations
from typing import Dict, Optional, Set, List, Tuple
from dataclasses import dataclass, field
import math
import itertools
from config import (
    GLOBAL_BUFFER_BYTES, ALLOW_EVICTION, ENFORCE_AT_LEAST_ONE_COPY,
    FORMAT_SIZE_MULTIPLIER, FORMAT_PLANNER_ALGO,
    FORMAT_PLANNER_MEMORY_BYTE_PENALTY, FORMAT_PLANNER_BEAM_WIDTH,
    
)
from utils import logger

@dataclass
class WeightRecord:
    size_bytes: int           # ND size
    formats: Set[str] = field(default_factory=set)
    future_uses: int = 0

class BufferManager:
    """
    HOST DRAM（GB）中的权重常驻与多格式副本管理。
    不再强制 ND 必存；每个权重至少要有一种格式。
    """
    def __init__(self, capacity_bytes: int = GLOBAL_BUFFER_BYTES) -> None:
        self.capacity = capacity_bytes
        self._weights: Dict[str, WeightRecord] = {}
        self._bytes_used: int = 0

    # ------- query -------
    def bytes_used(self) -> int: return self._bytes_used
    def bytes_free(self) -> int: return max(0, self.capacity - self._bytes_used)
    def has_weight(self, w: str) -> bool: return w in self._weights
    def weight_formats(self, w: str) -> Set[str]:
        return set(self._weights[w].formats) if w in self._weights else set()
    def weight_size(self, w: str, fmt: Optional[str] = None) -> int:
        if w not in self._weights:
            raise KeyError(f"Weight {w} not in buffer")
        sz_nd = self._weights[w].size_bytes
        if fmt is None: return sz_nd
        return int(sz_nd * FORMAT_SIZE_MULTIPLIER[fmt])
    def formats_map(self) -> Dict[str, Set[str]]:
        return {w: set(rec.formats) for w, rec in self._weights.items()}

    # ------- mutate -------
    def _alloc_bytes(self, n: int) -> bool:
        if n <= self.bytes_free():
            self._bytes_used += n
            return True
        return False

    def _free_bytes(self, n: int) -> None:
        self._bytes_used -= n
        if self._bytes_used < 0: self._bytes_used = 0

    def place_weight(self, w: str, size_bytes_nd: int, initial_format: str = "ND") -> None:
        """放入一种格式（首次创建）。不再强制 ND；由调用者决定 initial_format。"""
        if w in self._weights:
            return
        need = int(size_bytes_nd * FORMAT_SIZE_MULTIPLIER[initial_format])
        if not self._alloc_bytes(need):
            raise MemoryError(f"HOST OOM placing weight {w} as {initial_format}. need={need}, free={self.bytes_free()}")
        self._weights[w] = WeightRecord(size_bytes=size_bytes_nd, formats={initial_format})
        logger.debug(f"[HOST] place {w}:{initial_format}, used={self._bytes_used}/{self.capacity}")

    def ensure_format(self, w: str, fmt: str) -> bool:
        if w not in self._weights: raise KeyError(f"Weight {w} not present")
        rec = self._weights[w]
        if fmt in rec.formats: return True
        need = int(rec.size_bytes * FORMAT_SIZE_MULTIPLIER[fmt])
        if self._alloc_bytes(need):
            rec.formats.add(fmt)
            logger.debug(f"[HOST] +format {w}:{fmt}, +{need} bytes (used={self._bytes_used})")
            return True
        if ALLOW_EVICTION:
            # 只在同一权重内部逐个丢副本（至少保留一个）
            for f in sorted([x for x in rec.formats if x != fmt], key=lambda x: -FORMAT_SIZE_MULTIPLIER[x]):
                if ENFORCE_AT_LEAST_ONE_COPY and len(rec.formats) <= 1: break
                rec.formats.remove(f)
                freed = int(rec.size_bytes * FORMAT_SIZE_MULTIPLIER[f])
                self._free_bytes(freed)
                logger.debug(f"[HOST] evict {w}:{f}, -{freed} bytes")
                if self._alloc_bytes(need):
                    rec.formats.add(fmt)
                    logger.debug(f"[HOST] +format {w}:{fmt} (after evict), +{need}")
                    return True
        return False

    # ---------- Preload from plan (for next round) ----------
    def preload_plan(self, weights_size: Dict[str, int], plan: Dict[str, List[str]]) -> None:
        """
        根据上一轮的规划，把每个 weight 的格式副本预放入 HOST。
        如果超容量，会抛 MemoryError（你也可以在调用前先估算）。
        """
        for w, fmts in plan.items():
            if not fmts: continue
            base = fmts[0]
            self.place_weight(w, weights_size[w], initial_format=base)
            for f in fmts[1:]:
                ok = self.ensure_format(w, f)
                if not ok:
                    raise MemoryError(f"Preload plan exceeds capacity when adding {w}:{f}")

    # ---------- helpers shared by planners ----------
    def _build_baseline(self,
                        usage_by_format: Dict[str, Dict[str, int]],
                        weights_size: Dict[str, int]):
        """
        为每个权重选择单一基线格式（不强制ND），最小化全局转换时延（仅该权重视角）。
        返回:
          plan: {w: [baseline_fmt]}
          per_use_overhead: {w: {req_fmt: per_use_time_with_baseline}}
          baseline_bytes: int
        """
        from cost_model import CostModel
        cm = CostModel()
        plan: Dict[str, List[str]] = {}
        per_use_overhead: Dict[str, Dict[str, float]] = {}
        baseline_bytes = 0

        def candidate_formats(use_map: Dict[str,int]) -> Set[str]:
            s = set(use_map.keys())
            s.add("ND")
            return s

        for w, use_map in usage_by_format.items():
            size = weights_size[w]
            cands = candidate_formats(use_map)
            best_fmt = None
            best_sum = float("inf")
            best_per_use = {}

            for fmt in cands:
                cur_sum = 0.0
                per_use_map = {}
                for req_fmt, cnt in use_map.items():
                    t = cm.format_conversion_time(fmt, req_fmt, size) if req_fmt != fmt else 0.0
                    per_use_map[req_fmt] = t
                    cur_sum += t * cnt
                if cur_sum < best_sum:
                    best_sum = cur_sum
                    best_fmt = fmt
                    best_per_use = per_use_map

            plan[w] = [best_fmt]
            per_use_overhead[w] = best_per_use
            baseline_bytes += int(size * FORMAT_SIZE_MULTIPLIER[best_fmt])

        if baseline_bytes > self.capacity:
            raise MemoryError(f"Baseline formats exceed capacity: need={baseline_bytes}, cap={self.capacity}")

        return plan, per_use_overhead, baseline_bytes

    def _enumerate_bundles_for_weight(self,
                                      baseline_fmt: str,
                                      per_use_old: Dict[str, float],
                                      usage_map: Dict[str,int],
                                      size: int,
                                      penalty_per_byte: float):
        """
        对单个权重枚举“基线 + 额外格式集合”的所有 bundle：
        - 额外格式集合S取自 (需求格式 ∪ {ND}) \ {baseline}
        - bundle 的“附加字节”为 S 中各格式字节之和
        - “价值”为 相对基线的延迟降低（秒） - penalty*bytes
        返回 list[ (add_bytes:int, value:float, extras:list[str]) ]
        """
        from cost_model import CostModel
        cm = CostModel()
        cands = set(usage_map.keys()) | {"ND"}
        add_formats = sorted(list(cands - {baseline_fmt}))

        bundles = []
        # 包含空集合（仅基线） ->  add_bytes=0, value=0
        bundles.append((0, 0.0, []))

        # 枚举所有额外格式子集
        for r in range(1, len(add_formats)+1):
            for subset in itertools.combinations(add_formats, r):
                add_bytes = sum(int(size * FORMAT_SIZE_MULTIPLIER[f]) for f in subset)
                # 计算加入后每个 req_fmt 的 per_use 新成本 = min( baseline->req, min_{f in subset} f->req )
                red = 0.0
                for req_fmt, cnt in usage_map.items():
                    old_t = per_use_old.get(req_fmt, 0.0)
                    new_t = old_t
                    for f in subset:
                        cand_t = 0.0 if f == req_fmt else cm.format_conversion_time(f, req_fmt, size)
                        if cand_t < new_t:
                            new_t = cand_t
                    if new_t < old_t:
                        red += (old_t - new_t) * cnt
                value = red - penalty_per_byte * add_bytes
                bundles.append((add_bytes, value, list(subset)))
        return bundles

    # ---------- planners ----------
    def _plan_greedy(self,
                     baseline_plan: Dict[str, List[str]],
                     per_use_overhead: Dict[str, Dict[str, float]],
                     usage_by_format: Dict[str, Dict[str,int]],
                     weights_size: Dict[str,int],
                     remaining_capacity: int,
                     objective: str,
                     memory_byte_penalty: float) -> Tuple[Dict[str, List[str]], int]:
        """
        原有“基线 + 贪心增量”，每次选最优单位收益项；加入 ND 会缩短其它转换路径。
        """
        from cost_model import CostModel
        cm = CostModel()

        plan = {w: list(fmts) for w, fmts in baseline_plan.items()}
        total_bytes = sum(int(weights_size[w]*FORMAT_SIZE_MULTIPLIER[fmts[0]]) for w,fmts in baseline_plan.items())

        if objective == "memory":
            return plan, total_bytes  # 只保留基线

        while True:
            best_gain = 0.0
            best_choice: Optional[Tuple[str, str]] = None

            for w, use_map in usage_by_format.items():
                size = weights_size[w]
                for fmt in (set(use_map.keys()) | {"ND"}):
                    if fmt in plan[w]:  # 已有
                        continue
                    need = int(size * FORMAT_SIZE_MULTIPLIER[fmt])
                    if need > remaining_capacity:
                        continue
                    # 延迟收益：把该 weight 对应 req_fmt 的 per_use 降到 min(old, fmt->req)
                    red = 0.0
                    for req_fmt, cnt in use_map.items():
                        old_t = per_use_overhead[w].get(req_fmt, 0.0)
                        new_t = 0.0 if req_fmt == fmt else cm.format_conversion_time(fmt, req_fmt, size)
                        if new_t < old_t:
                            red += (old_t - new_t) * cnt
                    gain = red - (0.0 if objective=="latency" else memory_byte_penalty*need)
                    if gain > best_gain + 1e-15:
                        best_gain = gain
                        best_choice = (w, fmt)

            if not best_choice or best_gain <= 0.0:
                break

            w, fmt = best_choice
            size = weights_size[w]
            need = int(size * FORMAT_SIZE_MULTIPLIER[fmt])
            plan[w].append(fmt)
            remaining_capacity -= need
            total_bytes += need

            # 更新 per_use_overhead：各 req_fmt 取 min(旧值, fmt->req)
            for req_fmt, old_t in list(per_use_overhead[w].items()):
                new_t = 0.0 if req_fmt == fmt else cm.format_conversion_time(fmt, req_fmt, size)
                if new_t < old_t:
                    per_use_overhead[w][req_fmt] = new_t

        return plan, total_bytes

    def _plan_dp_mckp(self,
                      baseline_plan: Dict[str, List[str]],
                      per_use_overhead: Dict[str, Dict[str, float]],
                      usage_by_format: Dict[str, Dict[str,int]],
                      weights_size: Dict[str,int],
                      remaining_capacity: int,
                      objective: str,
                      memory_byte_penalty: float) -> Tuple[Dict[str, List[str]], int]:
        """
        多选背包（Multi-Choice Knapsack, MCKP）：
        - 每个权重选择一个“bundle”（基线 + 额外若干格式）
        - bundle 成本 = 额外字节；价值 = 延迟降低 - penalty*bytes
        - 预算 = 剩余容量（baseline 之外）
        - 用容量离散化（单位数受 FORMAT_PLANNER_DP_MAX_UNITS 控制）做伪多项式 DP
        """
        if objective == "memory":
            # 只保留基线
            total_bytes = sum(int(weights_size[w]*FORMAT_SIZE_MULTIPLIER[fmts[0]]) for w,fmts in baseline_plan.items())
            return {w:list(fmts) for w,fmts in baseline_plan.items()}, total_bytes

        # 离散化容量
        if remaining_capacity <= 0:
            total_bytes = sum(int(weights_size[w]*FORMAT_SIZE_MULTIPLIER[fmts[0]]) for w,fmts in baseline_plan.items())
            return {w:list(fmts) for w,fmts in baseline_plan.items()}, total_bytes

        max_units = max(1, FORMAT_PLANNER_DP_MAX_UNITS)
        unit = max(1, remaining_capacity // max_units)
        cap_units = remaining_capacity // unit

        weights = list(baseline_plan.keys())
        # 为每个 weight 枚举 bundles
        bundles_per_weight: List[List[Tuple[int,float,List[str]]]] = []
        for w in weights:
            size = weights_size[w]
            baseline_fmt = baseline_plan[w][0]
            use_map = usage_by_format[w]
            per_use_old = per_use_overhead[w]
            bundles = self._enumerate_bundles_for_weight(
                baseline_fmt, per_use_old, use_map, size,
                0.0 if objective=="latency" else memory_byte_penalty
            )
            # 把 bytes 转成单位
            bundles_units = []
            for add_bytes, value, extras in bundles:
                k = add_bytes // unit
                if add_bytes % unit != 0:
                    k += 1  # ceil
                bundles_units.append((k, value, extras))
            bundles_per_weight.append(bundles_units)

        # DP: 层=权重索引，列=容量单位
        NEG_INF = -1e100
        dp_layers = [[NEG_INF]*(cap_units+1)]
        dp_layers[0][0] = 0.0
        choice_idx: List[List[int]] = []      # choice_idx[i][c] = 选了第 i 个权重的哪个 bundle
        choice_cost: List[List[int]] = []     # 对应消耗了多少单位

        for i, bundles in enumerate(bundles_per_weight, start=1):
            prev = dp_layers[i-1]
            cur = [NEG_INF]*(cap_units+1)
            ch_i = [-1]*(cap_units+1)
            cc_i = [0]*(cap_units+1)
            for c_prev in range(cap_units+1):
                if prev[c_prev] <= NEG_INF/2:
                    continue
                for idx, (k_units, val, extras) in enumerate(bundles):
                    c = c_prev + k_units
                    if c > cap_units:
                        continue
                    v = prev[c_prev] + val
                    if v > cur[c]:
                        cur[c] = v
                        ch_i[c] = idx
                        cc_i[c] = k_units
            dp_layers.append(cur)
            choice_idx.append(ch_i)
            choice_cost.append(cc_i)

        # 回溯
        last = dp_layers[-1]
        best_c = max(range(cap_units+1), key=lambda c: last[c])
        plan = {w: list(fmts) for w, fmts in baseline_plan.items()}
        # 选中的 extras 合并入 plan
        for i in range(len(weights), 0, -1):
            ch = choice_idx[i-1][best_c]
            k = choice_cost[i-1][best_c]
            if ch == -1:
                # 不应发生（至少有空bundle）
                pass
            else:
                # 找到 extras
                size = weights_size[weights[i-1]]
                baseline_fmt = plan[weights[i-1]][0]
                use_map = usage_by_format[weights[i-1]]
                per_use_old = per_use_overhead[weights[i-1]]
                # 重新枚举拿回extras
                bundles = self._enumerate_bundles_for_weight(
                    baseline_fmt, per_use_old, use_map, size,
                    0.0 if objective=="latency" else memory_byte_penalty
                )
                extras = bundles[ch][2]
                plan[weights[i-1]].extend(extras)
            best_c -= k
            if best_c < 0: best_c = 0

        total_bytes = sum(int(weights_size[w]*FORMAT_SIZE_MULTIPLIER[f]) for w,fmts in plan.items() for f in fmts)
        return plan, total_bytes

    def _plan_beam_mckp(self,
                        baseline_plan: Dict[str, List[str]],
                        per_use_overhead: Dict[str, Dict[str, float]],
                        usage_by_format: Dict[str, Dict[str,int]],
                        weights_size: Dict[str,int],
                        remaining_capacity: int,
                        objective: str,
                        memory_byte_penalty: float) -> Tuple[Dict[str, List[str]], int]:
        """
        束搜索（beam）在“按权重依次选择 bundle”的搜索树上保留前K个状态（K=beam width）。
        - 状态 = (已处理权重索引i, 已用字节, 价值, 计划的extras选择)
        - 每个扩展：为第 i 个权重选择一个 bundle（基线+extras），若不超预算便加入候选
        - 下一层保留前K个价值最高的状态
        """
        if objective == "memory" or remaining_capacity <= 0:
            total_bytes = sum(int(weights_size[w]*FORMAT_SIZE_MULTIPLIER[fmts[0]]) for w,fmts in baseline_plan.items())
            return {w:list(fmts) for w,fmts in baseline_plan.items()}, total_bytes

        beam_width = max(1, FORMAT_PLANNER_BEAM_WIDTH)
        weights = list(baseline_plan.keys())

        # 预生成每个 weight 的 bundles（真正字节，不离散）
        per_weight_bundles: List[List[Tuple[int,float,List[str]]]] = []
        for w in weights:
            size = weights_size[w]
            baseline_fmt = baseline_plan[w][0]
            use_map = usage_by_format[w]
            per_use_old = per_use_overhead[w]
            bundles = self._enumerate_bundles_for_weight(
                baseline_fmt, per_use_old, use_map, size,
                0.0 if objective=="latency" else memory_byte_penalty
            )
            per_weight_bundles.append(bundles)

        # 初始状态
        State = Tuple[int, int, float, List[List[str]]]  # (idx, bytes, value, extras_each_weight)
        init_extras = [[] for _ in weights]
        beam: List[State] = [(0, 0, 0.0, init_extras)]

        for i in range(len(weights)):
            new_beam: List[State] = []
            for (idx, used_bytes, value, extras_list) in beam:
                # 扩展 i 的所有 bundle
                for add_bytes, val, extras in per_weight_bundles[i]:
                    nb = used_bytes + add_bytes
                    if nb > remaining_capacity:
                        continue
                    nv = value + val
                    new_extras = [list(x) for x in extras_list]
                    new_extras[i] = extras
                    new_beam.append((i+1, nb, nv, new_extras))
            if not new_beam:
                # 若无可扩展，强制使用“空 bundle”继续（不会超预算）
                for (idx, used_bytes, value, extras_list) in beam:
                    new_extras = [list(x) for x in extras_list]
                    new_beam.append((i+1, used_bytes, value, new_extras))
            # 选取前 K
            new_beam.sort(key=lambda s: (s[2], -s[1]), reverse=True)
            beam = new_beam[:beam_width]

        # 取最优
        best = max(beam, key=lambda s: (s[2], -s[1]))
        _, add_bytes, _, extras_list = best

        plan = {w: list(fmts) for w, fmts in baseline_plan.items()}
        for w, extras in zip(weights, extras_list):
            plan[w].extend(extras)

        total_bytes = sum(int(weights_size[w]*FORMAT_SIZE_MULTIPLIER[f]) for w,fmts in plan.items() for f in fmts)
        return plan, total_bytes

    # ---------- ENTRY: Format planner for NEXT round ----------
    def plan_weight_formats(self,
                            usage_by_format: Dict[str, Dict[str, int]],
                            weights_size: Dict[str, int],
                            objective: str = "balanced",
                            memory_byte_penalty: float = FORMAT_PLANNER_MEMORY_BYTE_PENALTY
                            ) -> Tuple[Dict[str, List[str]], int]:
        """
        在【调度结束后】调用：基于本轮统计 usage_by_format（每个权重被各目标格式使用次数），
        规划下一轮在 GB 中应常驻的格式集合（每个权重至少一种，不强制 ND）。
        返回：(plan, total_bytes)，其中 plan[w]=[fmt1, fmt2,...]。
        该函数不会修改当前 Buffer 内容。
        """
        # 1) 单权重基线
        baseline_plan, per_use_overhead, baseline_bytes = self._build_baseline(usage_by_format, weights_size)
        remaining_capacity = max(0, self.capacity - baseline_bytes)

        # 2) 三种规划器可选
        algo = (FORMAT_PLANNER_ALGO or "greedy").lower()
        if algo == "greedy":
            plan, total_bytes = self._plan_greedy(
                baseline_plan, per_use_overhead, usage_by_format, weights_size,
                remaining_capacity, objective, memory_byte_penalty
            )
        elif algo == "dp":
            plan, total_bytes = self._plan_dp_mckp(
                baseline_plan, per_use_overhead, usage_by_format, weights_size,
                remaining_capacity, objective, memory_byte_penalty
            )
        elif algo == "beam":
            plan, total_bytes = self._plan_beam_mckp(
                baseline_plan, per_use_overhead, usage_by_format, weights_size,
                remaining_capacity, objective, memory_byte_penalty
            )
        else:
            raise ValueError(f"Unknown FORMAT_PLANNER_ALGO: {FORMAT_PLANNER_ALGO}")

        return plan, total_bytes
