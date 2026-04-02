from __future__ import annotations
import os
import re
from config import attach_local_debug_filter
from dataclasses import dataclass,field
from typing import Dict, List, Tuple, Optional, Any, Iterable, OrderedDict, Hashable
try:  # pragma: no cover
    from typing import override  # type: ignore
except Exception:  # pragma: no cover
    from typing_extensions import override  # type: ignore
from collections import defaultdict
from collections import ChainMap
from collections.abc import Hashable, Mapping
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from plan_label import PlanLabel
from hardware import Cluster, DeviceSpec
from task_graph import TaskGraph, TaskNode, JointTaskGraph, JointNodeMeta
from cost_model import CostModel
from cost_model import _normalize_weight_format_token as _cm_normalize_weight_format_token
from cost_model import _resolve_npu_weight_conversion_steps as _cm_resolve_npu_weight_conversion_steps
from buffer_manager import GlobalMemoryManager, LRUCache
from config import (
    RANKU_INCLUDE_AVG_WEIGHT_LOAD,
    PIM_RUNTIME_LRU_THRESHOLD,
    SCHED_JOINT_LK_ENABLE,
    SCHED_JOINT_LK_H,
    SCHED_JOINT_LK_GAMMA,
    SCHED_JOINT_LK_CONSIST_LAMBDA,
    SCHED_JOINT_LK_PLAN_HINT_MAX,
    SCHED_WEIGHT_BIAS_ETA,
)
from types import SimpleNamespace
import logging
import sys
import math
import random
import copy
import heapq
import itertools
from stats_recorder import StatsRecorder
from comm_primitives import (
    normalize_topology,
    ring_allreduce,
    reduce_to_host,
    gather_to_host,
    scatter_from_host,
    transfer_p2p,
)
from weight_stage_models import clamp_overlap_ratio, overlap_time

_MISSING = object()
DEBUG_SCHEDULER = True
logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: DEBUG_SCHEDULER)


def _sched_normalize_weight_format_token(fmt: str, *, allow_compute: bool = False) -> str:
    if _cm_normalize_weight_format_token is not None:
        return _cm_normalize_weight_format_token(fmt, allow_compute=allow_compute)
    s = str(fmt or 'ND').strip().upper().replace('_', '-')
    alias = {
        'NPU-OPT': 'NZ',
        'PIM-OPT': 'PIM-OPT',
        'DUAL': 'DUAL',
        'DUAL-COPY': 'DUAL',
        'NZ+PIM-OPT': 'DUAL',
    }
    s = alias.get(s, s)
    storage_ok = {'ND', 'NZ', 'PIM-OPT', 'DUAL'}
    compute_ok = {'ZN', 'ZZ'} if allow_compute else set()
    ok = storage_ok | compute_ok
    if s not in ok:
        raise ValueError(f'Unsupported weight format token: {fmt}')
    return s


def _sched_resolve_npu_weight_conversion_steps(src_fmt: str, dst_fmt: str) -> List[Tuple[str, str]]:
    if _cm_resolve_npu_weight_conversion_steps is not None:
        return _cm_resolve_npu_weight_conversion_steps(src_fmt, dst_fmt)
    src = _sched_normalize_weight_format_token(src_fmt, allow_compute=True)
    dst = _sched_normalize_weight_format_token(dst_fmt, allow_compute=True)
    if src == 'DUAL':
        src = 'NZ'
    if src == dst:
        return []
    if src == 'ND':
        if dst == 'NZ':
            return [('ND', 'NZ')]
        if dst in ('ZN', 'ZZ'):
            return [('ND', 'NZ'), ('NZ', dst)]
    if src == 'NZ':
        if dst in ('ZN', 'ZZ'):
            return [('NZ', dst)]
        if dst == 'ND':
            return [('NZ', 'ND')]
    if src == 'PIM-OPT':
        if dst == 'ND':
            return [('PIM-OPT', 'ND')]
        if dst == 'NZ':
            return [('PIM-OPT', 'ND'), ('ND', 'NZ')]
        if dst in ('ZN', 'ZZ'):
            return [('PIM-OPT', 'ND'), ('ND', 'NZ'), ('NZ', dst)]
    raise ValueError(f'Unsupported NPU weight conversion path: {src}->{dst}')

