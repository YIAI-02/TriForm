"""Shared imports, policy naming helpers, and logger setup for main CLI modules."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from buffer_manager import GlobalMemoryManager
from config import (
    ENABLE_PIM_WEIGHT_PRELOAD,
    PIM_STATIC_ALLOC_RATIO,
    attach_local_debug_filter,
    setup_logging,
)
from cost_model import CostModel, DTYPE_BYTES
from costmodel_impl.cost_model_pim_backend import _make_shared_model_dict
from dtype_utils import dtype_bytes, normalize_dtype_token
from hardware import Cluster, demo_cluster
from model_parser import build_graph
from plan_label import PlanLabel
from scheduler import BifocalScheduler, HEFTScheduler, NaiveTopoScheduler
from stats_recorder import reset_simulation_logger
from task_graph import TaskGraph, TaskNode

logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: True)

_WEIGHT_SUGGEST_DEBUG_SUMMARY_ONLY = False
_WEIGHT_SUGGEST_PROGRESS_ENABLED = False
_WEIGHT_SUGGEST_AL_LOGGER: logging.Logger | None = None
_WEIGHT_SUGGEST_AL_LOG_PATH: str | None = None

_POLICY_DISPLAY_NAMES: Dict[str, str] = {
    'HEFT': 'HEFT',
    'Bifocal': 'Bifocal',
    'Naive': 'Naive',
    'PD': 'PD',
    'AF': 'AF',
    'PD+Linear': 'PD+Linear',
    'PD+Attn': 'PD+Attn',
    'PD+FFN': 'PD+FFN',
    'weights_on_pim': 'weights_on_pim',
    'NeuPIMs': 'NeuPIMs',
    'ColdMoE': 'ColdMoE',
}

_ALGO_TOKENS = frozenset({'HEFT', 'Bifocal', 'Naive'})
_BASELINE_TOKENS = frozenset({'PD', 'AF', 'PD+Linear', 'PD+Attn', 'PD+FFN', 'weights_on_pim', 'NeuPIMs', 'ColdMoE'})


def _normalize_policy_lookup_key(name: Any) -> str:
    return str(name or '').strip().replace('＋', '+')


def _canonical_policy_token(name: Any) -> str:
    return _normalize_policy_lookup_key(name)


def _normalize_algo_name(name: Any) -> str:
    token = _canonical_policy_token(name)
    return token if token in _ALGO_TOKENS else token


def _normalize_baseline_name(name: Any) -> str:
    token = _canonical_policy_token(name)
    return token if token in _BASELINE_TOKENS else token


def _display_policy_name(name: Any) -> str:
    token = _canonical_policy_token(name)
    if token in _POLICY_DISPLAY_NAMES:
        return _POLICY_DISPLAY_NAMES[token]
    raw = str(name or '').strip()
    return raw or token


def _policy_label(name: Any) -> str:
    return f'algo:{_display_policy_name(name)}'


def _policy_dir_name(name: Any) -> str:
    return f'algo_{_display_policy_name(name)}'


__all__ = [name for name in globals() if not name.startswith('__')]
