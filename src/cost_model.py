"""Public CostModel facade assembled from smaller implementation modules."""

from __future__ import annotations

from costmodel_impl.compute_mixin import CostModelComputeMixin
from costmodel_impl.estimate_mixin import CostModelEstimateMixin
from costmodel_impl.npu_backends import (
    NpuAscend310BLutBackend,
    NpuBackendBase,
    NpuFastBackend,
    NpuLlmCompassBackend,
    NpuOpContext,
    build_npu_backend,
)
from costmodel_impl.pim_backends import (
    PimBackendBase,
    PimFastBackend,
    PimOpContext,
    PimTraceBackend,
    build_pim_backend,
)
from costmodel_impl.runtime_mixin import CostModelRuntimeMixin
from costmodel_impl.shared import (
    DTYPE_BYTES,
    LocalWeightLoadCost,
    NpuFastModeConfigError,
    NpuWeightRuntimeModel,
    PimFastModeConfigError,
    PimWeightRuntimeModel,
    _device_family_key_from_name,
    _is_norm_like,
    _lookup_cfg_by_device_name,
    _normalize_npu_backend_safe,
    _normalize_npu_op_key,
    _normalize_npu_weight_op_name,
    _normalize_weight_format_token,
    _resolve_npu_weight_conversion_steps,
    _resolve_pim_weight_load_steps,
)


class CostModel(
    CostModelRuntimeMixin,
    CostModelComputeMixin,
    CostModelEstimateMixin,
):
    """Cost model combining runtime lookup, kernel timing, and graph estimates."""


__all__ = [
    "CostModel",
    "DTYPE_BYTES",
    "LocalWeightLoadCost",
    "NpuAscend310BLutBackend",
    "NpuBackendBase",
    "NpuFastBackend",
    "NpuFastModeConfigError",
    "NpuLlmCompassBackend",
    "NpuOpContext",
    "NpuWeightRuntimeModel",
    "PimBackendBase",
    "PimFastBackend",
    "PimFastModeConfigError",
    "PimOpContext",
    "PimTraceBackend",
    "PimWeightRuntimeModel",
    "_normalize_weight_format_token",
    "_resolve_npu_weight_conversion_steps",
    "build_npu_backend",
    "build_pim_backend",
]
