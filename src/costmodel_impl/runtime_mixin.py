"""Initialization, runtime-model, and weight-format helpers for CostModel."""

from __future__ import annotations

from .shared import *
from .npu_backends import *
from .pim_backends import *

class CostModelRuntimeMixin:
    def __init__(
        self,
        cluster: Cluster,
        dtype: str = 'fp16',
        pim_config_path: Optional[Path] = None,
        ramulator_config_path: Optional[Path] = None,
        simulation_log_file: Optional[Path] = None,
        debug_traces: bool = False,
        model_dict: Optional[Dict] = None,
        pim_fast_mode: bool = False,
        npu_backend: Optional[str] = None,
        npu_lut_strict: bool = False,
        tp_qkv: int = 1,
        tp_ffn: int = 1,
        tp_moe: int = 1,
        pim_ramulator_bin: Optional[Path] = None,
        pim_ramulator_timeout_s: Optional[int] = None,
        pim_trace_strict: bool = False,
        pim_trace_keep_traces: bool = False,
        pim_trace_dir: Optional[Path] = None,
    ):
        self.cluster = cluster
        self.dtype = normalize_dtype_token(dtype, default='fp16')
        self.pim_config_path = pim_config_path
        self.ramulator_config_path = ramulator_config_path
        # -------------------------------------------------------------------
        # PIM trace runtime knobs (keep CostModel + schedule_deploy_verify consistent)
        # -------------------------------------------------------------------
        self.pim_ramulator_bin: Optional[Path] = None
        if pim_ramulator_bin:
            try:
                self.pim_ramulator_bin = Path(pim_ramulator_bin).expanduser().resolve()
            except Exception:
                self.pim_ramulator_bin = Path(str(pim_ramulator_bin)).expanduser()
            # cost_model_pim_backend resolves ramulator2 via env vars.
            os.environ['RAMULATOR2_BIN'] = str(self.pim_ramulator_bin)

        try:
            self.pim_ramulator_timeout_s: int = int(pim_ramulator_timeout_s) if pim_ramulator_timeout_s is not None else 3000
        except Exception:
            self.pim_ramulator_timeout_s = 3000

        self.pim_trace_strict: bool = bool(pim_trace_strict)
        self.pim_trace_keep_traces: bool = bool(pim_trace_keep_traces or debug_traces)
        self.pim_trace_dir: Optional[Path] = None
        if pim_trace_dir is not None:
            try:
                self.pim_trace_dir = Path(pim_trace_dir).expanduser().resolve()
            except Exception:
                self.pim_trace_dir = Path(str(pim_trace_dir)).expanduser()

        # When user doesn't provide a shared model_dict, build per-(dim,heads,kv_heads,ffn,seqlen) dicts on demand.
        self._pim_model_dict_cache: Dict[Tuple[int, int, int, int, int], Dict[str, Any]] = {}
        self.debug_traces = debug_traces
        self.pim_fast_mode = pim_fast_mode  # When True, skip all trace simulations
        self.npu_backend = _normalize_npu_backend_safe(npu_backend)
        self.npu_lut_strict = bool(npu_lut_strict)
        try:
            self.tp_qkv = max(1, int(tp_qkv or 1))
            self.tp_ffn = max(1, int(tp_ffn or 1))
            self.tp_moe = max(1, int(tp_moe or 1))
        except Exception:
            self.tp_ffn = 1
            self.tp_qkv = 1
            self.tp_moe = 1
        self.logger = get_simulation_logger(simulation_log_file)
        self.pim_cache_enabled = True
        self._shared_model_dict: Optional[Dict] = model_dict
        self._format_runtime_model_path: Optional[Path] = None
        self._format_runtime_model_raw: Optional[Dict[str, Any]] = None
        self._npu_weight_runtime_model: Optional[NpuWeightRuntimeModel] = None
        self._pim_weight_runtime_model: Optional[PimWeightRuntimeModel] = None
        self.kv_pd_separation: bool = False
        if (not self.pim_fast_mode) and pim_config_path:
            if not pim_config_path.exists():
                raise ValueError(f'PIM config not found: {pim_config_path}')
            if model_dict is None:
                logger.debug(str('[WARNING] PIM config provided but model_dict is None. Call set_model_dict() before using PIM operations.'))
        if (not self.pim_fast_mode) and ramulator_config_path:
            if not ramulator_config_path.exists():
                raise ValueError(f'Ramulator config not found: {ramulator_config_path}')

        self._npu_backend_impl_name: Optional[str] = None
        self._npu_backend_impl: NpuBackendBase = NpuFastBackend()
        self._pim_backend_fast_mode: Optional[bool] = None
        self._pim_backend_impl: PimBackendBase = PimFastBackend()
        self._ensure_backend_impls()

    def _ensure_backend_impls(self) -> None:
        """(Re)build backend objects if user changes npu_backend / pim_fast_mode after __init__."""
        npu_name = _normalize_npu_backend_safe(self.npu_backend)
        if npu_name is None:
            npu_name = 'fast'
        if npu_name != getattr(self, '_npu_backend_impl_name', None):
            self._npu_backend_impl = build_npu_backend(npu_name)
            self._npu_backend_impl_name = npu_name

        pim_fast = bool(getattr(self, 'pim_fast_mode', False))
        if pim_fast != getattr(self, '_pim_backend_fast_mode', None):
            self._pim_backend_impl = build_pim_backend(pim_fast)
            self._pim_backend_fast_mode = pim_fast

    def set_model_dict(self, model_dict: Dict):
        if model_dict is None:
            raise ValueError('model_dict cannot be None')
        self._shared_model_dict = model_dict
        logger.debug(str(f'[CostModel] Model dictionary set with keys: {list(model_dict.keys())[:5]}...'))

    def get_model_dict(self) -> Dict:
        if self._shared_model_dict is None:
            raise RuntimeError('Model dictionary not set. You must call set_model_dict() or provide model_dict during initialization before using PIM operations.')
        return self._shared_model_dict

    def has_model_dict(self) -> bool:
        return self._shared_model_dict is not None


    def get_or_make_pim_model_dict(
        self,
        *,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        seqlen: int,
    ) -> Dict[str, Any]:
        """Return a model_dict for the CENT/AiM PIM trace backend.

        If the user provided a shared model_dict (set_model_dict), that is returned.
        Otherwise we lazily synthesize one and cache it by (dim, n_heads, n_kv_heads, ffn_dim, seqlen).
        """
        if self._shared_model_dict is not None:
            return self._shared_model_dict

        try:
            key = (int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim), int(seqlen))
        except Exception:
            key = (0, 0, 0, 0, 0)

        md = self._pim_model_dict_cache.get(key)
        if md is not None:
            return md

        md = _make_shared_model_dict(int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim), int(seqlen))
        self._pim_model_dict_cache[key] = md
        return md

    def get_host_device(self) -> DeviceSpec:    
        if HOST_NAME in self.cluster.devices:
            return self.cluster.devices[HOST_NAME]
        cpus = self.cluster.devices_by_type('cpu')
        return cpus[0] if cpus else next(iter(self.cluster.devices.values()))

    def _first_device_of_type(self, dev_type: str) -> Optional[DeviceSpec]:
        try:
            devs = list(getattr(self.cluster, 'devices_by_type', lambda *_: [])(str(dev_type)))
        except Exception:
            devs = []
        return devs[0] if devs else None

    def _npu_fast_hw_only_mode(self) -> bool:
        self._ensure_backend_impls()
        return str(getattr(self, '_npu_backend_impl_name', '') or '').strip().lower() == 'fast'

    def _pim_fast_hw_only_mode(self) -> bool:
        return bool(getattr(self, 'pim_fast_mode', False))

    def device_preferred_fmt(self, dev: DeviceSpec) -> str:
        return DEVICE_PREFERRED_FORMAT.get(dev.type, 'ND')

    def format_size(self, size_bytes: int, fmt: str) -> int:
        m = float(FORMAT_SIZE_MULTIPLIER.get(fmt, 1.0))
        return int(size_bytes * m)

    # ------------------------------------------------------------------
    # Explicit weight-format model
    # ------------------------------------------------------------------
    def _discover_npu_weight_runtime_model_path(self) -> Path:
        env_path = str(os.environ.get('NPU_WEIGHT_RUNTIME_JSON', '') or '').strip()
        if env_path:
            rp = Path(env_path).expanduser().resolve()
            if not rp.exists():
                raise ValueError(f'NPU runtime-model JSON not found: {rp}')
            return rp

        runtime_dir_candidates = list(getattr(_config, 'NPU_RUNTIME_MODEL_DIR_CANDIDATES', ()) or ())
        runtime_dir = str(getattr(_config, 'NPU_RUNTIME_MODEL_DIR', './run_time_model') or './run_time_model').strip()
        if runtime_dir:
            runtime_dir_candidates.append(runtime_dir)
        if './runtime_models' not in runtime_dir_candidates:
            runtime_dir_candidates.append('./runtime_models')

        search_roots = [
            Path.cwd(),
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parent.parent,
        ]
        uniq_hits: List[Path] = []
        seen_hits = set()
        for runtime_dir in runtime_dir_candidates:
            for root in search_roots:
                d = (root / runtime_dir).resolve() if not Path(runtime_dir).is_absolute() else Path(runtime_dir).resolve()
                if not d.is_dir():
                    continue
                for q in sorted(d.glob('*.json')):
                    pth = q.resolve()
                    if pth not in seen_hits:
                        uniq_hits.append(pth)
                        seen_hits.add(pth)
        if not uniq_hits:
            raise ValueError(
                'No runtime-model JSON found. Expected exactly one *.json under one of '
                f"{runtime_dir_candidates}"
            )
        if len(uniq_hits) > 1:
            raise ValueError(
                'Multiple runtime-model JSON files found; please keep exactly one file in '
                f"{runtime_dir_candidates} or set NPU_WEIGHT_RUNTIME_JSON. found={[str(p) for p in uniq_hits]}"
            )
        return uniq_hits[0]

    def _ensure_runtime_model_raw(self) -> Tuple[Path, Dict[str, Any]]:
        cached = getattr(self, '_format_runtime_model_raw', None)
        cached_path = getattr(self, '_format_runtime_model_path', None)
        if cached is not None and cached_path is not None:
            return cached_path, cached

        path = self._discover_npu_weight_runtime_model_path()
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
        except Exception as e:
            raise ValueError(f'Failed to load runtime-model JSON: {path} ({e})') from e
        if not isinstance(raw, dict):
            raise ValueError(f'Runtime-model JSON must be a dict/object: {path}')
        self._format_runtime_model_raw = raw
        self._format_runtime_model_path = path
        return path, raw

    def _runtime_model_device_sections(self, dev_key: str) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
        path, raw = self._ensure_runtime_model_raw()
        bw_root = ((raw or {}).get('format_conv_bw_gbs') or {}).get(str(dev_key), {}) or {}
        ovh_root = ((raw or {}).get('format_conv_overhead_us') or {}).get(str(dev_key), {}) or {}
        if not isinstance(bw_root, dict) or not isinstance(ovh_root, dict):
            raise ValueError(
                f"Runtime-model JSON sections format_conv_bw_gbs/{dev_key} and format_conv_overhead_us/{dev_key} must be dicts: {path}"
            )
        return path, bw_root, ovh_root

    def _ensure_npu_weight_runtime_model(self) -> NpuWeightRuntimeModel:
        mdl = getattr(self, '_npu_weight_runtime_model', None)
        if mdl is not None:
            return mdl

        path, bw_root, ovh_root = self._runtime_model_device_sections('npu')
        bw_paths = bw_root.get('paths') or {}
        ovh_paths = ovh_root.get('paths') or {}
        if not isinstance(bw_paths, dict) or not isinstance(ovh_paths, dict) or not bw_paths:
            raise ValueError(
                f'NPU runtime-model JSON is missing format_conv_bw_gbs/format_conv_overhead_us npu paths: {path}'
            )
        bw = {str(k): float(v) for k, v in bw_paths.items()}
        ovh = {str(k): float(v) for k, v in ovh_paths.items()}
        mdl = NpuWeightRuntimeModel(path=path, bw_gbs=bw, overhead_us=ovh)
        self._npu_weight_runtime_model = mdl
        return mdl

    def _ensure_pim_weight_runtime_model(self) -> PimWeightRuntimeModel:
        mdl = getattr(self, '_pim_weight_runtime_model', None)
        if mdl is not None:
            return mdl

        path, bw_root, ovh_root = self._runtime_model_device_sections('pim')
        bw_paths = bw_root.get('paths') or {}
        ovh_paths = ovh_root.get('paths') or {}
        if not isinstance(bw_paths, dict) or not isinstance(ovh_paths, dict) or not bw_paths:
            raise ValueError(
                f'PIM runtime-model JSON is missing format_conv_bw_gbs/format_conv_overhead_us pim paths: {path}'
            )

        bw: Dict[str, float] = {}
        ovh: Dict[str, float] = {}
        for key in _PIM_WEIGHT_LOAD_DEFAULT_PATHS.keys():
            if key not in bw_paths:
                raise ValueError(
                    f"Missing PIM runtime-model bandwidth entry for path '{key}' in {path}"
                )
            bw_gbs = float(bw_paths[key] or 0.0)
            overhead_us = float(ovh_paths.get(key, 0.0) or 0.0)
            if bw_gbs <= 0.0:
                raise ValueError(
                    f"PIM runtime-model path '{key}' must have bw_gbs > 0 in {path}, got {bw_gbs}"
                )
            bw[key] = float(bw_gbs)
            ovh[key] = float(overhead_us)

        # Semantic guardrail for the PIM local-load model:
        # ND->PIM-OPT is interpreted as "pack/format-convert + write/program PIM",
        # while PIM-OPT->PIM-OPT is interpreted as "write/program PIM only".
        # Therefore the total ND->PIM-OPT path should not be *faster* than the
        # pure-write path for the same logical ND payload. The JSON stores effective
        # bandwidth/overhead parameters, not raw times.
        try:
            bw_total = float(bw.get('ND->PIM-OPT', 0.0) or 0.0)
            bw_write = float(bw.get('PIM-OPT->PIM-OPT', 0.0) or 0.0)
            ovh_total = float(ovh.get('ND->PIM-OPT', 0.0) or 0.0)
            ovh_write = float(ovh.get('PIM-OPT->PIM-OPT', 0.0) or 0.0)
            if bw_total > bw_write or ovh_total < ovh_write:
                logger.warning(
                    '[PIM-RUNTIME-MODEL] Suspicious local-load semantics in %s: '
                    'ND->PIM-OPT is modeled as total(pack+write) but appears faster than '
                    'PIM-OPT->PIM-OPT(write-only). Current values: bw_total=%s bw_write=%s '
                    'ovh_total=%s ovh_write=%s. If you intended to change *time*, update '
                    'format_conv_bw_gbs / format_conv_overhead_us with calibrated effective values, '
                    'not raw latency numbers.',
                    str(path),
                    bw_total,
                    bw_write,
                    ovh_total,
                    ovh_write,
                )
        except Exception:
            pass

        mdl = PimWeightRuntimeModel(
            source=str(path),
            bw_gbs=bw,
            overhead_us=ovh,
        )
        self._pim_weight_runtime_model = mdl
        return mdl

    def weight_storage_format(self, fmt: str) -> str:
        return _normalize_weight_format_token(fmt, allow_compute=False)

    def weight_storage_bytes(self, size_nd_bytes: int, fmt: str) -> int:
        return int(self.format_size(int(size_nd_bytes), self.weight_storage_format(fmt)))

    def weight_host_source_format(self, fmt: str, dev_or_type: DeviceSpec | str) -> str:
        src = self.weight_storage_format(str(fmt))
        dev_type = str(getattr(dev_or_type, 'type', dev_or_type) or '').strip().lower()
        if src != 'DUAL':
            return str(src)
        if dev_type == 'pim':
            return 'PIM-OPT'
        if dev_type == 'npu':
            return 'NZ'
        return 'NZ'

    def weight_resident_format(self, host_src_fmt: str, dev: DeviceSpec) -> str:
        dev_type = str(getattr(dev, 'type', '') or '').lower()
        src = _normalize_weight_format_token(str(host_src_fmt), allow_compute=False)
        if dev_type == 'pim':
            return 'PIM-OPT'
        if src == 'DUAL':
            return self.weight_host_source_format(src, dev)
        return str(src)

    def npu_weight_compute_format(self, node: TaskNode) -> str:
        attrs = getattr(node, 'attrs', {}) or {}
        for key in ('npu_weight_target_format', 'npu_cube_weight_format', 'weight_compute_format', 'cube_weight_format'):
            val = attrs.get(key)
            if val not in (None, ''):
                return _normalize_weight_format_token(str(val), allow_compute=True)
        op = _normalize_npu_weight_op_name(node)
        mapping = getattr(_config, 'NPU_WEIGHT_TARGET_FORMAT_BY_OP', {}) or {}
        if op not in mapping:
            raise ValueError(
                f"Missing explicit NPU cube target format for weight op='{op}'. "
                'Please set config.NPU_WEIGHT_TARGET_FORMAT_BY_OP or annotate node.attrs.'
            )
        return _normalize_weight_format_token(str(mapping[op]), allow_compute=True)

    def _weight_runtime_path_total_s(
        self,
        *,
        size_nd_bytes: int,
        steps: List[Tuple[str, str]],
        bw_gbs: Dict[str, float],
        overhead_us: Dict[str, float],
        source_desc: str,
    ) -> float:
        size_nd_bytes = int(size_nd_bytes or 0)
        if size_nd_bytes <= 0 or not steps:
            return 0.0
        total_s = 0.0
        for a, b in steps:
            key = f'{a}->{b}'
            bw = bw_gbs.get(key)
            if bw is None:
                raise ValueError(f"Missing runtime-model bandwidth entry for path '{key}' in {source_desc}")
            eff_bw = float(bw) * 1e9
            if eff_bw <= 0.0:
                raise ValueError(f"Non-positive runtime-model bandwidth for path {key}: {bw}")
            total_s += float(overhead_us.get(key, 0.0) or 0.0) * 1e-6
            total_s += float(size_nd_bytes) / float(eff_bw)
        return float(total_s)

    def npu_weight_conversion_time(self, size_nd_bytes: int, src_fmt: str, dst_fmt: str, *, dev: Optional[DeviceSpec] = None) -> float:
        size_nd_bytes = int(size_nd_bytes or 0)
        if size_nd_bytes <= 0:
            return 0.0

        # Fast-mode simplification: do not use any runtime-model ND/NZ/NPU-opt
        # conversion data. Model NPU internal weight fetch purely from the
        # hardware JSON memory bandwidth.
        if self._npu_fast_hw_only_mode():
            dev_eff = dev if dev is not None else self._first_device_of_type('npu')
            if dev_eff is None:
                return 0.0
            src = _normalize_weight_format_token(src_fmt, allow_compute=True)
            weight_bytes = int(self.weight_storage_bytes(int(size_nd_bytes), str(src)))
            return float(self.mem_time(int(weight_bytes), dev_eff))

        src = _normalize_weight_format_token(src_fmt, allow_compute=True)
        dst = _normalize_weight_format_token(dst_fmt, allow_compute=True)
        if src == dst:
            return 0.0
        mdl = self._ensure_npu_weight_runtime_model()
        steps = _resolve_npu_weight_conversion_steps(src, dst)
        return float(self._weight_runtime_path_total_s(
            size_nd_bytes=int(size_nd_bytes),
            steps=list(steps),
            bw_gbs=dict(mdl.bw_gbs),
            overhead_us=dict(mdl.overhead_us),
            source_desc=str(mdl.path),
        ))

    def npu_local_weight_load_cost(self, size_nd_bytes: int, src_storage_fmt: str, dst_compute_fmt: str, *, from_cache: bool = False, dev: Optional[DeviceSpec] = None) -> WeightLoadStageBreakdown:
        total = float(self.npu_weight_conversion_time(int(size_nd_bytes), str(src_storage_fmt), str(dst_compute_fmt), dev=dev))
        return WeightLoadStageBreakdown(
            total_s=float(total),
            host_src_fmt=str(src_storage_fmt),
            resident_fmt=str(dst_compute_fmt),
            l2_local_s=float(total),
            combine_rule='serial',
            bytes_nd=int(size_nd_bytes or 0),
            bytes_src=int(self.weight_storage_bytes(int(size_nd_bytes or 0), str(src_storage_fmt))),
        )

    def pim_local_weight_load_time(self, size_nd_bytes: int, src_storage_fmt: str, *, dev: Optional[DeviceSpec] = None) -> float:
        size_nd_bytes = int(size_nd_bytes or 0)
        if size_nd_bytes <= 0:
            return 0.0

        # Fast-mode simplification: do not use ND->PIM-OPT runtime-model data.
        # Treat local load as direct programming/write into PIM memory using the
        # PIM line-latency model derived from hardware JSON.
        if self._pim_fast_hw_only_mode():
            dev_eff = dev if dev is not None else self._first_device_of_type('pim')
            if dev_eff is None:
                return 0.0
            src = self.weight_storage_format(str(src_storage_fmt))
            bytes_local = int(self.weight_storage_bytes(int(size_nd_bytes), str(src)))
            return float(self.pim_write_time(int(bytes_local), dev_eff))

        mdl = self._ensure_pim_weight_runtime_model()
        steps = _resolve_pim_weight_load_steps(str(src_storage_fmt))
        return float(self._weight_runtime_path_total_s(
            size_nd_bytes=int(size_nd_bytes),
            steps=list(steps),
            bw_gbs=dict(mdl.bw_gbs),
            overhead_us=dict(mdl.overhead_us),
            source_desc=str(mdl.source),
        ))

    def pim_local_weight_write_only_time(self, size_nd_bytes: int, *, dev: Optional[DeviceSpec] = None) -> float:
        size_nd_bytes = int(size_nd_bytes or 0)
        if size_nd_bytes <= 0:
            return 0.0

        if self._pim_fast_hw_only_mode():
            dev_eff = dev if dev is not None else self._first_device_of_type('pim')
            if dev_eff is None:
                return 0.0
            return float(self.pim_write_time(int(size_nd_bytes), dev_eff))

        mdl = self._ensure_pim_weight_runtime_model()
        return float(self._weight_runtime_path_total_s(
            size_nd_bytes=int(size_nd_bytes),
            steps=[('PIM-OPT', 'PIM-OPT')],
            bw_gbs=dict(mdl.bw_gbs),
            overhead_us=dict(mdl.overhead_us),
            source_desc=str(mdl.source),
        ))

    def pim_local_weight_pack_only_est_time(self, size_nd_bytes: int, src_storage_fmt: str, *, dev: Optional[DeviceSpec] = None) -> float:
        if self._pim_fast_hw_only_mode():
            return 0.0
        total = float(self.pim_local_weight_load_time(int(size_nd_bytes), str(src_storage_fmt), dev=dev))
        write_only = float(self.pim_local_weight_write_only_time(int(size_nd_bytes), dev=dev))
        return float(max(0.0, total - write_only))

    def pim_local_weight_load_cost(self, size_nd_bytes: int, src_storage_fmt: str, *, from_cache: bool = False, dev: Optional[DeviceSpec] = None) -> WeightLoadStageBreakdown:
        total = float(self.pim_local_weight_load_time(int(size_nd_bytes), str(src_storage_fmt), dev=dev))
        return WeightLoadStageBreakdown(
            total_s=float(total),
            host_src_fmt=str(src_storage_fmt),
            resident_fmt='PIM-OPT',
            l2_local_s=float(total),
            combine_rule='serial',
            bytes_nd=int(size_nd_bytes or 0),
            bytes_src=int(self.weight_storage_bytes(int(size_nd_bytes or 0), str(src_storage_fmt))),
        )

    def weight_transfer_comm_bytes(self, size_nd_bytes: int, src_storage_fmt: str, *, dev_or_type: DeviceSpec | str | None = None) -> int:
        fmt = str(src_storage_fmt)
        if dev_or_type is not None:
            fmt = self.weight_host_source_format(fmt, dev_or_type)
        else:
            fmt_norm = self.weight_storage_format(fmt)
            if fmt_norm == 'DUAL':
                raise ValueError('weight_transfer_comm_bytes requires dev_or_type when src_storage_fmt is DUAL')
            fmt = fmt_norm
        return int(self.weight_storage_bytes(int(size_nd_bytes), str(fmt)))
