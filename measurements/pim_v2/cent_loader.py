from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
import importlib, importlib.util, inspect, sys, os, types
from typing import Dict, List, Callable, Any, Tuple, Union

Pathish = Union[str, os.PathLike]

def _as_paths_list(paths: Union[Pathish, Sequence[Pathish]]) -> List[Path]:
    """把单个路径或路径序列统一成 List[Path]；防止传进来的是 os.path/posixpath/module 等非序列对象。"""
    if isinstance(paths, (str, os.PathLike)):
        return [Path(paths)]
    if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes, bytearray)):
        return [Path(p) for p in paths]
    return [Path(paths)]

@contextmanager
def _use_paths_front(paths: Union[Pathish, Sequence[Pathish]]):
    """把一个或多个路径临时插入 sys.path 的最前面；退出时按插入顺序移除。"""
    to_push = [str(p.resolve()) for p in _as_paths_list(paths)]
    try:
        for s in to_push:
            sys.path.insert(0, s)
        yield
    finally:
        # 逐个移除首次出现的记录
        for s in to_push:
            try:
                sys.path.remove(s)
            except ValueError:
                pass

def _normalize_module_path(cent_root: Path, module_path: str) -> str:
    p = Path(module_path)
    if p.suffix == ".py":
        rel = p.resolve().relative_to(cent_root.resolve())
        return str(rel.with_suffix("")).replace(os.sep, ".")
    return module_path

def _package_info(cent_root: Path, module_path: str) -> Tuple[str, Path, Path]:
    parts = module_path.split(".")
    if len(parts) < 2:
        raise ValueError(f"module_path should be like 'pkg.submod', got: {module_path}")
    pkg_name = parts[0]
    pkg_dir  = cent_root / pkg_name
    return pkg_name, pkg_dir, cent_root

def _file_for_module(cent_root: Path, module_path: str) -> Tuple[Path, Path, str, str]:
    parts = module_path.split(".")
    if len(parts) < 2:
        raise ValueError(f"module_path should be like 'pkg.submod', got: {module_path}")
    pkg_name = parts[0]
    sub_path = "/".join(parts[1:])
    pkg_dir  = cent_root / pkg_name
    file     = pkg_dir / (sub_path + ".py")
    return pkg_dir, file, pkg_name, sub_path

def _import_module_with_fallback(cent_root: Path, module_path: str):
    """优先用标准包导入；失败则注册伪包并按文件路径导入子模块。"""
    pkg_name, pkg_dir, _ = _package_info(cent_root, module_path)
    # ☆ 同时把 cent_root 和 pkg_dir（cent_simulation/）放到 sys.path，修复 TransformerBlock 里的 'import aim_sim'
    with _use_paths_front([cent_root, pkg_dir]):
        try:
            return importlib.import_module(module_path)
        except ModuleNotFoundError:
            # 兜底：构造伪包并从文件加载
            pkg_dir2, file, pkg_name2, _ = _file_for_module(cent_root, module_path)
            if not file.exists():
                raise
            if pkg_name2 not in sys.modules:
                pkg_mod = types.ModuleType(pkg_name2)
                pkg_mod.__path__ = [str(pkg_dir2)]
                sys.modules[pkg_name2] = pkg_mod
            spec = importlib.util.spec_from_file_location(module_path, file)
            if spec is None or spec.loader is None:
                raise
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_path] = mod
            mod.__package__ = pkg_name2
            spec.loader.exec_module(mod)
            return mod
        
@dataclass
class CentModule:
    module: object
    funcs:Dict[str,Callable]

def load_cent_functions(
    cent_root: str | Path,
    module_path: str,
    function_names: List[str],
    *, #后续参数都是关键字参数，必须通过key=value的形式传递，不能通过位置
    strict: bool = True,
    show_signatures: bool = False,
)-> CentModule:
    
    cent_root = Path(cent_root).resolve()
    with _use_paths_front(cent_root):
        mod = importlib.import_module(module_path) #和import 类似，但是可以在运行时动态指定模块名

        funcs:Dict[str,Callable] = {}
        missing: List[str] = []

        for name in function_names:
            if hasattr(mod,name):
                fn = getattr(mod, name)
                funcs[name] = fn
                if show_signatures:
                    try:
                        print(f"[CENT] {module_path}.{name}{inspect.signature(fn)}")
                    except Exception:
                        print(f"[CENT]{module_path}.{name}{...}")
                    
            else:
                missing.append(name)

        if missing and strict:
            raise ImportError(
                f"[CENT] Missing in {module_path}: {', '.join(missing)}. "
                f"Check the module path or update the function names."
            )
            
    return CentModule(module=mod, funcs=funcs)
 
def load_cent_functions(
    cent_root: str | Path,
    module_path: str,
    function_names: List[str],
    *,
    strict: bool = True,
    show_signatures: bool = False,
) -> CentModule:
    cent_root = Path(cent_root).resolve()
    # 允许传 "xxx.py" 或 "pkg.Mod" 或 "Mod"
    module_path = _normalize_module_path(cent_root, module_path)

    # 用带兜底的导入，自动把 cent_root 和包目录都放进 sys.path
    mod = _import_module_with_fallback(cent_root, module_path)

    funcs: Dict[str, Callable] = {}
    missing: List[str] = []
    for name in function_names:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            if callable(fn):
                funcs[name] = fn
                if show_signatures:
                    try:
                        print(f"[CENT] {module_path}.{name}{inspect.signature(fn)}")
                    except Exception:
                        print(f"[CENT] {module_path}.{name}(...)")
            else:
                missing.append(name)
        else:
            missing.append(name)

    if missing and strict:
        raise ImportError(
            f"[CENT] Missing in {module_path}: {', '.join(missing)}."
        )

    return CentModule(module=mod, funcs=funcs)   

class CentAPI:
    def __init__(self,cent_root:str | Path, module_path: str):
        self.cent_root = Path(cent_root).resolve()
        with _use_path_front(self, cent_root):
             self._mod = importlib.import_module(module_path)

    def __getattr__ (self, item:str):
        attr = getattr(self._mod, item)
        if callable(attr):#如果是一个可调用对象
            def _wrapped(*args, **kwargs):
                with _use_path_front(self.cent_root):
                    return attr(*args, **kwargs)
            return _wrapped
        return attr