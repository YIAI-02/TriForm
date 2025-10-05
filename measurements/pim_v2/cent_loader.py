from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Callable
import importlib, inspect, sys

@contextmanager
def _use_path_front(path:Path):
    '''
    temperarily prepend a path to sys.path to make CENT imports resolve correctly
    '''
    path = Path(path).resolve() #转换为绝对路径

    try:
        yield #将控制权交给with块
    finally: #不管with中有没有异常，finally 都会执行
        for i,p in enumerate(sys.path):
            if p == str(path):
                sys.path.pop(i) #最后将临时加入的path 删除
                break

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
    with _use_path_front(cent_root):
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