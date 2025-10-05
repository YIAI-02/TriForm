from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import os, csv, re, subprocess, inspect

from cent_loader import load_cent_functions
from trace_composer import concat_traces, write_trace

DEFAULT_FUNCTIONS = [
    "store_to_DRAM_multi_channel",
    "load_from_DRAM_multi_channel",
    "broadcast_store_query",
    "broadcast_load_query",
    "Vector_Matrix_Mul_weight_pim_only_trace",
    "Vector_Matrix_Mul_score_pim_only_trace",
    "Vector_Matrix_Mul_output_pim_only_trace",
    "Vector_Matrix_Mul_weight_pim",
    "Vector_Matrix_Mul_weight_af_pim",
    "Vector_Matrix_Mul_score_pim",
    "Vector_Matrix_Mul_output_pim",
    "store_for_neighbor_bank_input_only_trace",
    "store_for_input_only_trace",
    "store_for_EWMUL_input_only_trace",
    "load_from_input_only_trace",
    "load_from_EWMUL_input_only_trace",
    "store_for_EWMUL_score_only_trace",
    "load_from_EWMUL_score_only_trace",
    "store_for_score_only_trace",
    "load_for_score_only_trace",
]

AIM_BIN_ENV = "AIM_BIN"  # path to aim_simulator/build/ramulator2
AIM_CFG_ENV = "AIM_CFG"  # path to aim_simulator/test/example.yaml

def _bind_args(fn, kwargs:Dict[str,Any])-> Dict[str,Any]:
    sig = inspect.signature(fn) #获取函数的signature(包含函数的名称，参数信息，类型，默认值等)
    params = sig.parameters #获取的一个函数parameters的有序字典
    call_kwargs: Dict[str,Any] = {}
    missing: List[str] = []

    for name, p in params.items():
        '''
        name是参数名，p是参数的详细信息
        p.kind是参数的类型，可能是仅限位置参数（POSITIONAL_ONLY），位置/关键字参数 POSITIONAL_OR_KEYWORD，仅限关键字参数KEYWORD_ONLY，VAR_POSITIONAL可变位置参数 如*args，VAR_KEYWORD可变关键字参数 **kwargs
        '''
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name in kwargs: #如果参数在传入的kwargs中，就加入call_kwargs
            call_kwargs[name] = kwargs[name]
        elif p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            missing.append(name) #如果参数没有默认值，且参数属于后三者中的一个，加入missing
        if missing:
            raise TypeError(f"Missing required args for {fn.__name__}{sig}: {', '.join(missing)}")
    return call_kwargs #返回的是所有绑定参数

@dataclass
class CentOps:
    module_path: str
    funcs: Dict[str,Any]

class CentTraceAdapter:
    def __init__(self, cent_root: str, module_path: str, function_names: List[str] = None):
        function_names = function_names or DEFAULT_FUNCTIONS
        cm = load_cent_functions(cent_root, module_path, function_names, show_signatures=True, strict=False)
        self.ops = CentOps(module_path=module_path, funcs=cm.funcs)

    def build_trace_txt(self,steps: List[Dict[str, Any]], defaults: Dict[str, Any]) -> str:
        pieces: List[str] = []
        for i,st in enumerate(steps):
            fn_name = st['fn']
            fn_args = st.get("args",{})
            if fn_name not in self.ops.funcs:
                raise KeyError(f"[CENT] Function '{fn_name}' not loaded from {self.ops.module_path}.")
            fn = self.ops.funcs[fn_name]

            merged = {**defaults, **fn_args} 
            call_kwargs = _bind_args(fn, merged)
            ret = fn(**call_kwargs)

            if isinstance(ret, list):
                pieces.append("".join([line if line.endswith("\n") else line + "\n" for line in ret]))
            elif isinstance(ret, str):
                pieces.append(ret if ret.endswith("\n") else ret + "\n")
            elif ret is None:
                continue
            else:
                pieces.append(str(ret) + ("\n" if not str(ret).endswith("\n") else ""))
        return "".join(pieces)

    def write_trace(self, text: str, out_path: str) -> str:
        return write_trace(text, out_path)
    
    @staticmethod
    def run_aim(trace_path:str, aim_bin:Optional[str]= None, aim_cfg: Optional[str] = None) -> Tuple[float,float,float]:
        aim_bin = aim_bin or os.environ.get(AIM_BIN_ENV)
        aim_cfg = aim_cfg or os.environ.get(AIM_CFG_ENV)
        if not aim_bin or not os.path.exists(aim_bin):
            raise FileNotFoundError(f"AiM binary not found. Set {AIM_BIN_ENV} or pass aim_bin.")
        if not aim_cfg or not os.path.exists(aim_cfg):
            raise FileNotFoundError(f"AiM config YAML not found. Set {AIM_CFG_ENV} or pass aim_cfg.")
        
        proc = subprocess.run([aim_bin, "-f", aim_cfg, "-t", trace_path],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        
        out = proc.stdout

        m_cycles = re.search(r"memory_system_cycles\s*[:=]\s*([0-9eE+\-.]+)", out)
        if not m_cycles:
            raise RuntimeError("Failed to parse AiM output. Please update regex.")
        cycles = float(m_cycles.group(1))
        return cycles

    def batch_profile(
        self,
        grid: List[Tuple[int,int,int]],
        steps_builder: callable,   # (M,N,K, dtype, channels) -> steps(list[dict])
        dtype: str,
        out_csv: str,
        channels: int = 32,
        aim_bin: Optional[str] = None,
        aim_cfg: Optional[str] = None,
        tmp_dir: str = "trace/tmp"
    ) -> None:
        
        rows: List[Dict[str, Any]] = []
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        for (M,N,K) in grid:
            defaults = dict(M=M, N=N, K=K, dtype=dtype, channels=channels)
            steps = steps_builder(M, N, K, dtype, channels)
            text = self.build_trace_text(steps, defaults)
            out_path = f"{tmp_dir}/gemm_M{M}_N{N}_K{K}.trace"
            self.write_trace(text, out_path)
            cycles = self.run_aim(out_path, aim_bin=aim_bin, aim_cfg=aim_cfg)
            rows.append({"op_class":"GEMM","M":M,"N":N,"K":K,"dtype":dtype,"channels":channels,"cycles":cycles})

        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())) #DictWriter专门用于将dict写入csv
            w.writeheader() #把rows[0] key 定义的列名写进去
            w.writerows(rows) #这里只写行不会写列名

def default_weight_vmm_steps(M:int, N:int, K:int, dtype:str, channels:int) -> List[Dict[str, Any]]:
    "example step"
    return [
        {"fn":"store_for_input_only_trace", "args":{}},          
        {"fn":"store_to_DRAM_multi_channel", "args":{}},         
        {"fn":"broadcast_store_query", "args":{}},               
        {"fn":"Vector_Matrix_Mul_weight_pim_only_trace", "args":{}},
        {"fn":"broadcast_load_query", "args":{}}, 
        {"fn":"load_from_DRAM_multi_channel", "args":{}},
        {"fn":"load_from_input_only_trace", "args":{}},  
    ]

if __name__ == "__main__":
    cent_root   = "../../submodules/CENT"
    module_path = "../../submodules/CENT/cent_simulation/TransformerBlock.py" 
    adapter = CentTraceAdapter(cent_root, module_path, DEFAULT_FUNCTIONS)

    grid = [(2048,4096,4096), (4096,4096,4096)]
    out_csv = "profiles/aim_pim_latencies.csv"
    dtype = "fp16"
    adapter.batch_profile(
        grid=grid,
        steps_builder=default_weight_vmm_steps,
        dtype=dtype,
        out_csv=out_csv,
        channels=32,
        aim_bin="../../submodules/CENT/aim_simulator/build/ramulator2",
        aim_cfg="./../submodules/CENT/aim_simulator/test/example.yaml",
    )
    print(f"[OK] wrote {out_csv}")