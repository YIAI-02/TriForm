
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path

DEFAULT_SUMMARY_JSON = Path("measurements/pim/out_run/results/model_fit_summary.json")
ALT_FORMULA_JSON = Path("measurements/pim/out_run/results/model_formula.json")

@dataclass
class OpFormula:
    basis: List[str]
    coeffs: List[float]
    def eval_cycles(self, *, seqlen: int = 0, vector_dim: int = 0, matrix_col: int = 0, n_heads: int = 0) -> float:
        def val(name: str) -> float:
            if name == "1":  return 1.0
            if name == "L":  return float(seqlen)
            if name == "L2": return float(seqlen)**2
            if name == "H":  return float(n_heads)
            if name == "LxH": return float(seqlen)*float(n_heads)
            if name == "V":  return float(vector_dim)
            if name == "N":  return float(matrix_col)
            if name == "VxN": return float(vector_dim)*float(matrix_col)
            return 0.0
        return sum(float(c)*val(b) for b,c in zip(self.basis, self.coeffs))

@dataclass
class PIMFormulaLatency:
    per_op: Dict[str, OpFormula]

    @staticmethod
    def _load_summary_json(path: Path) -> "PIMFormulaLatency":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if "per_op_formula" in obj:
            per = obj["per_op_formula"]
        elif "per_op" in obj:
            per = obj["per_op"]
        else:
            raise ValueError(f"Unrecognized JSON structure in {path}")
        per_op: Dict[str, OpFormula] = {}
        for k, v in per.items():
            if "basis" in v and "coeffs" in v:
                per_op[k] = OpFormula(basis=v["basis"], coeffs=[float(x) for x in v["coeffs"]])
        return PIMFormulaLatency(per_op=per_op)

    @classmethod
    def load(cls, path: Path) -> "PIMFormulaLatency":
        return cls._load_summary_json(path)

    @classmethod
    def load_default(cls) -> "PIMFormulaLatency":
        if DEFAULT_SUMMARY_JSON.is_file():
            return cls.load(DEFAULT_SUMMARY_JSON)
        if ALT_FORMULA_JSON.is_file():
            return cls.load(ALT_FORMULA_JSON)
        for p in [Path("out_run/results/model_fit_summary.json"),
                  Path("out_run/results/model_formula.json")]:
            if p.is_file():
                return cls.load(p)
        raise FileNotFoundError("summary json not found")
