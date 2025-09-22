from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ModelShape:
    layer_num: int
    dim: int
    ffn_dim: int
    n_heads: int
    n_kv_heads: int

    @property
    def head_dim(self) -> int:
        return self.dim // max(1, self.n_heads)

class BaseModelDef:
    name: str = "base"
    def layer_blueprint(self):
        raise NotImplementedError

class LLaMADef(BaseModelDef):
    name = "llama"
    def layer_blueprint(self):
        ops = [
            "X","LN1","Q","K","V","QK","Softmax","SV","O","Add1","LN2","FFN_W1","FFN_W3","SwiGLU","FFN_W2","Add2",
        ]
        edges = [
            ("X", "LN1"),
            ("LN1", "Q"), ("LN1", "K"), ("LN1", "V"),
            ("Q", "QK"), ("K", "QK"),
            ("QK", "Softmax"),
            ("Softmax", "SV"), ("V", "SV"),
            ("SV", "O"),
            ("O", "Add1"), ("X", "Add1"),
            ("Add1", "LN2"),
            ("LN2", "FFN_W1"), ("LN2", "FFN_W3"),
            ("FFN_W1", "SwiGLU"), ("FFN_W3", "SwiGLU"),
            ("SwiGLU", "FFN_W2"),
            ("FFN_W2", "Add2"), ("Add1", "Add2"),
        ]
        return ops, edges

class OPTDef(BaseModelDef):
    name = "opt"
    def layer_blueprint(self):
        ops = [
            "X","LN1","Q","K","V","QK","Softmax","SV","O","Add1","LN2","FFN_W1","GELU","FFN_W2","Add2",
        ]
        edges = [
            ("X","LN1"),
            ("LN1","Q"), ("LN1","K"), ("LN1","V"),
            ("Q","QK"), ("K","QK"),
            ("QK","Softmax"),
            ("Softmax","SV"), ("V","SV"),
            ("SV","O"),
            ("O","Add1"), ("X","Add1"),
            ("Add1","LN2"),
            ("LN2","FFN_W1"),
            ("FFN_W1","GELU"),
            ("GELU","FFN_W2"),
            ("FFN_W2","Add2"), ("Add1","Add2"),
        ]
        return ops, edges

class PaLMDef(BaseModelDef):
    name = "palm"
    def layer_blueprint(self):
        return LLaMADef().layer_blueprint()

def get_model_def(model_type: str) -> BaseModelDef:
    t = model_type.lower()
    if t == "llama":
        return LLaMADef()
    if t == "opt":
        return OPTDef()
    if t == "palm":
        return PaLMDef()
    raise ValueError(f"Unknown model type: {model_type}")
