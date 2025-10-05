import unittest
import json
from pathlib import Path
from mvm import emit_mvm
from dialect import TraceDialect
from weight_layout import plan_weight_layout_for_linear

class TestMVM(unittest.TestCase):
    
    def test_small_matrix_vector_multiply(self):
        """测试向量矩阵乘法的trace生成"""
        K = 2048  # 输入向量长度
        N = 4     # 输出向量长度
        B = 1     # 批大小
        S = 1     # 序列长度

        config_path = Path(__file__).parent.parent.parent / "configs" / "pim.json"
        with open(config_path, 'r') as f:
            pim_config = json.load(f)
        
        # 创建测试用的dialect
        dialect = TraceDialect(
            aim_ops={"AiM MAC_ABK", "AiM MAC_SBK", "AiM RD_MAC", "AiM WR_GB", 
                    "AiM WR_BIAS", "AiM EOC"},
            rw_ops={"W CFR"}
        )
        
        op = {
            "inputs": [{"shape": [B, S, K]}],  
            "outputs": [{"shape": [B, S, N]}],  
            "weights": [{"shape": [N, K]}]   
        }
        
        # BF16格式，每个元素2字节
        bpe = 2
        
        # 创建权重布局 - 物理存储为4*4096
        weight_layout = plan_weight_layout_for_linear([N, K], pim_config, bpe)
        
        # 调用emit_mvm生成trace
        trace_lines = emit_mvm(
            op=op, 
            pim=pim_config, 
            dialect=dialect, 
            bpe=bpe,
            weight_layout=weight_layout,
            defer_final_rdmac_to_activation=False
        )
        
        trace_path = Path(__file__).parent / "test_mvm_trace.txt"
        with open(trace_path, "w") as f:
            for line in trace_lines:
                f.write(line + "\n")


if __name__ == "__main__":
    unittest.main()