import json
import unittest
from pathlib import Path
from weight_layout import plan_weight_layout_for_linear, emit_weight_write_trace
from dialect import TraceDialect

class TestWeightLayout(unittest.TestCase):
    
    def setUp(self):
        # 从配置文件加载PIM参数
        config_path = Path(__file__).parent.parent.parent / "configs" / "pim.json"
        with open(config_path, 'r') as f:
            self.pim_config = json.load(f)
        
        # 创建一个基本的dialect，包含W MEM指令
        self.dialect = TraceDialect(aim_ops=set(), rw_ops={"W MEM"})
        
        # 矩阵大小 4096x4096
        self.shape_k_n = [4096, 4]
        self.bpe = 2  # BF16 格式，每个元素2字节
    
    def test_plan_weight_layout_for_linear(self):
        """测试权重布局规划函数"""
        
        # 调用被测函数
        layout = plan_weight_layout_for_linear(self.shape_k_n, self.pim_config, self.bpe)

        print(layout)
        
    #     # 验证segments分布
    #     first_row = layout.rows[0]
    #     self.assertTrue(len(first_row.segs) > 0)
    #     total_cols = sum(seg.cols for seg in first_row.segs)
    #     self.assertEqual(total_cols, 4096)  # 总K元素应为4096
    
    def test_emit_weight_write_trace(self):
        """测试权重写入trace生成函数"""
        
        # 先获取布局
        layout = plan_weight_layout_for_linear(self.shape_k_n, self.pim_config, self.bpe)
        
        # 调用被测函数
        trace_lines = emit_weight_write_trace(layout, self.dialect)

        print("\n".join(trace_lines))
        
    #     # 验证trace文件基本格式
    #     self.assertTrue(len(trace_lines) > 0)
    #     self.assertTrue(trace_lines[0].startswith('#'))  # 第一行应是注释
        
    #     # 统计W MEM指令数量
    #     w_mem_count = sum(1 for line in trace_lines if line.startswith("W MEM"))
        
    #     # 总segment数量就是总指令数
    #     total_segments = sum(len(row.segs) for row in layout.rows)
    #     self.assertEqual(w_mem_count, total_segments)
        
    #     # 检查第一个非注释指令格式
    #     w_mem_line = next(line for line in trace_lines if line.startswith("W MEM"))
    #     parts = w_mem_line.split()
    #     self.assertEqual(len(parts), 5)  # W MEM channel bank row
    #     self.assertEqual(parts[0], "W")
    #     self.assertEqual(parts[1], "MEM")
        
    #     # 验证至少覆盖了所有bank
    #     banks_covered = set()
    #     for line in trace_lines:
    #         if line.startswith("W MEM"):
    #             parts = line.split()
    #             banks_covered.add(int(parts[3]))
        
    #     self.assertEqual(len(banks_covered), 8)  # 应该覆盖所有8个bank

if __name__ == "__main__":
    unittest.main()