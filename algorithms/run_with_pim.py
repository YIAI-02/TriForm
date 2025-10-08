#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_with_pim.py
在不改你原始源码的前提下，把 PIM 公式 CostModel 注入到运行时：
- 用 cost_model_pim.CostModelPIM 覆盖 cost_model.CostModel
- 然后调用 main.main() 启动原有 CLI

用法：
    python run_with_pim.py --model llama --shape ../configs/mpt_shape.json --batch 1 --seq 2048
"""
import sys, importlib, json, os

# 先加载我们的替代实现，并把它注册成模块名 "cost_model"
import cost_model as _cmp
sys.modules["cost_model"] = _cmp

# 再加载你的原 main，并调用入口
import main as _main

if __name__ == "__main__":
    # 尝试从 --shape 读取 n_heads，写入环境变量，帮助 PIM 公式更准确
    try:
        if "--shape" in sys.argv:
            i = sys.argv.index("--shape")
            if i+1 < len(sys.argv):
                sp = sys.argv[i+1]
                with open(sp, "r") as f:
                    sj = json.load(f)
                nh = sj.get("q_head_num") or sj.get("num_attention_heads") or sj.get("n_head")
                if nh:
                    os.environ.setdefault("N_HEADS", str(int(nh)))
    except Exception:
        pass
    _main.main()
