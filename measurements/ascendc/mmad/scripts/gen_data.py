#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import argparse, os, json

# def gen_golden_data():
#     M = 32
#     N = 32
#     K = 32

#     x1_gm = np.random.uniform(1, 10, [M, K]).astype(np.float16)
#     x2_gm = np.random.uniform(1, 10, [K, N]).astype(np.float16)
#     bias_gm = np.random.uniform(1, 10, [N]).astype(np.float16)
#     golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32)) + bias_gm.astype(np.float32)).astype(np.float32)
#     os.system("mkdir -p input")
#     os.system("mkdir -p output")
#     x1_gm.tofile("./input/x1_gm.bin")
#     x2_gm.tofile("./input/x2_gm.bin")
#     bias_gm.tofile("./input/bias_gm.bin")
#     golden.tofile("./output/golden.bin")


# if __name__ == "__main__":
#     gen_golden_data()


def _env_int(name, default=None):
    v = os.getenv(name)
    return int(v) if v is not None and v != "" else default

def parse_args():
    p = argparse.ArgumentParser(description="Generate GEMM inputs (fp16) and golden (fp32)")
    p.add_argument("--m", type=int, default=_env_int("M", 32), help="Rows of A/C (default: 32 or env M)")
    p.add_argument("--n", type=int, default=_env_int("N", 32), help="Cols of B/C (default: 32 or env N)")
    p.add_argument("--k", type=int, default=_env_int("K", 32), help="Cols of A / Rows of B (default: 32 or env K)")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument("--input-dir", default="input", help="Directory to write A/B/bias bins")
    p.add_argument("--output-dir", default="output", help="Directory to write golden/meta")
    p.add_argument("--pad16", action="store_true",
                   help="Pad A/B/bias to multiples of 16 when writing inputs; golden still uses real MxN")
    return p.parse_args()

def round_up16(x): return (x + 15) // 16 * 16

def main():
    args = parse_args()
    M, N, K = int(args.m), int(args.n), int(args.k)
    Mp = round_up16(M) if args.pad16 else M
    Np = round_up16(N) if args.pad16 else N
    Kp = round_up16(K) if args.pad16 else K

    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    # 生成真实尺寸的随机数，再拷到（可能对齐后的）大张量里
    A = np.zeros((Mp, Kp), dtype=np.float16)
    B = np.zeros((Kp, Np), dtype=np.float16)
    bias = np.zeros((Np,), dtype=np.float16)
    A[:M, :K] = rng.normal(0, 1, size=(M, K)).astype(np.float16)
    B[:K, :N] = rng.normal(0, 1, size=(K, N)).astype(np.float16)
    bias[:N] = rng.normal(0, 1, size=(N,)).astype(np.float16)

    # 写入二进制（与 C++ 侧一致：A/B/bias 都是 half=2 字节）：
    A.tofile(os.path.join(args.input_dir, "x1_gm.bin"))
    B.tofile(os.path.join(args.input_dir, "x2_gm.bin"))
    bias.tofile(os.path.join(args.input_dir, "bias_gm.bin"))

    # 计算金标（fp32，按真实 MxN，不包含 pad 区域）
    C_pad = A.astype(np.float32) @ B.astype(np.float32) + bias.astype(np.float32)
    C_true = C_pad[:M, :N].astype(np.float32)
    C_true.tofile(os.path.join(args.output_dir, "golden.bin"))
    print(f"[INFO] Generated inputs ({Mp}x{Kp}, {Kp}x{Np}, {Np}) and golden ({M}x{N}).")

if __name__ == "__main__":
    main()
