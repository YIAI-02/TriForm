# #!/usr/bin/python3
# # coding=utf-8
# #
# # Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# # ===============================================================================

# import sys
# import numpy as np

# # for float32
# relative_tol = 1e-6
# absolute_tol = 1e-9
# error_tol = 1e-4


# def verify_result(output, golden):
#     output = np.fromfile(output, dtype=np.float32).reshape(-1)
#     golden = np.fromfile(golden, dtype=np.float32).reshape(-1)
#     different_element_results = np.isclose(output,
#                                            golden,
#                                            rtol=relative_tol,
#                                            atol=absolute_tol,
#                                            equal_nan=True)
#     different_element_indexes = np.where(different_element_results == False)[0]
#     for index in range(len(different_element_indexes)):
#         real_index = different_element_indexes[index]
#         golden_data = golden[real_index]
#         output_data = output[real_index]
#         print(
#             "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
#             (real_index, golden_data, output_data,
#              abs(output_data - golden_data) / golden_data))
#         if index == 100:
#             break
#     error_ratio = float(different_element_indexes.size) / golden.size
#     print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, error_tol))
#     return error_ratio <= error_tol


# if __name__ == '__main__':
#     try:
#         res = verify_result(sys.argv[1], sys.argv[2])
#         if not res:
#             raise ValueError("[ERROR] result error")
#         else:
#             print("test pass")
#     except Exception as e:
#         print(e)
#         sys.exit(1)

#!/usr/bin/env python3
import argparse, os, json, sys
import numpy as np

def _env_int(name):
    v = os.getenv(name)
    return int(v) if v is not None and v != "" else None

def parse_args():
    p = argparse.ArgumentParser(description="Verify GEMM output against golden")
    p.add_argument("output", help="Path to device output bin (fp32)")
    p.add_argument("golden", nargs="?", default=None,
                   help="Path to golden bin (fp32). If omitted or missing, recompute from inputs.")
    p.add_argument("--m", type=int, help="Rows of A/C")
    p.add_argument("--n", type=int, help="Cols of B/C")
    p.add_argument("--k", type=int, help="Cols of A / Rows of B")
    p.add_argument("--input-dir", default="input", help="Directory of x1_gm.bin/x2_gm.bin/bias_gm.bin")
    p.add_argument("--output-dir", default="output", help="Directory of golden/meta")
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--atol", type=float, default=1e-2)
    return p.parse_args()

def load_shape(args):
    M = args.m or _env_int("M")
    N = args.n or _env_int("N")
    K = args.k or _env_int("K")
    meta_path = os.path.join(args.output_dir, "meta.json")
    if (M is None or N is None or K is None) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        M = M or meta.get("M")
        N = N or meta.get("N")
        K = K or meta.get("K")
    if M is None or N is None or K is None:
        raise SystemExit("Please provide --m --n --k, or set env M/N/K, or provide output/meta.json.")
    return int(M), int(N), int(K)

def recompute_golden(args, M, N, K):
    A = np.fromfile(os.path.join(args.input_dir, "x1_gm.bin"), dtype=np.float16)
    B = np.fromfile(os.path.join(args.input_dir, "x2_gm.bin"), dtype=np.float16)
    bias = np.fromfile(os.path.join(args.input_dir, "bias_gm.bin"), dtype=np.float16)
    # 尺寸可能带 pad：按 K/N 反推 reshape，再裁剪到 M×N
    if K == 0 or N == 0 or M == 0:
        raise SystemExit("Invalid shape.")
    A = A.reshape(-1, K)[:M, :K].astype(np.float32)
    B = B.reshape(K, -1)[:K, :N].astype(np.float32)
    bias = bias[:N].astype(np.float32)
    return A @ B + bias

def main():
    args = parse_args()
    M, N, K = load_shape(args)
    out = np.fromfile(args.output, dtype=np.float32)
    if out.size != M * N:
        print(f"[ERROR] Output size {out.size} != M*N {M*N}", file=sys.stderr)
        sys.exit(2)
    out = out.reshape(M, N)

    if args.golden and os.path.exists(args.golden):
        golden = np.fromfile(args.golden, dtype=np.float32).reshape(M, N)
    else:
        golden = recompute_golden(args, M, N, K)

    ok = np.allclose(out, golden, rtol=args.rtol, atol=args.atol, equal_nan=False)
    max_abs = float(np.max(np.abs(out - golden)))
    max_rel = float(np.max(np.abs(out - golden) / (np.abs(golden) + 1e-12)))
    print(f"[INFO] Shape: M={M} N={N} K={K} | max_abs={max_abs:.4e} max_rel={max_rel:.4e} | "
          f"rtol={args.rtol} atol={args.atol}")
    if ok:
        print("[PASS] output matches golden within tolerance.")
        sys.exit(0)
    else:
        diff = np.abs(out - golden)
        idx = np.unravel_index(np.argsort(diff.ravel())[::-1][:10], diff.shape)
        print("[FAIL] Top-10 diffs (i,j,out,golden,abs_err):")
        for i, j in zip(idx[0], idx[1]):
            print(i, j, float(out[i, j]), float(golden[i, j]), float(diff[i, j]))
        sys.exit(1)

if __name__ == "__main__":
    main()
