#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, subprocess, tempfile, re, sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

def load_model(model_json: Path) -> Dict[str, Any]:
    """加载模型JSON文件"""
    with model_json.open("r", encoding="utf-8") as f:
        return json.load(f)

def predict_with_model(model_data: Dict[str, Any], op: str, **kwargs) -> float:
    """使用模型预测cycles，复用03的逻辑"""
    from importlib import import_module

    models = model_data["models"]
    
    MATMUL_OPS = ["q_proj", "k_proj", "v_proj", "wo_proj", 
                  "ffn_up", "ffn_gate", "ffn_down"]
    
    if op in MATMUL_OPS:
        dim = kwargs.get("dim")
        n_heads = kwargs.get("n_heads")
        n_kv_heads = kwargs.get("n_kv_heads", n_heads)
        ffn_dim_mul = kwargs.get("ffn_dim_mul")
        if dim is None or n_heads is None or ffn_dim_mul is None:
            raise ValueError(f"Missing required parameters for {op}")
        ffn_dim = int(ffn_dim_mul * dim)
        if op == "q_proj":
            kwargs["vector_dim"], kwargs["matrix_col"] = dim, dim
        elif op == "k_proj":
            kwargs["vector_dim"], kwargs["matrix_col"] = dim, n_kv_heads * (dim // n_heads)
        elif op == "v_proj":
            kwargs["vector_dim"], kwargs["matrix_col"] = dim, n_kv_heads * (dim // n_heads)
        elif op == "wo_proj":
            kwargs["vector_dim"], kwargs["matrix_col"] = dim, dim
        elif op == "ffn_up":
            kwargs["vector_dim"], kwargs["matrix_col"] = dim, ffn_dim
        elif op == "ffn_gate":
            kwargs["vector_dim"], kwargs["matrix_col"] = dim, ffn_dim
        elif op == "ffn_down":
            kwargs["vector_dim"], kwargs["matrix_col"] = ffn_dim, dim
        op = "matmul"
    
    if op not in models:
        raise ValueError(f"Op '{op}' not found in model")
    
    model = models[op]
    feature_names = model["feature_names"]
    model_params = model["model_params"]
    model_type = model_params["type"]
    
    # 提取特征
    query_features = []
    for feat in feature_names:
        if feat not in kwargs:
            raise ValueError(f"Missing feature '{feat}' for op '{op}'")
        query_features.append(float(kwargs[feat]))
    query_point = np.array([query_features])

    # 可选特征标准化
    if "feature_scaler" in model_params:
        scaler = model_params["feature_scaler"]
        mean = np.array(scaler.get("mean", [0]*len(query_features)))
        scale = np.array(scaler.get("scale", [1]*len(query_features)))
        query_point = (query_point - mean) / scale

    if model_type == "linear_1d":
        prediction = model_params["coef"] * query_features[0] + model_params.get("intercept", 0.0)

    elif model_type in ("poly_2d", "poly_ridge", "poly"):
        # 统一处理任意维度多项式
        from sklearn.preprocessing import PolynomialFeatures
        degree = model_params.get("degree", 2)
        poly_features = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly_features.fit_transform(query_point)
        coef = np.array(model_params["coef"])
        intercept = model_params.get("intercept", 0.0)
        if X_poly.shape[1] != coef.shape[0]:
            raise ValueError(f"Polynomial feature size mismatch: X_poly={X_poly.shape}, coef={coef.shape}")
        prediction = float(X_poly @ coef + intercept)

    elif model_type == "linear_nd":
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        X_train = np.array(model_params["X"])
        y_train = np.array(model_params["y"])
        interpolator = LinearNDInterpolator(X_train, y_train, fill_value=np.nan)
        prediction = interpolator(query_point)[0]
        if np.isnan(prediction):
            nearest_interp = NearestNDInterpolator(X_train, y_train)
            prediction = nearest_interp(query_point)[0]

    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return float(prediction)

def run_ground_truth(
    op: str,
    pim_config: Path,
    ramulator_config: Path,
    dim: int,
    n_heads: int,
    n_kv_heads: Optional[int] = None,
    seqlen: Optional[int] = None,
    ffn_dim_mul: float = 4.0,
    with_af: bool = False
) -> Tuple[float, Path]:

    if n_kv_heads is None:
        n_kv_heads = n_heads
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp(prefix="validate_"))
    
    # 构建01命令
    cmd_parts = [
        "python3", "01_gentrace.py",
        "--pim-config", str(pim_config),
        "--ops", op,
        "--out-dir", str(temp_dir),
        "--mode", "single",
        "--dim", str(dim),
        "--n-heads", str(n_heads),
        "--n-kv-heads", str(n_kv_heads),
        "--ffn-mult", str(ffn_dim_mul)
    ]
    
    if seqlen:
        cmd_parts.extend(["--seqlens", str(seqlen)])
    
    if with_af:
        cmd_parts.append("--with-af")
    
    print(f"  Generating trace: {' '.join(cmd_parts)}")
    result = subprocess.run(cmd_parts, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"01_gentrace.py failed:\n{result.stderr}")
    
    # 查找生成的trace文件
    trace_files = list(temp_dir.rglob("*.trace"))
    if not trace_files:
        raise RuntimeError(f"No trace file generated in {temp_dir}")
    
    trace_path = trace_files[0]
    print(f"  Generated trace: {trace_path}")
    
    # 运行ramulator
    cmd = f"./ramulator2 -f {ramulator_config} -t {trace_path}"
    print(f"  Running ramulator: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Ramulator failed:\n{result.stderr}")
    
    # 解析cycles
    pattern = r"memory_system_cycles:\s*([0-9]+)"
    match = re.search(pattern, result.stdout)
    
    if not match:
        raise RuntimeError(f"Could not parse cycles from ramulator output:\n{result.stdout}")
    
    cycles = int(match.group(1))
    print(f"  Ground truth cycles: {cycles}")
    
    return cycles, trace_path

def validate_single(
    model_json: Path,
    pim_config: Path,
    ramulator_config: Path,
    op: str,
    dim: int,
    n_heads: int,
    n_kv_heads: Optional[int] = None,
    seqlen: Optional[int] = None,
    ffn_dim_mul: float = 4.0,
    with_af: bool = False,
    cleanup: bool = True
) -> Dict[str, Any]:

    print(f"\n{'='*60}")
    print(f"Validating: op={op}, dim={dim}, n_heads={n_heads}, "
          f"n_kv_heads={n_kv_heads or n_heads}, seqlen={seqlen}, "
          f"ffn_dim_mul={ffn_dim_mul}")
    print(f"{'='*60}")
    
    # 加载模型
    model_data = load_model(model_json)
    
    # 准备预测参数
    pred_kwargs = {
        "dim": dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads or n_heads,
        "ffn_dim_mul": ffn_dim_mul
    }
    
    if seqlen is not None:
        pred_kwargs["seqlen"] = seqlen
    
    # 预测
    print("\n[1/2] Predicting with model...")
    try:
        predicted_cycles = predict_with_model(model_data, op, **pred_kwargs)
        print(f"  Predicted cycles: {predicted_cycles:.2f}")
    except Exception as e:
        print(f"  Prediction failed: {e}")
        return {"error": str(e)}
    
    # 获取ground truth
    print("\n[2/2] Running ground truth (01 + ramulator)...")
    try:
        ground_truth_cycles, trace_path = run_ground_truth(
            op=op,
            pim_config=pim_config,
            ramulator_config=ramulator_config,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            seqlen=seqlen,
            ffn_dim_mul=ffn_dim_mul,
            with_af=with_af
        )
    except Exception as e:
        print(f"  Ground truth failed: {e}")
        return {"error": str(e), "predicted": predicted_cycles}
    
    # 计算误差
    absolute_error = abs(predicted_cycles - ground_truth_cycles)
    relative_error = (absolute_error / ground_truth_cycles) * 100 if ground_truth_cycles > 0 else float('inf')
    
    result = {
        "op": op,
        "dim": dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads or n_heads,
        "seqlen": seqlen,
        "ffn_dim_mul": ffn_dim_mul,
        "predicted_cycles": predicted_cycles,
        "ground_truth_cycles": ground_truth_cycles,
        "absolute_error": absolute_error,
        "relative_error_pct": relative_error,
        "trace_path": str(trace_path)
    }
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Predicted:       {predicted_cycles:>12.2f} cycles")
    print(f"Ground Truth:    {ground_truth_cycles:>12} cycles")
    print(f"Absolute Error:  {absolute_error:>12.2f} cycles")
    print(f"Relative Error:  {relative_error:>12.2f}%")
    print(f"{'='*60}")
    
    # 清理临时文件
    if cleanup:
        import shutil
        shutil.rmtree(trace_path.parent.parent, ignore_errors=True)
        print(f"\nCleaned up temporary files")
    else:
        print(f"\nTrace saved at: {trace_path}")
    
    return result

def validate_batch(
    model_json: Path,
    pim_config: Path,
    ramulator_config: Path,
    test_configs: list[Dict[str, Any]],
    out_json: Optional[Path] = None,
    cleanup: bool = True
) -> list[Dict[str, Any]]:
    """
    批量验证多个配置
    """
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n\n{'#'*60}")
        print(f"Test {i}/{len(test_configs)}")
        print(f"{'#'*60}")
        
        result = validate_single(
            model_json=model_json,
            pim_config=pim_config,
            ramulator_config=ramulator_config,
            cleanup=cleanup,
            **config
        )
        
        results.append(result)
    
    # 统计总结
    successful = [r for r in results if "error" not in r]
    
    if successful:
        errors = [r["relative_error_pct"] for r in successful]
        print(f"\n\n{'='*60}")
        print(f"BATCH VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests:        {len(results)}")
        print(f"Successful:         {len(successful)}")
        print(f"Failed:             {len(results) - len(successful)}")
        print(f"Mean rel. error:    {np.mean(errors):.2f}%")
        print(f"Median rel. error:  {np.median(errors):.2f}%")
        print(f"Max rel. error:     {np.max(errors):.2f}%")
        print(f"Min rel. error:     {np.min(errors):.2f}%")
        print(f"{'='*60}")
    
    # 保存结果
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_json}")
    
    return results

def main():
    ap = argparse.ArgumentParser(
        description="Validate latency model predictions against ground truth"
    )
    
    ap.add_argument("--model-json", type=Path, required=True)
    ap.add_argument("--pim-config", type=Path, required=True)
    ap.add_argument("--ramulator-config", type=Path, required=True)
    
    # Test configuration
    ap.add_argument("--op", type=str, required=True,
                    help="Operation to test")
    ap.add_argument("--dim", type=int, required=True,
                    help="Model dimension")
    ap.add_argument("--n-heads", type=int, required=True,
                    help="Number of attention heads")
    ap.add_argument("--n-kv-heads", type=int, default=None,
                    help="Number of KV heads (default: same as n_heads)")
    ap.add_argument("--seqlen", type=int, default=None,
                    help="Sequence length (for attention ops)")
    ap.add_argument("--ffn-dim-mul", type=float, default=4.0,
                    help="FFN dimension multiplier")
    ap.add_argument("--with-af", action="store_true",
                    help="Include activation function")
    
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Keep temporary trace files")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Save validation results to JSON")
    
    args = ap.parse_args()
    
    result = validate_single(
        model_json=args.model_json,
        pim_config=args.pim_config,
        ramulator_config=args.ramulator_config,
        op=args.op,
        dim=args.dim,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        seqlen=args.seqlen,
        ffn_dim_mul=args.ffn_dim_mul,
        with_af=args.with_af,
        cleanup=not args.no_cleanup
    )
    
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {args.out_json}")

    if "error" in result:
        sys.exit(1)
    elif result.get("relative_error_pct", 0) > 20:  # 20%阈值
        print("\nWarning: Relative error > 20%")
        sys.exit(2)

if __name__ == "__main__":
    main()