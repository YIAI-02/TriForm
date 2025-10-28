#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 操作特征规格
OP_FEATURE_SPEC: Dict[str, List[str]] = {
    # attention：based on seqlen and n_heads
    "score":      ["seqlen", "n_heads"],
    "attn_score": ["seqlen", "n_heads"],
    "softmax":    ["seqlen", "n_heads"],
    "attn_out":   ["seqlen", "n_heads"],
    "output":     ["seqlen", "n_heads"],
    
    # weight projection: based on vector_dim and matrix_col
    "weight":     ["vector_dim", "matrix_col"],
    "weight_af":  ["vector_dim", "matrix_col"],
    "q_proj":     ["vector_dim", "matrix_col"],
    "k_proj":     ["vector_dim", "matrix_col"],
    "v_proj":     ["vector_dim", "matrix_col"],
    "wo_proj":    ["vector_dim", "matrix_col"],
    "ffn_up":     ["vector_dim", "matrix_col"],
    "ffn_gate":   ["vector_dim", "matrix_col"],
    "ffn_down":   ["vector_dim", "matrix_col"],
    
    # vector: based on vector_dim (dim)
    "rmsnorm":    ["dim"],
    "rope":       ["dim"],
    "silu":       ["ffn_dim"],
    "gelu":       ["ffn_dim"],
    "residual":   ["dim"],
}

# 操作类型分类
OP_CATEGORIES = {
    "vector": ["rmsnorm", "rope", "silu", "gelu", "residual"],
    "weight_projection": ["weight", "weight_af", "q_proj", "k_proj", "v_proj", 
                          "wo_proj", "ffn_up", "ffn_gate", "ffn_down"],
    "attention": ["score", "attn_score", "softmax", "attn_out", "output"],
}

# 为每种操作类型指定拟合方法
FITTING_STRATEGIES = {
    "vector": "linear_1d",           # 一维线性拟合
    "weight_projection": "poly_2d",  # 二维多项式拟合
    "attention": "linear_nd",        # 多维线性插值
}

def read_results_csv(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def extract_samples_for_op(rows: List[Dict[str, Any]], op: str) -> Tuple[np.ndarray, np.ndarray]:

    if op not in OP_FEATURE_SPEC:
        raise ValueError(f"Unknown op: {op}. Available ops: {list(OP_FEATURE_SPEC.keys())}")
    
    feature_names = OP_FEATURE_SPEC[op]
    samples_X = []
    samples_y = []
    
    for row in rows:
        if row.get("op", "").strip() != op:
            continue
        cycles_str = row.get("cycles", "").strip()
        if not cycles_str or cycles_str == "N/A":
            continue
        
        try:
            cycles = float(cycles_str)
        except ValueError:
            continue
        
        features = []
        valid = True
        for feat_name in feature_names:
            val_str = row.get(feat_name, "").strip()
            if not val_str:
                valid = False
                break
            try:
                features.append(float(val_str))
            except ValueError:
                valid = False
                break
        
        if valid:
            samples_X.append(features)
            samples_y.append(cycles)
    
    X = np.array(samples_X) if samples_X else np.zeros((0, len(feature_names)))
    y = np.array(samples_y) if samples_y else np.zeros((0,))
    
    return X, y

def get_op_category(op: str) -> str:
    """获取操作的类别"""
    for category, ops in OP_CATEGORIES.items():
        if op in ops:
            return category
    return "unknown"

def get_fitting_strategy(op: str) -> str:
    """获取操作的拟合策略"""
    category = get_op_category(op)
    return FITTING_STRATEGIES.get(category, "linear_nd")

# ...existing code...

def fit_linear_1d(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    一维线性拟合，适用于基于单一维度的向量操作
    """
    if X.shape[1] != 1:
        raise ValueError(f"Expected 1D input, got {X.shape[1]}D")
    
    # 使用线性回归
    model = LinearRegression()
    model.fit(X, y)
    
    # 计算拟合优度
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    
    return {
        "type": "linear_1d",
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(r2),
        "X_range": [float(X.min()), float(X.max())],
    }

def fit_poly_2d(X: np.ndarray, y: np.ndarray, degree: int = 2) -> Dict[str, Any]:
    """
    二维多项式拟合，适用于 weight projection 操作
    使用多项式特征 (vector_dim, matrix_col) 及其交叉项
    """
    if X.shape[1] != 2:
        raise ValueError(f"Expected 2D input, got {X.shape[1]}D")
    
    # 创建多项式特征管道
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
        ('linear', LinearRegression())
    ])
    
    poly_model.fit(X, y)
    
    # 获取多项式特征
    poly_features = poly_model.named_steps['poly']
    linear_model = poly_model.named_steps['linear']
    
    # 计算拟合优度
    y_pred = poly_model.predict(X)
    r2 = poly_model.score(X, y)
    
    return {
        "type": "poly_2d",
        "degree": degree,
        "coef": linear_model.coef_.tolist(),
        "intercept": float(linear_model.intercept_),
        "feature_names": poly_features.get_feature_names_out().tolist(),
        "r2": float(r2),
        "X_range": [[float(X[:, i].min()), float(X[:, i].max())] for i in range(X.shape[1])],
    }

def fit_linear_nd(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    多维线性插值，适用于 attention 操作
    """
    # 使用线性插值器
    interpolator = LinearNDInterpolator(X, y, fill_value=np.nan)
    
    # 评估插值效果
    y_pred = interpolator(X)
    valid_mask = ~np.isnan(y_pred)
    
    if np.sum(valid_mask) > 0:
        residuals = y[valid_mask] - y_pred[valid_mask]
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y[valid_mask] - np.mean(y[valid_mask]))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        r2 = 0
    
    return {
        "type": "linear_nd",
        "X": X.tolist(),
        "y": y.tolist(),
        "r2": float(r2),
    }

def fit_model_for_op(X: np.ndarray, y: np.ndarray, op: str) -> Dict[str, Any]:
    """
    根据操作类型选择合适的拟合方法
    """
    strategy = get_fitting_strategy(op)
    
    if strategy == "linear_1d":
        return fit_linear_1d(X, y)
    elif strategy == "poly_2d":
        return fit_poly_2d(X, y, degree=2)
    elif strategy == "linear_nd":
        return fit_linear_nd(X, y)
    else:
        raise ValueError(f"Unknown fitting strategy: {strategy}")

def evaluate_model(model_params: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    评估模型性能
    """
    model_type = model_params["type"]
    
    if model_type == "linear_1d":
        y_pred = model_params["coef"] * X[:, 0] + model_params["intercept"]
    elif model_type == "poly_2d":
        poly_features = PolynomialFeatures(degree=model_params["degree"], include_bias=True)
        X_poly = poly_features.fit_transform(X)
        y_pred = np.dot(X_poly, model_params["coef"]) + model_params["intercept"]
    elif model_type == "linear_nd":
        X_train = np.array(model_params["X"])
        y_train = np.array(model_params["y"])
        interpolator = LinearNDInterpolator(X_train, y_train, fill_value=np.nan)
        y_pred = interpolator(X)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 计算指标
    valid_mask = ~np.isnan(y_pred)
    if np.sum(valid_mask) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "valid_samples": 0}
    
    y_true_valid = y[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    residuals = y_true_valid - y_pred_valid
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true_valid - np.mean(y_true_valid))**2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    
    return {"rmse": rmse, "mae": mae, "r2": r2, "valid_samples": int(np.sum(valid_mask))}

def fit_all_ops(results_csv: Path, out_model: Path, 
                out_summary_csv: Optional[Path] = None) -> None:
    """
    为所有操作拟合模型
    """
    rows = read_results_csv(results_csv)
    op_counts = {}
    for row in rows:
        op = row.get("op", "").strip()
        if op and row.get("cycles", "").strip():
            op_counts[op] = op_counts.get(op, 0) + 1
    
    print(f"Found {len(op_counts)} unique ops with data:")
    for op, count in sorted(op_counts.items()):
        strategy = get_fitting_strategy(op)
        print(f"  {op}: {count} samples (strategy: {strategy})")
    
    models = {}
    summary_rows = []
    
    for op in OP_FEATURE_SPEC.keys():
        if op not in op_counts:
            print(f"\nSkipping {op}: no data")
            continue
        
        strategy = get_fitting_strategy(op)
        print(f"\nFitting {op} (strategy: {strategy})...", end=" ")
        
        try:
            X, y = extract_samples_for_op(rows, op)
            
            if X.shape[0] < 3:
                print(f"SKIP (insufficient samples: {X.shape[0]})")
                continue

            model_params = fit_model_for_op(X, y, op)
            metrics = evaluate_model(model_params, X, y)
            
            models[op] = {
                "feature_names": OP_FEATURE_SPEC[op],
                "num_samples": int(X.shape[0]),
                "model_params": model_params,
                "metrics": metrics,
            }
            
            summary_rows.append({
                "op": op,
                "num_samples": X.shape[0],
                "features": ",".join(OP_FEATURE_SPEC[op]),
                "strategy": strategy,
                "rmse": f"{metrics['rmse']:.2f}",
                "mae": f"{metrics['mae']:.2f}",
                "r2": f"{metrics['r2']:.4f}",
            })
            
            print(f"OK (n={X.shape[0]}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.4f})")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not models:
        print("\nError: No models were fitted!", file=sys.stderr)
        sys.exit(1)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    with out_model.open("w", encoding="utf-8") as f:
        json.dump({"models": models}, f, indent=2)
    
    print(f"\n[SUCCESS] Saved model to {out_model}")

    if out_summary_csv:
        with out_summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["op", "num_samples", "features", 
                                                     "strategy", "rmse", "mae", "r2"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[SUCCESS] Saved summary to {out_summary_csv}")

def predict_cycles(model_json: Path, op: str, **kwargs) -> float:
    """
    使用拟合的模型预测延迟
    """
    with model_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    models = data["models"]
    
    if op not in models:
        raise ValueError(f"Op '{op}' not found in model. Available ops: {list(models.keys())}")
    
    model = models[op]
    feature_names = model["feature_names"]
    model_params = model["model_params"]
    model_type = model_params["type"]

    # 提取查询特征
    query_features = []
    for feat in feature_names:
        if feat not in kwargs:
            raise ValueError(f"Missing feature '{feat}' for op '{op}'. Required: {feature_names}")
        query_features.append(float(kwargs[feat]))

    query_point = np.array([query_features])
    
    # 根据模型类型进行预测
    if model_type == "linear_1d":
        prediction = model_params["coef"] * query_features[0] + model_params["intercept"]
        
        # 检查是否超出训练范围
        x_min, x_max = model_params["X_range"]
        if query_features[0] < x_min or query_features[0] > x_max:
            print(f"Warning: Query point {query_features[0]} is outside training range [{x_min}, {x_max}]", 
                  file=sys.stderr)
    
    elif model_type == "poly_2d":
        poly_features = PolynomialFeatures(degree=model_params["degree"], include_bias=True)
        X_poly = poly_features.fit_transform(query_point)
        prediction = float(np.dot(X_poly, model_params["coef"]) + model_params["intercept"])
        
        # 检查是否超出训练范围
        for i, (x_min, x_max) in enumerate(model_params["X_range"]):
            if query_features[i] < x_min or query_features[i] > x_max:
                print(f"Warning: Feature {i} ({query_features[i]}) is outside training range [{x_min}, {x_max}]", 
                      file=sys.stderr)
    
    elif model_type == "linear_nd":
        X_train = np.array(model_params["X"])
        y_train = np.array(model_params["y"])
        interpolator = LinearNDInterpolator(X_train, y_train, fill_value=np.nan)
        prediction = interpolator(query_point)[0]
        
        if np.isnan(prediction):
            print(f"Warning: Linear interpolation failed (extrapolation). Using nearest neighbor.", 
                  file=sys.stderr)
            nearest_interp = NearestNDInterpolator(X_train, y_train)
            prediction = nearest_interp(query_point)[0]
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return float(prediction)

def main():
    ap = argparse.ArgumentParser(description="Fit and predict latency models with adaptive strategies")

    # fit
    sub = ap.add_subparsers(dest="cmd", required=True)
    fit_parser = sub.add_parser("fit", help="Fit models with adaptive strategies")
    fit_parser.add_argument("--results-csv", type=Path, required=True,
                            help="CSV file from 02_run_ramulator.py")
    fit_parser.add_argument("--out-model", type=Path, required=True,
                            help="Output model JSON file")
    fit_parser.add_argument("--out-summary-csv", type=Path, default=None,
                            help="Optional summary CSV")
    
    # Predict
    pred_parser = sub.add_parser("predict", help="Predict cycles for a given configuration")
    pred_parser.add_argument("--model-json", type=Path, required=True,
                             help="Model JSON file from fit command")
    pred_parser.add_argument("--op", type=str, required=True,
                             help="Operation name")
    pred_parser.add_argument("--dim", type=int, help="Model dimension")
    pred_parser.add_argument("--seqlen", type=int, help="Sequence length")
    pred_parser.add_argument("--n-heads", type=int, help="Number of heads")
    pred_parser.add_argument("--vector-dim", type=int, help="Vector dimension")
    pred_parser.add_argument("--matrix-col", type=int, help="Matrix columns")
    pred_parser.add_argument("--ffn-dim", type=int, help="FFN dimension")
    
    args = ap.parse_args()
    
    if args.cmd == "fit":
        fit_all_ops(args.results_csv, args.out_model, args.out_summary_csv)
    
    elif args.cmd == "predict":
        kwargs = {}
        if args.dim is not None:
            kwargs["dim"] = args.dim
        if args.seqlen is not None:
            kwargs["seqlen"] = args.seqlen
        if args.n_heads is not None:
            kwargs["n_heads"] = args.n_heads
        if args.vector_dim is not None:
            kwargs["vector_dim"] = args.vector_dim
        if args.matrix_col is not None:
            kwargs["matrix_col"] = args.matrix_col
        if args.ffn_dim is not None:
            kwargs["ffn_dim"] = args.ffn_dim
        
        try:
            cycles = predict_cycles(args.model_json, args.op, **kwargs)
            print(f"{cycles:.2f}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()