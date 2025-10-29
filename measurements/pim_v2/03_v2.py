# -*- coding: utf-8 -*-
"""
Improved latency model fitting with better model selection
"""

from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

OP_FEATURE_SPEC: Dict[str, List[str]] = {
    "score": ["seqlen", "n_heads"],
    "attn_score": ["seqlen", "n_heads"],
    "softmax": ["seqlen", "n_heads"],
    "attn_out": ["seqlen", "n_heads"],
    "output": ["seqlen", "n_heads"],
    "matmul": ["vector_dim", "matrix_col"],
    "weight_af": ["vector_dim", "matrix_col"],
    "rmsnorm": ["dim"],
    "rope": ["dim"],
    "silu": ["ffn_dim"],
    "gelu": ["ffn_dim"],
    "residual": ["dim"],
}

MATMUL_OPS = ["q_proj", "k_proj", "v_proj", "wo_proj", 
              "ffn_up", "ffn_gate", "ffn_down"]

def read_results_csv(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def extract_samples_for_op(rows: List[Dict[str, Any]], op: str) -> tuple[np.ndarray, np.ndarray]:
    if op not in OP_FEATURE_SPEC:
        raise ValueError(f"Unknown op: {op}")
    
    feature_names = OP_FEATURE_SPEC[op]
    target_ops = MATMUL_OPS if op == "matmul" else [op]
    
    samples_X = []
    samples_y = []
    
    for row in rows:
        row_op = row.get("op", "").strip()
        if row_op not in target_ops:
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

def find_best_model(X: np.ndarray, y: np.ndarray, verbose: bool = False) -> Pipeline:
    """自动选择最佳模型"""
    candidates = []
    
    # Try different polynomial degrees and regularization
    for degree in [1, 2, 3]:
        for alpha in [0.0, 0.1, 1.0, 10.0, 100.0]:
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=alpha))
            ])
            
            try:
                pipeline.fit(X, y)
                
                # Use cross-validation for better evaluation
                cv_scores = cross_val_score(pipeline, X, y, cv=min(5, len(y)), 
                                           scoring='r2')
                cv_r2 = cv_scores.mean()
                
                # Train metrics
                y_pred = pipeline.predict(X)
                train_r2 = pipeline.score(X, y)
                train_mape = np.mean(np.abs((y - y_pred) / y)) * 100
                
                candidates.append({
                    "pipeline": pipeline,
                    "degree": degree,
                    "alpha": alpha,
                    "cv_r2": cv_r2,
                    "train_r2": train_r2,
                    "train_mape": train_mape
                })
                
                if verbose:
                    print(f"  deg={degree}, alpha={alpha:6.1f}: "
                          f"CV-R²={cv_r2:.4f}, Train-R²={train_r2:.4f}, "
                          f"MAPE={train_mape:.2f}%")
            
            except Exception as e:
                if verbose:
                    print(f"  deg={degree}, alpha={alpha:6.1f}: FAILED ({e})")
                continue
    
    if not candidates:
        raise RuntimeError("No valid models found")
    
    # Sort by CV R² (prefer generalization)
    candidates.sort(key=lambda x: x["cv_r2"], reverse=True)
    best = candidates[0]
    
    if verbose:
        print(f"\nSelected: degree={best['degree']}, alpha={best['alpha']}")
        print(f"  CV R² = {best['cv_r2']:.4f}")
        print(f"  Train R² = {best['train_r2']:.4f}")
        print(f"  MAPE = {best['train_mape']:.2f}%")
    
    return best["pipeline"]

def fit_model_for_op(X: np.ndarray, y: np.ndarray, op: str, 
                     verbose: bool = False) -> Dict[str, Any]:
    """Fit best model for operation"""
    
    if verbose:
        print(f"\nFinding best model for {op}...")
    
    best_pipeline = find_best_model(X, y, verbose=verbose)
    
    # Extract model parameters
    poly = best_pipeline.named_steps['poly']
    scaler = best_pipeline.named_steps['scaler']
    ridge = best_pipeline.named_steps['ridge']
    
    feature_names = OP_FEATURE_SPEC[op]
    poly_feature_names = poly.get_feature_names_out(feature_names).tolist()
    
    model_params = {
        "type": "poly_ridge",
        "degree": poly.degree,
        "alpha": ridge.alpha,
        "feature_names": poly_feature_names,
        "coef": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
        "scale_mean": scaler.mean_.tolist(),
        "scale_std": scaler.scale_.tolist(),
        "X_range": [[float(X[:, i].min()), float(X[:, i].max())] 
                    for i in range(X.shape[1])]
    }
    
    return model_params

def evaluate_model(model_params: Dict[str, Any], X: np.ndarray, 
                   y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Evaluate model performance"""
    
    # Reconstruct pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=model_params["degree"], include_bias=True)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=model_params.get("alpha", 1.0)))
    ])
    
    # Set fitted parameters
    pipeline.named_steps['scaler'].mean_ = np.array(model_params["scale_mean"])
    pipeline.named_steps['scaler'].scale_ = np.array(model_params["scale_std"])
    pipeline.named_steps['scaler'].n_features_in_ = len(model_params["scale_mean"])
    pipeline.named_steps['ridge'].coef_ = np.array(model_params["coef"])
    pipeline.named_steps['ridge'].intercept_ = model_params["intercept"]
    
    # Predict
    X_poly = pipeline.named_steps['poly'].fit_transform(X)
    X_scaled = pipeline.named_steps['scaler'].transform(X_poly)
    y_pred = pipeline.named_steps['ridge'].predict(X_scaled)
    
    # Metrics
    residuals = y - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / y)) * 100)
    
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "valid_samples": int(len(y))
    }

def fit_all_ops(results_csv: Path, out_model: Path,
                out_summary_csv: Optional[Path] = None,
                verbose: bool = False) -> None:
    """Fit models for all operations"""
    
    rows = read_results_csv(results_csv)
    
    # Count samples per op
    op_counts = {}
    for row in rows:
        op = row.get("op", "").strip()
        if op and row.get("cycles", "").strip():
            op_counts[op] = op_counts.get(op, 0) + 1
    
    print(f"Found {len(op_counts)} unique ops with data:")
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count} samples")
    
    matmul_count = sum(op_counts.get(op, 0) for op in MATMUL_OPS)
    if matmul_count > 0:
        print(f"\n  matmul (merged): {matmul_count} samples")
    
    models = {}
    summary_rows = []
    
    for op in OP_FEATURE_SPEC.keys():
        if op == "matmul":
            if matmul_count == 0:
                print(f"\nSkipping {op}: no data")
                continue
        else:
            if op not in op_counts:
                print(f"\nSkipping {op}: no data")
                continue
        
        print(f"\n{'='*60}")
        print(f"Fitting {op}")
        print(f"{'='*60}")
        
        try:
            X, y = extract_samples_for_op(rows, op)
            
            if X.shape[0] < 3:
                print(f"SKIP: insufficient samples ({X.shape[0]})")
                continue
            
            model_params = fit_model_for_op(X, y, op, verbose=verbose)
            metrics = evaluate_model(model_params, X, y, OP_FEATURE_SPEC[op])
            
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
                "degree": model_params["degree"],
                "alpha": model_params.get("alpha", 0.0),
                "rmse": f"{metrics['rmse']:.2f}",
                "mae": f"{metrics['mae']:.2f}",
                "mape": f"{metrics['mape']:.2f}",
                "r2": f"{metrics['r2']:.4f}",
            })
            
            print(f"\nResults:")
            print(f"  Samples: {X.shape[0]}")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.1f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not models:
        print("\nError: No models were fitted!")
        sys.exit(1)
    
    # Save model
    out_model.parent.mkdir(parents=True, exist_ok=True)
    with out_model.open("w", encoding="utf-8") as f:
        json.dump({"models": models}, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: Saved {len(models)} models to {out_model}")
    print(f"{'='*60}")
    
    # Save summary
    if out_summary_csv and summary_rows:
        with out_summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved summary to {out_summary_csv}")

def predict_cycles(model_json: Path, op: str, **kwargs) -> float:
    """Predict cycles using improved model"""
    with model_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    models = data["models"]
    
    # Handle matmul ops
    if op in MATMUL_OPS:
        dim = kwargs.get("dim")
        n_heads = kwargs.get("n_heads")
        n_kv_heads = kwargs.get("n_kv_heads", n_heads)
        ffn_dim_mul = kwargs.get("ffn_dim_mul")
        
        if dim is None or n_heads is None or ffn_dim_mul is None:
            raise ValueError(f"Missing parameters for {op}")
        
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
    
    # Extract query features
    query_features = []
    for feat in feature_names:
        if feat not in kwargs:
            raise ValueError(f"Missing feature '{feat}' for op '{op}'")
        query_features.append(float(kwargs[feat]))
    
    X_query = np.array([query_features])
    
    # Predict using poly_ridge model
    poly = PolynomialFeatures(degree=model_params["degree"], include_bias=True)
    X_poly = poly.fit_transform(X_query)
    
    # Scale
    X_scaled = (X_poly - np.array(model_params["scale_mean"])) / np.array(model_params["scale_std"])
    
    # Predict
    prediction = np.dot(X_scaled, model_params["coef"]) + model_params["intercept"]
    
    return float(prediction[0])

def main():
    ap = argparse.ArgumentParser(description="Improved latency model fitting")
    
    sub = ap.add_subparsers(dest="cmd", required=True)
    
    # Fit
    fit_parser = sub.add_parser("fit", help="Fit improved models")
    fit_parser.add_argument("--results-csv", type=Path, required=True)
    fit_parser.add_argument("--out-model", type=Path, required=True)
    fit_parser.add_argument("--out-summary-csv", type=Path, default=None)
    fit_parser.add_argument("--verbose", action="store_true")
    
    # Predict
    pred_parser = sub.add_parser("predict", help="Predict cycles")
    pred_parser.add_argument("--model-json", type=Path, required=True)
    pred_parser.add_argument("--op", type=str, required=True)
    pred_parser.add_argument("--dim", type=int)
    pred_parser.add_argument("--seqlen", type=int)
    pred_parser.add_argument("--n-heads", type=int)
    pred_parser.add_argument("--n-kv-heads", type=int)
    pred_parser.add_argument("--ffn-dim-mul", type=float)
    pred_parser.add_argument("--ffn-dim", type=int)
    
    args = ap.parse_args()
    
    if args.cmd == "fit":
        fit_all_ops(args.results_csv, args.out_model, 
                   args.out_summary_csv, args.verbose)
    
    elif args.cmd == "predict":
        kwargs = {}
        for arg in ["dim", "seqlen", "n_heads", "n_kv_heads", "ffn_dim_mul", "ffn_dim"]:
            val = getattr(args, arg, None)
            if val is not None:
                kwargs[arg] = val
        
        try:
            cycles = predict_cycles(args.model_json, args.op, **kwargs)
            print(f"{cycles:.2f}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
