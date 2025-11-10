#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced LayerNorm latency model fitting (fixed)
- 多模型：linear / wls / logy / hinge / power / robust / auto
- 自变量：x = batch*dim，可附加 B、D
- 修复：wlsq(..., weights=None) -> wlsq(..., None)
用法示例：
  python fit_layernorm_model_advanced.py \
    --glob "./results/*.csv" \
    --target-file "layernorm" \
    --out-dir "./layernorm_fit_out_adv" \
    --model auto --extra-feats "B,D" --plots 1
"""

import os
import re
import glob
import json
import argparse
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------

def _norm_lines_to_set(lines_spec) -> List[int]:
    if lines_spec is None:
        return set()
    if isinstance(lines_spec, int):
        return {int(lines_spec)}
    if isinstance(lines_spec, (list, tuple, set)):
        return {int(x) for x in lines_spec}
    if isinstance(lines_spec, str):
        s = lines_spec.strip()
        if not s:
            return set()
        return {int(x.strip()) for x in s.split(",") if x.strip()}
    raise ValueError(f"Unsupported TARGET_LINES type: {type(lines_spec)}")

def parse_batch_dim_from_path(path: str) -> Tuple[int, int]:
    s = os.path.basename(path)
    combo_patterns = [
        r"batch[=_\- ]*(\d+)[^\d]+(?:dim|hidden|d)[=_\- ]*(\d+)",
        r"(?:bs|b)[=_\- ]*(\d+)[^\d]+(?:dim|hidden|d)[=_\- ]*(\d+)",
        r"(\d+)[xX](\d+)",
        r"b(\d+)[^\d]+d(\d+)",
        r"bs(\d+)[^\d]+dim(\d+)",
    ]
    for p in combo_patterns:
        m = re.search(p, s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
    b = d = None
    for p in [r"[^\d]batch[=_\- ]*(\d+)", r"[^\d](?:bs|b)[=_\- ]*(\d+)", r"\bB(\d+)\b"]:
        m = re.search(p, s, flags=re.IGNORECASE)
        if m: b = int(m.group(1)); break
    for p in [r"[^\d](?:dim|hidden|d)[=_\- ]*(\d+)", r"\bD(\d+)\b", r"\bH(\d+)\b"]:
        m = re.search(p, s, flags=re.IGNORECASE)
        if m: d = int(m.group(1)); break
    if isinstance(b, int) and isinstance(d, int):
        return b, d
    ints = re.findall(r"(\d+)", s)
    if len(ints) == 2:
        return int(ints[0]), int(ints[1])
    return -1, -1

def read_code_exec_csv(p: str, time_col: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(p)
    except Exception as e:
        raise ValueError(f"{p}: {e}")
    if "code" not in df.columns or time_col not in df.columns:
        raise ValueError(f"{p}: 必须包含列 'code' 与 '{time_col}'")
    m = df["code"].astype(str).str.extract(r'(?P<file>[^:]+):(?P<line>\d+)$')
    df["file"] = m["file"]
    df["line"] = pd.to_numeric(m["line"], errors="coerce").astype("Int64")
    return df

def sum_target_time_us(df: pd.DataFrame, target_file: str, target_lines: List[int], time_col: str) -> float:
    file_series = df["file"].astype(str)
    mask_file = file_series.str.endswith(target_file, na=False) | file_series.str.contains(re.escape(target_file), na=False)
    if len(target_lines) > 0:
        mask_line = df["line"].isin(list(target_lines))
        mask = mask_file & mask_line
    else:
        mask = mask_file
    return float(df.loc[mask, time_col].sum())

def summarize_one_csv(csv_path: str, target_file: str, target_lines: List[int], time_col: str) -> Dict:
    df = read_code_exec_csv(csv_path, time_col)
    b, d = parse_batch_dim_from_path(csv_path)
    BD = b * d if (b > 0 and d > 0) else -1
    meas_us = sum_target_time_us(df, target_file, target_lines, time_col)
    file_total_us = float(df[df["file"].astype(str).str.contains(re.escape(target_file), na=False)][time_col].sum())
    grand_total_us = float(df[time_col].sum())
    present_lines = sorted(df[df["file"].astype(str).str.contains(re.escape(target_file), na=False)]["line"].dropna().unique().tolist())
    found_lines = sorted(df[(df["file"].astype(str).str.contains(re.escape(target_file), na=False)) & (df["line"].isin(list(target_lines)))]["line"].dropna().unique().tolist())
    note = ""
    if (len(target_lines) > 0) and (not found_lines):
        note = f"WARNING: 指定的行号 {sorted(list(target_lines))} 在 {os.path.basename(csv_path)} 中未出现；该点的 meas_us=0"
    if BD <= 0:
        note = (note + " | " if note else "") + "WARNING: 无法从文件名解析 batch 与 dim"
    return {
        "csv": csv_path, "batch": int(b), "dim": int(d), "x": int(BD),
        "meas_us": float(meas_us), "file_total_us": float(file_total_us),
        "grand_total_us": float(grand_total_us), "present_lines": present_lines,
        "found_lines": found_lines, "note": note
    }

# ---------- linear algebra & metrics ----------

def lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coef

def wlsq(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    if w is None:
        return lstsq(X, y)
    sw = np.sqrt(np.asarray(w)).reshape(-1, 1)
    return lstsq(X * sw, y * sw[:, 0])

def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([X, np.ones((X.shape[0], 1))])

def metrics_basic(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    resid = y - yhat
    mae = float(np.mean(np.abs(resid)))
    mape = float(np.mean(np.abs(resid) / np.maximum(np.abs(y), eps)) * 100.0)
    p50 = float(np.percentile(np.abs(resid) / np.maximum(np.abs(y), eps) * 100.0, 50))
    p90 = float(np.percentile(np.abs(resid) / np.maximum(np.abs(y), eps) * 100.0, 90))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)
    return {"MAE_us": mae, "MAPE_pct": mape, "P50": p50, "P90": p90, "R2": r2}

def loocv_metrics(predict_fn, fit_fn, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    preds = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        Xi, yi = X[mask], y[mask]
        try:
            state = fit_fn(Xi, yi)
            preds[i] = predict_fn(state, X[i:i+1])[0]
        except Exception:
            preds[i] = np.nan
    valid = ~np.isnan(preds)
    if valid.sum() == 0:
        return {"LOOCV_MAE_us": np.nan, "LOOCV_MAPE_pct": np.nan}
    yv, pv = y[valid], preds[valid]
    eps = 1e-12
    resid = yv - pv
    mae = float(np.mean(np.abs(resid)))
    mape = float(np.mean(np.abs(resid) / np.maximum(np.abs(yv), eps)) * 100.0)
    return {"LOOCV_MAE_us": mae, "LOOCV_MAPE_pct": mape}

# ---------- feature builder ----------

def build_X(features: Dict[str, np.ndarray], extra_feats: List[str]) -> np.ndarray:
    cols = []
    if "x" in features: cols.append(features["x"])
    for name in extra_feats:
        name = name.strip().upper()
        if name == "B" and "B" in features: cols.append(features["B"])
        elif name == "D" and "D" in features: cols.append(features["D"])
    if not cols: raise ValueError("No features to build design matrix.")
    return np.column_stack(cols)

# ---------- model families ----------

def fit_linear_family(X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]=None, robust: bool=False, huber_delta: float=1.0, max_iter: int=50, tol: float=1e-6):
    Xd = add_intercept(X)
    if not robust:
        coef = wlsq(Xd, y, weights)
        return {"coef": coef, "family": "wls" if weights is not None else "linear"}
    # Huber IRLS（修复点：调用 wlsq(Xd, y, None)）
    coef = wlsq(Xd, y, None)
    for _ in range(max_iter):
        yhat = Xd @ coef
        resid = y - yhat
        abs_r = np.abs(resid) + 1e-12
        w = np.where(abs_r <= huber_delta, 1.0, huber_delta / abs_r)
        coef_new = wlsq(Xd, y, w)
        if np.linalg.norm(coef_new - coef) <= tol * (np.linalg.norm(coef) + 1e-12):
            coef = coef_new; break
        coef = coef_new
    return {"coef": coef, "family": "robust", "huber_delta": huber_delta}

def predict_linear_family(state: Dict, X: np.ndarray) -> np.ndarray:
    Xd = add_intercept(X); return Xd @ state["coef"]

def fit_logy(X: np.ndarray, y: np.ndarray, eps_y: float=1e-9) -> Dict:
    y_safe = np.maximum(y, eps_y); Xd = add_intercept(X)
    beta = lstsq(Xd, np.log(y_safe)); return {"beta": beta, "family": "logy", "eps_y": eps_y}

def predict_logy(state: Dict, X: np.ndarray) -> np.ndarray:
    Xd = add_intercept(X); return np.exp(Xd @ state["beta"])

def fit_hinge(X: np.ndarray, y: np.ndarray, xcol: int=0, c_grid: Optional[List[float]]=None) -> Dict:
    x = X[:, xcol]
    if c_grid is None:
        qs = np.linspace(0.3, 0.9, 13)
        c_grid = np.quantile(x, qs).tolist()
        c_grid = sorted(list(set(map(float, c_grid))))
    best = None; Xbase = X.copy()
    for c in c_grid:
        h = np.maximum(0.0, x - c)
        Xd = add_intercept(np.column_stack([Xbase, h]))
        coef = lstsq(Xd, y); yhat = Xd @ coef; sse = float(np.sum((y - yhat)**2))
        if (best is None) or (sse < best["sse"]):
            best = {"coef": coef, "c": float(c), "sse": sse}
    best["family"] = "hinge"; best["xcol"] = xcol; return best

def predict_hinge(state: Dict, X: np.ndarray) -> np.ndarray:
    x = X[:, state.get("xcol", 0)]
    h = np.maximum(0.0, x - state["c"])
    Xd = add_intercept(np.column_stack([X, h])); return Xd @ state["coef"]

def fit_power_single(x: np.ndarray, y: np.ndarray, eps: float=1e-12) -> Dict:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = (x > 0) & (y > 0)
    if m.sum() < 2: raise ValueError("power: 需要至少两个 x>0,y>0 的样本")
    lx = np.log(x[m]); ly = np.log(y[m])
    Xd = add_intercept(lx.reshape(-1, 1)); beta = lstsq(Xd, ly)
    lnA, p = float(beta[1]), float(beta[0]); A = float(np.exp(lnA))
    return {"A": A, "p": p, "idx_mask": m, "family": "power"}

def predict_power(state: Dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float); yhat = np.full_like(x, np.nan, dtype=float)
    m = x > 0; yhat[m] = state["A"] * np.power(x[m], state["p"]); return yhat

# ---------- train & select ----------

def compute_all_and_select(X: np.ndarray, y: np.ndarray, features_used: List[str], plots: int, out_dir: str) -> Dict:
    res_all = []

    def fit_lin(Xt, yt): return fit_linear_family(Xt, yt, weights=None, robust=False)
    state = fit_lin(X, y); yhat = predict_linear_family(state, X)
    met = metrics_basic(y, yhat)
    lo = loocv_metrics(predict_linear_family, fit_lin, X, y)
    res_all.append({"name":"linear","state":state,"yhat":yhat,"metrics":met,"loocv":lo})

    def wls_weights(yy): return 1.0 / np.maximum(yy, 1e-12)**2
    def fit_wls(Xt, yt): return fit_linear_family(Xt, yt, weights=wls_weights(yt), robust=False)
    state = fit_linear_family(X, y, weights=wls_weights(y), robust=False)
    yhat = predict_linear_family(state, X); met = metrics_basic(y, yhat)
    lo = loocv_metrics(predict_linear_family, fit_wls, X, y)
    res_all.append({"name":"wls","state":state,"yhat":yhat,"metrics":met,"loocv":lo})

    def fit_rob(Xt, yt): return fit_linear_family(Xt, yt, weights=None, robust=True, huber_delta=1.0)
    state = fit_rob(X, y); yhat = predict_linear_family(state, X)
    met = metrics_basic(y, yhat); lo = loocv_metrics(predict_linear_family, fit_rob, X, y)
    res_all.append({"name":"robust","state":state,"yhat":yhat,"metrics":met,"loocv":lo})

    def fit_log(Xt, yt): return fit_logy(Xt, yt)
    state = fit_log(X, y); yhat = predict_logy(state, X)
    met = metrics_basic(y, yhat); lo = loocv_metrics(predict_logy, fit_log, X, y)
    res_all.append({"name":"logy","state":state,"yhat":yhat,"metrics":met,"loocv":lo})

    try:
        state = fit_hinge(X, y, xcol=0, c_grid=None)
        def fit_hg(Xt, yt): return fit_hinge(Xt, yt, xcol=0, c_grid=None)
        yhat = predict_hinge(state, X); met = metrics_basic(y, yhat)
        lo = loocv_metrics(predict_hinge, fit_hg, X, y)
        res_all.append({"name":"hinge","state":state,"yhat":yhat,"metrics":met,"loocv":lo})
    except Exception:
        pass

    if X.shape[1] >= 1:
        try:
            state = fit_power_single(X[:,0], y)
            def fit_p(Xt, yt): return fit_power_single(Xt[:,0], yt)
            yhat = predict_power(state, X[:,0]); met = metrics_basic(y, yhat)
            lo = loocv_metrics(lambda st, Xq: predict_power(st, Xq[:,0]), fit_p, X, y)
            res_all.append({"name":"power","state":state,"yhat":yhat,"metrics":met,"loocv":lo})
        except Exception:
            pass

    best = min(res_all, key=lambda r: (np.inf if np.isnan(r["loocv"]["LOOCV_MAPE_pct"]) else r["loocv"]["LOOCV_MAPE_pct"]))
    eqn = pretty_equation(best["name"], best["state"], features_used)

    if plots and X.shape[1] == 1:
        x = X[:,0]; xg = np.linspace(float(x.min()), float(x.max()), 200)
        if best["name"] in ("linear","wls","robust","logy","hinge"):
            Xg = xg.reshape(-1,1)
            if best["name"] == "logy": yg = predict_logy(best["state"], Xg)
            elif best["name"] == "hinge": yg = predict_hinge(best["state"], Xg)
            else: yg = predict_linear_family(best["state"], Xg)
        elif best["name"] == "power":
            yg = predict_power(best["state"], xg)
        else:
            yg = None
        plt.figure(); plt.scatter(x, y, label="measured")
        if yg is not None: plt.plot(xg, yg, label=f"{best['name']} fit")
        plt.xlabel("x = batch * dim"); plt.ylabel("Latency (us)"); plt.legend()
        plt.title("LayerNorm latency fit (advanced)"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "advanced_fit_curve.png"), dpi=160)

        plt.figure(); plt.scatter(y, best["yhat"])
        ymin, ymax = float(min(np.nanmin(y), np.nanmin(best["yhat"]))), float(max(np.nanmax(y), np.nanmax(best["yhat"])))
        plt.plot([ymin, ymax], [ymin, ymax])
        plt.xlabel("Measured (us)"); plt.ylabel("Predicted (us)")
        plt.title(f"Measured vs Predicted ({best['name']})"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "advanced_meas_vs_pred.png"), dpi=160)

    out = {"chosen_family": best["name"], "equation": eqn,
           "state": serialize_state(best["name"], best["state"], features_used),
           "metrics": best["metrics"], "loocv_metrics": best["loocv"]}
    return out, res_all

def pretty_equation(name: str, state: Dict, features_used: List[str]) -> str:
    feat_names = ["x"] + [f for f in features_used if f in ("B","D")]
    if name in ("linear","wls","robust"):
        coef = state["coef"].reshape(-1); terms = []
        for i, fn in enumerate(feat_names): terms.append(f"{coef[i]:.6g}*{fn}")
        terms.append(f"{coef[-1]:.6g}"); return "T_us = " + " + ".join(terms)
    if name == "logy":
        beta = state["beta"].reshape(-1); terms = []
        for i, fn in enumerate(feat_names): terms.append(f"{beta[i]:.6g}*{fn}")
        terms.append(f"{beta[-1]:.6g}"); return "T_us = exp(" + " + ".join(terms) + ")"
    if name == "hinge":
        coef = state["coef"].reshape(-1); k = len(feat_names); terms = []
        for i, fn in enumerate(feat_names): terms.append(f"{coef[i]:.6g}*{fn}")
        terms.append(f"{coef[k]:.6g}*max(0, x - {state['c']:.6g})"); terms.append(f"{coef[-1]:.6g}")
        return "T_us = " + " + ".join(terms)
    if name == "power": return f"T_us = {state['A']:.6g} * x^{state['p']:.6g}"
    return "unknown"

def serialize_state(name: str, state: Dict, features_used: List[str]) -> Dict:
    if name in ("linear","wls","robust"):
        return {"coef": state["coef"].tolist(), "intercept_included": True, "features": ["x"] + [f for f in features_used if f in ("B","D")]}
    if name == "logy":
        return {"beta": state["beta"].tolist(), "intercept_included": True, "features": ["x"] + [f for f in features_used if f in ("B","D")]}
    if name == "hinge":
        return {"coef": state["coef"].tolist(), "c": state["c"], "xcol": state["xcol"], "features": ["x"] + [f for f in features_used if f in ("B","D")]}
    if name == "power":
        return {"A": state["A"], "p": state["p"], "features": ["x"]}
    return state

def main():
    parser = argparse.ArgumentParser(description="Advanced LayerNorm latency model fitting (fixed)")
    parser.add_argument("--glob", default="./results/*.csv")
    parser.add_argument("--target-file", default="layernorm")
    parser.add_argument("--target-lines", default="")
    parser.add_argument("--time-col", default="running_time(us)")
    parser.add_argument("--out-dir", default="./layernorm_fit_out_adv")
    parser.add_argument("--model", default="auto", choices=["auto","linear","wls","logy","hinge","power","robust"])
    parser.add_argument("--extra-feats", default="")  # e.g., "B,D"
    parser.add_argument("--plots", type=int, default=1)
    args = parser.parse_args()

    target_lines = _norm_lines_to_set(args.target_lines)

    csv_list = sorted(glob.glob(args.glob, recursive=True))
    if not csv_list:
        raise FileNotFoundError(f"未找到任何 CSV：{args.glob}")

    rows = []
    for p in csv_list:
        try:
            rows.append(summarize_one_csv(p, args.target_file, target_lines, args.time_col))
        except Exception as e:
            print(f"[WARN] 跳过 {p}: {e}")

    df = pd.DataFrame(rows)
    df = df[(df["x"] > 0) & df["meas_us"].notna()]
    if df.empty:
        raise RuntimeError("no valid data (无法解析出 batch*dim 或无有效时间)")
    df = df.sort_values(by="x").reset_index(drop=True)

    features = {"x": df["x"].to_numpy(dtype=float),
                "B": df["batch"].to_numpy(dtype=float),
                "D": df["dim"].to_numpy(dtype=float)}
    extra_feats = [s.strip().upper() for s in args.extra_feats.split(",") if s.strip()]
    X = build_X(features, extra_feats)  # 第一列是 x
    y = df["meas_us"].to_numpy(dtype=float)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.model == "auto":
        best, all_res = compute_all_and_select(X, y, extra_feats, args.plots, args.out_dir)
        chosen = best
        compare = []
        for r in all_res:
            d = {"family": r["name"], **{f"train_{k}": v for k, v in r["metrics"].items()}, **r["loocv"]}
            compare.append(d)
        pd.DataFrame(compare).to_csv(os.path.join(args.out_dir, "candidates_metrics.csv"), index=False)
        df["pred_us"] = next(r["yhat"] for r in all_res if r["name"] == chosen["chosen_family"])
    else:
        if args.model == "linear":
            state = fit_linear_family(X, y, weights=None, robust=False); yhat = predict_linear_family(state, X)
        elif args.model == "wls":
            def wls_weights(yy): return 1.0 / np.maximum(yy, 1e-12)**2
            state = fit_linear_family(X, y, weights=wls_weights(y), robust=False); yhat = predict_linear_family(state, X)
        elif args.model == "robust":
            state = fit_linear_family(X, y, weights=None, robust=True, huber_delta=1.0); yhat = predict_linear_family(state, X)
        elif args.model == "logy":
            state = fit_logy(X, y); yhat = predict_logy(state, X)
        elif args.model == "hinge":
            state = fit_hinge(X, y, xcol=0, c_grid=None); yhat = predict_hinge(state, X)
        elif args.model == "power":
            state = fit_power_single(X[:,0], y); yhat = predict_power(state, X[:,0])
        else:
            raise ValueError("unknown model")
        met = metrics_basic(y, yhat)
        if args.model in ("linear","wls","robust"):
            if args.model == "wls":
                def fit_wls(Xt, yt): return fit_linear_family(Xt, yt, weights=1.0/np.maximum(yt,1e-12)**2, robust=False)
                lo = loocv_metrics(predict_linear_family, fit_wls, X, y)
            elif args.model == "robust":
                def fit_rob(Xt, yt): return fit_linear_family(Xt, yt, weights=None, robust=True, huber_delta=1.0)
                lo = loocv_metrics(predict_linear_family, fit_rob, X, y)
            else:
                def fit_lin(Xt, yt): return fit_linear_family(Xt, yt, weights=None, robust=False)
                lo = loocv_metrics(predict_linear_family, fit_lin, X, y)
        elif args.model == "logy":
            def fit_log(Xt, yt): return fit_logy(Xt, yt)
            lo = loocv_metrics(predict_logy, fit_log, X, y)
        elif args.model == "hinge":
            def fit_hg(Xt, yt): return fit_hinge(Xt, yt, xcol=0, c_grid=None)
            lo = loocv_metrics(predict_hinge, fit_hg, X, y)
        elif args.model == "power":
            def fit_p(Xt, yt): return fit_power_single(Xt[:,0], yt)
            lo = loocv_metrics(lambda st, Xq: predict_power(st, Xq[:,0]), fit_p, X, y)
        else:
            lo = {"LOOCV_MAE_us": np.nan, "LOOCV_MAPE_pct": np.nan}
        chosen = {"chosen_family": args.model, "equation": pretty_equation(args.model, state, extra_feats),
                  "state": serialize_state(args.model, state, extra_feats), "metrics": met, "loocv_metrics": lo}
        df["pred_us"] = yhat

    df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump({"model": chosen["equation"], "family": chosen["chosen_family"], "state": chosen["state"],
                   "metrics": chosen["metrics"], "loocv_metrics": chosen["loocv_metrics"], "n_samples": int(len(df))},
                  f, ensure_ascii=False, indent=2)

    print("[OK] chosen family:", chosen["chosen_family"])
    print("Equation:", chosen["equation"])
    print("Train metrics:", chosen["metrics"])
    print("LOOCV metrics:", chosen["loocv_metrics"])
    print(f"Outputs -> {args.out_dir}/summary.csv, {args.out_dir}/model.json")
    if args.plots and X.shape[1] == 1:
        print(f"Plots   -> {args.out_dir}/advanced_fit_curve.png, {args.out_dir}/advanced_meas_vs_pred.png")

if __name__ == "__main__":
    main()
