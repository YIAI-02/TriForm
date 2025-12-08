import pandas as pd
import numpy as np

# ====== 你只需要改这里的文件名 ======
A_NAME = "attacc"
A_OPS  = "128x2048_ops_trace.csv"
A_COMM = "128x2048_comms_trace.csv"

B_NAME = "heft"
B_OPS  = "heft_kv_first_128x2048_ops_trace.csv"
B_COMM = "heft_kv_first_128x2048_comms_trace.csv"
# ===================================

def load_ops(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["start", "end", "duration"]:
        df[c] = df[c].astype(float)
    return df

def load_comms(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["start", "end", "duration"]:
        df[c] = df[c].astype(float)
    df["bytes"] = df["bytes"].astype(int)
    return df

def makespan(df: pd.DataFrame, phase: str) -> float:
    sub = df[df["phase"] == phase]
    return float(sub["end"].max()) if len(sub) else float("nan")

def add_occ_idx(ops: pd.DataFrame, phase: str) -> pd.DataFrame:
    """
    同一个 node_id 在 decode 会出现多次（你的 trace 是采样的）。
    用 node_id 内部按 start 排序，然后 cumcount 作为 occ_idx，对齐两个算法的“第几次出现”。
    """
    sub = ops[ops["phase"] == phase].copy()
    sub = sub.sort_values(["node_id", "start", "end"]).reset_index(drop=True)
    sub["occ_idx"] = sub.groupby("node_id").cumcount()
    return sub

def comm_summary(comms: pd.DataFrame, phase: str) -> pd.DataFrame:
    sub = comms[comms["phase"] == phase].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    out = (sub.groupby("tag")
              .agg(count=("tag", "size"),
                   bytes=("bytes", "sum"),
                   duration=("duration", "sum"))
              .sort_values("duration", ascending=False))
    return out

def pim_ratio_by_node(ops_phase: pd.DataFrame) -> pd.Series:
    # ops_phase 必须是 add_occ_idx 后、且已经筛好 phase
    g = (ops_phase.groupby("node_id")["device_type"]
               .value_counts(normalize=True)
               .unstack(fill_value=0))
    return g["pim"] if "pim" in g.columns else pd.Series(0.0, index=g.index)

def pim_ratio_by_op(ops_phase: pd.DataFrame) -> pd.Series:
    g = (ops_phase.groupby("op")["device_type"]
               .value_counts(normalize=True)
               .unstack(fill_value=0))
    return g["pim"] if "pim" in g.columns else pd.Series(0.0, index=g.index)

def compare_ops(opsA: pd.DataFrame, opsB: pd.DataFrame, phase="decode", topk=20):
    A = add_occ_idx(opsA, phase)
    B = add_occ_idx(opsB, phase)

    # 对齐每个 node 的第 occ_idx 次出现
    m = A.merge(
        B[["node_id", "occ_idx", "device", "device_type", "start", "end", "duration"]],
        on=["node_id", "occ_idx"],
        suffixes=("_A", "_B"),
        how="inner"
    )

    # 1) 最早分叉点（按时间）
    diff = m[m["device_A"] != m["device_B"]].copy()
    if len(diff):
        diff["tmin"] = np.minimum(diff["start_A"], diff["start_B"])
        first_diff = diff.sort_values("tmin").head(10)[
            ["node_id", "op", "occ_idx",
             "device_A", "device_B",
             "start_A", "start_B",
             "duration_A", "duration_B"]
        ]
    else:
        first_diff = pd.DataFrame()

    # 2) node 级别：PIM 占比差异最大
    pimA = pim_ratio_by_node(A)
    pimB = pim_ratio_by_node(B)
    node_summary = pd.DataFrame({
        "op": A.groupby("node_id")["op"].first(),
        f"{A_NAME}_pim_ratio": pimA,
        f"{B_NAME}_pim_ratio": pimB,
    })
    node_summary["abs_diff"] = (node_summary[f"{A_NAME}_pim_ratio"] - node_summary[f"{B_NAME}_pim_ratio"]).abs()
    node_summary = node_summary.sort_values("abs_diff", ascending=False).head(topk)

    # 3) op 类型级别：PIM 占比差异最大
    op_pimA = pim_ratio_by_op(A)
    op_pimB = pim_ratio_by_op(B)
    op_summary = pd.DataFrame({
        f"{A_NAME}_pim_ratio": op_pimA,
        f"{B_NAME}_pim_ratio": op_pimB,
    })
    op_summary["abs_diff"] = (op_summary[f"{A_NAME}_pim_ratio"] - op_summary[f"{B_NAME}_pim_ratio"]).abs()
    op_summary = op_summary.sort_values("abs_diff", ascending=False).head(topk)

    # 4) “对性能影响最大”：node 总 duration 差值（B-A）
    durA = A.groupby(["node_id", "op"])["duration"].sum()
    durB = B.groupby(["node_id", "op"])["duration"].sum()
    impact = (durB - durA).sort_values(ascending=False).head(topk).reset_index()
    impact.columns = ["node_id", "op", "delta_total_duration(B-A)"]

    return first_diff, node_summary, op_summary, impact

def main():
    opsA = load_ops(A_OPS)
    opsB = load_ops(B_OPS)
    commA = load_comms(A_COMM)
    commB = load_comms(B_COMM)

    for phase in ["prefill", "decode"]:
        print(f"\n=== Makespan ({phase}) ===")
        print(f"{A_NAME}: {makespan(opsA, phase):.6f} s")
        print(f"{B_NAME}: {makespan(opsB, phase):.6f} s")

    print("\n=== Comms summary (decode) ===")
    print(f"\n[{A_NAME}]")
    print(comm_summary(commA, "decode"))
    print(f"\n[{B_NAME}]")
    print(comm_summary(commB, "decode"))

    # 一个小技巧：kv_load 条数基本等于 “K_read+V_read 在 NPU 的次数”
    kvB = commB[(commB["phase"] == "decode") & (commB["tag"] == "kv_load")]
    print(f"\n[{B_NAME}] kv_load count={len(kvB)}, sum_dur={kvB['duration'].sum():.6f}s, sum_bytes={kvB['bytes'].sum():,}")

    first_diff, top_nodes, top_ops, impact = compare_ops(opsA, opsB, phase="decode", topk=20)

    print("\n=== Earliest device divergence (decode, first 10 events) ===")
    print(first_diff if len(first_diff) else "No diff found.")

    print("\n=== Nodes with largest device assignment diff (by PIM ratio) ===")
    print(top_nodes)

    print("\n=== Op-types with largest device assignment diff (by PIM ratio) ===")
    print(top_ops)

    print("\n=== Biggest runtime-impact nodes (delta total duration in sampled trace) ===")
    print(impact)

if __name__ == "__main__":
    main()
