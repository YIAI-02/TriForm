from __future__ import annotations
import json, argparse, sys
from typing import Dict
from config import (
    NPU_COUNT, PIM_COUNT, LOG_LEVEL, OBJECTIVE, GLOBAL_BUFFER_BYTES,
    FORMAT_SIZE_MULTIPLIER
)
from hardware import HardwareManager
from buffer_manager import BufferManager
from cost_model import CostModel
from scheduler import HeftScheduler, Assignment
from graph_io import load_graph_from_json, graph_to_json
from model_parser import parse_shape_json, build_model_graph, ParserConfig
from utils import setup_logger

def _compute_weight_usage_by_format(assignments: Dict[str, Assignment], nodes_by_id: Dict[str, any]) -> Dict[str, Dict[str,int]]:
    """统计：每个 weight 在本轮中被各 '目标格式' 使用的次数。
       NPU -> NPU_OPT, PIM -> PIM_OPT, HYBRID -> 两者都 +1。
    """
    stats: Dict[str, Dict[str,int]] = {}
    for nid, a in assignments.items():
        node = nodes_by_id[nid]
        if not node.weight_id:
            continue
        m = stats.setdefault(node.weight_id, {"NPU_OPT":0, "PIM_OPT":0})
        if a.output_device == "NPU":
            m["NPU_OPT"] += 1
        elif a.output_device == "PIM":
            m["PIM_OPT"] += 1
        elif a.output_device == "HYBRID":
            m["NPU_OPT"] += 1
            m["PIM_OPT"] += 1
    return stats

def _weights_size_from_graph(g) -> Dict[str, int]:
    return g.weights()

def _plan_total_bytes(plan: Dict[str, list], weights_size: Dict[str,int]) -> int:
    total = 0
    for w, fmts in plan.items():
        sz = weights_size.get(w, 0)
        for f in fmts:
            total += int(sz * FORMAT_SIZE_MULTIPLIER[f])
    return total

def run_static_graph(g, init_plan_path: str = "") -> Dict:
    cm = CostModel()
    hw = HardwareManager(NPU_COUNT, PIM_COUNT)
    buf = BufferManager()

    # optional: preload plan for THIS round
    weights_size = _weights_size_from_graph(g)
    if init_plan_path:
        with open(init_plan_path, "r") as f:
            init_obj = json.load(f)
        # 支持两种键名：weight_plan 或 weight_formats
        plan = init_obj.get("weight_plan") or init_obj.get("weight_formats") or {}
        est_bytes = _plan_total_bytes(plan, weights_size)
        if est_bytes > GLOBAL_BUFFER_BYTES:
            print(f"[WARN] init plan total bytes={est_bytes} > capacity={GLOBAL_BUFFER_BYTES}")
        buf.preload_plan(weights_size, plan)

    sched = HeftScheduler(cm, hw, buf)
    assigns = sched.schedule(g)

    # 本轮统计使用信息（按目标格式计数）
    usage_by_format = _compute_weight_usage_by_format(assigns, g.nodes)
    # 规划下一轮格式
    plan_next, plan_bytes = buf.plan_weight_formats(usage_by_format, weights_size, objective=OBJECTIVE)

    makespan = max(a.finish_time for a in assigns.values()) if assigns else 0.0
    return {
        "objective": OBJECTIVE,
        "makespan": makespan,
        "buffer_usage_bytes_this_round": buf.bytes_used(),
        "recommended_plan_bytes": plan_bytes,
        "recommended_weight_plan": {w: list(fmts) for w, fmts in plan_next.items()},
        "assignments": {k: a.__dict__ for k,a in assigns.items()},
    }

def main():
    setup_logger(LOG_LEVEL)
    ap = argparse.ArgumentParser(description="HEFT + AttAcc Interconnects (Static) with Greedy/DP/Beam Format Planners")
    ap.add_argument("--graph", type=str, help="JSON path (prebuilt DAG)")
    ap.add_argument("--model", type=str, choices=["llama","opt","palm"], help="Model type for parsing")
    ap.add_argument("--shape", type=str, help="Shape.json path for the model")
    ap.add_argument("--batch", type=int, default=1, help="Batch size")
    ap.add_argument("--seq", type=int, default=128, help="Sequence length")
    ap.add_argument("--init-plan", type=str, default="", help="Preload GB with a prior plan (JSON)")
    ap.add_argument("--emit-plan", type=str, default="", help="Emit the NEXT-round plan to this JSON")
    ap.add_argument("--out", type=str, default="", help="Output JSON path (default: stdout)")
    ap.add_argument("--mode", type=str, choices=["model_parser","analysis"], default="analysis", help="model_parser: only build & dump graph then exit; analysis: full scheduling & planning (default)")
    ap.add_argument("--graph-dir", type=str, default="", help="Directory to dump parsed graph JSON (file name: graph.json). Used in model_parser mode or when also saving during analysis.")
    args = ap.parse_args()

    if args.model and args.shape:
        model_type, shp = parse_shape_json(args.shape)
        if args.model:
            model_type = args.model
        g = build_model_graph(model_type, shp, ParserConfig(batch=args.batch, seq_len=args.seq))
        # Save graph if requested
        if args.graph_dir:
            import os, json as _json
            from pathlib import Path
            gd = Path(args.graph_dir)
            gd.mkdir(parents=True, exist_ok=True)
            (gd / 'graph.json').write_text(_json.dumps(graph_to_json(g), indent=2), encoding='utf-8')
            print(f"Saved graph to {gd / args.model + 'graph.json'}")
        if args.mode == "model_parser":
            # Only parse & dump
            if args.out:
                with open(args.out, 'w') as f:
                    f.write(json.dumps({"status":"graph_dumped","graph_path": str((Path(args.graph_dir)/'graph.json') if args.graph_dir else '')}, indent=2))
            else:
                print(json.dumps({"status":"graph_dumped"}, indent=2))
            return
        # analysis mode continues
        res = run_static_graph(g, init_plan_path=args.init_plan)
    elif args.graph:
        with open(args.graph, "r") as f:
            obj = json.load(f)
        g = load_graph_from_json(obj)
        if args.mode == "model_parser":
            # If user provides an already-built graph with model_parser mode, just optionally re-dump and exit
            if args.graph_dir:
                import json as _json
                from pathlib import Path
                gd = Path(args.graph_dir)
                gd.mkdir(parents=True, exist_ok=True)
                (gd / 'graph.json').write_text(_json.dumps(graph_to_json(g), indent=2), encoding='utf-8')
                print(f"Saved graph to {gd / 'graph.json'}")
            print(json.dumps({"status":"graph_dumped_from_input"}, indent=2))
            return
        res = run_static_graph(g, init_plan_path=args.init_plan)
    else:
        ap.print_help()
        sys.exit(1)

    # emit plan (for next round)
    if args.emit_plan:
        plan = {
            "weight_plan": res["recommended_weight_plan"],
            "total_bytes": res["recommended_plan_bytes"],
            "capacity_bytes": GLOBAL_BUFFER_BYTES
        }
        with open(args.emit_plan, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"Saved NEXT-round plan to: {args.emit_plan}")

    js = json.dumps(res, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(js)
        print(f"Saved result to: {args.out}")
    else:
        print(js)

if __name__ == "__main__":
    main()
