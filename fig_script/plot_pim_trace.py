import sys
import json
import matplotlib.pyplot as plt

def load_trace(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容：pim_trace 可能直接在顶层 times 里，也可能在 payload 里
    if isinstance(data, dict) and "pim_trace" in data:
        return data["pim_trace"]
    
    #best_summary 
    if isinstance(data,dict) and ('prefill_schedule' in data or 'decode_steps' in data):
        trace = []
        prefill = data.get("prefill_schedule") or []
        for ev in prefill:
            if isinstance(ev, dict):
                trace.append(ev)
        
        decode = data.get("decode_steps") or []
        for step in decode:
            if not isinstance(step, dict):
                continue
            seq_len = step.get("seq_len", 0)
            schedule = step.get("schedule") or []
            if not isinstance(schedule, list):
                continue

            for ev in schedule:
                if not isinstance(ev, dict):
                    continue
                e = dict(ev)
                if "seq_len" not in e and seq_len > 0:
                    e["seq_len"] = seq_len
                trace.append(e)
    if trace:            
        return trace
    
    if isinstance(data, list):
        return data
    
    return data

def plot_pim_memory(trace, x_axis="time"):
    """
    x_axis: "time" 或 "seq_len"
    """
    if not isinstance(trace, list):
        print(f"Error: The loaded data is not a list (got {type(trace)}).")
        print("Please ensure the JSON file contains a 'pim_trace' field or is a list of trace events.")
        return

    devs = sorted({e["device"] for e in trace if "device" in e})
    
    # Filter valid events
    valid_trace = []
    for e in trace:
        if x_axis == "seq_len" and "seq_len" not in e:
            continue
        if x_axis == "time" and "finish" not in e:
            continue
        valid_trace.append(e)

    for metric in ["kv_used_bytes", "weight_used_bytes", "act_used_bytes"]:
        plt.figure()
        for dev in devs:
            evs = [e for e in valid_trace if e.get("device") == dev]
            if not evs:
                continue
            
            # Sort events
            if x_axis == "time":
                evs.sort(key=lambda x: x.get("finish", 0))
                xs = [e["finish"] for e in evs]
                ys = [e.get(metric, 0) / (1024**3) for e in evs]  # 转成 GiB
                plt.step(xs, ys, where="post", label=dev)
            else:
                # x_axis == "seq_len"
                # Aggregate by seq_len to find peak memory usage per sequence length
                # This avoids vertical lines and provides a continuous trend
                data_by_len = {}
                for e in evs:
                    sl = e.get("seq_len", 0)
                    val = e.get(metric, 0)
                    if sl not in data_by_len:
                        data_by_len[sl] = val
                    else:
                        data_by_len[sl] = max(data_by_len[sl], val)
                
                sorted_sl = sorted(data_by_len.keys())
                xs = sorted_sl
                ys = [data_by_len[sl] / (1024**3) for sl in sorted_sl] # 转成 GiB
                
                # Use plot instead of step for continuous line between discrete seq_len points
                plt.plot(xs, ys, label=dev, marker='.')

        plt.xlabel("time (s)" if x_axis == "time" else "sequence length")
        plt.ylabel(metric.replace("_bytes", " (GiB)"))
        plt.title(f"Memory Usage - {metric}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_pim_trace.py <path_to_json>")
        sys.exit(1)

    trace = load_trace(sys.argv[1])
    plot_pim_memory(trace, x_axis="seq_len")
