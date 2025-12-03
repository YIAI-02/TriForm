import sys
import json
import matplotlib.pyplot as plt

def load_trace(path: str):
    """
    读取 JSON:
    - 如果顶层是 {"pim_trace": [...]} 就直接取 pim_trace
    - 如果是 best_summary_xxx.json, 把 prefill_schedule + decode_steps 展平
    - 如果本身就是 list, 直接返回
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) 直接携带 pim_trace 的格式
    if isinstance(data, dict) and "pim_trace" in data:
        return data["pim_trace"]

    trace = []

    # 2) best_summary_xxx.json 这种结构
    if isinstance(data, dict) and ("prefill_schedule" in data or "decode_steps" in data):
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
                # 把 seq_len 塞进每个 event 里，方便按长度画
                if "seq_len" not in e and seq_len is not None:
                    e["seq_len"] = int(seq_len)
                trace.append(e)

    # 如果上面解析到了 trace，就用它
    if trace:
        return trace

    # 3) 顶层本身就是 list
    if isinstance(data, list):
        return data

    # 其他情况直接返回原始数据，让后续逻辑报错提示
    return data


def _filter_and_sort(trace, x_axis="seq_len"):
    """
    把 trace 里不符合要求的条目过滤掉，并按 x 排序。
    """
    valid = []
    for e in trace:
        if not isinstance(e, dict):
            continue
        if "device" not in e:
            continue
        if x_axis == "seq_len" and "seq_len" not in e:
            continue
        if x_axis == "time" and "finish" not in e:
            continue
        valid.append(e)

    key = (lambda ev: ev.get("seq_len", 0)) if x_axis == "seq_len" else (lambda ev: ev.get("finish", 0.0))
    valid.sort(key=key)
    return valid


def plot_pim_memory(trace, x_axis="seq_len"):
    """
    画三条曲线：
    - KV 占用
    - weight 占用
    - activation 占用
    再加 total 占用一条总线
    x_axis: "time" 或 "seq_len"
    """
    if not isinstance(trace, list):
        print(f"Error: loaded trace is {type(trace)}, expect list of dict")
        return

    # 只保留 PIM 设备上的事件
    trace = [e for e in trace if isinstance(e, dict) and str(e.get("device", "")).upper().startswith("PIM")]
    if not trace:
        print("No PIM events found in trace.")
        return

    devs = sorted({e["device"] for e in trace if "device" in e})
    valid_trace = _filter_and_sort(trace, x_axis=x_axis)

    for metric in ["kv_used_bytes", "weight_used_bytes", "act_used_bytes", "total_used_bytes"]:
        plt.figure()
        for dev in devs:
            evs = [e for e in valid_trace if e.get("device") == dev]
            if not evs:
                continue

            if x_axis == "time":
                xs = [e.get("finish", 0.0) for e in evs]
            else:
                xs = [e.get("seq_len", 0) for e in evs]

            ys = [e.get(metric, 0) / (1024**3) for e in evs]  # 转成 GiB
            # 用阶梯线更符合“使用量变化”的直觉
            plt.step(xs, ys, where="post", label=dev)

        plt.xlabel("time (s)" if x_axis == "time" else "sequence length")
        plt.ylabel(metric.replace("_bytes", " (GiB)"))
        plt.title(f"PIM memory usage - {metric} ({x_axis})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()


def plot_event_timeline(trace, x_axis="seq_len"):
    """
    额外画一张 event timeline，把 KV 读写等事件打成点，方便看是哪一层/哪个 op 触发的。
    这里假设 trace 里有字段:
      - event: 比如 "KV_BLOCK_WRITE" / "KV_BLOCK_READ"
      - node_id: 对应算子名字 (L0_K, L1_V_read ...)
      - block_key / weight_id: 如果你在 scheduler 里已经加到 extra 里，也会被展示
    如果这些字段缺失，代码会自动跳过相关信息。
    """
    if not isinstance(trace, list):
        return
    trace = [e for e in trace if isinstance(e, dict) and str(e.get("device", "")).upper().startswith("PIM")]
    if not trace:
        return

    valid_trace = _filter_and_sort(trace, x_axis=x_axis)

    # 只挑出有 event 字段的
    evts = [e for e in valid_trace if "event" in e]
    if not evts:
        return

    plt.figure()
    # 不同类型事件映射到不同的 y 位置，方便区分
    event_types = sorted({str(e.get("event")) for e in evts})
    y_map = {et: i for i, et in enumerate(event_types)}

    for e in evts:
        et = str(e.get("event"))
        y = y_map.get(et, 0)
        if x_axis == "time":
            x = e.get("finish", 0.0)
        else:
            x = e.get("seq_len", 0)

        plt.scatter([x], [y], s=10)

        # 给点加上简单注释：node_id / block_key / weight_id
        label_parts = []
        nid = e.get("node_id")
        if nid:
            label_parts.append(str(nid))
        # 如果你在 scheduler._record_pim_trace 里把 block_key / weight_id 塞进 extra 里，
        # 并在写入 rec 时 rec.update(extra)，这里就能显示“替换哪个块/哪个 weight”了。
        if "block_key" in e:
            label_parts.append(f"blk={e['block_key']}")
        if "weight_id" in e:
            label_parts.append(f"wid={e['weight_id']}")
        if label_parts:
            txt = "\n".join(label_parts)
            plt.annotate(
                txt,
                xy=(x, y),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=6,
            )

    plt.yticks(list(y_map.values()), list(y_map.keys()))
    plt.xlabel("time (s)" if x_axis == "time" else "sequence length")
    plt.title(f"PIM events timeline ({x_axis})")
    plt.grid(True, axis="x")
    plt.tight_layout()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_pim_trace.py <path_to_json> [time|seq_len]")
        sys.exit(1)

    path = sys.argv[1]
    x_axis = "seq_len"
    if len(sys.argv) >= 3 and sys.argv[2] in ("time", "seq_len"):
        x_axis = sys.argv[2]

    trace = load_trace(path)
    if not isinstance(trace, list):
        print(f"Loaded object is {type(trace)}, cannot plot.")
        return

    plot_pim_memory(trace, x_axis=x_axis)
    plot_event_timeline(trace, x_axis=x_axis)
    plt.show()


if __name__ == "__main__":
    main()
