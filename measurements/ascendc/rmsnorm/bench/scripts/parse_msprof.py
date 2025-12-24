#!/usr/bin/env python3
import sys, os, csv, glob

def find_csvs(dirpath):
    # Common names: op_summary_*.csv, aicore_op_summary_*.csv, op_statistic_*.csv
    patterns = ["*.csv"]
    out = []
    for pat in patterns:
        out.extend(glob.glob(os.path.join(dirpath, pat)))
    return out

def parse_dir(d):
    recs = []
    # Prefer op-level summary CSV if present
    csvs = find_csvs(d)
    for c in csvs:
        try:
            with open(c, "r", newline="") as f:
                # Fallback to default dialect if sniff fails
                try:
                    sniffer = csv.Sniffer()
                    sample = f.read(4096); f.seek(0)
                    dialect = sniffer.sniff(sample)
                except Exception:
                    dialect = csv.excel
                reader = csv.reader(f, dialect)
                header = next(reader, [])
                cols = {h:i for i,h in enumerate(header)}
                # Try multiple header variants across msprof versions
                name_keys = ["op_name","OP Name","Op Name","KernelName","Kernel Name"]
                time_keys = ["avg_time(us)","Average Time(us)","Avg Time(us)","Avg Time(us.)","Mean(us)","Average Op Time(us)"]
                shape_keys = ["shape","Input Shape","input_shape","Tensor Shapes"]
                dtype_keys = ["data_type","Dtype","DataType","Data Type"]
                def find(klist):
                    for k in klist:
                        if k in cols: return cols[k]
                    return None
                i_name = find(name_keys)
                i_time = find(time_keys)
                i_shape = find(shape_keys)
                i_dtype = find(dtype_keys)
                for row in reader:
                    if i_name is None or i_time is None:
                        continue
                    if len(row) <= max(i_name, i_time):
                        continue
                    name = row[i_name]
                    # Heuristic: only keep rows containing rmsnorm
                    if "rmsnorm" in name.lower() or "rms_norm" in name.lower():
                        recs.append({
                            "msprof_csv": os.path.basename(c),
                            "op_name": name,
                            "avg_time_us": float(row[i_time]) if row[i_time] else None,
                            "shape": row[i_shape] if i_shape is not None and i_shape < len(row) else "",
                            "dtype": row[i_dtype] if i_dtype is not None and i_dtype < len(row) else "",
                        })
        except Exception:
            pass
    return recs

def main():
    if len(sys.argv) < 2:
        print("Usage: parse_msprof.py <profile_root>", file=sys.stderr)
        sys.exit(2)
    root = sys.argv[1]
    rows = []
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d): continue
        recs = parse_dir(d)
        if recs:
            r = recs[0]
            rows.append([sub, r.get("avg_time_us",""), r.get("op_name",""), r.get("msprof_csv","")])
        else:
            rows.append([sub, "", "", ""])
    w = csv.writer(sys.stdout)
    w.writerow(["shape","avg_time_us","op_name","from_csv"])
    w.writerows(rows)

if __name__ == "__main__":
    main()
