#!/usr/bin/env python3
import os, re, sys, csv, json
base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "profile"))
rows = []
for root, dirs, files in os.walk(base):
    for f in files:
        if f == "run.log":
            path = os.path.join(root, f)
            op = root.split(os.sep)[-2]
            shape = root.split(os.sep)[-1]
            avg = p50 = None
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    m = re.search(r"\[RESULT\].*avg_ms=([0-9.]+).*p50_ms=([0-9.]+)", line)
                    if m:
                        avg = float(m.group(1)); p50 = float(m.group(2))
            rows.append({"op": op, "shape": shape, "avg_ms": avg, "p50_ms": p50, "log": path})
rows.sort(key=lambda r: (r["op"], r["shape"]))
out_csv = os.path.join(base, "summary.csv")
with open(out_csv, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=["op","shape","avg_ms","p50_ms","log"])
    w.writeheader()
    for r in rows: w.writerow(r)
print(f"[OK] Wrote {out_csv} with {len(rows)} rows")
