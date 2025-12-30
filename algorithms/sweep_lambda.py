#!/usr/bin/env python3
# sweep_lambda.py
import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PARAM = "SCHED_JOINT_LK_CONSIST_LAMBDA"

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def replace_param(text: str, new_value: float) -> tuple[str, float]:
    """
    Replace a line like:
      SCHED_JOINT_LK_CONSIST_LAMBDA: float = 10
    or:
      SCHED_JOINT_LK_CONSIST_LAMBDA = 10
    Return (new_text, old_value).
    """
    # capture old numeric value
    pat = re.compile(
        rf"^({PARAM}\s*(?::\s*float)?\s*=\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$",
        re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"在 config.py 里没找到参数行：{PARAM}")

    old_val = float(m.group(2))
    # keep prefix, replace number
    new_text = pat.sub(lambda mm: f"{mm.group(1)}{new_value}", text, count=1)
    return new_text, old_val

def run_one(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n")
        f.write(f"# time: {dt.datetime.now().isoformat(timespec='seconds')}\n\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return p.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.py", help="config.py 路径")
    ap.add_argument("--script", default="./command_single.sh", help="要执行的 bash 脚本")
    ap.add_argument(
        "--values",
        nargs="+",
        type=float,
        required=True,
        help="要扫描的 lambda 值列表，如：--values 0 1 2 5 10",
    )
    ap.add_argument(
        "--outdir",
        default="./output/sweep_lambda",
        help="日志输出目录",
    )
    args, extra = ap.parse_known_args()

    config_path = Path(args.config)
    script_path = Path(args.script)
    outdir = Path(args.outdir)

    if not config_path.exists():
        print(f"找不到 {config_path}", file=sys.stderr)
        sys.exit(2)
    if not script_path.exists():
        print(f"找不到 {script_path}", file=sys.stderr)
        sys.exit(2)

    orig_text = read_text(config_path)
    backup_path = config_path.with_suffix(config_path.suffix + ".bak")
    shutil.copy2(config_path, backup_path)

    try:
        # 读出原始值（用于打印/校验）
        _, old_val = replace_param(orig_text, float(args.values[0]))
        print(f"[info] {PARAM} 当前值: {old_val}")
        print(f"[info] 将扫描: {args.values}")
        print(f"[info] 备份已写入: {backup_path}")

        for v in args.values:
            print(f"\n=== Running {PARAM}={v} ===")

            new_text, _ = replace_param(read_text(config_path), v)
            write_text(config_path, new_text)

            # 每个取值一个独立日志文件
            tag = f"{PARAM}={v}"
            safe_tag = tag.replace("=", "_").replace(".", "p")
            log_path = outdir / safe_tag / "run.log"

            # 统一用 bash 执行，避免权限问题
            cmd = ["bash", str(script_path)]
            if extra:
                cmd += extra

            rc = run_one(cmd, log_path)
            print(f"[done] rc={rc} log={log_path}")
            if rc != 0:
                print("[warn] 命令返回非 0，中断后续扫描。")
                sys.exit(rc)

    finally:
        # 恢复原 config.py
        write_text(config_path, orig_text)
        print(f"\n[info] 已恢复 {config_path} 为原始内容")

if __name__ == "__main__":
    main()
