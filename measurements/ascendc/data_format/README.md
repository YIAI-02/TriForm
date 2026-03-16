# Ascend format-conversion microbench (sim / msprof)

这套工程用于在 **Ascend 910/910B simulator + `msprof op simulator`** 流程下，测几条和 `0315_algo` 新格式模型直接对应的转换边：

- `ND -> NZ`（A-path / B-path）
- `NZ -> ZZ`（A-path）
- `NZ -> ZN`（B-path）
- `ND -> ZZ`（链式：`ND -> NZ -> ZZ`）
- `ND -> ZN`（链式：`ND -> NZ -> ZN`）

## 目录

- `format_conv_bench.cpp` / `format_conv_bench_kernel.h`：Ascend C kernel
- `main.cpp`：host 侧驱动，支持 `cpu / sim / npu`
- `run_sim_msprof_sweep.sh`：批量编译 + 运行 + `msprof`
- `parse_msprof_summary.py`：递归抓取 `msprof` 输出里的 CSV，汇总到一个结果表
- `fit_to_0315_config.py`：把结果拟合成 `0315_algo/config.py` 可直接替换的 `FORMAT_CONV_BW_GBs / FORMAT_CONV_OVERHEAD_US`

## kernel 模式

- `nd2nz_a`：把 `A(M,K)` 当作 ND 输入，逐 tile 做 `ND -> NZ`
- `nd2nz_b`：把 `B(K,N)` 当作 ND 输入，逐 tile 做 `ND -> NZ`
- `nz2zz_a`：把 A-path 的 NZ tile 输入到 `LoadData(ifTranspose=false)`
- `nz2zn_b`：把 B-path 的 NZ tile 输入到 `LoadData(ifTranspose=true)`
- `nd2zz_a`：链式 `ND -> NZ -> ZZ`
- `nd2zn_b`：链式 `ND -> NZ -> ZN`

## 一次性批量跑

```bash
cd ascend_format_bench
RUN_MODE=sim \
SOC_VERSION=Ascend910B1 \
MODES=nd2nz_a,nz2zz_a,nz2zn_b,nd2zz_a,nd2zn_b \
CASES=32x32x32,64x64x64,64x128x64,128x128x128,127x129x255 \
REPEAT=10 \
INNER_LOOPS=64 \
bash run_sim_msprof_sweep.sh
```

跑完后默认会生成：

- `profile/format_conv_results.csv`
- `profile/format_conv_fit.json`

## 单次运行

```bash
./out/bin/format_conv_bench_app --mode nd2zn_b --m 128 --n 128 --k 128 --repeat 10 --inner_loops 64
```

## 和 0315_algo 的关系

`0315_algo` 现在已经改成 **shape-aware + path-based**：

- 权重 host format 候选：`ND / NZ / ZZ / ZN`
- NPU 权重 consumer-side preferred format：`ZN`
- 转换时间按 conversion graph 组合，而不是单个常数 multiplier

这个 bench 对应的是最核心的三条直接边：

- `ND->NZ`
- `NZ->ZZ`
- `NZ->ZN`

反向边（如 `ZZ->ND` / `ZN->ND`）在当前代码里先用正向边参数占位；如果你后面要把反向边也做实测，建议再补一个专门的 `Fixpipe`/输出链 bench。


## 常见问题

如果脚本一启动就退出，优先检查这几件事：

- 是否从任意目录执行脚本。新版脚本会自动切到脚本目录；旧版不会。
- `ASCEND_INSTALL_PATH` / `ASCEND_HOME_PATH` 是否指向真实的 CANN 安装目录。
- `setenv.bash` 是否存在，以及在 `set -u` 下是否会触发未定义变量。新版脚本已在 `source` 前后临时关闭 `nounset`。
- `msprof` 是否在 `PATH` 里。

打开详细日志：

```bash
SCRIPT_DEBUG=1 bash run_sim_msprof_sweep.sh
```
