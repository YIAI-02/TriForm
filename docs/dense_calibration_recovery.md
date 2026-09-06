# Dense 扰动与 CAMC 校准实验导出

本次配套 Het-Infer 的 Dense min-cut 反馈重分配实验。DOPS 负责生成原始放置、逐轮网络与计时 bundle；运行期干预、反馈校准和重分配在 Het-Infer 完成。

- commands/export_hetinfer_dense_calibration_suite.py 定义 W1–W7 的模型形状、batch、prefill 与 decode 轮数。已有符合条件的1.8B bundle可复用；W7（Qwen7B仓库形状，B4/P256/H24）复用精确计时覆盖并重新导出原始DOPS放置。
- experiment.json 的 decode_rounds 读取配置 decode_len，支持24、32、128轮。
- 导出不注入扰动，也不手工修改 Dense 的默认设备放置。干预只作用于 Het-Infer measurement 服务时间。
- 需已有严格NPU LUT、CENT/AIM与Ramulator2环境。脚本中的HPC路径按当前实验机器配置；原始bundle及计时缓存保留在HPC output目录。

在登录节点进入DOPS根目录，提交计算节点作业：

```bash
mkdir -p results
sbatch commands/run_dense_calibration_export.slurm --case W1
sbatch commands/run_dense_calibration_export.slurm --case W6
```

完整setup、19组结果、67点扫描、三个原版CAMC正例及公式说明见 [Het-Infer 实验报告](https://github.com/YIAI-02/Het-Infer/blob/v1_CAMA.md/docs/experiments/2026-09-06_Dense_校准重分配闭环_实验报告.md)。

该实验是NPU LUT/PIM AIM与时间线执行器的模拟回放，未执行真实模型张量或生成token。主例为QK@PIM0×1.25、Softmax@PIM0×2、SwiGLU@NPU×4；分别从第9轮第3、2、3层起发生目标迁移。完整阶段较各自CAMC OFF减少1.904862%、0.034860%、0.014651%；不能把小占比算子的机制示例表述为显著硬件端到端加速。

发布验证：Slurm 14780090 的 DOPS 相关导出回归31项通过；配套 Het-Infer 独立发布版本80项通过、1项跳过，并完整复现上述三个正例。
