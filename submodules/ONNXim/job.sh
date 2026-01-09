#!/bin/bash
#SBATCH -J build_onnxim           # 作业名
#SBATCH --partition=C064M0256G
#SBATCH --qos=low                 # QOS(按需改)
#SBATCH -c 8                      # 核心数
#SBATCH -t 01:00:00               # 时限
#SBATCH -o build.%j.out           # 输出日志

set -eo pipefail

# 1) 加载工具链
module purge
module load cmake/3.25.0

source /lustre/home/2501111916/anaconda3/etc/profile.d/conda.sh
conda activate onnxim

# 2) 回到工程目录，重新生成一个干净的 build 目录
cd /lustre/home/2501111916/workspace/ONNXim
rm -rf build
mkdir -p build
cd build

# 4) 配置 + 编译（优先用 Ninja；没有 Ninja 就用 Make）

conan install ..

if command -v ninja >/dev/null 2>&1; then
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
  ninja -j "$SLURM_CPUS_PER_TASK"
else
  cmake -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . -j "$SLURM_CPUS_PER_TASK"
fi


