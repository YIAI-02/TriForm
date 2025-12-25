
source ~/.bashrc
conda activate fig

# python analyze_speedup_comparison.py --output-root ../algorithms/output/experiment_2npu --out-dir ../figs/experiment_2npu
# python analyze_speedup_comparison.py --output-root ../algorithms/output/experiment_npu --out-dir ../figs/experiment_npu

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_2npu/hw_2npu_4aim \
    --out  ../algorithms/output/experiment_2npu/hw_2npu_4aim_stat_out \
    --baseline ALL\
    --main-algo heft

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_npu/hw_npu_2aim \
    --out  ../algorithms/output/experiment_npu/hw_npu_2aim_stat_out \
    --baseline ALL\
    --main-algo heft

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_npu/hw_npu_4aim \
    --out  ../algorithms/output/experiment_npu/hw_npu_4aim_stat_out \
    --baseline ALL\
    --main-algo heft

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_npu/hw_npu_6aim \
    --out  ../algorithms/output/experiment_npu/hw_npu_6aim_stat_out \
    --baseline ALL\
    --main-algo heft

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_npu/hw_npu_8aim\
    --out  ../algorithms/output/experiment_npu/hw_npu_8aim_npu_stat_out \
    --baseline ALL\
    --main-algo heft

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_npu/hw_npu_10aim\
    --out  ../algorithms/output/experiment_npu/hw_npu_10aim_stat_out \
    --baseline ALL\
    --main-algo heft