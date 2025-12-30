
source ~/.bashrc
conda activate fig

python analyze_speedup_comparison.py --output-root ../algorithms/output/experiment_halfnpu --out-dir ../figs/experiment_halfnpu
# python analyze_speedup_comparison.py --output-root ../algorithms/output/experiment_npu --out-dir ../figs/experiment_npu

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_halfnpu/hw_halfnpu_aim \
    --out  ../algorithms/output/experiment_halfnpu/hw_halfnpu_aim_stat_out \
    --baseline ALL\
    --main-algo heft

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_halfnpu/hw_halfnpu_2aim \
    --out  ../algorithms/output/experiment_halfnpu/hw_halfnpu_2aim_stat_out \
    --baseline ALL\
    --main-algo heft

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_halfnpu/hw_halfnpu_4aim \
    --out  ../algorithms/output/experiment_halfnpu/hw_halfnpu_4aim_stat_out \
    --baseline ALL\
    --main-algo heft

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_halfnpu/hw_halfnpu_6aim \
    --out  ../algorithms/output/experiment_halfnpu/hw_halfnpu_6aim_stat_out \
    --baseline ALL\
    --main-algo heft

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_halfnpu/hw_halfnpu_8aim\
    --out  ../algorithms/output/experiment_halfnpu/hw_halfnpu_8aim_stat_out \
    --baseline ALL\
    --main-algo heft

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_halfnpu/hw_halfnpu_10aim\
    --out  ../algorithms/output/experiment_halfnpu/hw_halfnpu_10aim_stat_out \
    --baseline ALL\
    --main-algo heft