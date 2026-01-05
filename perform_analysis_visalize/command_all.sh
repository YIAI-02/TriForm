
source ~/.bashrc
conda activate fig

python analyze_speedup_comparison.py --output-root ../algorithms/output/experiment_2npu --out-dir ../figs/experiment_2npu
python analyze_speedup_comparison.py --output-root ../algorithms/output/experiment_4npu --out-dir ../figs/experiment_4npu

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_2npu/hw_2npu_aim \
    --out  ../algorithms/output/experiment_2npu/hw_2npu_aim_stat_out \
    --baseline ALL\
    --main-algo hefthint

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_2npu/hw_2npu_2aim \
    --out  ../algorithms/output/experiment_2npu/hw_2npu_2aim_stat_out \
    --baseline ALL\
    --main-algo hefthint

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_2npu/hw_2npu_4aim \
    --out  ../algorithms/output/experiment_2npu/hw_2npu_4aim_stat_out \
    --baseline ALL\
    --main-algo hefthint

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_2npu/hw_2npu_6aim \
    --out  ../algorithms/output/experiment_2npu/hw_2npu_6aim_stat_out \
    --baseline ALL\
    --main-algo hefthint

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_2npu/hw_2npu_8aim\
    --out  ../algorithms/output/experiment_2npu/hw_2npu_8aim_stat_out \
    --baseline ALL\
    --main-algo hefthint


python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_4npu/hw_4npu_aim \
    --out  ../algorithms/output/experiment_4npu/hw_4npu_aim_stat_out \
    --baseline ALL\
    --main-algo hefthint

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_4npu/hw_4npu_2aim \
    --out  ../algorithms/output/experiment_4npu/hw_4npu_2aim_stat_out \
    --baseline ALL\
    --main-algo hefthint

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_4npu/hw_4npu_4aim \
    --out  ../algorithms/output/experiment_4npu/hw_4npu_4aim_stat_out \
    --baseline ALL\
    --main-algo hefthint

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_4npu/hw_4npu_6aim \
    --out  ../algorithms/output/experiment_4npu/hw_4npu_6aim_stat_out \
    --baseline ALL\
    --main-algo hefthint

python analyze_heft_statistics.py \
    --root ../algorithms/output/experiment_4npu/hw_4npu_8aim\
    --out  ../algorithms/output/experiment_4npu/hw_4npu_8aim_stat_out \
    --baseline ALL\
    --main-algo hefthint