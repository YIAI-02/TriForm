
source ~/.bashrc
conda activate fig

python speedup.py --output-root ../algorithms/output/lens_eval_sweep --out-dir ../figs/speedup

python analyze_heft_speedup_all_in_one.py \
    --root ../algorithms/output/lens_eval_sweep/hw_scale_down_npu_large \
    --out  ../algorithms/output/hw_scale_down_npu_large \
    --baseline ALL

python analyze_heft_speedup_all_in_one.py \
    --root ../algorithms/output/lens_eval_sweep/hw_scale_down_pima \
    --out  ../algorithms/output/hw_scale_down_pima \
    --baseline ALL

python analyze_heft_speedup_all_in_one.py \
    --root ../algorithms/output/lens_eval_sweep/hw_scale_down_pima_large \
    --out  ../algorithms/output/hw_scale_down_pima_large \
    --baseline ALL

python analyze_heft_speedup_all_in_one.py \
    --root ../algorithms/output/lens_eval_sweep/hw_scale_down_pima_small \
    --out  ../algorithms/output/hw_scale_down_pima_small \
    --baseline ALL