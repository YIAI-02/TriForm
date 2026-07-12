# Experiment Hyperparameters for Paper

This page is a checklist for the hyperparameters used in the paper experiments. 

## Bifocal scheduler hyperparameter table

| Hyperparameter / config key | Meaning | Sec. 6.2 value | Sec. 6.3 value | Sec. 6.4 value |
|---|---|---:|---:|---|
| `SCHED_JOINT_LK_ENABLE` | Enable DAG-window lookahead term | True | True | True |
| `SCHED_JOINT_LK_H` | Lookahead depth/window size | 3 | 3 | 3 | 
| `SCHED_JOINT_LK_GAMMA` | Lookahead score weight | {0.05, 0.2, 0.4} | {0.05, 0.2} | 0.05 |
| `SCHED_JOINT_LK_CONSIST_LAMBDA` | Near-future placement-consistency penalty/weight | {1.0, 4.0} | 1.0 | 0.05 |
| `SCHED_JOINT_LK_PLAN_HINT_MAX` | Maximum number of retained near-future hints | 3 | 3 | 3 |
| `SCHED_WEIGHT_BIAS_ETA` | Weight-reuse / phase-reuse bias multiplier | {0.02, 0.1, 2.0} | {0.02, 0.1} | 0.1 |
| `SCHED_DECODE_AMORT_ENABLE` | Enable decode token-amortization bias | True | True | True |
| `SCHED_DECODE_AMORT_ALPHA` | Token-amortization bias weight | 1 | 1 | 1 |
| `SCHED_DECODE_AMORT_RMIN` | Minimum remaining decode horizon threshold | 1 | 1 | 1 |
| `SCHED_DECODE_AMORT_REUSE_PROB` | Reuse-probability multiplier | 1.0 | 1.0 | 1.0 |
| `decode_sample_stride` | Decode-step sampling stride used during simulation/export | 2 | 2 | 2 |
| `decode_plan_refresh_stride` | Decode scheduling-plan refresh stride | 2 | 2 | 2 |
| `weight_load_compute_overlap_ratio` | Assumed overlap between weight load and compute | 1.0 | 1.0 | 1.0 | Required if overridden in configs |


