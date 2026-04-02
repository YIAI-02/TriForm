# DOPS: Dynamic OPerator Sorting for Heterogeneous NPU–PIM LLM Inference

DOPS is a simulation-and-analysis framework for studying decoder-only LLM inference on heterogeneous **NPU–PIM** platforms. It accompanies the paper **Beyond Prefill-Decode Partition: Dissecting LLM Inference for Heterogeneous Platforms via Dynamic OPerator Sorting** and provides a practical code for graph construction, hardware abstraction, runtime modeling, scheduling, weight-layout search, trace export, and result analysis.

**Quick links:** Demo video *(coming soon)* 

---

## DOPS Framework

DOPS is built around a closed loop that starts from three inputs:

1. an **LLM model card**, 
2. a **hardware abstraction** of the target heterogeneous system, and
3. a **workload configuration** (batch, prefill length, decode length, scheduling/search settings).

From these inputs, DOPS:

- builds a stage-aware execution DAG for prefill and decode,
- instantiates **performance models** and **communication topology models**,
- searches for a dynamic operator-to-device mapping using the **Bifocal scheduler**,
- optionally searches for a blockwise persistent-weight layout using the **Weight Layout Arbiter**,
- exports schedules, traces, and summary JSON files for downstream comparison and visualization.

At a high level, DOPS follows the workflow below.

```mermaid
flowchart LR
    A[LLM model card<br/>+ optimization annotations] --> D[Computation Graph Builder]
    B[Hardware abstraction<br/>NPU / PIM / links / capacities] --> D
    C[Workload configuration<br/>batch / prefill / decode / TP / search knobs] --> D

    D --> E[Performance model<br/>+ communication topology]
    E --> F[Bifocal scheduler]
    E --> G[Weight Layout Arbiter]

    F --> H[Operator placement<br/>and execution timeline]
    G --> I[Blockwise weight-layout map]

    H --> J[Simulated latency / traces]
    I --> J
    J --> K[Verify / deploy / compare]
```

---

## Repository layout

```text
.
├── algorithms/      # Main implementation of DOPS
│   ├── main.py                              # Main CLI entry point. Supports `evaluate` and `weight-suggest`.
│   ├── model_parser.py                      # Loads shape cards, validates tensor-parallel settings, builds the DAG, and applies optimization annotations.
│   ├── model_definition.py                  # Defines decoder-only graph construction logic, including attention/FFN subgraphs, TP sharding, and collectives such as reduce, scatter, and all-reduce.
│   ├── task_graph.py                        # Core graph data structures.
│   ├── hardware.py                          # Parses hardware JSON files, normalizes topology, builds device/link objects, and checks PIM capacity consistency against the address map.
│   ├── cost_model.py                        # Unified runtime model. Handles compute, memory, communication, weight loading, format conversion, cache behavior, and backend dispatch.
│   ├── cost_model_npu_ascend_backend.py     # LUT-based NPU backend for Ascend-style operator latency lookup/interpolation.
│   ├── cost_model_npu_llmcompass_backend.py # Optional LLMCompass integration for NPU operator estimation.
│   ├── cost_model_pim_backend.py            # PIM backend helpers, trace generation, Ramulator2 integration, and shared PIM model-dictionary construction.
│   ├── scheduler.py                         # Aggregates available scheduler classes.
│   ├── scheduler_heft.py                    # Baseline HEFT scheduler.
│   ├── scheduler_heft_commaware.py          # Bifocal scheduling flow.
│   ├── scheduler_naive.py                   # Simple topology/order baseline scheduler.
│   ├── scheduler_common.py                  # Shared scheduling utilities, hint logic, buffer/cache interaction, communication accounting, and trace/stat collection.
│   ├── scheduler_comm.py                    # Communication-related scheduling helpers.
│   ├── comm_primitives.py                   # Collective communication primitives and topology-aware transfer helpers.
│   ├── buffer_manager.py                    # Global memory manager and LRU-style cache abstractions.
│   ├── plan_label.py                        # Metadata carrier for KV placement, PIM capacity, pinned weights, and trace artifacts.
│   ├── optimizations.py                     # Optional graph annotations for quantization, weight sparsity, activation sparsity, and attention sparsity.
│   ├── stats_recorder.py                    # Writes raw op/comm traces and overlap summaries.
│   ├── weight_stage_trace_tools.py          # Post-processes op traces into weight-stage summaries grouped by phase, device type, or operator.
│   ├── weight_stage_models.py               # Utility functions for overlap-ratio modeling.
│   ├── runtime_models/                      # Packaged runtime tables.
│   ├── examples/                            # Example configs and hardware descriptions used by `main.py evaluate` and `main.py weight-suggest`.
│   ├── aim_simulator/                       # Example AiM/PIM config files used by the trace-based PIM path.
│   ├── pkl/                                 # Cached latency/model artifacts for PIM trace execution.
│   ├── sweep_*.py                           # Sweep-related scripts.
│   ├── *.sh                                 # Shell scripts.
│   └── *.slurm                              # Slurm job scripts.
├── configs/         # model shape cards (Llama, Qwen, Mixtral, ...)
├── experiment/      # paper-figure scripts and interactive schedule visualization
├── measurements/    # microbenchmarks, profiling utilities, LUT-generation scripts
└── submodules/      # external backends such as LLMCompass and CENT / Ramulator-based flows
```
---

## Dependencies

Recommended Python version: **3.10+**

### Minimal installation for the current CLI path

If your goal is to reuse only the analytical/runtime-model part of the framework, the smallest practical subset is:

```bash
pip install torch typing_extensions
```

### Optional external runtimes and toolchains

These are only needed for specific backends or measurement pipelines.

- **Ramulator2 / trace-based PIM flow**
  Needed when you want the trace-based PIM backend instead of the analytical fast path. Place the submodule under `submodules/CENT/` and install the dependencies required by that project.

- **Huawei CANN / Ascend-C / msprof**
  Needed for reproducing measured results.

- **LLMCompass**
  Needed when `npu_backend=llmcompass`. Place the submodule under `submodules/LLMCompass/` and install the dependencies required by that project.

- **Plotting / analysis stack**
  The full paper-reproduction workflow may additionally use a standard scientific Python stack such as `numpy`, `pandas`, `matplotlib`, `scipy`, `seaborn`, and graph/IO helpers depending on the scripts you run.

---

## How to start

The usual workflow is:

1. prepare a **hardware JSON** under `algorithms/examples/`,
2. prepare an **evaluation config** such as `evaluate_test_config.json`,
3. run `evaluate` to compare DOPS/Bifocal against static baselines,
4. run `weight-suggest` to search for a mixed persistent-weight layout.

### Step 1. Prepare a hardware JSON

The hardware JSON can be wrapped either as `{"hardware": ...}` or `{"cluster": ...}`. The parser accepts two major topology styles:

- **`fc`**: fully connected fabric. You can either provide one shared bandwidth (`fc_bw_GBs`) or explicit `links`.
- **`star`**: host-centric fabric. Use explicit links that connect all accelerators through the host.

A minimal example looks like this:

```json
{
  "hardware": {
    "topology": "fc",
    "devices": [
      {
        "name": "CPU0",
        "type": "cpu",
        "tflops": 0.01,
        "mem_bw_GBs": 32768.0,
        "mem_capacity_GB": 8192.0,
        "cpu_read_latency_ns": 70.0,
        "cpu_write_latency_ns": 85.0,
        "cpu_cacheline_B": 64
      },
      {
        "name": "Ascend_910B_NPU0",
        "type": "npu",
        "tflops": 280.0,
        "mem_bw_GBs": 819.2,
        "mem_capacity_GB": 16.0
      },
      {
        "name": "PIM0",
        "type": "pim",
        "pim_type": "accel",
        "tflops": 16.0,
        "mem_bw_GBs": 16384.0,
        "mem_capacity_GB": 16.0,
        "pim_read_latency_ns": 5.0,
        "pim_write_latency_ns": 30.0,
        "freq_ghz": 0.8,
        "pim_memory": {
          "addr_map_unit": "bits",
          "addr_map": {
            "row": 14,
            "channel": 1,
            "bank": 2,
            "column": 8,
            "offset": 12
          }
        }
      }
    ],
    "fc_bw_GBs": 32.0,
    "link_defaults": {
      "latency_s": 0.0,
      "overhead_s": 0.0,
      "flit_size_B": 16,
      "max_payload_B": 256
    }
  }
}
```

#### Hardware JSON: top-level keys

| Key         | Meaning                                                       | How to set it                                                                            |
| ----------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `topology`  | Interconnect type.                                            | Use `fc` for a shared fully connected fabric, or `star` for host-mediated communication. |
| `devices`   | List of device descriptors.                                   | Add one entry per concrete CPU/NPU/PIM device.                                           |
| `fc_bw_GBs` | Default bandwidth for all unspecified pairs in `fc` topology. | Use when you want one shared link bandwidth instead of enumerating all pairs.            |
| `links`     | Explicit point-to-point links.                                | Required for custom fabrics.                                                             |

#### Hardware JSON: device fields

| Key                                                   | Meaning                                                       | Notes                                                                                                                   |
| ----------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `name`                                                | Unique device name.                                           | Used throughout traces and schedule output.                                                                             |
| `type`                                                | Device class.                                                 | Must be one of `cpu`, `npu`, or `pim`.                                                                                  |
| `tflops`                                              | Peak compute throughput.                                      | Used by the analytical cost model.                                                                                      |
| `mem_bw_GBs`                                          | Device memory bandwidth.                                      | For NPU/CPU this is off-chip bandwidth; for PIM it can represent the near-memory bandwidth model used by the simulator. |
| `mem_capacity_GB`                                     | Device memory capacity.                                       | Used for feasibility checks, KV placement, and preload budgeting.                                                       |
| `arch` / `llmcompass_kind`                            | Optional architecture tag.                                    | Useful when `npu_backend=llmcompass`.                                                                                   |
| `cpu_read_latency_ns`, `cpu_write_latency_ns`         | Optional host-access latency parameters.                      | Meaningful for CPU/host modeling.                                                                                       |
| `cpu_cacheline_B` / `cpu_access_bytes_B`              | Host access granularity.                                      | Defaults to 64 B if omitted.                                                                                            |
| `pim_read_latency_ns`, `pim_write_latency_ns`         | PIM local memory latency parameters.                          | Used by the PIM backend.                                                                                                |
| `freq_ghz`                                            | PIM operating frequency.                                      | Used by some trace-mode flows.                                                                                          |
| `pim_memory`                                          | PIM address-map description.                                  | Required for strict capacity checking.                                                                                  |
| `addr_map_unit`                                       | Whether `addr_map` values are address **bits** or **counts**. | The current examples use `"bits"`.                                                                                      |
| `addr_map.row`, `channel`, `bank`, `column`, `offset` | PIM address-map decomposition.                                | The parser checks that this implied capacity matches `mem_capacity_GB`.                                                 |
| `capacity_bytes` / `capacity_B`                       | Optional explicit capacity override.                          | Useful when the address map is unavailable.                                                                             |

#### Hardware JSON: link fields

| Key                                        | Meaning                                                     | Notes                                                        |
| ------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| `a`, `b`                                   | Source and destination device names.                        | Link direction is treated symmetrically in the parser.       |
| `bw_GBs`                                   | Link bandwidth in GB/s.                                     | Required for explicit links.                                 |
| `latency_s`, `latency_us`, `latency_ns`    | Optional fixed latency term.                                | Use whichever unit is most convenient.                       |
| `overhead_s`, `overhead_us`, `overhead_ns` | Optional packet/message overhead term.                      | Useful when you want a LogGP/AHEAD-like communication model. |
| `flit_size_B`                              | Packetization granularity.                                  | Optional.                                                    |
| `max_payload_B`                            | Payload size before extra packet headers/flits are charged. | Optional.                                                    |

#### Practical guidance

- Use **one CPU host** to model shared memory / host-side routing even if you do not schedule much work on CPU.
- For every **PIM** device, make sure `mem_capacity_GB` matches the capacity implied by `pim_memory.addr_map`; the parser validates this and will raise an error if they disagree.

---

### Step 2. Prepare `evaluate_test_config.json`

A representative config looks like this:

```json
{
  "model_family": "llama",
  "model_variant": "7b",
  "dtype": "fp16",
  "batch": 1,
  "prefill_len": 512,
  "decode_len": 128,
  "decode_sample_stride": 2,
  "decode_plan_refresh_stride": 2,
  "pim_config_path": "./aim_simulator/PIM_AiM.json",
  "gb_config_path": "./aim_simulator/gb.json",
  "ramulator_config_path": "./aim_simulator/example.yaml",
  "result_dir": "./output/evaluate_single_test/",
  "hardware_json": "./examples/hardware_1npu_2aim.json",
  "algo": ["bifocal"],
  "baselines": ["pd"],
  "tp_qkv": 2,
  "tp_ffn": 2,
  "npu_backend": "lut",
  "dump_graph": false,
  "dump_graph_dir": "./output/graph_dumps",
  "pim_weight_load_overlap_ratio": 0.0,
  "weight_load_compute_overlap_ratio": 0.0
}
```

#### Core run keys

| Key                          | Meaning                                                                                | How to set it                                                                                                                 |
| ---------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `model_family`               | Model family name.                                                                     | Use a packaged family such as `llama`, `qwen`, or `mixtral`, unless you provide `shape_file`.                                 |
| `model_variant`              | Model-card variant.                                                                    | Example: `7b`, `13b`, `70b`, `1.8b`, `8x7b`.                                                                                  |
| `dtype`                      | Logical datatype used by the simulator.                                                | `fp16` is the common default.                                                                                                 |
| `result_dir`                 | Base output directory.                                                                 | DOPS will append a run-specific folder name such as `<family>_<variant>_<dtype>_b<batch>[_s<stride>]`.                        |
| `hardware_json`              | Path to the hardware description.                                                      | Usually a file under `algorithms/examples/`.                                                                                  |
| `npu_backend`                | NPU operator-latency backend.                                                          | Use `fast`/`fast_mode` for analytical estimation, `lut` for LUT-backed Ascend-style modeling, or `llmcompass` for LLMCompass. |
| `decode_plan_refresh_stride` | Re-run an exact decode search every `N` tokens and replay a fixed plan in between.     | Smaller values are more accurate but slower; larger values are faster. `0` means do not refresh after the warm-up plan.       |
| `decode_sample_stride`       | Controls how densely per-token schedules are stored in the exported decode trace JSON. | This affects output detail rather than the simulated makespan itself. Tokens `0`, `1`, and the final token are always kept.   |
| `debug`                      | Verbose logging flag.                                                                  | Can be enabled in JSON or with `--debug`.                                                                                     |

#### Scheduler and baseline controls

| Key         | Meaning                 | How to set it                                                                                                       |
| ----------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `algo`      | Dynamic scheduler list. | Typical choices are `Bifocal`, `heft`.                                                                              |
| `baselines` | Static policy list.     | Common choices are `pd`, `weights_on_pim`, `attn_on_pim`, `ianus`, `neupims`, `facil`, and `attacc`. （这里需要改） |

#### Parallelism / graph-sharding controls

| Key      | Meaning                                                                 | How to set it                                                                                                                          |
| -------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `tp_qkv` | Tensor-parallel split for Q/K/V generation and attention head sharding. | Must divide both `n_heads` and `n_kv_heads`, unless you intentionally exceed `n_heads` and let the code fall back to KV-head sharding. |
| `tp_ffn` | Tensor-parallel split for FFN.                                          | Must divide `ffn_dim`.                                                                                                                 |
| `tp_moe` | Expert-parallel split for Mixtral/MoE.                                  | Only meaningful for MoE models and must divide the per-layer expert count.                                                             |

#### Model-card override and graph annotations

| Key                                        | Meaning                                                           | How to set it                                                                |
| ------------------------------------------ | ----------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `shape_file`                               | Path to a custom model-card JSON.                                 | Use this instead of editing the packaged `configs/` mapping.                 |
| `quantization`                             | Optional quantization annotation block.                           | Supports modes such as weight-only and W8A8-style annotations.               |
| `weight_sparsity`                          | Optional weight sparsity annotation block.                        | Used to adjust bytes/FLOPs and annotate graph nodes.                         |
| `activation_sparsity`                      | Optional activation sparsity block.                               | Can be phase-dependent.                                                      |
| `attention_sparsity`                       | Optional sparse-attention block.                                  | Used to model reduced attention work.                                        |
| `optimizations` / `optimization` / `optim` | Alternative root object containing the optimization blocks above. | Use whichever naming style is more convenient; the parser accepts all three. |

#### Runtime-overlap controls

| Key                                 | Meaning                                                                                 | How to set it                                   |
| ----------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `pim_weight_load_overlap_ratio`     | Overlap ratio between host-to-PIM transfer and local PIM-format conversion/programming. | Use values in `[0, 1]`; `0` means fully serial. |
| `weight_load_compute_overlap_ratio` | Overlap ratio between weight loading and compute for weight-bearing operators.          | Use values in `[0, 1]`; `0` means fully serial. |



#### Weight-layout search keys

These keys are used by `weight-suggest`.

| Key                                | Meaning                                                                     | How to set it                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `format_block_layer_span`          | Group the same weight across every `N` layers into one block.               | `4` or `8` are common. Larger values reduce search space; smaller values allow finer control.       |
| `format_block_change_percent`      | Maximum fraction of blocks allowed to change in one outer iteration.        | Smaller values are more conservative.                                                               |
| `format_outer_max_iters`           | Legacy compatibility knob.                                                  | If `format_block_change_percent` is not set, the code derives it from `1 / format_outer_max_iters`. |
| `format_inner_max_blocks`          | Cap on inner refinement candidates.                                         | `0` means no cap.                                                                                   |
| `format_nd_margin_init`            | Initial neutrality margin around the default dense layout `Linear` .        | Larger values keep more blocks in `Linear` early in the search.                                     |
| `format_nd_margin_decay`           | Per-outer-iteration decay factor for the neutral band.                      | Controls how quickly the search becomes more aggressive.                                            |
| `format_nd_margin_min`             | Lower bound on the neutral band.                                            | Prevents over-splitting late in the search.                                                         |
| `format_inner_improve_eps`         | Minimum improvement required to accept an inner flip.                       | Helps reject noise-level changes.                                                                   |
| `format_outer_stop_eps`            | Early-stop tolerance between outer iterations.                              | Stops when the outer-stage update is no longer beneficial enough.                                   |
| `format_reload_count_mode`         | How block reload pressure is normalized when comparing NPU vs PIM pressure. | Use `raw`, `per_device`, or `soft_per_device`.                                                      |
| `format_reload_device_count_alpha` | Soft normalization factor used by `soft_per_device`.                        | Tune this when you want to reduce device-count bias without fully averaging it out.                 |

---

### Step 3. Run the Bifocal scheduler

```bash
cd algorithms

python main.py evaluate \
  --config ./examples/evaluate_test_config.json \
  --npu_backend fast_mode \
  --pim_fast_mode \
  --debug
```

This command performs the full scheduling/evaluation flow. In order, it:

1. loads the model shape card (or your custom `shape_file`),
2. validates tensor-parallel settings,
3. builds the execution DAG for prefill and decode,
4. loads the hardware JSON and instantiates the device/link topology,
5. builds the cost model with the selected NPU backend and PIM mode,
6. runs the requested dynamic scheduler(s) and static baseline(s),
7. exports per-policy schedules, traces, and summaries,
8. writes one combined comparison JSON for easy downstream plotting.

#### Typical output layout

The run directory is automatically named as:

```text
<result_root>/<family>_<variant>_<dtype>_b<batch>[_s<refresh_stride>]/
```

A representative `evaluate` output tree looks like this:

```text
<result_dir>/
├── baseline_compare_<prefill>x<decode>.json
├── driver_debug.txt
├── algo_pd/
│   ├── best_summary_<prefill>x<decode>.json
│   ├── debug_<prefill>x<decode>.txt
│   ├── pd_prefill-<prefill>xdecode_<decode>_ops_trace.csv
│   ├── pd_prefill-<prefill>xdecode_<decode>_comms_trace.csv
│   └── pim_sim_<prefill>x<decode>.txt
├── algo_xxxx/
│   └── ...
└── algo_hefthint/
    ├── best_summary_<prefill>x<decode>.json
    ├── debug_<prefill>x<decode>.txt
    ├── hefthint_prefill-<prefill>xdecode_<decode>_ops_trace.csv
    ├── hefthint_prefill-<prefill>xdecode_<decode>_comms_trace.csv
    └── pim_sim_<prefill>x<decode>.txt
```

Exact filenames can vary slightly when you add custom artifact tags or storage-mode tags, but the structure stays the same.

#### How to read the results

- **`baseline_compare_*.json`** is the easiest file to consume in scripts. It stores the top-level config and a flat list of results, each with `prefill_time_s`, `decode_time_s`, and `total_time_s`.
- **`algo_<policy>/best_summary_*.json`** contains the richer schedule export for one policy. It typically records the chosen KV policy, serialized prefill schedule, sampled decode schedules, and pointers to generated trace files.
- **`*_ops_trace.csv`** contains operator execution events with device assignment, timing, and weight-stage details. It is the main input for timeline visualizers and overlap breakdown tools.
- **`*_comms_trace.csv`** contains inter-device communication events and is useful when studying topology bottlenecks or collective overhead.

#### What you can do with `evaluate`

`evaluate` is the right mode for:

- comparing **Bifocal (`hefthint`)** against **PD**, **AF-style attention-offload**, **IANUS-inspired**, **FACIL-inspired**, **AttAcc-inspired**, and other baselines,
- running **hardware-scaling studies** by swapping `hardware_json` files,
- generating **schedule traces** for downstream visualization,
- studying how performance changes with **batch size**, **prefill length**, **decode length**, and **TP sharding**.

---

### Step 4. Run the Weight Layout Arbiter

```bash
cd algorithms

python main.py weight-suggest \
  --config ./examples/evaluate_test_config.json \
  --npu_backend fast_mode \
  --pim_fast_mode \
  --debug
```

This mode runs the paper’s **Weight Layout Arbiter** on top of the scheduling flow. In broad terms, it:

1. groups persistent weights into stable blocks,
2. starts from a default storage mode (the current implementation treats `ND` / Linear as the default),
3. evaluates schedules under that layout,
4. performs an **outer dominance-assignment** update using block reload pressure,
5. performs an **inner targeted-refinement** update for blocks that still disagree with observed loading behavior,
6. saves the best blockwise storage map and comparison reports.

#### Typical output layout

```text
<result_dir>/
├── all_passes_<tag>.json
├── best_summary_<tag>.json
├── weight_storage_suggestion_<tag>.json
├── weight_storage_suggestion_<tag>_full.json
├── weight_storage_suggestion_<tag>_compare.json
├── weight_suggest_al_debug.txt
└── pim_sim_<tag>.txt
```

#### How to read the results

- **`all_passes_*.json`** records the entire search history. Use it when you want to inspect every outer/inner iteration, not just the final answer.
- **`best_summary_*.json`** stores the best pass, best times, best schedule exports, and the improvement relative to other passes.
- **`weight_storage_suggestion_*.json`** stores the compact blockwise format map.
- **`weight_storage_suggestion_*_full.json`** expands the blockwise decision to the per-weight map, which is convenient if you want to feed the result into another toolchain.
- **`weight_storage_suggestion_*_compare.json`** compares the searched layout against built-in fixed references such as `PD + Linear`, `PD + DUAL`, `hefthint + Linear`, and `hefthint + DUAL`.
- **`weight_suggest_al_debug.txt`** is the most useful file for debugging the arbiter itself. It logs initialization, outer-stage updates, accepted inner flips, and stop conditions.

#### What you can do with `weight-suggest`

This mode is the right choice when you want to study:

- **Linear vs optimized weight-layout comparisons**,
- **mixed blockwise layout search** instead of one single global storage rule,
- **iteration traces** that show where the arbiter converges,
- the interaction between **weight layout** and **dynamic scheduling** under the same workload/hardware setting.

---

## How to extend the framework

### Add a new model

1. Add a shape JSON under `configs/`.
2. Register it in `algorithms/model_parser.py` (`FILE_MAP`).
3. Extend `algorithms/model_definition.py` if the graph structure differs from existing decoder families.
4. If the model has MoE-specific behavior, also check the `tp_moe` path and any expert-routing assumptions.

### Add a new hardware target

1. Create a new hardware JSON under `algorithms/examples/`.
2. Make sure device names, capacities, and topology are consistent with the backend you want to use.
3. For PIM devices, make sure `pim_memory.addr_map` matches `mem_capacity_GB`.
4. Pass the file through `--hardware_json` or set it in the config JSON.

### Add a new scheduler or baseline

- **New scheduler**: implement it in `algorithms/scheduler_*.py`, export it through `algorithms/scheduler.py`, and wire it into `_make_scheduler()` in `algorithms/main.py`.
- **New baseline**: register it in `_BASELINE_REGISTRY` in `algorithms/main.py` with the `@register_baseline(...)` decorator.

### Add a new NPU or PIM backend

- Extend the backend logic in `algorithms/cost_model.py`.
- Add a new NPU backend in `algorithms/cost_model_npu_*.py` or a new PIM path in `algorithms/cost_model_pim_backend.py`.
- Place supporting tables under `algorithms/runtime_models/` when needed.

### Add a new optimization annotation

If you want to model a new graph-side optimization such as another quantization or sparsity scheme, the main entry point is:

- `algorithms/optimizations.py`

This file already contains the parsing and graph-annotation flow for quantization, weight sparsity, activation sparsity, and attention sparsity.

---

## Visualization

### `experiment/experiment_fig/`

These scripts generate the paper figures. Their headers are intended to document runnable examples and plotting assumptions.

### Experiment 1: scheduling benefits

- `plot_exp1_simulated.py`  
  Simulated latency and speedup comparison across policies.
- `plot_exp1_batch.py`  
  Batch-sensitivity analysis.
- `plot_exp1_utilization_violin.py`  
  Distribution-style utilization and co-utilization plots.
- `plot_exp1_gantt.py`  
  Case-study Gantt/timeline figure.

### Experiment 2: hardware scaling

- `plot_exp2_heatmap.py`  
  Speedup heatmaps across prefill/decode/model/hardware axes.
- `plot_exp2_baseline_marginal.py`  
  Marginal-return curves as more PIM budget is added.

### Experiment 3: Weight Layout Arbiter

- `plot_exp3_layout_compare.py`  
  Compares conventional/global layouts against arbiter-optimized results.
- `plot_exp3_iter_from_txt_json.py`  
  Rebuilds arbiter iteration traces from JSON/log output.
- `plot_exp3_layout_compare_cfg_example.json`  
  Example plotting config.

### `experiment/demo/`

#### `server.py`

Launch the local HTTP demo server:

```bash
cd experiment/demo
python server.py
```

The browser demo is designed to visualize **simulated CSV traces** generated by the framework, rather than live hardware telemetry. In practice, the most useful inputs are the exported files such as:

- `*_ops_trace.csv`
- `*_comms_trace.csv`
- derived weight-stage summary CSV/JSON files

These files are generated by `evaluate`, so the typical workflow is to run a simulation first and then load the produced CSVs into the demo.

**Demo assets**

- Demo video: *to be added*
- Demo screenshot: *to be added*
