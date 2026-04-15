# Configuration Reference

This document collects the configuration keys and runtime options used by DOPS.

## Contents

- [Required files](#required-files)
- [Hardware JSON reference](#hardware-json-reference)
- [Evaluation config reference](#evaluation-config-reference)
- [Weight-layout search options](#weight-layout-search-options)

---

## Required files

A typical run needs two inputs:

1. a **hardware JSON** that describes devices and interconnects, and
2. an **evaluation config JSON** that describes the model, workload, and runtime settings.

Example locations:

- `src/examples/hardware_1npu_2aim.json`
- `src/examples/evaluate_test_config.json`

---

## Hardware JSON reference

The hardware file can be wrapped as either:

```json
{ "hardware": { ... } }
```

or:

```json
{ "cluster": { ... } }
```

### Top-level keys

| Key             | Meaning                            | Notes                                          |
| --------------- | ---------------------------------- | ---------------------------------------------- |
| `topology`      | Interconnect type                  | Use `fc` or `star`.                            |
| `devices`       | List of device descriptors         | One entry per CPU / NPU / PIM device.          |
| `fc_bw_GBs`     | Shared bandwidth for `fc` topology | Optional if you provide explicit `links`.      |
| `links`         | Explicit point-to-point links      | Recommended for custom fabrics.                |
| `link_defaults` | Default communication parameters   | Optional helper for latency / packet settings. |

### Device fields

| Key                                           | Meaning                        | Notes                                      |
| --------------------------------------------- | ------------------------------ | ------------------------------------------ |
| `name`                                        | Unique device name             | Used in traces and schedule output.        |
| `type`                                        | Device class                   | Must be `cpu`, `npu`, or `pim`.            |
| `tflops`                                      | Peak compute throughput        | Used by the analytical cost model.         |
| `mem_bw_GBs`                                  | Device memory bandwidth        | Required by the runtime model.             |
| `mem_capacity_GB`                             | Device memory capacity         | Used for feasibility and placement checks. |
| `arch` / `llmcompass_kind`                    | Optional architecture tag      | Useful with `npu_backend=llmcompass`.      |
| `llmcompass_device` / `llmcompass_device_name` / `llmcompass_arch` | LLMCompass device key | Use a built-in key such as `A100_80GB_fp16`, `TPUv3`, `MI210`, or `TPUv3_new`. |
| `cpu_read_latency_ns`, `cpu_write_latency_ns` | Host-access latency parameters | CPU-only.                                  |
| `cpu_cacheline_B` / `cpu_access_bytes_B`      | Host access granularity        | Optional.                                  |
| `pim_read_latency_ns`, `pim_write_latency_ns` | PIM local-memory latency       | PIM-only.                                  |
| `freq_ghz`                                    | PIM operating frequency        | Used by trace-mode flows.                  |
| `pim_memory`                                  | PIM address-map description    | Required for PIM capacity checks.          |

### PIM address map fields

Inside `pim_memory`:

| Key                             | Meaning                         | Notes |
| ------------------------------- | ------------------------------- | ----- |
| `addr_map_unit`                 | Address-map interpretation      | `bits` treats each field as address bits; other values are treated as counts. |
| `addr_map`                      | Address-map object              | Required by the current capacity checker. |
| `addr_map.row`                  | Row decomposition               | Also accepts `line`, `lines`, or `rows`. |
| `addr_map.channel`              | Channel decomposition           | Also accepts `channels`. |
| `addr_map.bank`                 | Bank decomposition              | Also accepts `banks`. |
| `addr_map.column`               | Column decomposition            | Also accepts `columns`. |
| `addr_map.offset`               | Offset decomposition            | Also accepts `offset_bits` or `offset_bytes`. |
| `capacity_bytes` / `capacity_B` | Explicit capacity override      | Only read inside `pim_memory`; device-level keys with these names are ignored. Current code still requires `addr_map` to exist as an object before applying this override. |


### Link fields

| Key                                        | Meaning                                            | Notes                        |
| ------------------------------------------ | -------------------------------------------------- | ---------------------------- |
| `a`, `b`                                   | Endpoint device names                              | Parsed symmetrically.        |
| `bw_GBs`                                   | Link bandwidth in GB/s                             | Required for explicit links. |
| `latency_s`, `latency_us`, `latency_ns`    | Fixed latency term                                 | Optional.                    |
| `overhead_s`, `overhead_us`, `overhead_ns` | Packet/message overhead                            | Optional.                    |
| `flit_size_B`                              | Packetization granularity                          | Optional.                    |
| `max_payload_B`                            | Payload threshold before extra headers/flits apply | Optional.                    |

### Practical notes

- Use a host CPU entry when modeling host-mediated routing or shared-memory behavior.
- For every PIM device, provide `pim_memory.addr_map` and make sure `mem_capacity_GB` matches the capacity implied by it.

---

## Evaluation config reference

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
  "ramulator_config_path": "./aim_simulator/example.yaml",
  "result_dir": "./output/evaluate_single_test/",
  "hardware_json": "./examples/hardware_1npu_2aim.json",
  "algo": ["bifocal"],
  "baselines": ["pd"],
  "tp_qkv": 2,
  "tp_ffn": 2,
  "npu_backend": "lut",
  "dump_graph": false,
  "dump_graph_dir": "./output/graph_dumps"
}
```

### Core keys

| Key             | Meaning                          | Notes                                                    |
| --------------- | -------------------------------- | -------------------------------------------------------- |
| `model_family`  | Model family name                | Example: `llama`, `qwen`, `mixtral`.                     |
| `model_variant` | Model-card variant               | Example: `7b`, `13b`, `70b`, `1.8b`, `8x7b`.             |
| `dtype`         | Logical datatype                 | `fp16` is the common default.                            |
| `batch`         | Batch size                       | Affects graph size and scheduling.                       |
| `prefill_len`   | Prefill sequence length          | Required for the workload.                               |
| `decode_len`    | Decode length                    | Required for the workload.                               |
| `result_dir`    | Base output directory            | DOPS creates run-specific folders inside it.             |
| `hardware_json` | Path to the hardware description | Usually under `src/examples/`.                           |
| `npu_backend`   | NPU latency backend              | Common values: `fast`, `fast_mode`, `lut`, `llmcompass`. |
| `debug`         | Verbose logging                  | Can be set in JSON or passed with `--debug`.             |

### Scheduler and baseline controls

| Key         | Meaning                | Notes                                                                 |
| ----------- | ---------------------- | --------------------------------------------------------------------- |
| `algo`      | Dynamic scheduler list | Typical values include `bifocal` and `heft`.                          |
| `baselines` | Static baseline list   | Common choices include `pd` and other built-in baselines in `main.py`. |

### Decode control

| Key                          | Meaning                                     | Notes                                        |
| ---------------------------- | ------------------------------------------- | -------------------------------------------- |
| `decode_plan_refresh_stride` | Re-run exact decode search every `N` tokens | Smaller values are more accurate but slower. |
| `decode_sample_stride`       | How densely decode schedules are exported   | Affects trace detail, not the simulated makespan. |

### Parallelism and graph-sharding keys

| Key      | Meaning                                             | Notes                                      |
| -------- | --------------------------------------------------- | ------------------------------------------ |
| `tp_qkv` | Tensor-parallel split for Q/K/V and attention heads | Must be compatible with model head counts. |
| `tp_ffn` | Tensor-parallel split for FFN                       | Must divide `ffn_dim`.                     |
| `tp_moe` | Expert-parallel split for MoE models                | Only meaningful for MoE.                   |

### Optional model override and annotation keys

| Key                                        | Meaning                          | Notes                                         |
| ------------------------------------------ | -------------------------------- | --------------------------------------------- |
| `shape_file`                               | Custom model-card JSON           | Use instead of packaged configs.              |
| `quantization`                             | Quantization annotation block    | Supports weight-only and W8A8-style modeling. |
| `weight_sparsity`                          | Weight sparsity annotation block | Adjusts bytes / FLOPs.                        |
| `activation_sparsity`                      | Activation sparsity block        | Can be phase-dependent.                       |
| `attention_sparsity`                       | Sparse-attention block           | Models reduced attention work.                |
| `optimizations` / `optimization` / `optim` | Alternative root object name     | The parser accepts all three names.           |

#### Optimization annotation schema

These annotations are parsed by `src/optimizations.py` and attached to graph nodes before cost modeling. You may place the sections at the top level (`quantization`, `weight_sparsity`, etc.) or under `optimizations`. For sparsity, the nested form `optimizations.sparsity.weight`, `optimizations.sparsity.activation`, and `optimizations.sparsity.attention` is also accepted.

**Quantization**

| Key | Meaning | Notes |
| --- | --- | --- |
| `enable` / `enabled` | Enables quantization annotations | Must be true and `mode` must not be `none`. |
| `mode` / `type` / `scheme` | Quantization mode | Examples: `weight_only`, `w8a8`, `w4a16`, `w4a8`. |
| `method` / `algorithm` | Algorithm label | Examples: `awq`, `gptq`, `smoothquant`; stored as metadata. |
| `weight_bits` / `w_bits` / `bits` | Weight bit width | Changes stored weight size. |
| `activation_bits` / `a_bits` | Activation bit width | Used when `activation_io` is `int8` or `int4`. |
| `activation_io` / `act_io` | Inter-op activation dtype | `fp16` keeps activation I/O unchanged; `int8`/`int4` tags activation and KV bytes. |
| `group_size` | Quantization group size | Adds scale overhead when `weight_bits < 16`. |
| `per_channel` / `per_row` | Per-channel metadata flag | Stored as metadata. |
| `scale_dtype_bits` / `scale_bits` | Scale dtype width | Defaults to 16. |
| `speedup` | Optional device-type speedup hint | Example: `{"npu": 1.15}`. |
| `apply_to`, `exclude` | Op-name substring filters | Empty `apply_to` means common linear weights. |

**Weight sparsity**

| Key | Meaning | Notes |
| --- | --- | --- |
| `enable` / `enabled` | Enables weight sparsity annotations | Requires positive sparsity or valid N:M pattern. |
| `method` / `mode` | Pruning method label | Examples: `magnitude`, `wanda`, `sparsegpt`, `global`, `layerwise`. |
| `pattern` / `scheme` / `type` | Sparsity pattern | `unstructured`, `block`, `n:m`; `2:4` is accepted as a shortcut for `n=2,m=4`. |
| `sparsity` / `ratio` | Zero fraction | Used for unstructured/block sparsity. |
| `n`, `m` | N:M structured sparsity | Density is `n/m`. |
| `storage` | Stored format assumption | `dense` keeps bytes unchanged; `compressed` scales bytes by density plus metadata. |
| `assume_sparse_compute` / `sparse_compute` | FLOP reduction switch | If true, compute cost uses sparse density. |
| `metadata_bytes_per_nnz` / `index_bytes` | Compressed metadata overhead | Extra bytes per nonzero. |
| `speedup`, `apply_to`, `exclude` | Same as quantization | Optional. |

**Activation and attention sparsity**

| Section | Important keys | Effect |
| --- | --- | --- |
| `activation_sparsity` or `optimizations.sparsity.activation` | `enable`, `method`/`mode`, `sparsity` or `density`, `density_by_phase`, `storage`, `assume_sparse_compute` | Scales activation bytes when compressed; scales compute only when `assume_sparse_compute=true`. |
| `attention_sparsity` or `optimizations.sparsity.attention` | `enable`, `pattern`, `window_left`, `window_right`, `block_size`, `blocks_left`, `blocks_right`, `density` | Reduces effective QK / Softmax / SV attention pairs for local, block, or sparse-matrix patterns. |

Minimal nested example:

```json
{
  "optimizations": {
    "quantization": {
      "enable": true,
      "mode": "w4a16",
      "method": "awq",
      "weight_bits": 4,
      "activation_io": "fp16",
      "group_size": 128
    },
    "sparsity": {
      "weight": {
        "enable": true,
        "method": "magnitude",
        "pattern": "2:4",
        "storage": "compressed",
        "assume_sparse_compute": true
      }
    }
  }
}
```

See `src/examples/evaluate_quant_sparse_config.json` for a complete runnable config.

### Optional export and debug keys

| Key              | Meaning                           | Notes                                           |
| ---------------- | --------------------------------- | ----------------------------------------------- |
| `dump_graph`     | Whether to export the constructed graph | Useful for debugging graph construction.     |
| `dump_graph_dir` | Output directory for dumped graphs | Used when `dump_graph=true`.                    |

### Runtime-overlap keys

| Key                                 | Meaning                                                                      | Notes                    |
| ----------------------------------- | ---------------------------------------------------------------------------- | ------------------------ |
| `pim_weight_load_overlap_ratio`     | Overlap between host-to-PIM transfer and local PIM formatting/programming    | Use a value in `[0, 1]`. |
| `weight_load_compute_overlap_ratio` | Overlap between weight loading and compute                                   | Use a value in `[0, 1]`. |

### Trace-mode / backend-specific paths

| Key                     | Meaning              | Notes                                               |
| ----------------------- | -------------------- | --------------------------------------------------- |
| `pim_config_path`       | PIM simulator config | Needed for trace-based PIM flows.                   |
| `ramulator_config_path` | Ramulator2 config    | Needed when using the Ramulator2-based PIM backend. |

---

## Weight-layout search options

These keys are mainly used by `weight-suggest`.

| Key                                | Meaning                                                      | Notes                                             |
| ---------------------------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| `format_block_layer_span`          | Group the same weight across every `N` layers into one block | `4` or `8` are common.                            |
| `format_block_change_percent`      | Maximum fraction of blocks allowed to change per outer iteration | Smaller values are more conservative.             |
| `format_outer_max_iters`           | Legacy compatibility knob                                    | Used when `format_block_change_percent` is unset. |
| `format_inner_max_blocks`          | Cap on inner refinement candidates                           | `0` means no cap.                                 |
| `format_nd_margin_init`            | Initial neutrality margin around the default dense layout    | Larger values keep more blocks in `Linear` early. |
| `format_nd_margin_decay`           | Per-iteration decay factor for the neutral band              | Controls aggressiveness over time.                |
| `format_nd_margin_min`             | Lower bound on the neutral band                              | Prevents over-splitting late in the search.       |
| `format_inner_improve_eps`         | Minimum improvement required to accept an inner flip         | Helps reject noise-level changes.                 |
| `format_outer_stop_eps`            | Early-stop tolerance between outer iterations                | Stops when improvement is too small.              |
| `format_reload_count_mode`         | How reload pressure is normalized                            | Use `raw`, `per_device`, or `soft_per_device`.    |
| `format_reload_device_count_alpha` | Soft normalization factor for `soft_per_device`              | Reduces device-count bias.                        |
