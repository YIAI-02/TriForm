# TriForm

## Mesurements
### Global Buffer
#### End-to-End Workflow
```
configs ──▶ 01_generate_traces.py ──▶ <weight traces>
                │
                ▼
            02_run_ramulator.py   ──▶ <Ramulator statistics / logs>
                │
                ▼
             03_parse_and_fit     ──▶ <linear-fit params for algorithms>
```

- 01_generate_traces.py

    Input: configs/ (model params).

    Output: DRAM traces that represent ND-format weight accesses.
- 02_run_ramulator.py

    Runs Ramulator on the generated traces and exports timing/latency stats.

- 03_parse_and_fit

    Parses the simulation outputs and fits a linear model used by the algorithm side.

#### One-Command Runner

Use ./run_dram.sh at repo root to chain the full pipeline (01→03). It accepts flags to override sensible defaults.
Flags

- `-m` — model(s) to run (comma or space separated).

- `-d` — DRAM type (as expected by your Ramulator config).

- `-b` — path to Ramulator (binary or root, depending on your script’s logic).

The script also honors these environment variables (with defaults):
MODELS (default: mpt)
- DRAM (default: DDR3)
- RAMULATOR_BIN (default: ./ramulator, relative to measurements/dram/)
- OUT_ND_DIR (default: ./out_nd)
- RUN_OUT_DIR (default: ./out_runs)

Examples
```shell
# Full run: choose model, DRAM type, 8-way parallel, and point to Ramulator
./run_dram.sh -m mpt -d DDR3 -b /path/to/ramulator

# Multiple models, different DRAM type
./run_dram.sh -m "mistral,qwen2" -d HBM  -b /path/to/ramulator
```
Make sure Ramulator is compiled and the -b path is correct.

#### Outputs

- Traces (from step 01): ND-format weight access traces for DRAM simulation.
- Simulation stats/logs (from step 02): latency and other DRAM metrics from Ramulator.
- Linear fit results (from step 03): coefficients/params directly consumable by algorithms.