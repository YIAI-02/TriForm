# MICRO 2026 Artifact Evaluation Guide

This guide covers the reviewer-facing **Artifacts Evaluated - Functional**
workflow for DOPS. The submission also applies for **Artifact Available**; the
final Zenodo DOI must be added to HotCRP after the release archive is frozen.
The submission does not currently apply for **Results Reproduced**.

## Artifact inventory

- `src/`: DOPS model, hardware, cost-model, scheduling, and trace-export code.
- `configs/`: model shape cards.
- `ae/`: small analytical smoke test and automatic verifier.
- `commands/`: full evaluation and sweep launchers.
- `experiment/`: analysis, plotting, and trace-visualization utilities.
- `measurements/`: optional measured-backend and microbenchmark sources.
- `submodules/`: partial optional third-party source snapshots; see
  `THIRD_PARTY_NOTICES.md`.

## Required functional workflow

### Hardware

A Linux x86-64 workstation or server is recommended, with at least 4 CPU cores,
8 GB RAM, and 2 GB free disk. The analytical workflow does not require a GPU,
NPU, PIM device, or proprietary toolchain.

### Software

- Bash
- CPython 3.10 or later
- packages in `requirements-core.txt`

Create an isolated environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
```

### Run and validate

```bash
bash ae/run_smoke.sh
```

The script explicitly selects the analytical NPU and PIM backends. It runs a
small Qwen-1.8B workload with the PD baseline, a fixed Bifocal tie-breaking
seed, and the Bifocal scheduler, then checks
that both policies produced:

- a finite, positive prefill/decode/total latency;
- one `best_summary_*.json` file;
- one operator trace CSV; and
- one communication trace CSV.

Success ends with:

```text
[AE] PASS: analytical fast-mode smoke test produced valid PD and Bifocal artifacts
```

The default output root is `output/ae_smoke/`. The script refuses to overwrite
an existing output directory. Set `OUTPUT_ROOT=/new/path` to run again.

## Longer example

The following uses Qwen-7B, batch 4, prefill 128, decode 512, and scheduling
stride 2. It also explicitly defaults to the analytical fast backends:

```bash
bash commands/command_single_evaluate.sh
```

Expected outputs include a combined comparison JSON, PD/Bifocal summary JSON
files, and operator/communication traces. Runtime depends on the host; the
authors observed approximately seven minutes on their evaluation system.

## Optional backends

The following are not required for the Functional-badge smoke test:

- Ramulator2/CENT for trace-based PIM simulation;
- LLMCompass for an alternate NPU model;
- Huawei CANN, Ascend-C, and `msprof` for hardware measurements; and
- the scientific Python plotting stack for paper figures.

These optional paths have additional platform requirements and are documented
in the main README and their respective directories. The partial CENT and
LLMCompass snapshots are not standalone installations; obtain their missing
nested dependencies as described in `THIRD_PARTY_NOTICES.md`.

## Known scope

This Functional submission validates installation, configuration parsing,
execution of PD and Bifocal, and artifact generation. It does not claim that the
small smoke workload reproduces every numerical result in the paper.

## Zenodo release procedure

1. Confirm that the root `LICENSE` and `THIRD_PARTY_NOTICES.md` match the
   intended release terms and bundled snapshots.
2. Run the smoke test in the final clean environment.
3. Commit all release changes and ensure `git status --short` is empty.
4. Run `bash scripts/make_zenodo_archive.sh`.
5. Upload the generated archive and `SHA256SUMS` to Zenodo.
6. Publish the Zenodo record, then add its DOI URL to HotCRP.

Do not archive an arbitrary working directory or a moving branch. Archive the
exact reviewed commit and record its commit SHA in the Zenodo description.
