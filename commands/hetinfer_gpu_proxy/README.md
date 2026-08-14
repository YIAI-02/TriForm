# Het-Infer static-prior export from DOPS

This directory previously contained a score-profile grid and manifest builder.
That workflow is retired. The public schema name is still
`dops.hetinfer_prior.v1`, but it now names one strict static artifact with
exactly three complete tables:

1. `expert_placement` from the completed Bifocal placement;
2. `t_service` for every legal operator-device pair, including unselected
   candidates;
3. `t_move` for every declared legal movement route, including resident
   self-routes.

The artifact also carries strict `inputs` and `collective_contexts` execution
manifests so a consumer can bind tensor residency, fixed collective staging,
and atomic internal transport without inventing a fourth timing table.

The retired score-bearing v1 payload, profile-grid manifest, source sidecar,
and `DQN_PRIOR` bundle are not compatible with this contract. There is no v2
alias and no compatibility loader.

## Evidence boundary

DOPS performs placement and local cost-model evaluation. Export is a pure
post-placement projection and does not change the selected devices. It does
not invoke a Value model, trainer, Het-Infer runtime, or online ATLAS
simulation. `GPU0` may still be represented by DOPS' `npu` device type while
retaining the exported device id `GPU0`.

The artifact is an offline cost contract. It does not prove physical GPU/PIM
execution, tensor correctness, overlap, end-to-end inference latency, or
speedup.

## Two-stage ATLAS workflow

Run all Python work inside a Slurm compute allocation. The login node is only
for inspection and job submission.

Use the exact same config, workload overrides, scheduler seed, and Bifocal
algorithm in both DOPS runs. First export the exact PIM-dependent service and
movement keys that require offline ATLAS measurements:

```bash
python3 src/main.py evaluate \
  --config /absolute/path/to/evaluate.json \
  --algo Bifocal \
  --scheduler_seed 0 \
  --hetinfer-atlas-manifest-out /absolute/path/to/atlas_request.json
```

The request schema is `dops.hetinfer_atlas_timing_request.v1`. It carries the
derived `graph_id` and `workload_id`, a timing-context SHA-256 bound to the
complete snapshot/config/input-file bytes, and exact descriptor-bearing
`service` and `movement` arrays. Run ATLAS offline for those keys and create a strict
`dops.hetinfer_atlas_timings.v1` JSON with the same identity and key fields.
Each result entry adds only:

```json
{
  "cycles": 250,
  "frequency_MHz": 500
}
```

DOPS converts each result to seconds using
`cycles / (frequency_MHz * 1_000_000)`. Missing, duplicate, extra, or
identity-mismatched entries are rejected. ATLAS is never recomputed during
prior export.

Then rerun the same schedule with the completed timings and write the static
prior:

```bash
python3 src/main.py evaluate \
  --config /absolute/path/to/evaluate.json \
  --algo Bifocal \
  --scheduler_seed 0 \
  --hetinfer-atlas-timings /absolute/path/to/atlas_timings.json \
  --hetinfer-prior-out /absolute/path/to/dops_hetinfer_prior.json
```

If a schedule has no ATLAS-marked PIM service or movement key, the timings
file is optional. A `.json` output argument is the exact output file; a
directory argument receives an automatic workload filename. Repeating either
command with the same output path atomically replaces the previous complete
file, which is the supported same-name v1 update behavior.

The exact schema, completeness rules, and validation boundary are documented
in `docs/HETINFER_PRIOR_CONTRACT.md`.
