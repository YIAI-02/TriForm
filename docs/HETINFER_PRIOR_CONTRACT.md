# DOPS to Het-Infer placement-prior contract

`dops.hetinfer_prior.v1` is the versioned boundary between the offline DOPS
planner and the online Het-Infer runtime. DOPS supplies a baseline placement,
the legal action set, and the score terms that produced that baseline.
Het-Infer still makes the final runtime device decision.

The artifact is intentionally separate from the existing AE
`best_summary_*.json`. Existing timelines and optimized-layout outputs are not
renamed or removed.

## Produce a scored prior

Run Bifocal through the normal evaluate path and request the additional file:

```bash
python src/main.py evaluate \
  --config src/examples/evaluate_test_config.json \
  --algo Bifocal \
  --hetinfer-prior-out output/hetinfer_prior.json
```

The hook captures the candidate set at the exact point where Bifocal computes
its placement score. It does not re-run the scheduler after the fact. A
canonical workload profile contains both phases:

```text
profiles[].workload
profiles[].phases.prefill.operators[]
profiles[].phases.decode.operators[]
```

For `weight-suggest`, the same flag is supported with `--algo Bifocal`. After
the layout search chooses `best_formats`, DOPS performs one explicit Bifocal
evaluation of that exact map and exports those candidate scores. This avoids
mixing scores from a rejected layout into the runtime prior.

When DOPS refreshes the decode plan at more than one context length, the
exporter emits one profile per captured decode context and pairs each with the
same workload's prefill capture. Het-Infer may select/interpolate profiles, but
must never add a device that is absent from `legal_devices`.

## Candidate metrics

All time fields use seconds and the `_s` suffix.

- `dops_score_s`: the actual Bifocal ranking value.
- `eft_s`: candidate earliest-finish time in DOPS's current simulated state.
- `window_s`: Bifocal lookahead-window completion estimate.
- `compute_s`: operator compute component reported by the same cost-model
  evaluation. It is `null` when the specialized primitive path does not expose
  a separable compute component.
- `reload_s`: end-to-end weight reload service from the weight-service profile,
  including its configured internal overlap. It is not recomputed as
  `eft-compute`.
- `comm_s`: critical-path delay before candidate start caused by dependency/KV
  movement, relative to predecessor and device readiness. It is not the sum of
  every link reservation and can be zero when transfers are hidden.
- `weight_reuse_bias_s`: Bifocal's short-horizon residency/consistency bias.
- `decode_amort_bias_s`: Bifocal's cross-token decode amortization bias.

The exact score is:

```text
(1 - gamma) * eft_s
+ gamma * window_s
+ weight_reuse_bias_s
+ decode_amort_bias_s
```

`dynamic_eligible` is true only when DOPS exposed more than one legal concrete
device and the node is neither a pinned KV write nor a communication primitive.
This is an offline eligibility hint; Het-Infer must apply the legal-device mask
again before every DQN action.

## Provenance

Each artifact records the full normalized graph and hardware snapshots plus
their SHA-256 digests, the complete input config snapshot, model family and
revision, TP/PP/EP fields, DOPS git revision, policy, and source-capture digest.
Every exported prior also writes an adjacent canonical source sidecar named
`<artifact_id>.source.json`. `provenance.source_artifact_path` is that relative
filename and `source_artifact_sha256` is the SHA-256 of the sidecar's exact
bytes. The bundle validator rejects a missing, modified, or non-canonically
encoded sidecar. SHA-256 fields are canonical lowercase 64-hex strings;
`digest_algorithm` is stored separately where applicable.

The prior and source sidecar are one indivisible handoff bundle. When staging
a prior for Het-Infer DQN training, copy both files into the same destination
directory without renaming the sidecar; then point `DQN_PRIOR` or
`--dops-prior` at the copied prior JSON. Copying only `prior.json` is expected
to fail closed in Het-Infer. Grid manifests expose the exact pair in each
entry's `bundle_files` list and separately record both SHA-256 values.

`provenance.status=partial` and `missing_fields` are explicit. Consumers that
require reproducible graph/hardware identity should reject a partial artifact.

## Import an old best summary

The compatibility tool preserves an old baseline without fabricating scores:

```bash
python src/export_hetinfer_prior.py \
  --best-summary output/best_summary.json \
  --config path/to/original_config.json \
  --output output/legacy_hetinfer_prior.json
```

Legacy candidates contain JSON `null` for unavailable metrics,
`dynamic_eligible=false`, and only the historically selected device in the
legal set. Such an artifact can replay or audit the baseline, but it is not
DQN-bootstrap training data. Use the live Bifocal hook for scored alternatives.

The machine-readable schema is
`schemas/dops.hetinfer_prior.v1.schema.json`. The Python validator additionally
checks cross-field invariants: unique profiles/operators, baseline membership,
exact equality between candidate keys and legal devices, and the adjacent
source sidecar's byte-level digest.
