# DOPS -> Het-Infer static prior contract v1

`dops.hetinfer_prior.v1` is the versioned, score-free file boundary between
the DOPS offline expert and the Het-Infer online scheduler. The same v1 name
replaces the retired score-bearing payload in place; there is no v2 alias and
no compatibility loader for the old format.

The artifact exports one fully instantiated graph/workload and contains:

1. the exact final `expert_placement` selected by DOPS;
2. `t_service` for every legal operator/device candidate;
3. `t_move` for every declared legal movement route;
4. graph, input-residency, and collective-context manifests that make the
   domains of those three tables independently checkable.

The export path observes a completed schedule. It does not rescore, replace,
or otherwise change the expert placement, and the file contains no Value or
trainer state.

## Identity, units, and strictness

- `schema` is exactly `dops.hetinfer_prior.v1`.
- `schema_version` is exactly `1`; both version fields are checked together.
- `graph_id` and `workload_id` identify the instantiated graph/workload.
- `time_unit` is exactly `seconds`.
- Every time field is named `duration_s`, finite, and non-negative.
- Unknown fields, duplicate JSON keys, and non-standard numeric literals are
  rejected.
- Arrays whose meaning is unordered are sorted by the canonical writer.

In particular, `eft_s`, `window_s`, `reload_s`, `comm_s`, composite DOPS
scores, Value estimates, and trainer state cannot appear in service or
movement records.

## Operators and placement

Each `operators[]` record binds one stable `op_id` to its dependency IDs and
complete `legal_devices` set. `expert_placement[]` must cover every operator
exactly once and select a legal device.

A collective is represented as one atomic operator. Its public
`legal_devices` set contains exactly its `canonical_device_id`, and its expert
placement must equal that canonical device. The participant and resource sets
are described separately by `collective_contexts[]`; they are not fabricated
operator-placement candidates.

## Input manifest

Every declared operator dependency has exactly one matching `inputs[]` record.
An external request or KV input uses `producer_op_id: null`. The stable input
key is:

```text
(consumer_op_id, producer_op_id, tensor_id)
```

The same `tensor_id` must bind one producer and byte count throughout the
artifact. Each source residency binds a physical `device_id` and the tensor's
source `layout`. `destination_devices` states where that input is permitted or
required to arrive. Exactly three semantics are accepted:

### `data`

- `source_residencies` and `destination_devices` are non-empty.
- `destination_devices` equals the consumer's complete `legal_devices` set.
- For every source residency and every destination, the corresponding route
  must appear in `legal_movement_routes` and `t_move`.
- A collective dependency may not be labeled `data`.

### `barrier`

- The producer must be a declared dependency.
- `bytes` is zero.
- `source_residencies` and `destination_devices` are both empty.
- It is dependency ordering only, so it creates no movement route.

### `collective_staging`

- The producer must be a declared dependency of a collective operator.
- It has non-empty source residencies and exactly one fixed staging
  destination.
- That destination is a declared participant of the collective.
- The route closure from every possible source residency to the fixed staging
  destination must be present in `legal_movement_routes` and `t_move`.
- Across all staging inputs of a collective, the set of fixed destinations
  must equal `participant_device_ids` exactly.

This separation is important: movement from an online tensor residency to its
fixed collective staging device is `T_move`; communication internal to the
collective is not another public movement edge.

## Collective context and no-double-count rule

Every collective has one exact `collective_contexts[]` record containing its
primitive, topology, canonical device, participant devices, output devices,
resource devices, and tensor byte count. All referenced devices must be in
the artifact; every participant and output device must also be in
`resource_device_ids`; the output set includes the canonical device.

`internal_transport` is a required constant:

```text
included_in_t_service
```

It states that `t_service` for the atomic collective already includes its
internal ring/reduce/scatter/transfer communication. Exporters and consumers
must not emit or charge a second `T_move` for that internal transport. Only the
explicit `collective_staging` routes described above are movement-table
lookups.

## Service table

The unique key of `t_service[]` is:

```text
(op_id, device_id)
```

Its key set equals the Cartesian expansion of each operator's
`legal_devices`; no unselected legal candidate may be omitted. A collective
has one service entry on its canonical device and that duration is the atomic,
context-bound collective cost.

## Movement table

The unique key shared by `legal_movement_routes[]` and `t_move[]` is:

```text
(tensor_id, source_device_id, destination_device_id, bytes, layout)
```

The two complete key sets must be equal and both devices must be declared.
`layout` is a case-sensitive identifier for the tensor at its source; the
duration may include the required conversion. A resident route has equal
source and destination and must have `duration_s = 0`.

For every non-barrier input, the manifest must contain at least the complete
Cartesian closure:

```text
source_residencies x destination_devices
```

The route manifest may additionally contain explicit legal spill/store routes,
but `t_move` must still cover every declared route exactly once.

## Shared files

The canonical machine-readable schema is
`schemas/dops.hetinfer_prior.v1.schema.json`. The shared fixture is
`tests/fixtures/dops_hetinfer_prior_v1_minimal.json`; it contains normal data,
a zero-byte barrier, and a two-participant collective whose inputs are staged
on fixed GPU/PIM participants. DOPS and Het-Infer carry byte-identical schema,
fixture, contract prose, and equivalent strict readers/tests.
