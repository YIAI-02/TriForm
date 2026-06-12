# Demo Trace Examples

This directory is the recommended storage location for reviewer-facing browser-demo trace examples. Demo can be simply opens local CSV files selected through the two file pickers in `experiment/demo/index.html`.

## Directory layout

Add each demo case as one subdirectory:

```text
experiment/demo/examples/
└── <case_id>/
    ├── <baseline>_ops_trace.csv
    └── <Bifocal>_ops_trace.csv
```

The browser timeline needs the operator trace CSV files exported by `evaluate`, i.e., files ending with `_ops_trace.csv`. The corresponding `_comms_trace.csv` files are optional for the browser demo, although they may still be useful for offline plotting scripts.

A typical case can be stored as:

```text
experiment/demo/examples/qwen7b_fp16_b16_p512_d512/
├── PD_linear_prefill-512xdecode_512_ops_trace.csv
└── Bifocal_linear_prefill-512xdecode_512_ops_trace.csv
```

For a single-policy timeline check, select the same `_ops_trace.csv` file as both the baseline and Bifocal trace.

## Opening the demo

Open this file directly in a browser:

```text
experiment/demo/index.html
```

Then select the two local CSV traces from the left control panel:

1. **Baseline trace (CSV)**: choose the baseline policy trace, such as the PD trace.
2. **Bifocal trace (CSV)**: choose the Bifocal to compare against the baseline.

