# Demo Trace Examples

This directory is the canonical place for browser-demo trace examples used to replay the scheduling timeline in `experiment/demo/index.html`.

## Directory layout

Add each demo case as one subdirectory:

```text
experiment/demo/examples/
├── manifest.json
└── <case_id>/
    ├── <baseline>_ops_trace.csv
    └── <heuristic>_ops_trace.csv
    └── ...
```

The browser timeline needs the operator trace CSV files exported by `evaluate`, i.e., files ending with `_ops_trace.csv`. The corresponding `_comms_trace.csv` files are optional for the browser demo, although they may still be useful for offline plotting scripts.

The current browser view is a side-by-side comparison. If a reviewer-facing case contains only one policy trace, register the same `_ops_trace.csv` path as both `baseline` and `heuristic` so the timeline can still be opened.

A typical case can be stored as:

```text
experiment/demo/examples/qwen_7b_b4_128x512/
├── PD_linear_prefill-128xdecode_512_ops_trace.csv
└── Bifocal_linear_prefill-128xdecode_512_ops_trace.csv
└── ...
```

## Registering a case in `manifest.json`

After copying the CSV files, add an entry to `manifest.json`:

```json
{
  "version": 1,
  "examples": [
    {
      "id": "qwen_7b_b4_128x512",
      "label": "Qwen-7B, B=4, prefill=128, decode=512, PD vs Bifocal",
      "model": "auto",
      "batch": 4,
      "prefill": 128,
      "decode": 512,
      "dtype": "fp16",
      "baseline": "examples/qwen_7b_b4_128x512/PD_linear_prefill-128xdecode_512_ops_trace.csv",
      "heuristic": "examples/qwen_7b_b4_128x512/Bifocal_linear_prefill-128xdecode_512_ops_trace.csv",
      "notes": "Optional short description shown in the demo panel."
    }
  ]
}
```

Paths in `manifest.json` are resolved relative to `experiment/demo/index.html`, so they should usually start with `examples/...`.

## Opening the demo

From the repository root:

```bash
cd experiment/demo
python server.py
```

Then open `http://localhost:8015/index.html`. The Bundled example dropdown is populated from `examples/manifest.json`. Select a registered case to load the baseline and heuristic traces automatically.

You can also skip `manifest.json` and load two local `_ops_trace.csv` files manually with the Baseline trace (CSV) and Heuristic trace (CSV) file pickers.


