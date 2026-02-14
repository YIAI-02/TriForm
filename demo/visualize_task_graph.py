#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Render one graph to SVG (default):
python visualize_task_graph.py -i ./output/graph_dumps/*_full.json

# Render multiple graphs to a directory, output PNG:
python visualize_task_graph.py -i ../algorithms/output/evaluate_single_test/hardware_config_scale_down_12pima/graph_dumps/llama_7b_B1_S4096_T1024_int8_1770879232535227374_full.json -o ./viz --format png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _collect_inputs(inputs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for s in inputs:
        if not s:
            continue
        # Expand globs if user provides patterns
        if any(ch in s for ch in ['*', '?', '[']):
            for m in sorted(glob.glob(s)):
                p = Path(m)
                if p.is_file():
                    out.append(p)
        else:
            p = Path(s)
            if p.is_file():
                out.append(p)
    # De-dup while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in out:
        ap = str(p.resolve())
        if ap not in seen:
            seen.add(ap)
            uniq.append(p)
    return uniq


def _infer_edges(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    edges = payload.get('edges', None)
    if isinstance(edges, list) and edges and isinstance(edges[0], (list, tuple)):
        out: List[Tuple[str, str]] = []
        for e in edges:
            if not isinstance(e, (list, tuple)) or len(e) != 2:
                continue
            out.append((str(e[0]), str(e[1])))
        return out

    # Fallback: derive edges from per-node succ lists
    out2: List[Tuple[str, str]] = []
    nodes = payload.get('nodes', []) or []
    if isinstance(nodes, list):
        for n in nodes:
            if not isinstance(n, dict):
                continue
            u = str(n.get('id', ''))
            for v in (n.get('succ') or []):
                out2.append((u, str(v)))
    return out2


def _safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', s).strip('_')
    return s or 'graph'


def _build_node_label(
    nid: str,
    node: Dict[str, Any],
    mode: str,
    *,
    max_line: int = 120,
) -> str:
    name = str(node.get('name', '') or '')
    wid = node.get('weight_id', None)
    wsz = node.get('weight_size', None)

    def _clip(x: str) -> str:
        x = str(x)
        if max_line > 0 and len(x) > max_line:
            return x[: max_line - 1] + '…'
        return x

    mode = (mode or 'id_name').strip().lower()
    if mode == 'none':
        return ''
    if mode == 'id':
        return _clip(nid)
    if mode == 'name':
        return _clip(name or nid)
    if mode in ('id_name_weight', 'id_name_w', 'id_name_weights'):
        extra = ''
        if wid is not None:
            extra = f"\n{_clip(str(wid))} ({int(wsz or 0)} B)"
        if name:
            return _clip(f"{nid}\n{name}") + extra
        return _clip(nid) + extra

    # default: id + name
    if name:
        return _clip(f"{nid}\n{name}")
    return _clip(nid)


def render_graph_json(
    json_path: Path,
    *,
    out_dir: Path,
    fmt: str = 'svg',
    rankdir: str = 'LR',
    node_label: str = 'id_name',
    prog: str = 'dot',
    open_after: bool = False,
    write_dot: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Render a single dumped graph JSON to an image.

    Returns: (rendered_image_path, dot_path)
    """
    payload = _read_json(json_path)
    nodes = payload.get('nodes', []) or []
    edges = _infer_edges(payload)

    try:
        import pydot  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pydot is required for visualization. Install with: pip install pydot"
        ) from e

    dot = pydot.Dot(graph_type='digraph', rankdir=str(rankdir or 'LR'))

    # Nodes
    if not isinstance(nodes, list):
        nodes = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = str(n.get('id', ''))
        if not nid:
            continue
        label = _build_node_label(nid, n, node_label)
        dot.add_node(pydot.Node(nid, label=label, shape='box'))

    # Edges
    for u, v in edges:
        if not u or not v:
            continue
        dot.add_edge(pydot.Edge(str(u), str(v)))

    out_dir.mkdir(parents=True, exist_ok=True)
    base = _safe_filename(json_path.stem)

    dot_path: Optional[Path] = None
    if write_dot:
        dot_path = out_dir / f"{base}.dot"
        dot.write(str(dot_path), format='raw', prog=prog, encoding='utf-8')

    img_path = out_dir / f"{base}.{fmt}"
    dot.write(str(img_path), format=str(fmt), prog=prog)

    if open_after:
        try:
            webbrowser.open('file://' + str(img_path.resolve()))
        except Exception:
            pass

    return img_path, dot_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Visualize TaskGraph JSON dumps (nodes/edges).')
    p.add_argument(
        '-i', '--input', nargs='+', required=True,
        help='One or more *_full.json dump files. Globs are supported.'
    )
    p.add_argument(
        '-o', '--out-dir', default='./graph_viz',
        help='Output directory for rendered images (default: ./graph_viz).'
    )
    p.add_argument(
        '--format', default='svg',
        help='Output format: svg (default), png, pdf, ... (Graphviz-supported formats).'
    )
    p.add_argument(
        '--rankdir', default='LR',
        help='Graphviz rankdir: LR (default) or TB.'
    )
    p.add_argument(
        '--node-label', default='id_name',
        choices=['id_name', 'id', 'name', 'id_name_weight', 'none'],
        help='How to label nodes.'
    )
    p.add_argument(
        '--prog', default='dot',
        help='Graphviz program to use (default: dot).'
    )
    p.add_argument(
        '--open', action='store_true',
        help='Open the rendered file in your default browser/viewer.'
    )
    p.add_argument(
        '--write-dot', action='store_true',
        help='Also write the intermediate .dot file.'
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    inputs = _collect_inputs(args.input)
    if not inputs:
        print('[viz] No valid input files found.', file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    fmt = str(args.format or 'svg').strip().lower()

    print(f"[viz] inputs: {len(inputs)} file(s)")
    print(f"[viz] out_dir: {out_dir}")
    print(f"[viz] format: {fmt} | rankdir: {args.rankdir} | node_label: {args.node_label}")

    for jpath in inputs:
        try:
            img_path, dot_path = render_graph_json(
                jpath,
                out_dir=out_dir,
                fmt=fmt,
                rankdir=args.rankdir,
                node_label=args.node_label,
                prog=args.prog,
                open_after=bool(args.open),
                write_dot=bool(args.write_dot),
            )
            rel = os.path.relpath(str(img_path), os.getcwd()) if img_path else str(img_path)
            print(f"[viz] OK: {jpath} -> {rel}")
            if dot_path is not None:
                rel2 = os.path.relpath(str(dot_path), os.getcwd())
                print(f"      dot: {rel2}")
        except Exception as e:
            print(f"[viz] FAIL: {jpath}: {e}", file=sys.stderr)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
