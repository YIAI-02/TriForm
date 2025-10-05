from __future__ import annotations
from pathlib import Path
from typing import List, Union

TextOrLines = Union[str, List[str], None]

def _to_text(ret: TextOrLines) -> str:
    if ret is None:
        return ""
    if isinstance(ret, str):
        return ret if ret.endswith("\n") else ret + "\n"
    return "".join([line if line.endswith("\n") else (line + "\n") for line in ret])

def concat_trace(pieces: List[TextOrLines]) -> str:
    return "".join([_to_text(piece) for piece in pieces])

def write_trace(ret: TextOrLines, out_path: str | Path) -> str:
    text = _to_text(ret)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    return str(out_path)