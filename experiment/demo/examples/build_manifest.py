from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"

def safe_id(path: str) -> str:
    text = path.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

examples = []

for case_dir in sorted([p for p in ROOT.rglob("*") if p.is_dir()]):
    pd_files = sorted(case_dir.glob("PD*_ops_trace.csv"))
    bifocal_files = sorted(case_dir.glob("Bifocal*_ops_trace.csv"))

    if not pd_files or not bifocal_files:
        continue

    rel_dir = case_dir.relative_to(ROOT).as_posix()
    entry_id = safe_id(rel_dir)

    examples.append({
        "id": entry_id,
        "label": rel_dir.replace("/", " | "),
        "baseline": f"examples/{rel_dir}/{pd_files[0].name}",
        "heuristic": f"examples/{rel_dir}/{bifocal_files[0].name}",
        "notes": "Auto-generated demo example entry."
    })

manifest = {
    "version": 1,
    "examples": examples
}

MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Wrote {MANIFEST} with {len(examples)} examples.")