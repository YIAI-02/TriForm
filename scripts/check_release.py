#!/usr/bin/env python3
"""Fail closed when a generated release tree contains common packaging hazards."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REQUIRED_FILES = (
    "ARTIFACT_EVALUATION.md",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements-core.txt",
    "ae/run_smoke.sh",
    "ae/smoke_config.json",
    "ae/verify_smoke.py",
)

PROHIBITED_PARTS = {".git", ".vscode", "__pycache__", "build", "kernel_meta", "profile"}
PROHIBITED_SUFFIXES = {".bak", ".log", ".o", ".pkl", ".pyc", ".so"}
BINARY_MAGIC_PREFIXES = (
    b"\x7fELF",
    b"MZ",
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
)
TEXT_PATTERNS = {
    # Split the literals so this checker does not flag its own signatures.
    "private HPC path": re.compile("/" + r"lustre/home/"),
    "private macOS path": re.compile("/" + r"Users/"),
    "private key": re.compile("BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "Wi-Fi credential command": re.compile("nm" + r"cli[^\n]*(?:password|psk)", re.IGNORECASE),
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_release.py RELEASE_ROOT")
    root = Path(sys.argv[1]).resolve()
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (root / rel).is_file():
            errors.append(f"missing required file: {rel}")

    license_path = root / "LICENSE"
    if license_path.is_file() and license_path.stat().st_size < 200:
        errors.append("LICENSE is unexpectedly short")

    notices_path = root / "THIRD_PARTY_NOTICES.md"
    if notices_path.is_file():
        notices = notices_path.read_text(encoding="utf-8", errors="replace").lower()
        if ("revision " + "pending") in notices:
            errors.append("THIRD_PARTY_NOTICES.md contains an unresolved revision")

    citation_path = root / "CITATION.cff"
    if citation_path.is_file():
        citation = citation_path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"^license:\s*[A-Za-z0-9.+-]+\s*$", citation, re.MULTILINE):
            errors.append("CITATION.cff is missing a root SPDX license identifier")

    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in PROHIBITED_PARTS for part in rel.parts):
            errors.append(f"prohibited generated/private path: {rel}")
            continue
        if path.is_file() and path.suffix.lower() in PROHIBITED_SUFFIXES:
            errors.append(f"prohibited generated file: {rel}")
            continue
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                header = handle.read(4)
        except OSError:
            header = b""
        if any(header.startswith(magic) for magic in BINARY_MAGIC_PREFIXES):
            errors.append(f"prebuilt executable or library: {rel}")
            continue
        if path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in TEXT_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label}: {rel}")

    if errors:
        print("Release validation failed:", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Release validation passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
