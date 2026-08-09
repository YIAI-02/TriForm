"""Shared, dependency-free helpers for the GPU calibration tools."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

RAW_SCHEMA = "dops.gpu_microbench.raw.v1"
FIT_SCHEMA = "dops.gpu_calibration.fit.v1"
RUNTIME_MODEL_SCHEMA = "dops.gpu_runtime_model.v1"


def load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_positive(values: Iterable[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0.0:
            output.append(number)
    return output


def quantile(values: Iterable[Any], probability: float) -> float:
    ordered = sorted(finite_positive(values))
    if not ordered:
        raise ValueError("quantile requires at least one finite positive value")
    p = min(1.0, max(0.0, float(probability)))
    if len(ordered) == 1:
        return ordered[0]
    position = p * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summary(values: Iterable[Any]) -> dict[str, float | int]:
    cleaned = finite_positive(values)
    if not cleaned:
        raise ValueError("summary requires at least one finite positive value")
    return {
        "count": len(cleaned),
        "min": min(cleaned),
        "p10": quantile(cleaned, 0.10),
        "p25": quantile(cleaned, 0.25),
        "median": quantile(cleaned, 0.50),
        "p75": quantile(cleaned, 0.75),
        "p90": quantile(cleaned, 0.90),
        "max": max(cleaned),
    }


def ordinary_least_squares(
    points: Iterable[tuple[float, float]],
) -> dict[str, float | int]:
    clean = [
        (float(x), float(y))
        for x, y in points
        if math.isfinite(float(x))
        and math.isfinite(float(y))
        and float(x) >= 0.0
        and float(y) > 0.0
    ]
    if len(clean) < 2:
        raise ValueError("linear fit requires at least two finite points")
    count = len(clean)
    mean_x = sum(x for x, _ in clean) / count
    mean_y = sum(y for _, y in clean) / count
    variance_x = sum((x - mean_x) ** 2 for x, _ in clean)
    if variance_x <= 0.0:
        raise ValueError("linear fit x values must not all be equal")
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in clean)
    slope = covariance / variance_x
    intercept = mean_y - slope * mean_x
    predictions = [intercept + slope * x for x, _ in clean]
    residual = sum(
        (y - prediction) ** 2 for (_, y), prediction in zip(clean, predictions)
    )
    total = sum((y - mean_y) ** 2 for _, y in clean)
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return {
        "count": count,
        "intercept": intercept,
        "slope": slope,
        "r_squared": r_squared,
    }
