from __future__ import annotations

import csv
import gzip
import shutil
from collections import defaultdict, deque
from pathlib import Path
from random import Random
from typing import Iterable

from ragebait_detector.utils.io import ensure_parent

DEFAULT_CONFIDENCE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)
DEFAULT_LABEL_RATIOS = {0: 0.5, 1: 0.5}


def gzip_path_for(path: str | Path) -> Path:
    csv_path = Path(path)
    return csv_path.with_suffix(f"{csv_path.suffix}.gz")


def ensure_csv_available(path: str | Path) -> Path:
    csv_path = Path(path)
    if csv_path.exists():
        return csv_path
    compressed_path = gzip_path_for(csv_path)
    if not compressed_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path} or compressed variant {compressed_path}"
        )
    ensure_parent(csv_path)
    with gzip.open(compressed_path, "rb") as src, csv_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return csv_path


def compress_csv(path: str | Path, *, keep_original: bool = False) -> Path:
    csv_path = ensure_csv_available(path)
    compressed_path = gzip_path_for(csv_path)
    ensure_parent(compressed_path)
    with csv_path.open("rb") as src, gzip.open(compressed_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    if not keep_original:
        csv_path.unlink()
    return compressed_path


def parse_label(row: dict[str, str]) -> int | None:
    label_value = (row.get("label") or "").strip().lower()
    if label_value in {"0", "1"}:
        return int(label_value)
    bool_value = (row.get("is_ragebait") or "").strip().lower()
    if bool_value in {"true", "false"}:
        return 1 if bool_value == "true" else 0
    return None


def parse_confidence(row: dict[str, str]) -> float | None:
    value = (row.get("confidence") or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def is_ok_row(row: dict[str, str]) -> bool:
    status = (row.get("labeling_status") or "").strip().lower()
    return not status or status == "ok"


def threshold_tag(threshold: float) -> str:
    percent = f"{threshold * 100:.2f}".rstrip("0").rstrip(".")
    return percent.replace(".", "_")


def parse_label_ratio(value: str) -> dict[int, float]:
    parts = [part.strip().rstrip("%") for part in value.split("/")]
    if len(parts) != 2:
        raise ValueError("label ratio must look like 50/50 or 60/40 in label0/label1 order")
    try:
        raw_values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("label ratio values must be numeric") from exc
    if any(part < 0 for part in raw_values):
        raise ValueError("label ratio values must be non-negative")
    total = sum(raw_values)
    if total <= 0:
        raise ValueError("label ratio values must sum to more than zero")
    return {0: raw_values[0] / total, 1: raw_values[1] / total}


def label_ratio_tag(label_ratios: dict[int, float]) -> str:
    left = _ratio_component(label_ratios.get(0, 0.0))
    right = _ratio_component(label_ratios.get(1, 0.0))
    return f"r{left}_{right}"


def analyze_labeled_rows(
    rows: Iterable[dict[str, str]], thresholds: Iterable[float]
) -> dict[str, object]:
    total_rows = 0
    valid_rows = 0
    ragebait_rows = 0
    threshold_stats = {
        threshold: {"rows": 0, "ragebait": 0} for threshold in sorted(set(thresholds))
    }

    for row in rows:
        total_rows += 1
        if not is_ok_row(row):
            continue
        label = parse_label(row)
        confidence = parse_confidence(row)
        if label is None or confidence is None:
            continue
        valid_rows += 1
        ragebait_rows += label
        for threshold, stats in threshold_stats.items():
            if confidence >= threshold:
                stats["rows"] += 1
                stats["ragebait"] += label

    return {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "ragebait_rows": ragebait_rows,
        "ragebait_ratio": (ragebait_rows / valid_rows) if valid_rows else 0.0,
        "thresholds": [
            {
                "threshold": threshold,
                "rows": stats["rows"],
                "ragebait_rows": stats["ragebait"],
                "ragebait_ratio": (
                    stats["ragebait"] / stats["rows"] if stats["rows"] else 0.0
                ),
                "coverage_ratio": (stats["rows"] / valid_rows) if valid_rows else 0.0,
            }
            for threshold, stats in threshold_stats.items()
        ],
    }


def load_filtered_rows(
    path: str | Path, *, confidence_threshold: float
) -> tuple[list[str], list[dict[str, str]], dict[str, int]]:
    csv_path = ensure_csv_available(path)
    filtered_rows: list[dict[str, str]] = []
    skipped = {"invalid_status": 0, "invalid_label": 0, "low_confidence": 0}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if not is_ok_row(row):
                skipped["invalid_status"] += 1
                continue
            label = parse_label(row)
            confidence = parse_confidence(row)
            if label is None or confidence is None:
                skipped["invalid_label"] += 1
                continue
            if confidence < confidence_threshold:
                skipped["low_confidence"] += 1
                continue
            row["_normalized_label"] = str(label)
            row["_normalized_confidence"] = f"{confidence:.4f}"
            filtered_rows.append(row)
    return fieldnames, filtered_rows, skipped


def balance_rows(
    rows: Iterable[dict[str, str]],
    *,
    limit: int,
    seed: int,
    label_ratios: dict[int, float] | None = None,
) -> tuple[list[dict[str, str]], dict[str, int | float]]:
    rng = Random(seed)
    normalized_ratios = dict(DEFAULT_LABEL_RATIOS if label_ratios is None else label_ratios)
    rows_by_label: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_label[int(row["_normalized_label"])].append(row)

    available_per_class = {
        0: len(rows_by_label.get(0, [])),
        1: len(rows_by_label.get(1, [])),
    }
    actual_limit = _max_limit_for_ratio(limit, available_per_class, normalized_ratios)
    label_targets = _allocate_label_targets(
        actual_limit, normalized_ratios, available_per_class
    )

    balanced: list[dict[str, str]] = []
    for label in (0, 1):
        balanced.extend(
            _sample_diverse_rows(rows_by_label.get(label, []), label_targets[label], rng)
        )
    rng.shuffle(balanced)
    return balanced, {
        "requested_limit": limit,
        "actual_limit": len(balanced),
        "target_label_0": label_targets[0],
        "target_label_1": label_targets[1],
        "available_label_0": available_per_class[0],
        "available_label_1": available_per_class[1],
        "requested_ratio_label_0": normalized_ratios[0],
        "requested_ratio_label_1": normalized_ratios[1],
    }


def _sample_diverse_rows(
    rows: list[dict[str, str]], target: int, rng: Random
) -> list[dict[str, str]]:
    if target <= 0 or not rows:
        return []

    buckets: dict[str, dict[str, deque[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    for row in rows:
        source = row.get("source") or "[unknown_source]"
        author = row.get("author_id") or "[unknown_author]"
        buckets[source][author].append(row)

    sources = list(buckets)
    rng.shuffle(sources)
    author_orders: dict[str, list[str]] = {}
    author_offsets: dict[str, int] = {}
    for source in sources:
        authors = list(buckets[source])
        rng.shuffle(authors)
        author_orders[source] = authors
        author_offsets[source] = 0
        for author in authors:
            bucket_rows = list(buckets[source][author])
            rng.shuffle(bucket_rows)
            buckets[source][author] = deque(bucket_rows)

    selected: list[dict[str, str]] = []
    active_sources = deque(sources)
    while active_sources and len(selected) < target:
        source = active_sources.popleft()
        authors = author_orders[source]
        offset = author_offsets[source]
        picked = False
        for step in range(len(authors)):
            author_index = (offset + step) % len(authors)
            author = authors[author_index]
            bucket = buckets[source][author]
            if bucket:
                selected.append(bucket.popleft())
                author_offsets[source] = (author_index + 1) % len(authors)
                picked = True
                break
        if picked and any(buckets[source][author] for author in authors):
            active_sources.append(source)
    return selected


def _ratio_component(value: float) -> str:
    percent = f"{value * 100:.2f}".rstrip("0").rstrip(".")
    return percent.replace(".", "_")


def _max_limit_for_ratio(
    requested_limit: int,
    available_per_class: dict[int, int],
    label_ratios: dict[int, float],
) -> int:
    feasible_limit = requested_limit
    for label, ratio in label_ratios.items():
        if ratio <= 0:
            continue
        feasible_limit = min(feasible_limit, int(available_per_class[label] / ratio))
    return feasible_limit


def _allocate_label_targets(
    total: int,
    label_ratios: dict[int, float],
    available_per_class: dict[int, int],
) -> dict[int, int]:
    raw_targets = {
        label: total * label_ratios.get(label, 0.0) for label in (0, 1)
    }
    targets = {
        label: min(int(raw_targets[label]), available_per_class[label]) for label in (0, 1)
    }

    remainder = total - sum(targets.values())
    if remainder <= 0:
        return targets

    labels_by_fraction = sorted(
        (0, 1),
        key=lambda label: (raw_targets[label] - int(raw_targets[label]), -label),
        reverse=True,
    )
    while remainder > 0:
        assigned = False
        for label in labels_by_fraction:
            if targets[label] < available_per_class[label]:
                targets[label] += 1
                remainder -= 1
                assigned = True
                if remainder == 0:
                    break
        if not assigned:
            break
    return targets
