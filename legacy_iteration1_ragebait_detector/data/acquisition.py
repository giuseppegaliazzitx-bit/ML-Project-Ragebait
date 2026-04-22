from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ragebait_detector.utils.io import read_csv, read_jsonl, write_csv

UNIFIED_FIELDS = [
    "post_id",
    "author_id",
    "created_at",
    "language",
    "text",
    "source",
    "label",
]

_ID_KEYS = ("post_id", "id", "tweet_id")
_AUTHOR_KEYS = ("author_id", "user_id", "username")
_CREATED_AT_KEYS = ("created_at", "timestamp", "date")
_LANG_KEYS = ("language", "lang")
_TEXT_KEYS = ("text", "full_text", "content", "body")
_LABEL_KEYS = ("label", "ragebait", "is_ragebait")


def _first_present(row: dict, keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def normalize_raw_record(row: dict, source: str) -> dict[str, str]:
    return {
        "post_id": _first_present(row, _ID_KEYS),
        "author_id": _first_present(row, _AUTHOR_KEYS),
        "created_at": _first_present(row, _CREATED_AT_KEYS),
        "language": _first_present(row, _LANG_KEYS, default="unknown"),
        "text": _first_present(row, _TEXT_KEYS),
        "source": source,
        "label": _first_present(row, _LABEL_KEYS),
    }


def load_records(path: str | Path) -> list[dict[str, str]]:
    source_path = Path(path)
    if source_path.suffix.lower() == ".jsonl":
        rows = read_jsonl(source_path)
    elif source_path.suffix.lower() == ".csv":
        rows = read_csv(source_path)
    else:
        raise ValueError(f"Unsupported input format for {source_path}")
    return [normalize_raw_record(row, source=source_path.name) for row in rows]


def normalize_exports(input_paths: Iterable[str | Path], output_path: str | Path) -> Path:
    rows: list[dict[str, str]] = []
    for path in input_paths:
        rows.extend(load_records(path))
    deduped = deduplicate_records(rows)
    return write_csv(output_path, deduped, UNIFIED_FIELDS)


def deduplicate_records(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        record_id = row["post_id"] or f"{row['source']}::{len(deduped)}"
        if record_id in seen:
            continue
        seen.add(record_id)
        row["post_id"] = record_id
        deduped.append(row)
    return deduped


def build_annotation_template(
    unified_posts_path: str | Path,
    output_path: str | Path,
) -> Path:
    rows = read_csv(unified_posts_path)
    annotation_rows = [
        {
            "post_id": row["post_id"],
            "text": row["text"],
            "label": "",
            "guideline_bucket": "",
            "notes": "",
        }
        for row in rows
    ]
    return write_csv(
        output_path,
        annotation_rows,
        ["post_id", "text", "label", "guideline_bucket", "notes"],
    )


def merge_annotations(
    unified_posts_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
) -> Path:
    posts = {row["post_id"]: row for row in read_csv(unified_posts_path)}
    labeled_rows: list[dict[str, str]] = []

    for annotation in read_csv(annotations_path):
        post_id = annotation["post_id"]
        label = annotation.get("label", "").strip()
        if post_id not in posts:
            continue
        if label not in {"0", "1"}:
            continue
        row = dict(posts[post_id])
        row["label"] = label
        labeled_rows.append(row)

    return write_csv(output_path, labeled_rows, UNIFIED_FIELDS)


def validate_volume(rows: list[dict[str, str]], minimum_size: int) -> tuple[bool, int]:
    usable = sum(1 for row in rows if row.get("text", "").strip())
    return usable >= minimum_size, usable

