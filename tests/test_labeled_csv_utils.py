from __future__ import annotations

import csv
from pathlib import Path

from ragebait_detector.utils.labeled_csv import (
    balance_rows,
    compress_csv,
    ensure_csv_available,
    parse_label_ratio,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_ensure_csv_available_restores_gzip(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    _write_csv(
        csv_path,
        [
            {"label": "0", "confidence": "0.91", "author_id": "a1", "source": "s1"},
        ],
    )
    compressed_path = compress_csv(csv_path)

    assert not csv_path.exists()
    assert compressed_path.exists()

    restored_path = ensure_csv_available(csv_path)
    assert restored_path.exists()
    assert restored_path.read_text(encoding="utf-8").startswith("label,confidence")


def test_balance_rows_keeps_classes_even() -> None:
    rows: list[dict[str, str]] = []
    for idx in range(6):
        rows.append(
            {
                "label": "0",
                "author_id": f"author_{idx}",
                "source": "source_a" if idx % 2 == 0 else "source_b",
                "_normalized_label": "0",
            }
        )
        rows.append(
            {
                "label": "1",
                "author_id": f"author_{idx}",
                "source": "source_a" if idx % 2 == 0 else "source_b",
                "_normalized_label": "1",
            }
        )

    balanced, stats = balance_rows(rows, limit=8, seed=7)

    assert stats["actual_limit"] == 8
    assert sum(1 for row in balanced if row["_normalized_label"] == "0") == 4
    assert sum(1 for row in balanced if row["_normalized_label"] == "1") == 4


def test_balance_rows_supports_custom_ratio() -> None:
    rows: list[dict[str, str]] = []
    for idx in range(12):
        rows.append(
            {
                "label": "0",
                "author_id": f"author_{idx}",
                "source": "source_a" if idx % 2 == 0 else "source_b",
                "_normalized_label": "0",
            }
        )
    for idx in range(8):
        rows.append(
            {
                "label": "1",
                "author_id": f"author_{idx}",
                "source": "source_a" if idx % 2 == 0 else "source_b",
                "_normalized_label": "1",
            }
        )

    balanced, stats = balance_rows(
        rows,
        limit=20,
        seed=7,
        label_ratios=parse_label_ratio("60/40"),
    )

    assert stats["actual_limit"] == 20
    assert sum(1 for row in balanced if row["_normalized_label"] == "0") == 12
    assert sum(1 for row in balanced if row["_normalized_label"] == "1") == 8
