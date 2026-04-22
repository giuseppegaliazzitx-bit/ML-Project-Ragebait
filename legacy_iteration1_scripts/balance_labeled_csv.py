import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragebait_detector.utils.io import ensure_parent
from ragebait_detector.utils.labeled_csv import (
    balance_rows,
    label_ratio_tag,
    load_filtered_rows,
    parse_label_ratio,
    threshold_tag,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a balanced labeled CSV while spreading samples across sources and authors."
    )
    parser.add_argument(
        "--input",
        default="data/labeled/vllm_all_qwen_ragebait_labels.csv",
        help="Path to the labeled CSV. If missing, the script restores it from .csv.gz.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Maximum number of rows to write.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        required=True,
        help="Minimum confidence required to keep a row.",
    )
    parser.add_argument(
        "--label-ratio",
        default="50/50",
        help="Requested label split in label0/label1 order, for example 50/50 or 60/40.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tie-breaking and final row order.",
    )
    parser.add_argument(
        "--output",
        help="Optional explicit output path. Defaults to data/labeled/balanced_<limit>_c<threshold>_r<label0>_<label1>.csv",
    )
    return parser


def default_output_path(
    input_path: str | Path,
    limit: int,
    threshold: float,
    label_ratios: dict[int, float],
) -> Path:
    labeled_dir = Path(input_path).resolve().parent
    return (
        labeled_dir
        / f"balanced_{limit}_c{threshold_tag(threshold)}_{label_ratio_tag(label_ratios)}.csv"
    )


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("--confidence-threshold must be between 0 and 1")
    try:
        label_ratios = parse_label_ratio(args.label_ratio)
    except ValueError as exc:
        raise SystemExit(f"--label-ratio error: {exc}") from exc

    fieldnames, filtered_rows, skipped = load_filtered_rows(
        args.input,
        confidence_threshold=args.confidence_threshold,
    )
    balanced, stats = balance_rows(
        filtered_rows,
        limit=args.limit,
        seed=args.seed,
        label_ratios=label_ratios,
    )

    output_path = Path(args.output) if args.output else default_output_path(
        args.input, args.limit, args.confidence_threshold, label_ratios
    )
    destination = ensure_parent(output_path)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in balanced:
            clean_row = {key: value for key, value in row.items() if not key.startswith("_")}
            writer.writerow(clean_row)

    label_counts = {0: 0, 1: 0}
    for row in balanced:
        label_counts[int(row["_normalized_label"])] += 1

    print(f"Output: {destination}")
    print(f"Rows written: {stats['actual_limit']:,}")
    print(
        "Requested label ratio (0/1): "
        f"{stats['requested_ratio_label_0']:.0%}/{stats['requested_ratio_label_1']:.0%}"
    )
    print(f"Label 0 rows: {label_counts[0]:,}")
    print(f"Label 1 rows: {label_counts[1]:,}")
    print(f"Available label 0 rows at threshold: {stats['available_label_0']:,}")
    print(f"Available label 1 rows at threshold: {stats['available_label_1']:,}")
    print(f"Skipped invalid status rows: {skipped['invalid_status']:,}")
    print(f"Skipped invalid label/confidence rows: {skipped['invalid_label']:,}")
    print(f"Skipped low-confidence rows: {skipped['low_confidence']:,}")
    if stats["actual_limit"] < args.limit:
        print(
            "Requested limit could not be met while preserving the requested class ratio at the chosen threshold."
        )
