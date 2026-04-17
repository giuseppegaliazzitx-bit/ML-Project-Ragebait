import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragebait_detector.utils.labeled_csv import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    analyze_labeled_rows,
    ensure_csv_available,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a labeled CSV and report ragebait prevalence at several confidence thresholds."
    )
    parser.add_argument(
        "--input",
        default="data/labeled/vllm_all_qwen_ragebait_labels.csv",
        help="Path to the labeled CSV. If missing, the script restores it from .csv.gz.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=list(DEFAULT_CONFIDENCE_THRESHOLDS),
        help="Confidence thresholds to evaluate.",
    )
    return parser


def _format_ratio(value: float) -> str:
    return f"{value:.2%}"


if __name__ == "__main__":
    args = build_parser().parse_args()
    csv_path = ensure_csv_available(args.input)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        summary = analyze_labeled_rows(csv.DictReader(handle), args.thresholds)

    print(f"Input: {csv_path}")
    print(f"Total rows: {summary['total_rows']:,}")
    print(f"Valid labeled rows: {summary['valid_rows']:,}")
    print(f"Detected ragebait: {summary['ragebait_rows']:,}")
    print(f"Overall ragebait ratio: {_format_ratio(summary['ragebait_ratio'])}")
    print("")
    print("threshold | rows | ragebait | ragebait_ratio | coverage")
    print("----------|------|----------|----------------|---------")
    for item in summary["thresholds"]:
        print(
            f"{item['threshold']:.2f} | "
            f"{item['rows']:,} | "
            f"{item['ragebait_rows']:,} | "
            f"{_format_ratio(item['ragebait_ratio'])} | "
            f"{_format_ratio(item['coverage_ratio'])}"
        )
