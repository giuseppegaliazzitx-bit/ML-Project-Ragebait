import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragebait_detector.utils.labeled_csv import compress_csv, gzip_path_for


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress a labeled CSV to .csv.gz so it is easier to store in git."
    )
    parser.add_argument(
        "--input",
        default="data/labeled/vllm_all_qwen_ragebait_labels.csv",
        help="Path to the raw labeled CSV.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep the uncompressed CSV after writing the .csv.gz file.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    compressed_path = compress_csv(args.input, keep_original=args.keep_original)
    print(f"Compressed to {compressed_path}")
    if args.keep_original:
        print(f"Original retained at {Path(args.input)}")
    else:
        print(f"Original removed; restore on demand from {gzip_path_for(args.input)}")
