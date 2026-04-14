import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragebait_detector.pipeline import generate_mock_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate mock rage-bait training data")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--output")
    return parser


if __name__ == "__main__":
    generate_mock_dataset(build_parser().parse_args())
