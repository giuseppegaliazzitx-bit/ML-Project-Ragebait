import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragebait_detector.config import load_settings
from ragebait_detector.data.unifier import run_interactive_import
from ragebait_detector.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactively compile raw Kaggle tweet files into one clean CSV")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-dir")
    parser.add_argument("--output")
    parser.add_argument("--manifest")
    return parser


if __name__ == "__main__":
    configure_logging()
    args = build_parser().parse_args()
    settings = load_settings(args.config)
    summary = run_interactive_import(
        settings=settings,
        input_dir=args.input_dir,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print(summary)

