import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragebait_detector.config import load_settings
from ragebait_detector.labeling.ollama_labeler import label_csv_with_ollama
from ragebait_detector.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label compiled tweets with Ollama and a tool-calling chat model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--summary")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--host")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int)
    return parser


if __name__ == "__main__":
    configure_logging()
    args = build_parser().parse_args()
    settings = load_settings(args.config)
    if args.host:
        settings.ollama.host = args.host
    if args.model:
        settings.ollama.model = args.model
    if args.workers:
        settings.ollama.max_workers = args.workers
    summary = label_csv_with_ollama(
        input_path=args.input or settings.paths.unlabeled_posts_path,
        output_path=args.output or settings.paths.ollama_labels_path,
        summary_path=args.summary or Path(settings.paths.output_dir) / "ollama_labeling_summary.json",
        settings=settings,
        limit=args.limit,
    )
    print(summary)
