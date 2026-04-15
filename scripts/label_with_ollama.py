import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragebait_detector.config import load_settings
from ragebait_detector.labeling.ollama_labeler import label_csv_with_ollama
from ragebait_detector.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label compiled tweets with Ollama using batched native JSON responses"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--summary")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--host")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--random-selection", type=bool)
    parser.add_argument("--random-seed", type=int)
    return parser


if __name__ == "__main__":
    configure_logging()
    args = build_parser().parse_args()
    settings = load_settings(args.config)
    if args.host:
        settings.ollama.host = args.host
    if args.model:
        settings.ollama.model = args.model
    if args.workers is not None:
        settings.ollama.max_workers = args.workers
    if args.batch_size is not None:
        settings.ollama.batch_size = args.batch_size
    if args.random_selection is not None:
        settings.ollama.random_selection = args.random_selection
    if args.random_seed is not None:
        settings.ollama.random_seed = args.random_seed
    summary = label_csv_with_ollama(
        input_path=args.input or settings.paths.unlabeled_posts_path,
        output_path=args.output or settings.paths.ollama_labels_path,
        summary_path=args.summary
        or Path(settings.paths.output_dir) / "ollama_labeling_summary.json",
        settings=settings,
        limit=args.limit,
    )
    print(summary)
