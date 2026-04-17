from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragebait_detector.config import load_settings
from ragebait_detector.inference import RageBaitPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score one or more texts with a trained rage-bait BERT checkpoint."
    )
    parser.add_argument("--config", default="configs/bert_32k.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-dir")
    parser.add_argument("--artifacts")
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Text to score. Repeat to score multiple texts.",
    )
    parser.add_argument(
        "--text-file",
        help="Optional text file with one post per line.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read one post per line from standard input.",
    )
    parser.add_argument(
        "--no-force-english",
        action="store_false",
        dest="force_english",
        help="Re-enable inference-time language rejection instead of assuming the input should be scored as English.",
    )
    parser.set_defaults(force_english=True)
    return parser


def resolve_checkpoint(args, settings) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    if args.run_dir:
        return Path(args.run_dir) / "bert" / "checkpoint.pt"
    if args.artifacts:
        with Path(args.artifacts).open("r", encoding="utf-8") as handle:
            artifacts = json.load(handle)
        return Path(artifacts["checkpoint_path"])

    output_dir = Path(settings.paths.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(
            f"No output directory found at {output_dir}. Pass --checkpoint, --run-dir, or --artifacts."
        )

    candidates = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_dir() and (path / "bert" / "checkpoint.pt").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No trained checkpoints found under {output_dir}. Pass --checkpoint, --run-dir, or --artifacts."
        )
    return candidates[-1] / "bert" / "checkpoint.pt"


def collect_texts(args) -> list[str]:
    texts: list[str] = list(args.text)
    if args.text_file:
        with Path(args.text_file).open("r", encoding="utf-8") as handle:
            texts.extend(line.rstrip("\n") for line in handle if line.strip())
    if args.stdin:
        texts.extend(line.rstrip("\n") for line in sys.stdin if line.strip())
    return texts


def score_texts(texts: Iterable[str], predictor: RageBaitPredictor) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for text in texts:
        result = predictor.predict_text(text)
        results.append({"text": text, **asdict(result)})
    return results


@contextlib.contextmanager
def suppress_transformers_load_report():
    logger_names = [
        "transformers.modeling_utils",
        "transformers.utils.loading_report",
    ]
    previous = []
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        previous.append((logger, logger.disabled))
        logger.disabled = True
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        for logger, was_disabled in previous:
            logger.disabled = was_disabled


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings(args.config)
    if args.force_english:
        settings.data.drop_non_english = False
    texts = collect_texts(args)
    if not texts:
        raise SystemExit("Provide at least one input via --text, --text-file, or --stdin.")

    checkpoint_path = resolve_checkpoint(args, settings)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    with suppress_transformers_load_report():
        predictor = RageBaitPredictor.from_checkpoint(checkpoint_path, settings)
    results = score_texts(texts, predictor)
    payload: object
    if len(results) == 1:
        payload = {
            "checkpoint_path": str(checkpoint_path),
            "force_english": args.force_english,
            **results[0],
        }
    else:
        payload = {
            "checkpoint_path": str(checkpoint_path),
            "force_english": args.force_english,
            "results": results,
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
