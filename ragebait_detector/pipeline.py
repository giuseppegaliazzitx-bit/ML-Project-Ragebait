from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ragebait_detector.config import load_settings
from ragebait_detector.data.acquisition import (
    build_annotation_template,
    merge_annotations,
    normalize_exports,
    validate_volume,
)
from ragebait_detector.data.dataset import load_processed_records, stratified_split
from ragebait_detector.data.preprocessing import prepare_labeled_dataset
from ragebait_detector.data.unifier import run_interactive_import
from ragebait_detector.evaluation import evaluate_checkpoint_on_labeled_csv
from ragebait_detector.labeling.vllm_labeler import label_csv_with_vllm
from ragebait_detector.utils.io import dump_json, ensure_parent, read_csv, write_csv
from ragebait_detector.utils.logging import configure_logging
from ragebait_detector.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)


def prepare_exports(args) -> dict[str, Any]:
    settings = load_settings(args.config)
    destination = normalize_exports(
        args.inputs, args.output or settings.paths.unified_posts_path
    )
    rows = read_csv(destination)
    meets_volume, usable_rows = validate_volume(rows, settings.data.min_posts)
    summary = {
        "unified_posts_path": str(destination),
        "usable_rows": usable_rows,
        "meets_volume_requirement": meets_volume,
        "minimum_required_posts": settings.data.min_posts,
    }
    dump_json(Path(settings.paths.output_dir) / "prepare_summary.json", summary)
    return summary


def build_annotation_sheet(args) -> dict[str, Any]:
    settings = load_settings(args.config)
    destination = build_annotation_template(
        args.input or settings.paths.unified_posts_path,
        args.output or settings.paths.annotation_template_path,
    )
    return {"annotation_template_path": str(destination)}


def merge_annotation_sheet(args) -> dict[str, Any]:
    settings = load_settings(args.config)
    destination = merge_annotations(
        unified_posts_path=args.posts or settings.paths.unified_posts_path,
        annotations_path=args.annotations,
        output_path=args.output or settings.paths.labeled_posts_path,
    )
    return {"labeled_posts_path": str(destination)}


def preprocess_dataset(args) -> dict[str, Any]:
    settings = load_settings(args.config)
    summary = prepare_labeled_dataset(
        input_path=args.input or settings.paths.labeled_posts_path,
        output_path=args.output or settings.paths.processed_dataset_path,
        settings=settings,
    )
    dump_json(Path(settings.paths.output_dir) / "preprocess_summary.json", summary)
    return summary


def interactive_import_dataset(args) -> dict[str, Any]:
    settings = load_settings(args.config)
    summary = run_interactive_import(
        settings=settings,
        input_dir=args.input_dir,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    return summary


def label_dataset_with_vllm(args) -> dict[str, Any]:
    settings = load_settings(args.config)
    if args.model:
        settings.vllm.model = args.model
    if args.quantization:
        settings.vllm.quantization = args.quantization
    if args.gpu_memory_utilization is not None:
        settings.vllm.gpu_memory_utilization = args.gpu_memory_utilization
    if args.max_model_len is not None:
        settings.vllm.max_model_len = args.max_model_len
    if args.temperature is not None:
        settings.vllm.temperature = args.temperature
    output_path = args.output or settings.paths.vllm_labels_path
    summary_path = (
        args.summary or Path(settings.paths.output_dir) / "vllm_labeling_summary.json"
    )
    summary = label_csv_with_vllm(
        input_path=args.input or settings.paths.unlabeled_posts_path,
        output_path=output_path,
        summary_path=summary_path,
        settings=settings,
        limit=args.limit,
        random_seed=args.random_seed,
        enable_random=args.enable_random,
        balance_by_source=args.balance_by_source,
    )
    return summary


def evaluate_manual_dataset(args) -> dict[str, Any]:
    settings = load_settings(args.config)
    checkpoint_path = _resolve_checkpoint_path(args, settings)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_root = (
        Path(args.run_dir)
        if args.run_dir
        else _infer_run_root_from_checkpoint(checkpoint_path)
    )
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "manual_eval"
    return evaluate_checkpoint_on_labeled_csv(
        checkpoint_path=checkpoint_path,
        input_path=args.input or settings.paths.manual_eval_path,
        settings=settings,
        output_dir=output_dir,
        force_english=args.force_english,
    )


def run_training_pipeline(args) -> dict[str, Any]:
    from ragebait_detector.models.baselines import run_baseline_suite
    from ragebait_detector.training.trainer import train_bert_classifier

    settings = load_settings(args.config)
    seed_everything(settings.training.seed)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = ensure_parent(
        Path(args.output_dir or settings.paths.output_dir) / timestamp / "summary.json"
    ).parent

    preprocess_summary = prepare_labeled_dataset(
        input_path=args.input or settings.paths.labeled_posts_path,
        output_path=settings.paths.processed_dataset_path,
        settings=settings,
    )
    records = load_processed_records(settings.paths.processed_dataset_path)
    splits = stratified_split(
        records=records,
        validation_size=settings.data.validation_size,
        test_size=settings.data.test_size,
        seed=settings.training.seed,
    )

    split_summary = {
        "train_size": len(splits.train),
        "validation_size": len(splits.validation),
        "test_size": len(splits.test),
    }

    baseline_results = {}
    if settings.baselines.enabled and not args.skip_baselines:
        baseline_results = run_baseline_suite(
            splits=splits,
            output_dir=run_root / "baselines",
            max_features=settings.baselines.max_features,
            ngram_range=tuple(settings.baselines.ngram_range),
            seed=settings.training.seed,
        )

    bert_results = {}
    if not args.baselines_only:
        bert_results = train_bert_classifier(
            splits=splits,
            settings=settings,
            output_dir=run_root,
        )

    summary = {
        "preprocess_summary": preprocess_summary,
        "split_summary": split_summary,
        "baseline_results": baseline_results,
        "bert_results": bert_results,
    }
    dump_json(run_root / "summary.json", summary)
    return summary


def generate_mock_dataset(args) -> dict[str, Any]:
    settings = load_settings(args.config)
    rows: list[dict[str, Any]] = []
    templates = {
        1: [
            "everyone who disagrees with this obvious truth is pathetic",
            "if this post makes you mad then i clearly won the argument",
            "watch the comments melt down because i know this bait works",
            "you are all proving my point by reacting exactly like this",
        ],
        0: [
            "today was exhausting and i am genuinely upset about the layoffs",
            "sharing an article about transit policy changes in the city",
            "i disagree with the decision but want to understand the facts",
            "here is a neutral update on the match and the final score",
        ],
    }
    total_rows = args.rows or 500
    for index in range(total_rows):
        label = 1 if index % 5 == 0 else 0
        text = templates[label][index % len(templates[label])]
        rows.append(
            {
                "post_id": f"mock-{index}",
                "author_id": f"user-{index % 25}",
                "created_at": "2026-01-01T00:00:00Z",
                "language": "en",
                "text": text,
                "source": "mock_generator",
                "label": label,
            }
        )
    destination = args.output or settings.paths.labeled_posts_path
    write_csv(
        destination,
        rows,
        ["post_id", "author_id", "created_at", "language", "text", "source", "label"],
    )
    return {"mock_dataset_path": str(destination), "rows": total_rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rage-bait detector pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-exports")
    prepare_parser.add_argument("--inputs", nargs="+", required=True)
    prepare_parser.add_argument("--output")

    interactive_import_parser = subparsers.add_parser("interactive-import")
    interactive_import_parser.add_argument("--input-dir")
    interactive_import_parser.add_argument("--output")
    interactive_import_parser.add_argument("--manifest")

    annotation_parser = subparsers.add_parser("build-annotation-template")
    annotation_parser.add_argument("--input")
    annotation_parser.add_argument("--output")

    merge_parser = subparsers.add_parser("merge-annotations")
    merge_parser.add_argument("--posts")
    merge_parser.add_argument("--annotations", required=True)
    merge_parser.add_argument("--output")

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--input")
    preprocess_parser.add_argument("--output")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--input")
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--skip-baselines", action="store_true")
    run_parser.add_argument("--baselines-only", action="store_true")

    mock_parser = subparsers.add_parser("generate-mock-data")
    mock_parser.add_argument("--rows", type=int)
    mock_parser.add_argument("--output")

    vllm_parser = subparsers.add_parser("label-with-vllm")
    vllm_parser.add_argument("--input")
    vllm_parser.add_argument("--output")
    vllm_parser.add_argument("--summary")
    vllm_parser.add_argument("--limit", type=int)
    vllm_parser.add_argument("--random-seed", type=int)
    vllm_parser.add_argument(
        "--enable-random",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    vllm_parser.add_argument(
        "--balance-by-source",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    vllm_parser.add_argument("--model")
    vllm_parser.add_argument("--quantization")
    vllm_parser.add_argument("--gpu-memory-utilization", type=float)
    vllm_parser.add_argument("--max-model-len", type=int)
    vllm_parser.add_argument("--temperature", type=float)

    manual_eval_parser = subparsers.add_parser("evaluate-manual")
    manual_eval_parser.add_argument("--input")
    manual_eval_parser.add_argument("--checkpoint")
    manual_eval_parser.add_argument("--run-dir")
    manual_eval_parser.add_argument("--artifacts")
    manual_eval_parser.add_argument("--output-dir")
    manual_eval_parser.add_argument(
        "--no-force-english",
        action="store_false",
        dest="force_english",
        help="Re-enable inference-time language rejection instead of scoring the manual set as English by default.",
    )
    manual_eval_parser.set_defaults(force_english=True)

    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "prepare-exports": prepare_exports,
        "interactive-import": interactive_import_dataset,
        "build-annotation-template": build_annotation_sheet,
        "merge-annotations": merge_annotation_sheet,
        "preprocess": preprocess_dataset,
        "run": run_training_pipeline,
        "generate-mock-data": generate_mock_dataset,
        "label-with-vllm": label_dataset_with_vllm,
        "evaluate-manual": evaluate_manual_dataset,
    }
    result = commands[args.command](args)
    LOGGER.info("Pipeline result: %s", result)


def _resolve_checkpoint_path(args, settings) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    if args.run_dir:
        return Path(args.run_dir) / "bert" / "checkpoint.pt"
    if args.artifacts:
        import json

        with Path(args.artifacts).open("r", encoding="utf-8") as handle:
            artifacts = json.load(handle)
        return Path(artifacts["checkpoint_path"])

    output_dir = Path(settings.paths.output_dir)
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


def _infer_run_root_from_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    if checkpoint.name == "checkpoint.pt" and checkpoint.parent.name == "bert":
        return checkpoint.parent.parent
    return checkpoint.parent


if __name__ == "__main__":
    main()
