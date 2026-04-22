"""BERT fine-tuning for Iteration 2: Experiment 1."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.make_dataset import load_config
from src.data.transformer_dataset import build_transformer_dataloaders
from src.evaluation.evaluate import evaluate_logits, save_json
from src.models.bert_classifier import (
    create_adamw_optimizer,
    create_bert_classifier,
    create_linear_scheduler,
)


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path from the Iteration 2 project root."""
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for Iteration 2: Experiment 1 BERT training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_bce_targets(labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    """Convert integer class ids into one-hot BCE targets."""
    return torch_functional.one_hot(labels, num_classes=num_labels).float()


def run_validation(
    model,
    dataloader,
    criterion,
    device: torch.device,
    num_labels: int,
    confusion_matrix_path: Path,
) -> tuple[float, dict[str, Any]]:
    """Run validation and compute metrics from raw logits."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels: list[int] = []
    logits_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            targets = compute_bce_targets(labels, num_labels=num_labels)
            loss = criterion(logits, targets)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            all_labels.extend(labels.detach().cpu().tolist())
            logits_batches.append(logits.detach().cpu().numpy())

    average_loss = total_loss / max(total_examples, 1)
    all_logits = np.concatenate(logits_batches, axis=0)
    evaluation = evaluate_logits(
        y_true=all_labels,
        logits=all_logits,
        confusion_matrix_path=confusion_matrix_path,
        title="BERT Validation Confusion Matrix",
    )
    return average_loss, evaluation


def save_checkpoint(
    model,
    checkpoint_path: Path,
    epoch: int,
    validation_loss: float,
    validation_metrics: dict[str, float],
    config: dict[str, Any],
) -> Path:
    """Save the best validation checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "state_dict": copy.deepcopy(model.state_dict()),
        "validation_loss": validation_loss,
        "validation_metrics": validation_metrics,
        "model_config": config["model"],
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_baseline_results(summary_path: Path) -> dict[str, Any]:
    """Load Tier 1 and Tier 2 baseline results if available."""
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train_bert_model(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train and evaluate the Tier 3 BERT model."""
    dataloaders, tokenizer, split_frames = build_transformer_dataloaders(config)

    output_dir = resolve_project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = resolve_project_path(config["paths"]["tokenizer_dir"])
    tokenizer.save_pretrained(tokenizer_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_bert_classifier(config).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = create_adamw_optimizer(model, config)
    total_steps = len(dataloaders["train"]) * config["training"]["epochs"]
    scheduler = create_linear_scheduler(optimizer, total_steps, config)
    use_amp = config["training"].get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    history: dict[str, Any] = {
        "epochs": [],
        "device": str(device),
        "train_rows": int(len(split_frames["train"])),
        "val_rows": int(len(split_frames["val"])),
        "max_length": config["model"]["max_length"],
        "tokenizer_dir": str(tokenizer_dir),
    }

    best_f1 = -1.0
    best_epoch = 0
    best_val_loss = float("inf")
    best_checkpoint_path = resolve_project_path(config["paths"]["checkpoint_path"])
    best_state = None

    training_start = time.perf_counter()
    for epoch in range(1, config["training"]["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_total = 0.0
        train_examples = 0

        for batch in dataloaders["train"]:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None
            targets = compute_bce_targets(labels, num_labels=config["model"]["num_labels"])

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = labels.size(0)
            train_loss_total += loss.item() * batch_size
            train_examples += batch_size

        average_train_loss = train_loss_total / max(train_examples, 1)
        validation_loss, validation = run_validation(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            num_labels=config["model"]["num_labels"],
            confusion_matrix_path=output_dir / "bert_confusion_matrix.png",
        )
        validation_metrics = validation["metrics"]
        epoch_seconds = time.perf_counter() - epoch_start
        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": average_train_loss,
                "val_loss": validation_loss,
                "val_metrics": validation_metrics,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "epoch_seconds": epoch_seconds,
            }
        )
        logging.info(
            "Epoch %s | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | seconds=%.2f",
            epoch,
            average_train_loss,
            validation_loss,
            validation_metrics["f1"],
            epoch_seconds,
        )

        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            best_epoch = epoch
            best_val_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                model=model,
                checkpoint_path=best_checkpoint_path,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_metrics=validation_metrics,
                config=config,
            )
            logging.info("Saved new best checkpoint to %s", best_checkpoint_path)

    total_train_seconds = time.perf_counter() - training_start
    if best_state is None:
        raise RuntimeError("BERT training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    predict_start = time.perf_counter()
    validation_loss, validation = run_validation(
        model=model,
        dataloader=dataloaders["val"],
        criterion=criterion,
        device=device,
        num_labels=config["model"]["num_labels"],
        confusion_matrix_path=output_dir / "bert_confusion_matrix.png",
    )
    predict_seconds = time.perf_counter() - predict_start

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["checkpoint_path"] = str(best_checkpoint_path)
    bert_result = {
        "tier": "Tier 3",
        "model_type": type(model).__name__,
        "pretrained_model": config["model"]["pretrained_name"],
        "validation": validation,
        "timing_seconds": {
            "train": total_train_seconds,
            "predict": predict_seconds,
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(best_checkpoint_path),
        "tokenizer_dir": str(tokenizer_dir),
        "device": str(device),
        "max_length": config["model"]["max_length"],
    }
    return bert_result, history, split_frames


def build_summary(
    config: dict[str, Any],
    split_frames,
    bert_result: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build the Tier 3 comparative summary."""
    baseline_summary_path = resolve_project_path(config["paths"]["baseline_summary"])
    baseline_summary = load_baseline_results(baseline_summary_path)
    baseline_results = baseline_summary.get("results", {})

    all_results = {**baseline_results, "bert_base_uncased": bert_result}
    ranking = sorted(
        (
            {
                "model_name": model_name,
                "tier": result["tier"],
                "validation_f1": result["validation"]["metrics"]["f1"],
            }
            for model_name, result in all_results.items()
        ),
        key=lambda item: item["validation_f1"],
        reverse=True,
    )
    best_validation_model = ranking[0]["model_name"] if ranking else "bert_base_uncased"

    ffnn_result = baseline_results.get("ffnn")
    ffnn_delta = None
    if ffnn_result is not None:
        ffnn_delta = bert_result["validation"]["metrics"]["f1"] - ffnn_result["validation"]["metrics"]["f1"]

    return {
        "iteration": config["experiment"]["iteration"],
        "experiment": "Experiment 1",
        "task": "binary_classification",
        "dataset": {
            "train_rows": int(len(split_frames["train"])),
            "val_rows": int(len(split_frames["val"])),
            "test_rows": int(len(split_frames["test"])),
            "train_split_path": str(resolve_project_path(config["paths"]["train_split"])),
            "val_split_path": str(resolve_project_path(config["paths"]["val_split"])),
        },
        "baseline_summary_path": str(baseline_summary_path),
        "results": all_results,
        "best_validation_model": best_validation_model,
        "validation_f1_ranking": ranking,
        "bert_vs_ffnn_validation_f1_delta": ffnn_delta,
        "runtime_seconds": runtime_seconds,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for BERT fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Train the Tier 3 BERT model for Iteration 2: Experiment 1."
    )
    parser.add_argument(
        "--config",
        default="configs/exp1_binary_bert.yaml",
        help="Path to the Iteration 2 BERT config.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for Iteration 2: Experiment 1 Tier 3 training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(config["training"]["seed"])

    overall_start = time.perf_counter()
    bert_result, history, split_frames = train_bert_model(config)
    runtime_seconds = time.perf_counter() - overall_start

    output_dir = resolve_project_path(config["paths"]["output_dir"])
    save_json(history, output_dir / "training_history.json")
    summary = build_summary(
        config=config,
        split_frames=split_frames,
        bert_result=bert_result,
        runtime_seconds=runtime_seconds,
    )
    summary_path = save_json(summary, output_dir / "summary.json")
    logging.info("Saved BERT summary to %s", summary_path)


if __name__ == "__main__":
    main()
