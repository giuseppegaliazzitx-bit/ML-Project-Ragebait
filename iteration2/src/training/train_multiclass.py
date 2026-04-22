"""Multiclass baseline training for Iteration 2: Experiment 2."""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.make_dataset import (
    create_multiclass_splits,
    get_output_paths,
    get_target_column,
    load_config,
    load_multiclass_splits,
)
from src.data.preprocessing import (
    build_class_weight_tensor,
    build_ffnn_dataloaders,
    build_tfidf_features,
    compute_class_weights,
    summarize_class_distribution,
)
from src.evaluation.evaluate import evaluate_predictions, save_json
from src.models.baselines import create_tier1_models, create_tier2_model


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for Experiment 2 multiclass training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve config paths from the Iteration 2 project root."""
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def choose_device(device_name: str) -> torch.device:
    """Choose the training device for Iteration 2: Experiment 2."""
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_names(config: dict[str, Any]) -> list[str]:
    """Return the ordered multiclass labels."""
    return config["dataset"]["label_order"]


def prepare_class_weight_artifact(
    config: dict[str, Any],
    split_frames: dict[str, Any],
) -> tuple[dict[int, float], dict[str, Any]]:
    """Compute and save exact train-split class weights."""
    class_names = get_class_names(config)
    class_ids = list(range(len(class_names)))
    target_column = get_target_column(config)
    y_train = split_frames["train"][target_column].to_numpy()
    class_weights = compute_class_weights(y_train, class_ids=class_ids)
    distribution = summarize_class_distribution(
        y_train,
        class_ids=class_ids,
        class_names=class_names,
    )
    artifact = {
        "formula": "weight_i = N / (K * count_i)",
        "class_distribution": distribution,
        "class_weights": {
            class_name: {
                "index": class_id,
                "weight": class_weights[class_id],
            }
            for class_id, class_name in enumerate(class_names)
        },
    }
    class_weights_path = resolve_project_path(config["paths"]["class_weights_path"])
    save_json(artifact, class_weights_path)
    return class_weights, artifact


def train_tier1_models(
    config: dict[str, Any],
    split_frames,
    output_dir: Path,
    class_weights: dict[int, float],
) -> dict[str, Any]:
    """Train weighted Tier 1 multiclass baselines."""
    text_column = config["dataset"]["text_column"]
    target_column = get_target_column(config)
    class_names = get_class_names(config)
    vectorizer, features = build_tfidf_features(
        split_frames["train"],
        split_frames["val"],
        split_frames["test"],
        config["preprocessing"]["tfidf"],
        text_column=text_column,
    )
    logging.info("TF-IDF vocabulary size: %s", len(vectorizer.vocabulary_))

    y_train = split_frames["train"][target_column].to_numpy()
    y_val = split_frames["val"][target_column].to_numpy()
    models = create_tier1_models(config, class_weight=class_weights)
    results: dict[str, Any] = {}

    for model_name, model in models.items():
        logging.info("Training Tier 1 multiclass model: %s", model_name)
        train_start = time.perf_counter()
        model.fit(features["train"], y_train)
        train_seconds = time.perf_counter() - train_start

        predict_start = time.perf_counter()
        y_pred = model.predict(features["val"])
        predict_seconds = time.perf_counter() - predict_start

        evaluation = evaluate_predictions(
            y_true=y_val,
            y_pred=y_pred,
            confusion_matrix_path=output_dir / f"{model_name}_confusion_matrix.png",
            task_type="multiclass",
            class_names=class_names,
            title=f"{model_name} Validation Confusion Matrix",
        )
        results[model_name] = {
            "tier": "Tier 1",
            "model_type": type(model).__name__,
            "validation": evaluation,
            "timing_seconds": {
                "train": train_seconds,
                "predict": predict_seconds,
            },
        }
    return results


def predict_ffnn(model, dataloader, device, criterion):
    """Run FFNN inference and collect multiclass validation inputs."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for batch in dataloader["val"]:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            all_targets.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / max(total_examples, 1)
    return average_loss, all_targets, all_predictions


def train_tier2_ffnn(
    config: dict[str, Any],
    split_frames,
    output_dir: Path,
    class_weights: dict[int, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train a weighted multiclass FFNN with early stopping on macro F1."""
    training_config = config["training"]
    sequence_config = config["preprocessing"]["sequence"]
    target_column = get_target_column(config)
    class_names = get_class_names(config)
    class_ids = list(range(len(class_names)))
    dataloaders, vocabulary = build_ffnn_dataloaders(
        split_frames["train"],
        split_frames["val"],
        split_frames["test"],
        sequence_config=sequence_config,
        batch_size=training_config["batch_size"],
        seed=training_config["seed"],
        text_column=config["dataset"]["text_column"],
        label_column=target_column,
        task_type="multiclass",
    )
    device = choose_device(training_config["device"])
    model = create_tier2_model(
        config,
        vocab_size=vocabulary.size,
        pad_index=vocabulary.pad_index,
        output_dim=len(class_names),
    ).to(device)

    class_weight_tensor = build_class_weight_tensor(class_weights, class_ids).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    best_state = None
    best_macro_f1 = -1.0
    best_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0
    history = {"epochs": []}
    total_train_start = time.perf_counter()

    for epoch in range(1, training_config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        example_count = 0
        for batch in dataloaders["train"]:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            example_count += batch_size

        train_loss = epoch_loss / max(example_count, 1)
        val_loss, val_targets, val_predictions = predict_ffnn(
            model=model,
            dataloader=dataloaders,
            device=device,
            criterion=criterion,
        )
        evaluation = evaluate_predictions(
            y_true=val_targets,
            y_pred=val_predictions,
            confusion_matrix_path=output_dir / "ffnn_confusion_matrix.png",
            task_type="multiclass",
            class_names=class_names,
            title="FFNN Validation Confusion Matrix",
        )
        val_macro_f1 = evaluation["metrics"]["macro_f1"]
        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": evaluation["metrics"],
            }
        )
        logging.info(
            "Epoch %s | train_loss=%.4f | val_loss=%.4f | val_macro_f1=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_macro_f1,
        )

        if val_macro_f1 > best_macro_f1:
            best_state = copy.deepcopy(model.state_dict())
            best_macro_f1 = val_macro_f1
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training_config["patience"]:
            logging.info("Early stopping triggered at epoch %s", epoch)
            break

    total_train_seconds = time.perf_counter() - total_train_start
    if best_state is None:
        raise RuntimeError("FFNN training did not produce a valid multiclass checkpoint.")

    model.load_state_dict(best_state)
    predict_start = time.perf_counter()
    val_loss, val_targets, val_predictions = predict_ffnn(
        model=model,
        dataloader=dataloaders,
        device=device,
        criterion=criterion,
    )
    predict_seconds = time.perf_counter() - predict_start
    evaluation = evaluate_predictions(
        y_true=val_targets,
        y_pred=val_predictions,
        confusion_matrix_path=output_dir / "ffnn_confusion_matrix.png",
        task_type="multiclass",
        class_names=class_names,
        title="FFNN Validation Confusion Matrix",
    )
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_loss
    history["device"] = str(device)
    history["vocab_size"] = vocabulary.size
    history["max_length"] = sequence_config["max_length"]

    results = {
        "tier": "Tier 2",
        "model_type": type(model).__name__,
        "validation": evaluation,
        "timing_seconds": {
            "train": total_train_seconds,
            "predict": predict_seconds,
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "device": str(device),
        "vocab_size": vocabulary.size,
    }
    return results, history


def build_summary(
    config: dict[str, Any],
    split_frames,
    tier1_results: dict[str, Any],
    tier2_result: dict[str, Any],
    class_weight_artifact: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build the structured Experiment 2 summary."""
    linear_svc_macro_f1 = tier1_results["linear_svc"]["validation"]["metrics"]["macro_f1"]
    ffnn_macro_f1 = tier2_result["validation"]["metrics"]["macro_f1"]
    warning = None
    if ffnn_macro_f1 <= linear_svc_macro_f1:
        warning = (
            "Tier 2 FFNN did not beat Tier 1 linear_svc on validation macro F1. "
            "Try larger embeddings, wider hidden layers, longer sequence limits, or a 1D-CNN."
        )
        logging.warning(warning)

    all_results = {**tier1_results, "ffnn": tier2_result}
    best_model = max(
        all_results.items(),
        key=lambda item: item[1]["validation"]["metrics"]["macro_f1"],
    )[0]
    output_paths = get_output_paths(config)

    return {
        "iteration": config["experiment"]["iteration"],
        "experiment": "Experiment 2",
        "task": "multiclass_classification",
        "dataset": {
            "raw_data": str(resolve_project_path(config["paths"]["raw_data"])),
            "train_rows": int(len(split_frames["train"])),
            "val_rows": int(len(split_frames["val"])),
            "test_rows": int(len(split_frames["test"])),
            "label_map_path": str(output_paths["label_map"]),
        },
        "class_weights_path": str(resolve_project_path(config["paths"]["class_weights_path"])),
        "class_weights": class_weight_artifact,
        "results": all_results,
        "best_validation_model": best_model,
        "warning": warning,
        "runtime_seconds": runtime_seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multiclass baselines for Iteration 2: Experiment 2."
    )
    parser.add_argument(
        "--config",
        default="configs/exp2_multiclass.yaml",
        help="Path to the Iteration 2 multiclass config.",
    )
    parser.add_argument(
        "--force-splits",
        action="store_true",
        help="Regenerate the canonical split files before training.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for Iteration 2: Experiment 2 multiclass training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(config["training"]["seed"])

    output_dir = resolve_project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    create_multiclass_splits(config, force=args.force_splits)
    split_frames = load_multiclass_splits(config)
    class_weights, class_weight_artifact = prepare_class_weight_artifact(config, split_frames)

    overall_start = time.perf_counter()
    tier1_results = train_tier1_models(config, split_frames, output_dir, class_weights)
    tier2_result, history = train_tier2_ffnn(config, split_frames, output_dir, class_weights)
    runtime_seconds = time.perf_counter() - overall_start

    save_json(history, output_dir / "ffnn_history.json")
    summary = build_summary(
        config=config,
        split_frames=split_frames,
        tier1_results=tier1_results,
        tier2_result=tier2_result,
        class_weight_artifact=class_weight_artifact,
        runtime_seconds=runtime_seconds,
    )
    summary_path = save_json(summary, output_dir / "summary.json")
    logging.info("Saved Experiment 2 summary to %s", summary_path)


if __name__ == "__main__":
    main()
