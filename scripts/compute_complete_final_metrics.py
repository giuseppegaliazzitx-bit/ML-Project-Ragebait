"""Compute complete final validation/test metrics for the ragebait project.

This script is intentionally kept outside iteration2/ so it can write final
evaluation artifacts under deliverables/ without overwriting saved experiment
outputs. It retrains the cheap baselines from the frozen splits and can
optionally retrain binary BERT to recover the missing held-out test metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITERATION2_ROOT = PROJECT_ROOT / "iteration2"
if str(ITERATION2_ROOT) not in sys.path:
    sys.path.insert(0, str(ITERATION2_ROOT))

from src.data.make_dataset import get_target_column, load_config, load_dataset_splits
from src.data.preprocessing import (
    build_class_weight_tensor,
    build_ffnn_dataloaders,
    build_tfidf_features,
    compute_class_weights,
)
from src.data.transformer_dataset import build_transformer_dataloaders
from src.evaluation.evaluate import evaluate_logits, evaluate_predictions, logits_to_predictions, save_json
from src.models.baselines import create_tier1_models, create_tier2_model
from src.models.bert_classifier import create_adamw_optimizer, create_bert_classifier, create_linear_scheduler


def resolve_iteration2_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ITERATION2_ROOT / path


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def choose_device(device_name: str = "auto") -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def task_info(config: dict[str, Any]) -> tuple[str, str, list[str]]:
    task_type = config.get("task", {}).get("type", "binary")
    if task_type == "multiclass":
        return task_type, get_target_column(config), config["dataset"]["label_order"]
    return task_type, "binary_label", ["Normal", "Ragebait"]


def train_and_evaluate_tier1(
    config: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, Any]:
    task_type, target_column, class_names = task_info(config)
    text_column = config["dataset"]["text_column"]
    vectorizer, features = build_tfidf_features(
        split_frames["train"],
        split_frames["val"],
        split_frames["test"],
        config["preprocessing"]["tfidf"],
        text_column=text_column,
    )

    class_weights = None
    if task_type == "multiclass":
        class_ids = list(range(len(class_names)))
        class_weights = compute_class_weights(
            split_frames["train"][target_column].to_numpy(),
            class_ids=class_ids,
        )

    y_train = split_frames["train"][target_column].to_numpy()
    y_val = split_frames["val"][target_column].to_numpy()
    y_test = split_frames["test"][target_column].to_numpy()
    models = create_tier1_models(config, class_weight=class_weights)
    results: dict[str, Any] = {}

    for model_name, model in models.items():
        train_start = time.perf_counter()
        model.fit(features["train"], y_train)
        train_seconds = time.perf_counter() - train_start

        val_start = time.perf_counter()
        val_pred = model.predict(features["val"])
        val_seconds = time.perf_counter() - val_start

        test_start = time.perf_counter()
        test_pred = model.predict(features["test"])
        test_seconds = time.perf_counter() - test_start

        kwargs = {}
        if task_type == "multiclass":
            kwargs = {"task_type": "multiclass", "class_names": class_names}

        val_eval = evaluate_predictions(
            y_true=y_val,
            y_pred=val_pred,
            confusion_matrix_path=output_dir / f"{model_name}_val_confusion_matrix.png",
            title=f"{model_name} Validation Confusion Matrix",
            **kwargs,
        )
        test_eval = evaluate_predictions(
            y_true=y_test,
            y_pred=test_pred,
            confusion_matrix_path=output_dir / f"{model_name}_test_confusion_matrix.png",
            title=f"{model_name} Test Confusion Matrix",
            **kwargs,
        )
        results[model_name] = {
            "tier": "Tier 1",
            "model_type": type(model).__name__,
            "validation": val_eval,
            "test": test_eval,
            "timing_seconds": {
                "train": train_seconds,
                "validation_predict": val_seconds,
                "test_predict": test_seconds,
            },
            "tfidf_vocabulary_size": len(vectorizer.vocabulary_),
        }
    return results


def predict_ffnn(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    criterion,
    task_type: str,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            if task_type == "multiclass":
                predictions = logits.argmax(dim=1)
                targets = labels.long()
            else:
                predictions = (torch.sigmoid(logits) >= 0.5).long()
                targets = labels.long()

            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
    return total_loss / max(total_examples, 1), all_targets, all_predictions


def train_and_evaluate_ffnn(
    config: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_type, target_column, class_names = task_info(config)
    training_config = config["training"]
    sequence_config = config["preprocessing"]["sequence"]
    dataloaders, vocabulary = build_ffnn_dataloaders(
        split_frames["train"],
        split_frames["val"],
        split_frames["test"],
        sequence_config=sequence_config,
        batch_size=training_config["batch_size"],
        seed=training_config["seed"],
        text_column=config["dataset"]["text_column"],
        label_column=target_column,
        task_type=task_type,
    )
    device = choose_device(training_config.get("device", "auto"))
    output_dim = 1 if task_type == "binary" else len(class_names)
    model = create_tier2_model(
        config,
        vocab_size=vocabulary.size,
        pad_index=vocabulary.pad_index,
        output_dim=output_dim,
    ).to(device)

    if task_type == "multiclass":
        class_ids = list(range(len(class_names)))
        class_weights = compute_class_weights(
            split_frames["train"][target_column].to_numpy(),
            class_ids=class_ids,
        )
        criterion = torch.nn.CrossEntropyLoss(
            weight=build_class_weight_tensor(class_weights, class_ids).to(device)
        )
        selection_metric = "macro_f1"
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        selection_metric = "f1"

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    best_state = None
    best_metric = -1.0
    best_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: dict[str, Any] = {"epochs": []}
    train_start = time.perf_counter()

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
            epoch_loss += loss.item() * labels.size(0)
            example_count += labels.size(0)

        train_loss = epoch_loss / max(example_count, 1)
        val_loss, val_targets, val_predictions = predict_ffnn(
            model, dataloaders["val"], device, criterion, task_type
        )
        eval_kwargs = {}
        if task_type == "multiclass":
            eval_kwargs = {"task_type": "multiclass", "class_names": class_names}
        val_eval = evaluate_predictions(
            y_true=val_targets,
            y_pred=val_predictions,
            confusion_matrix_path=output_dir / "ffnn_val_confusion_matrix.png",
            title="FFNN Validation Confusion Matrix",
            **eval_kwargs,
        )
        metric_value = val_eval["metrics"][selection_metric]
        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_eval["metrics"],
            }
        )
        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= training_config["patience"]:
            break

    train_seconds = time.perf_counter() - train_start
    if best_state is None:
        raise RuntimeError("FFNN training did not produce a best state.")

    model.load_state_dict(best_state)
    eval_kwargs = {}
    if task_type == "multiclass":
        eval_kwargs = {"task_type": "multiclass", "class_names": class_names}

    val_start = time.perf_counter()
    val_loss, val_targets, val_predictions = predict_ffnn(
        model, dataloaders["val"], device, criterion, task_type
    )
    val_seconds = time.perf_counter() - val_start
    val_eval = evaluate_predictions(
        y_true=val_targets,
        y_pred=val_predictions,
        confusion_matrix_path=output_dir / "ffnn_val_confusion_matrix.png",
        title="FFNN Validation Confusion Matrix",
        **eval_kwargs,
    )

    test_start = time.perf_counter()
    test_loss, test_targets, test_predictions = predict_ffnn(
        model, dataloaders["test"], device, criterion, task_type
    )
    test_seconds = time.perf_counter() - test_start
    test_eval = evaluate_predictions(
        y_true=test_targets,
        y_pred=test_predictions,
        confusion_matrix_path=output_dir / "ffnn_test_confusion_matrix.png",
        title="FFNN Test Confusion Matrix",
        **eval_kwargs,
    )

    history.update(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "device": str(device),
            "vocab_size": vocabulary.size,
            "max_length": sequence_config["max_length"],
        }
    )
    result = {
        "tier": "Tier 2",
        "model_type": type(model).__name__,
        "validation": val_eval,
        "test": test_eval,
        "timing_seconds": {
            "train": train_seconds,
            "validation_predict": val_seconds,
            "test_predict": test_seconds,
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "device": str(device),
        "vocab_size": vocabulary.size,
    }
    return result, history


def compute_bce_targets(labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_labels).float()


def evaluate_bert_split(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    config: dict[str, Any],
    confusion_matrix_path: Path,
    title: str,
    predictions_path: Path | None = None,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels: list[int] = []
    all_row_ids: list[int] = []
    logits_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            targets = compute_bce_targets(labels, num_labels=config["model"]["num_labels"])
            loss = criterion(logits, targets)
            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_row_ids.extend(batch["row_id"].cpu().tolist())
            logits_batches.append(logits.cpu().numpy())

    logits_array = np.concatenate(logits_batches, axis=0)
    evaluation = evaluate_logits(
        y_true=all_labels,
        logits=logits_array,
        confusion_matrix_path=confusion_matrix_path,
        title=title,
    )
    if predictions_path is not None:
        exp_logits = np.exp(logits_array - np.max(logits_array, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        predictions = logits_to_predictions(logits_array)
        pd.DataFrame(
            {
                "row_id": all_row_ids,
                "true_label": all_labels,
                "predicted_label": predictions,
                "confidence": probabilities.max(axis=1),
                "normal_probability": probabilities[:, 0],
                "ragebait_probability": probabilities[:, 1],
            }
        ).to_csv(predictions_path, index=False)
    return total_loss / max(total_examples, 1), evaluation


def train_binary_bert(config: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    set_global_seed(config["training"]["seed"], deterministic=True)
    dataloaders, tokenizer, split_frames = build_transformer_dataloaders(config)
    tokenizer.save_pretrained(output_dir / "tokenizer")

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
        "test_rows": int(len(split_frames["test"])),
        "max_length": config["model"]["max_length"],
        "tokenizer_dir": str(output_dir / "tokenizer"),
    }
    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    best_val_loss = float("inf")
    train_start = time.perf_counter()

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
                logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss_total += loss.item() * labels.size(0)
            train_examples += labels.size(0)

        train_loss = train_loss_total / max(train_examples, 1)
        val_loss, val_eval = evaluate_bert_split(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            config=config,
            confusion_matrix_path=output_dir / "bert_val_confusion_matrix.png",
            title="BERT Binary Validation Confusion Matrix",
        )
        val_metrics = val_eval["metrics"]
        epoch_seconds = time.perf_counter() - epoch_start
        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "epoch_seconds": epoch_seconds,
            }
        )
        print(
            f"BERT epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.4f} "
            f"seconds={epoch_seconds:.2f}",
            flush=True,
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    train_seconds = time.perf_counter() - train_start
    if best_state is None:
        raise RuntimeError("Binary BERT training did not produce a best state.")
    model.load_state_dict(best_state)

    val_start = time.perf_counter()
    val_loss, val_eval = evaluate_bert_split(
        model=model,
        dataloader=dataloaders["val"],
        criterion=criterion,
        device=device,
        config=config,
        confusion_matrix_path=output_dir / "bert_val_confusion_matrix.png",
        title="BERT Binary Validation Confusion Matrix",
        predictions_path=output_dir / "bert_val_predictions.csv",
    )
    val_seconds = time.perf_counter() - val_start

    test_start = time.perf_counter()
    test_loss, test_eval = evaluate_bert_split(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
        config=config,
        confusion_matrix_path=output_dir / "bert_test_confusion_matrix.png",
        title="BERT Binary Test Confusion Matrix",
        predictions_path=output_dir / "bert_test_predictions.csv",
    )
    test_seconds = time.perf_counter() - test_start

    history.update({"best_epoch": best_epoch, "best_val_loss": best_val_loss})
    result = {
        "tier": "Tier 3",
        "model_type": type(model).__name__,
        "pretrained_model": config["model"]["pretrained_name"],
        "validation": val_eval,
        "test": test_eval,
        "timing_seconds": {
            "train": train_seconds,
            "validation_predict": val_seconds,
            "test_predict": test_seconds,
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "device": str(device),
        "max_length": config["model"]["max_length"],
    }
    return result, history


def load_existing_multiclass_bert() -> dict[str, Any]:
    summary_path = ITERATION2_ROOT / "outputs/exp2_multiclass_bert/summary.json"
    history_path = ITERATION2_ROOT / "outputs/exp2_multiclass_bert/training_history.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    bert = summary["results"]["bert_base_uncased"]
    return {
        **bert,
        "validation": {
            "metrics": history["epochs"][-1]["val_metrics"],
            "source": str(history_path),
        },
        "test": bert["validation"],
        "test_source_note": "Existing summary field is named validation, but it was produced on the held-out test dataloader.",
    }


def train_existing_multiclass_bert(config: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retrain multiclass BERT through the project training implementation."""
    from src.training.train_multiclass_bert import (
        set_global_seed as set_multiclass_bert_seed,
        train_bert_model as train_multiclass_bert_model,
    )

    set_multiclass_bert_seed(config["training"]["seed"])
    config = copy.deepcopy(config)
    config["paths"]["output_dir"] = str(output_dir)
    config["paths"]["tokenizer_dir"] = str(output_dir / "tokenizer")
    config["paths"]["checkpoint_path"] = str(Path("/tmp") / "ragebait_complete_multiclass_bert_best.pt")
    config["paths"]["hard_errors_path"] = str(output_dir / "hard_errors.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    result, history, _split_frames = train_multiclass_bert_model(config)
    test_evaluation = result["validation"]
    result = {
        **result,
        "validation": {
            "metrics": history["epochs"][-1]["val_metrics"],
            "source": "training_history_last_epoch",
        },
        "test": test_evaluation,
        "test_source_note": "The project trainer returns the held-out test evaluation in the field named validation.",
    }
    checkpoint_path = Path(config["paths"]["checkpoint_path"])
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return result, history


def run_complete_metrics(run_binary_bert: bool, run_multiclass_bert: bool) -> dict[str, Any]:
    output_dir = PROJECT_ROOT / "deliverables/final_assets/complete_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_config = load_config(ITERATION2_ROOT / "configs/exp1_binary.yaml")
    multiclass_config = load_config(ITERATION2_ROOT / "configs/exp2_multiclass.yaml")
    set_global_seed(binary_config["training"]["seed"])

    results: dict[str, Any] = {
        "generated_at_unix": time.time(),
        "output_dir": str(output_dir),
        "binary": {},
        "multiclass": {},
    }

    binary_splits = load_dataset_splits(binary_config)
    binary_out = output_dir / "binary"
    binary_out.mkdir(parents=True, exist_ok=True)
    results["binary"]["tier1"] = train_and_evaluate_tier1(binary_config, binary_splits, binary_out)
    binary_ffnn, binary_ffnn_history = train_and_evaluate_ffnn(binary_config, binary_splits, binary_out)
    results["binary"]["ffnn"] = binary_ffnn
    save_json(binary_ffnn_history, binary_out / "ffnn_history.json")

    if run_binary_bert:
        bert_config = load_config(ITERATION2_ROOT / "configs/exp1_binary_bert.yaml")
        # Use the current YAML as the canonical rerun configuration.
        bert_config["paths"]["output_dir"] = str(output_dir / "binary_bert")
        bert_config["paths"]["tokenizer_dir"] = str(output_dir / "binary_bert/tokenizer")
        bert_config["paths"]["checkpoint_path"] = str(Path("/tmp") / "ragebait_complete_binary_bert_best.pt")
        bert_out = output_dir / "binary_bert"
        bert_out.mkdir(parents=True, exist_ok=True)
        bert_result, bert_history = train_binary_bert(bert_config, bert_out)
        results["binary"]["bert_base_uncased"] = bert_result
        save_json(bert_history, bert_out / "training_history.json")
    else:
        results["binary"]["bert_base_uncased"] = {
            "status": "not_run",
            "reason": "Run with --run-binary-bert to retrain binary BERT on CPU and evaluate the test split.",
        }

    multiclass_splits = load_dataset_splits(multiclass_config)
    multiclass_out = output_dir / "multiclass"
    multiclass_out.mkdir(parents=True, exist_ok=True)
    results["multiclass"]["tier1"] = train_and_evaluate_tier1(
        multiclass_config, multiclass_splits, multiclass_out
    )
    multiclass_ffnn, multiclass_ffnn_history = train_and_evaluate_ffnn(
        multiclass_config, multiclass_splits, multiclass_out
    )
    results["multiclass"]["ffnn"] = multiclass_ffnn
    save_json(multiclass_ffnn_history, multiclass_out / "ffnn_history.json")
    if run_multiclass_bert:
        multiclass_bert_config = load_config(ITERATION2_ROOT / "configs/exp2_multiclass_bert.yaml")
        multiclass_bert_result, multiclass_bert_history = train_existing_multiclass_bert(
            multiclass_bert_config,
            output_dir / "multiclass_bert",
        )
        results["multiclass"]["bert_base_uncased"] = multiclass_bert_result
        save_json(multiclass_bert_history, output_dir / "multiclass_bert/training_history.json")
    else:
        results["multiclass"]["bert_base_uncased"] = load_existing_multiclass_bert()

    save_json(results, output_dir / "complete_metrics.json")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute complete final metrics.")
    parser.add_argument(
        "--run-binary-bert",
        action="store_true",
        help="Retrain binary BERT and evaluate the held-out test split.",
    )
    parser.add_argument(
        "--run-multiclass-bert",
        action="store_true",
        help="Retrain multiclass BERT and evaluate the held-out test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_complete_metrics(
        run_binary_bert=args.run_binary_bert,
        run_multiclass_bert=args.run_multiclass_bert,
    )
    output = Path(results["output_dir"]) / "complete_metrics.json"
    print(f"Saved complete metrics to {output}")


if __name__ == "__main__":
    main()
