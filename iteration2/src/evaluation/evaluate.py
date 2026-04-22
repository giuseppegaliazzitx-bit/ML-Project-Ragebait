"""Evaluation utilities for Iteration 2 experiments."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_binary_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate binary metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def calculate_multiclass_metrics(
    y_true,
    y_pred,
    class_names: list[str],
) -> dict[str, Any]:
    """Calculate multiclass metrics with per-class reporting."""
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        class_name: {
            "precision": float(report[class_name]["precision"]),
            "recall": float(report[class_name]["recall"]),
            "f1": float(report[class_name]["f1-score"]),
            "support": int(report[class_name]["support"]),
        }
        for class_name in class_names
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "classification_report": per_class,
    }


def save_confusion_matrix(
    y_true,
    y_pred,
    output_path: str | Path,
    labels: list[int] | None = None,
    display_labels: list[str] | tuple[str, ...] | None = None,
    title: str = "Validation Confusion Matrix",
) -> Path:
    """Save a confusion matrix PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    if display_labels is None:
        display_labels = [str(label) for label in labels]

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_size = max(len(labels), 2)
    figure, axis = plt.subplots(figsize=(max(5.0, matrix_size * 1.3), max(4.5, matrix_size * 1.1)))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=display_labels,
    ).plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(title)
    axis.tick_params(axis="x", labelrotation=35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def evaluate_predictions(
    y_true,
    y_pred,
    confusion_matrix_path: str | Path,
    task_type: str = "binary",
    class_names: list[str] | None = None,
    title: str = "Validation Confusion Matrix",
) -> dict[str, Any]:
    """Evaluate predictions and save the confusion matrix artifact."""
    if task_type == "multiclass":
        if class_names is None:
            raise ValueError("class_names must be provided for multiclass evaluation")
        metrics = calculate_multiclass_metrics(y_true, y_pred, class_names=class_names)
        labels = list(range(len(class_names)))
        display_labels = class_names
    else:
        metrics = calculate_binary_metrics(y_true, y_pred)
        labels = [0, 1]
        display_labels = class_names or ["Normal", "Ragebait"]

    saved_path = save_confusion_matrix(
        y_true,
        y_pred,
        confusion_matrix_path,
        labels=labels,
        display_labels=display_labels,
        title=title,
    )
    return {
        "metrics": metrics,
        "confusion_matrix_path": str(saved_path),
    }


def logits_to_predictions(logits) -> np.ndarray:
    """Convert model logits into hard class predictions."""
    logits_array = np.asarray(logits)
    if logits_array.ndim == 1:
        probabilities = 1.0 / (1.0 + np.exp(-logits_array))
        return (probabilities >= 0.5).astype(int)
    if logits_array.ndim == 2 and logits_array.shape[1] == 1:
        probabilities = 1.0 / (1.0 + np.exp(-logits_array[:, 0]))
        return (probabilities >= 0.5).astype(int)
    return logits_array.argmax(axis=-1).astype(int)


def evaluate_logits(
    y_true,
    logits,
    confusion_matrix_path: str | Path,
    task_type: str = "binary",
    class_names: list[str] | None = None,
    title: str = "Validation Confusion Matrix",
) -> dict[str, Any]:
    """Evaluate raw logits by first mapping them to hard predictions."""
    predictions = logits_to_predictions(logits)
    evaluation = evaluate_predictions(
        y_true=y_true,
        y_pred=predictions,
        confusion_matrix_path=confusion_matrix_path,
        task_type=task_type,
        class_names=class_names,
        title=title,
    )
    evaluation["logits_shape"] = list(np.asarray(logits).shape)
    return evaluation


def save_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Persist a JSON payload."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path
