from __future__ import annotations

from pathlib import Path
from typing import Any

from ragebait_detector.utils.io import dump_json, ensure_parent


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["not_ragebait", "ragebait"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy,
        "precision_by_class": {"0": precision[0], "1": precision[1]},
        "recall_by_class": {"0": recall[0], "1": recall[1]},
        "f1_by_class": {"0": f1[0], "1": f1[1]},
        "support_by_class": {"0": int(support[0]), "1": int(support[1])},
        "report": report,
    }
    return _to_builtin(metrics)


def save_metrics_report(metrics: dict[str, Any], output_path: str | Path) -> Path:
    return dump_json(output_path, metrics)


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    output_path: str | Path,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    destination = ensure_parent(output_path)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Rage-Bait", "Rage-Bait"],
        yticklabels=["Not Rage-Bait", "Rage-Bait"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()
    return destination
