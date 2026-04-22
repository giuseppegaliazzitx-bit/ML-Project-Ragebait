from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from ragebait_detector.data.preprocessing import clean_text, detect_language, normalize_label
from ragebait_detector.utils.io import dump_json, ensure_parent, read_csv, write_csv


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


def evaluate_checkpoint_on_labeled_csv(
    *,
    checkpoint_path: str | Path,
    input_path: str | Path,
    settings,
    output_dir: str | Path,
    force_english: bool = True,
) -> dict[str, Any]:
    from ragebait_detector.inference import RageBaitPredictor

    predictor_settings = deepcopy(settings)
    if force_english:
        predictor_settings.data.drop_non_english = False

    predictor = RageBaitPredictor.from_checkpoint(checkpoint_path, predictor_settings)
    rows = read_csv(input_path)

    predictions: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    abstained_reasons: Counter[str] = Counter()
    invalid_label_rows = 0

    for index, row in enumerate(rows):
        text = _resolve_text(row, settings)
        true_label = _resolve_label(row, settings)
        if true_label is None:
            invalid_label_rows += 1

        detected_language = detect_language(text) if text.strip() else "unknown"
        result = predictor.predict_text(text)
        if result.reason:
            abstained_reasons[result.reason] += 1

        predictions.append(
            {
                "row_index": index,
                "post_id": row.get("post_id", ""),
                "source": row.get("source", ""),
                "language": row.get("language", ""),
                "detected_language": detected_language,
                "text": text,
                "clean_text": clean_text(text),
                "true_label": "" if true_label is None else true_label,
                "predicted_label": "" if result.label is None else result.label,
                "confidence": f"{result.confidence:.6f}",
                "prob_not_ragebait": f"{result.probabilities['not_ragebait']:.6f}",
                "prob_ragebait": f"{result.probabilities['ragebait']:.6f}",
                "reason": result.reason or "",
                "chunks_scored": result.chunks_scored,
            }
        )

        if true_label is not None and result.label is not None:
            y_true.append(true_label)
            y_pred.append(result.label)

    destination = ensure_parent(Path(output_dir) / "summary.json").parent
    predictions_path = destination / "predictions.csv"
    metrics_path = destination / "metrics.json"
    matrix_path = destination / "confusion_matrix.png"

    write_csv(
        predictions_path,
        predictions,
        [
            "row_index",
            "post_id",
            "source",
            "language",
            "detected_language",
            "text",
            "clean_text",
            "true_label",
            "predicted_label",
            "confidence",
            "prob_not_ragebait",
            "prob_ragebait",
            "reason",
            "chunks_scored",
        ],
    )

    metrics: dict[str, Any] | None = None
    if y_true:
        metrics = compute_classification_metrics(y_true, y_pred)
        save_metrics_report(metrics, metrics_path)
        plot_confusion_matrix(
            y_true,
            y_pred,
            matrix_path,
            title="Manual Evaluation - Rage-Bait Detector",
        )

    summary = {
        "input_path": str(input_path),
        "checkpoint_path": str(checkpoint_path),
        "force_english": force_english,
        "total_rows": len(rows),
        "valid_labeled_rows": len(rows) - invalid_label_rows,
        "invalid_label_rows": invalid_label_rows,
        "scored_rows": len(y_true),
        "abstained_rows": sum(abstained_reasons.values()),
        "scored_coverage_ratio": (
            len(y_true) / max(len(rows) - invalid_label_rows, 1)
            if (len(rows) - invalid_label_rows) > 0
            else 0.0
        ),
        "abstained_reasons": dict(abstained_reasons),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path) if metrics is not None else None,
        "confusion_matrix_path": str(matrix_path) if metrics is not None else None,
        "metrics": metrics,
    }
    dump_json(destination / "summary.json", summary)
    return summary


def _resolve_text(row: dict[str, str], settings) -> str:
    candidate_keys = [
        settings.data.text_column,
        "text",
        "tweet",
        "raw_text",
        "post_text",
        "content",
    ]
    for key in candidate_keys:
        value = row.get(key)
        if value and value.strip():
            return value
    return ""


def _resolve_label(row: dict[str, str], settings) -> int | None:
    candidate_keys = [
        settings.data.label_column,
        "label",
        "is_ragebait",
        "gold_label",
        "manual_label",
    ]
    for key in candidate_keys:
        value = row.get(key)
        normalized = normalize_label(value)
        if normalized is not None:
            return normalized
    return None
