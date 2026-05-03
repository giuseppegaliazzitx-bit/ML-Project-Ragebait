"""Render top-level final asset charts with the report flowchart theme."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-ragebait")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "deliverables" / "final_assets"
METRICS_PATH = ASSET_DIR / "complete_metrics" / "complete_metrics.json"

BG = "#f7f9fc"
INK = "#172033"
MUTED = "#4f5f73"
GRID = "#cbd5e1"
BLUE = "#2563eb"
TEAL = "#0f766e"
AMBER = "#b45309"
GREEN = "#15803d"
RED = "#b91c1c"
PURPLE = "#7c3aed"
SLATE = "#64748b"

MODEL_LABELS = {
    "logistic_regression": "Logistic\nRegression",
    "linear_svc": "Linear\nSVC",
    "ffnn": "FFNN",
    "bert_base_uncased": "BERT",
}
MODEL_ORDER = ["logistic_regression", "linear_svc", "ffnn", "bert_base_uncased"]
MODEL_COLORS = [TEAL, BLUE, AMBER, RED]


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def model_block(metrics: dict, task: str, model: str) -> dict:
    if model in metrics[task].get("tier1", {}):
        return metrics[task]["tier1"][model]
    return metrics[task][model]


def test_metric(metrics: dict, task: str, model: str, key: str) -> float:
    return model_block(metrics, task, model)["test"]["metrics"][key]


def timing_value(metrics: dict, task: str, model: str) -> tuple[float, float]:
    timing = model_block(metrics, task, model)["timing_seconds"]
    train = timing["train"]
    if "test_predict" in timing:
        predict = timing["test_predict"]
    else:
        predict = timing["predict"]
    return train, predict


def figure_for(filename: str) -> tuple[plt.Figure, plt.Axes]:
    sizes = {
        "binary_model_comparison.png": (2354, 1341),
        "multiclass_model_comparison.png": (2354, 1341),
        "classwise_f1_comparison.png": (2354, 1341),
        "binary_vs_multiclass_bert.png": (2162, 1209),
        "compute_time_comparison.png": (2702, 1220),
    }
    width, height = sizes[filename]
    dpi = 170
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(colors=MUTED, labelsize=14)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=1.2, alpha=0.7)
    ax.set_axisbelow(True)


def title(ax: plt.Axes, headline: str, subtitle: str) -> None:
    ax.text(0, 1.12, headline, transform=ax.transAxes, ha="left", va="bottom", fontsize=26, fontweight="bold", color=INK)
    ax.text(0, 1.06, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=15, color=MUTED)


def label_bars(ax: plt.Axes, bars, *, size: int = 13, fmt: str = "{:.4f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=size,
            color=INK,
            fontweight="bold",
        )


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(ASSET_DIR / filename, facecolor=BG, bbox_inches=None)
    plt.close(fig)


def binary_model_comparison(metrics: dict) -> None:
    fig, ax = figure_for("binary_model_comparison.png")
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    accuracy = [test_metric(metrics, "binary", m, "accuracy") for m in MODEL_ORDER]
    f1 = [test_metric(metrics, "binary", m, "f1") for m in MODEL_ORDER]
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color=BLUE)
    bars2 = ax.bar(x + width / 2, f1, width, label="F1", color=TEAL)
    title(ax, "Binary Held-out Performance", "BERT leads on both accuracy and positive-class F1; lexical baselines remain competitive.")
    ax.set_ylim(0.78, 0.95)
    ax.set_ylabel("Score", color=MUTED, fontsize=15)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=14)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.01), fontsize=14)
    style_axis(ax)
    label_bars(ax, bars1, size=11)
    label_bars(ax, bars2, size=11)
    fig.tight_layout(pad=4)
    save(fig, "binary_model_comparison.png")


def multiclass_model_comparison(metrics: dict) -> None:
    fig, ax = figure_for("multiclass_model_comparison.png")
    x = np.arange(len(MODEL_ORDER))
    width = 0.28
    accuracy = [test_metric(metrics, "multiclass", m, "accuracy") for m in MODEL_ORDER]
    macro = [test_metric(metrics, "multiclass", m, "macro_f1") for m in MODEL_ORDER]
    weighted = [test_metric(metrics, "multiclass", m, "weighted_f1") for m in MODEL_ORDER]
    ax.bar(x - width, accuracy, width, label="Accuracy", color=BLUE)
    ax.bar(x, macro, width, label="Macro F1", color=RED)
    ax.bar(x + width, weighted, width, label="Weighted F1", color=TEAL)
    title(ax, "Five-class Held-out Performance", "Macro F1 shows the minority-class challenge more clearly than accuracy.")
    ax.set_ylim(0.48, 0.80)
    ax.set_ylabel("Score", color=MUTED, fontsize=15)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=14)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=14)
    style_axis(ax)
    for container in ax.containers:
        label_bars(ax, container, size=10)
    fig.tight_layout(pad=4)
    save(fig, "multiclass_model_comparison.png")


def classwise_f1_comparison(metrics: dict) -> None:
    fig, ax = figure_for("classwise_f1_comparison.png")
    classes = ["Normal", "Profanity", "Trolling", "Derogatory", "Hate Speech"]
    logit_report = model_block(metrics, "multiclass", "logistic_regression")["test"]["metrics"]["classification_report"]
    bert_report = model_block(metrics, "multiclass", "bert_base_uncased")["test"]["metrics"]["classification_report"]
    logit = [logit_report[c]["f1"] for c in classes]
    bert = [bert_report[c]["f1"] for c in classes]
    x = np.arange(len(classes))
    width = 0.36
    ax.bar(x - width / 2, logit, width, label="Logistic Regression", color=SLATE)
    ax.bar(x + width / 2, bert, width, label="BERT", color=RED)
    title(ax, "Class-wise F1 on the Held-out Test Split", "BERT improves every class, while Derogatory and Hate Speech remain the hardest boundaries.")
    ax.set_ylim(0.20, 0.98)
    ax.set_ylabel("F1", color=MUTED, fontsize=15)
    ax.set_xticks(x, ["Normal", "Profanity", "Trolling", "Derogatory", "Hate\nSpeech"], fontsize=14)
    ax.legend(frameon=False, loc="upper right", fontsize=14)
    style_axis(ax)
    for container in ax.containers:
        label_bars(ax, container, size=10)
    fig.tight_layout(pad=4)
    save(fig, "classwise_f1_comparison.png")


def binary_vs_multiclass(metrics: dict) -> None:
    fig, ax = figure_for("binary_vs_multiclass_bert.png")
    labels = ["Binary\nF1", "Multiclass\nMacro F1"]
    values = [
        test_metric(metrics, "binary", "bert_base_uncased", "f1"),
        test_metric(metrics, "multiclass", "bert_base_uncased", "macro_f1"),
    ]
    bars = ax.bar(labels, values, color=[TEAL, RED], width=0.55)
    title(ax, "Task Framing Changes Difficulty", "Binary detection collapses abuse labels; multiclass prediction must separate overlapping categories.")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", color=MUTED, fontsize=15)
    style_axis(ax)
    label_bars(ax, bars, size=16)
    fig.tight_layout(pad=4)
    save(fig, "binary_vs_multiclass_bert.png")


def compute_time(metrics: dict) -> None:
    fig, ax = figure_for("compute_time_comparison.png")
    labels = [
        "Binary\nLogReg",
        "Binary\nSVC",
        "Binary\nFFNN",
        "Binary\nBERT",
        "Multi\nLogReg",
        "Multi\nSVC",
        "Multi\nFFNN",
        "Multi\nBERT",
    ]
    pairs = [
        ("binary", "logistic_regression"),
        ("binary", "linear_svc"),
        ("binary", "ffnn"),
        ("binary", "bert_base_uncased"),
        ("multiclass", "logistic_regression"),
        ("multiclass", "linear_svc"),
        ("multiclass", "ffnn"),
        ("multiclass", "bert_base_uncased"),
    ]
    train = []
    predict = []
    for task, model in pairs:
        t, p = timing_value(metrics, task, model)
        train.append(t)
        predict.append(p)
    x = np.arange(len(labels))
    width = 0.36
    bars1 = ax.bar(x - width / 2, train, width, label="Train", color=BLUE)
    bars2 = ax.bar(x + width / 2, predict, width, label="Predict / evaluate", color=AMBER)
    title(ax, "Training and Evaluation Time", "Log scale keeps classical baselines and BERT visible on the same chart.")
    ax.set_yscale("log")
    ax.set_ylabel("Seconds, log scale", color=MUTED, fontsize=15)
    ax.set_xticks(x, labels, fontsize=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=14)
    style_axis(ax)
    label_bars(ax, bars1, size=9, fmt="{:.3g}")
    label_bars(ax, bars2, size=9, fmt="{:.3g}")
    fig.tight_layout(pad=4)
    save(fig, "compute_time_comparison.png")


def main() -> None:
    metrics = load_metrics()
    binary_model_comparison(metrics)
    multiclass_model_comparison(metrics)
    classwise_f1_comparison(metrics)
    binary_vs_multiclass(metrics)
    compute_time(metrics)


if __name__ == "__main__":
    main()
