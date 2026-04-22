"""Baseline models for Iteration 2 experiments."""

from __future__ import annotations

from typing import Any

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torch import nn


def _get_task_type(config: dict[str, Any]) -> str:
    return config.get("task", {}).get("type", "binary").lower()


def create_tier1_models(
    config: dict[str, Any],
    class_weight: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Instantiate Tier 1 classical baselines for the configured task."""
    seed = config["training"]["seed"]
    task_type = _get_task_type(config)
    logistic_config = config["models"]["logistic_regression"]
    svc_config = config["models"]["linear_svc"]

    if task_type == "multiclass":
        return {
            "logistic_regression": LogisticRegression(
                C=logistic_config["c"],
                max_iter=logistic_config["max_iter"],
                random_state=seed,
                solver=logistic_config.get("solver", "lbfgs"),
                class_weight=class_weight,
            ),
            "linear_svc": LinearSVC(
                C=svc_config["c"],
                random_state=seed,
                class_weight=class_weight,
            ),
        }

    return {
        "logistic_regression": LogisticRegression(
            C=logistic_config["c"],
            max_iter=logistic_config["max_iter"],
            random_state=seed,
            solver="liblinear",
            class_weight=class_weight,
        ),
        "linear_svc": LinearSVC(
            C=svc_config["c"],
            random_state=seed,
            class_weight=class_weight,
        ),
    }


class FFNNClassifier(nn.Module):
    """Simple sequence-fed FFNN for Iteration 2 baselines."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float,
        pad_index: int,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.pad_index = pad_index
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)

        layers: list[nn.Module] = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        mask = (input_ids != self.pad_index).unsqueeze(-1)
        masked_embeddings = embeddings * mask
        pooled_embeddings = masked_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.classifier(pooled_embeddings)
        return logits.squeeze(-1) if self.output_dim == 1 else logits


def create_tier2_model(
    config: dict[str, Any],
    vocab_size: int,
    pad_index: int,
    output_dim: int | None = None,
) -> FFNNClassifier:
    """Instantiate the Tier 2 FFNN for the configured task."""
    model_config = config["models"]["ffnn"]
    task_type = _get_task_type(config)
    if output_dim is None:
        output_dim = 1 if task_type == "binary" else len(config["dataset"]["label_order"])
    return FFNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=model_config["embedding_dim"],
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config["dropout"],
        pad_index=pad_index,
        output_dim=output_dim,
    )
