"""BERT classifier module for Iteration 2: Experiment 1."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup


class BertSequenceClassifier(nn.Module):
    """Fine-tunable BERT classifier for Iteration 2: Experiment 1."""

    def __init__(
        self,
        pretrained_name: str,
        num_labels: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        local_files_only: bool,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            local_files_only=local_files_only,
        )
        self.encoder = AutoModel.from_pretrained(
            pretrained_name,
            config=self.config,
            local_files_only=local_files_only,
        )
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute classification logits for a batch."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled_output))


def create_bert_classifier(config: dict[str, Any]) -> BertSequenceClassifier:
    """Instantiate the Tier 3 BERT classifier for GPU-accelerated fine-tuning."""
    model_config = config["model"]
    return BertSequenceClassifier(
        pretrained_name=model_config["pretrained_name"],
        num_labels=model_config["num_labels"],
        hidden_dropout_prob=model_config.get("hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=model_config.get("attention_probs_dropout_prob", 0.1),
        local_files_only=model_config.get("local_files_only", True),
    )


def create_adamw_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Create an AdamW optimizer with standard BERT parameter grouping."""
    training_config = config["training"]
    no_decay = {"bias", "LayerNorm.bias", "LayerNorm.weight"}
    parameter_groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and not any(term in name for term in no_decay)
            ],
            "weight_decay": training_config["weight_decay"],
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and any(term in name for term in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=training_config["learning_rate"],
        eps=training_config.get("adam_epsilon", 1e-8),
    )


def create_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    config: dict[str, Any],
):
    """Create a linear warmup-decay scheduler for BERT fine-tuning."""
    training_config = config["training"]
    warmup_ratio = training_config.get("warmup_ratio", 0.0)
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
