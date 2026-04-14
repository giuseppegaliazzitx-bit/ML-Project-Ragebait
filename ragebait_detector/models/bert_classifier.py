from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class BertRageBaitClassifier(nn.Module):
    def __init__(self, model_name: str, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(encoder_hidden_size, hidden_dim)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]

        hidden = self.dropout(self.activation(self.hidden(self.dropout(pooled))))
        logits = self.classifier(hidden).squeeze(-1)
        probabilities = torch.softmax(
            torch.stack([torch.zeros_like(logits), logits], dim=-1),
            dim=-1,
        )
        return {
            "logits": logits,
            "probabilities": probabilities,
            "positive_probability": probabilities[:, 1],
        }


@dataclass
class ModelBundle:
    model: BertRageBaitClassifier
    tokenizer: AutoTokenizer


def build_model_bundle(
    model_name: str,
    hidden_dim: int,
    dropout: float,
) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertRageBaitClassifier(
        model_name=model_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    return ModelBundle(model=model, tokenizer=tokenizer)


def load_checkpoint(
    checkpoint_path: str | Path,
    model_name: str,
    hidden_dim: int,
    dropout: float,
    device: torch.device,
) -> ModelBundle:
    bundle = build_model_bundle(model_name, hidden_dim, dropout)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bundle.model.load_state_dict(state["model_state_dict"])
    bundle.model.to(device)
    bundle.model.eval()
    return bundle
