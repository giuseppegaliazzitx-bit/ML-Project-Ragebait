from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ragebait_detector.data.dataset import chunk_for_inference
from ragebait_detector.data.preprocessing import clean_text, detect_language, is_media_only_or_empty
from ragebait_detector.models.bert_classifier import load_checkpoint
from ragebait_detector.training.trainer import resolve_device


@dataclass
class PredictionResult:
    label: int | None
    confidence: float
    probabilities: dict[str, float]
    reason: str | None = None
    chunks_scored: int = 1


class RageBaitPredictor:
    def __init__(self, model, tokenizer, settings, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, settings):
        device = resolve_device(settings.training.device)
        bundle = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model_name=settings.model.model_name,
            hidden_dim=settings.model.hidden_dim,
            dropout=settings.model.dropout,
            device=device,
        )
        return cls(bundle.model, bundle.tokenizer, settings, device)

    def predict_text(self, text: str) -> PredictionResult:
        cleaned = clean_text(text)
        if is_media_only_or_empty(cleaned):
            return PredictionResult(
                label=None,
                confidence=0.0,
                probabilities={"not_ragebait": 0.0, "ragebait": 0.0},
                reason="empty_or_media_only",
                chunks_scored=0,
            )

        detected_language = detect_language(text)
        if (
            self.settings.data.drop_non_english
            and detected_language not in self.settings.data.supported_languages
        ):
            return PredictionResult(
                label=None,
                confidence=0.0,
                probabilities={"not_ragebait": 0.0, "ragebait": 0.0},
                reason=f"unsupported_language:{detected_language}",
                chunks_scored=0,
            )

        chunks = chunk_for_inference(
            cleaned,
            tokenizer=self.tokenizer,
            max_length=self.settings.model.max_length,
            stride=self.settings.model.stride,
        )
        rage_scores = [self._score_chunk(chunk) for chunk in chunks]
        ragebait_probability = sum(rage_scores) / len(rage_scores)
        not_ragebait_probability = 1.0 - ragebait_probability
        label = int(ragebait_probability >= self.settings.training.decision_threshold)
        confidence = max(ragebait_probability, not_ragebait_probability)

        return PredictionResult(
            label=label,
            confidence=confidence,
            probabilities={
                "not_ragebait": not_ragebait_probability,
                "ragebait": ragebait_probability,
            },
            reason=None,
            chunks_scored=len(chunks),
        )

    def _score_chunk(self, text: str) -> float:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.settings.model.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoded)
        return float(outputs["positive_probability"].detach().cpu().item())

