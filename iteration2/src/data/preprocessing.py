"""Feature engineering and weighting for Iteration 2 experiments."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset

TOKEN_PATTERN = re.compile(r"\b\w+\b|[^\w\s]")


def _tokenize(text: str, lowercase: bool) -> list[str]:
    normalized = str(text).strip()
    if lowercase:
        normalized = normalized.lower()
    return TOKEN_PATTERN.findall(normalized)


def build_tfidf_features(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    tfidf_config: dict[str, Any],
    text_column: str = "text",
):
    """Build TF-IDF matrices for Tier 1 models."""
    vectorizer = TfidfVectorizer(
        max_features=tfidf_config["max_features"],
        ngram_range=tuple(tfidf_config["ngram_range"]),
        min_df=tfidf_config["min_df"],
        max_df=tfidf_config["max_df"],
        sublinear_tf=tfidf_config.get("sublinear_tf", True),
    )
    x_train = vectorizer.fit_transform(train_frame[text_column])
    x_val = vectorizer.transform(val_frame[text_column])
    x_test = vectorizer.transform(test_frame[text_column])
    return vectorizer, {"train": x_train, "val": x_val, "test": x_test}


def compute_class_weights(labels: Iterable[int], class_ids: Iterable[int]) -> dict[int, float]:
    """Compute exact balanced class weights from the training labels."""
    labels_array = np.asarray(list(labels), dtype=np.int64)
    class_id_list = [int(class_id) for class_id in class_ids]
    total_examples = int(labels_array.shape[0])
    num_classes = len(class_id_list)
    class_weights: dict[int, float] = {}

    for class_id in class_id_list:
        class_count = int((labels_array == class_id).sum())
        if class_count == 0:
            raise ValueError(f"Cannot compute class weight for missing class id {class_id}")
        class_weights[class_id] = float(total_examples / (num_classes * class_count))
    return class_weights


def build_class_weight_tensor(
    class_weights: dict[int, float],
    class_ids: Iterable[int],
) -> torch.Tensor:
    """Build a torch tensor of class weights ordered by class id."""
    ordered_weights = [class_weights[int(class_id)] for class_id in class_ids]
    return torch.tensor(ordered_weights, dtype=torch.float32)


def summarize_class_distribution(
    labels: Iterable[int],
    class_ids: Iterable[int],
    class_names: list[str] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Summarize counts and shares for each class."""
    labels_array = np.asarray(list(labels), dtype=np.int64)
    total_examples = int(labels_array.shape[0])
    class_id_list = [int(class_id) for class_id in class_ids]
    distribution: dict[str, dict[str, float | int]] = {}

    for index, class_id in enumerate(class_id_list):
        class_name = class_names[index] if class_names is not None else str(class_id)
        class_count = int((labels_array == class_id).sum())
        distribution[class_name] = {
            "index": class_id,
            "count": class_count,
            "share": float(class_count / total_examples if total_examples else 0.0),
        }
    return distribution


@dataclass(frozen=True)
class Vocabulary:
    """Vocabulary container for Iteration 2 sequence features."""

    token_to_id: dict[str, int]

    @property
    def pad_index(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_index(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def size(self) -> int:
        return len(self.token_to_id)


def build_vocabulary(
    texts: Iterable[str],
    max_vocab_size: int,
    min_freq: int,
    lowercase: bool,
) -> Vocabulary:
    """Build a train-only vocabulary for Tier 2 sequence encoding."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(_tokenize(text, lowercase=lowercase))

    token_to_id = {"<pad>": 0, "<unk>": 1}
    most_common_tokens = [
        token for token, count in counter.most_common() if count >= min_freq
    ]
    available_slots = max(max_vocab_size - len(token_to_id), 0)
    for token in most_common_tokens[:available_slots]:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    return Vocabulary(token_to_id=token_to_id)


def encode_text(
    text: str,
    vocabulary: Vocabulary,
    max_length: int,
    lowercase: bool,
) -> tuple[np.ndarray, int]:
    """Encode a single text into a fixed-length token id sequence."""
    tokens = _tokenize(text, lowercase=lowercase)[:max_length]
    sequence = [vocabulary.token_to_id.get(token, vocabulary.unk_index) for token in tokens]
    length = len(sequence)
    if length < max_length:
        sequence.extend([vocabulary.pad_index] * (max_length - length))
    return np.asarray(sequence, dtype=np.int64), max(length, 1)


class SequenceDataset(Dataset):
    """Torch dataset for Tier 2 FFNN training."""

    def __init__(
        self,
        texts: Iterable[str],
        labels: Iterable[int],
        vocabulary: Vocabulary,
        max_length: int,
        lowercase: bool,
        label_dtype: torch.dtype,
    ) -> None:
        sequences = []
        lengths = []
        for text in texts:
            encoded, length = encode_text(text, vocabulary, max_length, lowercase=lowercase)
            sequences.append(encoded)
            lengths.append(length)

        self.input_ids = torch.tensor(np.stack(sequences), dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.labels = torch.tensor(list(labels), dtype=label_dtype)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "lengths": self.lengths[index],
            "labels": self.labels[index],
        }


def build_ffnn_dataloaders(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    sequence_config: dict[str, Any],
    batch_size: int,
    seed: int,
    text_column: str = "text",
    label_column: str = "binary_label",
    task_type: str = "binary",
):
    """Build dataloaders and a train-only vocabulary for Tier 2 models."""
    vocabulary = build_vocabulary(
        texts=train_frame[text_column],
        max_vocab_size=sequence_config["max_vocab_size"],
        min_freq=sequence_config["min_freq"],
        lowercase=sequence_config.get("lowercase", True),
    )
    label_dtype = torch.long if task_type == "multiclass" else torch.float32
    datasets = {
        "train": SequenceDataset(
            train_frame[text_column],
            train_frame[label_column],
            vocabulary=vocabulary,
            max_length=sequence_config["max_length"],
            lowercase=sequence_config.get("lowercase", True),
            label_dtype=label_dtype,
        ),
        "val": SequenceDataset(
            val_frame[text_column],
            val_frame[label_column],
            vocabulary=vocabulary,
            max_length=sequence_config["max_length"],
            lowercase=sequence_config.get("lowercase", True),
            label_dtype=label_dtype,
        ),
        "test": SequenceDataset(
            test_frame[text_column],
            test_frame[label_column],
            vocabulary=vocabulary,
            max_length=sequence_config["max_length"],
            lowercase=sequence_config.get("lowercase", True),
            label_dtype=label_dtype,
        ),
    }

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }
    return dataloaders, vocabulary
