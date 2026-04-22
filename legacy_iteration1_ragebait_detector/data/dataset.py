from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ragebait_detector.utils.io import read_csv

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    Dataset = object


@dataclass
class DatasetSplits:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]


def load_processed_records(path: str) -> list[dict[str, Any]]:
    rows = read_csv(path)
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                **row,
                "label": int(row["label"]),
                "is_supported_language": str(row["is_supported_language"]).lower() == "true",
                "was_augmented": str(row["was_augmented"]).lower() == "true",
            }
        )
    return records


def stratified_split(
    records: list[dict[str, Any]],
    validation_size: float,
    test_size: float,
    seed: int,
) -> DatasetSplits:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[int(record["label"])].append(record)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for label, bucket in buckets.items():
        rng.shuffle(bucket)
        bucket_size = len(bucket)
        test_count = max(1, math.floor(bucket_size * test_size))
        validation_count = max(1, math.floor(bucket_size * validation_size))

        test.extend(bucket[:test_count])
        validation.extend(bucket[test_count : test_count + validation_count])
        train.extend(bucket[test_count + validation_count :])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return DatasetSplits(train=train, validation=validation, test=test)


def compute_class_weights(labels: list[int]) -> dict[int, float]:
    total = len(labels)
    unique_labels = sorted(set(labels))
    return {
        label: total / (len(unique_labels) * labels.count(label))
        for label in unique_labels
    }


def build_sample_weights(labels: list[int]) -> list[float]:
    class_weights = compute_class_weights(labels)
    return [class_weights[label] for label in labels]


class BertTextDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], text_key: str = "clean_text") -> None:
        self.rows = rows
        self.text_key = text_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {"text": row[self.text_key], "label": int(row["label"])}


def build_collate_fn(tokenizer, max_length: int):
    if torch is None:  # pragma: no cover - requires torch
        raise RuntimeError("PyTorch must be installed to create a BERT collate function.")

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        encoded["texts"] = texts
        return encoded

    return collate


def chunk_for_inference(
    text: str,
    tokenizer,
    max_length: int,
    stride: int,
) -> list[str]:
    tokens = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,
    )["input_ids"]
    if len(tokens) <= max_length - 2:
        return [text]

    chunks: list[str] = []
    step = max_length - stride - 2
    for start in range(0, len(tokens), max(step, 1)):
        chunk_ids = tokens[start : start + max_length - 2]
        if not chunk_ids:
            continue
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if start + max_length - 2 >= len(tokens):
            break
    return chunks or [text]
