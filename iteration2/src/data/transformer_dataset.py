"""Transformer data loading for Iteration 2: Experiment 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path from the Iteration 2 project root."""
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_transformer_splits(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load the canonical processed splits for transformer training."""
    split_paths = {
        "train": resolve_project_path(config["paths"]["train_split"]),
        "val": resolve_project_path(config["paths"]["val_split"]),
        "test": resolve_project_path(config["paths"]["test_split"]),
    }
    missing_paths = [str(path) for path in split_paths.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing canonical processed split files. Expected: "
            + ", ".join(missing_paths)
        )
    return {
        split_name: pd.read_csv(path)
        for split_name, path in split_paths.items()
    }


def load_bert_tokenizer(config: dict[str, Any]):
    """Load the HuggingFace tokenizer for Iteration 2: Experiment 1."""
    model_config = config["model"]
    return AutoTokenizer.from_pretrained(
        model_config["pretrained_name"],
        use_fast=model_config.get("use_fast", True),
        local_files_only=model_config.get("local_files_only", True),
    )


class TransformerTextDataset(Dataset):
    """Tokenized dataset wrapper for Iteration 2: Experiment 1."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        text_column: str,
        label_column: str,
        max_length: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        tokenized = self.tokenizer(
            str(row[self.text_column]),
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        tokenized["labels"] = int(row[self.label_column])
        tokenized["row_id"] = int(row["row_id"])
        return tokenized


class TransformerBatchCollator:
    """Dynamic-padding collator for Iteration 2: Experiment 1 BERT batches."""

    def __init__(self, tokenizer) -> None:
        self.padder = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([feature.pop("labels") for feature in features], dtype=torch.long)
        row_ids = torch.tensor([feature.pop("row_id") for feature in features], dtype=torch.long)
        batch = self.padder(features)
        batch["labels"] = labels
        batch["row_id"] = row_ids
        return batch


def build_transformer_dataloaders(config: dict[str, Any]):
    """Build train, validation, and test dataloaders for BERT fine-tuning."""
    split_frames = load_transformer_splits(config)
    tokenizer = load_bert_tokenizer(config)
    collator = TransformerBatchCollator(tokenizer=tokenizer)

    dataset_config = config["dataset"]
    model_config = config["model"]
    training_config = config["training"]
    train_dataset = TransformerTextDataset(
        frame=split_frames["train"],
        tokenizer=tokenizer,
        text_column=dataset_config["text_column"],
        label_column=dataset_config["label_column"],
        max_length=model_config["max_length"],
    )
    val_dataset = TransformerTextDataset(
        frame=split_frames["val"],
        tokenizer=tokenizer,
        text_column=dataset_config["text_column"],
        label_column=dataset_config["label_column"],
        max_length=model_config["max_length"],
    )
    test_dataset = TransformerTextDataset(
        frame=split_frames["test"],
        tokenizer=tokenizer,
        text_column=dataset_config["text_column"],
        label_column=dataset_config["label_column"],
        max_length=model_config["max_length"],
    )

    generator = torch.Generator()
    generator.manual_seed(training_config["seed"])
    num_workers = training_config.get("num_workers", 0)
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=training_config["train_batch_size"],
            shuffle=True,
            collate_fn=collator,
            generator=generator,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=training_config.get("eval_batch_size", training_config["train_batch_size"]),
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=training_config.get("eval_batch_size", training_config["train_batch_size"]),
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
        ),
    }
    return dataloaders, tokenizer, split_frames
