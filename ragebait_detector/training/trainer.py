from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler

from ragebait_detector.data.dataset import BertTextDataset, build_collate_fn, build_sample_weights
from ragebait_detector.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    save_metrics_report,
)
from ragebait_detector.models.bert_classifier import build_model_bundle
from ragebait_detector.utils.io import dump_json, ensure_parent


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    best_score: float = float("-inf")
    bad_epochs: int = 0

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def resolve_device(preference: str) -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_bert_classifier(
    splits,
    settings,
    output_dir: str | Path,
) -> dict[str, Any]:
    device = resolve_device(settings.training.device)
    bundle = build_model_bundle(
        model_name=settings.model.model_name,
        tokenizer_name=settings.model.tokenizer_name,
        hidden_dim=settings.model.hidden_dim,
        dropout=settings.model.dropout,
    )
    bundle.model.to(device)

    checkpoint_dir = ensure_parent(Path(output_dir) / "bert" / "checkpoint.pt").parent
    tokenizer_dir = checkpoint_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    bundle.tokenizer.save_pretrained(tokenizer_dir)

    train_dataset = BertTextDataset(splits.train)
    validation_dataset = BertTextDataset(splits.validation)
    test_dataset = BertTextDataset(splits.test)

    sample_weights = build_sample_weights([row["label"] for row in splits.train])
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    collate_fn = build_collate_fn(bundle.tokenizer, settings.model.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.training.batch_size,
        sampler=sampler,
        num_workers=settings.training.num_workers,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        num_workers=settings.training.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        num_workers=settings.training.num_workers,
        collate_fn=collate_fn,
    )

    labels = [row["label"] for row in splits.train]
    positive_count = max(1, sum(labels))
    negative_count = max(1, len(labels) - positive_count)
    pos_weight = torch.tensor(negative_count / positive_count, device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(
        bundle.model.parameters(),
        lr=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )
    stopper = EarlyStopping(
        patience=settings.training.patience,
        min_delta=settings.training.min_delta,
    )

    history: list[dict[str, Any]] = []
    best_checkpoint_path = checkpoint_dir / "checkpoint.pt"
    best_validation_f1 = float("-inf")

    for epoch in range(1, settings.training.epochs + 1):
        train_loss = _run_training_epoch(
            model=bundle.model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        validation_metrics = _evaluate_model(
            model=bundle.model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
            threshold=settings.training.decision_threshold,
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["metrics"]["accuracy"],
            "validation_f1_ragebait": validation_metrics["metrics"]["f1_by_class"]["1"],
        }
        history.append(epoch_result)

        current_f1 = validation_metrics["metrics"]["f1_by_class"]["1"]
        if current_f1 > best_validation_f1 + settings.training.min_delta:
            best_validation_f1 = current_f1
            torch.save(
                {
                    "model_state_dict": bundle.model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics["metrics"],
                },
                best_checkpoint_path,
            )
        if stopper.step(current_f1):
            break

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    bundle.model.load_state_dict(checkpoint["model_state_dict"])
    bundle.model.eval()

    test_metrics = _evaluate_model(
        model=bundle.model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        threshold=settings.training.decision_threshold,
    )

    metrics_path = checkpoint_dir / "test_metrics.json"
    matrix_path = checkpoint_dir / "test_confusion_matrix.png"
    history_path = checkpoint_dir / "training_history.json"
    artifact_path = checkpoint_dir / "artifacts.json"

    save_metrics_report(test_metrics["metrics"], metrics_path)
    plot_confusion_matrix(
        test_metrics["labels"],
        test_metrics["predictions"],
        matrix_path,
        title="BERT Rage-Bait Detector",
    )
    dump_json(history_path, {"history": history})
    dump_json(
        artifact_path,
        {
            "checkpoint_path": str(best_checkpoint_path),
            "tokenizer_dir": str(tokenizer_dir),
            "device": str(device),
        },
    )

    return {
        "history": history,
        "test_metrics": test_metrics["metrics"],
        "checkpoint_path": str(best_checkpoint_path),
        "tokenizer_dir": str(tokenizer_dir),
        "artifacts_path": str(artifact_path),
    }


def _run_training_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        labels = batch.pop("labels").to(device)
        batch.pop("texts", None)
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / max(len(dataloader), 1)


def _evaluate_model(
    model,
    dataloader,
    criterion,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    labels_list: list[int] = []
    probabilities: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch.pop("texts", None)
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            loss = criterion(outputs["logits"], labels)
            losses.append(loss.item())

            probs = outputs["positive_probability"].detach().cpu().tolist()
            batch_predictions = [1 if probability >= threshold else 0 for probability in probs]
            probabilities.extend(probs)
            predictions.extend(batch_predictions)
            labels_list.extend(labels.detach().cpu().int().tolist())

    metrics = compute_classification_metrics(labels_list, predictions)
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "metrics": metrics,
        "labels": labels_list,
        "predictions": predictions,
        "probabilities": probabilities,
    }
