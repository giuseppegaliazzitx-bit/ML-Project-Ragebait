"""Deep Error Analysis for Multiclass BERT predictions."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from src.evaluation.evaluate import logits_to_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path from the Iteration 2 project root."""
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def perform_deep_error_analysis(
    split_frame: pd.DataFrame,
    logits: np.ndarray,
    config: dict[str, Any]
) -> None:
    """
    Perform deep error diagnostics by finding 'High Confidence, Wrong Answer' 
    cases and extracting specific confusion pairs programmatically.
    """
    class_names = config["dataset"]["label_order"]
    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    # Calculate softmax probabilities programmatically from logits
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Extract predicted classes and confidence
    predictions = logits_to_predictions(logits)
    confidence = np.max(probabilities, axis=1)
    
    # Ensure dataframe and arrays align
    df = split_frame.copy()
    df["predicted_label_id"] = predictions
    df["predicted_label_name"] = [class_names[p] for p in predictions]
    df["true_label_name"] = [class_names[t] for t in df[config["dataset"]["label_column"]]]
    df["confidence"] = confidence
    
    # Save the logits as string array for inspection
    df["logits"] = [str(list(np.round(l, 4))) for l in logits]
    
    # Filter for all misclassifications
    misclassified = df[df[config["dataset"]["label_column"]] != df["predicted_label_id"]]
    
    # Isolate specific confusion pairs programmatically:
    # 1. "Trolling" vs "Derogatory"
    # 2. "Profanity" vs "Hate Speech"
    trolling_id = label_to_id["Trolling"]
    derogatory_id = label_to_id["Derogatory"]
    profanity_id = label_to_id["Profanity"]
    hate_speech_id = label_to_id["Hate Speech"]
    
    mask_troll_derog = (
        (misclassified[config["dataset"]["label_column"]] == trolling_id) & (misclassified["predicted_label_id"] == derogatory_id)
    ) | (
        (misclassified[config["dataset"]["label_column"]] == derogatory_id) & (misclassified["predicted_label_id"] == trolling_id)
    )
    
    mask_prof_hate = (
        (misclassified[config["dataset"]["label_column"]] == profanity_id) & (misclassified["predicted_label_id"] == hate_speech_id)
    ) | (
        (misclassified[config["dataset"]["label_column"]] == hate_speech_id) & (misclassified["predicted_label_id"] == profanity_id)
    )
    
    # Combine the masks to filter down to just our target hard error pairs
    target_errors = misclassified[mask_troll_derog | mask_prof_hate]
    
    # Sort by confidence descending to find "High Confidence, Wrong Answer" cases
    target_errors = target_errors.sort_values(by="confidence", ascending=False)
    
    # Select final columns to save
    columns_to_save = [
        config["dataset"]["text_column"], 
        "true_label_name", 
        "predicted_label_name", 
        "confidence", 
        "logits"
    ]
    
    hard_errors_path = resolve_project_path(config["paths"]["hard_errors_path"])
    hard_errors_path.parent.mkdir(parents=True, exist_ok=True)
    target_errors[columns_to_save].to_csv(hard_errors_path, index=False)
