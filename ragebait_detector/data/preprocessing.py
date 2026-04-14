from __future__ import annotations

import random
import re
from typing import Any

from ragebait_detector.config import Settings
from ragebait_detector.utils.dependencies import MissingDependencyError, require_dependency
from ragebait_detector.utils.io import read_csv, write_csv

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
NUMBER_PATTERN = re.compile(r"\d+")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALPHA_PATTERN = re.compile(r"[^a-z_\s\[\]]+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
SPECIAL_TOKENS = {
    "[url]",
    "[user]",
    "[hashtag]",
    "[emoji]",
    "[empty_post]",
}


def normalize_label(value: str | int | None) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "ragebait", "rage-bait", "positive", "yes", "true"}:
        return 1
    if normalized in {"0", "not_ragebait", "not-ragebait", "negative", "no", "false"}:
        return 0
    return None


def detect_language(text: str) -> str:
    if not text.strip():
        return "unknown"
    try:
        langdetect = require_dependency("langdetect")
    except MissingDependencyError:
        ascii_ratio = sum(character.isascii() for character in text) / max(len(text), 1)
        return "en" if ascii_ratio > 0.95 else "unknown"
    return str(langdetect.detect(text))


def clean_text(text: str) -> str:
    if not text or not text.strip():
        return "[empty_post]"

    normalized = text.lower()
    normalized = URL_PATTERN.sub(" [url] ", normalized)
    normalized = MENTION_PATTERN.sub(" [user] ", normalized)
    normalized = HASHTAG_PATTERN.sub(" [hashtag] ", normalized)
    normalized = EMOJI_PATTERN.sub(" [emoji] ", normalized)
    normalized = NUMBER_PATTERN.sub(" ", normalized)
    normalized = NON_ALPHA_PATTERN.sub(" ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized or "[empty_post]"


def is_media_only_or_empty(cleaned_text: str) -> bool:
    tokens = [token for token in cleaned_text.split() if token not in SPECIAL_TOKENS]
    return not tokens


def meaningful_length(cleaned_text: str) -> int:
    tokens = [token for token in cleaned_text.split() if token not in SPECIAL_TOKENS]
    return sum(len(token) for token in tokens)


def augment_text(text: str, seed: int | None = None) -> str:
    rng = random.Random(seed)
    tokens = text.split()
    movable_positions = [
        index
        for index, token in enumerate(tokens)
        if token not in SPECIAL_TOKENS and len(token) > 2
    ]
    if len(movable_positions) < 4:
        return text

    chosen = movable_positions[:]
    rng.shuffle(chosen)
    window = chosen[: min(6, len(chosen))]
    shuffled_tokens = [tokens[index] for index in window]
    rng.shuffle(shuffled_tokens)

    augmented = tokens[:]
    for index, replacement in zip(window, shuffled_tokens):
        augmented[index] = replacement
    return " ".join(augmented)


def prepare_labeled_dataset(
    input_path: str,
    output_path: str,
    settings: Settings,
) -> dict[str, Any]:
    rows = read_csv(input_path)
    processed_rows: list[dict[str, str | int | bool]] = []
    dropped_empty = 0
    dropped_non_english = 0
    dropped_unlabeled = 0

    for row in rows:
        label = normalize_label(row.get(settings.data.label_column))
        if label is None:
            dropped_unlabeled += 1
            continue

        raw_text = row.get(settings.data.text_column, "")
        detected_language = detect_language(raw_text)
        is_supported_language = detected_language in settings.data.supported_languages
        cleaned = clean_text(raw_text)
        is_empty = is_media_only_or_empty(cleaned)
        is_too_short = meaningful_length(cleaned) < settings.data.min_text_length

        if is_empty or is_too_short:
            dropped_empty += 1
            continue
        if settings.data.drop_non_english and not is_supported_language:
            dropped_non_english += 1
            continue

        processed_rows.append(
            {
                "post_id": row.get("post_id", ""),
                "raw_text": raw_text,
                "clean_text": cleaned,
                "label": label,
                "source": row.get("source", ""),
                "language": row.get("language", "unknown"),
                "detected_language": detected_language,
                "is_supported_language": is_supported_language,
                "was_augmented": False,
            }
        )

    if settings.data.augment_minority_class and processed_rows:
        processed_rows.extend(
            build_augmented_rows(
                processed_rows,
                copies=settings.data.augmentation_copies,
                seed=settings.training.seed,
            )
        )

    fieldnames = [
        "post_id",
        "raw_text",
        "clean_text",
        "label",
        "source",
        "language",
        "detected_language",
        "is_supported_language",
        "was_augmented",
    ]
    write_csv(output_path, processed_rows, fieldnames)

    return {
        "processed_rows": len(processed_rows),
        "dropped_empty": dropped_empty,
        "dropped_non_english": dropped_non_english,
        "dropped_unlabeled": dropped_unlabeled,
    }


def build_augmented_rows(
    rows: list[dict[str, Any]],
    copies: int,
    seed: int,
) -> list[dict[str, Any]]:
    labels = [int(row["label"]) for row in rows]
    if not labels:
        return []

    class_counts = {label: labels.count(label) for label in set(labels)}
    minority_label = min(class_counts, key=class_counts.get)
    minority_rows = [row for row in rows if int(row["label"]) == minority_label]
    rng = random.Random(seed)
    augmented_rows: list[dict[str, Any]] = []

    for copy_index in range(copies):
        sampled_rows = minority_rows[:]
        rng.shuffle(sampled_rows)
        for row in sampled_rows:
            augmented_rows.append(
                {
                    **row,
                    "post_id": f"{row['post_id']}::aug::{copy_index}",
                    "clean_text": augment_text(
                        str(row["clean_text"]),
                        seed=rng.randint(0, 10_000_000),
                    ),
                    "was_augmented": True,
                }
            )
    return augmented_rows
