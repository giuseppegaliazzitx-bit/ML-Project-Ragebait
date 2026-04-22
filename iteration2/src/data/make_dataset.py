"""Dataset creation for Iteration 2 experiments."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load an Iteration 2 YAML configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def get_task_type(config: dict[str, Any]) -> str:
    """Return the task type for the provided config."""
    task_type = config.get("task", {}).get("type", "binary").lower()
    if task_type not in {"binary", "multiclass"}:
        raise ValueError(f"Unsupported task type: {task_type}")
    return task_type


def get_target_column(config: dict[str, Any]) -> str:
    """Return the encoded target column name."""
    dataset_config = config["dataset"]
    task_type = get_task_type(config)
    default_target = "binary_label" if task_type == "binary" else "multiclass_label"
    return dataset_config.get("target_column", default_target)


def get_target_name_column(config: dict[str, Any]) -> str:
    """Return the encoded target-name column name."""
    dataset_config = config["dataset"]
    task_type = get_task_type(config)
    default_target_name = (
        "binary_label_name" if task_type == "binary" else "multiclass_label_name"
    )
    return dataset_config.get("target_name_column", default_target_name)


def get_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve processed artifact paths for the configured task."""
    paths_config = config["paths"]
    processed_dir = _resolve_path(paths_config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    task_type = get_task_type(config)
    prefix = paths_config.get(
        "split_prefix",
        "binary" if task_type == "binary" else "multiclass",
    )
    return {
        "train": _resolve_path(paths_config.get("train_split", processed_dir / f"{prefix}_train.csv")),
        "val": _resolve_path(paths_config.get("val_split", processed_dir / f"{prefix}_val.csv")),
        "test": _resolve_path(paths_config.get("test_split", processed_dir / f"{prefix}_test.csv")),
        "manifest": _resolve_path(
            paths_config.get("split_manifest_path", processed_dir / f"{prefix}_split_manifest.json")
        ),
        "label_map": _resolve_path(
            paths_config.get("label_map_path", processed_dir / f"{prefix}_label_map.json")
        ),
    }


def _prepare_raw_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    dataset_config = config["dataset"]
    raw_path = _resolve_path(config["paths"]["raw_data"])
    text_column = dataset_config["text_column"]
    label_column = dataset_config["label_column"]

    dataframe = pd.read_csv(raw_path)
    missing_columns = {text_column, label_column} - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {raw_path}: {sorted(missing_columns)}")

    dataframe = dataframe[[text_column, label_column]].copy()
    dataframe[text_column] = dataframe[text_column].fillna("").astype(str).str.strip()
    dataframe[label_column] = dataframe[label_column].fillna("").astype(str).str.strip()
    dataframe = dataframe.loc[
        (dataframe[text_column] != "") & (dataframe[label_column] != "")
    ].reset_index(drop=True)
    dataframe.insert(0, "row_id", range(len(dataframe)))
    return dataframe


def _build_binary_label_map(dataframe: pd.DataFrame, config: dict[str, Any]) -> dict[str, int]:
    dataset_config = config["dataset"]
    normal_label = dataset_config["normal_label"]
    observed_labels = sorted(dataframe[dataset_config["label_column"]].unique())
    return {
        label: 0 if label == normal_label else 1
        for label in observed_labels
    }


def _build_multiclass_label_map(config: dict[str, Any]) -> dict[str, int]:
    dataset_config = config["dataset"]
    return {
        label: index
        for index, label in enumerate(dataset_config["label_order"])
    }


def _prepare_dataframe(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    dataframe = _prepare_raw_dataframe(config)
    dataset_config = config["dataset"]
    label_column = dataset_config["label_column"]
    target_column = get_target_column(config)
    target_name_column = get_target_name_column(config)
    task_type = get_task_type(config)

    if task_type == "binary":
        label_map = _build_binary_label_map(dataframe, config)
        positive_label_name = dataset_config["positive_label_name"]
        normal_label = dataset_config["normal_label"]
        dataframe[target_column] = dataframe[label_column].map(label_map).astype(int)
        dataframe[target_name_column] = dataframe[target_column].map(
            {0: normal_label, 1: positive_label_name}
        )
        return dataframe, label_map

    label_map = _build_multiclass_label_map(config)
    unknown_labels = sorted(set(dataframe[label_column]) - set(label_map))
    if unknown_labels:
        raise ValueError(f"Found unexpected labels for multiclass task: {unknown_labels}")

    dataframe[target_column] = dataframe[label_column].map(label_map).astype(int)
    dataframe[target_name_column] = dataframe[label_column]
    return dataframe, label_map


def create_dataset_splits(config: dict[str, Any], force: bool = False) -> dict[str, Path]:
    """Create canonical split files for the configured Iteration 2 task."""
    output_paths = get_output_paths(config)
    if not force and all(path.exists() for path in output_paths.values()):
        return output_paths

    dataset_config = config["dataset"]
    split_config = dataset_config["split"]
    label_column = dataset_config["label_column"]
    target_column = get_target_column(config)

    train_size = split_config["train_size"]
    val_size = split_config["val_size"]
    test_size = split_config["test_size"]
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, found {total}")

    dataframe, label_map = _prepare_dataframe(config)
    stratify_column = dataset_config.get("stratify_column", label_column)
    temp_size = val_size + test_size
    train_frame, temp_frame = train_test_split(
        dataframe,
        test_size=temp_size,
        stratify=dataframe[stratify_column],
        random_state=split_config["seed"],
    )
    relative_test_size = test_size / temp_size
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=relative_test_size,
        stratify=temp_frame[stratify_column],
        random_state=split_config["seed"],
    )

    split_frames = {
        "train": train_frame.sort_values("row_id").reset_index(drop=True),
        "val": val_frame.sort_values("row_id").reset_index(drop=True),
        "test": test_frame.sort_values("row_id").reset_index(drop=True),
    }
    for split_name, frame in split_frames.items():
        frame["split"] = split_name
        frame.to_csv(output_paths[split_name], index=False)

    with output_paths["label_map"].open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2)

    manifest = {
        "iteration": config["experiment"]["iteration"],
        "experiment": config["experiment"]["name"],
        "task": get_task_type(config),
        "seed": split_config["seed"],
        "raw_data": str(_resolve_path(config["paths"]["raw_data"])),
        "processed_dir": str(output_paths["train"].parent),
        "target_column": target_column,
        "label_map_path": str(output_paths["label_map"]),
        "label_map": label_map,
        "splits": {},
    }
    for split_name, frame in split_frames.items():
        manifest["splits"][split_name] = {
            "path": str(output_paths[split_name]),
            "rows": int(len(frame)),
            "original_label_distribution": {
                key: int(value)
                for key, value in frame[label_column].value_counts().sort_index().items()
            },
            "target_distribution": {
                str(key): int(value)
                for key, value in frame[target_column].value_counts().sort_index().items()
            },
        }

    with output_paths["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return output_paths


def load_dataset_splits(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load train, validation, and test split files for the configured task."""
    output_paths = create_dataset_splits(config, force=False)
    return {
        split_name: pd.read_csv(path)
        for split_name, path in output_paths.items()
        if split_name in {"train", "val", "test"}
    }


def create_binary_splits(config: dict[str, Any], force: bool = False) -> dict[str, Path]:
    """Create canonical binary split files for Iteration 2: Experiment 1."""
    return create_dataset_splits(config, force=force)


def load_binary_splits(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load train, validation, and test binary split files."""
    return load_dataset_splits(config)


def create_multiclass_splits(config: dict[str, Any], force: bool = False) -> dict[str, Path]:
    """Create canonical multiclass split files for Iteration 2: Experiment 2."""
    return create_dataset_splits(config, force=force)


def load_multiclass_splits(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load train, validation, and test multiclass split files."""
    return load_dataset_splits(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create canonical Iteration 2 split files for the provided config."
    )
    parser.add_argument(
        "--config",
        default="configs/exp1_binary.yaml",
        help="Path to the Iteration 2 experiment config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the processed split files even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for Iteration 2 dataset creation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    output_paths = create_dataset_splits(config, force=args.force)
    logging.info("Saved %s splits to %s", get_task_type(config), output_paths["train"].parent)


if __name__ == "__main__":
    main()
