from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://192.168.191.205:5173",
)


@dataclass(frozen=True)
class ManualEvalSettings:
    repo_root: Path = REPO_ROOT
    unified_csv: Path = REPO_ROOT / "data/unlabeled/unified_unlabeled_posts.csv"
    ai_generated_csv: Path = REPO_ROOT / "data/unlabeled/ai_generated_set.csv"
    combined_csv: Path = REPO_ROOT / "data/unlabeled/manual_eval_pool.csv"
    manual_eval_csv: Path = REPO_ROOT / "data/labeled/manual_eval.csv"
    database_path: Path = REPO_ROOT / "manual_eval_app/backend/state/manual_eval.sqlite3"
    seed: int = 7
    ai_source_name: str = "gemma4 ai generated"
    ai_language: str = "en"
    allowed_origins: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_ALLOWED_ORIGINS
    )


def default_settings() -> ManualEvalSettings:
    return ManualEvalSettings()
