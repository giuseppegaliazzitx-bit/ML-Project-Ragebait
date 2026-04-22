from __future__ import annotations

import csv
from pathlib import Path

from fastapi.testclient import TestClient

from manual_eval_app.backend.app.config import ManualEvalSettings
from manual_eval_app.backend.app.main import create_app


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_manual_eval_flow_builds_pool_and_supports_undo(tmp_path: Path) -> None:
    unified_csv = tmp_path / "data/unlabeled/unified_unlabeled_posts.csv"
    ai_csv = tmp_path / "data/unlabeled/ai_generated_set.csv"
    combined_csv = tmp_path / "data/unlabeled/manual_eval_pool.csv"
    manual_eval_csv = tmp_path / "data/labeled/manual_eval.csv"
    database_path = tmp_path / "manual_eval_app/backend/state/manual_eval.sqlite3"

    _write_csv(
        unified_csv,
        ["post_id", "author_id", "created_at", "language", "text", "source"],
        [
            {
                "post_id": "u1",
                "author_id": "alpha",
                "created_at": "",
                "language": "en",
                "text": "First real tweet",
                "source": "real-source-a",
            },
            {
                "post_id": "u2",
                "author_id": "beta",
                "created_at": "",
                "language": "en",
                "text": "Second real tweet",
                "source": "real-source-b",
            },
            {
                "post_id": "u3",
                "author_id": "alpha",
                "created_at": "",
                "language": "en",
                "text": "Third real tweet",
                "source": "real-source-a",
            },
        ],
    )
    _write_csv(
        ai_csv,
        ["text", "is_rage_bait", "niche"],
        [
            {
                "text": "Synthetic one",
                "is_rage_bait": "1",
                "niche": "Persona A",
            },
            {
                "text": "Synthetic two",
                "is_rage_bait": "0",
                "niche": "Persona B",
            },
        ],
    )

    settings = ManualEvalSettings(
        repo_root=tmp_path,
        unified_csv=unified_csv,
        ai_generated_csv=ai_csv,
        combined_csv=combined_csv,
        manual_eval_csv=manual_eval_csv,
        database_path=database_path,
        seed=13,
        allowed_origins=(),
    )

    app = create_app(settings)
    with TestClient(app) as client:
        session = client.get("/api/session")
        assert session.status_code == 200
        payload = session.json()
        assert payload["stats"]["total_posts"] == 5
        assert payload["stats"]["labeled_count"] == 0
        assert payload["tweet"]["display_index"] == 1

        first_post_id = payload["tweet"]["post_id"]
        label_response = client.post(
            "/api/respond",
            json={"action": "label", "label": 1},
        )
        assert label_response.status_code == 200
        labeled_payload = label_response.json()
        assert labeled_payload["stats"]["labeled_count"] == 1
        assert labeled_payload["stats"]["completed_actions"] == 1
        assert labeled_payload["tweet"]["post_id"] != first_post_id

        skipped_post = labeled_payload["tweet"]["post_id"]
        skip_response = client.post("/api/respond", json={"action": "skip"})
        assert skip_response.status_code == 200
        skipped_payload = skip_response.json()
        assert skipped_payload["stats"]["skipped_count"] == 1

        undo_response = client.post("/api/undo")
        assert undo_response.status_code == 200
        undo_payload = undo_response.json()
        assert undo_payload["stats"]["skipped_count"] == 0
        assert undo_payload["tweet"]["post_id"] == skipped_post

    assert combined_csv.exists()
    combined_rows = list(csv.DictReader(combined_csv.open("r", encoding="utf-8")))
    assert any(row["source"] == "gemma4 ai generated" for row in combined_rows)
    assert any(row["author_id"] == "Persona A" for row in combined_rows)

    exported_rows = list(csv.DictReader(manual_eval_csv.open("r", encoding="utf-8")))
    assert len(exported_rows) == 1
    assert exported_rows[0]["post_id"] == first_post_id
    assert exported_rows[0]["label"] == "1"
