from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
from random import Random
from typing import Iterator

from .config import ManualEvalSettings

POOL_FIELDNAMES = ["post_id", "author_id", "created_at", "language", "text", "source"]
MANUAL_EVAL_FIELDNAMES = [
    "queue_position",
    "post_id",
    "author_id",
    "created_at",
    "language",
    "source",
    "text",
    "label",
    "labeled_at",
]


class ManualEvalService:
    def __init__(self, settings: ManualEvalSettings) -> None:
        self.settings = settings
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self.settings.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.combined_csv.parent.mkdir(parents=True, exist_ok=True)
        self.settings.manual_eval_csv.parent.mkdir(parents=True, exist_ok=True)
        needs_bootstrap = not self.settings.database_path.exists()
        if not needs_bootstrap:
            with self._connect() as conn:
                needs_bootstrap = not self._has_schema(conn)
        if needs_bootstrap:
            self._bootstrap_database()
        elif not self.settings.combined_csv.exists():
            self._export_combined_pool_csv()
        self._export_manual_eval_csv()
        self._initialized = True

    def get_session(self) -> dict[str, object]:
        with self._connect() as conn:
            return self._build_session_payload(conn)

    def submit_action(self, action: str, label: int | None = None) -> dict[str, object]:
        if action not in {"label", "skip"}:
            raise ValueError("Action must be 'label' or 'skip'.")
        if action == "label" and label not in {0, 1}:
            raise ValueError("Labels must be 0 for not ragebait or 1 for ragebait.")
        if action == "skip" and label is not None:
            raise ValueError("Skip actions cannot include a label.")

        with self._connect() as conn:
            current = self._fetch_current_tweet(conn)
            if current is None:
                raise ValueError("No tweets remain in the queue.")
            conn.execute(
                """
                INSERT INTO actions (position, post_id, action_type, label, acted_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    current["position"],
                    current["post_id"],
                    action,
                    label,
                    self._timestamp(),
                ),
            )
            payload = self._build_session_payload(conn)
        self._export_manual_eval_csv()
        return payload

    def undo(self) -> dict[str, object]:
        with self._connect() as conn:
            latest = conn.execute(
                "SELECT sequence FROM actions ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            if latest is not None:
                conn.execute("DELETE FROM actions WHERE sequence = ?", (latest["sequence"],))
            payload = self._build_session_payload(conn)
        self._export_manual_eval_csv()
        return payload

    def _bootstrap_database(self) -> None:
        self.settings.database_path.unlink(missing_ok=True)
        buckets: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

        with self._connect() as conn, self.settings.combined_csv.open(
            "w", encoding="utf-8", newline=""
        ) as combined_handle:
            self._create_schema(conn)
            writer = csv.DictWriter(combined_handle, fieldnames=POOL_FIELDNAMES)
            writer.writeheader()

            post_rows: list[tuple[str, str, str, str, str, str]] = []
            for row in self._iter_pool_rows():
                writer.writerow(row)
                post_rows.append(
                    (
                        row["post_id"],
                        row["author_id"],
                        row["created_at"],
                        row["language"],
                        row["text"],
                        row["source"],
                    )
                )
                buckets[row["source"]][row["author_id"]].append(row["post_id"])
                if len(post_rows) >= 5000:
                    conn.executemany(
                        """
                        INSERT INTO posts
                        (post_id, author_id, created_at, language, text, source)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        post_rows,
                    )
                    post_rows.clear()

            if post_rows:
                conn.executemany(
                    """
                    INSERT INTO posts
                    (post_id, author_id, created_at, language, text, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    post_rows,
                )

            order_rows: list[tuple[int, str]] = []
            for index, post_id in enumerate(self._build_balanced_order(buckets)):
                order_rows.append((index, post_id))
                if len(order_rows) >= 5000:
                    conn.executemany(
                        "INSERT INTO served_order (position, post_id) VALUES (?, ?)",
                        order_rows,
                    )
                    order_rows.clear()
            if order_rows:
                conn.executemany(
                    "INSERT INTO served_order (position, post_id) VALUES (?, ?)",
                    order_rows,
                )

    def _iter_pool_rows(self) -> Iterator[dict[str, str]]:
        yield from self._iter_unified_rows()
        yield from self._iter_ai_rows()

    def _iter_unified_rows(self) -> Iterator[dict[str, str]]:
        with self.settings.unified_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for fallback_index, row in enumerate(reader, start=1):
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                post_id = (row.get("post_id") or "").strip() or f"unified-{fallback_index}"
                author_id = (row.get("author_id") or "").strip() or "[unknown author]"
                created_at = (row.get("created_at") or "").strip()
                language = (row.get("language") or "").strip()
                source = (row.get("source") or "").strip() or "[unknown source]"
                yield {
                    "post_id": post_id,
                    "author_id": author_id,
                    "created_at": created_at,
                    "language": language,
                    "text": text,
                    "source": source,
                }

    def _iter_ai_rows(self) -> Iterator[dict[str, str]]:
        with self.settings.ai_generated_csv.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                niche = (row.get("niche") or "").strip() or "[unknown ai niche]"
                yield {
                    "post_id": f"gemma4-ai-{index:06d}",
                    "author_id": niche,
                    "created_at": "",
                    "language": self.settings.ai_language,
                    "text": text,
                    "source": self.settings.ai_source_name,
                }

    def _build_balanced_order(
        self, buckets: dict[str, dict[str, list[str]]]
    ) -> list[str]:
        rng = Random(self.settings.seed)
        shuffled_sources = list(buckets)
        rng.shuffle(shuffled_sources)

        queued: dict[str, dict[str, deque[str]]] = {}
        author_orders: dict[str, list[str]] = {}
        author_offsets: dict[str, int] = {}

        for source in shuffled_sources:
            authors = list(buckets[source])
            rng.shuffle(authors)
            author_orders[source] = authors
            author_offsets[source] = 0
            queued[source] = {}
            for author in authors:
                post_ids = list(buckets[source][author])
                rng.shuffle(post_ids)
                queued[source][author] = deque(post_ids)

        order: list[str] = []
        active_sources = deque(
            source for source in shuffled_sources if author_orders.get(source)
        )
        while active_sources:
            source = active_sources.popleft()
            authors = author_orders[source]
            offset = author_offsets[source]
            picked = False
            for step in range(len(authors)):
                author_index = (offset + step) % len(authors)
                author = authors[author_index]
                bucket = queued[source][author]
                if bucket:
                    order.append(bucket.popleft())
                    author_offsets[source] = (author_index + 1) % len(authors)
                    picked = True
                    break
            if picked and any(queued[source][author] for author in authors):
                active_sources.append(source)
        return order

    def _build_session_payload(self, conn: sqlite3.Connection) -> dict[str, object]:
        total_posts = conn.execute("SELECT COUNT(*) AS value FROM served_order").fetchone()[
            "value"
        ]
        completed_actions = conn.execute(
            "SELECT COUNT(*) AS value FROM actions"
        ).fetchone()["value"]
        labeled_count = conn.execute(
            "SELECT COUNT(*) AS value FROM actions WHERE action_type = 'label'"
        ).fetchone()["value"]
        skipped_count = completed_actions - labeled_count
        current = self._fetch_current_tweet(conn)
        labeled_by_source = [
            {"source": row["source"], "labeled": row["labeled"]}
            for row in conn.execute(
                """
                SELECT posts.source AS source, COUNT(*) AS labeled
                FROM actions
                JOIN posts ON posts.post_id = actions.post_id
                WHERE actions.action_type = 'label'
                GROUP BY posts.source
                ORDER BY labeled DESC, posts.source ASC
                """
            ).fetchall()
        ]

        if current is None:
            tweet = None
            next_display_index = total_posts if total_posts else 0
        else:
            tweet = {
                "post_id": current["post_id"],
                "position": current["position"],
                "display_index": current["position"] + 1,
                "total": total_posts,
                "source": current["source"],
                "author_id": current["author_id"],
                "created_at": current["created_at"],
                "language": current["language"],
                "text": current["text"],
            }
            next_display_index = current["position"] + 1

        return {
            "tweet": tweet,
            "stats": {
                "total_posts": total_posts,
                "completed_actions": completed_actions,
                "labeled_count": labeled_count,
                "skipped_count": skipped_count,
                "remaining_count": max(total_posts - completed_actions, 0),
                "next_display_index": next_display_index,
                "can_undo": completed_actions > 0,
                "labeled_by_source": labeled_by_source,
            },
        }

    def _fetch_current_tweet(self, conn: sqlite3.Connection) -> sqlite3.Row | None:
        current_position = conn.execute(
            "SELECT COUNT(*) AS value FROM actions"
        ).fetchone()["value"]
        return conn.execute(
            """
            SELECT served_order.position, posts.post_id, posts.author_id, posts.created_at,
                   posts.language, posts.text, posts.source
            FROM served_order
            JOIN posts ON posts.post_id = served_order.post_id
            WHERE served_order.position = ?
            """,
            (current_position,),
        ).fetchone()

    def _export_manual_eval_csv(self) -> None:
        self.settings.manual_eval_csv.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn, self.settings.manual_eval_csv.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=MANUAL_EVAL_FIELDNAMES)
            writer.writeheader()
            for row in conn.execute(
                """
                SELECT actions.position AS queue_position, posts.post_id, posts.author_id,
                       posts.created_at, posts.language, posts.source, posts.text,
                       actions.label, actions.acted_at AS labeled_at
                FROM actions
                JOIN posts ON posts.post_id = actions.post_id
                WHERE actions.action_type = 'label'
                ORDER BY actions.position ASC
                """
            ).fetchall():
                writer.writerow(
                    {
                        "queue_position": row["queue_position"],
                        "post_id": row["post_id"],
                        "author_id": row["author_id"],
                        "created_at": row["created_at"],
                        "language": row["language"],
                        "source": row["source"],
                        "text": row["text"],
                        "label": row["label"],
                        "labeled_at": row["labeled_at"],
                    }
                )

    def _export_combined_pool_csv(self) -> None:
        self.settings.combined_csv.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn, self.settings.combined_csv.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=POOL_FIELDNAMES)
            writer.writeheader()
            for row in conn.execute(
                """
                SELECT posts.post_id, posts.author_id, posts.created_at,
                       posts.language, posts.text, posts.source
                FROM served_order
                JOIN posts ON posts.post_id = served_order.post_id
                ORDER BY served_order.position ASC
                """
            ).fetchall():
                writer.writerow({field: row[field] for field in POOL_FIELDNAMES})

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS posts (
                post_id TEXT PRIMARY KEY,
                author_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                language TEXT NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS served_order (
                position INTEGER PRIMARY KEY,
                post_id TEXT NOT NULL UNIQUE REFERENCES posts(post_id)
            );

            CREATE TABLE IF NOT EXISTS actions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                position INTEGER NOT NULL UNIQUE,
                post_id TEXT NOT NULL REFERENCES posts(post_id),
                action_type TEXT NOT NULL CHECK (action_type IN ('label', 'skip')),
                label INTEGER CHECK (label IN (0, 1) OR label IS NULL),
                acted_at TEXT NOT NULL
            );
            """
        )

    def _has_schema(self, conn: sqlite3.Connection) -> bool:
        required = {"posts", "served_order", "actions"}
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        existing = {row["name"] for row in rows}
        return required.issubset(existing)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.settings.database_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
