from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SourceCount(BaseModel):
    source: str
    labeled: int


class TweetPayload(BaseModel):
    post_id: str
    position: int
    display_index: int
    total: int
    source: str
    author_id: str
    created_at: str
    language: str
    text: str


class SessionStats(BaseModel):
    total_posts: int
    completed_actions: int
    labeled_count: int
    skipped_count: int
    remaining_count: int
    next_display_index: int
    can_undo: bool
    labeled_by_source: list[SourceCount]


class SessionResponse(BaseModel):
    tweet: TweetPayload | None
    stats: SessionStats


class SubmitActionRequest(BaseModel):
    action: Literal["label", "skip"]
    label: int | None = None
