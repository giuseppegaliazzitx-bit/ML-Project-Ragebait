from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import requests

from ragebait_detector.config import Settings
from ragebait_detector.utils.io import dump_json, read_csv, write_csv

LABEL_COLUMNS = [
    "post_id",
    "author_id",
    "created_at",
    "language",
    "text",
    "source",
    "is_ragebait",
    "label",
    "confidence",
    "reason",
    "llm_model",
    "tool_name",
    "used_tool_call",
    "labeling_status",
    "error",
    "latency_ms",
]
SYSTEM_PROMPT = """
You are labeling tweets for rage-bait detection.
Rage-bait means intentionally provocative wording designed to trigger anger, dogpiles, or hostile engagement.
Not rage-bait includes neutral discussion and sincere outrage about a real event.
You must call the classify_ragebait tool exactly once for every tweet.
If uncertain, prefer false and explain your uncertainty briefly.
""".strip()
TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "classify_ragebait",
        "description": "Return whether the tweet is rage-bait.",
        "parameters": {
            "type": "object",
            "properties": {
                "is_ragebait": {
                    "type": "boolean",
                    "description": "True only if the post appears intentionally crafted to provoke anger or hostile engagement.",
                },
                "confidence": {
                    "type": "number",
                    "description": "A confidence score between 0 and 1.",
                },
                "reason": {
                    "type": "string",
                    "description": "A short explanation for the decision.",
                },
            },
            "required": ["is_ragebait"],
        },
    },
}


@dataclass
class OllamaLabelResult:
    is_ragebait: bool | None
    confidence: float | None
    reason: str
    tool_name: str | None
    used_tool_call: bool
    labeling_status: str
    error: str = ""
    latency_ms: int = 0

    @property
    def numeric_label(self) -> str:
        if self.is_ragebait is None:
            return ""
        return "1" if self.is_ragebait else "0"


def build_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Classify this tweet.\n"
                "Return the decision by calling the tool.\n\n"
                f"Tweet:\n{text}"
            ),
        },
    ]


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def parse_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def extract_tool_result(response_json: dict[str, Any]) -> OllamaLabelResult:
    message = response_json.get("message", {})
    tool_calls = message.get("tool_calls") or response_json.get("tool_calls") or []

    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        tool_name = function.get("name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        decision = parse_bool(arguments.get("is_ragebait"))
        if decision is None:
            continue
        return OllamaLabelResult(
            is_ragebait=decision,
            confidence=parse_confidence(arguments.get("confidence")),
            reason=str(arguments.get("reason", "")).strip(),
            tool_name=tool_name,
            used_tool_call=True,
            labeling_status="ok",
        )

    content = str(message.get("content", "")).strip()
    if content:
        fallback = extract_fallback_result(content)
        if fallback is not None:
            return fallback

    return OllamaLabelResult(
        is_ragebait=None,
        confidence=None,
        reason="",
        tool_name=None,
        used_tool_call=False,
        labeling_status="error",
        error="No usable tool call or fallback content found in Ollama response.",
    )


def extract_fallback_result(content: str) -> OllamaLabelResult | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        decision = parse_bool(payload.get("is_ragebait"))
        if decision is not None:
            return OllamaLabelResult(
                is_ragebait=decision,
                confidence=parse_confidence(payload.get("confidence")),
                reason=str(payload.get("reason", "")).strip(),
                tool_name=None,
                used_tool_call=False,
                labeling_status="ok_fallback",
            )

    lowered = content.lower()
    if "true" in lowered and "false" not in lowered:
        return OllamaLabelResult(
            is_ragebait=True,
            confidence=None,
            reason=content[:240],
            tool_name=None,
            used_tool_call=False,
            labeling_status="ok_fallback",
        )
    if "false" in lowered and "true" not in lowered:
        return OllamaLabelResult(
            is_ragebait=False,
            confidence=None,
            reason=content[:240],
            tool_name=None,
            used_tool_call=False,
            labeling_status="ok_fallback",
        )
    return None


def classify_text_with_ollama(
    text: str,
    *,
    host: str,
    model: str,
    timeout_seconds: int,
    temperature: float,
    max_retries: int,
) -> OllamaLabelResult:
    if not text.strip():
        return OllamaLabelResult(
            is_ragebait=None,
            confidence=None,
            reason="",
            tool_name=None,
            used_tool_call=False,
            labeling_status="skipped",
            error="empty_text",
            latency_ms=0,
        )

    payload = {
        "model": model,
        "stream": False,
        "messages": build_messages(text),
        "tools": [TOOL_SPEC],
        "options": {
            "temperature": temperature,
        },
    }
    endpoint = host.rstrip("/") + "/api/chat"

    for attempt in range(max_retries + 1):
        started_at = perf_counter()
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            result = extract_tool_result(response.json())
            result.latency_ms = int((perf_counter() - started_at) * 1000)
            if result.labeling_status.startswith("ok"):
                return result
            if attempt == max_retries:
                return result
        except requests.RequestException as exc:
            if attempt == max_retries:
                return OllamaLabelResult(
                    is_ragebait=None,
                    confidence=None,
                    reason="",
                    tool_name=None,
                    used_tool_call=False,
                    labeling_status="error",
                    error=str(exc),
                    latency_ms=int((perf_counter() - started_at) * 1000),
                )

    return OllamaLabelResult(
        is_ragebait=None,
        confidence=None,
        reason="",
        tool_name=None,
        used_tool_call=False,
        labeling_status="error",
        error="Labeling attempt exhausted retries without a usable response.",
        latency_ms=0,
    )


def merge_row_with_label(
    row: dict[str, str],
    result: OllamaLabelResult,
    model_name: str,
) -> dict[str, str]:
    return {
        "post_id": row.get("post_id", ""),
        "author_id": row.get("author_id", ""),
        "created_at": row.get("created_at", ""),
        "language": row.get("language", ""),
        "text": row.get("text", ""),
        "source": row.get("source", ""),
        "is_ragebait": "" if result.is_ragebait is None else str(result.is_ragebait).lower(),
        "label": result.numeric_label,
        "confidence": "" if result.confidence is None else f"{result.confidence:.4f}",
        "reason": result.reason,
        "llm_model": model_name,
        "tool_name": result.tool_name or "",
        "used_tool_call": str(result.used_tool_call).lower(),
        "labeling_status": result.labeling_status,
        "error": result.error,
        "latency_ms": str(result.latency_ms),
    }


def label_csv_with_ollama(
    *,
    input_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    settings: Settings,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = read_csv(input_path)
    if limit is not None:
        rows = rows[:limit]

    unique_texts = {
        row.get("text", "").strip()
        for row in rows
        if row.get("text", "").strip()
    }
    cached_results: dict[str, OllamaLabelResult] = {}

    with ThreadPoolExecutor(max_workers=settings.ollama.max_workers) as executor:
        futures = {
            executor.submit(
                classify_text_with_ollama,
                text,
                host=settings.ollama.host,
                model=settings.ollama.model,
                timeout_seconds=settings.ollama.request_timeout_seconds,
                temperature=settings.ollama.temperature,
                max_retries=settings.ollama.max_retries,
            ): text
            for text in unique_texts
        }
        for future in as_completed(futures):
            text = futures[future]
            cached_results[text] = future.result()

    labeled_rows: list[dict[str, str]] = []
    success_count = 0
    error_count = 0

    for row in rows:
        text = row.get("text", "").strip()
        result = cached_results.get(text)
        if result is None:
            result = OllamaLabelResult(
                is_ragebait=None,
                confidence=None,
                reason="",
                tool_name=None,
                used_tool_call=False,
                labeling_status="skipped",
                error="empty_text",
            )
        if result.labeling_status.startswith("ok"):
            success_count += 1
        elif result.labeling_status == "error":
            error_count += 1

        labeled_rows.append(
            merge_row_with_label(
                row=row,
                result=result,
                model_name=settings.ollama.model,
            )
        )

    write_csv(output_path, labeled_rows, LABEL_COLUMNS)
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "rows_requested": len(rows),
        "rows_labeled_successfully": success_count,
        "rows_with_errors": error_count,
        "model": settings.ollama.model,
        "host": settings.ollama.host,
        "max_workers": settings.ollama.max_workers,
    }
    dump_json(summary_path, summary)
    return summary
