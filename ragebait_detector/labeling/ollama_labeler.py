from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any

import requests
from requests.adapters import HTTPAdapter

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

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
You are labeling short social media posts for rage-bait detection.
Rage-bait means intentionally provocative wording designed to trigger anger, dogpiles, or hostile engagement.
Not rage-bait includes neutral discussion and sincere outrage about a real event.
Return JSON only.
For each item, include: id, is_ragebait, confidence, reason.
Confidence must be between 0 and 1.
If uncertain, prefer false and explain your uncertainty briefly.
""".strip()


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


@dataclass(frozen=True)
class OllamaBatchItem:
    item_id: str
    text: str


def build_messages(batch: list[OllamaBatchItem]) -> list[dict[str, str]]:
    prompt_rows = [
        {
            "id": item.item_id,
            "text": item.text,
        }
        for item in batch
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Classify the {len(batch)} tweets in the input JSON array.\n"
                'Return only a JSON array of objects with keys "id", "is_ragebait", "confidence", and "reason".\n'
                "Preserve every id exactly once and do not omit any item.\n\n"
                f"Input tweets:\n{json.dumps(prompt_rows, ensure_ascii=False)}"
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


def build_error_result(error: str, *, latency_ms: int = 0) -> OllamaLabelResult:
    return OllamaLabelResult(
        is_ragebait=None,
        confidence=None,
        reason="",
        tool_name=None,
        used_tool_call=False,
        labeling_status="error",
        error=error,
        latency_ms=latency_ms,
    )


def build_skipped_result(error: str) -> OllamaLabelResult:
    return OllamaLabelResult(
        is_ragebait=None,
        confidence=None,
        reason="",
        tool_name=None,
        used_tool_call=False,
        labeling_status="skipped",
        error=error,
        latency_ms=0,
    )


def load_json_content(content: Any) -> Any | None:
    if isinstance(content, (dict, list)):
        return content
    if not isinstance(content, str):
        return None

    stripped = content.strip()
    if not stripped:
        return None

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            stripped = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


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

    fallback_payload = load_json_content(message.get("content", ""))
    if fallback_payload is not None:
        fallback = extract_fallback_result(fallback_payload)
        if fallback is not None:
            return fallback

    content = str(message.get("content", "")).strip()
    if content:
        fallback = extract_fallback_result(content)
        if fallback is not None:
            return fallback

    return build_error_result("No usable tool call or fallback content found in Ollama response.")


def extract_fallback_result(content: Any) -> OllamaLabelResult | None:
    payload = content
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

    if not isinstance(content, str):
        return None

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


def extract_batch_payload(response_json: dict[str, Any]) -> list[dict[str, Any]] | None:
    message = response_json.get("message", {})
    payload = load_json_content(message.get("content", ""))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if all(key in payload for key in ("id", "is_ragebait")):
            return [payload]
        for key in ("results", "items", "predictions", "labels"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        nested_values = [value for value in payload.values() if isinstance(value, dict)]
        if nested_values and all("is_ragebait" in value for value in nested_values):
            return nested_values
    return None


def extract_batch_results(
    response_json: dict[str, Any],
    batch: list[OllamaBatchItem],
    *,
    latency_ms: int,
) -> dict[str, OllamaLabelResult]:
    if len(batch) == 1:
        single_payload = load_json_content(response_json.get("message", {}).get("content", ""))
        if isinstance(single_payload, dict) and "is_ragebait" in single_payload:
            parsed_single = extract_fallback_result(single_payload)
            if parsed_single is not None:
                parsed_single.labeling_status = "ok_json"
                parsed_single.latency_ms = latency_ms
                return {batch[0].text: parsed_single}

    expected_items = {item.item_id: item for item in batch}
    parsed_items = extract_batch_payload(response_json)
    if parsed_items is None:
        return {
            item.text: build_error_result(
                "No usable JSON batch payload found in Ollama response.",
                latency_ms=latency_ms,
            )
            for item in batch
        }

    results = {
        item.text: build_error_result(
            "Batch response did not contain a result for this tweet.",
            latency_ms=latency_ms,
        )
        for item in batch
    }
    seen_ids: set[str] = set()

    for parsed_item in parsed_items:
        item_id = str(parsed_item.get("id", "")).strip()
        if not item_id or item_id in seen_ids or item_id not in expected_items:
            continue
        seen_ids.add(item_id)

        decision = parse_bool(parsed_item.get("is_ragebait"))
        if decision is None:
            results[expected_items[item_id].text] = build_error_result(
                "Batch response contained an invalid is_ragebait value.",
                latency_ms=latency_ms,
            )
            continue

        results[expected_items[item_id].text] = OllamaLabelResult(
            is_ragebait=decision,
            confidence=parse_confidence(parsed_item.get("confidence")),
            reason=str(parsed_item.get("reason", "")).strip(),
            tool_name=None,
            used_tool_call=False,
            labeling_status="ok_json",
            latency_ms=latency_ms,
        )

    return results


def has_complete_batch_results(
    results: dict[str, OllamaLabelResult],
    batch: list[OllamaBatchItem],
) -> bool:
    return all(
        results.get(item.text, build_error_result("missing")).labeling_status.startswith("ok")
        for item in batch
    )


def recover_incomplete_batch_results(
    batch: list[OllamaBatchItem],
    current_results: dict[str, OllamaLabelResult],
    *,
    host: str,
    model: str,
    timeout_seconds: int,
    temperature: float,
    max_retries: int,
    session: requests.Session,
) -> dict[str, OllamaLabelResult]:
    unresolved_items = [
        item
        for item in batch
        if not current_results.get(item.text, build_error_result("missing")).labeling_status.startswith("ok")
    ]
    if not unresolved_items or len(unresolved_items) == len(batch) == 1:
        return current_results

    midpoint = max(1, len(unresolved_items) // 2)
    recovered_results = dict(current_results)

    for sub_batch in (unresolved_items[:midpoint], unresolved_items[midpoint:]):
        if not sub_batch:
            continue
        split_results = classify_batch_with_ollama(
            sub_batch,
            host=host,
            model=model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_retries=max_retries,
            session=session,
        )
        recovered_results.update(split_results)

    return recovered_results


def classify_batch_with_ollama(
    batch: list[OllamaBatchItem],
    *,
    host: str,
    model: str,
    timeout_seconds: int,
    temperature: float,
    max_retries: int,
    session: requests.Session,
) -> dict[str, OllamaLabelResult]:
    results: dict[str, OllamaLabelResult] = {}
    populated_batch: list[OllamaBatchItem] = []

    for item in batch:
        if item.text.strip():
            populated_batch.append(item)
        else:
            results[item.text] = build_skipped_result("empty_text")

    if not populated_batch:
        return results

    payload = {
        "model": model,
        "stream": False,
        "messages": build_messages(populated_batch),
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }
    endpoint = host.rstrip("/") + "/api/chat"

    for attempt in range(max_retries + 1):
        started_at = perf_counter()
        try:
            response = session.post(endpoint, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            latency_ms = int((perf_counter() - started_at) * 1000)
            batch_results = extract_batch_results(
                response.json(),
                populated_batch,
                latency_ms=latency_ms,
            )
            if has_complete_batch_results(batch_results, populated_batch):
                results.update(batch_results)
                return results
            if attempt == max_retries:
                if len(populated_batch) > 1:
                    batch_results = recover_incomplete_batch_results(
                        populated_batch,
                        batch_results,
                        host=host,
                        model=model,
                        timeout_seconds=timeout_seconds,
                        temperature=temperature,
                        max_retries=max_retries,
                        session=session,
                    )
                results.update(batch_results)
                return results
        except requests.RequestException as exc:
            if attempt == max_retries:
                latency_ms = int((perf_counter() - started_at) * 1000)
                error_result = build_error_result(str(exc), latency_ms=latency_ms)
                for item in populated_batch:
                    results[item.text] = error_result
                return results

    exhausted_result = build_error_result(
        "Labeling attempt exhausted retries without a usable response.",
        latency_ms=0,
    )
    for item in populated_batch:
        results[item.text] = exhausted_result
    return results


def classify_text_with_ollama(
    text: str,
    *,
    host: str,
    model: str,
    timeout_seconds: int,
    temperature: float,
    max_retries: int,
    session: requests.Session,
) -> OllamaLabelResult:
    results = classify_batch_with_ollama(
        [OllamaBatchItem(item_id="1", text=text)],
        host=host,
        model=model,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_retries=max_retries,
        session=session,
    )
    return results.get(text, build_error_result("Single-item batch did not return a result."))


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
    random_seed: int = 42,
    select_randomly: bool = True,
) -> dict[str, Any]:
    rows = read_csv(input_path)
    if limit is not None:
        if select_randomly:
            shuffled_rows = rows[:]
            random.Random(random_seed).shuffle(shuffled_rows)
            rows = shuffled_rows[:limit]
        else:
            rows = rows[:limit]

    unique_texts: list[str] = []
    seen_texts: set[str] = set()
    for row in rows:
        text = row.get("text", "").strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        unique_texts.append(text)

    batch_size = max(1, int(settings.ollama.batch_size))
    batched_texts = [
        [
            OllamaBatchItem(item_id=str(index + 1), text=text)
            for index, text in enumerate(unique_texts[start : start + batch_size])
        ]
        for start in range(0, len(unique_texts), batch_size)
    ]
    cached_results: dict[str, OllamaLabelResult] = {}
    max_workers = max(1, int(settings.ollama.max_workers))

    print(
        f"Preparing to label {len(unique_texts)} unique tweets in "
        f"{len(batched_texts)} batch(es) using {settings.ollama.model}..."
    )

    with requests.Session() as http_session:
        adapter = HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
        http_session.mount("http://", adapter)
        http_session.mount("https://", adapter)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    classify_batch_with_ollama,
                    batch,
                    host=settings.ollama.host,
                    model=settings.ollama.model,
                    timeout_seconds=settings.ollama.request_timeout_seconds,
                    temperature=settings.ollama.temperature,
                    max_retries=settings.ollama.max_retries,
                    session=http_session,
                ): batch
                for batch in batched_texts
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Calling Ollama API",
                unit="batch",
            ):
                cached_results.update(future.result())

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
        "unique_texts_requested": len(unique_texts),
        "batches_submitted": len(batched_texts),
        "batch_size": batch_size,
        "rows_labeled_successfully": success_count,
        "rows_with_errors": error_count,
        "model": settings.ollama.model,
        "host": settings.ollama.host,
        "max_workers": max_workers,
        "random_seed": random_seed,
        "select_randomly": select_randomly,
    }
    dump_json(summary_path, summary)
    return summary
