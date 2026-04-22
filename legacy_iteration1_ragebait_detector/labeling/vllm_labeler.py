from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import shutil
import sysconfig
from typing import Any

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

try:
    from vllm import LLM, SamplingParams
    VLLM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    LLM = None
    SamplingParams = None
    VLLM_IMPORT_ERROR = exc

# Try to import GuidedDecodingParams separately for older vLLM versions
try:
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    GuidedDecodingParams = None

from ragebait_detector.config import Settings
from ragebait_detector.utils.dependencies import MissingDependencyError
from ragebait_detector.utils.io import dump_json, read_csv, write_csv

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_ragebait": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["is_ragebait", "confidence", "reason"],
}

SYSTEM_PROMPT = (
    "You are labeling short social media posts for rage-bait detection. "
    "Rage-bait means intentionally provocative wording designed to trigger anger, "
    "dogpiles, or hostile engagement. Not rage-bait includes neutral discussion "
    "and sincere outrage about a real event. "
    "You MUST respond ONLY with a valid JSON object containing exactly three keys: "
    "'is_ragebait' (boolean), 'confidence' (float between 0.0 and 1.0), and 'reason' (string)."
)

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
    "labeling_status",
    "parse_mode",
    "error",
    # "raw_response",
]


@dataclass
class VLLMLabelResult:
    is_ragebait: bool | None
    confidence: float | None
    reason: str
    labeling_status: str
    parse_mode: str = ""
    error: str = ""
    raw_response: str = ""

    @property
    def numeric_label(self) -> str:
        if self.is_ragebait is None:
            return ""
        return "1" if self.is_ragebait else "0"


def format_qwen_prompt(text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nClassify this tweet:\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def format_gemma_prompt(text: str) -> str:
    return (
        f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\nClassify this tweet:\n{text}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def parse_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def build_error_result(error: str) -> VLLMLabelResult:
    return VLLMLabelResult(
        is_ragebait=None,
        confidence=None,
        reason="",
        labeling_status="error",
        parse_mode="error",
        error=error,
    )


def build_skipped_result(error: str) -> VLLMLabelResult:
    return VLLMLabelResult(
        is_ragebait=None,
        confidence=None,
        reason="",
        labeling_status="skipped",
        parse_mode="skipped",
        error=error,
    )


def _extract_json_object_candidate(payload_text: str) -> str:
    stripped = payload_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1] == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].lstrip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1:
        if end != -1 and end >= start:
            return stripped[start : end + 1]
        return stripped[start:]
    return stripped


def _extract_reason_value(payload_text: str) -> str | None:
    match = re.search(r'"reason"\s*:\s*"', payload_text)
    if match is None:
        return None

    index = match.end()
    chars: list[str] = []
    escaped = False
    while index < len(payload_text):
        char = payload_text[index]
        if escaped:
            chars.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return "".join(chars).strip()
        else:
            chars.append(char)
        index += 1

    return "".join(chars).strip()


def _salvage_label_payload(payload_text: str) -> dict[str, Any] | None:
    candidate = _extract_json_object_candidate(payload_text)
    decision_match = re.search(r'"is_ragebait"\s*:\s*(true|false)', candidate)
    confidence_match = re.search(
        r'"confidence"\s*:\s*(-?(?:\d+(?:\.\d+)?|\.\d+))',
        candidate,
    )
    reason = _extract_reason_value(candidate)

    if decision_match is None or confidence_match is None or reason is None:
        return None

    return {
        "is_ragebait": decision_match.group(1) == "true",
        "confidence": float(confidence_match.group(1)),
        "reason": reason,
    }


def _normalise_single_quoted_payload(payload_text: str) -> str:
    candidate = _extract_json_object_candidate(payload_text)
    if '"' in candidate or "'" not in candidate:
        return candidate
    candidate = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'\s*:", r'"\1":', candidate)
    candidate = re.sub(r":\s*'([^'\\]*(?:\\.[^'\\]*)*)'", r': "\1"', candidate)
    return candidate


def extract_label_result(payload_text: str) -> VLLMLabelResult:
    candidate = _extract_json_object_candidate(payload_text)
    try:
        payload = json.loads(candidate)
        parse_mode = "strict_json"
    except json.JSONDecodeError as exc:
        normalised_candidate = _normalise_single_quoted_payload(candidate)
        if normalised_candidate != candidate:
            try:
                payload = json.loads(normalised_candidate)
                parse_mode = "single_quote_normalized"
            except json.JSONDecodeError:
                payload = _salvage_label_payload(normalised_candidate)
                parse_mode = "salvaged_regex" if payload is not None else ""
        else:
            payload = _salvage_label_payload(candidate)
            parse_mode = "salvaged_regex" if payload is not None else ""
        if payload is None:
            result = build_error_result(f"Invalid guided JSON response: {exc}")
            result.raw_response = payload_text
            return result

    decision = payload.get("is_ragebait")
    if not isinstance(decision, bool):
        result = build_error_result(
            "Guided JSON response did not contain a boolean is_ragebait value."
        )
        result.raw_response = payload_text
        return result

    confidence = parse_confidence(payload.get("confidence"))
    if confidence is None:
        result = build_error_result(
            "Guided JSON response did not contain a numeric confidence value."
        )
        result.raw_response = payload_text
        return result

    reason = payload.get("reason")
    if not isinstance(reason, str):
        result = build_error_result(
            "Guided JSON response did not contain a string reason value."
        )
        result.raw_response = payload_text
        return result

    return VLLMLabelResult(
        is_ragebait=decision,
        confidence=confidence,
        reason=reason.strip(),
        labeling_status="ok",
        parse_mode=parse_mode,
        raw_response=payload_text,
    )


def merge_row_with_label(
    row: dict[str, str],
    result: VLLMLabelResult,
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
        "labeling_status": result.labeling_status,
        "parse_mode": result.parse_mode,
        "error": result.error,
        # "raw_response": result.raw_response,
    }


def _require_vllm_runtime() -> tuple[Any, Any]:
    if LLM is None or SamplingParams is None:
        if VLLM_IMPORT_ERROR is not None:
            raise MissingDependencyError(
                "Unable to import 'vllm'. This is often caused by an incompatible "
                "'transformers' version. Install project dependencies with "
                "'transformers>=4.51.1,<4.52'."
            ) from VLLM_IMPORT_ERROR
        raise MissingDependencyError(
            "Missing optional dependency 'vllm'. Install project dependencies before labeling."
        )
    if not torch.cuda.is_available():
        raise MissingDependencyError(
            "vLLM labeling requires CUDA, but this Python environment cannot see an "
            "NVIDIA GPU. In WSL2, confirm GPU passthrough is working and that "
            "'nvidia-smi' succeeds inside WSL before running this script."
        )
    if not any(shutil.which(binary) for binary in ("cc", "gcc", "clang")):
        raise MissingDependencyError(
            "vLLM/Triton requires a C compiler, but none was found in PATH. "
            "Install WSL build tools, for example: 'sudo apt update && sudo apt install -y build-essential'."
        )
    include_dir = sysconfig.get_config_var("INCLUDEPY")
    if not include_dir or not Path(include_dir, "Python.h").exists():
        raise MissingDependencyError(
            "vLLM/Triton requires Python development headers, but 'Python.h' was not found. "
            "Install them for this interpreter, for example: 'sudo apt install -y python3.12-dev'."
        )
    return LLM, SamplingParams


def _select_rows_for_labeling(
    rows: list[dict[str, str]],
    limit: int | None,
    enable_random: bool,
    random_seed: int,
    balance_by_source: bool,
) -> list[dict[str, str]]:
    if limit is None:
        return list(rows)
    if limit < 0:
        raise ValueError("limit must be greater than or equal to 0")

    sample_size = min(limit, len(rows))
    if not enable_random:
        return rows[:sample_size]

    if balance_by_source:
        rng = random.Random(random_seed)
        source_to_indices: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            source = (row.get("source") or "").strip() or "[unknown_source]"
            source_to_indices.setdefault(source, []).append(index)

        for indices in source_to_indices.values():
            rng.shuffle(indices)

        selected_indices: list[int] = []
        while len(selected_indices) < sample_size:
            available_sources = [
                source for source, indices in source_to_indices.items() if indices
            ]
            if not available_sources:
                break
            rng.shuffle(available_sources)
            for source in available_sources:
                if len(selected_indices) >= sample_size:
                    break
                selected_indices.append(source_to_indices[source].pop())

        selected_indices.sort()
        return [rows[index] for index in selected_indices]

    selected_indices = sorted(
        random.Random(random_seed).sample(range(len(rows)), k=sample_size)
    )
    return [rows[index] for index in selected_indices]


def label_csv_with_vllm(
    input_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    settings: Settings,
    limit: int | None = None,
    random_seed: int | None = None,
    enable_random: bool | None = None,
    balance_by_source: bool | None = None,
) -> dict[str, Any]:
    rows = read_csv(input_path)
    total_rows_available = len(rows)
    effective_limit = settings.vllm.limit if limit is None else limit
    effective_random_seed = (
        settings.vllm.random_seed if random_seed is None else random_seed
    )
    effective_enable_random = (
        settings.vllm.enable_random if enable_random is None else enable_random
    )
    effective_balance_by_source = (
        settings.vllm.balance_by_source
        if balance_by_source is None
        else balance_by_source
    )
    rows = _select_rows_for_labeling(
        rows=rows,
        limit=effective_limit,
        enable_random=effective_enable_random,
        random_seed=effective_random_seed,
        balance_by_source=effective_balance_by_source,
    )

    unique_texts: list[str] = []
    seen_texts: set[str] = set()
    for row in rows:
        text = row.get("text", "").strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        unique_texts.append(text)

    cached_results: dict[str, VLLMLabelResult] = {}
    #prompts = [format_qwen_prompt(text) for text in unique_texts]
    prompts = [format_gemma_prompt(text) for text in unique_texts]

    if prompts:
        llm_class, sampling_params_class = _require_vllm_runtime()
        llm = llm_class(
            model=settings.vllm.model,
            quantization=settings.vllm.quantization,
            gpu_memory_utilization=settings.vllm.gpu_memory_utilization,
            max_model_len=settings.vllm.max_model_len,
            enforce_eager=True,
        )
        guided_schema = json.dumps(OUTPUT_SCHEMA)
        sampling_kwargs = {
            "temperature": settings.vllm.temperature,
            "max_tokens": 512,
            "stop": ["<end_of_turn>"], # for gemma
        }
        if GuidedDecodingParams is not None:
            sampling_kwargs["guided_decoding"] = GuidedDecodingParams(
                json=guided_schema,
                backend="auto",
            )
        else:
            sampling_kwargs["guided_json"] = guided_schema
            sampling_kwargs["guided_decoding_backend"] = "outlines"
        sampling_params = sampling_params_class(**sampling_kwargs)
        outputs = llm.generate(prompts, sampling_params)
        prompt_to_text = dict(zip(prompts, unique_texts))

        for output in tqdm(
            outputs,
            total=len(outputs),
            desc="Parsing vLLM outputs",
            unit="tweet",
        ):
            prompt = getattr(output, "prompt", "")
            text = prompt_to_text.get(prompt)
            if text is None:
                continue
            generations = getattr(output, "outputs", [])
            if not generations:
                cached_results[text] = build_error_result(
                    "vLLM returned no generations for this prompt."
                )
                continue
            cached_results[text] = extract_label_result(generations[0].text)

    labeled_rows: list[dict[str, str]] = []
    success_count = 0
    error_count = 0

    for row in rows:
        text = row.get("text", "").strip()
        if not text:
            result = build_skipped_result("empty_text")
        else:
            result = cached_results.get(
                text,
                build_error_result("No cached vLLM response was available for this tweet."),
            )

        if result.labeling_status == "ok":
            success_count += 1
        elif result.labeling_status == "error":
            error_count += 1

        labeled_rows.append(
            merge_row_with_label(
                row=row,
                result=result,
                model_name=settings.vllm.model,
            )
        )

    write_csv(output_path, labeled_rows, LABEL_COLUMNS)
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "total_rows_available": total_rows_available,
        "rows_requested": len(rows),
        "unique_texts_requested": len(unique_texts),
        "prompts_submitted": len(prompts),
        "rows_labeled_successfully": success_count,
        "rows_with_errors": error_count,
        "limit": effective_limit,
        "enable_random": effective_enable_random,
        "random_seed": effective_random_seed,
        "balance_by_source": effective_balance_by_source,
        "model": settings.vllm.model,
        "quantization": settings.vllm.quantization,
        "gpu_memory_utilization": settings.vllm.gpu_memory_utilization,
        "max_model_len": settings.vllm.max_model_len,
        "temperature": settings.vllm.temperature,
    }
    dump_json(summary_path, summary)
    return summary
