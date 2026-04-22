import json
import random
import unittest
from unittest.mock import patch

from ragebait_detector.config import Settings
from ragebait_detector.labeling.vllm_labeler import (
    OUTPUT_SCHEMA,
    extract_label_result,
    format_qwen_prompt,
    label_csv_with_vllm,
    merge_row_with_label,
)


class _FakeGeneratedText:
    def __init__(self, text):
        self.text = text


class _FakeRequestOutput:
    def __init__(self, prompt, text):
        self.prompt = prompt
        self.outputs = [_FakeGeneratedText(text)]


class _FakeLLM:
    init_kwargs = None
    generate_calls = []

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def generate(self, prompts, sampling_params):
        type(self).generate_calls.append((list(prompts), sampling_params))
        return [
            _FakeRequestOutput(
                prompt,
                json.dumps(
                    {
                        "is_ragebait": "stay mad" in prompt.lower(),
                        "confidence": 0.92 if "stay mad" in prompt.lower() else 0.08,
                        "reason": "provocative bait"
                        if "stay mad" in prompt.lower()
                        else "neutral update",
                    }
                ),
            )
            for prompt in prompts
        ]


class _FakeSamplingParams:
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs


class VLLMLabelerTests(unittest.TestCase):
    def setUp(self):
        _FakeLLM.init_kwargs = None
        _FakeLLM.generate_calls = []
        _FakeSamplingParams.init_kwargs = None

    def test_format_qwen_prompt_uses_chatml(self):
        prompt = format_qwen_prompt("stay mad online")
        self.assertTrue(prompt.startswith("<|im_start|>system\n"))
        self.assertIn("Classify this tweet:\nstay mad online", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_extract_label_result_reads_guided_json(self):
        result = extract_label_result(
            '{"is_ragebait": true, "confidence": 0.91, "reason": "deliberately provocative"}'
        )
        self.assertTrue(result.is_ragebait)
        self.assertAlmostEqual(result.confidence or 0.0, 0.91)
        self.assertEqual(result.reason, "deliberately provocative")
        self.assertEqual(result.labeling_status, "ok")
        self.assertEqual(result.parse_mode, "strict_json")

    def test_extract_label_result_salvages_missing_comma_json(self):
        result = extract_label_result(
            '{"is_ragebait": false "confidence": 0.27, "reason": "calm discussion"}'
        )
        self.assertFalse(result.is_ragebait)
        self.assertAlmostEqual(result.confidence or 0.0, 0.27)
        self.assertEqual(result.reason, "calm discussion")
        self.assertEqual(result.labeling_status, "ok")
        self.assertEqual(result.parse_mode, "salvaged_regex")

    def test_extract_label_result_salvages_wrapped_json(self):
        result = extract_label_result(
            '```json\n{"is_ragebait": true, "confidence": 0.88, "reason": "trying to provoke"}\n```'
        )
        self.assertTrue(result.is_ragebait)
        self.assertAlmostEqual(result.confidence or 0.0, 0.88)
        self.assertEqual(result.reason, "trying to provoke")
        self.assertEqual(result.labeling_status, "ok")
        self.assertEqual(result.parse_mode, "strict_json")

    def test_extract_label_result_salvages_truncated_reason_json(self):
        result = extract_label_result(
            '{"is_ragebait": true, "confidence": 0.66, "reason": "intentionally inflammatory"'
        )
        self.assertTrue(result.is_ragebait)
        self.assertAlmostEqual(result.confidence or 0.0, 0.66)
        self.assertEqual(result.reason, "intentionally inflammatory")
        self.assertEqual(result.labeling_status, "ok")
        self.assertEqual(result.parse_mode, "salvaged_regex")

    def test_extract_label_result_normalizes_single_quoted_json(self):
        result = extract_label_result(
            "{'is_ragebait': false, 'confidence': 0.13, 'reason': 'ordinary post'}"
        )
        self.assertFalse(result.is_ragebait)
        self.assertAlmostEqual(result.confidence or 0.0, 0.13)
        self.assertEqual(result.reason, "ordinary post")
        self.assertEqual(result.labeling_status, "ok")
        self.assertEqual(result.parse_mode, "single_quote_normalized")

    def test_merge_row_with_label_outputs_expected_columns(self):
        result = extract_label_result(
            '{"is_ragebait": false, "confidence": 0.12, "reason": "plain update"}'
        )
        merged = merge_row_with_label(
            {
                "post_id": "1",
                "author_id": "7",
                "created_at": "",
                "language": "en",
                "text": "plain update",
                "source": "set_a",
            },
            result=result,
            model_name="Qwen/Qwen2.5-3B-Instruct-AWQ",
        )
        self.assertEqual(merged["label"], "0")
        self.assertEqual(merged["is_ragebait"], "false")
        self.assertEqual(merged["llm_model"], "Qwen/Qwen2.5-3B-Instruct-AWQ")
        self.assertEqual(merged["parse_mode"], "strict_json")

    def test_label_csv_with_vllm_uses_flat_prompt_list(self):
        settings = Settings()
        rows = [
            {
                "post_id": "1",
                "author_id": "a",
                "created_at": "",
                "language": "en",
                "text": "stay mad",
                "source": "sample",
            },
            {
                "post_id": "2",
                "author_id": "b",
                "created_at": "",
                "language": "en",
                "text": "here is a neutral update",
                "source": "sample",
            },
            {
                "post_id": "3",
                "author_id": "c",
                "created_at": "",
                "language": "en",
                "text": "stay mad",
                "source": "sample",
            },
            {
                "post_id": "4",
                "author_id": "d",
                "created_at": "",
                "language": "en",
                "text": "",
                "source": "sample",
            },
        ]

        with patch(
            "ragebait_detector.labeling.vllm_labeler.read_csv",
            return_value=rows,
        ), patch(
            "ragebait_detector.labeling.vllm_labeler.write_csv"
        ) as write_csv_mock, patch(
            "ragebait_detector.labeling.vllm_labeler.dump_json"
        ), patch(
            "ragebait_detector.labeling.vllm_labeler.LLM",
            _FakeLLM,
        ), patch(
            "ragebait_detector.labeling.vllm_labeler.SamplingParams",
            _FakeSamplingParams,
        ):
            summary = label_csv_with_vllm(
                input_path="input.csv",
                output_path="output.csv",
                summary_path="summary.json",
                settings=settings,
            )

        self.assertEqual(summary["unique_texts_requested"], 2)
        self.assertEqual(summary["prompts_submitted"], 2)
        self.assertEqual(_FakeLLM.init_kwargs["model"], settings.vllm.model)
        self.assertEqual(_FakeLLM.init_kwargs["quantization"], settings.vllm.quantization)
        if "guided_decoding" in _FakeSamplingParams.init_kwargs:
            guided_decoding = _FakeSamplingParams.init_kwargs["guided_decoding"]
            self.assertEqual(guided_decoding.json, json.dumps(OUTPUT_SCHEMA))
            self.assertEqual(guided_decoding.backend, "auto")
        else:
            self.assertEqual(
                _FakeSamplingParams.init_kwargs["guided_json"],
                json.dumps(OUTPUT_SCHEMA),
            )
            self.assertEqual(
                _FakeSamplingParams.init_kwargs["guided_decoding_backend"],
                "outlines",
            )

        submitted_prompts, _ = _FakeLLM.generate_calls[0]
        self.assertEqual(len(submitted_prompts), 2)
        self.assertTrue(
            all(prompt.count("Classify this tweet:") == 1 for prompt in submitted_prompts)
        )
        self.assertTrue(any("stay mad" in prompt for prompt in submitted_prompts))
        self.assertTrue(
            any("here is a neutral update" in prompt for prompt in submitted_prompts)
        )

        labeled_rows = write_csv_mock.call_args.args[1]
        self.assertEqual(labeled_rows[0]["label"], "1")
        self.assertEqual(labeled_rows[1]["label"], "0")
        self.assertEqual(labeled_rows[2]["label"], "1")
        self.assertEqual(labeled_rows[3]["labeling_status"], "skipped")
        self.assertEqual(labeled_rows[3]["error"], "empty_text")
        self.assertEqual(labeled_rows[0]["parse_mode"], "strict_json")

    def test_label_csv_with_vllm_uses_seeded_random_limit_without_duplicate_rows(self):
        settings = Settings()
        settings.vllm.enable_random = True
        settings.vllm.random_seed = 7
        settings.vllm.balance_by_source = False
        rows = [
            {
                "post_id": str(index),
                "author_id": f"user-{index}",
                "created_at": "",
                "language": "en",
                "text": f"tweet {index}",
                "source": "sample",
            }
            for index in range(6)
        ]

        with patch(
            "ragebait_detector.labeling.vllm_labeler.read_csv",
            return_value=rows,
        ), patch(
            "ragebait_detector.labeling.vllm_labeler.write_csv"
        ) as write_csv_mock, patch(
            "ragebait_detector.labeling.vllm_labeler.dump_json"
        ), patch(
            "ragebait_detector.labeling.vllm_labeler.LLM",
            _FakeLLM,
        ), patch(
            "ragebait_detector.labeling.vllm_labeler.SamplingParams",
            _FakeSamplingParams,
        ):
            summary = label_csv_with_vllm(
                input_path="input.csv",
                output_path="output.csv",
                summary_path="summary.json",
                settings=settings,
                limit=3,
            )

        expected_indices = sorted(random.Random(7).sample(range(len(rows)), k=3))
        expected_post_ids = [rows[index]["post_id"] for index in expected_indices]

        labeled_rows = write_csv_mock.call_args.args[1]
        selected_post_ids = [row["post_id"] for row in labeled_rows]

        self.assertEqual(summary["rows_requested"], 3)
        self.assertEqual(summary["limit"], 3)
        self.assertTrue(summary["enable_random"])
        self.assertEqual(summary["random_seed"], 7)
        self.assertFalse(summary["balance_by_source"])
        self.assertEqual(selected_post_ids, expected_post_ids)
        self.assertEqual(len(selected_post_ids), len(set(selected_post_ids)))

    def test_label_csv_with_vllm_balances_random_sampling_by_source(self):
        settings = Settings()
        settings.vllm.enable_random = True
        settings.vllm.random_seed = 11
        settings.vllm.balance_by_source = True
        rows = [
            {
                "post_id": "a1",
                "author_id": "user-a1",
                "created_at": "",
                "language": "en",
                "text": "tweet a1",
                "source": "set_a",
            },
            {
                "post_id": "a2",
                "author_id": "user-a2",
                "created_at": "",
                "language": "en",
                "text": "tweet a2",
                "source": "set_a",
            },
            {
                "post_id": "a3",
                "author_id": "user-a3",
                "created_at": "",
                "language": "en",
                "text": "tweet a3",
                "source": "set_a",
            },
            {
                "post_id": "a4",
                "author_id": "user-a4",
                "created_at": "",
                "language": "en",
                "text": "tweet a4",
                "source": "set_a",
            },
            {
                "post_id": "b1",
                "author_id": "user-b1",
                "created_at": "",
                "language": "en",
                "text": "tweet b1",
                "source": "set_b",
            },
            {
                "post_id": "b2",
                "author_id": "user-b2",
                "created_at": "",
                "language": "en",
                "text": "tweet b2",
                "source": "set_b",
            },
        ]

        with patch(
            "ragebait_detector.labeling.vllm_labeler.read_csv",
            return_value=rows,
        ), patch(
            "ragebait_detector.labeling.vllm_labeler.write_csv"
        ) as write_csv_mock, patch(
            "ragebait_detector.labeling.vllm_labeler.dump_json"
        ), patch(
            "ragebait_detector.labeling.vllm_labeler.LLM",
            _FakeLLM,
        ), patch(
            "ragebait_detector.labeling.vllm_labeler.SamplingParams",
            _FakeSamplingParams,
        ):
            summary = label_csv_with_vllm(
                input_path="input.csv",
                output_path="output.csv",
                summary_path="summary.json",
                settings=settings,
                limit=4,
            )

        labeled_rows = write_csv_mock.call_args.args[1]
        source_counts: dict[str, int] = {}
        for row in labeled_rows:
            source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1

        self.assertEqual(summary["rows_requested"], 4)
        self.assertTrue(summary["balance_by_source"])
        self.assertEqual(source_counts, {"set_a": 2, "set_b": 2})
        self.assertEqual(len(labeled_rows), 4)


if __name__ == "__main__":
    unittest.main()
