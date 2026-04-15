import unittest
from unittest.mock import patch

from ragebait_detector.labeling.ollama_labeler import (
    OllamaBatchItem,
    OllamaLabelResult,
    extract_batch_results,
    extract_tool_result,
    recover_incomplete_batch_results,
    merge_row_with_label,
)


class OllamaLabelerTests(unittest.TestCase):
    def test_extract_tool_result_reads_dict_arguments(self):
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "classify_ragebait",
                            "arguments": {
                                "is_ragebait": True,
                                "confidence": 0.91,
                                "reason": "deliberately provocative",
                            },
                        }
                    }
                ]
            }
        }
        result = extract_tool_result(response)
        self.assertTrue(result.is_ragebait)
        self.assertEqual(result.labeling_status, "ok")
        self.assertTrue(result.used_tool_call)

    def test_extract_tool_result_reads_string_arguments(self):
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "classify_ragebait",
                            "arguments": '{"is_ragebait": false, "confidence": 0.3, "reason": "sincere complaint"}',
                        }
                    }
                ]
            }
        }
        result = extract_tool_result(response)
        self.assertFalse(result.is_ragebait)
        self.assertAlmostEqual(result.confidence or 0.0, 0.3)

    def test_extract_batch_results_reads_json_array_content(self):
        response = {
            "message": {
                "content": (
                    '[{"id":"1","is_ragebait":true,"confidence":0.91,"reason":"deliberately provocative"},'
                    '{"id":"2","is_ragebait":false,"confidence":0.17,"reason":"plain update"}]'
                )
            }
        }
        batch = [
            OllamaBatchItem(item_id="1", text="stay mad"),
            OllamaBatchItem(item_id="2", text="here is a neutral update"),
        ]

        results = extract_batch_results(response, batch, latency_ms=321)

        self.assertTrue(results["stay mad"].is_ragebait)
        self.assertEqual(results["stay mad"].labeling_status, "ok_json")
        self.assertEqual(results["stay mad"].latency_ms, 321)
        self.assertFalse(results["here is a neutral update"].is_ragebait)

    def test_extract_batch_results_reads_wrapped_payload(self):
        response = {
            "message": {
                "content": (
                    '{"results":[{"id":"1","is_ragebait":"false","confidence":"0.25","reason":"sincere complaint"}]}'
                )
            }
        }
        batch = [OllamaBatchItem(item_id="1", text="i am upset about the layoffs")]

        results = extract_batch_results(response, batch, latency_ms=99)

        result = results["i am upset about the layoffs"]
        self.assertFalse(result.is_ragebait)
        self.assertAlmostEqual(result.confidence or 0.0, 0.25)
        self.assertEqual(result.labeling_status, "ok_json")

    def test_extract_batch_results_reads_single_item_without_id(self):
        response = {
            "message": {
                "content": (
                    '{"is_ragebait":false,"confidence":1.0,"reason":"neutral announcement"}'
                )
            }
        }
        batch = [OllamaBatchItem(item_id="1", text="launch update")]

        results = extract_batch_results(response, batch, latency_ms=77)

        result = results["launch update"]
        self.assertFalse(result.is_ragebait)
        self.assertAlmostEqual(result.confidence or 0.0, 1.0)
        self.assertEqual(result.labeling_status, "ok_json")
        self.assertEqual(result.latency_ms, 77)

    def test_extract_batch_results_reads_keyed_object_payload(self):
        response = {
            "message": {
                "content": (
                    '{"0":{"id":"1","is_ragebait":false,"confidence":1.0,"reason":"neutral announcement"}}'
                )
            }
        }
        batch = [OllamaBatchItem(item_id="1", text="weekly nasa update")]

        results = extract_batch_results(response, batch, latency_ms=88)

        result = results["weekly nasa update"]
        self.assertFalse(result.is_ragebait)
        self.assertAlmostEqual(result.confidence or 0.0, 1.0)
        self.assertEqual(result.labeling_status, "ok_json")

    def test_extract_batch_results_marks_missing_items_as_errors(self):
        response = {"message": {"content": '[{"id":"1","is_ragebait":true,"reason":"bait"}]'}}
        batch = [
            OllamaBatchItem(item_id="1", text="stay mad"),
            OllamaBatchItem(item_id="2", text="plain update"),
        ]

        results = extract_batch_results(response, batch, latency_ms=50)

        self.assertTrue(results["stay mad"].is_ragebait)
        self.assertEqual(results["plain update"].labeling_status, "error")
        self.assertIn("did not contain", results["plain update"].error)

    def test_recover_incomplete_batch_results_splits_unresolved_items(self):
        batch = [
            OllamaBatchItem(item_id="1", text="stay mad"),
            OllamaBatchItem(item_id="2", text="plain update"),
            OllamaBatchItem(item_id="3", text="another update"),
        ]
        current_results = {
            "stay mad": OllamaLabelResult(
                is_ragebait=True,
                confidence=0.9,
                reason="bait",
                tool_name=None,
                used_tool_call=False,
                labeling_status="ok_json",
            ),
            "plain update": OllamaLabelResult(
                is_ragebait=None,
                confidence=None,
                reason="",
                tool_name=None,
                used_tool_call=False,
                labeling_status="error",
                error="missing",
            ),
            "another update": OllamaLabelResult(
                is_ragebait=None,
                confidence=None,
                reason="",
                tool_name=None,
                used_tool_call=False,
                labeling_status="error",
                error="missing",
            ),
        }

        def fake_classify(sub_batch, **kwargs):
            return {
                item.text: OllamaLabelResult(
                    is_ragebait=False,
                    confidence=0.2,
                    reason=f"recovered {item.item_id}",
                    tool_name=None,
                    used_tool_call=False,
                    labeling_status="ok_json",
                )
                for item in sub_batch
            }

        with patch("ragebait_detector.labeling.ollama_labeler.classify_batch_with_ollama", side_effect=fake_classify):
            recovered = recover_incomplete_batch_results(
                batch,
                current_results,
                host="http://127.0.0.1:11434",
                model="qwen2.5:3b-instruct-q4_K_M",
                timeout_seconds=30,
                temperature=0.0,
                max_retries=0,
                session=None,
            )

        self.assertTrue(recovered["stay mad"].is_ragebait)
        self.assertFalse(recovered["plain update"].is_ragebait)
        self.assertFalse(recovered["another update"].is_ragebait)
        self.assertEqual(recovered["another update"].reason, "recovered 3")

    def test_merge_row_with_label_outputs_expected_columns(self):
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "classify_ragebait",
                            "arguments": {"is_ragebait": True},
                        }
                    }
                ]
            }
        }
        result = extract_tool_result(response)
        merged = merge_row_with_label(
            {
                "post_id": "1",
                "author_id": "7",
                "created_at": "",
                "language": "en",
                "text": "stay mad",
                "source": "set_a",
            },
            result=result,
            model_name="llama3.1:8b",
        )
        self.assertEqual(merged["label"], "1")
        self.assertEqual(merged["is_ragebait"], "true")
        self.assertEqual(merged["llm_model"], "llama3.1:8b")


if __name__ == "__main__":
    unittest.main()
