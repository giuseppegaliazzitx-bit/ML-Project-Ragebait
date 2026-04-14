import unittest

from ragebait_detector.labeling.ollama_labeler import extract_tool_result, merge_row_with_label


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
