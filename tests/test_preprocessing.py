import unittest

from ragebait_detector.data.preprocessing import (
    augment_text,
    clean_text,
    is_media_only_or_empty,
    meaningful_length,
    normalize_label,
)


class PreprocessingTests(unittest.TestCase):
    def test_clean_text_replaces_platform_artifacts(self):
        text = "Visit https://example.com @friend #Drama 123 !!!"
        cleaned = clean_text(text)
        self.assertIn("[url]", cleaned)
        self.assertIn("[user]", cleaned)
        self.assertIn("[hashtag]", cleaned)
        self.assertNotIn("123", cleaned)

    def test_media_only_detection_flags_token_only_posts(self):
        cleaned = clean_text("@user https://t.co/abc #wow")
        self.assertTrue(is_media_only_or_empty(cleaned))

    def test_meaningful_length_ignores_special_tokens(self):
        self.assertEqual(meaningful_length("[url] [user] [hashtag]"), 0)
        self.assertGreater(meaningful_length("this is real text"), 0)

    def test_normalize_label_supports_common_aliases(self):
        self.assertEqual(normalize_label("ragebait"), 1)
        self.assertEqual(normalize_label("not_ragebait"), 0)
        self.assertIsNone(normalize_label("maybe"))

    def test_augmentation_preserves_token_count(self):
        text = "this post is intentionally trying to trigger angry replies"
        augmented = augment_text(text, seed=7)
        self.assertEqual(len(text.split()), len(augmented.split()))


if __name__ == "__main__":
    unittest.main()
