import unittest

import pandas as pd

from ragebait_detector.data.unifier import (
    compile_rows_from_dataframe,
    deduplicate_rows,
    extract_sql_table_definitions,
    parse_numeric_selection,
)


class UnifierTests(unittest.TestCase):
    def test_parse_numeric_selection_supports_ranges_and_lists(self):
        self.assertEqual(parse_numeric_selection("1,3-4", upper_bound=5), [0, 2, 3])
        self.assertEqual(parse_numeric_selection("2:5", upper_bound=5), [1, 2, 3, 4])

    def test_extract_sql_table_definitions_reads_create_table_columns(self):
        sql = """
        CREATE TABLE tweets (
            id BIGINT,
            body TEXT,
            lang TEXT,
            PRIMARY KEY (id)
        );
        """
        tables = extract_sql_table_definitions(sql)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].name, "tweets")
        self.assertEqual(tables[0].columns, ["id", "body", "lang"])

    def test_compile_rows_from_dataframe_builds_final_shape(self):
        frame = pd.DataFrame(
            [
                {"body": "Hello world", "user_id": 7, "lang": "EN"},
                {"body": "   ", "user_id": 9, "lang": "EN"},
            ]
        )
        batch = compile_rows_from_dataframe(
            dataframe=frame,
            mapping={
                "text": "body",
                "author_id": "user_id",
                "created_at": None,
                "language": "lang",
            },
            source_name="kaggle_set_a",
            starting_post_id=1,
        )
        self.assertEqual(len(batch.rows), 1)
        self.assertEqual(batch.rows[0]["post_id"], "1")
        self.assertEqual(batch.rows[0]["author_id"], "7")
        self.assertEqual(batch.rows[0]["language"], "en")
        self.assertEqual(batch.rows[0]["text"], "Hello world")
        self.assertEqual(batch.skipped_empty_text, 1)

    def test_deduplicate_rows_reassigns_incremental_post_ids(self):
        rows = [
            {
                "post_id": "10",
                "author_id": "1",
                "created_at": "",
                "language": "en",
                "text": "same",
                "source": "set",
            },
            {
                "post_id": "11",
                "author_id": "1",
                "created_at": "",
                "language": "en",
                "text": "same",
                "source": "set",
            },
        ]
        deduped, removed = deduplicate_rows(rows)
        self.assertEqual(removed, 1)
        self.assertEqual(deduped[0]["post_id"], "1")


if __name__ == "__main__":
    unittest.main()

