from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person1.conversion import convert_ragtruth_to_person1


class Person1ConversionTests(unittest.TestCase):
    def test_conversion_creates_person1_jsonl_rows(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        response_jsonl = repo_root / "dataset" / "response.jsonl"
        source_info_jsonl = repo_root / "dataset" / "source_info.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_jsonl = Path(tmpdir) / "converted.jsonl"
            summary = convert_ragtruth_to_person1(
                response_jsonl=response_jsonl,
                source_info_jsonl=source_info_jsonl,
                output_jsonl=output_jsonl,
                limit=2,
            )

            self.assertEqual(summary["rows_written"], 2)
            self.assertTrue(output_jsonl.exists())

            rows = [
                json.loads(line)
                for line in output_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 2)
            self.assertIn("sample_id", rows[0])
            self.assertIn("retrieved_context", rows[0])
            self.assertIn("answer", rows[0])


if __name__ == "__main__":
    unittest.main()
