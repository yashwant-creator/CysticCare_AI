import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.utils.paper_identity import (
    _load_benchmark_title_ids,
    normalize_paper_identity,
    stable_paper_id,
)


class PaperIdentityTests(unittest.TestCase):
    def test_normalization_is_filename_and_punctuation_resilient(self):
        self.assertEqual(
            normalize_paper_identity("A PKD Paper: Results!"),
            normalize_paper_identity("a pkd-paper results"),
        )

    def test_benchmark_title_mapping_preserves_existing_paper_id(self):
        with patch(
            "app.utils.paper_identity.benchmark_title_ids",
            return_value={"physiologicmechanismsunderlyingpkd": "f8ddf0fd"},
        ):
            paper_id = stable_paper_id(
                "Physiologic mechanisms underlying PKD",
                "Boletta_2025",
            )
        self.assertEqual(paper_id, "f8ddf0fd")

    def test_fallback_is_deterministic_and_chroma_safe(self):
        with patch("app.utils.paper_identity.benchmark_title_ids", return_value={}):
            first = stable_paper_id("A new paper", "new.pdf")
            second = stable_paper_id("A new paper", "renamed.pdf")
        self.assertEqual(first, second)
        self.assertRegex(first, r"^paper_[a-f0-9]{12}$")

    def test_generic_pdf_title_uses_filename_identity(self):
        with patch("app.utils.paper_identity.benchmark_title_ids", return_value={}):
            unknown_one = stable_paper_id("Unknown", "first_paper.pdf")
            unknown_two = stable_paper_id("Untitled", "second_paper.pdf")

        self.assertNotEqual(unknown_one, unknown_two)

    def test_loader_reads_source_and_passage_title_ids(self):
        records = [{
            "source_papers": [{"paper_id": "paper-a", "paper_title": "A Paper"}],
            "supporting_passages": [{"paper_id": "paper-b", "paper_title": "B Paper"}],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            mapping = _load_benchmark_title_ids(path)
        self.assertEqual(mapping["apaper"], "paper-a")
        self.assertEqual(mapping["bpaper"], "paper-b")


if __name__ == "__main__":
    unittest.main()
