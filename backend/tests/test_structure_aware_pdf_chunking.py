import unittest
from unittest.mock import patch

from app.utils import openai_utils


class StructureAwarePdfChunkingTests(unittest.TestCase):
    def setUp(self):
        # Keep the tests offline and deterministic even when the tiktoken model
        # definition has not been cached in the test environment.
        self.original_tokenizer = openai_utils._tokenizer
        openai_utils._tokenizer = openai_utils._WhitespaceTokenizer()

    def tearDown(self):
        openai_utils._tokenizer = self.original_tokenizer

    def test_chunks_keep_section_page_parent_and_neighbor_metadata(self):
        results_lines = "\n".join(
            (
                f"Result {index} shows that tolvaptan slowed total kidney-volume growth "
                "in adults with rapidly progressive autosomal dominant polycystic kidney disease."
            )
            for index in range(1, 9)
        )
        pages = [
            {
                "page_number": 1,
                "text": "ABSTRACT\nThis study evaluates treatment outcomes in ADPKD.",
            },
            {"page_number": 2, "text": f"RESULTS\n{results_lines}"},
        ]

        chunks, metadatas = openai_utils.chunk_pdf_pages(
            pages,
            document_key="tolvaptan paper",
            document_title="Tolvaptan in ADPKD",
            chunk_size_tokens=64,
            chunk_overlap_tokens=12,
        )

        self.assertEqual(len(chunks), len(metadatas))
        self.assertGreater(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("[Document: Tolvaptan in ADPKD | Section: ABSTRACT]"))
        self.assertEqual(metadatas[0]["page_number"], 1)
        self.assertEqual(metadatas[0]["section_heading"], "ABSTRACT")

        results = [metadata for metadata in metadatas if metadata["section_heading"] == "RESULTS"]
        self.assertGreaterEqual(len(results), 2)
        self.assertTrue(all(metadata["page_number"] == 2 for metadata in results))
        self.assertTrue(all(metadata["parent_document_id"] == "tolvaptan_paper" for metadata in metadatas))
        self.assertEqual([metadata["chunk_index"] for metadata in metadatas], list(range(len(metadatas))))
        self.assertEqual(results[0]["previous_chunk_index"], -1)
        self.assertEqual(results[0]["next_chunk_index"], results[1]["chunk_index"])
        self.assertEqual(results[1]["previous_chunk_index"], results[0]["chunk_index"])
        self.assertLess(results[0]["chunk_start"], results[0]["chunk_end"])

        for metadata in metadatas:
            self.assertTrue(
                all(isinstance(value, (str, int, float, bool)) for value in metadata.values())
            )

    @patch("app.utils.openai_utils.extract_metadata")
    @patch("app.utils.openai_utils.extract_text_pages_from_pdf")
    def test_metadata_process_helper_aligns_metadata_to_chunks(self, mock_extract_pages, mock_extract_metadata):
        mock_extract_pages.return_value = [
            {"page_number": 3, "text": "DISCUSSION\nThe finding supports individualized treatment."},
        ]
        mock_extract_metadata.return_value = {
            "title": "A Study",
            "author": "Author",
            "file_name": "a_study.pdf",
        }

        chunks, basic_metadata, chunk_metadatas = openai_utils.process_pdf_file_with_metadata(
            "/papers/a_study.pdf",
            chunk_size_tokens=64,
            chunk_overlap_tokens=8,
        )

        self.assertEqual(basic_metadata["title"], "A Study")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunk_metadatas), 1)
        self.assertEqual(chunk_metadatas[0]["page_number"], 3)
        self.assertEqual(chunk_metadatas[0]["section_heading"], "DISCUSSION")
        self.assertEqual(chunk_metadatas[0]["document_key"], "a_study")
        self.assertEqual(chunk_metadatas[0]["chunk_id"], "a_study_chunk_0")

    @patch("app.utils.openai_utils.extract_metadata")
    @patch("app.utils.openai_utils.extract_text_from_pdf")
    def test_legacy_process_pdf_file_still_returns_two_items(self, mock_extract_text, mock_extract_metadata):
        mock_extract_text.return_value = "one two three"
        mock_extract_metadata.return_value = {"title": "Legacy"}

        chunks, metadata = openai_utils.process_pdf_file("legacy.pdf")

        self.assertEqual(chunks, ["one two three"])
        self.assertEqual(metadata, {"title": "Legacy"})


if __name__ == "__main__":
    unittest.main()
