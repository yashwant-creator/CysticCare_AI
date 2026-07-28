"""Offline contracts for recovering a legacy Chroma-only corpus."""

from __future__ import annotations

import json
import unittest

from app.services.legacy_corpus_recovery import (
    LEGACY_CHUNK_ORIGIN,
    LegacyChromaRow,
    LegacyCorpusRecoveryError,
    iter_legacy_chroma_rows,
    recover_legacy_collection,
    recover_legacy_rows,
)


class FakeCollection:
    """Minimal public Chroma-like collection; no database or network needed."""

    def __init__(self, rows, *, name="legacy_fixture"):
        self.name = name
        self.rows = list(rows)
        self.calls = []

    def count(self):
        return len(self.rows)

    def get(self, *, limit, offset, include):
        self.calls.append({"limit": limit, "offset": offset, "include": list(include)})
        page = self.rows[offset : offset + limit]
        return {
            "ids": [row["id"] for row in page],
            "documents": [row["document"] for row in page],
            "metadatas": [row["metadata"] for row in page],
        }


class MalformedCollection(FakeCollection):
    def get(self, *, limit, offset, include):
        page = super().get(limit=limit, offset=offset, include=include)
        page["metadatas"] = page["metadatas"][:-1]
        return page


def _row(legacy_id, document, file_name, title, *, file_path=None):
    return {
        "id": legacy_id,
        "document": document,
        "metadata": {
            "file_name": file_name,
            "file_path": file_path or f"/legacy/papers/{file_name}",
            "title": title,
            "author": "Researcher",
            "year": "2024",
            "source_type": "scientific_paper",
        },
    }


def _paper_id(title, fallback_name):
    return f"paper_{title.casefold().replace(' ', '_')}"


class LegacyCorpusRecoveryTests(unittest.TestCase):
    def test_public_pagination_deduplicates_only_complete_alias_source_sets(self):
        # The two Alpha filenames have an identical complete set of chunks, so
        # they are aliases.  Beta's different content remains a separate paper.
        collection = FakeCollection(
            [
                _row(
                    "Alpha_2024_pdf_chunk_1",
                    "alpha conclusion",
                    "Alpha_2024.pdf",
                    "Alpha paper",
                ),
                _row(
                    "Beta_2023_pdf_chunk_0",
                    "beta evidence",
                    "Beta_2023.pdf",
                    "Beta paper",
                ),
                _row(
                    "Alpha-2024_pdf_chunk_0",
                    "alpha evidence",
                    "Alpha-2024.pdf",
                    "Alpha Paper (author copy)",
                ),
                _row(
                    "Alpha_2024_pdf_chunk_0",
                    "alpha evidence",
                    "Alpha_2024.pdf",
                    "Alpha paper",
                ),
                _row(
                    "Alpha-2024_pdf_chunk_1",
                    "alpha conclusion",
                    "Alpha-2024.pdf",
                    "Alpha Paper (author copy)",
                ),
            ]
        )

        recovery = recover_legacy_collection(
            collection,
            page_size=2,
            paper_id_factory=_paper_id,
        )

        self.assertEqual(
            collection.calls,
            [
                {"limit": 2, "offset": 0, "include": ["documents", "metadatas"]},
                {"limit": 2, "offset": 2, "include": ["documents", "metadatas"]},
                {"limit": 1, "offset": 4, "include": ["documents", "metadatas"]},
            ],
        )
        inventory = recovery.inventory
        self.assertEqual(inventory.source_row_count, 5)
        self.assertEqual(inventory.source_file_count, 3)
        self.assertEqual(inventory.logical_paper_count, 2)
        self.assertEqual(inventory.recovered_chunk_count, 3)
        self.assertEqual(inventory.duplicate_alias_source_set_count, 1)
        self.assertEqual(inventory.aliased_source_file_count, 2)
        self.assertEqual(inventory.duplicate_alias_source_file_count, 1)
        self.assertEqual(inventory.duplicate_alias_row_count, 2)
        self.assertEqual(recovery.legacy_vector_count, 5)
        self.assertEqual(recovery.legacy_document_count, 3)
        self.assertEqual(recovery.source_backup["collection_name"], "legacy_fixture")

        alpha_records = [
            record
            for record in recovery.records
            if record.metadata["source_file_name"] == "Alpha-2024.pdf"
        ]
        self.assertEqual([record.document for record in alpha_records], ["alpha evidence", "alpha conclusion"])
        self.assertEqual(
            [record.metadata["legacy_embedding_id"] for record in alpha_records],
            ["Alpha-2024_pdf_chunk_0", "Alpha-2024_pdf_chunk_1"],
        )
        self.assertEqual(
            [record.metadata["chunk_id"] for record in alpha_records],
            [record.id for record in alpha_records],
        )
        self.assertEqual(alpha_records[0].metadata["previous_chunk_id"], "")
        self.assertEqual(alpha_records[0].metadata["next_chunk_id"], alpha_records[1].id)
        self.assertEqual(alpha_records[1].metadata["previous_chunk_id"], alpha_records[0].id)
        self.assertEqual(alpha_records[1].metadata["next_chunk_id"], "")

        metadata = alpha_records[0].metadata
        self.assertEqual(metadata["chunk_origin"], LEGACY_CHUNK_ORIGIN)
        self.assertFalse(metadata["structure_available"])
        self.assertTrue(metadata["structure_unavailable"])
        self.assertEqual(metadata["structure_status"], "unavailable_from_legacy_chroma")
        self.assertEqual(metadata["page_number"], 0)
        self.assertEqual(metadata["page_start"], 0)
        self.assertEqual(metadata["page_end"], 0)
        self.assertEqual(metadata["section_heading"], "")
        self.assertEqual(metadata["file_name"], "Alpha-2024.pdf")
        self.assertEqual(
            json.loads(metadata["legacy_source_file_aliases_json"]),
            ["Alpha-2024.pdf", "Alpha_2024.pdf"],
        )
        self.assertIn("file:alpha2024", recovery.legacy_match_keys)
        self.assertIn("file:alpha2024", recovery.legacy_match_keys)

        payload = recovery.as_chroma_payload()
        self.assertEqual(payload["ids"], [record.id for record in recovery.records])
        self.assertEqual(payload["documents"], [record.document for record in recovery.records])
        self.assertEqual(payload["metadatas"][0]["chunk_origin"], LEGACY_CHUNK_ORIGIN)

    def test_same_title_with_nonidentical_content_is_not_merged(self):
        rows = (
            LegacyChromaRow(
                legacy_embedding_id="one_chunk_0",
                document="first distinct paper",
                metadata={"file_name": "one.pdf", "title": "Shared title"},
                source_position=0,
            ),
            LegacyChromaRow(
                legacy_embedding_id="two_chunk_0",
                document="second distinct paper",
                metadata={"file_name": "two.pdf", "title": "Shared title"},
                source_position=1,
            ),
        )

        recovery = recover_legacy_rows(rows, paper_id_factory=lambda *_: "paper_shared")

        self.assertEqual(recovery.inventory.logical_paper_count, 2)
        paper_ids = {record.metadata["paper_id"] for record in recovery.records}
        self.assertEqual(len(paper_ids), 2)
        self.assertIn("paper_shared", paper_ids)
        self.assertTrue(any(paper_id.startswith("paper_shared_") for paper_id in paper_ids))

    def test_rejects_misaligned_public_get_page(self):
        collection = MalformedCollection(
            [_row("one_chunk_0", "text", "one.pdf", "One")]
        )
        with self.assertRaisesRegex(LegacyCorpusRecoveryError, "not aligned"):
            list(iter_legacy_chroma_rows(collection, page_size=10))


if __name__ == "__main__":
    unittest.main()
