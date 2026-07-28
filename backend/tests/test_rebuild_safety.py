"""Safety contracts for rebuilding and serving the vector index.

These tests intentionally mock every external boundary.  They ensure an
incomplete collection is never queried or implicitly replaced, and that the
standalone rebuild script does not open/delete a collection until preflight
has succeeded.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import rebuild_vectordb
from app.services import openai_rag_init


class ValidatedService:
    """OpenAI stand-in that makes no network calls."""

    def __init__(self, **_kwargs):
        self.get_embedding = MagicMock()

    def validate_connection(self):
        return True


class BuildingCollection:
    def __init__(self, count=7):
        self._count = count
        self.metadata = {
            "index_schema_version": openai_rag_init.INDEX_SCHEMA_VERSION,
            "index_build_state": "building",
        }
        self.query = MagicMock()

    def count(self):
        return self._count


class RebuildSafetyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_client = openai_rag_init.openai_chroma_client
        self.original_collection = openai_rag_init.openai_collection
        self.original_service = openai_rag_init.openai_service
        self.original_helper_service = openai_rag_init.openai_helper_service

    def tearDown(self):
        openai_rag_init.openai_chroma_client = self.original_client
        openai_rag_init.openai_collection = self.original_collection
        openai_rag_init.openai_service = self.original_service
        openai_rag_init.openai_helper_service = self.original_helper_service

    async def test_initialization_refuses_nonempty_incomplete_collection_without_deleting_it(self):
        """Startup must not replace a partially built nonempty index."""
        collection = BuildingCollection(count=7)
        client = MagicMock()
        client.get_collection.return_value = collection

        config = {
            "embedding_model": "test-embedding",
            "chat_model": "test-chat",
            "vision_model": "test-vision",
            "max_retries": 0,
            "retry_delay": 0,
        }
        with (
            patch.object(openai_rag_init, "load_session_config", return_value=config),
            patch.object(openai_rag_init, "OpenAIService", ValidatedService),
            patch.object(openai_rag_init, "_resolve_pdf_directory", return_value="/no-pdfs"),
            patch.object(openai_rag_init.os, "makedirs"),
            patch.object(openai_rag_init.chromadb, "PersistentClient", return_value=client),
            patch.object(openai_rag_init, "process_pdf_file_with_metadata") as process_pdf,
        ):
            result = await openai_rag_init.build_openai_rag_system()

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["requires_rebuild"])
        self.assertEqual(result["total_vectors"], 7)
        self.assertIn("incomplete", result["message"].lower())
        client.delete_collection.assert_not_called()
        client.create_collection.assert_not_called()
        process_pdf.assert_not_called()

    async def test_search_refuses_building_collection_before_embedding_or_chroma_query(self):
        """A partial index must not serve a mixed/partial retrieval result."""
        collection = BuildingCollection()
        service = MagicMock()

        with (
            patch.object(openai_rag_init, "openai_collection", collection),
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_helper_service", service),
        ):
            result = await openai_rag_init.search_knowledge_base("What is ADPKD?")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["results"], [])
        self.assertIn("incomplete", result["message"].lower())
        service.get_embedding.assert_not_called()
        collection.query.assert_not_called()

    async def test_ready_state_update_excludes_immutable_hnsw_settings(self):
        metadata = openai_rag_init._mutable_collection_metadata(
            {
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 100,
                "index_schema_version": "2",
                "index_build_state": "building",
            }
        )

        self.assertNotIn("hnsw:space", metadata)
        self.assertNotIn("hnsw:construction_ef", metadata)
        self.assertEqual(metadata["index_schema_version"], "2")
        self.assertEqual(metadata["index_build_state"], "building")

    async def test_preflight_stops_before_openai_configuration_when_no_pdfs_exist(self):
        """A missing corpus does not trigger a network/API configuration path."""
        with (
            patch.object(openai_rag_init, "_resolve_pdf_directory", return_value="/no-pdfs"),
            patch("app.utils.openai_utils.get_pdf_files", return_value=[]),
            patch("app.services.openai_service.OpenAIService") as service_class,
        ):
            result = await rebuild_vectordb.preflight()

        self.assertFalse(result)
        service_class.assert_not_called()

    async def test_rebuild_main_does_not_open_or_delete_collection_when_preflight_fails(self):
        """The destructive step is unreachable until the preflight succeeds."""
        client_class = MagicMock()

        with (
            patch.object(rebuild_vectordb, "preflight", new=AsyncMock(return_value=False)),
            patch("chromadb.PersistentClient", client_class),
        ):
            exit_code = await rebuild_vectordb.main()

        self.assertEqual(exit_code, 1)
        client_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
