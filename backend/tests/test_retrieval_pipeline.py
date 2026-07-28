import os
import unittest
from unittest.mock import patch

from app.services import openai_rag_init
from app.services.hybrid_retriever import HybridRetriever


class FakeCollection:
    def query(self, query_embeddings, n_results, include):
        return {
            "documents": [["Relevant evidence", "Unrelated background"]],
            "metadatas": [[
                {"paper_id": "paper-a", "title": "Paper A", "chunk_id": "a-1"},
                {"paper_id": "paper-b", "title": "Paper B", "chunk_id": "b-1"},
            ]],
            "ids": [["a-1", "b-1"]],
            "distances": [[0.10, 0.15]],
        }


class FakeRerankService:
    def get_embedding(self, query):
        return [0.1, 0.2]

    def get_structured_chat_completion(self, **kwargs):
        return {
            "ratings": [
                {"id": "a-1", "score": 0.91},
                {"id": "b-1", "score": 0.24},
            ]
        }


class FakeMedCPT:
    def __init__(self, config):
        self.config = config

    def score_pairs(self, query, passages):
        # Reverse the dense order so this verifies that the factory actually
        # selected MedCPT rather than the existing LLM judge.
        return [-2.0, 3.0]


class RetrievalPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {
                "RAG_RERANK_ENABLED": "true",
                "RAG_RERANK_CANDIDATES": "5",
                "RAG_MIN_RELEVANCE_SCORE": "0.50",
                "RAG_RERANKER_BACKEND": "llm",
            },
            clear=False,
        )
        self.environment.start()
        # Force the pure-dense path for this unit test; hybrid behavior has its
        # own tests and this isolates the semantic gate contract.
        self.old_instance = HybridRetriever._instance
        HybridRetriever._instance = None

    def tearDown(self):
        HybridRetriever._instance = self.old_instance
        self.environment.stop()

    async def test_search_returns_only_context_safe_semantic_results(self):
        service = FakeRerankService()
        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_helper_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            result = await openai_rag_init.search_knowledge_base("question", top_k=2)

        self.assertEqual(result["status"], "success")
        self.assertEqual([chunk["id"] for chunk in result["results"]], ["a-1"])
        self.assertEqual(result["results"][0]["relevance_score"], 0.91)
        self.assertEqual(result["retrieval_metadata"]["score_type"], "semantic")
        self.assertTrue(result["retrieval_metadata"]["accepted"])

    async def test_search_uses_medcpt_backend_when_configured(self):
        service = FakeRerankService()
        from app.services import cross_encoder_reranker

        with (
            patch.dict(
                os.environ,
                {
                    "RAG_RERANKER_BACKEND": "medcpt",
                    "RAG_CROSS_ENCODER_FALLBACK_TO_LLM": "false",
                    "RAG_CROSS_ENCODER_MIN_LOGIT": "",
                },
                clear=False,
            ),
            patch.object(cross_encoder_reranker, "MedCPTCrossEncoder", FakeMedCPT),
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_helper_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            result = await openai_rag_init.search_knowledge_base("question", top_k=2)

        self.assertEqual(result["status"], "success")
        self.assertEqual([chunk["id"] for chunk in result["results"]], ["b-1", "a-1"])
        self.assertTrue(result["retrieval_metadata"]["reranker_used"])
        self.assertEqual(result["retrieval_metadata"]["reranker_backend"], "medcpt")
        self.assertEqual(result["retrieval_metadata"]["score_type"], "cross_encoder_logit")
        self.assertEqual(result["retrieval_metadata"]["confidence"], "ranked_unthresholded")

    async def test_search_performs_parent_child_swapping_when_enabled(self):
        class FakeParentChildCollection:
            def query(self, query_embeddings, n_results, include):
                return {
                    "documents": [["Child text snippet"]],
                    "metadatas": [[{
                        "paper_id": "paper-a",
                        "title": "Paper A",
                        "chunk_id": "a-1",
                        "parent_text": "Detailed parent context covering previous, current and next chunks."
                    }]],
                    "ids": [["a-1"]],
                    "distances": [[0.10]],
                }

        service = FakeRerankService()

        # Test 1: Enabled (Default)
        with (
            patch.dict(os.environ, {"RAG_PARENT_CHILD_ENABLED": "true"}),
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_helper_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeParentChildCollection()),
        ):
            result = await openai_rag_init.search_knowledge_base("question", top_k=1)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"][0]["document"], "Detailed parent context covering previous, current and next chunks.")

        # Test 2: Disabled
        with (
            patch.dict(os.environ, {"RAG_PARENT_CHILD_ENABLED": "false"}),
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_helper_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeParentChildCollection()),
        ):
            result = await openai_rag_init.search_knowledge_base("question", top_k=1)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"][0]["document"], "Child text snippet")

    async def test_search_performs_hyde_retrieval_when_enabled(self):
        class FakeQueryRewriter:
            def __init__(self, service):
                pass
            async def generate_hyde_passage(self, query):
                return "A hypothetical paper passage answering the question."

        service = FakeRerankService()
        original_get_embedding = service.get_embedding
        embedded_texts = []

        def get_embedding_spy(text):
            embedded_texts.append(text)
            return original_get_embedding(text)

        service.get_embedding = get_embedding_spy

        from app.services import query_rewriter

        # Test 1: HyDE Enabled (Default)
        with (
            patch.dict(os.environ, {"RAG_HYDE_ENABLED": "true"}),
            patch.object(query_rewriter, "get_query_rewriter", FakeQueryRewriter),
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_helper_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            result = await openai_rag_init.search_knowledge_base("what is the best dose of tolvaptan?", top_k=1)

        self.assertEqual(result["status"], "success")
        self.assertIn("A hypothetical paper passage answering the question.", embedded_texts)

        # Test 2: HyDE Disabled
        embedded_texts.clear()
        with (
            patch.dict(os.environ, {"RAG_HYDE_ENABLED": "false"}),
            patch.object(query_rewriter, "get_query_rewriter", FakeQueryRewriter),
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_helper_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            result = await openai_rag_init.search_knowledge_base("what is the best dose of tolvaptan?", top_k=1)

        self.assertEqual(result["status"], "success")
        self.assertNotIn("A hypothetical paper passage answering the question.", embedded_texts)
        self.assertIn("what is the best dose of tolvaptan?", embedded_texts)


if __name__ == "__main__":
    unittest.main()
