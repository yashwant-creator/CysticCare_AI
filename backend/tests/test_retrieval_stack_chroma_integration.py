"""Deterministic integration coverage for the Chroma -> hybrid -> rerank path.

This test deliberately uses explicit vectors and a local temporary Chroma
database.  It therefore exercises the real Chroma query/get APIs and the
BM25 index built from stored chunks without requiring an embedding endpoint or
an OpenAI request.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import chromadb
from chromadb.config import Settings

from app.services.hybrid_retriever import HybridRetriever
from app.services.semantic_reranker import RerankerConfig, SemanticReranker


class DeterministicJudge:
    """A complete, inspectable semantic-judge substitute with no network I/O."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def get_structured_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "ratings": [
                {"id": candidate_id, "score": score}
                for candidate_id, score in self.scores.items()
            ]
        }


class RetrievalStackChromaIntegrationTests(unittest.TestCase):
    """Exercise real local storage, hybrid fusion, and semantic safeguards."""

    _TARGET_IDS = {
        "noise-lexical",
        "tolvaptan-efficacy-1",
        "tolvaptan-efficacy-2",
        "tolvaptan-safety",
        "guideline-support",
    }

    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.client = chromadb.PersistentClient(
            path=self.temp_directory.name,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="retrieval_stack_integration",
            metadata={"hnsw:space": "cosine"},
        )
        self._seed_real_chroma_collection()

        # HybridRetriever is intentionally a singleton in production. Reset it
        # here so this test uses only the temporary collection, not state from a
        # prior unit/integration test.
        HybridRetriever._instance = None
        self.retriever = HybridRetriever(self.collection)

    def tearDown(self):
        HybridRetriever._instance = None
        try:
            self.client.delete_collection("retrieval_stack_integration")
        finally:
            self.temp_directory.cleanup()

    def _seed_real_chroma_collection(self):
        """Store explicit vectors, including enough rows for a 30-result pool."""
        records = [
            {
                "id": "noise-lexical",
                "document": (
                    "Tolvaptan tolvaptan tolvaptan slows eGFR decline: registry code."
                ),
                "metadata": {
                    "paper_id": "administrative-paper",
                    "title": "Administrative treatment registry",
                    "section_heading": "Registry fields",
                    "page_number": 2,
                },
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "id": "tolvaptan-efficacy-1",
                "document": (
                    "In adults with ADPKD, tolvaptan slows eGFR decline and supports "
                    "disease-modifying treatment."
                ),
                "metadata": {
                    "paper_id": "trial-paper",
                    "title": "Tolvaptan efficacy trial",
                    "section_heading": "Primary outcome",
                    "page_number": 7,
                },
                "embedding": [0.40, 0.916515, 0.0],
            },
            {
                "id": "tolvaptan-efficacy-2",
                "document": (
                    "The trial found that tolvaptan slows eGFR decline versus control "
                    "during long-term follow-up."
                ),
                "metadata": {
                    "paper_id": "trial-paper",
                    "title": "Tolvaptan efficacy trial",
                    "section_heading": "Long-term follow-up",
                    "page_number": 10,
                },
                "embedding": [0.35, 0.936750, 0.0],
            },
            {
                "id": "tolvaptan-safety",
                "document": (
                    "Tolvaptan treatment requires monitoring for aquaresis and liver "
                    "enzyme abnormalities."
                ),
                "metadata": {
                    "paper_id": "trial-paper",
                    "title": "Tolvaptan efficacy trial",
                    "section_heading": "Safety",
                    "page_number": 13,
                },
                "embedding": [0.30, 0.953939, 0.0],
            },
            {
                "id": "guideline-support",
                "document": (
                    "ADPKD guidance describes when to initiate tolvaptan for patients "
                    "at risk of progressive eGFR decline."
                ),
                "metadata": {
                    "paper_id": "guideline-paper",
                    "title": "ADPKD treatment guideline",
                    "section_heading": "Treatment recommendations",
                    "page_number": 5,
                },
                "embedding": [0.20, 0.979796, 0.0],
            },
        ]

        # HybridRetriever requests at least 30 dense candidates. Add irrelevant
        # stored chunks so the real Chroma call is fully satisfied without an
        # undersized-result warning, while keeping target chunks well ahead of
        # them in cosine space and BM25.
        for index in range(25):
            records.append(
                {
                    "id": f"irrelevant-{index:02d}",
                    "document": f"Unrelated radiology archive note {index} about scanner maintenance.",
                    "metadata": {
                        "paper_id": f"irrelevant-paper-{index:02d}",
                        "title": "Unrelated archive note",
                        "section_heading": "Maintenance",
                        "page_number": 1,
                    },
                    "embedding": [-1.0, 0.0, 0.0],
                }
            )

        self.collection.add(
            ids=[record["id"] for record in records],
            documents=[record["document"] for record in records],
            metadatas=[record["metadata"] for record in records],
            embeddings=[record["embedding"] for record in records],
        )
        self.assertEqual(self.collection.count(), 30)

    def _hybrid_candidates(self):
        # RRF k=1 intentionally makes the synthetic dense+lexical distractor
        # rank first, proving that the semantic stage can correct fusion order.
        with patch.dict(os.environ, {"RAG_HYBRID_CANDIDATE_POOL": "30"}):
            return self.retriever.hybrid_search(
                query="tolvaptan slows egfr decline",
                query_embedding=[1.0, 0.0, 0.0],
                top_k=3,
                result_limit=5,
                candidate_pool_limit=30,
                rrf_k=1,
            )

    @staticmethod
    def _reranker_config(**overrides):
        options = {
            "enabled": True,
            "candidate_limit": 5,
            "min_relevance_score": 0.50,
            "min_score_margin": 0.05,
            "max_chunks_per_document": 2,
            "max_chars_per_candidate": 1_000,
            "max_completion_tokens": 400,
        }
        options.update(overrides)
        return RerankerConfig(**options)

    def test_real_chroma_candidates_are_semantically_reranked_and_diversified(self):
        candidates = self._hybrid_candidates()

        # The collection was actually queried and its stored content became the
        # BM25 corpus. The deliberately superficial record wins raw fusion.
        self.assertTrue(self.retriever.initialized)
        self.assertEqual(self.retriever.corpus_size, 30)
        self.assertEqual({item["id"] for item in candidates}, self._TARGET_IDS)
        self.assertEqual(candidates[0]["id"], "noise-lexical")
        self.assertGreater(candidates[0]["sparse_score"], 0.0)
        self.assertGreater(candidates[0]["dense_score"], 0.99)

        judge = DeterministicJudge(
            {
                "noise-lexical": 0.26,
                "tolvaptan-efficacy-1": 0.98,
                "tolvaptan-efficacy-2": 0.90,
                "tolvaptan-safety": 0.88,
                "guideline-support": 0.82,
            }
        )
        reranker = SemanticReranker(judge, self._reranker_config())

        results, metadata = reranker.rerank(
            "Does tolvaptan slow eGFR decline?", candidates, top_k=3
        )

        # Semantic evidence outranks fused rank. The third trial passage is
        # skipped because max_chunks_per_document=2, letting independent
        # guideline evidence into the answer context.
        self.assertEqual(
            [item["id"] for item in results],
            ["tolvaptan-efficacy-1", "tolvaptan-efficacy-2", "guideline-support"],
        )
        self.assertEqual(
            [item["metadata"]["paper_id"] for item in results],
            ["trial-paper", "trial-paper", "guideline-paper"],
        )
        self.assertEqual([item["relevance_score"] for item in results], [0.98, 0.90, 0.82])
        self.assertEqual(results[0]["fused_score"], candidates[1]["relevance_score"])
        self.assertTrue(metadata["reranker_used"])
        self.assertTrue(metadata["accepted"])
        self.assertEqual(metadata["score_type"], "semantic")
        self.assertEqual(metadata["confidence"], "high")
        self.assertEqual(metadata["selected_count"], 3)

        self.assertEqual(len(judge.calls), 1)
        rating_schema = judge.calls[0]["schema"]["properties"]["ratings"]
        self.assertEqual(rating_schema["minItems"], 5)
        self.assertEqual(rating_schema["maxItems"], 5)

    def test_semantic_threshold_rejects_high_rrf_candidates_from_real_chroma(self):
        candidates = self._hybrid_candidates()
        self.assertGreater(max(item["relevance_score"] for item in candidates), 0.50)

        judge = DeterministicJudge(
            {
                "noise-lexical": 0.49,
                "tolvaptan-efficacy-1": 0.44,
                "tolvaptan-efficacy-2": 0.39,
                "tolvaptan-safety": 0.31,
                "guideline-support": 0.28,
            }
        )
        reranker = SemanticReranker(judge, self._reranker_config())

        results, metadata = reranker.rerank(
            "Does tolvaptan slow eGFR decline?", candidates, top_k=3
        )

        # High normalized RRF is a rank signal, never a confidence bypass.
        self.assertEqual(results, [])
        self.assertTrue(metadata["reranker_used"])
        self.assertFalse(metadata["accepted"])
        self.assertEqual(metadata["score_type"], "semantic")
        self.assertEqual(metadata["confidence"], "insufficient")
        self.assertEqual(metadata["top_score"], 0.49)
        self.assertEqual(metadata["selected_count"], 0)


if __name__ == "__main__":
    unittest.main()
