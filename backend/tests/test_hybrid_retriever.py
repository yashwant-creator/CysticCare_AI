import unittest
import math
from unittest.mock import MagicMock, patch
from app.services.hybrid_retriever import HybridRetriever

class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        # Reset the Singleton instance for isolation between tests
        self.retriever = HybridRetriever()

        # Restore any class methods on the singleton that might have been mocked in previous tests
        if hasattr(self.retriever, 'sparse_search') and isinstance(self.retriever.sparse_search, MagicMock):
            self.retriever.sparse_search = HybridRetriever.sparse_search.__get__(self.retriever, HybridRetriever)

        self.retriever.initialized = False
        self.retriever.documents = []
        self.retriever.metadatas = []
        self.retriever.ids = []
        self.retriever.corpus_size = 0
        self.retriever.avg_doc_len = 0.0
        self.retriever.doc_lengths = []
        self.retriever.doc_freqs = {}
        self.retriever.term_freqs = []
        self.retriever.collection = None

    def test_singleton_behavior(self):
        """Verify that HybridRetriever is indeed a singleton."""
        instance1 = HybridRetriever()
        instance2 = HybridRetriever()
        self.assertIs(instance1, instance2)

    def test_tokenization_edge_cases(self):
        """Verify the alphanumeric word/number tokenization rules."""
        # Simple lowercase conversion & punctuation stripping
        tokens = self.retriever._tokenize("Hello, World! ADPKD PKD1.")
        self.assertEqual(tokens, ["hello", "world", "adpkd", "pkd1"])

        # Multiple contiguous whitespaces & special characters
        tokens = self.retriever._tokenize("   PKD2  ---  and   tolvaptan  123   ")
        self.assertEqual(tokens, ["pkd2", "and", "tolvaptan", "123"])

        # Empty/None checks
        self.assertEqual(self.retriever._tokenize(""), [])
        self.assertEqual(self.retriever._tokenize(None), [])

    def test_idf_smoothing_and_positivity(self):
        """Verify that IDF score calculations remain positive and correctly smoothed."""
        self.retriever.corpus_size = 100

        # Term appearing in 0 docs (smooth to prevent zero division)
        self.retriever.doc_freqs = {}
        idf_0 = self.retriever._idf("unknown_word")
        self.assertGreater(idf_0, 0.0)

        # Term appearing in all docs (ensure no negative scores are returned)
        self.retriever.doc_freqs = {"common_word": 100}
        idf_100 = self.retriever._idf("common_word")
        self.assertGreaterEqual(idf_100, 0.0001)

        # Normal occurrence
        self.retriever.doc_freqs = {"pkd1": 10}
        idf_10 = self.retriever._idf("pkd1")
        # Formula: math.log((100 - 10 + 0.5) / (10 + 0.5) + 1.0)
        expected_idf = math.log((90.5 / 10.5) + 1.0)
        self.assertAlmostEqual(idf_10, expected_idf, places=4)

    def test_sparse_bm25_search(self):
        """Verify sparse keyword scoring rankings."""
        self.retriever.documents = [
            "PKD1 is a gene on chromosome 16 that encodes polycystin-1.",
            "Tolvaptan is a selective vasopressin V2 receptor antagonist used to treat ADPKD.",
            "ADPKD can be diagnosed using genetics and kidney imaging methods.",
            "This document is completely unrelated to kidney diseases."
        ]
        self.retriever.metadatas = [
            {"citation": "Xu (2023)", "id": "doc1"},
            {"citation": "Hammond (2024)", "id": "doc2"},
            {"citation": "Harris (2022)", "id": "doc3"},
            {"citation": "Other (2020)", "id": "doc4"}
        ]
        self.retriever.ids = ["doc1", "doc2", "doc3", "doc4"]
        self.retriever.corpus_size = 4

        # Build index parameters manually
        self.retriever.doc_lengths = []
        self.retriever.term_freqs = []
        self.retriever.doc_freqs = {}
        total_len = 0
        for doc in self.retriever.documents:
            tokens = self.retriever._tokenize(doc)
            self.retriever.doc_lengths.append(len(tokens))
            total_len += len(tokens)

            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.retriever.term_freqs.append(tf)

            for t in tf.keys():
                self.retriever.doc_freqs[t] = self.retriever.doc_freqs.get(t, 0) + 1

        self.retriever.avg_doc_len = total_len / 4
        self.retriever.initialized = True

        # Test query matching 'PKD1' - should rank doc1 at the top
        results = self.retriever.sparse_search("PKD1", top_k=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc1")
        self.assertEqual(results[0]["metadata"]["citation"], "Xu (2023)")

        # Test query matching 'ADPKD' - doc2 and doc3 contain it, doc1/doc4 do not.
        results_adpkd = self.retriever.sparse_search("ADPKD", top_k=3)
        self.assertEqual(len(results_adpkd), 2)
        matched_ids = [r["id"] for r in results_adpkd]
        self.assertIn("doc2", matched_ids)
        self.assertIn("doc3", matched_ids)

        # Test query matching none - empty results
        results_empty = self.retriever.sparse_search("unrelated_query", top_k=2)
        self.assertEqual(results_empty, [])

    def test_reciprocal_rank_fusion_logic(self):
        """Test hybrid RRF combination, sorting, and score normalization."""
        mock_collection = MagicMock()

        # Stub dense search results (collection.query)
        mock_collection.query.return_value = {
            "documents": [["DocA", "DocB"]],
            "metadatas": [[{"citation": "CiteA"}, {"citation": "DocB"}]],
            "ids": [["idA", "idB"]],
            "distances": [[0.3, 0.4]]  # relevance scores will be 1 - 0.3 = 0.7, and 1 - 0.4 = 0.6
        }

        self.retriever.collection = mock_collection
        self.retriever.initialized = True
        self.retriever.corpus_size = 2
        self.retriever.documents = ["DocA", "DocB"]
        self.retriever.metadatas = [{"citation": "CiteA"}, {"citation": "DocB"}]
        self.retriever.ids = ["idA", "idB"]

        # Mock the sparse_search to return a different rank order: DocB, then DocA
        self.retriever.sparse_search = MagicMock(return_value=[
            {"document": "DocB", "metadata": {"citation": "CiteB"}, "id": "idB", "sparse_score": 5.0},
            {"document": "DocA", "metadata": {"citation": "CiteA"}, "id": "idA", "sparse_score": 1.0}
        ])

        # Execute hybrid search
        results = self.retriever.hybrid_search(
            query="test query",
            query_embedding=[0.1, 0.2],
            top_k=2,
            rrf_k=60
        )

        # Verify results and scores
        self.assertEqual(len(results), 2)

        # Dense ranks: idA: 1st (rank 0), idB: 2nd (rank 1)
        # Sparse ranks: idB: 1st (rank 0), idA: 2nd (rank 1)
        # RRF Score formula:
        # idA = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934 + 0.016129 = 0.032522
        # idB = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.0163934 = 0.032522
        # Since scores are identical, sorting is stable. Let's verify scores are calculated accurately.
        max_possible_rrf = 2.0 / (60 + 1)  # ≈ 0.032786
        expected_norm_score = round(0.032522 / max_possible_rrf, 4)  # ≈ 0.992

        for item in results:
            self.assertAlmostEqual(item["relevance_score"], expected_norm_score, delta=0.01)
            self.assertIn("dense_score", item)
            self.assertIn("sparse_score", item)
            self.assertIn("rrf_score", item)

    def test_uninitialized_sparse_fallback(self):
        """Verify that when sparse retriever is uninitialized, we fall back to pure dense."""
        mock_collection = MagicMock()
        mock_collection.get.side_effect = Exception("ChromaDB Error")

        # Instantiating retriever with failing collection
        retriever = HybridRetriever(mock_collection)
        self.assertFalse(retriever.initialized)

        # Try building index - should handle exception gracefully and remain uninitialized
        retriever.build_sparse_index()
        self.assertFalse(retriever.initialized)

if __name__ == "__main__":
    unittest.main()
