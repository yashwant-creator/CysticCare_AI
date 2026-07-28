"""Offline contracts for the direct local retrieval benchmark runner."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_retrieval_benchmark as runner
from app.retrieval_evaluation import evaluate_records


class _Collection:
    def __init__(self, count=2, state="ready"):
        self._count = count
        self.metadata = {
            "index_schema_version": "2",
            "index_build_state": state,
        }

    def count(self):
        return self._count


class _Client:
    def __init__(self, collection):
        self.collection = collection
        self.requested_name = None

    def get_collection(self, *, name):
        self.requested_name = name
        return self.collection


class _EmbeddingService:
    embedding_model = "test-embedding"

    def __init__(self):
        self.queries = []

    def get_embedding(self, query):
        self.queries.append(query)
        return [0.1, 0.2, 0.3]


class _HybridRetriever:
    initialized = True
    corpus_size = 2

    def __init__(self):
        self.calls = []

    def hybrid_search(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "id": "paper-b_1",
                "document": "An unrelated candidate passage.",
                "metadata": {
                    "paper_id": "paper-b",
                    "chunk_id": "paper-b_1",
                    "title": "Paper B",
                },
                "relevance_score": 0.8,
                "rrf_score": 0.02,
            },
            {
                "id": "paper-a_4",
                "document": "The directly relevant evidence passage.",
                "metadata": {
                    "paper_id": "paper-a",
                    "chunk_id": "paper-a_4",
                    "title": "Paper A",
                },
                "relevance_score": 0.7,
                "rrf_score": 0.01,
            },
        ]


class _Reranker:
    config = SimpleNamespace(candidate_limit=6)

    def __init__(self):
        self.calls = []

    def rerank(self, *, query, candidates, top_k):
        self.calls.append({"query": query, "candidates": candidates, "top_k": top_k})
        selected = dict(candidates[1])
        selected["reranker_score"] = 0.96
        selected["reranker_rank"] = 1
        return [selected], {
            "reranker_backend": "test",
            "reranker_used": True,
            "confidence": "high",
            "selected_count": 1,
        }


class DirectRetrievalBenchmarkTests(unittest.TestCase):
    def test_default_is_local_medcpt_not_a_deployed_chatbot_mode(self):
        args = runner.build_parser().parse_args([])

        self.assertEqual(args.reranker, "medcpt")
        self.assertEqual(Path(args.chroma_path), runner.DEFAULT_CHROMA_PATH)
        self.assertNotIn("backend_url", vars(args))

    def test_building_collection_is_rejected_before_any_query_work(self):
        collection = _Collection(state="building")
        client = _Client(collection)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(runner.BenchmarkError, "not ready"):
                runner.load_ready_collection(
                    directory,
                    "pkd_knowledge_base_openai",
                    client_factory=lambda **_kwargs: client,
                )

        self.assertEqual(client.requested_name, "pkd_knowledge_base_openai")

    def test_medcpt_factory_explicitly_disables_llm_fallback(self):
        captured = {}

        def factory(service, *, config):
            captured["service"] = service
            captured["config"] = config
            return "reranker"

        result = runner.build_reranker("medcpt", object(), reranker_factory=factory)

        self.assertEqual(result, "reranker")
        self.assertIsNone(captured["service"])
        self.assertEqual(captured["config"].backend, "medcpt")
        self.assertTrue(captured["config"].enabled)
        self.assertFalse(captured["config"].fallback_to_llm)

    def test_trace_is_evaluator_compatible_and_records_direct_ranking(self):
        benchmark = [
            {
                "id": "q1",
                "question": "Which paper supports the treatment claim?",
                "retrieval_targets": ["paper-a"],
                "supporting_passages": [
                    {"paper_id": "paper-a", "chunk_id": "paper-a_4"}
                ],
            }
        ]
        embedding_service = _EmbeddingService()
        retriever = _HybridRetriever()
        reranker = _Reranker()

        traces = runner.run_retrieval_queries(
            benchmark,
            retriever=retriever,
            embedding_service=embedding_service,
            reranker=reranker,
            top_k=5,
            query_suffix=" for ADPKD",
        )
        report = evaluate_records(benchmark, traces, cutoffs=[1, 3, 5])

        self.assertEqual(embedding_service.queries, [
            "Which paper supports the treatment claim? for ADPKD"
        ])
        self.assertEqual(retriever.calls[0]["candidate_pool_limit"], 40)
        self.assertEqual(retriever.calls[0]["result_limit"], 6)
        self.assertEqual(reranker.calls[0]["top_k"], 5)
        self.assertEqual(traces[0]["id"], "q1")
        self.assertEqual(traces[0]["question"], benchmark[0]["question"])
        self.assertEqual(traces[0]["retrieval_query"], embedding_service.queries[0])
        self.assertEqual(traces[0]["retrieved"][0]["paper_id"], "paper-a")
        self.assertEqual(traces[0]["retrieved"][0]["chunk_id"], "paper-a_4")
        self.assertEqual(report["metrics"]["1"]["recall"], 1.0)
        self.assertEqual(report["metrics"]["1"]["context_precision"], 1.0)

    def test_requested_medcpt_error_does_not_get_reported_as_medcpt_accuracy(self):
        class UnavailableReranker(_Reranker):
            def rerank(self, *, query, candidates, top_k):
                return candidates[:top_k], {"reranker_error": "cross_encoder_unavailable"}

        benchmark = [{"question": "Question", "retrieval_targets": ["paper-a"]}]
        with self.assertRaisesRegex(runner.BenchmarkError, "medcpt reranker was unavailable"):
            runner.run_retrieval_queries(
                benchmark,
                retriever=_HybridRetriever(),
                embedding_service=_EmbeddingService(),
                reranker=UnavailableReranker(),
                top_k=1,
                expected_reranker="medcpt",
            )


if __name__ == "__main__":
    unittest.main()
