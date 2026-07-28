import unittest

from app.services.semantic_reranker import RerankerConfig, SemanticReranker


class FakeStructuredService:
    def __init__(self, ratings=None, error=None):
        self.ratings = ratings or []
        self.error = error
        self.calls = []

    def get_structured_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"ratings": self.ratings}


def candidate(candidate_id, paper_id, score=0.7):
    return {
        "id": candidate_id,
        "document": f"Evidence passage for {candidate_id}",
        "metadata": {"paper_id": paper_id, "title": paper_id},
        "relevance_score": score,
    }


class SemanticRerankerTests(unittest.TestCase):
    def config(self, **overrides):
        defaults = {
            "enabled": True,
            "candidate_limit": 10,
            "min_relevance_score": 0.50,
            "min_score_margin": 0.05,
            "max_chunks_per_document": 2,
            "max_chars_per_candidate": 1000,
        }
        defaults.update(overrides)
        return RerankerConfig(**defaults)

    def test_uses_semantic_scores_and_diversifies_papers(self):
        service = FakeStructuredService([
            {"id": "a1", "score": 0.96},
            {"id": "a2", "score": 0.92},
            {"id": "a3", "score": 0.89},
            {"id": "b1", "score": 0.82},
        ])
        reranker = SemanticReranker(
            service,
            self.config(max_chunks_per_document=2),
        )

        results, metadata = reranker.rerank(
            "What evidence supports tolvaptan use?",
            [
                candidate("a1", "paper-a"),
                candidate("a2", "paper-a"),
                candidate("a3", "paper-a"),
                candidate("b1", "paper-b"),
            ],
            top_k=3,
        )

        self.assertTrue(metadata["reranker_used"])
        self.assertEqual(metadata["score_type"], "semantic")
        self.assertEqual([result["id"] for result in results], ["a1", "a2", "b1"])
        self.assertEqual(results[0]["relevance_score"], 0.96)
        self.assertEqual(results[0]["fused_score"], 0.7)
        self.assertEqual(len(service.calls), 1)
        ratings_schema = service.calls[0]["schema"]["properties"]["ratings"]
        self.assertEqual(ratings_schema["minItems"], 4)
        self.assertEqual(ratings_schema["maxItems"], 4)
        self.assertEqual(service.calls[0]["max_tokens"], 2000)

    def test_threshold_rejects_insufficient_evidence(self):
        service = FakeStructuredService([
            {"id": "a1", "score": 0.49},
            {"id": "b1", "score": 0.31},
        ])
        reranker = SemanticReranker(service, self.config())

        results, metadata = reranker.rerank(
            "Question", [candidate("a1", "paper-a"), candidate("b1", "paper-b")], top_k=2
        )

        self.assertEqual(results, [])
        self.assertFalse(metadata["accepted"])
        self.assertEqual(metadata["confidence"], "insufficient")
        self.assertEqual(metadata["top_score"], 0.49)

    def test_falls_back_to_fused_results_when_reranker_fails(self):
        reranker = SemanticReranker(
            FakeStructuredService(error=RuntimeError("temporary API failure")),
            self.config(),
        )

        results, metadata = reranker.rerank(
            "Question", [candidate("a1", "paper-a"), candidate("b1", "paper-b")], top_k=2
        )

        self.assertEqual([result["id"] for result in results], ["a1", "b1"])
        self.assertFalse(metadata["reranker_used"])
        self.assertEqual(metadata["confidence"], "fallback")

    def test_incomplete_ratings_fail_open_instead_of_zeroing_candidates(self):
        reranker = SemanticReranker(
            FakeStructuredService([{"id": "a1", "score": 0.95}]),
            self.config(),
        )

        results, metadata = reranker.rerank(
            "Question", [candidate("a1", "paper-a"), candidate("b1", "paper-b")], top_k=2
        )

        self.assertEqual([result["id"] for result in results], ["a1", "b1"])
        self.assertEqual(metadata["confidence"], "fallback")

    def test_disabled_reranker_never_applies_a_rank_score_threshold(self):
        reranker = SemanticReranker(None, self.config(enabled=False, min_relevance_score=0.99))

        results, metadata = reranker.rerank(
            "Question", [candidate("a1", "paper-a", score=0.05)], top_k=1
        )

        self.assertEqual([result["id"] for result in results], ["a1"])
        self.assertTrue(metadata["accepted"])
        self.assertEqual(metadata["score_type"], "fused_rank")

    def test_non_positive_top_k_never_selects_a_candidate(self):
        reranker = SemanticReranker(
            FakeStructuredService([{"id": "a1", "score": 0.95}]),
            self.config(),
        )

        results, metadata = reranker.rerank(
            "Question", [candidate("a1", "paper-a")], top_k=0
        )

        self.assertEqual(results, [])
        self.assertEqual(metadata["selected_count"], 0)


if __name__ == "__main__":
    unittest.main()
