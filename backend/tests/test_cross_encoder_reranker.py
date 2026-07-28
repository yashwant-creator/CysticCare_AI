import unittest

from app.services.cross_encoder_reranker import CrossEncoderConfig, CrossEncoderReranker
from app.services.semantic_reranker import RerankerConfig, create_evidence_reranker


def candidate(candidate_id, paper_id, score=0.7):
    return {
        "id": candidate_id,
        "document": f"Evidence passage for {candidate_id}",
        "metadata": {
            "paper_id": paper_id,
            "title": f"Title for {paper_id}",
            "section_heading": "Results",
            "page_number": 3,
        },
        "relevance_score": score,
    }


class DeterministicCrossEncoder:
    def __init__(self, scores=None, error=None):
        self.scores = list(scores or [])
        self.error = error
        self.calls = []

    def score_pairs(self, query, passages):
        self.calls.append((query, list(passages)))
        if self.error:
            raise self.error
        return list(self.scores)


class FakeStructuredService:
    def __init__(self, ratings):
        self.ratings = ratings
        self.calls = []

    def get_structured_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"ratings": self.ratings}


class CrossEncoderRerankerTests(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        defaults = {
            "enabled": True,
            "candidate_limit": 8,
            "max_chunks_per_document": 2,
            "max_chars_per_candidate": 1_000,
            "model_name": "ncbi/MedCPT-Cross-Encoder",
            "max_length": 512,
            "batch_size": 8,
            "device": "cpu",
            "local_files_only": True,
            "min_logit": None,
            "min_logit_margin": 0.0,
        }
        defaults.update(overrides)
        return CrossEncoderConfig(**defaults)

    def test_ranks_raw_logits_and_diversifies_papers(self):
        scorer = DeterministicCrossEncoder([3.2, 2.7, 2.4, 1.1])
        reranker = CrossEncoderReranker(self.config(), scorer=scorer)
        results, metadata = reranker.rerank(
            "Does tolvaptan slow eGFR decline?",
            [
                candidate("a1", "trial-paper"),
                candidate("a2", "trial-paper"),
                candidate("a3", "trial-paper"),
                candidate("b1", "guideline-paper"),
            ],
            top_k=3,
        )

        self.assertEqual([result["id"] for result in results], ["a1", "a2", "b1"])
        self.assertEqual([result["reranker_raw_score"] for result in results], [3.2, 2.7, 1.1])
        self.assertGreater(results[0]["relevance_score"], 0.9)
        self.assertEqual(results[0]["fused_score"], 0.7)
        self.assertTrue(metadata["reranker_used"])
        self.assertEqual(metadata["reranker_backend"], "medcpt")
        self.assertEqual(metadata["score_type"], "cross_encoder_logit")
        self.assertEqual(metadata["threshold_type"], "disabled")
        self.assertEqual(metadata["confidence"], "ranked_unthresholded")
        self.assertTrue(metadata["accepted"])
        self.assertEqual(len(scorer.calls), 1)
        passage = scorer.calls[0][1][0]
        self.assertIn("Paper: Title for trial-paper", passage)
        self.assertIn("Section: Results", passage)
        self.assertIn("Page: 3", passage)

    def test_no_llm_probability_cutoff_is_applied_to_raw_logits(self):
        scorer = DeterministicCrossEncoder([-2.0, -3.0])
        reranker = CrossEncoderReranker(self.config(min_logit=None), scorer=scorer)

        results, metadata = reranker.rerank(
            "Question", [candidate("a1", "paper-a"), candidate("b1", "paper-b")], top_k=2
        )

        # The displayed sigmoid score is below 0.50, but no threshold is
        # applied until a MedCPT raw-logit cutoff has been benchmark-calibrated.
        self.assertEqual([result["id"] for result in results], ["a1", "b1"])
        self.assertLess(results[0]["relevance_score"], 0.5)
        self.assertTrue(metadata["accepted"])
        self.assertEqual(metadata["confidence"], "ranked_unthresholded")

    def test_calibrated_raw_logit_threshold_rejects_insufficient_evidence(self):
        scorer = DeterministicCrossEncoder([0.8, 0.4])
        reranker = CrossEncoderReranker(self.config(min_logit=1.0), scorer=scorer)

        results, metadata = reranker.rerank(
            "Question", [candidate("a1", "paper-a"), candidate("b1", "paper-b")], top_k=2
        )

        self.assertEqual(results, [])
        self.assertFalse(metadata["accepted"])
        self.assertEqual(metadata["confidence"], "insufficient")
        self.assertEqual(metadata["top_raw_score"], 0.8)
        self.assertEqual(metadata["threshold"], 1.0)
        self.assertEqual(metadata["threshold_type"], "raw_logit")

    def test_cross_encoder_failure_fails_open_to_fused_results(self):
        reranker = CrossEncoderReranker(
            self.config(),
            scorer=DeterministicCrossEncoder(error=RuntimeError("model unavailable")),
        )
        results, metadata = reranker.rerank(
            "Question", [candidate("a1", "paper-a"), candidate("b1", "paper-b")], top_k=2
        )

        self.assertEqual([result["id"] for result in results], ["a1", "b1"])
        self.assertFalse(metadata["reranker_used"])
        self.assertEqual(metadata["reranker_error"], "cross_encoder_unavailable")
        self.assertEqual(metadata["confidence"], "fallback")

    def test_factory_falls_back_to_llm_when_cross_encoder_cannot_load(self):
        service = FakeStructuredService(
            [{"id": "a1", "score": 0.91}, {"id": "b1", "score": 0.14}]
        )
        config = RerankerConfig(
            enabled=True,
            candidate_limit=5,
            min_relevance_score=0.50,
            min_score_margin=0.05,
            max_chunks_per_document=2,
            max_chars_per_candidate=1_000,
            backend="medcpt",
            fallback_to_llm=True,
        )
        reranker = create_evidence_reranker(
            service,
            config=config,
            cross_encoder_scorer=DeterministicCrossEncoder(
                error=RuntimeError("model unavailable")
            ),
        )
        results, metadata = reranker.rerank(
            "Question", [candidate("a1", "paper-a"), candidate("b1", "paper-b")], top_k=2
        )

        self.assertEqual([result["id"] for result in results], ["a1"])
        self.assertTrue(metadata["reranker_used"])
        self.assertEqual(metadata["score_type"], "semantic")
        self.assertEqual(metadata["requested_reranker_backend"], "medcpt")
        self.assertEqual(metadata["fallback_reranker_backend"], "llm")
        self.assertEqual(metadata["primary_reranker_error"], "cross_encoder_unavailable")
        self.assertEqual(len(service.calls), 1)

    def test_factory_off_keeps_fused_results_without_loading_a_model(self):
        reranker = create_evidence_reranker(
            None,
            config=RerankerConfig(backend="off", enabled=True),
        )
        results, metadata = reranker.rerank(
            "Question", [candidate("a1", "paper-a"), candidate("b1", "paper-b")], top_k=2
        )

        self.assertEqual([result["id"] for result in results], ["a1", "b1"])
        self.assertFalse(metadata["reranker_used"])
        self.assertEqual(metadata["reranker_backend"], "off")
        self.assertEqual(metadata["score_type"], "fused_rank")


if __name__ == "__main__":
    unittest.main()
