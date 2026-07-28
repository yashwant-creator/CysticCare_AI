import csv
import math
import tempfile
import unittest
from pathlib import Path

from app.retrieval_evaluation import (
    evaluate_records,
    extract_gold_labels,
    extract_retrieved_items,
    load_records,
    parse_cutoffs,
)


def benchmark_row(record_id="q1"):
    return {
        "id": record_id,
        "question": "Which research papers answer this question?",
        "retrieval_targets": ["paper-a", "paper-b"],
        "source_papers": [
            {"paper_id": "paper-a", "paper_title": "Paper A"},
            {"paper_id": "paper-b", "paper_title": "Paper B"},
        ],
        "supporting_passages": [
            {"chunk_id": "paper-a_3", "paper_id": "paper-a", "paper_title": "Paper A"},
            {"chunk_id": "paper-b_8", "paper_id": "paper-b", "paper_title": "Paper B"},
        ],
    }


class TestRetrievalEvaluation(unittest.TestCase):
    def test_paper_metrics_use_rank_and_multiple_gold_targets(self):
        predictions = [
            {
                "id": "q1",
                "retrieved": [
                    {"paper_id": "not-relevant"},
                    {"paper_id": "paper-b"},
                    {"paper_id": "paper-a"},
                ],
            }
        ]

        report = evaluate_records([benchmark_row()], predictions, cutoffs=[1, 2, 3])
        metrics = report["metrics"]

        self.assertEqual(report["queries_evaluated"], 1)
        self.assertEqual(metrics["1"]["recall"], 0.0)
        self.assertEqual(metrics["1"]["mrr"], 0.0)
        self.assertEqual(metrics["1"]["context_precision"], 0.0)

        self.assertEqual(metrics["2"]["recall"], 0.5)
        self.assertEqual(metrics["2"]["mrr"], 0.5)
        self.assertEqual(metrics["2"]["context_precision"], 0.5)
        expected_ndcg_at_2 = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(metrics["2"]["ndcg"], expected_ndcg_at_2)

        self.assertEqual(metrics["3"]["recall"], 1.0)
        self.assertEqual(metrics["3"]["mrr"], 0.5)
        self.assertAlmostEqual(metrics["3"]["context_precision"], 2 / 3)

    def test_duplicate_paper_does_not_inflate_metrics(self):
        predictions = [
            {
                "id": "q1",
                "retrieved": [
                    {"paper_id": "paper-a"},
                    {"paper_id": "paper-a"},
                    {"paper_id": "paper-b"},
                ],
            }
        ]

        report = evaluate_records([benchmark_row()], predictions, cutoffs=[2])
        metric = report["metrics"]["2"]
        self.assertEqual(metric["recall"], 1.0)
        self.assertEqual(metric["context_precision"], 1.0)

    def test_historical_ids_for_one_title_are_one_paper_target(self):
        """A fresh index keeps one canonical ID when the PDF was duplicated."""
        benchmark = {
            "id": "duplicates",
            "question": "Duplicate historical identifiers",
            "retrieval_targets": ["old-a", "old-b"],
            "source_papers": [
                {"paper_id": "old-a", "paper_title": "Same Paper"},
                {"paper_id": "old-b", "paper_title": "Same Paper"},
            ],
        }
        prediction = {"id": "duplicates", "retrieved": [{"paper_id": "old-a"}]}

        report = evaluate_records([benchmark], [prediction], cutoffs=[1])
        metric = report["metrics"]["1"]

        self.assertEqual(metric["recall"], 1.0)
        self.assertEqual(metric["context_precision"], 1.0)

    def test_canonical_reindexed_id_matches_historic_target_via_title_metadata(self):
        benchmark = {
            "id": "canonical-alias",
            "retrieval_targets": ["historic-id"],
            "source_papers": [
                {"paper_id": "historic-id", "paper_title": "A Landmark PKD Paper"},
            ],
        }
        prediction = {
            "id": "canonical-alias",
            "retrieved": [
                {
                    "paper_id": "canonical-reindexed-id",
                    "metadata": {"title": "A Landmark PKD Paper"},
                }
            ],
        }

        report = evaluate_records([benchmark], [prediction], cutoffs=[1])
        self.assertEqual(report["metrics"]["1"]["recall"], 1.0)

    def test_legacy_file_title_fallback_matches_current_metadata_shape(self):
        benchmark = benchmark_row()
        benchmark["retrieval_targets"] = ["paper-a"]
        benchmark["source_papers"] = [
            {"paper_id": "paper-a", "paper_title": "A Landmark PKD Paper"}
        ]
        benchmark["supporting_passages"] = []
        prediction = {
            "id": "q1",
            "cysticcare_metadata": {
                "sources": [
                    {
                        "file": "A Landmark PKD Paper",
                        "display_name": "A Landmark PKD Paper",
                    }
                ]
            },
        }

        report = evaluate_records([benchmark], [prediction], cutoffs=[1], label_level="paper")
        self.assertEqual(report["metrics"]["1"]["recall"], 1.0)
        self.assertEqual(report["metrics"]["1"]["mrr"], 1.0)

    def test_chunk_level_requires_exact_supporting_chunk(self):
        prediction = {
            "id": "q1",
            "retrieved_chunks": [{"paper_id": "paper-a", "chunk_id": "paper-a_99"}],
        }

        paper_report = evaluate_records([benchmark_row()], [prediction], cutoffs=[1], label_level="paper")
        chunk_report = evaluate_records([benchmark_row()], [prediction], cutoffs=[1], label_level="chunk")

        self.assertEqual(paper_report["metrics"]["1"]["recall"], 0.5)
        self.assertEqual(chunk_report["metrics"]["1"]["recall"], 0.0)
        self.assertEqual(chunk_report["metrics"]["1"]["context_precision"], 0.0)

    def test_nested_retrieved_chunks_take_priority_over_display_sources(self):
        prediction = {
            "id": "q1",
            "cysticcare_metadata": {
                "retrieved_chunks": [{"paper_id": "paper-a", "chunk_id": "paper-a_3"}],
                "sources": [{"paper_id": "not-relevant"}],
            },
        }

        report = evaluate_records([benchmark_row()], [prediction], cutoffs=[1], label_level="chunk")

        self.assertEqual(report["metrics"]["1"]["recall"], 0.5)

    def test_missing_prediction_is_counted_as_a_zero_score(self):
        first = benchmark_row("q1")
        second = benchmark_row("q2")
        predictions = [{"id": "q1", "retrieved": [{"paper_id": "paper-a"}]}]

        report = evaluate_records([first, second], predictions, cutoffs=[1])

        self.assertEqual(report["queries_evaluated"], 2)
        self.assertEqual(report["prediction_diagnostics"]["missing_prediction"], 1)
        # First record gets 1/2 recall; missing output gets 0, so mean is 1/4.
        self.assertEqual(report["metrics"]["1"]["recall"], 0.25)

    def test_legacy_string_labels_and_sources_are_supported(self):
        benchmark = {
            "id": "legacy-q",
            "question": "Legacy question",
            "source_papers": "A Legacy Paper — Researcher One",
            "supporting_passages": "[P1] A Legacy Paper (legacy-paper_7)\nEvidence text.",
        }
        prediction = {
            "id": "legacy-q",
            "cysticcare_sources": "1. A Legacy Paper (rel=0.9841)",
        }

        labels = extract_gold_labels(benchmark)
        items = extract_retrieved_items(prediction)
        self.assertIn("legacy-paper", labels.paper_grades)
        self.assertEqual(items[0].title_aliases[0], "alegacypaper")

        report = evaluate_records([benchmark], [prediction], cutoffs=[1])
        self.assertEqual(report["metrics"]["1"]["recall"], 1.0)

    def test_graded_label_map_is_used_for_ndcg(self):
        benchmark = {
            "id": "graded-q",
            "question": "Graded question",
            "retrieval_targets": {"high-value": 3, "lower-value": 1},
        }
        prediction = {
            "id": "graded-q",
            "retrieved": [{"paper_id": "lower-value"}, {"paper_id": "high-value"}],
        }

        report = evaluate_records([benchmark], [prediction], cutoffs=[2])

        # A high-relevance paper at rank two should not receive a perfect nDCG.
        self.assertGreater(report["metrics"]["2"]["ndcg"], 0.0)
        self.assertLess(report["metrics"]["2"]["ndcg"], 1.0)

    def test_csv_loader_decodes_serialized_retrieval_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "predictions.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "retrieved"])
                writer.writeheader()
                writer.writerow({"id": "q1", "retrieved": '[{"paper_id":"paper-a"}]'})

            rows = load_records(csv_path)

        self.assertEqual(rows, [{"id": "q1", "retrieved": [{"paper_id": "paper-a"}]}])

    def test_parse_cutoffs_rejects_invalid_values(self):
        self.assertEqual(parse_cutoffs("10,1,5,5"), (1, 5, 10))
        with self.assertRaises(ValueError):
            parse_cutoffs("0,5")


if __name__ == "__main__":
    unittest.main()
