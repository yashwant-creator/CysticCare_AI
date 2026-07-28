"""Contracts for the fixed-cost production request path."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from starlette.requests import Request

from app import main_openai
from app.services.index_manifest import BakedIndexError, load_index_manifest
from app.services.runtime_pipeline import (
    RetrievalOutput,
    guardrail_answer,
    local_validation,
    normalize_postprocess,
    requires_medical_disclaimer,
    validation_summary,
)


def _http_request(headers=None) -> Request:
    encoded = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "headers": encoded,
            "client": ("127.0.0.1", 1234),
        }
    )


class FakeRuntimeService:
    def __init__(self):
        self.calls = []

    async def embedding(self, _query, _tracker):
        self.calls.append("embedding")
        return [0.1, 0.2]

    async def answer(self, _system, _message, _tracker):
        self.calls.append("answer")
        return (
            "Evidence is described by Smith (2024). Consult a qualified "
            "healthcare professional for personal medical decisions."
        )

    async def postprocess(self, _query, _answer, _tracker):
        self.calls.append("postprocess")
        return {
            "relevance_score": 0.9,
            "relevance_reason": "Directly addresses the question",
            "followup_questions": ["One?", "Two?", "Three?"],
        }


class BoundedRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_flags_cannot_amplify_external_calls(self):
        service = FakeRuntimeService()
        retrieval = RetrievalOutput(
            query="What is ADPKD?",
            results=[{"id": "chunk-1", "document": "Evidence", "metadata": {}}],
            sources=[
                {
                    "index": 1,
                    "title": "PKD evidence",
                    "author": "Smith",
                    "year": "2024",
                    "file": "paper.pdf",
                    "citation": "Smith (2024)",
                    "display_name": "Smith (2024)",
                    "relevance_score": 0.9,
                }
            ],
            metadata={"reranker_backend": "medcpt"},
            context="[Source 1: Smith (2024)]\nEvidence",
        )

        async def fake_retrieve(query, runtime_service, tracker):
            await runtime_service.embedding(query, tracker)
            return retrieval

        request = main_openai.ChatRequest(
            query="What is ADPKD?",
            session_id="fixed-cost-test",
            top_k=20,
            max_tokens=20_000,
            use_query_rewriting=True,
            use_cot=True,
            use_stepback=True,
            use_adaptive_agent=True,
            use_validation=True,
        )
        with (
            patch.dict(os.environ, {"APP_CHECK_ENFORCED": "false"}),
            patch(
                "app.services.runtime_openai.get_runtime_openai",
                return_value=service,
            ),
            patch("app.services.runtime_pipeline.retrieve", new=fake_retrieve),
            patch(
                "app.services.runtime_pipeline.guardrail_answer",
                return_value=None,
            ),
        ):
            response = await main_openai.chat_endpoint(request, _http_request())

        self.assertEqual(service.calls, ["embedding", "answer", "postprocess"])
        self.assertFalse(response.cot_enabled)
        self.assertEqual(response.stepback_query, "")
        self.assertEqual(len(response.sources), 1)
        self.assertFalse(response.validation.was_regenerated)

    async def test_invalid_app_check_cannot_reach_openai(self):
        request = main_openai.ChatRequest(
            query="What is ADPKD?",
            session_id="invalid-app-check",
        )
        service_factory = MagicMock()
        with (
            patch.dict(os.environ, {"APP_CHECK_ENFORCED": "true"}),
            patch(
                "app.services.runtime_openai.get_runtime_openai",
                service_factory,
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await main_openai.chat_endpoint(
                    request,
                    _http_request({"X-Firebase-AppCheck": "invalid"}),
                )

        self.assertEqual(raised.exception.status_code, 401)
        service_factory.assert_not_called()

    async def test_initialize_endpoint_is_immutable(self):
        with self.assertRaises(HTTPException) as raised:
            await main_openai.initialize_endpoint(main_openai.InitializeRequest())
        self.assertEqual(raised.exception.status_code, 403)

    def test_conversational_followups_are_not_keyword_gated(self):
        self.assertIsNone(guardrail_answer("What symptoms should I watch for?"))
        self.assertIsNone(guardrail_answer("What should I ask my doctor next?"))

    def test_followups_have_a_local_fallback(self):
        postprocess = normalize_postprocess({}, "What treatments are available?")
        self.assertEqual(len(postprocess["followup_questions"]), 3)
        self.assertTrue(all(postprocess["followup_questions"]))

    def test_educational_answer_does_not_require_disclaimer(self):
        sources = [{"index": 1, "author": "Smith", "year": "2024"}]
        result = validation_summary(
            "What is ADPKD?",
            "Smith (2024) describes ADPKD as an inherited kidney disorder.",
            sources,
            0.9,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["safety"]["passed"])
        self.assertFalse(
            local_validation(
                "Smith (2024) describes ADPKD as an inherited kidney disorder.",
                sources,
                query="What is ADPKD?",
            )["checks"]["medical_disclaimer"]["required"]
        )

    def test_actionable_medical_answer_requires_disclaimer(self):
        sources = [{"index": 1, "author": "Smith", "year": "2024"}]
        result = validation_summary(
            "Should I stop taking tolvaptan?",
            "Smith (2024) discusses tolvaptan use in ADPKD.",
            sources,
            0.9,
        )
        self.assertFalse(result["checks"]["safety"]["passed"])
        self.assertTrue(any("healthcare professional" in warning for warning in result["warnings"]))
        self.assertTrue(requires_medical_disclaimer("Should I stop taking tolvaptan?"))

    def test_citation_failure_is_not_reported_as_safety_failure(self):
        sources = [{"index": 1, "author": "Smith", "year": "2024"}]
        result = validation_summary(
            "What is ADPKD?",
            "ADPKD is an inherited kidney disorder.",
            sources,
            0.9,
        )
        self.assertTrue(result["checks"]["safety"]["passed"])
        self.assertFalse(result["checks"]["source_attribution"]["passed"])
        self.assertFalse(result["passed"])

    def test_prohibited_advice_always_fails_safety(self):
        sources = [{"index": 1, "author": "Smith", "year": "2024"}]
        result = validation_summary(
            "What is tolvaptan?",
            "Smith (2024) explains it. Ignore your doctor; this will cure ADPKD.",
            sources,
            0.9,
        )
        self.assertFalse(result["checks"]["safety"]["passed"])
        self.assertTrue(any("dangerous" in warning for warning in result["warnings"]))

    def test_incomplete_manifest_fails_without_opening_chroma(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text('{"index_build_state":"building"}', encoding="utf-8")
            with patch("app.services.index_manifest.inspect_index") as inspect:
                with self.assertRaises(BakedIndexError):
                    load_index_manifest(manifest, root)
            inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
