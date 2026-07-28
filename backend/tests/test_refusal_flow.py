import json
import sys
import types
import unittest
from unittest.mock import patch

from app import main_openai
from app.services import (
    cot_rag_service,
    followup_agent,
    openai_rag_init,
    query_rewriter,
    stepback_agent,
    validation_agent,
)
from app.utils.refusal_utils import (
    INTRO_MESSAGE,
    REFUSAL_MESSAGE,
    is_intro_query,
    is_query_in_scope,
    is_refusal_response,
    normalize_refusal_response,
)


def configure_service_aliases():
    services_pkg = sys.modules.get("services")
    if services_pkg is None:
        services_pkg = types.ModuleType("services")
        sys.modules["services"] = services_pkg

    module_map = {
        "openai_rag_init": openai_rag_init,
        "query_rewriter": query_rewriter,
        "followup_agent": followup_agent,
        "cot_rag_service": cot_rag_service,
        "stepback_agent": stepback_agent,
        "validation_agent": validation_agent,

    }

    for name, module in module_map.items():
        setattr(services_pkg, name, module)
        sys.modules[f"services.{name}"] = module


configure_service_aliases()


class FakeCollection:
    def query(self, query_embeddings, n_results, include):
        return {
            "documents": [[
                "ADPKD is commonly monitored with kidney function testing and imaging."
            ]],
            "metadatas": [[{
                "title": "ADPKD Review",
                "author": "Torres",
                "year": "2024",
                "file_name": "adpkd_review.pdf",
                "citation": "Torres (2024)",
                "display_name": "ADPKD Review",
                "file": "adpkd_review.pdf",
            }]],
            "distances": [[0.05]],
        }


class FakeOpenAIService:
    def __init__(self, chat_responses=None, context_responses=None):
        self.chat_responses = list(chat_responses or [])
        self.context_responses = list(context_responses or [])
        self.chat_calls = 0
        self.context_calls = 0

    def get_embedding(self, text):
        return [0.1, 0.2, 0.3]

    def get_chat_completion(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        self.chat_calls += 1
        if not self.chat_responses:
            raise AssertionError("No fake chat response left")
        return self.chat_responses.pop(0)

    def get_chat_completion_with_context(
        self,
        context,
        user_query,
        system_instruction,
        temperature=0.7,
        max_tokens=2000,
    ):
        self.context_calls += 1
        if not self.context_responses:
            raise AssertionError("No fake context response left")
        return self.context_responses.pop(0)


async def fake_search_knowledge_base(query, top_k=5):
    return {
        "status": "success",
        "query": query,
        "count": 1,
        "results": [{
            "document": (
                "ADPKD treatment commonly includes blood pressure control, "
                "symptom monitoring, and sometimes tolvaptan."
            ),
            "metadata": {
                "title": "ADPKD Treatment Review",
                "author": "Torres",
                "year": "2024",
                "file_name": "adpkd_treatment_review.pdf",
                "citation": "Torres (2024)",
                "display_name": "ADPKD Treatment Review",
                "file": "adpkd_treatment_review.pdf",
            },
            "relevance_score": 0.93,
        }],
    }


async def fake_empty_search_knowledge_base(query, top_k=5):
    return {
        "status": "success",
        "query": query,
        "count": 0,
        "results": [],
    }


class FakeValidationResult:
    def to_dict(self):
        return {
            "passed": False,
            "overall_score": 0.2,
            "checks": {
                "relevance": {"passed": False, "score": 0.2},
                "source_attribution": {"passed": True, "score": 0.8},
                "safety": {"passed": True, "score": 0.8},
            },
            "warnings": ["Forced refusal for test"],
        }


class FakeValidationAgent:
    def __init__(self, openai_service):
        self.openai_service = openai_service

    async def validate_and_retry(
        self,
        query,
        answer,
        retrieved_chunks,
        agent_type="standard_rag",
        temperature=0.7,
        max_tokens=2000,
    ):
        return {
            "answer": REFUSAL_MESSAGE,
            "validation": FakeValidationResult(),
            "was_regenerated": True,
        }


class RefusalFlowTests(unittest.IsolatedAsyncioTestCase):
    def build_chat_request(self, query, **overrides):
        request_data = {
            "query": query,
            "use_adaptive_agent": False,
            "use_cot": False,
            "use_stepback": False,
            "use_validation": False,
            "use_query_rewriting": True,
            "top_k": 1,
        }
        request_data.update(overrides)
        return main_openai.ChatRequest(**request_data)

    async def test_refusal_utility_matches_wrapped_text(self):
        wrapped = f'  "{REFUSAL_MESSAGE}"  '
        self.assertTrue(is_refusal_response(wrapped))
        self.assertFalse(is_refusal_response("ADPKD is a genetic kidney disease."))

    async def test_intro_query_utility_returns_friendly_intro(self):
        self.assertTrue(is_intro_query("who are you"))
        self.assertFalse(is_refusal_response(INTRO_MESSAGE))

    async def test_refusal_utility_matches_verbose_off_topic_response(self):
        verbose_refusal = (
            "It seems there has been a mix-up in the context of your question. "
            "I am specialized in discussing Autosomal Dominant Polycystic Kidney Disease (ADPKD), "
            "not Euler's theorem. If you have questions related to ADPKD, I would be happy to help."
        )

        self.assertTrue(is_refusal_response(verbose_refusal))
        self.assertEqual(normalize_refusal_response(verbose_refusal), REFUSAL_MESSAGE)

    async def test_query_scope_is_pkd_only(self):
        self.assertTrue(is_query_in_scope("How is ADPKD treated?"))
        self.assertTrue(is_query_in_scope("When is tolvaptan used in PKD?"))
        self.assertFalse(is_query_in_scope("Explain Euler's theorem"))
        self.assertFalse(is_query_in_scope("What is chronic kidney disease?"))

    async def test_followup_agent_skips_llm_for_refusal(self):
        service = FakeOpenAIService(chat_responses=['["q1?","q2?","q3?"]'])

        questions = await followup_agent.FollowUpAgent(service).generate_followup_questions(
            query="Tell me about sports",
            response=REFUSAL_MESSAGE,
        )

        self.assertEqual(questions, [])
        self.assertEqual(service.chat_calls, 0)

    async def test_followup_agent_skips_llm_for_verbose_refusal(self):
        service = FakeOpenAIService(chat_responses=['["q1?","q2?","q3?"]'])
        verbose_refusal = (
            "I can only answer questions about PKD and ADPKD. "
            "For other topics, I recommend consulting a relevant expert."
        )

        questions = await followup_agent.FollowUpAgent(service).generate_followup_questions(
            query="Tell me about sports",
            response=verbose_refusal,
        )

        self.assertEqual(questions, [])
        self.assertEqual(service.chat_calls, 0)

    async def test_standard_rag_returns_refused_payload_without_sources(self):
        service = FakeOpenAIService(context_responses=[REFUSAL_MESSAGE])
        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            result = await openai_rag_init.get_rag_response("Tell me about basketball", top_k=1)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["refused"])
        self.assertEqual(result["response"], REFUSAL_MESSAGE)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["retrieved_chunks"], [])

    async def test_standard_rag_pre_gates_off_topic_without_llm(self):
        service = FakeOpenAIService()
        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            result = await openai_rag_init.get_rag_response("Explain Euler's theorem", top_k=1)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["refused"])
        self.assertEqual(result["response"], REFUSAL_MESSAGE)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["retrieved_chunks"], [])
        self.assertEqual(service.context_calls, 0)

    async def test_standard_rag_pre_gates_intro_without_llm(self):
        service = FakeOpenAIService()
        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            result = await openai_rag_init.get_rag_response("who are you", top_k=1)

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["refused"])
        self.assertEqual(result["response"], INTRO_MESSAGE)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["retrieved_chunks"], [])
        self.assertEqual(service.context_calls, 0)

    async def test_standard_rag_grounded_answer_keeps_sources(self):
        service = FakeOpenAIService(context_responses=["ADPKD is a genetic kidney disease."])
        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            result = await openai_rag_init.get_rag_response("What is ADPKD?", top_k=1)

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["refused"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(len(result["retrieved_chunks"]), 1)

    async def test_standard_rag_returns_empty_kb_message_without_llm_answer(self):
        service = FakeOpenAIService()
        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
            patch.object(openai_rag_init, "search_knowledge_base", fake_empty_search_knowledge_base),
        ):
            result = await openai_rag_init.get_rag_response("What is ADPKD?", top_k=1)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["refused"])
        self.assertEqual(result["response"], openai_rag_init.EMPTY_KNOWLEDGE_BASE_MESSAGE)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["retrieved_chunks"], [])
        self.assertEqual(service.context_calls, 0)

    async def test_cot_off_topic_returns_refused_payload(self):
        service = FakeOpenAIService(chat_responses=["OFF_TOPIC"])
        cot_service = cot_rag_service.ChainOfThoughtRAG(service)

        result = await cot_service.get_cot_rag_response("Who won the Super Bowl?")

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["refused"])
        self.assertEqual(result["response"], REFUSAL_MESSAGE)
        self.assertEqual(result["sources"], [])

    async def test_cot_synthesis_refusal_returns_refused_payload(self):
        service = FakeOpenAIService(chat_responses=[
            "1. What is ADPKD?",
            "**Finding:** ADPKD is an inherited kidney disease.\n**Evidence:** The review describes ADPKD as a genetic cystic kidney disease.",
            REFUSAL_MESSAGE,
        ])
        cot_service = cot_rag_service.ChainOfThoughtRAG(service)

        with patch.object(openai_rag_init, "search_knowledge_base", fake_search_knowledge_base):
            result = await cot_service.get_cot_rag_response("How is ADPKD treated?", top_k_per_step=1)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["refused"])
        self.assertEqual(result["response"], REFUSAL_MESSAGE)
        self.assertEqual(result["sources"], [])

    async def test_cot_empty_kb_returns_message_without_sources(self):
        service = FakeOpenAIService(chat_responses=["1. What is ADPKD?"])
        cot_service = cot_rag_service.ChainOfThoughtRAG(service)

        with patch.object(openai_rag_init, "search_knowledge_base", fake_empty_search_knowledge_base):
            result = await cot_service.get_cot_rag_response("What is ADPKD?", top_k_per_step=1)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["refused"])
        self.assertEqual(result["response"], openai_rag_init.EMPTY_KNOWLEDGE_BASE_MESSAGE)
        self.assertEqual(result["sources"], [])

    async def test_streaming_cot_abstains_when_every_step_has_no_context(self):
        service = FakeOpenAIService(chat_responses=["1. What is ADPKD?"])
        request = self.build_chat_request("What is ADPKD?", use_cot=True)

        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "search_knowledge_base", fake_empty_search_knowledge_base),
        ):
            events = [
                json.loads(event)
                async for event in main_openai.generate_chat_stream(request)
            ]

        chunks = [event["data"] for event in events if event["type"] == "chunk"]
        self.assertEqual(chunks, [openai_rag_init.EMPTY_KNOWLEDGE_BASE_MESSAGE])
        self.assertFalse(any(event["type"] == "error" for event in events))

    async def test_stepback_off_topic_returns_refused_payload(self):
        service = FakeOpenAIService(chat_responses=["OFF_TOPIC"])
        agent = stepback_agent.StepbackAgent(service)

        result = await agent.answer_with_stepback("What is the weather today?", top_k=1)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["refused"])
        self.assertEqual(result["response"], REFUSAL_MESSAGE)
        self.assertEqual(result["sources"], [])

    async def test_stepback_empty_kb_returns_message_without_sources(self):
        service = FakeOpenAIService(chat_responses=[
            "What treatment approaches are used for ADPKD?",
        ])
        agent = stepback_agent.StepbackAgent(service)

        with patch.object(openai_rag_init, "search_knowledge_base", fake_empty_search_knowledge_base):
            result = await agent.answer_with_stepback("How is ADPKD treated?", top_k=1)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["refused"])
        self.assertEqual(result["response"], openai_rag_init.EMPTY_KNOWLEDGE_BASE_MESSAGE)
        self.assertEqual(result["sources"], [])

    async def test_stepback_combines_distinct_new_index_papers_without_file_field(self):
        agent = stepback_agent.StepbackAgent(FakeOpenAIService())
        original = {
            "results": [
                {"id": "a-1", "metadata": {"paper_id": "paper-a"}},
                {"id": "b-1", "metadata": {"paper_id": "paper-b"}},
            ],
            "retrieval_metadata": {"confidence": "high"},
        }
        stepback = {
            "results": [
                {"id": "a-2", "metadata": {"paper_id": "paper-a"}},
                {"id": "c-1", "metadata": {"paper_id": "paper-c"}},
            ],
            "retrieval_metadata": {"confidence": "moderate"},
        }

        combined = agent._combine_results(original, stepback, "specific", "broader")

        self.assertEqual([doc["id"] for doc in combined["results"]], ["a-1", "b-1", "c-1"])
        self.assertEqual(combined["combined_count"], 3)
        self.assertEqual(combined["retrieval_metadata"]["confidence"], "context_available")

    async def test_stepback_answer_refusal_returns_refused_payload(self):
        service = FakeOpenAIService(chat_responses=[
            "What treatment approaches are used for ADPKD?",
            REFUSAL_MESSAGE,
        ])
        agent = stepback_agent.StepbackAgent(service)

        with patch.object(openai_rag_init, "search_knowledge_base", fake_search_knowledge_base):
            result = await agent.answer_with_stepback("How is ADPKD treated?", top_k=1)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["refused"])
        self.assertEqual(result["response"], REFUSAL_MESSAGE)
        self.assertEqual(result["sources"], [])

    @unittest.skip("Legacy endpoint pipeline replaced by test_bounded_runtime")
    async def test_chat_endpoint_refusal_returns_no_sources_and_no_followups(self):
        service = FakeOpenAIService(
            context_responses=[REFUSAL_MESSAGE],
            chat_responses=['["followup 1?","followup 2?","followup 3?"]'],
        )

        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            response = await main_openai.chat_endpoint(
                self.build_chat_request("Tell me about basketball")
            )

        self.assertEqual(response.status, "success")
        self.assertEqual(response.response, REFUSAL_MESSAGE)
        self.assertEqual(len(response.sources), 0)
        self.assertEqual(response.followup_questions, [])

    @unittest.skip("Legacy endpoint pipeline replaced by test_bounded_runtime")
    async def test_chat_endpoint_semantic_refusal_returns_no_sources(self):
        verbose_refusal = (
            "It seems there has been a mix-up in the context of your question. "
            "I am specialized in discussing Autosomal Dominant Polycystic Kidney Disease (ADPKD), "
            "not Euler's theorem. If you have questions related to ADPKD, I would be more than happy to assist."
        )
        service = FakeOpenAIService(
            context_responses=[verbose_refusal],
            chat_responses=['["followup 1?","followup 2?","followup 3?"]'],
        )

        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            response = await main_openai.chat_endpoint(
                self.build_chat_request("Explain Euler's theorem")
            )

        self.assertEqual(response.status, "success")
        self.assertEqual(response.response, REFUSAL_MESSAGE)
        self.assertEqual(len(response.sources), 0)
        self.assertEqual(response.followup_questions, [])

    @unittest.skip("Public retrieval-debug overrides are disabled in production")
    async def test_chat_debug_preserves_retrieval_when_answer_refuses(self):
        service = FakeOpenAIService(context_responses=[REFUSAL_MESSAGE])
        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            response = await main_openai.chat_endpoint(
                self.build_chat_request("What is ADPKD?", include_retrieval_debug=True)
            )

        self.assertEqual(response.response, REFUSAL_MESSAGE)
        self.assertEqual(response.sources, [])
        self.assertEqual(len(response.retrieved_chunks), 1)

    @unittest.skip("Legacy endpoint pipeline replaced by test_bounded_runtime")
    async def test_chat_endpoint_intro_returns_no_sources_and_no_followups(self):
        service = FakeOpenAIService()

        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            response = await main_openai.chat_endpoint(
                self.build_chat_request("who are you")
            )

        self.assertEqual(response.status, "success")
        self.assertEqual(response.response, INTRO_MESSAGE)
        self.assertEqual(len(response.sources), 0)
        self.assertEqual(response.followup_questions, [])
        self.assertEqual(service.context_calls, 0)
        self.assertEqual(service.chat_calls, 0)

    @unittest.skip("Legacy endpoint pipeline replaced by test_bounded_runtime")
    async def test_chat_endpoint_grounded_answer_keeps_sources_and_followups(self):
        service = FakeOpenAIService(
            context_responses=["ADPKD is a genetic kidney disease."],
            chat_responses=['["How is ADPKD monitored?","What symptoms should I watch for?","When is tolvaptan considered?"]'],
        )

        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
        ):
            response = await main_openai.chat_endpoint(
                self.build_chat_request("What is ADPKD?")
            )

        self.assertEqual(response.status, "success")
        self.assertEqual(response.response, "ADPKD is a genetic kidney disease.")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(len(response.followup_questions), 3)

    @unittest.skip("Corrective regeneration is disabled in production")
    async def test_chat_endpoint_validation_refusal_returns_no_sources(self):
        service = FakeOpenAIService(
            context_responses=["ADPKD is a genetic kidney disease."],
            chat_responses=['["How is ADPKD monitored?","What symptoms should I watch for?","When is tolvaptan considered?"]'],
        )

        with (
            patch.object(openai_rag_init, "openai_service", service),
            patch.object(openai_rag_init, "openai_collection", FakeCollection()),
            patch.object(validation_agent, "ValidationAgent", FakeValidationAgent),
        ):
            response = await main_openai.chat_endpoint(
                self.build_chat_request("What is ADPKD?", use_validation=True)
            )

        self.assertEqual(response.status, "success")
        self.assertEqual(response.response, REFUSAL_MESSAGE)
        self.assertEqual(len(response.sources), 0)
        self.assertEqual(response.followup_questions, [])
        self.assertIsNone(response.validation)


if __name__ == "__main__":
    unittest.main()
