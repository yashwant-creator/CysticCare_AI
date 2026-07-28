"""The bounded production RAG path: one embedding, local retrieval, one answer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import os
import re
import time
from typing import Any, Dict, List, Tuple

import tiktoken

from .cross_encoder_reranker import CrossEncoderConfig, CrossEncoderReranker
from .hybrid_retriever import HybridRetriever
from .runtime_openai import RuntimeOpenAI
from .usage_metrics import RequestUsage
from ..utils.refusal_utils import INTRO_MESSAGE, REFUSAL_MESSAGE, is_intro_query

TOP_K = 5
CONTEXT_TOKEN_BUDGET = 3_000
_rerank_semaphore = asyncio.Semaphore(2)

ANSWER_SYSTEM_PROMPT = """You are CysticCare AI, a warm and practical medical-information assistant specializing in PKD, ADPKD, and kidney disease.

The user may ask a short conversational follow-up without repeating "PKD" or
"kidney." Infer that context from the supplied PKD literature instead of
rejecting the question merely because it lacks a domain keyword.

Use the supplied medical-literature excerpts as evidence. Start with a clear,
useful answer, explain technical language, and cite supporting sources using
their displayed author and year. State uncertainty when the excerpts do not
support a claim. For personal medical decisions, encourage the user to discuss
the information with a qualified healthcare professional.

If the question is clearly unrelated to kidney disease and cannot reasonably be
a conversational follow-up, briefly explain that CysticCare focuses on PKD and
kidney disease. Never force an unrelated question into the retrieved evidence."""


@dataclass
class RetrievalOutput:
    query: str
    results: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    context: str


def hash_session_id(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def _expand_abbreviations(query: str) -> str:
    expansions = {
        "PKD": "Polycystic Kidney Disease",
        "ADPKD": "Autosomal Dominant Polycystic Kidney Disease",
        "ARPKD": "Autosomal Recessive Polycystic Kidney Disease",
        "TKV": "Total Kidney Volume",
        "eGFR": "estimated Glomerular Filtration Rate",
        "ESRD": "End-Stage Renal Disease",
        "CKD": "Chronic Kidney Disease",
    }
    expanded = query.strip()
    for abbreviation, full_name in expansions.items():
        expanded = re.sub(
            rf"\b{re.escape(abbreviation)}\b",
            full_name,
            expanded,
            flags=re.IGNORECASE,
        )
    return expanded


def _context_with_budget(results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = None
    used = 0
    selected: List[Dict[str, Any]] = []
    parts: List[str] = []
    for index, result in enumerate(results[:TOP_K], start=1):
        metadata = result.get("metadata") or {}
        display = metadata.get("display_name") or metadata.get("title") or f"Source {index}"
        text = str(result.get("document") or "")
        tokens = encoding.encode(text) if encoding is not None else text.split()
        remaining = CONTEXT_TOKEN_BUDGET - used
        if remaining <= 0:
            break
        if len(tokens) > remaining:
            tokens = tokens[:remaining]
            text = (
                encoding.decode(tokens)
                if encoding is not None
                else " ".join(tokens)
            )
        used += len(tokens)
        selected.append(result)
        parts.append(f"[Source {index}: {display}]\n{text}")
    return "\n\n".join(parts), selected


def _format_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        metadata = result.get("metadata") or {}
        key = str(
            metadata.get("paper_id")
            or metadata.get("display_name")
            or metadata.get("file_name")
            or result.get("id")
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "index": len(sources) + 1,
                "title": metadata.get("title", "Unknown"),
                "author": metadata.get("author", "Unknown"),
                "year": str(metadata.get("year", "Unknown")),
                "file": metadata.get("file_name", key),
                "citation": metadata.get("citation", ""),
                "display_name": metadata.get("display_name", ""),
                "relevance_score": float(result.get("relevance_score", 0)),
            }
        )
    return sources


async def retrieve(
    query: str,
    service: RuntimeOpenAI,
    tracker: RequestUsage,
) -> RetrievalOutput:
    from . import openai_rag_init

    if openai_rag_init.openai_collection is None:
        raise RuntimeError("Baked vector database is not loaded")
    started = time.perf_counter()
    search_query = _expand_abbreviations(query)
    query_embedding = await service.embedding(search_query, tracker)
    retriever = HybridRetriever()
    if not retriever.initialized:
        raise RuntimeError("Hybrid retriever is not initialized")
    candidates = await asyncio.to_thread(
        retriever.hybrid_search,
        query=search_query,
        query_embedding=query_embedding,
        top_k=TOP_K,
        candidate_pool_limit=40,
        result_limit=24,
    )
    config = CrossEncoderConfig.from_environment()
    reranker = CrossEncoderReranker(config)
    async with _rerank_semaphore:
        ranked, metadata = await asyncio.to_thread(
            reranker.rerank,
            search_query,
            candidates,
            TOP_K,
        )
    for item in ranked:
        parent_text = (item.get("metadata") or {}).get("parent_text")
        if parent_text:
            item["document"] = parent_text
    context, selected = _context_with_budget(ranked)
    tracker.retrieval_ms = round((time.perf_counter() - started) * 1000, 2)
    return RetrievalOutput(
        query=query,
        results=selected,
        sources=_format_sources(selected),
        metadata=metadata,
        context=context,
    )


def answer_user_message(query: str, retrieval: RetrievalOutput) -> str:
    return (
        f"QUESTION:\n{query}\n\n"
        f"MEDICAL LITERATURE EXCERPTS:\n{retrieval.context}\n\n"
        "Provide the answer."
    )


_ACTIONABLE_MEDICAL_QUERY_PATTERNS = (
    r"\b(?:should|can|could|may)\s+i\b",
    r"\b(?:is|would)\s+it\s+(?:be\s+)?safe\b",
    r"\b(?:safe|appropriate|recommended)\s+for\s+me\b",
    r"\b(?:start|stop|skip|restart|increase|decrease|change|switch)\b",
    r"\b(?:dose|dosage|how much|how often)\b",
    r"\b(?:drug|medication|supplement)\s+interaction\b",
    r"\b(?:urgent|emergency|emergency room|go to (?:the )?er)\b",
    r"\bwhat\s+symptoms?\s+should\s+i\b",
)


def requires_medical_disclaimer(query: str) -> bool:
    """Return whether a question asks for an actionable personal decision."""
    normalized = " ".join(query.lower().split())
    return any(
        re.search(pattern, normalized)
        for pattern in _ACTIONABLE_MEDICAL_QUERY_PATTERNS
    )


def local_validation(
    answer: str,
    sources: List[Dict[str, Any]],
    query: str = "",
) -> Dict[str, Any]:
    """Run deterministic safety and attribution checks without conflating them."""
    lowered = answer.lower()
    has_disclaimer = any(
        phrase in lowered
        for phrase in (
            "consult",
            "healthcare provider",
            "healthcare professional",
            "medical professional",
            "doctor",
            "physician",
            "nephrologist",
            "seek medical",
            "seek care",
            "talk to your",
            "speak with your",
        )
    )
    disclaimer_required = requires_medical_disclaimer(query)
    disclaimer_passed = not disclaimer_required or has_disclaimer
    prohibited = any(
        phrase in lowered
        for phrase in (
            "you should definitely take",
            "stop taking your medication",
            "you don't need to see a doctor",
            "ignore your doctor",
            "this will cure",
        )
    )
    cited = any(
        (
            source.get("author")
            and str(source["author"]).lower() in lowered
        )
        or (
            source.get("year")
            and str(source["year"]).lower() in lowered
            and str(source["year"]).lower() != "unknown"
        )
        or f"source {source.get('index')}" in lowered
        for source in sources
    )
    sources_present = bool(sources)
    safety_passed = disclaimer_passed and not prohibited
    attribution_passed = sources_present and cited
    warnings: List[str] = []
    if disclaimer_required and not has_disclaimer:
        warnings.append(
            "This actionable medical answer should encourage consultation with "
            "a qualified healthcare professional."
        )
    if prohibited:
        warnings.append("The answer contains potentially dangerous absolute medical advice.")
    if not sources_present:
        warnings.append("No retrieved sources were available for attribution.")
    elif not cited:
        warnings.append("The answer does not clearly attribute a retrieved source.")

    return {
        "passed": safety_passed and attribution_passed,
        "groups": {
            "safety": {
                "passed": safety_passed,
                "score": 1.0 if safety_passed else 0.0,
            },
            "source_attribution": {
                "passed": attribution_passed,
                "score": 1.0 if attribution_passed else 0.0,
            },
        },
        "checks": {
            "medical_disclaimer": {
                "passed": disclaimer_passed,
                "score": 1.0 if disclaimer_passed else 0.0,
                "required": disclaimer_required,
                "present": has_disclaimer,
            },
            "prohibited_advice": {"passed": not prohibited, "score": 0.0 if prohibited else 1.0},
            "sources_present": {"passed": sources_present, "score": 1.0 if sources_present else 0.0},
            "citation_attribution": {"passed": cited, "score": 1.0 if cited else 0.0},
        },
        "warnings": warnings,
    }


def validation_summary(
    query: str,
    answer: str,
    sources: List[Dict[str, Any]],
    relevance_score: float,
) -> Dict[str, Any]:
    """Build the three public validation groups used by both chat endpoints."""
    local = local_validation(answer, sources, query=query)
    relevance_score = min(1.0, max(0.0, float(relevance_score)))
    relevance_passed = relevance_score >= 0.7
    safety = local["groups"]["safety"]
    attribution = local["groups"]["source_attribution"]
    warnings = list(local["warnings"])
    if not relevance_passed:
        warnings.insert(0, "The answer may not directly address the question.")
    passed = relevance_passed and safety["passed"] and attribution["passed"]
    return {
        "passed": passed,
        "overall_score": round(
            (relevance_score + safety["score"] + attribution["score"]) / 3,
            3,
        ),
        "checks": {
            "relevance": {
                "passed": relevance_passed,
                "score": relevance_score,
            },
            "source_attribution": attribution,
            "safety": safety,
        },
        "warnings": warnings,
        "was_regenerated": False,
    }


def guardrail_answer(query: str) -> str | None:
    """Production gate that preserves natural conversational follow-ups.

    The answer prompt performs the final scope decision without adding another
    classifier call. Only empty input and deterministic introductions are
    handled before retrieval.
    """
    if is_intro_query(query):
        return INTRO_MESSAGE
    if not query.strip():
        return REFUSAL_MESSAGE
    return None


def fallback_followup_questions(query: str) -> List[str]:
    """Always provide useful follow-ups if the helper response is unavailable."""
    lowered = query.lower()
    if any(term in lowered for term in ("treat", "medication", "tolvaptan")):
        return [
            "How are treatment benefits and side effects monitored?",
            "Which symptoms or test results should I discuss with my nephrologist?",
            "What lifestyle measures can support the treatment plan?",
        ]
    if any(term in lowered for term in ("symptom", "pain", "sign")):
        return [
            "Which symptoms should prompt urgent medical attention?",
            "How are these symptoms usually evaluated in people with PKD?",
            "What can I track before my next nephrology appointment?",
        ]
    return [
        "How is this usually monitored over time in people with PKD?",
        "Which symptoms or warning signs should prompt medical follow-up?",
        "What questions would be useful to ask a nephrologist about this?",
    ]


def normalize_postprocess(payload: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Validate helper output and supply deterministic non-LLM fallbacks."""
    raw_questions = payload.get("followup_questions", [])
    if not isinstance(raw_questions, list):
        raw_questions = []
    questions = [
        str(question).strip()
        for question in raw_questions
        if str(question).strip()
    ][:3]
    if len(questions) != 3:
        questions = fallback_followup_questions(query)
    try:
        relevance_score = float(payload.get("relevance_score", 0.7))
    except (TypeError, ValueError):
        relevance_score = 0.7
    return {
        **payload,
        "relevance_score": min(1.0, max(0.0, relevance_score)),
        "relevance_reason": str(
            payload.get("relevance_reason")
            or "Structured helper output was unavailable; local checks were used."
        ),
        "followup_questions": questions,
    }
