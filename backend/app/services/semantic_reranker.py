"""Second-stage, evidence-focused reranking for retrieved research passages.

Dense/BM25 fusion is intentionally broad: it finds plausible candidates.  This
module performs the narrower question-to-passage judgement before material is
sent to the answer model.  It is deliberately fail-open: an unavailable helper
model never prevents the knowledge base from returning its fused candidates.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        logger.warning("Invalid %s value; using %s", name, default)
        return default


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        return min(maximum, max(minimum, float(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        logger.warning("Invalid %s value; using %s", name, default)
        return default


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value in choices:
        return value
    logger.warning("Invalid %s value %r; using %s", name, value, default)
    return default


@dataclass(frozen=True)
class RerankerConfig:
    """Runtime controls for the semantic reranking and confidence gate.

    `min_relevance_score` is intentionally applied only to a semantic score
    produced by the reranker. RRF ranks are relative to the candidate set and
    must never be treated as calibrated confidence values.
    """

    enabled: bool = True
    candidate_limit: int = 24
    min_relevance_score: float = 0.50
    min_score_margin: float = 0.05
    max_chunks_per_document: int = 2
    max_chars_per_candidate: int = 1_200
    max_completion_tokens: int = 2_000
    backend: str = "llm"
    fallback_to_llm: bool = True

    @classmethod
    def from_environment(cls) -> "RerankerConfig":
        return cls(
            enabled=_env_bool("RAG_RERANK_ENABLED", True),
            candidate_limit=min(60, _env_int("RAG_RERANK_CANDIDATES", 24)),
            min_relevance_score=_env_float("RAG_MIN_RELEVANCE_SCORE", 0.50),
            min_score_margin=_env_float("RAG_MIN_SCORE_MARGIN", 0.05),
            max_chunks_per_document=min(10, _env_int("RAG_MAX_CHUNKS_PER_DOCUMENT", 2)),
            max_chars_per_candidate=min(
                4_000, _env_int("RAG_RERANK_MAX_CHARS", 1_200, minimum=200)
            ),
            max_completion_tokens=min(
                4_000, _env_int("RAG_RERANK_MAX_TOKENS", 2_000, minimum=400)
            ),
            # MedCPT is opt-in until its raw-logit threshold has been
            # calibrated on the labelled PKD benchmark.  This preserves the
            # established LLM evidence gate for existing deployments.
            backend=_env_choice(
                "RAG_RERANKER_BACKEND",
                "llm",
                {"llm", "medcpt", "off"},
            ),
            fallback_to_llm=_env_bool("RAG_CROSS_ENCODER_FALLBACK_TO_LLM", True),
        )


class SemanticReranker:
    """Use a helper LLM to score whether a candidate contains answer evidence."""

    _SYSTEM_PROMPT = """You are a rigorous medical-literature retrieval judge.
Score whether each candidate passage contains direct, usable evidence for the
user's exact question. Score evidence support, not superficial keyword overlap.
High scores require that the passage could substantively support an answer;
background-only, tangential, or contradictory passages should score low.
Do not infer facts absent from a passage. Return a score for every candidate."""

    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ratings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["id", "score"],
                },
            }
        },
        "required": ["ratings"],
    }

    def __init__(self, openai_service: Any, config: Optional[RerankerConfig] = None):
        self.openai_service = openai_service
        self.config = config or RerankerConfig.from_environment()

    @staticmethod
    def _document_key(candidate: Dict[str, Any]) -> str:
        metadata = candidate.get("metadata") or {}
        return str(
            metadata.get("paper_id")
            or metadata.get("parent_document_id")
            or metadata.get("document_key")
            or metadata.get("file_name")
            or candidate.get("id")
            or "unknown"
        )

    def _build_prompt(self, query: str, candidates: Iterable[Dict[str, Any]]) -> str:
        records: List[str] = []
        for candidate in candidates:
            metadata = candidate.get("metadata") or {}
            text = " ".join(str(candidate.get("document", "")).split())
            if len(text) > self.config.max_chars_per_candidate:
                text = text[: self.config.max_chars_per_candidate].rsplit(" ", 1)[0] + "…"
            title = metadata.get("title") or metadata.get("display_name") or "Unknown paper"
            section = metadata.get("section_heading") or "Unspecified section"
            page = metadata.get("page_number") or metadata.get("page_start") or "unknown"
            records.append(
                f"ID: {candidate.get('id')}\nPaper: {title}\nSection: {section}\n"
                f"Page: {page}\nPassage: {text}"
            )

        return f"Question: {query}\n\nCandidates:\n\n" + "\n\n---\n\n".join(records)

    @staticmethod
    def _clean_scores(payload: Any, expected_ids: Iterable[str]) -> Dict[str, float]:
        valid_ids = {str(candidate_id) for candidate_id in expected_ids}
        scores: Dict[str, float] = {}
        ratings = payload.get("ratings", []) if isinstance(payload, dict) else []
        for rating in ratings:
            if not isinstance(rating, dict):
                continue
            candidate_id = str(rating.get("id", ""))
            if candidate_id not in valid_ids:
                continue
            try:
                score = float(rating.get("score"))
            except (TypeError, ValueError):
                continue
            scores[candidate_id] = min(1.0, max(0.0, score))
        return scores

    def _score_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        if self.openai_service is None:
            raise RuntimeError("No helper OpenAI service is configured for reranking")
        payload = self.openai_service.get_structured_chat_completion(
            system_prompt=self._SYSTEM_PROMPT,
            user_message=self._build_prompt(query, candidates),
            schema_name="passage_relevance_ratings",
            schema=self._rating_schema(len(candidates)),
            # Reasoning models count hidden work against this budget. A
            # conservative configurable floor avoids a valid judge response
            # being truncated merely because its JSON payload is small.
            max_tokens=max(self.config.max_completion_tokens, len(candidates) * 30),
        )
        expected_ids = {str(candidate.get("id")) for candidate in candidates}
        scores = self._clean_scores(payload, expected_ids)
        if not scores:
            raise ValueError("Reranker returned no valid candidate scores")
        missing_ids = expected_ids.difference(scores)
        if missing_ids:
            raise ValueError(
                f"Reranker did not score {len(missing_ids)} candidate(s)"
            )
        return scores

    @classmethod
    def _rating_schema(cls, candidate_count: int) -> Dict[str, Any]:
        """Require one rating per candidate and validate it again locally.

        The constraints are supported by current OpenAI Structured Outputs.
        The local validation in :meth:`_score_candidates` remains necessary
        for the JSON-prompt fallback and compatible API gateways.
        """
        schema = deepcopy(cls._SCHEMA)
        ratings = schema["properties"]["ratings"]
        ratings["minItems"] = candidate_count
        ratings["maxItems"] = candidate_count
        return schema

    def _apply_document_diversity(
        self,
        candidates: Iterable[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        try:
            result_limit = int(top_k)
        except (TypeError, ValueError):
            return []
        if result_limit <= 0:
            return []

        selected: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        for candidate in candidates:
            document_key = self._document_key(candidate)
            if counts.get(document_key, 0) >= self.config.max_chunks_per_document:
                continue
            counts[document_key] = counts.get(document_key, 0) + 1
            selected.append(candidate)
            if len(selected) >= result_limit:
                break
        return selected

    @staticmethod
    def _confidence(top_score: float, margin: Optional[float], config: RerankerConfig) -> str:
        if top_score < config.min_relevance_score:
            return "insufficient"
        if margin is not None and margin < config.min_score_margin:
            return "ambiguous"
        if top_score >= min(1.0, config.min_relevance_score + 0.20):
            return "high"
        return "moderate"

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Return context-safe candidates and an auditable confidence decision."""
        candidate_subset = list(candidates[: min(self.config.candidate_limit, 60)])
        base_metadata: Dict[str, Any] = {
            "reranker_enabled": self.config.enabled,
            "reranker_backend": "off" if self.config.backend == "off" else "llm",
            "candidate_count": len(candidate_subset),
            "threshold": self.config.min_relevance_score,
            "min_score_margin": self.config.min_score_margin,
            "reranker_used": False,
            "score_type": "fused_rank",
            "confidence": "unavailable",
            "accepted": bool(candidate_subset),
        }
        if not candidate_subset:
            base_metadata["selected_count"] = 0
            return [], base_metadata

        if not self.config.enabled:
            selected = self._apply_document_diversity(candidate_subset, top_k)
            base_metadata["selected_count"] = len(selected)
            return selected, base_metadata

        try:
            scores = self._score_candidates(query, candidate_subset)
        except Exception as error:
            logger.warning("Semantic reranking failed; retaining fused candidates: %s", error)
            # Never expose provider exception text in API metadata. It can
            # contain request diagnostics while the server log retains detail.
            base_metadata.update(
                {
                    "reranker_error": "semantic_reranker_unavailable",
                    "confidence": "fallback",
                }
            )
            selected = self._apply_document_diversity(candidate_subset, top_k)
            base_metadata["selected_count"] = len(selected)
            return selected, base_metadata

        ranked: List[Dict[str, Any]] = []
        for candidate in candidate_subset:
            result = dict(candidate)
            result["metadata"] = dict(candidate.get("metadata") or {})
            score = scores.get(str(candidate.get("id")), 0.0)
            result["fused_score"] = candidate.get("relevance_score")
            result["reranker_score"] = round(score, 4)
            result["relevance_score"] = round(score, 4)
            ranked.append(result)
        ranked.sort(key=lambda item: item["reranker_score"], reverse=True)
        for rank, candidate in enumerate(ranked, start=1):
            candidate["reranker_rank"] = rank

        top_score = ranked[0]["reranker_score"]
        margin = (
            round(top_score - ranked[1]["reranker_score"], 4)
            if len(ranked) > 1
            else None
        )
        confidence = self._confidence(top_score, margin, self.config)
        accepted = confidence != "insufficient"
        thresholded = [
            candidate
            for candidate in ranked
            if candidate["reranker_score"] >= self.config.min_relevance_score
        ]
        selected = self._apply_document_diversity(thresholded, top_k) if accepted else []

        base_metadata.update(
            {
                "reranker_used": True,
                "score_type": "semantic",
                "top_score": top_score,
                "score_margin": margin,
                "confidence": confidence,
                "accepted": accepted,
                "selected_count": len(selected),
            }
        )
        return selected, base_metadata


class _FallbackReranker:
    """Use the established LLM judge only when the selected local model fails.

    A valid MedCPT result, including a low-evidence rejection, is never
    overridden.  The fallback exists for deployment failures such as a missing
    local model cache or unavailable inference dependency.
    """

    def __init__(self, primary: Any, fallback: SemanticReranker):
        self.primary = primary
        self.fallback = fallback
        # ``search_knowledge_base`` reads this to size the broad candidate
        # pool. Both rerankers use the same candidate-limit environment value.
        self.config = primary.config

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        results, metadata = self.primary.rerank(query, candidates, top_k)
        primary_error = metadata.get("reranker_error")
        if not primary_error:
            return results, metadata

        fallback_results, fallback_metadata = self.fallback.rerank(
            query,
            candidates,
            top_k,
        )
        fallback_metadata.update(
            {
                "requested_reranker_backend": getattr(
                    self.primary, "backend_name", "cross_encoder"
                ),
                "fallback_reranker_backend": "llm",
                "primary_reranker_error": primary_error,
            }
        )
        return fallback_results, fallback_metadata


def create_evidence_reranker(
    openai_service: Any,
    config: Optional[RerankerConfig] = None,
    cross_encoder_scorer: Optional[Any] = None,
) -> Any:
    """Return the configured reranker without importing ML dependencies eagerly.

    ``RAG_RERANKER_BACKEND=medcpt`` activates the local biomedical
    cross-encoder.  Its optional LLM fallback preserves the existing
    fail-open behavior if the model cannot load.  The default remains ``llm``
    until a MedCPT logit cutoff has been calibrated on the held-out benchmark.
    """
    reranker_config = config or RerankerConfig.from_environment()
    if reranker_config.backend == "off":
        return SemanticReranker(
            None,
            replace(reranker_config, enabled=False),
        )
    if reranker_config.backend != "medcpt":
        return SemanticReranker(openai_service, reranker_config)

    # Avoid a module-level import: the default LLM deployment and its unit
    # tests do not need torch/transformers merely to import the API server.
    from .cross_encoder_reranker import CrossEncoderConfig, CrossEncoderReranker

    primary = CrossEncoderReranker(
        CrossEncoderConfig.from_environment(reranker_config),
        scorer=cross_encoder_scorer,
    )
    if not reranker_config.fallback_to_llm:
        return primary
    return _FallbackReranker(primary, SemanticReranker(openai_service, reranker_config))
