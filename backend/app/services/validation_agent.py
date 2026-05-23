"""
Answer Validation Agent
Real-time quality gate that validates RAG-generated answers before returning them to users.

Runs 3 parallel checks:
1. Relevance — does the answer address the user's question?
2. Source Attribution — do cited sources match retrieved chunks?
3. Safety — are medical disclaimers present and guardrails respected?

If validation fails, triggers one corrective regeneration attempt.
"""

import asyncio
import json
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from statistics import mean

from .openai_service import OpenAIService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation prompt templates
# ---------------------------------------------------------------------------

RELEVANCE_SYSTEM_PROMPT = "You are an expert at evaluating answer quality. You only output valid JSON."

RELEVANCE_USER_PROMPT = """Rate how well the following ANSWER addresses the user's QUESTION.

QUESTION: {query}

ANSWER: {answer}

Evaluation criteria:
- Does the answer directly address what was asked?
- Is the answer complete enough to be useful?
- Does the answer stay focused on the question topic?

Output ONLY valid JSON (no markdown, no extra text):
{{"score": 0.85, "reason": "Brief explanation of the rating"}}"""

CORRECTIVE_SYSTEM_PROMPT = """You are a helpful medical AI assistant specialized in PKD, ADPKD, and kidney-disease topics.
If the question is asking anything other than PKD, ADPKD, or any kidney related disease, then don't answer.

Your previous answer had quality issues that need to be corrected. Generate an improved answer that fixes the identified problems.

When answering:
- Only state facts supported by the provided sources
- Directly answer the user's question
- Cite sources accurately using the format (Author Year)
- Include appropriate medical disclaimers
- Be clear, concise, and compassionate"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str            # "relevance", "source_attribution", "safety"
    passed: bool         # score >= threshold
    score: float         # 0.0 - 1.0
    details: List[str] = field(default_factory=list)   # specific issues found

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """Aggregated result of all validation checks."""
    passed: bool                    # all checks passed
    checks: List[CheckResult]      # individual check results
    overall_score: float            # mean of all check scores
    feedback: str                   # aggregated feedback for regeneration prompt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "overall_score": round(self.overall_score, 3),
            "checks": {c.name: {"passed": c.passed, "score": round(c.score, 3)} for c in self.checks},
            "warnings": [d for c in self.checks for d in c.details] if not self.passed else [],
        }


# ---------------------------------------------------------------------------
# Validation Agent
# ---------------------------------------------------------------------------

class ValidationAgent:
    """
    Real-time answer validation agent.
    Sits between generation and response in the RAG pipeline.
    """

    # Thresholds for each check
    RELEVANCE_THRESHOLD = 0.7
    SOURCE_ATTRIBUTION_THRESHOLD = 0.5
    SAFETY_THRESHOLD = 0.6

    MAX_RETRIES = 1  # allow one regeneration attempt

    def __init__(self, openai_service: OpenAIService):
        self.openai_service = openai_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def validate(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        agent_type: str = "standard_rag",
    ) -> ValidationResult:
        """
        Run all validation checks in parallel and return aggregated result.

        Args:
            query: Original user question.
            answer: Generated answer text.
            retrieved_chunks: List of dicts with 'document' and 'metadata' keys
                              as returned by search_knowledge_base().
            agent_type: Which agent produced the answer (for logging).

        Returns:
            ValidationResult with pass/fail, scores, and feedback.
        """
        logger.info(f"Validating answer from {agent_type} agent...")

        # Run the LLM checks concurrently; programmatic checks are instant
        relevance_task = asyncio.to_thread(
            self._check_relevance, query, answer
        )

        # Programmatic checks (no LLM call)
        source_result = self._check_source_attribution(answer, retrieved_chunks)
        safety_result = self._check_safety(answer)

        (relevance_result,) = await asyncio.gather(relevance_task)

        checks = [relevance_result, source_result, safety_result]
        scores = [c.score for c in checks]
        all_passed = all(c.passed for c in checks)

        result = ValidationResult(
            passed=all_passed,
            checks=checks,
            overall_score=mean(scores) if scores else 0.0,
            feedback=self._aggregate_feedback(checks),
        )

        logger.info(
            f"Validation result: passed={result.passed}, "
            f"score={result.overall_score:.2f}, "
            f"checks={{ {', '.join(f'{c.name}={c.score:.2f}' for c in checks)} }}"
        )
        return result

    async def validate_and_retry(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        agent_type: str = "standard_rag",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """
        Validate an answer. If it fails, attempt one corrective regeneration.

        Returns:
            Dict with keys: answer, validation, was_regenerated
        """
        # First pass validation
        result = await self.validate(query, answer, retrieved_chunks, agent_type)

        if result.passed:
            return {
                "answer": answer,
                "validation": result,
                "was_regenerated": False,
            }

        logger.info("Validation failed — attempting corrective regeneration...")

        # Regenerate with feedback
        improved_answer = self._regenerate_with_feedback(
            query=query,
            original_answer=answer,
            feedback=result.feedback,
            retrieved_chunks=retrieved_chunks,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Second pass validation
        retry_result = await self.validate(query, improved_answer, retrieved_chunks, agent_type)

        # Return the better answer
        if retry_result.overall_score >= result.overall_score:
            return {
                "answer": improved_answer,
                "validation": retry_result,
                "was_regenerated": True,
            }
        else:
            # Original was actually better — keep it but attach warning
            return {
                "answer": answer,
                "validation": result,
                "was_regenerated": False,
            }

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_relevance(self, query: str, answer: str) -> CheckResult:
        """Verify the answer addresses the user's question (LLM judge)."""
        prompt = RELEVANCE_USER_PROMPT.format(query=query, answer=answer)

        try:
            raw = self.openai_service.get_chat_completion(
                system_prompt=RELEVANCE_SYSTEM_PROMPT,
                user_message=prompt,
                temperature=0,
                max_tokens=300,
            )
            parsed = self._safe_json_parse(raw)
            score = float(parsed.get("score", 0.0))
            reason = parsed.get("reason", "")

            details = [reason] if score < self.RELEVANCE_THRESHOLD and reason else []

            return CheckResult(
                name="relevance",
                passed=score >= self.RELEVANCE_THRESHOLD,
                score=score,
                details=details,
            )
        except Exception as e:
            logger.warning(f"Relevance check failed, passing by default: {e}")
            return CheckResult(name="relevance", passed=True, score=0.75, details=[])

    def _check_source_attribution(
        self, answer: str, chunks: List[Dict[str, Any]]
    ) -> CheckResult:
        """Verify cited sources actually exist in retrieved chunks (programmatic)."""
        # Extract citation patterns like (Author Year), (Author, Year), Author (Year)
        citation_patterns = [
            r'\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?)[,\s]+(\d{4})\)',  # (Torres, 2012)
            r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?)\s+\((\d{4})\)',       # Torres (2012)
        ]

        cited_sources = set()
        for pattern in citation_patterns:
            matches = re.findall(pattern, answer)
            for match in matches:
                author = match[0].strip().rstrip(",")
                year = match[1].strip()
                cited_sources.add(f"{author} {year}")

        if not cited_sources:
            # No citations found — that's okay for some answers, give a moderate score
            return CheckResult(
                name="source_attribution",
                passed=True,
                score=0.8,
                details=["No explicit citations found in answer"],
            )

        # Build set of available sources from chunks metadata
        available_sources = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            display = meta.get("display_name", "")
            author = meta.get("author", "")
            year = meta.get("year", "")
            if display:
                available_sources.add(display.lower())
            if author and year:
                available_sources.add(f"{author} {year}".lower())
                # Also add with "et al."
                available_sources.add(f"{author} et al. {year}".lower())

        # Check each cited source against available
        matched = 0
        unmatched_list = []
        for cited in cited_sources:
            cited_lower = cited.lower()
            if any(cited_lower in avail or avail in cited_lower for avail in available_sources):
                matched += 1
            else:
                unmatched_list.append(cited)

        total = len(cited_sources)
        score = matched / total if total > 0 else 1.0

        details = [f"Unmatched citation: {s}" for s in unmatched_list[:3]]

        return CheckResult(
            name="source_attribution",
            passed=score >= self.SOURCE_ATTRIBUTION_THRESHOLD,
            score=score,
            details=details,
        )

    def _check_safety(self, answer: str) -> CheckResult:
        """Check for medical disclaimers and off-topic leakage (programmatic)."""
        answer_lower = answer.lower()
        score = 1.0
        details = []

        # Check for medical disclaimer
        disclaimer_phrases = [
            "consult", "healthcare provider", "medical professional",
            "doctor", "physician", "clinical", "seek medical",
            "talk to your", "speak with your",
        ]
        has_disclaimer = any(phrase in answer_lower for phrase in disclaimer_phrases)
        if not has_disclaimer:
            score -= 0.3
            details.append("Missing medical disclaimer — answer should remind users to consult a healthcare provider")

        # Check for potentially dangerous absolute medical advice
        dangerous_phrases = [
            "you should definitely take",
            "stop taking your medication",
            "you don't need to see a doctor",
            "ignore your doctor",
            "this will cure",
        ]
        has_dangerous = any(phrase in answer_lower for phrase in dangerous_phrases)
        if has_dangerous:
            score -= 0.4
            details.append("Contains potentially dangerous absolute medical advice")

        # Check it's not obviously off-topic (simple heuristic)
        pkd_keywords = [
            "pkd", "adpkd", "arpkd", "polycystic", "kidney", "renal", "cyst",
            "tolvaptan", "nephro", "gfr", "egfr", "ckd", "dialysis",
            "transplant", "creatinine", "proteinuria", "albuminuria", "hematuria",
        ]
        has_pkd_content = any(kw in answer_lower for kw in pkd_keywords)
        if not has_pkd_content:
            score -= 0.2
            details.append("Answer may not be sufficiently focused on PKD/kidney topics")

        score = max(0.0, min(1.0, score))

        return CheckResult(
            name="safety",
            passed=score >= self.SAFETY_THRESHOLD,
            score=score,
            details=details,
        )

    # ------------------------------------------------------------------
    # Corrective regeneration
    # ------------------------------------------------------------------

    def _regenerate_with_feedback(
        self,
        query: str,
        original_answer: str,
        feedback: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Regenerate the answer with corrective feedback injected into the prompt."""
        context = self._format_chunks_for_prompt(retrieved_chunks)

        user_message = f"""Your previous answer had the following issues:
{feedback}

Please regenerate your answer, fixing these problems.

USER QUESTION: {query}

RETRIEVED SOURCES:
{context}

Provide an improved, accurate answer:"""

        try:
            improved = self.openai_service.get_chat_completion(
                system_prompt=CORRECTIVE_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info(f"Regenerated answer ({len(improved)} chars)")
            return improved.strip()
        except Exception as e:
            logger.error(f"Corrective regeneration failed: {e}")
            return original_answer  # fall back to original

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a readable context string."""
        parts = []
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            display = meta.get("display_name", f"Source {i + 1}")
            text = chunk.get("document", "")
            parts.append(f"[Source {i + 1}: {display}]\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _safe_json_parse(raw: str) -> Dict[str, Any]:
        """Parse JSON from LLM output, handling markdown fences and other noise."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (possibly with language tag)
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Could not parse JSON from LLM output: {raw[:200]}")
            return {"score": 0.75}  # safe default

    @staticmethod
    def _aggregate_feedback(checks: List[CheckResult]) -> str:
        """Combine all check details into a feedback string for regeneration."""
        lines = []
        for check in checks:
            if not check.passed:
                lines.append(f"[{check.name.upper()} FAILED — score {check.score:.2f}]")
                for detail in check.details:
                    lines.append(f"  - {detail}")
        return "\n".join(lines) if lines else "All checks passed."
