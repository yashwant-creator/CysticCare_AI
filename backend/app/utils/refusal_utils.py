"""Shared guardrail helpers for PKD-only prompt-gated responses."""

import re
from typing import Optional


INTRO_MESSAGE = (
    "Hi, I'm CysticCare AI. I'm here to answer questions about PKD and ADPKD, "
    "including symptoms, diagnosis, treatment options, monitoring, and research."
)

REFUSAL_MESSAGE = "I am a chatbot designed to answer questions only about PKD and ADPKD."

_INTRO_QUERY_PATTERNS = (
    r"^who are you(?:[?.!]|$)",
    r"^what are you(?:[?.!]|$)",
    r"^(?:can you )?introduce yourself(?:[?.!]|$)",
    r"^tell me about yourself(?:[?.!]|$)",
    r"^what(?:'s| is) your name(?:[?.!]|$)",
)

_PKD_QUERY_PATTERNS = (
    r"\bpkd\b",
    r"\badpkd\b",
    r"\barpkd\b",
    r"\bpolycystic kidney disease\b",
    r"\bautosomal dominant polycystic kidney disease\b",
    r"\bautosomal recessive polycystic kidney disease\b",
    r"\bpkd1\b",
    r"\bpkd2\b",
    r"\bpkhd1\b",
    r"\bpolycystin(?:-?1|-?2)?\b",
    r"\btolvaptan\b",
    r"\bjynarque\b",
    r"\btotal kidney volume\b",
    r"\btkv\b",
    r"\bmayo imaging classification\b",
)

_REFUSAL_CUE_PATTERNS = (
    r"\bi can only answer\b",
    r"\bi can(?:not|n't)\b",
    r"\b(?:i am|i'm) (?:specifically designed|designed|specialized|focused|trained)\b",
    r"\boutside (?:my|the) scope\b",
    r"\bnot (?:within|in) (?:my|the) scope\b",
    r"\boff[- ]topic\b",
    r"\bfor other topics\b",
    r"\bi recommend consulting\b",
    r"\bi recommend .*expert\b",
    r"\bdon't answer\b",
)

_DOMAIN_CUE_PATTERNS = (
    r"\bpkd\b",
    r"\badpkd\b",
    r"\bpolycystic kidney disease\b",
    r"\bkidney[- ]related disease\b",
    r"\bkidney[- ]disease topics?\b",
)

_INSUFFICIENT_CONTEXT_PATTERNS = (
    r"^i can(?:not|n't) answer(?: that| this)?(?: question)?(?: based on .+)?$",
    r"^i do not have enough (?:information|context)\b",
    r"^i don't have enough (?:information|context)\b",
    r"^there (?:is|isn't) not enough (?:information|context)\b",
    r"^the provided (?:sources|context) (?:do|does) not contain enough\b",
    r"^unable to answer based on (?:the )?(?:provided )?(?:sources|context)\b",
)


def _normalize_text(text: Optional[str]) -> str:
    """Collapse whitespace and casing for stable string checks."""
    if not text:
        return ""
    return " ".join(text.strip().strip('"\'' ).lower().split())


def _matches_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    """Return True when any regex pattern matches the text."""
    return any(re.search(pattern, text) for pattern in patterns)


def is_query_in_scope(query: Optional[str]) -> bool:
    """Return True when the query is clearly about PKD or ADPKD."""
    normalized = _normalize_text(query)
    if not normalized:
        return False
    return _matches_any_pattern(normalized, _PKD_QUERY_PATTERNS)


def is_intro_query(query: Optional[str]) -> bool:
    """Return True when the user is asking for a basic bot introduction."""
    normalized = _normalize_text(query)
    if not normalized:
        return False
    return _matches_any_pattern(normalized, _INTRO_QUERY_PATTERNS)


def get_guardrail_response(query: Optional[str]) -> Optional[str]:
    """Return an immediate canned response for intro or out-of-scope queries."""
    if is_intro_query(query):
        return INTRO_MESSAGE
    if not is_query_in_scope(query):
        return REFUSAL_MESSAGE
    return None


def is_refusal_response(response: Optional[str]) -> bool:
    """Check whether a response is an explicit or semantic refusal."""
    normalized = _normalize_text(response)
    if not normalized:
        return False

    if normalized in {_normalize_text(REFUSAL_MESSAGE), "off_topic", "off-topic"}:
        return True

    if _matches_any_pattern(normalized, _INSUFFICIENT_CONTEXT_PATTERNS):
        return True

    has_refusal_cue = _matches_any_pattern(normalized, _REFUSAL_CUE_PATTERNS)
    has_domain_cue = _matches_any_pattern(normalized, _DOMAIN_CUE_PATTERNS)
    return has_refusal_cue and has_domain_cue


def normalize_refusal_response(response: Optional[str]) -> str:
    """Collapse refusal variants to the canonical user-facing refusal."""
    if is_refusal_response(response):
        return REFUSAL_MESSAGE
    return response or REFUSAL_MESSAGE
