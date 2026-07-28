"""
Prompt Guard - Centralized off-topic detection and guardrails for PKD chatbot
Ensures responses stay within the scope of PKD/kidney disease
"""

import logging
import re
from typing import Dict, Any, Optional
from ..services.openai_service import OpenAIService

logger = logging.getLogger(__name__)

# Centralized restriction message
OFF_TOPIC_MESSAGE = """I appreciate your question, but I'm specifically designed to help with Polycystic Kidney Disease (PKD), ADPKD, and kidney-disease topics.

I can help you with:
• Understanding PKD/ADPKD symptoms, diagnosis, and progression
• Treatment options, medications, and monitoring for kidney disease
• Dialysis, transplant, lab results, and kidney-related complications
• Kidney-focused lifestyle questions tied to renal disease or nephrology care

If you have questions about PKD, ADPKD, or another kidney-related disease, I'd be happy to help!"""

OFF_TOPIC_SENTINEL = "OFF_TOPIC"
TOPIC_SCOPE_DESCRIPTION = (
    "Polycystic Kidney Disease (PKD), ADPKD, kidney disease, renal disease, "
    "nephrology, or kidney-related medical care"
)

DIRECT_ON_TOPIC_PATTERNS = (
    r"\bpkd\b",
    r"\badpkd\b",
    r"\barpkd\b",
    r"\bpolycystic kidney disease\b",
    r"\bkidney disease\b",
    r"\brenal disease\b",
    r"\bchronic kidney disease\b",
    r"\bckd\b",
    r"\bacute kidney injury\b",
    r"\baki\b",
    r"\bend-stage renal disease\b",
    r"\besrd\b",
    r"\btolvaptan\b",
    r"\bdialysis\b",
    r"\bhemodialysis\b",
    r"\bperitoneal dialysis\b",
    r"\bkidney transplant(?:ation)?\b",
    r"\brenal transplant(?:ation)?\b",
    r"\bnephrolog(?:y|ist|ists)\b",
    r"\bnephrectom(?:y|ies)\b",
    r"\begfr\b",
    r"\bgfr\b",
    r"\bcreatinin(?:e)?\b",
    r"\bproteinuria\b",
    r"\balbuminuria\b",
    r"\bhematuria\b",
    r"\bglomerul\w*\b",
    r"\bnephrotic\b",
    r"\bnephritic\b",
    r"\bhydronephrosis\b",
    r"\bkidney stone(?:s)?\b",
    r"\brenal stone(?:s)?\b",
    r"\bkidney cyst(?:s)?\b",
    r"\brenal cyst(?:s)?\b",
    r"\balport syndrome\b",
    r"\biga nephropathy\b",
)

KIDNEY_ANCHOR_PATTERNS = (
    r"\bkidney(?:s)?\b",
    r"\brenal\b",
    r"\bnephro\w*\b",
)

KIDNEY_MEDICAL_CONTEXT_PATTERNS = (
    r"\bdisease\b",
    r"\bdisorder\b",
    r"\bsymptom(?:s)?\b",
    r"\bdiagnos(?:is|e|ed|ing|tic)\b",
    r"\btreat(?:ment|ments|ing)?\b",
    r"\btherap(?:y|ies)\b",
    r"\bmedication(?:s)?\b",
    r"\bdrug(?:s)?\b",
    r"\bside effect(?:s)?\b",
    r"\bmonitor(?:ing)?\b",
    r"\btest(?:s)?\b",
    r"\blab(?:s)?\b",
    r"\bblood pressure\b",
    r"\bhypertension\b",
    r"\bcancer\b",
    r"\btumou?r\b",
    r"\bstone(?:s)?\b",
    r"\bcyst(?:s)?\b",
    r"\binfection(?:s)?\b",
    r"\buti(?:s)?\b",
    r"\btransplant(?:ation)?\b",
    r"\bdialysis\b",
    r"\bfailure\b",
    r"\bfunction\b",
    r"\bdamage\b",
    r"\bprogress(?:ion)?\b",
    r"\bstage(?:s)?\b",
    r"\bgenetic(?:s)?\b",
    r"\bmutation(?:s)?\b",
    r"\bcreatinin(?:e)?\b",
    r"\begfr\b",
    r"\bgfr\b",
    r"\bprotein(?: in urine)?\b",
    r"\balbumin(?: in urine)?\b",
    r"\bhematuria\b",
    r"\bdonat(?:e|ion|or)\b",
)

CLEAR_OFF_TOPIC_PATTERNS = (
    r"\bkidney bean(?:s)?\b",
    r"\bkidney pie\b",
    r"\bkidney stew\b",
    r"\bkidney punch(?:es|ed|ing)?\b",
)


def _normalize_text(text: Optional[str]) -> str:
    """Normalize whitespace and casing for lightweight rule checks."""
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def _matches_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    """Return True when any regex pattern matches the text."""
    return any(re.search(pattern, text) for pattern in patterns)


def _build_topic_result(
    *,
    on_topic: bool,
    reason: str,
    classification: str,
    source: str,
) -> Dict[str, Any]:
    """Build a standardized topic-classification payload."""
    return {
        "on_topic": on_topic,
        "reason": reason,
        "classification": classification,
        "source": source,
    }


def classify_question_locally(query: str) -> Optional[Dict[str, Any]]:
    """
    Use deterministic scope rules for obviously in-scope and out-of-scope queries.

    Returns None when the query needs LLM disambiguation.
    """
    normalized_query = _normalize_text(query)

    if not normalized_query:
        return _build_topic_result(
            on_topic=False,
            reason="Empty query is outside PKD/kidney-disease scope",
            classification="OFF-TOPIC: empty query",
            source="heuristic",
        )

    if _matches_any_pattern(normalized_query, CLEAR_OFF_TOPIC_PATTERNS):
        return _build_topic_result(
            on_topic=False,
            reason="Question uses kidney terminology in a non-medical context",
            classification="OFF-TOPIC: non-medical kidney usage",
            source="heuristic",
        )

    if _matches_any_pattern(normalized_query, DIRECT_ON_TOPIC_PATTERNS):
        return _build_topic_result(
            on_topic=True,
            reason="Question clearly references PKD/ADPKD or kidney-disease care",
            classification="ON-TOPIC: direct kidney-domain keyword match",
            source="heuristic",
        )

    has_kidney_anchor = _matches_any_pattern(normalized_query, KIDNEY_ANCHOR_PATTERNS)
    has_medical_context = _matches_any_pattern(normalized_query, KIDNEY_MEDICAL_CONTEXT_PATTERNS)

    if has_kidney_anchor and has_medical_context:
        return _build_topic_result(
            on_topic=True,
            reason="Question references kidneys in a medical or disease context",
            classification="ON-TOPIC: kidney anchor plus medical context",
            source="heuristic",
        )

    if has_kidney_anchor:
        return _build_topic_result(
            on_topic=False,
            reason="Question mentions kidneys without a kidney-disease or nephrology-care context",
            classification="OFF-TOPIC: kidney mention without disease or care context",
            source="heuristic",
        )

    return None


def is_off_topic_response(response: Optional[str]) -> bool:
    """
    Check whether a model response matches the standardized off-topic refusal.
    """
    if not response:
        return False
    normalized_response = _normalize_text(response.strip(' "\''))
    normalized_message = _normalize_text(OFF_TOPIC_MESSAGE)
    return normalized_response == normalized_message


def get_system_prompt_with_guardrails(base_prompt: str) -> str:
    """
    Add guardrails to any system prompt to enforce topic restrictions
    
    Args:
        base_prompt: Base system prompt for the assistant
        
    Returns:
        Enhanced system prompt with guardrails
    """
    guardrail = f"""

CRITICAL GUARDRAILS:
- If the user's question is NOT related to {TOPIC_SCOPE_DESCRIPTION}, you MUST respond with EXACTLY this message:

"{OFF_TOPIC_MESSAGE}"

- DO NOT attempt to answer questions about: unrelated medical conditions, general health advice unrelated to kidneys, non-medical topics, current events, or personal advice unrelated to kidney-disease management
- Questions that only mention "kidney" in a non-medical or non-disease context are OFF-TOPIC
- ALWAYS stay within the scope of PKD/ADPKD and kidney-disease care
- If unsure whether a question is related, err on the side of caution and provide the off-topic message"""
    
    return base_prompt + guardrail


async def is_question_on_topic(
    query: str,
    openai_service: OpenAIService
) -> Dict[str, Any]:
    """
    Pre-check if a question is related to PKD/kidney disease before processing.
    Uses deterministic local rules first, then falls back to an LLM classifier.
    
    Args:
        query: User's question
        openai_service: OpenAI service instance
        
    Returns:
        Dictionary with on_topic (bool) and reason (str)
    """
    heuristic_result = classify_question_locally(query)
    if heuristic_result is not None:
        return heuristic_result

    classifier_prompt = f"""You are a question classifier for a Polycystic Kidney Disease (PKD) medical chatbot.

Your task: Determine if a user's question is related to {TOPIC_SCOPE_DESCRIPTION}.

ON-TOPIC questions include:
- PKD/ADPKD symptoms, diagnosis, treatment, and progression
- Kidney disease, renal disease, nephrology care, dialysis, transplant, or kidney-focused lab questions
- Medications for PKD or kidney disease (tolvaptan, blood pressure meds, etc.)
- Kidney-related complications like hypertension, kidney stones, or UTIs when discussed as renal care
- Named kidney diseases or nephrology conditions even if "kidney" is not explicitly stated

OFF-TOPIC questions include:
- Unrelated medical conditions with no kidney-disease context
- General health questions unrelated to kidneys
- General anatomy or biology questions that are not about kidney disease or renal medical care
- Non-medical uses of the word "kidney" (food, slang, etc.)
- Non-medical topics (sports, entertainment, politics, etc.)
- Personal advice unrelated to kidney-disease management
- Current events, news, weather

Respond in exactly one line using one of these formats:
ON-TOPIC: brief reason
OFF-TOPIC: brief reason"""
    
    user_message = f"Question: {query}\n\nClassification:"
    
    try:
        response = openai_service.get_chat_completion(
            system_prompt=classifier_prompt,
            user_message=user_message,
            temperature=0.1,  # Very low temperature for consistent classification
            max_tokens=1500  # reasoning model floor: tiny budgets return empty content
        )
        
        response = response.strip()
        normalized = response.upper()

        if normalized.startswith("OFF-TOPIC"):
            return _build_topic_result(
                on_topic=False,
                reason="Question is not related to PKD/kidney-disease scope",
                classification=response,
                source="llm",
            )
        if normalized.startswith("ON-TOPIC"):
            return _build_topic_result(
                on_topic=True,
                reason="Question is related to PKD/kidney-disease scope",
                classification=response,
                source="llm",
            )
        return _build_topic_result(
            on_topic=False,
            reason="Classifier returned an unexpected format; treating as off-topic",
            classification=response,
            source="llm",
        )
            
    except Exception as e:
        logger.error(f"Error in topic classification: {e}")
        return {
            "on_topic": False,
            "reason": "Topic classification failed, treating the question as off-topic",
            "classification": "OFF-TOPIC: classifier failure",
            "error": str(e),
            "source": "llm",
        }


def get_off_topic_response() -> Dict[str, Any]:
    """
    Generate standardized off-topic response
    
    Returns:
        Response dictionary with off-topic message
    """
    return {
        "status": "success",
        "response": OFF_TOPIC_MESSAGE,
        "sources": [],
        "query": "",
        "off_topic": True
    }


# Pre-defined system prompts for different services

STANDARD_RAG_SYSTEM_PROMPT = get_system_prompt_with_guardrails(
    """You are a helpful medical AI assistant specialized in Polycystic Kidney Disease (PKD), ADPKD, and kidney-disease topics. 
Provide accurate, evidence-based information based on the provided medical literature.

When answering questions:
- Be clear, concise, and compassionate
- Cite evidence from the provided sources when making claims
- Acknowledge uncertainty when information is limited
- Remind users to consult their healthcare provider for medical decisions
- Use professional medical terminology but explain complex concepts"""
)

COT_RAG_SYSTEM_PROMPT = get_system_prompt_with_guardrails(
    """You are a helpful medical AI assistant specialized in Polycystic Kidney Disease (PKD), ADPKD, and kidney-disease topics.

You've reasoned through a complex question step-by-step. Now synthesize your findings into a comprehensive, well-structured answer.

Structure your response:
1. Start with a direct answer to the main question
2. Provide detailed explanation with evidence from your reasoning
3. Include relevant clinical implications or recommendations if applicable
4. End with any important caveats or notes

Write clearly and professionally, as if explaining to a patient or medical student."""
)

STEPBACK_SYSTEM_PROMPT = get_system_prompt_with_guardrails(
    """You are a helpful medical AI assistant specialized in Polycystic Kidney Disease (PKD), ADPKD, and kidney-disease topics.

You have been provided with medical literature from two types of searches:
1. Specific search results directly related to the user's question
2. Broader conceptual search results providing foundational knowledge

Use both types of information to provide a comprehensive, accurate answer that:
- Directly addresses the specific question
- Provides relevant background and context from broader principles
- Cites evidence from the sources when making claims
- Acknowledges uncertainty when information is limited

Be clear, concise, and helpful. If the question involves medical decisions, remind users to consult their healthcare provider."""
)
