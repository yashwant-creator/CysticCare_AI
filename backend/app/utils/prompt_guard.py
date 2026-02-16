"""
Prompt Guard - Centralized off-topic detection and guardrails for PKD chatbot
Ensures responses stay within the scope of PKD/kidney disease
"""

import logging
from typing import Dict, Any, Optional
from services.openai_service import OpenAIService

logger = logging.getLogger(__name__)

# Centralized restriction message
OFF_TOPIC_MESSAGE = """I appreciate your question, but I'm specifically designed to help with Polycystic Kidney Disease (PKD) and kidney health topics.

I can help you with:
• Understanding PKD symptoms, diagnosis, and progression
• Treatment options and medications for PKD
• Lifestyle modifications and dietary recommendations
• Managing complications and comorbidities
• General kidney health information

If you have questions about PKD or kidney disease, I'd be happy to help!"""


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
- If the user's question is NOT related to Polycystic Kidney Disease (PKD), kidney disease, renal health, or general kidney-related topics, you MUST respond with EXACTLY this message:

"{OFF_TOPIC_MESSAGE}"

- DO NOT attempt to answer questions about: unrelated medical conditions, general health advice unrelated to kidneys, non-medical topics, current events, personal advice unrelated to PKD management
- ALWAYS stay within the scope of PKD and kidney health
- If unsure whether a question is related, err on the side of caution and provide the off-topic message"""
    
    return base_prompt + guardrail


async def is_question_on_topic(
    query: str,
    openai_service: OpenAIService
) -> Dict[str, Any]:
    """
    Pre-check if a question is related to PKD/kidney disease before processing
    Uses LLM to classify the question
    
    Args:
        query: User's question
        openai_service: OpenAI service instance
        
    Returns:
        Dictionary with on_topic (bool) and reason (str)
    """
    classifier_prompt = """You are a question classifier for a Polycystic Kidney Disease (PKD) medical chatbot.

Your task: Determine if a user's question is related to PKD, kidney disease, or renal health.

ON-TOPIC questions include:
- PKD symptoms, diagnosis, treatment, progression
- Kidney function, kidney disease, renal health
- Medications for PKD (tolvaptan, blood pressure meds, etc.)
- Diet and lifestyle for kidney health
- Complications like hypertension, kidney stones, UTIs
- Genetic aspects of PKD
- Kidney transplant related to PKD
- General nephrology questions

OFF-TOPIC questions include:
- Unrelated medical conditions (diabetes, cancer, heart disease - unless directly related to PKD)
- General health questions unrelated to kidneys
- Non-medical topics (sports, entertainment, politics, etc.)
- Personal advice unrelated to PKD management
- Current events, news, weather

Respond with ONLY "ON-TOPIC" or "OFF-TOPIC" followed by a brief reason."""
    
    user_message = f"Question: {query}\n\nClassification:"
    
    try:
        response = openai_service.get_chat_completion(
            system_prompt=classifier_prompt,
            user_message=user_message,
            temperature=0.1,  # Very low temperature for consistent classification
            max_tokens=50
        )
        
        response = response.strip().upper()
        
        if "ON-TOPIC" in response:
            return {
                "on_topic": True,
                "reason": "Question is related to PKD/kidney health",
                "classification": response
            }
        else:
            return {
                "on_topic": False,
                "reason": "Question is not related to PKD/kidney health",
                "classification": response
            }
            
    except Exception as e:
        logger.error(f"Error in topic classification: {e}")
        # Fail open - allow the question to proceed (system prompt will catch it)
        return {
            "on_topic": True,
            "reason": "Classification failed, allowing through to system prompt",
            "error": str(e)
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
    """You are a helpful medical AI assistant specialized in Polycystic Kidney Disease (PKD). 
Provide accurate, evidence-based information based on the provided medical literature.

When answering questions:
- Be clear, concise, and compassionate
- Cite evidence from the provided sources when making claims
- Acknowledge uncertainty when information is limited
- Remind users to consult their healthcare provider for medical decisions
- Use professional medical terminology but explain complex concepts"""
)

COT_RAG_SYSTEM_PROMPT = get_system_prompt_with_guardrails(
    """You are a helpful medical AI assistant specialized in Polycystic Kidney Disease (PKD).

You've reasoned through a complex question step-by-step. Now synthesize your findings into a comprehensive, well-structured answer.

Structure your response:
1. Start with a direct answer to the main question
2. Provide detailed explanation with evidence from your reasoning
3. Include relevant clinical implications or recommendations if applicable
4. End with any important caveats or notes

Write clearly and professionally, as if explaining to a patient or medical student."""
)

STEPBACK_SYSTEM_PROMPT = get_system_prompt_with_guardrails(
    """You are a helpful medical AI assistant specialized in Polycystic Kidney Disease (PKD).

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
