"""
Follow-Up Question Agent
Generates contextually relevant follow-up questions based on the user's query
and the AI-generated response, using OpenAI ChatGPT.
"""

import logging
import json
from typing import List, Dict, Any
from .openai_service import OpenAIService
from ..utils.refusal_utils import is_refusal_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FOLLOWUP_SYSTEM_PROMPT = """You are a helpful medical AI assistant specializing in PKD, ADPKD, and related kidney diseases.

If the question is asking anything other than PKD, ADPKD, or any kidney related disease, then don't answer.

Your task: Given a user's question and the AI-generated response, generate 3 relevant follow-up questions that the user might naturally want to ask next.

Guidelines for follow-up questions:
- They should be directly related to the topic discussed in the question and response
- They should explore deeper, adjacent, or complementary aspects of the topic
- They should be specific and actionable — not vague or overly broad
- They should be phrased as natural questions a patient, caregiver, or researcher would ask
- They MUST stay within the domain of PKD, kidney disease, and related medical topics
- Each question should be distinct and cover a different angle

Output format: Return ONLY a valid JSON array of exactly 3 strings. No other text, no markdown, no explanation.
Example: ["What are the early warning signs of PKD progression?", "How often should I get kidney function tests?", "Are there dietary changes that can slow cyst growth?"]
"""


class FollowUpAgent:
    """
    Generates follow-up question suggestions using OpenAI ChatGPT.
    Takes the user's original question and the generated response as context
    to produce relevant, natural follow-up questions.
    """

    def __init__(self, openai_service: OpenAIService):
        """
        Initialize Follow-Up Agent.

        Args:
            openai_service: OpenAI service instance for LLM calls
        """
        self.openai_service = openai_service

    async def generate_followup_questions(
        self,
        query: str,
        response: str,
        num_questions: int = 3,
        temperature: float = 0.8,
    ) -> List[str]:
        """
        Generate follow-up questions based on the user query and AI response.

        Args:
            query: The user's original question
            response: The AI-generated response to the query
            num_questions: Number of follow-up questions to generate (default 3)
            temperature: Sampling temperature for creativity (default 0.8)

        Returns:
            List of follow-up question strings
        """
        user_message = (
            f"User's question:\n{query}\n\n"
            f"AI response:\n{response}\n\n"
            f"Generate {num_questions} follow-up questions the user might want to ask next."
        )

        try:
            if is_refusal_response(response):
                logger.info("Skipping follow-up generation for off-topic refusal")
                return []

            raw_output = self.openai_service.get_chat_completion(
                system_prompt=FOLLOWUP_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=temperature,
                max_tokens=500,
            )

            # Parse JSON array from the response
            followup_questions = self._parse_questions(raw_output, num_questions)
            logger.info(f"Generated {len(followup_questions)} follow-up questions")
            return followup_questions

        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return self._fallback_questions()

    def _parse_questions(self, raw_output: str, expected: int) -> List[str]:
        """
        Parse the LLM output into a list of question strings.

        Args:
            raw_output: Raw text from OpenAI
            expected: Expected number of questions

        Returns:
            List of question strings
        """
        # Strip markdown code fences if present
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag)
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            questions = json.loads(cleaned)
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                return questions[:expected]
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from follow-up generation, attempting line split")

        # Fallback: split by newlines and filter
        lines = [
            line.strip().lstrip("0123456789.-) ").strip('"').strip("'")
            for line in raw_output.strip().splitlines()
            if line.strip() and "?" in line
        ]
        if lines:
            return lines[:expected]

        return self._fallback_questions()

    @staticmethod
    def _fallback_questions() -> List[str]:
        """Return generic kidney-disease follow-up questions as a fallback."""
        return [
            "How is this kidney condition typically monitored over time?",
            "What treatment options are commonly considered for this condition?",
            "Which symptoms or warning signs should prompt medical follow-up?",
        ]
