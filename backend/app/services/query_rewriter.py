"""
Query Rewriting Agent
Improves retrieval by rephrasing and expanding user queries
"""

import logging
from typing import List, Dict, Any
from .openai_service import OpenAIService

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Agent that rewrites user queries to improve retrieval accuracy
    """

    def __init__(self, openai_service: OpenAIService):
        """
        Initialize query rewriter with OpenAI service

        Args:
            openai_service: OpenAI service instance for LLM calls
        """
        self.openai_service = openai_service

        # Common medical abbreviations in PKD domain
        self.abbreviation_expansions = {
            "PKD": "Polycystic Kidney Disease",
            "ADPKD": "Autosomal Dominant Polycystic Kidney Disease",
            "ARPKD": "Autosomal Recessive Polycystic Kidney Disease",
            "TKV": "Total Kidney Volume",
            "eGFR": "estimated Glomerular Filtration Rate",
            "ESRD": "End-Stage Renal Disease",
            "CKD": "Chronic Kidney Disease",
            "HTN": "Hypertension",
            "ACE": "Angiotensin-Converting Enzyme",
            "ARB": "Angiotensin Receptor Blocker",
        }

    def expand_abbreviations(self, query: str) -> str:
        """
        Expand common medical abbreviations in the query

        Args:
            query: Original user query

        Returns:
            Query with expanded abbreviations
        """
        expanded_query = query
        for abbrev, full_form in self.abbreviation_expansions.items():
            # Match whole words only (case insensitive)
            import re
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded_query = re.sub(pattern, full_form, expanded_query, flags=re.IGNORECASE)

        return expanded_query

    async def rewrite_query_simple(self, query: str) -> str:
        """
        Simple query rewriting with abbreviation expansion
        Fast, no LLM call required

        Args:
            query: Original user query

        Returns:
            Rewritten query
        """
        # Expand abbreviations
        rewritten = self.expand_abbreviations(query)

        # Basic cleaning
        rewritten = rewritten.strip()

        logger.info(f"Simple rewrite: '{query}' → '{rewritten}'")
        return rewritten

    async def rewrite_query_advanced(self, query: str) -> Dict[str, Any]:
        """
        Advanced query rewriting using LLM to generate multiple variations
        More accurate but slower (requires LLM call)

        Args:
            query: Original user query

        Returns:
            Dictionary with original, rewritten, and variations
        """
        try:
            # First expand abbreviations
            expanded = self.expand_abbreviations(query)

            # Use LLM to generate medical-focused variations
            system_prompt = """You are a medical query expert specializing in Polycystic Kidney Disease (PKD).
If the question is asking anything other than PKD, ADPKD, or any kidney related disease, then don't answer.

Your task is to rewrite user queries to be more effective for searching medical literature.

Guidelines:
1. Make queries more specific and medical
2. Add relevant medical terminology
3. Expand casual language to formal medical terms
4. Keep the core question intent
5. Generate 2-3 variations that could find relevant papers

Respond in JSON format:
{
    "primary_query": "The best rewritten version",
    "variations": ["variation 1", "variation 2"],
    "medical_terms": ["term1", "term2"]
}"""

            user_prompt = f"""Original query: "{expanded}"

Rewrite this query to be more effective for searching PKD medical literature.
Consider:
- What specific medical concepts is the user asking about?
- What synonyms or related terms should be included?
- How would this question appear in a medical paper?"""

            # Call LLM for query rewriting
            response = self.openai_service.get_chat_completion(
                system_prompt=system_prompt,
                user_message=user_prompt,
                temperature=0.3,  # honored only for non-reasoning models
                max_tokens=1500  # reasoning model floor: tiny budgets return empty content
            )

            # Parse JSON response
            import json
            try:
                result = json.loads(response)

                logger.info(f"Advanced rewrite: '{query}' → '{result['primary_query']}'")
                logger.info(f"  Variations: {result.get('variations', [])}")

                return {
                    "original": query,
                    "expanded": expanded,
                    "primary": result.get("primary_query", expanded),
                    "variations": result.get("variations", []),
                    "medical_terms": result.get("medical_terms", [])
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse LLM response as JSON, using simple rewrite")
                return {
                    "original": query,
                    "expanded": expanded,
                    "primary": expanded,
                    "variations": [],
                    "medical_terms": []
                }

        except Exception as e:
            logger.error(f"Error in advanced query rewriting: {e}")
            # Fallback to simple rewrite
            return {
                "original": query,
                "expanded": expanded,
                "primary": expanded,
                "variations": [],
                "medical_terms": []
            }

    async def get_search_queries(
        self,
        query: str,
        use_advanced: bool = False,
        max_variations: int = 3
    ) -> List[str]:
        """
        Get list of queries to use for retrieval

        Args:
            query: Original user query
            use_advanced: Whether to use LLM-based rewriting
            max_variations: Maximum number of query variations to return

        Returns:
            List of queries to search with (includes original and variations)
        """
        if use_advanced:
            # Use LLM for advanced rewriting
            result = await self.rewrite_query_advanced(query)
            queries = [result["primary"]]

            # Add variations
            if result.get("variations"):
                queries.extend(result["variations"][:max_variations - 1])

            # Always include original as fallback
            if query not in queries:
                queries.append(query)

        else:
            # Simple rewriting (fast)
            rewritten = await self.rewrite_query_simple(query)
            queries = [rewritten]

            # Include original if different
            if rewritten != query:
                queries.append(query)

        return queries[:max_variations]

    async def generate_hyde_passage(self, query: str) -> str:
        """
        Generate a hypothetical medical literature passage answering the query.
        Used for SOTA Hypothetical Document Embeddings (HyDE) retrieval.

        Args:
            query: Original user query

        Returns:
            A hypothetical scientific text passage
        """
        try:
            expanded = self.expand_abbreviations(query)

            system_prompt = """You are an expert academic biomedical writer specializing in Polycystic Kidney Disease (PKD).
Your task is to write a single, highly technical, gold-standard paragraph from a scientific/medical research paper that directly and comprehensively answers the user's clinical question.

Guidelines:
- Write ONLY the paragraph itself. Do NOT include any introduction, conversational filler, or citation tags (e.g. do not write "Sure, here is a paragraph", do not write "[1]", etc.).
- Use formal academic language, advanced medical terminology, and precise scientific reasoning.
- Present factual, plausible medical evidence matching clinical literature on PKD/ADPKD."""

            user_prompt = f"""Write a hypothetical paper passage answering: "{expanded}\""""

            logger.info("Generating hypothetical document passage (HyDE)...")
            passage = self.openai_service.get_chat_completion(
                system_prompt=system_prompt,
                user_message=user_prompt,
                temperature=0.5,
                max_tokens=800
            )

            # Clean up the output in case LLM added extra surrounding quotes
            passage = passage.strip()
            if passage.startswith('"') and passage.endswith('"'):
                passage = passage[1:-1].strip()

            logger.info(f"HyDE passage successfully generated ({len(passage)} characters)")
            return passage

        except Exception as e:
            logger.error(f"Error generating HyDE passage: {e}")
            return query


# Singleton instance
_query_rewriter: QueryRewriter = None


def get_query_rewriter(openai_service: OpenAIService) -> QueryRewriter:
    """
    Get or create query rewriter singleton

    Args:
        openai_service: OpenAI service instance

    Returns:
        QueryRewriter instance
    """
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter(openai_service)
    return _query_rewriter
