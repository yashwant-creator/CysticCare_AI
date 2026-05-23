"""
Stepback Query Decomposition Agent
Improves RAG accuracy by generating broader, conceptual queries alongside specific ones
"""

import logging
from typing import Dict, Any
from .openai_service import OpenAIService
from .openai_rag_init import EMPTY_KNOWLEDGE_BASE_MESSAGE, search_knowledge_base
from ..utils.refusal_utils import REFUSAL_MESSAGE, get_guardrail_response, is_refusal_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OFF_TOPIC_SENTINEL = "OFF_TOPIC"
OFF_TOPIC_MESSAGE = REFUSAL_MESSAGE


STEPBACK_SYSTEM_PROMPT = f"""You are a helpful medical AI assistant specialized in Polycystic Kidney Disease (PKD), ADPKD, and kidney-related disease topics.

If the question is asking anything other than PKD, ADPKD, or any kidney related disease, respond EXACTLY with:
"{REFUSAL_MESSAGE}"

If the retrieved context is not sufficient to answer the question, respond EXACTLY with:
"{REFUSAL_MESSAGE}"

You have been provided with medical literature from two types of searches:
1. Specific search results directly related to the user's question
2. Broader conceptual search results providing foundational knowledge

Use both types of information to provide a comprehensive, accurate answer that:
- Directly addresses the specific question
- Provides relevant background and context from broader principles
- Cites evidence from the sources when making claims
- Acknowledges uncertainty when information is limited

Be clear, concise, and helpful. If the question involves medical decisions, remind users to consult their healthcare provider."""


class StepbackAgent:
    """
    Implements Stepback Query Decomposition:
    1. Takes the original specific query
    2. Generates a broader, more conceptual "stepback" query
    3. Retrieves documents for both queries
    4. Combines context to provide comprehensive answers
    
    Example:
    Original: "Can I take ibuprofen with tolvaptan?"
    Stepback: "What are the general drug interactions and contraindications for vasopressin receptor antagonists?"
    """
    
    def __init__(self, openai_service: OpenAIService):
        """
        Initialize Stepback Agent
        
        Args:
            openai_service: OpenAI service instance for LLM calls
        """
        self.openai_service = openai_service
    
    async def generate_stepback_query(self, original_query: str) -> str:
        """
        Generate a broader, more conceptual stepback query from the original question.
        The stepback query captures underlying principles and general concepts.
        
        Args:
            original_query: User's specific question
            
        Returns:
            Broader, conceptual stepback query
        """
        system_prompt = """You are an expert at identifying underlying kidney-disease concepts and principles.

If the question is asking anything other than PKD, ADPKD, or any kidney related disease, output EXACTLY:
OFF_TOPIC

Your task: Given a specific medical question about Polycystic Kidney Disease (PKD), ADPKD, or another kidney-related disease, generate a broader, more general question that captures the underlying concept or principle.

If the original question is NOT related to PKD, kidney disease, renal health, nephrology, or kidney-related medical care, output EXACTLY:
OFF_TOPIC

The stepback question should:
- Be more general and conceptual than the original
- Focus on the medical class, mechanism, or category rather than specifics
- Help retrieve foundational medical knowledge
- Still be relevant to PKD/kidney disease when applicable

Examples:

Original: "Can I take ibuprofen with my tolvaptan medication?"
Stepback: "What are the drug interactions and contraindications for vasopressin receptor antagonists?"

Original: "Will eating less salt help my PKD?"
Stepback: "How does dietary sodium intake affect kidney function and disease progression in chronic kidney disease?"

Original: "What is my risk of developing kidney stones with PKD?"
Stepback: "What are the common complications and comorbidities associated with polycystic kidney disease?"

Original: "Should I avoid caffeine if I have PKD?"
Stepback: "What dietary factors and lifestyle modifications influence cyst growth and kidney function in PKD?"

Output ONLY the stepback question, nothing else."""

        user_message = f"Original question: {original_query}\n\nStepback question:"
        
        try:
            stepback_query = self.openai_service.get_chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.3,  # Lower temperature for consistent, focused stepback
                max_tokens=200
            )
            
            stepback_query = stepback_query.strip().strip('"\'')
            if stepback_query.upper() == OFF_TOPIC_SENTINEL:
                logger.info("Stepback generation blocked as off-topic")
                return OFF_TOPIC_SENTINEL
            
            logger.info(f"Generated stepback query: '{stepback_query}'")
            return stepback_query
            
        except Exception as e:
            logger.error(f"Error generating stepback query: {e}")
            return f"What are the general principles and mechanisms related to {original_query}?"
    
    async def retrieve_with_stepback(
        self,
        original_query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Perform stepback retrieval:
        1. Generate stepback query
        2. Retrieve documents for both original and stepback queries
        3. Combine and deduplicate results
        
        Args:
            original_query: User's original question
            top_k: Number of documents to retrieve per query
            
        Returns:
            Combined retrieval results with metadata
        """
        try:
            stepback_query = await self.generate_stepback_query(original_query)
            if stepback_query == OFF_TOPIC_SENTINEL:
                return {
                    "status": "success",
                    "original_query": original_query,
                    "stepback_query": "",
                    "results": [],
                    "original_count": 0,
                    "stepback_count": 0,
                    "combined_count": 0,
                    "off_topic": True,
                }
            
            logger.info(f"Retrieving for original query: '{original_query}'")
            original_results = await search_knowledge_base(original_query, top_k)
            
            logger.info(f"Retrieving for stepback query: '{stepback_query}'")
            stepback_results = await search_knowledge_base(stepback_query, top_k)
            
            combined = self._combine_results(
                original_results,
                stepback_results,
                original_query,
                stepback_query
            )
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in stepback retrieval: {e}")
            return await search_knowledge_base(original_query, top_k)
    
    def _combine_results(
        self,
        original_results: Dict[str, Any],
        stepback_results: Dict[str, Any],
        original_query: str,
        stepback_query: str
    ) -> Dict[str, Any]:
        """
        Combine and deduplicate results from original and stepback queries
        
        Args:
            original_results: Results from original query
            stepback_results: Results from stepback query
            original_query: Original query string
            stepback_query: Stepback query string
            
        Returns:
            Combined results dictionary
        """
        original_docs = original_results.get("results", [])
        stepback_docs = stepback_results.get("results", [])
        
        seen_ids = set()
        combined_docs = []
        
        for doc in original_docs:
            doc_id = doc.get("metadata", {}).get("file", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                doc["retrieval_source"] = "original"
                combined_docs.append(doc)
        
        for doc in stepback_docs:
            doc_id = doc.get("metadata", {}).get("file", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                doc["retrieval_source"] = "stepback"
                combined_docs.append(doc)
        
        logger.info(
            f"Combined {len(original_docs)} original + {len(stepback_docs)} stepback "
            f"= {len(combined_docs)} unique documents"
        )
        
        return {
            "status": "success",
            "original_query": original_query,
            "stepback_query": stepback_query,
            "results": combined_docs,
            "original_count": len(original_docs),
            "stepback_count": len(stepback_docs),
            "combined_count": len(combined_docs)
        }
    
    async def answer_with_stepback(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Full stepback RAG pipeline:
        1. Generate stepback query
        2. Retrieve with both queries
        3. Generate answer using combined context
        
        Args:
            query: User's question
            top_k: Documents to retrieve per query
            temperature: Model temperature for answer generation
            max_tokens: Maximum tokens in response
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        try:
            guardrail_response = get_guardrail_response(query)
            if guardrail_response is not None:
                logger.info("Stepback RAG answered query via guardrail before retrieval")
                return {
                    "status": "success",
                    "response": guardrail_response,
                    "sources": [],
                    "query": query,
                    "stepback_query": "",
                    "retrieved_chunks": [],
                    "off_topic": guardrail_response == OFF_TOPIC_MESSAGE,
                    "refused": guardrail_response == OFF_TOPIC_MESSAGE,
                }

            retrieval_results = await self.retrieve_with_stepback(query, top_k)
            if retrieval_results.get("off_topic"):
                return {
                    "status": "success",
                    "response": OFF_TOPIC_MESSAGE,
                    "sources": [],
                    "query": query,
                    "stepback_query": "",
                    "retrieved_chunks": [],
                    "off_topic": True,
                    "refused": True,
                }

            if not retrieval_results.get("results"):
                logger.warning("Stepback RAG found no retrieved documents; returning empty-knowledge-base response")
                return {
                    "status": "success",
                    "response": EMPTY_KNOWLEDGE_BASE_MESSAGE,
                    "sources": [],
                    "query": query,
                    "stepback_query": retrieval_results.get("stepback_query", ""),
                    "retrieved_chunks": [],
                    "off_topic": False,
                    "refused": True,
                }
            
            context_parts = []
            for i, doc in enumerate(retrieval_results["results"][:top_k * 2]):
                metadata = doc.get("metadata", {})
                content = doc.get("document", "")
                source_type = doc.get("retrieval_source", "unknown")
                display_name = metadata.get("display_name", f"Source {i+1}")
                
                context_parts.append(
                    f"[Source {i+1} - {source_type}: {display_name}]\n"
                    f"{content}\n"
                )
            
            context = "\n".join(context_parts)
            system_prompt = STEPBACK_SYSTEM_PROMPT

            user_message = f"""USER QUESTION: {query}

RETRIEVED CONTEXT:
{context}

Please provide a comprehensive answer based on the sources above."""

            answer = self.openai_service.get_chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if is_refusal_response(answer):
                logger.info("Stepback answer returned refusal response")
                return {
                    "status": "success",
                    "response": OFF_TOPIC_MESSAGE,
                    "sources": [],
                    "query": query,
                    "stepback_query": "",
                    "retrieved_chunks": [],
                    "off_topic": True,
                    "refused": True,
                }

            unique_sources = {}
            for doc in retrieval_results["results"][:top_k * 2]:
                metadata = doc.get("metadata", {})
                disp = metadata.get("display_name", "")
                file_key = disp or metadata.get("file_name") or metadata.get("citation") or f"chunk_{i}"
                score = doc.get("relevance_score", 0.0)
                if file_key not in unique_sources or score > unique_sources[file_key]["relevance_score"]:
                    unique_sources[file_key] = {
                        "title": metadata.get("title", "Unknown"),
                        "author": metadata.get("author", "Unknown Author"),
                        "year": metadata.get("year", "Unknown"),
                        "file": file_key,
                        "citation": metadata.get("citation", ""),
                        "display_name": metadata.get("display_name", ""),
                        "relevance_score": score,
                    }
            formatted_sources = [
                {"index": i + 1, **s}
                for i, s in enumerate(
                    sorted(unique_sources.values(), key=lambda x: -x["relevance_score"])
                )
            ]
            
            return {
                "status": "success",
                "response": answer,
                "original_query": query,
                "stepback_query": retrieval_results.get("stepback_query", ""),
                "sources": formatted_sources,
                "retrieved_chunks": retrieval_results.get("results", []),
                "retrieval_metadata": {
                    "original_count": retrieval_results.get("original_count", 0),
                    "stepback_count": retrieval_results.get("stepback_count", 0),
                    "combined_count": retrieval_results.get("combined_count", 0)
                },
                "refused": False,
            }
            
        except Exception as e:
            logger.error(f"Error in stepback answer generation: {e}")
            return {
                "status": "error",
                "response": f"Error generating answer: {str(e)}",
                "original_query": query,
                "stepback_query": "",
                "sources": [],
                "retrieval_metadata": {}
            }
