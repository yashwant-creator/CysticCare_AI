"""
Stepback Query Decomposition Agent
Improves RAG accuracy by generating broader, conceptual queries alongside specific ones
"""

import logging
from typing import Dict, Any, List, Tuple
from services.openai_service import OpenAIService
from services.openai_rag_init import search_knowledge_base
from utils.prompt_guard import STEPBACK_SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        system_prompt = """You are an expert at identifying underlying medical concepts and principles.

Your task: Given a specific medical question about Polycystic Kidney Disease (PKD), generate a broader, more general question that captures the underlying concept or principle.

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
            
            stepback_query = stepback_query.strip()
            
            # Remove any quotes or extra formatting
            stepback_query = stepback_query.strip('"\'')
            
            logger.info(f"Generated stepback query: '{stepback_query}'")
            return stepback_query
            
        except Exception as e:
            logger.error(f"Error generating stepback query: {e}")
            # Fallback: return a generic broader query
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
            # Generate stepback query
            stepback_query = await self.generate_stepback_query(original_query)
            
            # Retrieve for both queries in parallel (conceptually)
            logger.info(f"Retrieving for original query: '{original_query}'")
            original_results = await search_knowledge_base(original_query, top_k)
            
            logger.info(f"Retrieving for stepback query: '{stepback_query}'")
            stepback_results = await search_knowledge_base(stepback_query, top_k)
            
            # Combine results
            combined = self._combine_results(
                original_results,
                stepback_results,
                original_query,
                stepback_query
            )
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in stepback retrieval: {e}")
            # Fallback to original query only
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
        # Extract documents
        original_docs = original_results.get("results", [])
        stepback_docs = stepback_results.get("results", [])
        
        # Deduplicate by document ID/source
        seen_ids = set()
        combined_docs = []
        
        # Prioritize original query results (more specific)
        for doc in original_docs:
            doc_id = doc.get("metadata", {}).get("file", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                doc["retrieval_source"] = "original"
                combined_docs.append(doc)
        
        # Add stepback results that aren't duplicates
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
            # Retrieve with stepback
            retrieval_results = await self.retrieve_with_stepback(query, top_k)
            
            # Build context from all retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieval_results["results"][:top_k * 2]):
                metadata = doc.get("metadata", {})
                content = doc.get("document", "")  # Fixed: use 'document' not 'content'
                source_type = doc.get("retrieval_source", "unknown")
                display_name = metadata.get("display_name", f"Source {i+1}")
                
                context_parts.append(
                    f"[Source {i+1} - {source_type}: {display_name}]\n"
                    f"{content}\n"
                )
            
            context = "\n".join(context_parts)
            
            # Generate answer with guardrails
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
            
            # Format sources for the main API endpoint
            formatted_sources = []
            for i, doc in enumerate(retrieval_results["results"][:top_k * 2]):
                metadata = doc.get("metadata", {})
                formatted_sources.append({
                    "index": i + 1,
                    "title": metadata.get("title", "Unknown"),
                    "author": metadata.get("author", "Unknown Author"),
                    "year": metadata.get("year", "Unknown"),
                    "file": metadata.get("file_name", "unknown.pdf"),
                    "citation": metadata.get("citation", ""),
                    "display_name": metadata.get("display_name", ""),
                    "relevance_score": doc.get("relevance_score", 0.0)
                })
            
            return {
                "status": "success",
                "response": answer,
                "original_query": query,
                "stepback_query": retrieval_results.get("stepback_query", ""),
                "sources": formatted_sources,
                "retrieved_chunks": retrieval_results.get("results", []),  # raw chunks for validation agent
                "retrieval_metadata": {
                    "original_count": retrieval_results.get("original_count", 0),
                    "stepback_count": retrieval_results.get("stepback_count", 0),
                    "combined_count": retrieval_results.get("combined_count", 0)
                }
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
