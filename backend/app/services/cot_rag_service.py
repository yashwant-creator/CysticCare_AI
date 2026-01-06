"""
Chain of Thought RAG Service
Implements multi-step reasoning for complex medical queries
"""

import logging
from typing import Dict, Any, List, Optional
from services.openai_service import OpenAIService
from services.openai_rag_init import search_knowledge_base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChainOfThoughtRAG:
    """
    Chain of Thought RAG implementation that:
    1. Decomposes complex queries into sub-questions
    2. Retrieves relevant documents for each step
    3. Reasons through findings step-by-step
    4. Synthesizes a final answer with clear reasoning
    """
    
    def __init__(self, openai_service: OpenAIService):
        """
        Initialize CoT RAG service
        
        Args:
            openai_service: OpenAI service instance for LLM calls
        """
        self.openai_service = openai_service
    
    async def decompose_query(self, query: str) -> List[str]:
        """
        Break down a complex query into logical sub-questions
        
        Args:
            query: Original user query
            
        Returns:
            List of sub-questions to answer sequentially
        """
        system_prompt = """You are an expert at breaking down complex medical questions into logical steps.

Given a user's question about Polycystic Kidney Disease (PKD), identify the key sub-questions that need to be answered to fully address their query.

Output ONLY a numbered list of sub-questions, one per line. Keep it concise (2-5 sub-questions maximum).

Example:
User: "How does PKD progress and what treatments can slow it down?"
Output:
1. What is the typical progression pattern of PKD?
2. What factors influence PKD progression rate?
3. What treatment options are available for PKD?
4. Which treatments have been shown to slow PKD progression?"""
        
        user_message = f"User question: {query}\n\nSub-questions:"
        
        try:
            response = self.openai_service.get_chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.3,  # Lower temperature for more focused decomposition
                max_tokens=500
            )
            
            # Parse numbered list
            sub_questions = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets
                    cleaned = line.lstrip('0123456789.-•) ').strip()
                    if cleaned:
                        sub_questions.append(cleaned)
            
            logger.info(f"Decomposed query into {len(sub_questions)} sub-questions")
            return sub_questions if sub_questions else [query]  # Fallback to original
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            return [query]  # Fallback to original query
    
    async def retrieve_for_step(
        self,
        sub_question: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents for a specific sub-question
        
        Args:
            sub_question: Sub-question to search for
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        try:
            search_results = await search_knowledge_base(sub_question, top_k)
            return search_results
        except Exception as e:
            logger.error(f"Error retrieving for sub-question: {e}")
            return {
                "status": "error",
                "message": str(e),
                "results": []
            }
    
    async def reason_through_step(
        self,
        sub_question: str,
        context: str,
        previous_findings: List[str]
    ) -> str:
        """
        Reason through a single step using retrieved context
        
        Args:
            sub_question: Current sub-question
            context: Retrieved context documents
            previous_findings: Findings from previous steps
            
        Returns:
            Reasoning and findings for this step
        """
        previous_context = ""
        if previous_findings:
            previous_context = "\n\nPREVIOUS FINDINGS:\n" + "\n".join(
                f"Step {i+1}: {finding}" 
                for i, finding in enumerate(previous_findings)
            )
        
        system_prompt = f"""You are a medical AI assistant reasoning through a complex question step-by-step.

Your task: Answer the current sub-question using the provided medical literature context. Be specific and cite what you find in the sources.

Format your response as:
**Finding:** [Your clear, evidence-based finding]
**Evidence:** [Brief explanation of what the sources say]

Keep it concise but informative (2-4 sentences).{previous_context}"""
        
        user_message = f"""CURRENT CONTEXT:
{context}

SUB-QUESTION: {sub_question}

Your reasoning:"""
        
        try:
            response = self.openai_service.get_chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.5,
                max_tokens=400
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error reasoning through step: {e}")
            return f"Unable to reason through this step: {str(e)}"
    
    async def synthesize_final_answer(
        self,
        original_query: str,
        reasoning_steps: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Synthesize final answer from all reasoning steps
        
        Args:
            original_query: Original user question
            reasoning_steps: List of reasoning steps with findings
            temperature: Model temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Final synthesized answer
        """
        # Build reasoning chain
        chain_of_thought = []
        for i, step in enumerate(reasoning_steps):
            chain_of_thought.append(
                f"**Step {i+1}:** {step['sub_question']}\n{step['reasoning']}"
            )
        
        reasoning_summary = "\n\n".join(chain_of_thought)
        
        system_prompt = """You are a helpful medical AI assistant specialized in Polycystic Kidney Disease (PKD).

You've reasoned through a complex question step-by-step. Now synthesize your findings into a comprehensive, well-structured answer.

Structure your response:
1. Start with a direct answer to the main question
2. Provide detailed explanation with evidence from your reasoning
3. Include relevant clinical implications or recommendations if applicable
4. End with any important caveats or notes

Write clearly and professionally, as if explaining to a patient or medical student."""
        
        user_message = f"""ORIGINAL QUESTION: {original_query}

CHAIN OF THOUGHT REASONING:
{reasoning_summary}

Based on this step-by-step analysis, provide a comprehensive answer:"""
        
        try:
            response = self.openai_service.get_chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error synthesizing final answer: {e}")
            return "Unable to synthesize answer due to an error."
    
    async def get_cot_rag_response(
        self,
        query: str,
        top_k_per_step: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Full Chain of Thought RAG pipeline
        
        Args:
            query: User query
            top_k_per_step: Number of documents to retrieve per step
            temperature: Model temperature for final answer
            max_tokens: Maximum tokens in final response
            max_steps: Maximum number of reasoning steps
            
        Returns:
            Dictionary with final answer, reasoning chain, and sources
        """
        try:
            logger.info(f"Starting CoT RAG for query: {query[:100]}...")
            
            # Step 1: Decompose query into sub-questions
            sub_questions = await self.decompose_query(query)
            sub_questions = sub_questions[:max_steps]  # Limit steps
            logger.info(f"Query decomposed into {len(sub_questions)} steps")
            
            # Step 2: Process each sub-question
            reasoning_steps = []
            all_sources = []
            previous_findings = []
            
            for i, sub_question in enumerate(sub_questions):
                logger.info(f"Processing step {i+1}: {sub_question[:80]}...")
                
                # Retrieve relevant documents
                search_results = await self.retrieve_for_step(sub_question, top_k_per_step)
                
                if search_results["status"] != "success" or not search_results["results"]:
                    logger.warning(f"No results for step {i+1}")
                    continue
                
                # Format context
                context_parts = []
                for j, result in enumerate(search_results["results"]):
                    meta = result.get("metadata", {})
                    display_name = meta.get("display_name", f"Source {len(all_sources)+j+1}")
                    context_parts.append(f"[{display_name}]\n{result['document']}")
                    
                    # Track all sources
                    source_info = {
                        "index": len(all_sources) + j + 1,
                        "title": meta.get("title", "Unknown"),
                        "author": meta.get("author", "Unknown"),
                        "year": meta.get("year", "Unknown"),
                        "file": meta.get("file_name", "Unknown"),
                        "citation": meta.get("citation", ""),
                        "display_name": display_name,
                        "relevance_score": result.get("relevance_score", 0),
                        "step": i + 1
                    }
                    all_sources.append(source_info)
                
                context = "\n\n".join(context_parts)
                
                # Reason through this step
                reasoning = await self.reason_through_step(
                    sub_question,
                    context,
                    previous_findings
                )
                
                reasoning_steps.append({
                    "step": i + 1,
                    "sub_question": sub_question,
                    "reasoning": reasoning,
                    "sources_used": len(search_results["results"])
                })
                
                previous_findings.append(reasoning)
            
            if not reasoning_steps:
                logger.error("No reasoning steps completed")
                return {
                    "status": "error",
                    "message": "Unable to complete reasoning process",
                    "response": None,
                    "reasoning_chain": [],
                    "sources": []
                }
            
            # Step 3: Synthesize final answer
            logger.info("Synthesizing final answer...")
            final_answer = await self.synthesize_final_answer(
                query,
                reasoning_steps,
                temperature,
                max_tokens
            )
            
            # Deduplicate sources by file name
            unique_sources = {}
            for source in all_sources:
                key = source["file"]
                if key not in unique_sources or source["relevance_score"] > unique_sources[key]["relevance_score"]:
                    unique_sources[key] = source
            
            # Re-index sources
            final_sources = []
            for i, source in enumerate(sorted(unique_sources.values(), key=lambda x: -x["relevance_score"]), 1):
                source["index"] = i
                final_sources.append(source)
            
            logger.info(f"CoT RAG complete: {len(reasoning_steps)} steps, {len(final_sources)} unique sources")
            
            return {
                "status": "success",
                "response": final_answer,
                "reasoning_chain": reasoning_steps,
                "sources": final_sources,
                "query": query,
                "cot_enabled": True
            }
            
        except Exception as e:
            logger.error(f"Error in CoT RAG pipeline: {e}")
            return {
                "status": "error",
                "message": f"CoT RAG failed: {str(e)}",
                "response": None,
                "reasoning_chain": [],
                "sources": []
            }


def get_cot_rag_service(openai_service: OpenAIService) -> ChainOfThoughtRAG:
    """
    Factory function to get CoT RAG service instance
    
    Args:
        openai_service: OpenAI service instance
        
    Returns:
        ChainOfThoughtRAG instance
    """
    return ChainOfThoughtRAG(openai_service)
