"""
Adaptive Agent Selector
Automatically selects the best agent combination (Stepback, CoT, or Standard RAG)
based on query characteristics.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AdaptiveAgentSelector:
    """
    Analyzes user queries and automatically selects the most appropriate
    agent combination for optimal results.
    """
    
    # Keywords that indicate drug interaction questions
    DRUG_INTERACTION_KEYWORDS = [
        'interaction', 'drug', 'medication', 'take with', 'avoid', 
        'contraindication', 'side effect with', 'nsaid', 'contraindicated',
        'can i take', 'can i use', 'compatible with', 'together with',
        'combine', 'combined with', 'alongside', 'along with'
    ]
    
    # Keywords that indicate multi-part/comparison questions
    MULTISTEP_KEYWORDS = [
        'how and', 'what and', 'compare', 'versus', 'vs',
        'difference between', 'differences', 'comparison',
        'pros and cons', 'advantages and disadvantages',
        'benefits and risks', 'should i', 'which is better',
        'which treatment', 'best', 'steps to', 'process of',
        'stages of', 'progression', 'how does'
    ]
    
    # Keywords that benefit from stepback (foundational knowledge)
    STEPBACK_KEYWORDS = [
        'why', 'mechanism', 'mechanism of action', 'how work',
        'principle', 'concept', 'understand', 'explain',
        'general', 'broad', 'overview', 'background'
    ]
    
    @classmethod
    async def select_agent(cls, query: str) -> Dict[str, Any]:
        """
        Analyze query and return optimal agent configuration.
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with agent flags and reasoning
        """
        query_lower = query.lower()
        
        # Check for drug interaction patterns
        if cls._matches_keywords(query_lower, cls.DRUG_INTERACTION_KEYWORDS):
            logger.info("Query matched: Drug Interaction Pattern → Using Stepback")
            return {
                'use_stepback': True,
                'use_cot': False,
                'reason': 'drug_interaction',
                'recommendation': 'stepback'
            }
        
        # Check for multi-step/comparison patterns
        if cls._matches_keywords(query_lower, cls.MULTISTEP_KEYWORDS):
            logger.info("Query matched: Multi-step Pattern → Using CoT")
            return {
                'use_stepback': False,
                'use_cot': True,
                'reason': 'multistep_reasoning',
                'recommendation': 'cot'
            }
        
        # Check for stepback patterns (foundational knowledge requests)
        if cls._matches_keywords(query_lower, cls.STEPBACK_KEYWORDS):
            logger.info("Query matched: Foundational Knowledge Pattern → Using Stepback")
            return {
                'use_stepback': True,
                'use_cot': False,
                'reason': 'foundational_knowledge',
                'recommendation': 'stepback'
            }
        
        # Default: Standard RAG
        logger.info("Query: Standard Pattern → Using Standard RAG")
        return {
            'use_stepback': False,
            'use_cot': False,
            'reason': 'standard_query',
            'recommendation': 'standard_rag'
        }
    
    @staticmethod
    def _matches_keywords(text: str, keywords: list) -> bool:
        """
        Check if any keywords match in the text.
        
        Args:
            text: Text to search (already lowercase)
            keywords: List of keywords to match
            
        Returns:
            True if any keyword matches
        """
        return any(keyword in text for keyword in keywords)
