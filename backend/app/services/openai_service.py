"""
OpenAI Service Module
Handles all interactions with OpenAI APIs (embeddings and chat completions)
"""

import logging
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from utils.openai_utils import get_openai_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIService:
    """
    Service for interacting with OpenAI APIs
    Handles embeddings generation and chat completions
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize OpenAI service
        
        Args:
            embedding_model: Model for embeddings (default: text-embedding-3-small)
            chat_model: Model for chat completions (default: gpt-4o)
            max_retries: Maximum retry attempts for API calls
            retry_delay: Delay in seconds between retries
        """
        api_key = get_openai_api_key()
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"OpenAI Service initialized with embedding_model={embedding_model}, chat_model={chat_model}")
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry logic
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (RateLimitError, APIConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"API error on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries exceeded: {e}")
                    raise
            except APIError as e:
                logger.error(f"API error: {e}")
                raise
        
        if last_exception:
            raise last_exception
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text string
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding (1536 dimensions)
        """
        # Clean text
        text = text.replace("\n", " ")
        
        def _call():
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        
        return self._retry_with_backoff(_call)
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Get embeddings for multiple texts (batch processing for efficiency)
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call (max 100)
            
        Returns:
            List of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Clean texts
            batch = [text.replace("\n", " ") for text in batch]
            
            def _call():
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                # Sort by index to maintain order
                embeddings = sorted(response.data, key=lambda x: x.index)
                return [e.embedding for e in embeddings]
            
            batch_embeddings = self._retry_with_backoff(_call)
            all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Embedded batch {i // batch_size + 1}: {len(batch)} texts")
        
        return all_embeddings
    
    def get_chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Get chat completion from OpenAI
        
        Args:
            system_prompt: System prompt for the model
            user_message: User's message/query
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response text
        """
        def _call():
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        return self._retry_with_backoff(_call)
    
    def get_chat_completion_with_context(
        self,
        context: str,
        user_query: str,
        system_instruction: str = "You are a helpful medical AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Get chat completion with context (RAG pattern)
        
        Args:
            context: Retrieved context from knowledge base
            user_query: User's question
            system_instruction: System instruction for the model
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response text
        """
        system_prompt = f"""{system_instruction}

Use the following context to answer the user's question. If the context doesn't contain relevant information, say so explicitly.

CONTEXT:
{context}

Instructions:
1. Answer based on the provided context
2. Cite sources if available
3. Be accurate and helpful
"""
        
        return self.get_chat_completion(
            system_prompt=system_prompt,
            user_message=user_query,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def validate_connection(self) -> bool:
        """
        Validate connection to OpenAI API
        
        Returns:
            True if connection is valid
        """
        try:
            # Try a simple embedding call
            self.get_embedding("test")
            logger.info("OpenAI API connection validated")
            return True
        except Exception as e:
            logger.error(f"OpenAI API connection failed: {e}")
            return False
