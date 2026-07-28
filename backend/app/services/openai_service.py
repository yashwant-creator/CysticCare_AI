"""
OpenAI Service Module
Handles all interactions with OpenAI APIs (embeddings and chat completions)
"""

import base64
import json
import logging
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from ..utils.openai_utils import get_openai_api_key

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
        chat_model: str = "gpt-5.5",
        vision_model: str = "gpt-5.5",
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize OpenAI service

        Args:
            embedding_model: Model for embeddings (default: text-embedding-3-small)
            chat_model: Model for chat completions (default: gpt-5.5)
            vision_model: Vision-capable model for describing figures (default: gpt-5.5)
            max_retries: Maximum retry attempts for API calls
            retry_delay: Delay in seconds between retries
        """
        api_key = get_openai_api_key()
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.vision_model = vision_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Reasoning models (gpt-5.x, o1/o3/o4) only accept the default temperature
        # (1) and spend the token budget on hidden reasoning. Detect them so we can
        # omit unsupported params instead of getting a 400 on every call.
        self.is_reasoning_model = self._is_reasoning_model(chat_model)

        logger.info(f"OpenAI Service initialized with embedding_model={embedding_model}, chat_model={chat_model}, vision_model={vision_model}")

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Reasoning models forbid custom temperature and bill reasoning tokens
        against max_completion_tokens (e.g. gpt-5/gpt-5.5, o1, o3, o4)."""
        m = (model or "").lower()
        return m.startswith(("gpt-5", "o1", "o3", "o4"))

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
            params = {
                "model": self.chat_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_completion_tokens": max_tokens,
            }
            # Reasoning models reject any temperature other than the default (1).
            if not self.is_reasoning_model:
                params["temperature"] = temperature
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content

        return self._retry_with_backoff(_call)

    def get_structured_chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        schema_name: str,
        schema: Dict[str, Any],
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """Return schema-constrained JSON from a chat completion.

        Retrieval uses this for an LLM relevance judge. Keeping the helper here
        makes the calling code independent of the OpenAI SDK response shape and
        provides one controlled fallback for deployments whose configured model
        does not support Structured Outputs.
        """
        def _structured_call():
            params = {
                "model": self.chat_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_completion_tokens": max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
            if not self.is_reasoning_model:
                params["temperature"] = 0
            return self.client.chat.completions.create(**params)

        try:
            response = self._retry_with_backoff(_structured_call)
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Structured completion returned no content")
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("Structured completion did not return a JSON object")
            return parsed
        except Exception as structured_error:
            # Older model snapshots and compatible gateways can lack
            # response_format=json_schema. The reranker remains available with
            # validated JSON rather than turning retrieval into a hard failure.
            logger.warning(
                "Structured output unavailable; falling back to JSON prompt: %s",
                structured_error,
            )
            fallback_prompt = (
                f"{system_prompt}\n\nReturn ONLY a JSON object matching this schema: "
                f"{json.dumps(schema, separators=(',', ':'))}"
            )
            content = self.get_chat_completion(
                system_prompt=fallback_prompt,
                user_message=user_message,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("JSON fallback did not return an object")
            return parsed

    def stream_chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Stream chat completions chunk-by-chunk from OpenAI.

        Args:
            system_prompt: System prompt for the model
            user_message: User's message/query
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response

        Yields:
            Text chunks of the response
        """
        params = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_completion_tokens": max_tokens,
            "stream": True
        }
        if not self.is_reasoning_model:
            params["temperature"] = temperature

        def _call():
            return self.client.chat.completions.create(**params)

        response_stream = self._retry_with_backoff(_call)
        for chunk in response_stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

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

    # Sentinel returned by the vision model for non-informative images (logos, icons, etc.)
    NO_CONTENT_MARKER = "NO_CONTENT"

    def describe_image(
        self,
        image_bytes: bytes,
        image_ext: str = "png",
        caption: str = "",
        max_tokens: int = 2000,
    ) -> Optional[str]:
        """
        Describe a figure/image from a research paper using a vision-capable model.

        The description is written to be searchable as text (figure type, axes/units,
        trends, key values, and the finding it supports).

        Args:
            image_bytes: Raw image bytes extracted from the PDF
            image_ext: Image format/extension (e.g. "png", "jpeg")
            caption: The figure's caption from the paper, used as a grounding hint
            max_tokens: Completion-token budget. NOTE: reasoning models (e.g.
                gpt-5.5) spend this budget on reasoning tokens first, so it must
                be large enough to cover reasoning AND the visible description.
                Too small (e.g. 500) returns empty content -> every figure is
                wrongly dropped as NO_CONTENT.

        Returns:
            A text description, or None if the model judged the image non-informative.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/{image_ext};base64,{b64}"

        prompt = (
            "You are analyzing a figure from a Polycystic Kidney Disease (PKD/ADPKD) "
            "research paper. Describe the figure so it can be retrieved as text: state "
            "the figure type (chart, micrograph, MRI, diagram, table image, etc.), the "
            "variables/axes and their units, key trends, notable numeric values, and the "
            "clinical or scientific finding it supports. Be factual and concise. "
            "If the image is decorative (logo, icon, journal banner) or carries no "
            f"scientific information, reply with exactly: {self.NO_CONTENT_MARKER}"
        )
        if caption:
            prompt += f'\n\nThe figure\'s caption from the paper is: "{caption}"'

        def _call():
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content

        description = self._retry_with_backoff(_call)
        if not description:
            return None
        description = description.strip()
        if not description or self.NO_CONTENT_MARKER in description:
            return None
        return description

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
