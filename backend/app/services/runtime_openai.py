"""Bounded asynchronous OpenAI calls used by the production request path."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from .usage_metrics import OperationUsage, RequestUsage, usage_fields

logger = logging.getLogger(__name__)


class ProviderUnavailable(RuntimeError):
    def __init__(self, code: str = "provider_unavailable"):
        super().__init__(code)
        self.code = code


def _error_code(error: Exception) -> str:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        code = body.get("code")
        if not code and isinstance(body.get("error"), dict):
            code = body["error"].get("code")
        if code:
            return str(code)
    return error.__class__.__name__.lower()


def _retryable(error: Exception) -> bool:
    code = _error_code(error)
    if code in {"insufficient_quota", "invalid_api_key"}:
        return False
    return isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)) or (
        isinstance(error, APIStatusError) and error.status_code >= 500
    )


class RuntimeOpenAI:
    def __init__(self) -> None:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise ProviderUnavailable("openai_api_key_missing")
        self.client = AsyncOpenAI(api_key=key, max_retries=0, timeout=30.0)
        self.answer_model = os.getenv("BASE_CHAT_MODEL", "gpt-5.5").strip()
        self.helper_model = os.getenv("HELPER_CHAT_MODEL", "gpt-4o-mini").strip()
        self.embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ).strip()

    async def _call_with_retry(
        self,
        operation: str,
        model: str,
        tracker: RequestUsage,
        call: Any,
    ) -> Any:
        retries = 0
        started = time.perf_counter()
        while True:
            try:
                response = await call()
                return response, retries, started
            except (
                AuthenticationError,
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                APIStatusError,
            ) as error:
                code = _error_code(error)
                if retries >= 1 or not _retryable(error):
                    tracker.record(
                        OperationUsage(
                            operation=operation,
                            model=model,
                            latency_ms=round((time.perf_counter() - started) * 1000, 2),
                            retries=retries,
                            error_code=code,
                        )
                    )
                    raise ProviderUnavailable(code) from error
                retries += 1
                await asyncio.sleep(0.5)

    async def embeddings(
        self,
        texts: List[str],
        tracker: RequestUsage,
    ) -> List[List[float]]:
        """Embed one or more retrieval queries in a single provider request."""
        cleaned = [text.replace("\n", " ") for text in texts if text.strip()]
        if not cleaned:
            return []

        async def call() -> Any:
            return await self.client.embeddings.create(
                model=self.embedding_model,
                input=cleaned,
            )

        response, retries, started = await self._call_with_retry(
            "query_embedding", self.embedding_model, tracker, call
        )
        tracker.record(
            OperationUsage(
                operation="query_embedding",
                model=self.embedding_model,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                retries=retries,
                **usage_fields(response.usage),
            )
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    async def embedding(self, text: str, tracker: RequestUsage) -> list[float]:
        vectors = await self.embeddings([text], tracker)
        return vectors[0]

    async def plan_queries(
        self,
        query: str,
        mode: str,
        tracker: RequestUsage,
    ) -> List[str]:
        """Generate a tightly bounded retrieval plan with the helper model."""
        if mode not in {"stepback_lite", "cot_lite"}:
            return []
        maximum = 1 if mode == "stepback_lite" else 3
        minimum = 1 if mode == "stepback_lite" else 2
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": minimum,
                    "maxItems": maximum,
                    "items": {"type": "string"},
                },
            },
            "required": ["questions"],
        }
        if mode == "stepback_lite":
            instruction = (
                "Write exactly one broader kidney-disease retrieval question that "
                "captures the clinical principle behind the user's question."
            )
        else:
            instruction = (
                "Decompose the question into two or three distinct kidney-disease "
                "retrieval questions. Do not answer them."
            )

        async def call() -> Any:
            return await self.client.chat.completions.create(
                model=self.helper_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{instruction} Return retrieval questions only. "
                            "Preserve PKD/ADPKD context and avoid duplication."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                max_completion_tokens=int(
                    os.getenv("PLANNER_MAX_TOKENS", "180")
                ),
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{mode}_retrieval_plan",
                        "strict": True,
                        "schema": schema,
                    },
                },
                store=False,
                prompt_cache_key="cysticcare-retrieval-plan-v1",
            )

        operation = f"{mode}_planner"
        response, retries, started = await self._call_with_retry(
            operation, self.helper_model, tracker, call
        )
        choice = response.choices[0]
        tracker.record(
            OperationUsage(
                operation=operation,
                model=self.helper_model,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                retries=retries,
                finish_reason=choice.finish_reason,
                **usage_fields(response.usage),
            )
        )
        try:
            payload = json.loads(choice.message.content or "{}")
        except json.JSONDecodeError:
            payload = {}
        raw = payload.get("questions", [])
        if not isinstance(raw, list):
            return []
        questions = [str(item).strip() for item in raw if str(item).strip()]
        return questions[:maximum]

    def _answer_params(
        self,
        system_prompt: str,
        user_message: str,
        *,
        stream: bool,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.answer_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_completion_tokens": int(os.getenv("ANSWER_MAX_TOKENS", "1000")),
            "stream": stream,
            "store": False,
            "reasoning_effort": os.getenv("ANSWER_REASONING_EFFORT", "low"),
            "prompt_cache_key": "cysticcare-answer-v1",
        }
        if stream:
            params["stream_options"] = {"include_usage": True}
        return params

    async def answer(self, system_prompt: str, user_message: str, tracker: RequestUsage) -> str:
        async def call() -> Any:
            return await self.client.chat.completions.create(
                **self._answer_params(system_prompt, user_message, stream=False)
            )

        response, retries, started = await self._call_with_retry(
            "answer", self.answer_model, tracker, call
        )
        choice = response.choices[0]
        tracker.record(
            OperationUsage(
                operation="answer",
                model=self.answer_model,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                retries=retries,
                finish_reason=choice.finish_reason,
                **usage_fields(response.usage),
            )
        )
        return choice.message.content or ""

    async def stream_answer(
        self,
        system_prompt: str,
        user_message: str,
        tracker: RequestUsage,
    ) -> AsyncIterator[str]:
        async def call() -> Any:
            return await self.client.chat.completions.create(
                **self._answer_params(system_prompt, user_message, stream=True)
            )

        stream, retries, started = await self._call_with_retry(
            "answer", self.answer_model, tracker, call
        )
        response_usage: Any = None
        finish_reason: Optional[str] = None
        try:
            async for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    response_usage = chunk.usage
                if chunk.choices:
                    choice = chunk.choices[0]
                    finish_reason = choice.finish_reason or finish_reason
                    content = getattr(choice.delta, "content", None)
                    if content:
                        if tracker.time_to_first_token_ms is None:
                            tracker.time_to_first_token_ms = round(
                                (time.perf_counter() - tracker.started_at) * 1000,
                                2,
                            )
                        yield content
        finally:
            tracker.record(
                OperationUsage(
                    operation="answer",
                    model=self.answer_model,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    retries=retries,
                    finish_reason=finish_reason,
                    **usage_fields(response_usage),
                )
            )

    async def postprocess(
        self,
        query: str,
        answer: str,
        tracker: RequestUsage,
    ) -> Dict[str, Any]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
                "relevance_reason": {"type": "string"},
                "followup_questions": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "string"},
                },
            },
            "required": [
                "relevance_score",
                "relevance_reason",
                "followup_questions",
            ],
        }
        system = (
            "Evaluate whether a kidney-disease answer addresses its question and "
            "suggest exactly three concise, distinct kidney-related follow-up questions. "
            "Do not add medical facts beyond the supplied answer."
        )
        user = f"QUESTION:\n{query}\n\nANSWER:\n{answer}"

        async def call() -> Any:
            return await self.client.chat.completions.create(
                model=self.helper_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=int(os.getenv("HELPER_MAX_TOKENS", "350")),
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_postprocessing",
                        "strict": True,
                        "schema": schema,
                    },
                },
                store=False,
                prompt_cache_key="cysticcare-postprocess-v1",
            )

        response, retries, started = await self._call_with_retry(
            "postprocess", self.helper_model, tracker, call
        )
        choice = response.choices[0]
        tracker.record(
            OperationUsage(
                operation="postprocess",
                model=self.helper_model,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                retries=retries,
                finish_reason=choice.finish_reason,
                **usage_fields(response.usage),
            )
        )
        try:
            payload = json.loads(choice.message.content or "{}")
        except json.JSONDecodeError:
            payload = {}
        return payload


_runtime_service: Optional[RuntimeOpenAI] = None


def get_runtime_openai() -> RuntimeOpenAI:
    global _runtime_service
    if _runtime_service is None:
        _runtime_service = RuntimeOpenAI()
    return _runtime_service


def reset_runtime_openai_for_tests() -> None:
    global _runtime_service
    _runtime_service = None
