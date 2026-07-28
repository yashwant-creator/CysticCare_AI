"""Request-scoped OpenAI usage and latency accounting without prompt logging."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OperationUsage:
    operation: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    retries: int = 0
    finish_reason: Optional[str] = None
    error_code: Optional[str] = None


@dataclass
class RequestUsage:
    request_id: str
    session_id: str
    started_at: float = field(default_factory=time.perf_counter)
    operations: List[OperationUsage] = field(default_factory=list)
    retrieval_ms: float = 0.0
    time_to_first_token_ms: Optional[float] = None
    disconnected: bool = False

    def record(self, usage: OperationUsage) -> None:
        self.operations.append(usage)
        logger.info(
            "openai_operation %s",
            json.dumps(
                {
                    "request_id": self.request_id,
                    **usage.__dict__,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    @property
    def total_tokens(self) -> int:
        return sum(item.total_tokens for item in self.operations)

    def finish(self) -> Dict[str, Any]:
        payload = {
            "request_id": self.request_id,
            "session_id_hash": self.session_id,
            "operation_count": len(self.operations),
            "total_tokens": self.total_tokens,
            "selected_mode": "single_pass_rag",
            "retrieval_ms": round(self.retrieval_ms, 2),
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "completion_ms": round((time.perf_counter() - self.started_at) * 1000, 2),
            "disconnected": self.disconnected,
        }
        logger.info(
            "chat_request_summary %s",
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )
        if len(self.operations) > 3:
            logger.error(
                "openai_operation_budget_exceeded request_id=%s count=%s",
                self.request_id,
                len(self.operations),
            )
        if self.total_tokens > 10_000:
            logger.error(
                "openai_token_budget_exceeded request_id=%s tokens=%s",
                self.request_id,
                self.total_tokens,
            )
        if self.time_to_first_token_ms is not None and self.time_to_first_token_ms > 5_000:
            logger.warning(
                "answer_ttft_slo_exceeded request_id=%s ttft_ms=%s",
                self.request_id,
                self.time_to_first_token_ms,
            )
        if payload["completion_ms"] > 20_000:
            logger.warning(
                "answer_completion_slo_exceeded request_id=%s completion_ms=%s",
                self.request_id,
                payload["completion_ms"],
            )
        return payload


def usage_fields(response_usage: Any) -> Dict[str, int]:
    if response_usage is None:
        return {}
    prompt_details = getattr(response_usage, "prompt_tokens_details", None)
    completion_details = getattr(response_usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": int(getattr(response_usage, "prompt_tokens", 0) or 0),
        "cached_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
        "completion_tokens": int(getattr(response_usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(completion_details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(getattr(response_usage, "total_tokens", 0) or 0),
    }
