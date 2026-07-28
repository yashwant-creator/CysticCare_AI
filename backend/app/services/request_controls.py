"""App attestation, throttling, single-flight, and daily call-budget controls."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import date
import os
import time
from typing import AsyncIterator, Deque, Dict, Optional

from fastapi import HTTPException, Request
from .usage_metrics import RequestUsage


class RequestControls:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active_sessions: set[str] = set()
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)
        self._budget_day = date.today()
        self._openai_calls = 0
        self._openai_tokens = 0

    async def authorize(self, request: Request) -> str:
        token = request.headers.get("X-Firebase-AppCheck", "").strip()
        client_key = request.client.host if request.client else "unknown"
        if os.getenv("APP_CHECK_ENFORCED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            if not token:
                raise HTTPException(status_code=401, detail="App attestation is required")
            try:
                import firebase_admin
                from firebase_admin import app_check

                try:
                    firebase_admin.get_app()
                except ValueError:
                    firebase_admin.initialize_app()
                claims = app_check.verify_token(token)
                client_key = str(claims.get("app_id") or claims.get("sub") or client_key)
            except HTTPException:
                raise
            except Exception as error:
                raise HTTPException(status_code=401, detail="Invalid app attestation") from error

        now = time.monotonic()
        limit = max(1, int(os.getenv("CHAT_REQUESTS_PER_MINUTE", "10")))
        async with self._lock:
            history = self._requests[client_key]
            while history and history[0] < now - 60:
                history.popleft()
            if len(history) >= limit:
                raise HTTPException(status_code=429, detail="Chat request rate limit exceeded")
            history.append(now)
            self._refresh_budget_day()
            daily_limit = max(1, int(os.getenv("DAILY_OPENAI_CALL_LIMIT", "10000")))
            daily_token_limit = max(
                1, int(os.getenv("DAILY_OPENAI_TOKEN_LIMIT", "5000000"))
            )
            if (
                self._openai_calls + 3 > daily_limit
                or self._openai_tokens >= daily_token_limit
            ):
                raise HTTPException(
                    status_code=503,
                    detail="The daily AI service budget has been reached",
                )
        return client_key

    def _refresh_budget_day(self) -> None:
        today = date.today()
        if self._budget_day != today:
            self._budget_day = today
            self._openai_calls = 0
            self._openai_tokens = 0

    async def record_usage(self, tracker: RequestUsage) -> None:
        async with self._lock:
            self._refresh_budget_day()
            self._openai_calls += max(0, len(tracker.operations))
            self._openai_tokens += max(0, tracker.total_tokens)

    @asynccontextmanager
    async def single_flight(self, session_id: str) -> AsyncIterator[None]:
        async with self._lock:
            if session_id in self._active_sessions:
                raise HTTPException(
                    status_code=409,
                    detail="A chat request is already active for this session",
                )
            self._active_sessions.add(session_id)
        try:
            yield
        finally:
            async with self._lock:
                self._active_sessions.discard(session_id)


request_controls = RequestControls()
