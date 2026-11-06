"""FastAPI entrypoint: validation, rate limiting, error mapping, static UI."""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .agent import SearchAssistant
from .config import Settings, get_settings
from .errors import AssistantError
from .schemas import SESSION_ID_PATTERN, ChatRequest, ChatResponse

log = logging.getLogger("app")


class RateLimiter:
    """Per-client sliding window. Swap for Redis when running more than one instance."""

    def __init__(self, per_minute: int):
        self.per_minute, self.hits = per_minute, defaultdict(deque)

    def check(self, key: str) -> None:
        now, q = time.monotonic(), self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            raise HTTPException(429, "Too many requests. Please slow down.")
        q.append(now)
        if len(self.hits) > 10_000:  # bound memory
            self.hits = defaultdict(deque, {k: v for k, v in self.hits.items() if v})


def create_app(settings: Settings | None = None, assistant: SearchAssistant | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    limiter = RateLimiter(settings.rate_limit_per_min)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if assistant is None and not settings.google_api_key:
            log.warning("GOOGLE_API_KEY is not set; /api/chat will fail until it is configured.")
        app.state.assistant = assistant or SearchAssistant(settings)
        yield

    app = FastAPI(title="AI Search Assistant", lifespan=lifespan)

    @app.exception_handler(AssistantError)
    async def _assistant_error(_: Request, exc: AssistantError):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException):
        return JSONResponse({"error": {"code": str(exc.status_code), "message": exc.detail}}, status_code=exc.status_code)

    @app.middleware("http")
    async def _headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    def rate_limit(request: Request) -> None:
        limiter.check(request.client.host if request.client else "unknown")

    @app.get("/health")
    async def health():
        return {"status": "ok", "search_configured": bool(settings.google_api_key)}

    @app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
    async def chat(req: ChatRequest, request: Request):
        if len(req.message) > settings.max_message_chars:
            raise HTTPException(422, f"Message too long (max {settings.max_message_chars} characters).")
        return await request.app.state.assistant.chat(req.message, req.session_id)

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def reset(session_id: str, request: Request):
        if not re.match(SESSION_ID_PATTERN, session_id):
            raise HTTPException(422, "Invalid session id.")
        await request.app.state.assistant.reset(session_id)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return JSONResponse({"error": {"code": "422", "message": "Please enter a question (valid text, not empty)."}}, status_code=422)

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
    return app


app = create_app()
