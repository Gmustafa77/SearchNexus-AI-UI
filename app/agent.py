"""Agent orchestration: ADK agent, sessions, timeouts, retries."""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import OrderedDict

import httpx
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from .config import Settings
from .errors import AssistantError
from .responses import build_response
from .schemas import ChatResponse, LLMAnswer
from .search import build_search_tool, extract_grounded_sources

log = logging.getLogger(__name__)
APP_NAME = "ai_search_assistant"

INSTRUCTION = """You are a careful web research assistant. Today's date is available from search results.

Workflow for every user message:
1. Work out the user's intent, using the conversation so far to resolve pronouns and follow-ups
   ("what about its price?" -> rewrite as a self-contained question).
2. Decide if the web is needed. Search for anything current, factual, niche, or checkable.
   Skip search only for greetings, pure rewriting/maths, or questions about this conversation.
3. If searching, call the google search tool with a short, specific, self-contained query.
   Use a second, differently-worded query when the first is thin or when claims need cross-checking.
4. Answer ONLY from what the search returned. Never invent facts, numbers, quotes, or URLs.
5. If sources disagree, say so and describe each position. Separate confirmed facts from inference.
6. If reliable information cannot be found, say that plainly, give a low confidence_score, and do not guess.

Output fields (JSON, exactly as the schema defines):
- answer: clear, complete, plain-language answer (short paragraphs).
- summary: one or two sentences.
- sources: leave as an empty list; the system attaches verified links.
- key_points: 3-6 short findings.
- confidence_score: 0-1. High (>0.8) only when several reliable sources agree. Lower for conflict or thin evidence.
- follow_up_questions: 2-3 useful next questions.
- search_performed: true only if you actually called the search tool.
"""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):  # DNS, connect, read timeouts
        return True
    return isinstance(exc, genai_errors.APIError) and exc.code in {429, 500, 502, 503, 504}


class SearchAssistant:
    def __init__(self, settings: Settings, runner: Runner | None = None, session_service=None):
        self.settings = settings
        if settings.google_api_key:
            os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)
        self.sessions = session_service or InMemorySessionService()
        self.runner = runner or Runner(
            app_name=APP_NAME, agent=self._build_agent(), session_service=self.sessions
        )
        self._known: OrderedDict[str, asyncio.Lock] = OrderedDict()  # LRU of live session ids

    def _build_agent(self) -> LlmAgent:
        return LlmAgent(
            name="search_assistant",
            model=self.settings.llm_model,
            instruction=INSTRUCTION,
            tools=[build_search_tool(self.settings.llm_model)],
            output_schema=LLMAnswer,
            generate_content_config=types.GenerateContentConfig(temperature=0.2),
        )

    async def _ensure_session(self, session_id: str | None) -> tuple[str, asyncio.Lock]:
        sid = session_id or uuid.uuid4().hex
        if sid not in self._known:
            if not await self.sessions.get_session(app_name=APP_NAME, user_id=sid, session_id=sid):
                await self.sessions.create_session(app_name=APP_NAME, user_id=sid, session_id=sid)
            self._known[sid] = asyncio.Lock()
            while len(self._known) > self.settings.max_sessions:
                old, _ = self._known.popitem(last=False)
                await self.sessions.delete_session(app_name=APP_NAME, user_id=old, session_id=old)
        self._known.move_to_end(sid)
        return sid, self._known[sid]

    async def reset(self, session_id: str) -> None:
        if self._known.pop(session_id, None):
            await self.sessions.delete_session(app_name=APP_NAME, user_id=session_id, session_id=session_id)

    async def _run_once(self, sid: str, text: str) -> tuple[str, dict[str, str]]:
        msg = types.Content(role="user", parts=[types.Part(text=text)])
        final, grounded = "", {}
        async with asyncio.timeout(self.settings.request_timeout_s):
            async for ev in self.runner.run_async(user_id=sid, session_id=sid, new_message=msg):
                grounded.update(extract_grounded_sources(ev))
                if ev.is_final_response() and ev.content and ev.content.parts:
                    final = "".join(p.text or "" for p in ev.content.parts if p.text)
        return final, grounded

    async def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        started = time.perf_counter()
        sid, lock = await self._ensure_session(session_id)
        async with lock:  # one in-flight turn per conversation keeps history ordered
            try:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception(_is_transient),
                    stop=stop_after_attempt(self.settings.max_retries),
                    wait=wait_random_exponential(multiplier=1, max=8),
                    reraise=True,
                ):
                    with attempt:
                        raw, grounded = await self._run_once(sid, message)
            except TimeoutError:
                log.error("timeout sid=%s", sid)
                raise AssistantError(504, "timeout", "The search took too long. Please try again.")
            except httpx.TransportError:
                log.error("network error reaching the AI service sid=%s", sid)
                raise AssistantError(502, "network", "Could not reach the AI service. Please try again.")
            except genai_errors.APIError as exc:
                log.error("upstream error sid=%s code=%s", sid, exc.code)
                if exc.code in {401, 403}:
                    raise AssistantError(502, "auth", "The AI service rejected the server's API key.")
                if exc.code == 429:
                    raise AssistantError(503, "busy", "The AI service is busy. Please retry shortly.")
                raise AssistantError(502, "upstream", "The AI service returned an error.")
            except Exception:
                log.exception("unexpected failure sid=%s", sid)
                raise AssistantError(500, "internal", "Something went wrong. Please try again.")
        latency = int((time.perf_counter() - started) * 1000)
        log.info("chat ok sid=%s sources=%d latency_ms=%d", sid, len(grounded), latency)
        return build_response(raw, grounded, sid, latency)
