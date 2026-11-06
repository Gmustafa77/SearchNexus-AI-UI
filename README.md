# Lookout: AI search assistant (Google ADK + GoogleSearchAgentTool)

Developed by **Gulam Mustafa**.

## Features
Voice input: tap the mic to dictate a question (English, Hindi, Urdu). Uses the browser's Web Speech API, so it
works in Chrome, Edge and Safari over HTTPS or localhost; the mic hides itself where unsupported (e.g. Firefox).
Note: Chrome sends the audio to Google's speech service to transcribe it.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GOOGLE_API_KEY (Gemini API key from aistudio.google.com)
uvicorn app.main:app --reload   # open http://localhost:8000
pytest                          # offline tests (fake ADK runner)
```
Docker: `docker build -t lookout . && docker run -p 8000:8000 --env-file .env lookout`

## Architecture
| File | Role |
|---|---|
| `app/search.py` | Builds `GoogleSearchAgentTool`; extracts Google's grounding chunks |
| `app/agent.py` | ADK `LlmAgent` (tool + `output_schema`), sessions, timeout, retries, error mapping |
| `app/responses.py` | Validates output; **citations come only from Google grounding data** |
| `app/schemas.py` | `LLMAnswer` (model contract), `ChatResponse` (API), lenient repair |
| `app/main.py` | FastAPI, validation, rate limiting, security headers |
| `app/static/index.html` | Chat UI |

## Scaling notes
Sessions and rate limits are in-memory (single instance). For multiple instances use ADK's
`DatabaseSessionService` and a Redis-backed limiter, and sticky sessions or a shared lock.
