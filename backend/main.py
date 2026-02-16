import os
import uuid
import json
import time
from typing import Any, Dict, List, Optional, Iterable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from openai import OpenAI
from httpx import Timeout

# ----------------------------
# Config
# ----------------------------
ARCHIA_BASE_URL = "https://registry.archia.app/v1"

# Default agent if frontend doesn't specify one
DEFAULT_MODEL = "agent:Prism Orchestrator"

ARCHIA_TOKEN = os.environ.get("ARCHIA_TOKEN")

# ---- Timeout knobs (very lenient for prototyping) ----
ARCHIA_TIMEOUT_S = float(os.environ.get("ARCHIA_TIMEOUT_S", "600"))  # 10 minutes
ARCHIA_CONNECT_TIMEOUT_S = float(os.environ.get("ARCHIA_CONNECT_TIMEOUT_S", "30"))

# ---- Optional prompt limit (disabled by default) ----
PROMPT_MAX_CHARS_ENV = os.environ.get("PROMPT_MAX_CHARS", "").strip()
PROMPT_MAX_CHARS: Optional[int] = int(PROMPT_MAX_CHARS_ENV) if PROMPT_MAX_CHARS_ENV else None

# ---- Streaming heartbeat ----
SSE_HEARTBEAT_S = float(os.environ.get("SSE_HEARTBEAT_S", "10"))

# Optional safety: allow-list models you expect to use (prevents accidental typos)
ALLOWED_MODELS = {
    "agent:Prism Orchestrator",
    "agent:Emotion Support Agent",
    "agent:Web Search Agent",
    "agent:Course Materials Agent",
}

client: Optional[OpenAI] = None
if ARCHIA_TOKEN:
    client = OpenAI(
        base_url=ARCHIA_BASE_URL,
        api_key="not-used",
        default_headers={"Authorization": f"Bearer {ARCHIA_TOKEN}"},
        timeout=Timeout(
            ARCHIA_TIMEOUT_S,
            connect=ARCHIA_CONNECT_TIMEOUT_S,
            read=ARCHIA_TIMEOUT_S,
            write=ARCHIA_TIMEOUT_S,
        ),
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Schemas
# ----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    context: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


class AgentTrace(BaseModel):
    agent: str
    info: str


class ChatResponse(BaseModel):
    run_id: str
    answer: str
    trace: List[AgentTrace]


# ----------------------------
# Helpers (non-stream)
# ----------------------------
def extract_text_from_responses_api(resp: Any) -> str:
    parts: List[str] = []
    output = getattr(resp, "output", None) or []
    for item in output:
        content = getattr(item, "content", None) or []
        for c in content:
            text = getattr(c, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text)

    joined = "\n".join(parts).strip()
    if joined:
        return joined

    try:
        return resp.output_text  # type: ignore[attr-defined]
    except Exception:
        return str(resp)


def build_prompt(history: Optional[List[ChatMessage]], message: str) -> str:
    lines: List[str] = []
    if history:
        for m in history:
            role = (m.role or "").lower().strip()
            label = "User" if role == "user" else "Assistant"
            content = (m.content or "").strip()
            if content:
                lines.append(f"{label}: {content}")
    lines.append(f"User: {message.strip()}")
    return "\n".join(lines).strip()


def maybe_truncate_prompt(prompt: str) -> str:
    if PROMPT_MAX_CHARS is None:
        return prompt
    if len(prompt) <= PROMPT_MAX_CHARS:
        return prompt
    return prompt[-PROMPT_MAX_CHARS:]


def resolve_model(req_model: Optional[str]) -> str:
    model = (req_model or DEFAULT_MODEL).strip()
    # If you DON'T want allow-list enforcement, just return model here.
    if model in ALLOWED_MODELS:
        return model
    # fallback to default if unknown (prevents hard failure)
    return DEFAULT_MODEL


def call_llm(prompt: str, model: str) -> str:
    if client is None:
        raise RuntimeError("ARCHIA_TOKEN is missing. Please export ARCHIA_TOKEN.")
    resp = client.responses.create(model=model, input=prompt)
    return extract_text_from_responses_api(resp)


# ----------------------------
# Helpers (streaming + SSE)
# ----------------------------
def _sse(event: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _ev_type(ev: Any) -> Optional[str]:
    if isinstance(ev, dict):
        return ev.get("type")
    return getattr(ev, "type", None)


def _ev_delta(ev: Any) -> Optional[str]:
    if isinstance(ev, dict):
        return ev.get("delta")
    return getattr(ev, "delta", None)


def stream_llm(prompt: str, model: str) -> Iterable[str]:
    if client is None:
        yield _sse("error", {"message": "ARCHIA_TOKEN is missing. Please export ARCHIA_TOKEN."})
        return

    last_heartbeat = time.time()

    try:
        stream = client.responses.create(
            model=model,
            input=prompt,
            stream=True,
        )

        yield _sse(
            "meta",
            {
                "status": "stream_start",
                "model": model,
                "prompt_chars": len(prompt),
                "note": "streaming enabled",
            },
        )

        for ev in stream:
            now = time.time()
            if now - last_heartbeat >= SSE_HEARTBEAT_S:
                last_heartbeat = now
                yield _sse("ping", {"t": int(now), "note": "keepalive"})

            t = _ev_type(ev)

            if t == "response.output_text.delta":
                d = _ev_delta(ev) or ""
                if d:
                    yield _sse("delta", {"delta": d})

            elif t == "response.completed":
                yield _sse("done", {"status": "completed"})
                return

            elif t == "response.failed":
                yield _sse("error", {"message": "Model response failed (response.failed)."})
                return

            elif t == "error":
                msg = None
                if isinstance(ev, dict):
                    msg = ev.get("message") or ev.get("error", {}).get("message")
                else:
                    msg = getattr(ev, "message", None)
                yield _sse("error", {"message": msg or "Unknown streaming error (event=error)."})
                return

        yield _sse("done", {"status": "ended"})

    except Exception as e:
        yield _sse(
            "error",
            {
                "message": "Streaming exception on backend.",
                "detail": str(e),
            },
        )


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "archia_token_present": bool(ARCHIA_TOKEN),
        "default_model": DEFAULT_MODEL,
        "allowed_models": sorted(list(ALLOWED_MODELS)),
        "archia_timeout_s": ARCHIA_TIMEOUT_S,
        "prompt_max_chars": PROMPT_MAX_CHARS,
        "sse_heartbeat_s": SSE_HEARTBEAT_S,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    run_id = str(uuid.uuid4())
    model = resolve_model(req.model)

    prompt = build_prompt(req.history, req.message)
    prompt = maybe_truncate_prompt(prompt)

    trace: List[AgentTrace] = [
        AgentTrace(agent="orchestrator", info="history-enabled mode (non-stream)"),
        AgentTrace(agent="llm_client", info=f"calling model={model} via {ARCHIA_BASE_URL}"),
        AgentTrace(agent="prompt", info=f"chars={len(prompt)} turns={len(req.history or []) + 1}"),
        AgentTrace(agent="timeouts", info=f"client_timeout_s={ARCHIA_TIMEOUT_S}"),
    ]

    answer = call_llm(prompt, model=model)
    trace.append(AgentTrace(agent="llm_client", info="response received"))
    return ChatResponse(run_id=run_id, answer=answer, trace=trace)


@app.post("/chat_stream")
def chat_stream(req: ChatRequest):
    model = resolve_model(req.model)

    prompt = build_prompt(req.history, req.message)
    prompt = maybe_truncate_prompt(prompt)

    return StreamingResponse(
        stream_llm(prompt, model=model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # If you later put nginx in front, keep this:
            # "X-Accel-Buffering": "no",
        },
    )
