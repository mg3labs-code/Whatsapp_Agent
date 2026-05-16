"""Langfuse tracing (SDK v4 best practices).

- OpenAI: use ``get_async_openai_client()`` (langfuse.openai drop-in).
- Per-message context: ``message_trace_context()`` with hashed user/session ids.
- Agent spans: ``@observe`` on agent entry points.
- Shutdown: ``flush_langfuse()`` so background workers send spans.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from typing import Any, Iterator

from langfuse import get_client, propagate_attributes


def hash_user_id(phone: str) -> str:
    """Short hashed id for Langfuse user_id / session_id — never raw phone."""
    if not phone:
        return "anonymous"
    return hashlib.sha256(phone.encode()).hexdigest()[:16]


def tracing_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_async_openai_client(*, api_key: str) -> Any:
    """Langfuse-instrumented AsyncOpenAI client (tokens, latency, model auto-captured)."""
    from langfuse.openai import AsyncOpenAI

    return AsyncOpenAI(api_key=api_key)


@contextmanager
def message_trace_context(
    *,
    trace_name: str,
    phone: str,
    message_id: str = "",
    feature: str = "orchestrator",
) -> Iterator[None]:
    """Wrap one inbound WhatsApp turn with trace-level attributes."""
    if not tracing_enabled():
        yield
        return

    user_id = hash_user_id(phone)
    tags = ["wasa", feature] if feature else ["wasa"]
    metadata: dict[str, str] = {}
    if message_id:
        metadata["message_id"] = message_id[:200]

    with propagate_attributes(
        trace_name=trace_name,
        user_id=user_id,
        session_id=user_id,
        tags=tags,
        metadata=metadata,
    ):
        yield


def set_span_io(*, input_data: Any = None, output_data: Any = None) -> None:
    """Set current observation input/output without logging full function args."""
    # SECURITY: callers must pass metadata only (message_len, intent, agent) — not raw bodies.
    if not tracing_enabled():
        return
    client = get_client()
    if input_data is not None:
        client.update_current_span(input=input_data)
    if output_data is not None:
        client.update_current_span(output=output_data)


def flush_langfuse() -> None:
    """Send buffered spans (call after each background webhook task)."""
    if not tracing_enabled():
        return
    get_client().flush()
