"""FAQ / RAG agent.

Embeds the buyer message, queries Pinecone (wasa-faq), filters by similarity,
then grounds GPT-4o-mini on retrieved chunk text only. No invention outside
retrieved context.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import pinecone
from langfuse import observe

from app.utils.tracing import get_async_openai_client, set_span_io

logger = logging.getLogger(__name__)

INDEX_NAME = os.getenv("PINECONE_INDEX", "wasa-faq")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 3
# Cosine similarity floor for including a Pinecone match (strict: chunk used only if score > floor).
#
# Tuned band 0.40–0.42 from `python -m scripts.analyze_faq_thresholds` on wasa-faq (PowerShell run):
#   T=0.40–0.42 → 0/4 labelled bad queries pass the filter; escalate_max (restricted) top-1 ≈ 0.394.
#   Same band → 3/22 labelled FAQ queries escalate on top-1 alone (acceptable vs T=0.70 → 21/22).
# Default 0.41 is midpoint; override with FAQ_PINECONE_MIN_SCORE (e.g. 0.40 or 0.42) without code change.
_default_floor = os.getenv("FAQ_PINECONE_MIN_SCORE", "0.41")
try:
    SCORE_MIN_EXCLUSIVE = float(_default_floor)
except ValueError:
    SCORE_MIN_EXCLUSIVE = 0.41

NO_CONTEXT_REPLY = (
    "I don't have specific information on that. Let me connect you with our team."
)

FAQ_SYSTEM_PROMPT = (
    "You are a helpful assistant for New Life Medicare pharmaceutical exports.\n"
    "Answer the buyer's question using ONLY the context provided below.\n"
    "If the answer is not in the context, say: 'I'll need to check on that and get back to you.\n"
    "Let me connect you with our team for this.'\n"
    "Never make up information about regulations, shipping times, or product specifications.\n"
    "Use *asterisks* for bold text. Keep answers concise and professional."
)

ERROR_REPLY = (
    "I'm having trouble searching our knowledge base right now. "
    "Let me connect you with our team."
)


def _normalize_matches(query_response: Any) -> list[dict[str, Any]]:
    """Turn Pinecone query response into [{score, metadata}, ...]."""
    matches = getattr(query_response, "matches", None)
    if matches is None and isinstance(query_response, dict):
        matches = query_response.get("matches", [])
    out: list[dict[str, Any]] = []
    for m in matches or []:
        score = getattr(m, "score", None)
        if score is None and isinstance(m, dict):
            score = m.get("score")
        md = getattr(m, "metadata", None)
        if md is None and isinstance(m, dict):
            md = m.get("metadata") or {}
        out.append({"score": score, "metadata": md if isinstance(md, dict) else {}})
    return out


def _pinecone_query_sync(api_key: str, vector: list[float]) -> Any:
    pc = pinecone.Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    return index.query(vector=vector, top_k=TOP_K, include_metadata=True)


@observe(name="faq_agent", capture_input=False)
async def run_faq_agent(
    message: str,
    phone: str = "",
    session: dict | None = None,
) -> str:
    """Run Pinecone RAG + GPT-4o-mini on one buyer message.

    Returns a WhatsApp-ready string. On missing env, retrieval failure, or no
    chunks above the score threshold, returns a safe escalation-style message
    (no LLM call when there is no qualifying context).
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        logger.error("OPENAI_API_KEY or PINECONE_API_KEY missing; FAQ agent cannot run")
        return ERROR_REPLY

    # SECURITY: Langfuse input — metadata only, not full message body
    set_span_io(input_data={"message_len": len(message)})
    client = get_async_openai_client(api_key=openai_key)

    try:
        embedding = await client.embeddings.create(
            input=message,
            model=EMBEDDING_MODEL,
        )
        vector = embedding.data[0].embedding

        raw_results = await asyncio.to_thread(_pinecone_query_sync, pinecone_key, vector)
        rows = _normalize_matches(raw_results)

        context_chunks: list[str] = []
        for r in rows:
            score = r.get("score")
            if score is None or score <= SCORE_MIN_EXCLUSIVE:
                continue
            text = (r.get("metadata") or {}).get("text")
            if isinstance(text, str) and text.strip():
                context_chunks.append(text.strip())

        if not context_chunks:
            set_span_io(output_data={"status": "no_context"})
            return NO_CONTEXT_REPLY

        context = "\n\n".join(context_chunks)
        user_content = f"Context:\n{context}\n\nBuyer question:\n{message}"
        chat_messages = [
            {"role": "system", "content": FAQ_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        chat = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=chat_messages,
        )
        reply = (chat.choices[0].message.content or "").strip() or NO_CONTEXT_REPLY
        set_span_io(
            output_data={
                "reply_len": len(reply),
                "chunks": len(context_chunks),
                "agent": "faq",
            }
        )
        return reply
    except Exception:
        # SECURITY: log agent name only — not message content
        logger.exception("FAQ agent failed")
        return ERROR_REPLY
