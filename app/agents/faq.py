"""FAQ / RAG agent.

Embeds the buyer message, queries Pinecone (wasa-faq), filters by similarity,
then grounds GPT-4o-mini on retrieved chunk text only. No invention outside
retrieved context.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

import pinecone
from langfuse import observe

from app.business.countries import (
    SHIPMENT_EXCLUDED_REFUSAL,
    canonicalize_country,
    is_shipment_excluded_country,
)
from app.business.shipping import get_shipping_options
from app.db.database import SessionLocal
from app.utils.tracing import get_async_openai_client, set_span_io

logger = logging.getLogger(__name__)

_DESTINATION_COUNTRY_RE = re.compile(
    r"\b(?:ship|deliver|send)\s+to\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})\b",
    re.IGNORECASE,
)

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

# First miss: honest, no false "connecting you" promise (escalation fires on 2nd miss).
NO_CONTEXT_REPLY = (
    "I don't have specific information on that in our knowledge base. "
    "Could you rephrase, or type *speak to team* if you'd like a specialist?"
)

CONTINUE_FAQ = "faq"
FAQ_MISS_ESCALATE_AFTER = 2

FAQ_SYSTEM_PROMPT = (
    "You are a helpful assistant for New Life Medicare pharmaceutical exports.\n"
    "Answer the buyer's question using ONLY the context provided below.\n"
    "If the answer is not in the context, say exactly: "
    "'I don't have specific information on that in our knowledge base. "
    "Could you rephrase, or type *speak to team* if you'd like a specialist?'\n"
    "Never make up information about regulations, shipping times, or product specifications.\n"
    "Use *asterisks* for bold text. Keep answers concise and professional."
)

ERROR_REPLY = (
    "I'm having trouble searching our knowledge base right now. "
    "Please try again in a moment, or type *speak to team* for help."
)

# Soft no-answer signals from LLM (legacy + current copy) — count toward miss escalation.
_SOFT_NO_ANSWER_MARKERS = (
    "don't have specific information",
    "i don't have specific information",
    "let me connect you with our team",
    "connect you with our team for this",
    "i'll need to check on that",
)


def _extract_destination_country(message: str) -> str | None:
    """Return canonical destination country when message asks ship/deliver/send to one."""
    match = _DESTINATION_COUNTRY_RE.search(message or "")
    if not match:
        return None
    words = match.group(1).strip().split()
    for length in range(len(words), 0, -1):
        canonical = canonicalize_country(" ".join(words[:length]))
        if canonical:
            return canonical
    return None


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


def _is_soft_no_answer(reply: str) -> bool:
    """True when LLM (or fallback) admits it cannot answer from context."""
    lowered = (reply or "").lower()
    return any(marker in lowered for marker in _SOFT_NO_ANSWER_MARKERS)


def _record_faq_miss(session: dict) -> tuple[str, str]:
    """Increment miss counter; return (buyer_reply, next_intent).

    On the 2nd consecutive miss, clear the counter, set escalation_reason, and
    return an empty buyer reply so escalation_agent owns the message (no duplicate
    "connect you" copy).
    """
    miss_count = int(session.get("faq_miss_count", 0) or 0) + 1
    session["faq_miss_count"] = miss_count
    if miss_count >= FAQ_MISS_ESCALATE_AFTER:
        session["faq_miss_count"] = 0
        session["escalation_reason"] = "faq_no_match_repeated"
        return "", "escalate"
    return NO_CONTEXT_REPLY, CONTINUE_FAQ


def _record_faq_error(session: dict) -> tuple[str, str]:
    """Same miss accounting for infra failures; first miss keeps ERROR_REPLY copy."""
    miss_count = int(session.get("faq_miss_count", 0) or 0) + 1
    session["faq_miss_count"] = miss_count
    if miss_count >= FAQ_MISS_ESCALATE_AFTER:
        session["faq_miss_count"] = 0
        session["escalation_reason"] = "faq_no_match_repeated"
        return "", "escalate"
    return ERROR_REPLY, CONTINUE_FAQ


@observe(name="faq_agent", capture_input=False)
async def run_faq_agent(
    message: str,
    phone: str = "",
    session: dict | None = None,
) -> tuple[str, dict, str]:
    """Run Pinecone RAG + GPT-4o-mini on one buyer message.

    Returns (reply_text, updated_session, next_intent). On missing env, retrieval
    failure, or no chunks above the score threshold, returns a safe no-match message
    (no LLM call when there is no qualifying context). After two consecutive misses
    (no chunks, soft LLM no-answer, empty reply, or infra error), next_intent is
    ``escalate`` so the orchestrator can hand off like speak-to-team. On escalate the
    reply text is empty — escalation_agent provides the buyer-facing handoff message.
    """
    session = dict(session or {})
    destination = _extract_destination_country(message)
    if destination:
        if is_shipment_excluded_country(destination):
            return SHIPMENT_EXCLUDED_REFUSAL, session, CONTINUE_FAQ
        db = SessionLocal()
        try:
            options = get_shipping_options(destination, total_g=0, db=db)
        finally:
            db.close()
        if options.get("available"):
            return (
                f"Yes, we ship to {destination}. Express (EMS): approximately 7-14 days. "
                "Standard LP: approximately 15-30 days. Reply 'place an order' when ready."
            ), session, CONTINUE_FAQ
        return (
            f"We do not have standard shipping rates configured for {destination} yet. "
            "Our team can confirm if shipping is possible — type 'speak to team' and "
            "someone will follow up."
        ), session, CONTINUE_FAQ

    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        logger.error("OPENAI_API_KEY or PINECONE_API_KEY missing; FAQ agent cannot run")
        reply, next_intent = _record_faq_error(session)
        return reply, session, next_intent

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
            reply, next_intent = _record_faq_miss(session)
            return reply, session, next_intent

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
        raw_reply = (chat.choices[0].message.content or "").strip()
        if not raw_reply or _is_soft_no_answer(raw_reply):
            set_span_io(output_data={"status": "soft_no_answer", "chunks": len(context_chunks)})
            reply, next_intent = _record_faq_miss(session)
            return reply, session, next_intent

        session["faq_miss_count"] = 0
        set_span_io(
            output_data={
                "reply_len": len(raw_reply),
                "chunks": len(context_chunks),
                "agent": "faq",
            }
        )
        return raw_reply, session, CONTINUE_FAQ
    except Exception:
        # SECURITY: log agent name only — not message content
        logger.exception("FAQ agent failed")
        reply, next_intent = _record_faq_error(session)
        return reply, session, next_intent
