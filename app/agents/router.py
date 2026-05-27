"""Intent classification for orchestrator routing."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langfuse import observe

from app.messages.conversation_ui import MENU_OPTION_IDS, mark_menu_selection
from app.utils.tracing import get_async_openai_client, set_span_io

logger = logging.getLogger(__name__)

CLASSIFIER_MODEL = "gpt-4o-mini"
CONFIDENCE_ESCALATE_THRESHOLD = 0.40
CLARIFICATION_ATTEMPTS_BEFORE_ESCALATE = 2

VALID_INTENTS = frozenset({"pricing", "faq", "order", "qualify", "escalate"})

HUMAN_KEYWORDS: tuple[str, ...] = (
    "human",
    "agent",
    "speak to someone",
    "real person",
    "not helpful",
    "complaint",
    "escalate",
    "talk to someone",
    "connect me",
)

CLASSIFIER_SYSTEM_PROMPT = (
    "Classify this pharmaceutical B2B WhatsApp message. Return ONLY valid JSON.\n"
    '{"intent": "pricing"|"faq"|"order"|"qualify"|"escalate", "confidence": 0.0-1.0}\n'
    "- pricing: asking about product price, MOQ, quantity pricing, discounts, quotes\n"
    "- faq: asking about shipping, documentation, policies, company info, regulations, timelines\n"
    "- order: wants to place an order, buy products, confirming purchase intent\n"
    "- qualify: new contact, introduction, general inquiry with no specific product/topic\n"
    "- escalate: complaint, frustration, urgent situation, not getting help needed"
)


def _matches_human_keyword(message: str) -> bool:
    text = (message or "").lower()
    return any(keyword in text for keyword in HUMAN_KEYWORDS)


def _menu_button_intent(message: str) -> str | None:
    """Map WhatsApp quick-reply button ids to orchestrator intents."""
    key = (message or "").strip().lower()
    if key == "speak":
        return "escalate"
    if key in MENU_OPTION_IDS:
        return key
    return None


def _keyword_fallback_intent(message: str) -> tuple[str, float]:
    """Heuristic intent when LLM is unavailable (no API key or error)."""
    text = (message or "").lower()
    menu = _menu_button_intent(message)
    if menu:
        return menu, 0.95
    order_markers = (
        "place an order",
        "place order",
        "want to order",
        "i want to order",
        "order ",
        " buy ",
        "purchase",
    )
    if any(m in text for m in order_markers):
        return "order", 0.75
    pricing_markers = (
        "price",
        "pricing",
        "cost",
        "quote",
        "moq",
        "per unit",
        "/unit",
    )
    if any(m in text for m in pricing_markers):
        return "pricing", 0.75
    if "units" in text and any(m in text for m in ("price", "pricing", "quote", "cost")):
        return "pricing", 0.75
    if any(m in text for m in ("hi", "hello", "introduction", "new buyer")):
        return "qualify", 0.7
    return "faq", 0.7


def _parse_classifier_response(raw: str) -> tuple[str, float]:
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", raw or "")
        if not match:
            return "faq", 0.5
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return "faq", 0.5

    intent = str(data.get("intent", "faq")).lower().strip()
    if intent not in VALID_INTENTS:
        intent = "faq"

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    return intent, confidence


@observe(name="router_classifier", capture_input=False)
async def _classify_with_llm(message: str, phone: str = "") -> tuple[str, float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY missing; using keyword intent fallback")
        return _keyword_fallback_intent(message)

    # SECURITY: Langfuse input — metadata only, not full message body
    set_span_io(input_data={"message_len": len(message or "")})
    client = get_async_openai_client(api_key=api_key)
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": message or ""},
    ]
    try:
        response = await client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0,
            max_tokens=80,
        )
        raw = (response.choices[0].message.content or "").strip()
        result = _parse_classifier_response(raw)
        set_span_io(output_data={"intent": result[0], "confidence": result[1]})
        return result
    except Exception:
        logger.exception("Intent classifier LLM call failed")
        return _keyword_fallback_intent(message)


async def classify_intent(message: str, session: dict) -> tuple[str, dict]:
    """Classify buyer message and apply qualify-before-pricing / low-confidence rules.

    Returns (intent, updated_session).
    """
    session = dict(session or {})

    if _matches_human_keyword(message):
        return "escalate", session

    menu_intent = _menu_button_intent(message)
    if menu_intent:
        session = mark_menu_selection(session, message)
        if not session.get("lead_qualified") and menu_intent != "escalate":
            if menu_intent != "qualify":
                session["pending_intent"] = menu_intent
            return "qualify", session
        return menu_intent, session

    phone = session.get("phone") or ""
    intent, confidence = await _classify_with_llm(message, phone=phone)

    if not session.get("lead_qualified"):
        if intent != "escalate":
            if intent != "qualify":
                session["pending_intent"] = intent
            return "qualify", session

    if confidence < 0.45 and session.get("lead_qualified"):
        prior_count = session.get("clarification_count", session.get("clarification_attempts", 0))
        count = prior_count + 1
        session["clarification_count"] = count
        session["clarification_attempts"] = count
        if count >= 2:
            session["clarification_count"] = 0
            session["clarification_attempts"] = 0
            return "escalate", session
        return "faq", session

    session.pop("clarification_count", None)
    session.pop("clarification_attempts", None)
    return intent, session
