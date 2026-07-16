"""Intent classification for orchestrator routing."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langfuse import observe

from app.agents.order import SELECT_PAYMENT, is_order_account_message, is_order_tracking_message
from app.messages.conversation_ui import MENU_OPTION_IDS, mark_menu_selection
from app.messages.session_flow import is_discount_request, is_speak_to_team_request
from app.utils.tracing import get_async_openai_client, set_span_io

logger = logging.getLogger(__name__)

CLASSIFIER_MODEL = "gpt-4o-mini"
CONFIDENCE_ESCALATE_THRESHOLD = 0.40
CLARIFICATION_ATTEMPTS_BEFORE_ESCALATE = 2

VALID_INTENTS = frozenset({"pricing", "faq", "order", "qualify", "escalate"})

# Order and pricing require lead qualification; FAQ and general chat do not.
INTENTS_REQUIRING_QUALIFICATION = frozenset({"order", "pricing"})

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

ORDER_ACTION_IDS = frozenset({"pay_bank", "pay_card", "new_order", "order_status", "my_orders"})

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
    """Classify buyer message and apply qualify-before-order/pricing rules.

    FAQ is available without qualification. Order and pricing require a qualified lead.

    Returns (intent, updated_session).
    """
    session = dict(session or {})

    if _matches_human_keyword(message) or is_speak_to_team_request(message):
        return "escalate", session

    if is_discount_request(message):
        session["escalation_reason"] = "discount_request"
        return "escalate", session

    key = (message or "").strip().lower()
    if is_order_tracking_message(message):
        return "order", session
    if is_order_account_message(message):
        return "order", session
    if key in ORDER_ACTION_IDS:
        if key == "speak":
            return "escalate", session
        return "order", session
    if session.get("order_state") == SELECT_PAYMENT:
        return "order", session

    menu_intent = _menu_button_intent(message)
    if menu_intent:
        session = mark_menu_selection(session, message)
        if menu_intent == "escalate":
            return "escalate", session
        if menu_intent == "my_orders":
            return "order", session
        # FAQ / already-qualified order+pricing skip qualification forever.
        if not session.get("lead_qualified") and menu_intent in INTENTS_REQUIRING_QUALIFICATION:
            session["pending_intent"] = menu_intent
            return "qualify", session
        return menu_intent, session

    phone = session.get("phone") or ""
    intent, confidence = await _classify_with_llm(message, phone=phone)

    if not session.get("lead_qualified"):
        if intent == "escalate":
            return "escalate", session
        # FAQ is available without qualification for new buyers.
        if intent == "faq":
            return "faq", session
        if intent in INTENTS_REQUIRING_QUALIFICATION:
            session["pending_intent"] = intent
            return "qualify", session
        if intent == "qualify":
            return "qualify", session
        session["pending_intent"] = intent
        return "qualify", session

    # Already qualified — never route back to qualify from free-text intents.
    if intent == "qualify":
        return "faq", session

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
