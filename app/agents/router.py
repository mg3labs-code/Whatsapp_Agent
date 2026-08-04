"""Intent classification for orchestrator routing.

Layered design (mature router):
  L1 — Deterministic: human/speak, discount, buttons, order actions, pure greetings
  L2 — Keyword fallback when LLM unavailable (clear price/order markers)
  L3 — LLM free-text classifier with session context for everything else
  Policy — qualify gate, never re-qual, low-confidence clarify path

Pure greetings → menu_refresh (not FAQ). Free text is classified by the LLM —
keyword lists are shortcuts only, not a full language model substitute.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langfuse import observe

from app.agents.order import (
    SELECT_PAYMENT,
    is_order_account_message,
    is_order_tracking_message,
    message_looks_like_catalog_product,
)
from app.messages.conversation_ui import MENU_OPTION_IDS, mark_menu_selection
from app.messages.session_flow import (
    is_discount_request,
    is_pure_greeting,
    is_speak_to_team_request,
)
from app.utils.tracing import get_async_openai_client, set_span_io

logger = logging.getLogger(__name__)

CLASSIFIER_MODEL = "gpt-4o-mini"
# Live low-confidence threshold (qualified buyers). Keep in sync with classify_intent.
CONFIDENCE_ESCALATE_THRESHOLD = 0.45
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
    '{"intent": "pricing"|"faq"|"order"|"qualify"|"escalate", "confidence": 0.0-1.0}\n\n'
    "Intent definitions:\n"
    "- pricing: product price, quotes, rates, cost per strip; "
    "also a bare product/brand/strength name alone (e.g. \"KLENSMART 60MG\", \"LASIX TAB\") "
    "when the buyer is likely asking about that product — NOT faq\n"
    "- faq: shipping, delivery times, documentation (COA/GMP), policies, regulations, "
    "timelines, company info, how/when questions about process\n"
    "- order: place/buy/purchase products, cart, checkout, confirm purchase intent\n"
    "- qualify: new contact introducing themselves with no clear pricing/order/faq ask, "
    "or general \"who are you / what do you do\" without a task\n"
    "- escalate: complaint, frustration, urgent human help, not getting help needed\n\n"
    "Rules:\n"
    "- Bare greetings (hi/hello) are handled outside this classifier — do not use this "
    "path for those.\n"
    "- If the message mixes a greeting with a request, classify the REQUEST "
    '(e.g. "hi, price for metformin" → pricing; "hello I want to order" → order).\n'
    "- Prefer the primary actionable intent when multiple could apply.\n"
    "- Typos, slang, short fragments, and Hinglish are OK — infer intent from meaning.\n"
    "- Use session context for follow-ups (e.g. a bare quantity after pricing → pricing).\n"
    "- confidence: 0.9+ clear; 0.6–0.85 likely; below 0.45 truly unclear.\n"
)

# Process/FAQ language — do not force pricing even if a drug name appears.
_FAQ_PROCESS_MARKERS: tuple[str, ...] = (
    "ship",
    "shipping",
    "deliver",
    "delivery",
    "document",
    "documents",
    "documentation",
    "policy",
    "policies",
    "timeline",
    "timelines",
    "how long",
    "when will",
    "customs",
    "regulation",
    "regulations",
    "coa",
    "who-gmp",
    "gmp",
    "license",
    "what documents",
    "do you ship",
    "can you ship",
)

_CLEAR_ORDER_MARKERS: tuple[str, ...] = (
    "place an order",
    "place order",
    "want to order",
    "i want to order",
    "i want to buy",
    "add to cart",
    "checkout",
)


def _looks_like_faq_process_question(message: str) -> bool:
    text = (message or "").lower()
    return any(marker in text for marker in _FAQ_PROCESS_MARKERS)


def _looks_like_clear_order_request(message: str) -> bool:
    text = (message or "").lower()
    return any(marker in text for marker in _CLEAR_ORDER_MARKERS)

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
    if is_pure_greeting(message):
        # classify_intent short-circuits greetings; keep fallback consistent.
        return "qualify", 0.7
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
        "per strip",
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


def _session_context_for_classifier(session: dict | None) -> str:
    """Compact session metadata for the LLM (no PII beyond country label)."""
    session = session or {}
    parts = [
        f"lead_qualified={bool(session.get('lead_qualified'))}",
        f"last_agent={session.get('last_agent') or 'none'}",
        f"order_state={session.get('order_state') or 'none'}",
        f"qual_state={session.get('qual_state') or 'none'}",
        f"country={session.get('country') or 'none'}",
        f"pending_intent={session.get('pending_intent') or 'none'}",
    ]
    return "Session context: " + "; ".join(parts)


@observe(name="router_classifier", capture_input=False)
async def _classify_with_llm(
    message: str,
    phone: str = "",
    session: dict | None = None,
) -> tuple[str, float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY missing; using keyword intent fallback")
        return _keyword_fallback_intent(message)

    # SECURITY: Langfuse input — metadata only, not full message body
    set_span_io(
        input_data={
            "message_len": len(message or ""),
            "lead_qualified": bool((session or {}).get("lead_qualified")),
            "last_agent": (session or {}).get("last_agent") or None,
        }
    )
    client = get_async_openai_client(api_key=api_key)
    user_content = (
        f"{_session_context_for_classifier(session)}\n\n"
        f"Buyer message: {message or ''}"
    )
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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


async def classify_intent(
    message: str,
    session: dict,
    db: Any | None = None,
) -> tuple[str, dict]:
    """Classify buyer message and apply qualify-before-order/pricing rules.

    FAQ is available without qualification. Order and pricing require a qualified lead.
    Pure greetings return menu_refresh (no LLM / no FAQ).
    Optional ``db`` enables catalog-aware routing for bare product names → pricing.

    Returns (intent, updated_session).
    """
    session = dict(session or {})

    # --- L1 deterministic ---
    if _matches_human_keyword(message) or is_speak_to_team_request(message):
        session.setdefault("escalation_reason", "speak_to_team")
        return "escalate", session

    if is_discount_request(message):
        session["escalation_reason"] = "discount_request"
        return "escalate", session

    # Bare greetings → main menu (not FAQ / not qualify trap for returning buyers).
    if is_pure_greeting(message):
        return "menu_refresh", session

    key = (message or "").strip().lower()
    if is_order_tracking_message(message):
        return "order", session
    if is_order_account_message(message):
        return "order", session
    if key in ORDER_ACTION_IDS:
        if key == "speak":
            session.setdefault("escalation_reason", "speak_to_team")
            return "escalate", session
        return "order", session
    if session.get("order_state") == SELECT_PAYMENT:
        return "order", session

    menu_intent = _menu_button_intent(message)
    if menu_intent:
        session = mark_menu_selection(session, message)
        if menu_intent == "escalate":
            session.setdefault("escalation_reason", "speak_to_team")
            return "escalate", session
        if menu_intent == "my_orders":
            return "order", session
        # FAQ / already-qualified order+pricing skip qualification forever.
        if not session.get("lead_qualified") and menu_intent in INTENTS_REQUIRING_QUALIFICATION:
            session["pending_intent"] = menu_intent
            session["pending_query"] = message
            return "qualify", session
        return menu_intent, session

    # Catalog product name (e.g. KLENSMART 60MG) → pricing, not FAQ miss loop.
    # Skip when message is clearly FAQ-process or an explicit order request.
    if (
        db is not None
        and not _looks_like_faq_process_question(message)
        and not _looks_like_clear_order_request(message)
        and message_looks_like_catalog_product(message, db)
    ):
        if not session.get("lead_qualified"):
            session["pending_intent"] = "pricing"
            session["pending_query"] = message
            return "qualify", session
        return "pricing", session

    # --- L3 LLM free-text ---
    phone = session.get("phone") or ""
    intent, confidence = await _classify_with_llm(message, phone=phone, session=session)

    if not session.get("lead_qualified"):
        if intent == "escalate":
            return "escalate", session
        # FAQ is available without qualification for new buyers.
        if intent == "faq":
            return "faq", session
        if intent in INTENTS_REQUIRING_QUALIFICATION:
            session["pending_intent"] = intent
            session["pending_query"] = message
            return "qualify", session
        if intent == "qualify":
            return "qualify", session
        session["pending_intent"] = intent
        session["pending_query"] = message
        return "qualify", session

    # Already qualified — never route back to qualify from free-text intents.
    # Greetings already returned menu_refresh above; remaining qualify → FAQ
    # (vague free text with no clear task).
    if intent == "qualify":
        return "faq", session

    if confidence < CONFIDENCE_ESCALATE_THRESHOLD and session.get("lead_qualified"):
        prior_count = session.get("clarification_count", session.get("clarification_attempts", 0))
        count = prior_count + 1
        session["clarification_count"] = count
        session["clarification_attempts"] = count
        if count >= CLARIFICATION_ATTEMPTS_BEFORE_ESCALATE:
            session["clarification_count"] = 0
            session["clarification_attempts"] = 0
            return "escalate", session
        return "faq", session

    session.pop("clarification_count", None)
    session.pop("clarification_attempts", None)
    return intent, session