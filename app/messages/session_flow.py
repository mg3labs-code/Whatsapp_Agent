"""Shared session helpers — human handoff resume, order reset, action buttons."""

from __future__ import annotations

import re

RESUME_BOT_ID = "resume_bot"

GREETING_IDS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening",
    }
)

RESUME_AFTER_HANDOFF_IDS = frozenset(
    {
        "main_menu",
        "order",
        "pricing",
        "faq",
        "new_order",
        "order_status",
        "my_orders",
        RESUME_BOT_ID,
        "menu",
        "menu_refresh",
    }
) | GREETING_IDS

ORDER_RESET_IDS = frozenset(
    {
        "new_order",
        "main_menu",
        "cancel",
        "clear",
        "reset",
        "stop",
        "abort",
        "start_over",
    }
)

ORDER_RESET_PHRASES = (
    "new order",
    "main menu",
    "start over",
    "start fresh",
    "not needed",
    "don't need",
    "do not need",
    "cancel order",
    "clear cart",
    "i need new order",
    "need new order",
    "not needed cancel",
)

DISCOUNT_KEYWORDS = (
    "discount",
    "discopunt",
    "discout",
    "best price",
    "lower price",
    "cheaper",
    "price reduction",
    "special price",
    "better rate",
    "reduce price",
    "rebate",
    "bulk deal",
)

SPEAK_TO_TEAM_KEYWORDS = (
    "speak to team",
    "talk to team",
    "speak to someone",
    "talk to someone",
    "customer service",
    "representative",
    "real person",
    "human agent",
    "connect me",
    "transfer me",
    "i want to speak",
    "need to speak",
)

CART_ACTION_BUTTONS = [
    {"id": "checkout", "title": "Checkout"},
    {"id": "add", "title": "Add More"},
]

CONFIRM_ORDER_BUTTONS = [
    {"id": "confirm", "title": "Confirm Order"},
    {"id": "edit", "title": "Edit Cart"},
]

PRODUCT_CONFIRM_BUTTONS = [
    {"id": "confirm", "title": "Yes"},
    {"id": "reject", "title": "No"},
]

RESUME_BOT_BUTTONS = [{"id": RESUME_BOT_ID, "title": "Continue with Bot"}]

BIZ_TYPE_ROWS: list[dict[str, str]] = [
    {"id": "biz_distributor", "title": "Distributor", "description": "Wholesale / bulk buyer"},
    {"id": "biz_pharmacy", "title": "Pharmacy / Clinic", "description": "Retail pharmacy or clinic"},
    {"id": "biz_doctor", "title": "Doctor", "description": "Prescriber / physician"},
    {"id": "biz_independent", "title": "Independent Buyer", "description": "Personal or small buyer"},
]

BIZ_TYPE_BUTTON_IDS = frozenset(row["id"] for row in BIZ_TYPE_ROWS)

BIZ_TYPE_ID_TO_LABEL: dict[str, str] = {
    "biz_distributor": "distributor wholesaler",
    "biz_pharmacy": "pharmacy clinic",
    "biz_doctor": "doctor physician",
    "biz_independent": "independent buyer",
}

_TYPED_BIZ_SHORTCUTS: dict[str, str] = {
    "clinic": "pharmacy clinic",
    "pharmacy": "pharmacy clinic",
    "chemist": "pharmacy clinic",
    "drugstore": "pharmacy clinic",
    "dr": "doctor physician",
    "gp": "doctor physician",
    "doc": "doctor physician",
    "distributor": "distributor wholesaler",
    "wholesale": "distributor wholesaler",
    "wholesaler": "distributor wholesaler",
    "independent": "independent buyer",
    "buyer": "independent buyer",
}


def is_greeting_message(message: str) -> bool:
    key = (message or "").strip().lower()
    if key in GREETING_IDS:
        return True
    return key.startswith(("hi ", "hello ", "hey "))


def is_discount_request(message: str) -> bool:
    text = (message or "").lower()
    return any(keyword in text for keyword in DISCOUNT_KEYWORDS)


def is_speak_to_team_request(message: str) -> bool:
    text = (message or "").lower()
    if text == "speak":
        return True
    return any(keyword in text for keyword in SPEAK_TO_TEAM_KEYWORDS)


def should_resume_from_human_handoff(message: str) -> bool:
    """Buyer wants the AI assistant again after a team handoff."""
    key = (message or "").strip().lower()
    if key in RESUME_AFTER_HANDOFF_IDS:
        return True
    if is_greeting_message(message):
        return True
    if key in {"faqs", "faq"}:
        return True
    return False


def clear_human_handoff(session: dict) -> dict:
    session = dict(session or {})
    session.pop("human_active", None)
    session.pop("escalation_reason", None)
    return session


def is_order_reset_request(message: str) -> bool:
    key = (message or "").strip().lower()
    if not key:
        return False
    if key in ORDER_RESET_IDS:
        return True
    if key.replace(" ", "_") in ORDER_RESET_IDS:
        return True
    return any(phrase in key for phrase in ORDER_RESET_PHRASES)


def resolve_business_type_button(text: str) -> str | None:
    """Backward-compatible alias."""
    return resolve_business_type_selection(text)


def resolve_business_type_selection(text: str) -> str | None:
    """Map list ids, titles, descriptions, or typed text to a business label."""
    raw = (text or "").strip()
    if not raw:
        return None

    key = raw.lower()
    if key in BIZ_TYPE_ID_TO_LABEL:
        return BIZ_TYPE_ID_TO_LABEL[key]
    if key in _TYPED_BIZ_SHORTCUTS:
        return _TYPED_BIZ_SHORTCUTS[key]

    normalized = re.sub(r"\s+", " ", key)
    for row in BIZ_TYPE_ROWS:
        row_id = row["id"]
        title = row["title"].lower()
        description = row["description"].lower()
        label = BIZ_TYPE_ID_TO_LABEL[row_id]

        if normalized == title or normalized == description:
            return label
        if title in normalized or normalized in title:
            return label
        if description in normalized or normalized in description:
            return label
        # WhatsApp sometimes shows "Title / Description" in the chat bubble.
        combined = f"{title} / {description}"
        if normalized == combined or combined in normalized or normalized in combined:
            return label

    if len(normalized) >= 3:
        return normalized
    return None
