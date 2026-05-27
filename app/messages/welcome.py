"""Conversation-start templates (AI disclosure + interactive main menu)."""

from __future__ import annotations

from app.integrations.whatsapp import send_interactive_list, send_message
from app.messages.conversation_ui import MENU_OPTION_IDS

AI_DISCLOSURE_MESSAGE = (
    "Hi — I'm the *AI assistant* for *New Life Medicare*\n"
    "I can help with pricing, orders & FAQs. Reply *\"human\"* anytime to reach our team."
)

GREETING_INTRO_TEXT = (
    "Hi there! I'm your *AI sales assistant* at *New Life Medicare* 🏥\n"
    "Your trusted partner for pharmaceutical exports worldwide. 🌍"
)

# Kept for tests/backward compatibility (reply-button shape).
GREETING_BODY_TEXT = (
    f"{GREETING_INTRO_TEXT}\n\n"
    "How can I help you today? Select an option below 👇"
)

GREETING_LIST_HEADER = "New Life Medicare"
GREETING_LIST_BODY = "How can I help you today? Select an option below 👇"
GREETING_LIST_FOOTER = "Pharmaceutical exports worldwide 🌍"
GREETING_LIST_BUTTON = "View Options"

GREETING_LIST_ROWS: list[dict[str, str]] = [
    {
        "id": "order",
        "title": "Place an Order",
        "description": "Start a new purchase order",
    },
    {
        "id": "pricing",
        "title": "Get Pricing",
        "description": "Product quotes and MOQ",
    },
    {
        "id": "faq",
        "title": "FAQs",
        "description": "Shipping, docs and policies",
    },
]

GREETING_BUTTONS: list[dict[str, str]] = [
    {"id": row["id"], "title": row["title"][:20]} for row in GREETING_LIST_ROWS
]

MENU_BUTTON_IDS = MENU_OPTION_IDS

SESSION_FLAG_AI_DISCLOSURE_SENT = "ai_disclosure_sent"
SESSION_FLAG_GREETING_BUTTONS_SENT = "greeting_buttons_sent"


def should_send_ai_disclosure(session: dict | None) -> bool:
    """True when this phone has not yet received the one-time AI disclosure."""
    session = session or {}
    if session.get("human_active"):
        return False
    return not session.get(SESSION_FLAG_AI_DISCLOSURE_SENT)


def should_send_greeting_buttons(session: dict | None) -> bool:
    """True when the interactive main menu has not been sent yet."""
    session = session or {}
    if session.get("human_active"):
        return False
    return not session.get(SESSION_FLAG_GREETING_BUTTONS_SENT)


def prepend_ai_disclosure(reply: str, session: dict | None) -> tuple[str, dict]:
    """Prepend disclosure to the outbound reply and mark session so it is sent once."""
    session = dict(session or {})
    if not should_send_ai_disclosure(session):
        return reply, session

    session[SESSION_FLAG_AI_DISCLOSURE_SENT] = True
    body = (reply or "").strip()
    if not body:
        return AI_DISCLOSURE_MESSAGE, session
    return f"{AI_DISCLOSURE_MESSAGE}\n\n{body}", session


async def send_greeting_menu(
    phone: str,
    session: dict | None,
    *,
    force: bool = False,
) -> tuple[dict, bool]:
    """Send intro text + list menu (View Options), matching MedSource-style UX."""
    session = dict(session or {})
    if not force and not should_send_greeting_buttons(session):
        return session, False

    await send_message(phone, GREETING_INTRO_TEXT)
    ok = await send_interactive_list(
        phone,
        header_text=GREETING_LIST_HEADER,
        body_text=GREETING_LIST_BODY,
        footer_text=GREETING_LIST_FOOTER,
        button_text=GREETING_LIST_BUTTON,
        rows=GREETING_LIST_ROWS,
    )
    if ok:
        session[SESSION_FLAG_GREETING_BUTTONS_SENT] = True
        session[SESSION_FLAG_AI_DISCLOSURE_SENT] = True
    return session, ok
