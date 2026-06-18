"""Conversation-start templates (AI disclosure + interactive main menu)."""

from __future__ import annotations

from app.integrations.whatsapp import send_message
from app.messages.conversation_ui import (
    MAIN_MENU_BODY,
    MAIN_MENU_LIST_ROWS,
    MENU_OPTION_IDS,
    send_main_menu_list,
)

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
GREETING_LIST_BODY = MAIN_MENU_BODY
GREETING_LIST_FOOTER = "Pharmaceutical exports worldwide 🌍"
GREETING_LIST_BUTTON = "View Options"
GREETING_LIST_ROWS = MAIN_MENU_LIST_ROWS

GREETING_BUTTONS: list[dict[str, str]] = [
    {"id": row["id"], "title": row["title"][:20]} for row in MAIN_MENU_LIST_ROWS[:3]
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
    ok = await send_main_menu_list(phone, body=GREETING_LIST_BODY)
    if ok:
        session[SESSION_FLAG_GREETING_BUTTONS_SENT] = True
        session[SESSION_FLAG_AI_DISCLOSURE_SENT] = True
    return session, ok
