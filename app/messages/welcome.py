"""Conversation-start templates (AI disclosure for Meta transparency)."""

from __future__ import annotations

AI_DISCLOSURE_MESSAGE = (
    "Hi — I'm the *AI assistant* for *New Life Medicare*\n"
    "I can help with pricing, orders & FAQs. Reply *\"human\"* anytime to reach our team."
)

SESSION_FLAG_AI_DISCLOSURE_SENT = "ai_disclosure_sent"


def should_send_ai_disclosure(session: dict | None) -> bool:
    """True when this phone has not yet received the one-time AI disclosure."""
    session = session or {}
    if session.get("human_active"):
        return False
    return not session.get(SESSION_FLAG_AI_DISCLOSURE_SENT)


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
