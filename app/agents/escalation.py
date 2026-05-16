"""Escalation agent and business-hours helpers."""

from __future__ import annotations

from app.business import hours as _hours
from app.business.hours import (
    get_next_business_open_str,
    get_off_hours_notice,
    get_operations_mode,
    get_public_holiday_name,
    is_business_hours,
    is_limited_operations,
    is_public_holiday,
)
from app.integrations.alerts import send_escalation_alert

# Tests monkeypatch wall clock via escalation._now_in_tz
_now_in_tz = _hours._now_in_tz

IN_HOURS_ETA = "within the next 30–60 minutes"
SUPPORT_EMAIL = "exports@newlifemedicare.com"


def _in_hours_reply(company: str | None) -> str:
    greeting = (
        f"I'm connecting you with our sales team right now, {company}!"
        if company
        else "I'm connecting you with our sales team right now!"
    )
    return (
        f"{greeting}\n\n"
        f"Our team will reach out to you directly {IN_HOURS_ETA}.\n"
        f"For urgent matters, you can also reach us at {SUPPORT_EMAIL}\n\n"
        "Reference your phone number when contacting us. 🙏"
    )


def _off_hours_reply() -> str:
    resume = get_next_business_open_str()
    return (
        "Thank you for reaching out to New Life Medicare!\n\n"
        f"Our team is currently offline. We'll respond {resume}.\n"
        "Our AI assistant is available 24/7 and your query has been flagged as a priority.\n\n"
        f"For urgent inquiries: {SUPPORT_EMAIL}"
    )


async def run_escalation_agent(
    message: str,
    session: dict,
    reason: str,
    *,
    phone: str = "",
) -> tuple[str, dict]:
    """Escalate to human team: buyer reply + WhatsApp alerts to ops numbers.

    Returns (reply_text, updated_session) with human_active=True.
    """
    session = dict(session or {})
    buyer_phone = phone or session.get("phone") or ""

    if is_business_hours():
        company = (session.get("company") or "").strip() or None
        reply = _in_hours_reply(company)
    else:
        reply = _off_hours_reply()

    session["human_active"] = True
    session["escalation_reason"] = reason

    if buyer_phone:
        await send_escalation_alert(buyer_phone, session, reason)
    else:
        await send_escalation_alert("unknown", session, reason)

    return reply, session


# Re-export for tests and backward compatibility
__all__ = [
    "get_next_business_open_str",
    "get_off_hours_notice",
    "get_operations_mode",
    "get_public_holiday_name",
    "is_business_hours",
    "is_limited_operations",
    "is_public_holiday",
    "run_escalation_agent",
]
