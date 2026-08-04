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
from app.session.lead_hydration import is_session_disqualified

# Tests monkeypatch wall clock via escalation._now_in_tz
_now_in_tz = _hours._now_in_tz

SUPPORT_EMAIL = "exports@newlifemedicare.com"

# Soft follow-up — no minute/hour SLA commitments in buyer copy.
IN_HOURS_FOLLOW_UP = "Our team will follow up with you soon."

# Optional pre-authored buyer copy set by qualify / pricing before routing here.
SESSION_ESCALATION_BUYER_REPLY = "escalation_buyer_reply"


# Soft handoff (buyer can keep using the bot) vs hard compliance lock.
COMPLIANCE_LOCK_REASONS = frozenset(
    {
        "disqualified",
        "excluded_country",
        "excluded_country_inquiry",
    }
)


def _in_hours_reply(company: str | None) -> str:
    greeting = (
        f"I'm connecting you with our export team right now, {company}!"
        if company
        else "I'm connecting you with our export team right now!"
    )
    return (
        f"{greeting}\n\n"
        f"{IN_HOURS_FOLLOW_UP}\n"
        f"For urgent matters, you can also reach us at {SUPPORT_EMAIL}\n\n"
        "Reference your phone number when contacting us. 🙏"
    )


def _off_hours_reply() -> str:
    resume = get_next_business_open_str()
    return (
        "Thank you for reaching out to New Life Medicare!\n\n"
        f"Our team is currently offline. We'll follow up when we're back "
        f"({resume}).\n"
        "Our AI assistant is available 24/7 and your query has been flagged "
        "as a priority.\n\n"
        f"For urgent inquiries: {SUPPORT_EMAIL}"
    )


def _default_reply_for_reason(reason: str, session: dict) -> str | None:
    """Reason-specific single messages (used when no custom reply is queued)."""
    key = (reason or "").strip().lower()
    if key in {"disqualified", "excluded_country", "excluded_country_inquiry"}:
        return (
            "Thank you for your interest. We're unable to process this request through "
            "our automated channel due to export compliance requirements. "
            "Our compliance team has been notified and will follow up if applicable."
        )
    if key == "manual_review":
        return (
            "Thank you for the details. Your enquiry needs a compliance review. "
            "Our team has been notified and will follow up with you."
        )
    if key == "hot_lead":
        return (
            "Thank you for the information! I'm connecting you with our export team now. "
            "They'll follow up with you soon."
        )
    if key in {"pricing_outage", "pricing_error"}:
        return (
            "I'm having trouble checking pricing right now. "
            "I've notified our team — they'll follow up with you."
        )
    return None


async def run_escalation_agent(
    message: str,
    session: dict,
    reason: str,
    *,
    phone: str = "",
) -> tuple[str, dict]:
    """Escalate to human team: one buyer reply + WhatsApp alerts to ops numbers.

    If ``session[escalation_buyer_reply]`` is set, that text is used (single message).
    Otherwise reason-specific or business-hours default copy is used.
    Returns (reply_text, updated_session) with human_active=True.
    """
    session = dict(session or {})
    buyer_phone = phone or session.get("phone") or ""
    reason = (reason or "buyer_request").strip() or "buyer_request"

    custom = (session.pop(SESSION_ESCALATION_BUYER_REPLY, None) or "").strip()
    if custom:
        reply = custom
    else:
        reason_reply = _default_reply_for_reason(reason, session)
        if reason_reply:
            reply = reason_reply
        elif is_business_hours():
            company = (session.get("company") or "").strip() or None
            reply = _in_hours_reply(company)
        else:
            reply = _off_hours_reply()

    reason_key = reason.strip().lower()
    compliance_lock = reason_key in COMPLIANCE_LOCK_REASONS or is_session_disqualified(
        session
    )

    if compliance_lock:
        # Hard lock: no "Continue with Bot" / human_active resume loop.
        session["human_active"] = False
        session["escalation_reason"] = reason
        session["compliance_locked"] = True
    else:
        session["human_active"] = True
        session["escalation_reason"] = reason

    # Avoid re-alerting every compliance retry (pre_guardrails also alerts once).
    should_alert = True
    if compliance_lock and session.get("_compliance_block_alerted"):
        should_alert = False
    if should_alert and buyer_phone:
        await send_escalation_alert(buyer_phone, session, reason)
        if compliance_lock:
            session["_compliance_block_alerted"] = True
    elif should_alert:
        await send_escalation_alert("unknown", session, reason)
        if compliance_lock:
            session["_compliance_block_alerted"] = True

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
    "SESSION_ESCALATION_BUYER_REPLY",
    "IN_HOURS_FOLLOW_UP",
    "SUPPORT_EMAIL",
]
