"""Lead qualification agent — country + business type only."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.agents.lead_scoring import (
    enrich_session_from_message,
    map_business_type_to_buyer_type,
    score_lead,
)
from app.business.countries import (
    SHIPMENT_EXCLUDED_REFUSAL,
    classify_country,
    is_shipment_excluded_country,
)
from app.session.lead_persistence import upsert_lead_from_session
from app.session.manager import normalize_phone
from app.utils.security import user_ref
from app.integrations.alerts import send_escalation_alert
from app.integrations.whatsapp import send_interactive_list
from app.messages.conversation_ui import MAIN_MENU_ID, MENU_OPTION_IDS, send_main_menu_list
from app.messages.onboarding import (
    COUNTRY_BUTTON_IDS,
    SESSION_AWAITING_CUSTOM_COUNTRY,
    country_prompt,
    custom_country_prompt,
    resolve_country_button,
    send_country_picker,
)
from app.messages.session_flow import (
    BIZ_TYPE_ROWS,
    resolve_business_type_selection,
)

logger = logging.getLogger(__name__)

SESSION_BIZ_TYPE_PICKER_SENT = "biz_type_picker_sent"

COLLECT_COUNTRY = "COLLECT_COUNTRY"
COLLECT_BIZ_TYPE = "COLLECT_BIZ_TYPE"
QUAL_COMPLETE = "QUAL_COMPLETE"

# Legacy states from older deployments — map to current steps
_LEGACY_COMPANY = "COLLECT_COMPANY"

CONTINUE_QUAL = "continue_qual"

_FILLER_WORDS = frozenset({
    "hi",
    "hello",
    "hey",
    "ok",
    "okay",
    "yes",
    "no",
    "sure",
    "thanks",
    "thank you",
    "pls",
    "please",
    "good",
    "fine",
    "noted",
    "k",
    "kk",
    "test",
    "testing",
    "nothing",
    "none",
    "na",
    "n/a",
})

_QUAL_COMPLETE_BUTTONS = [
    {"id": "order", "title": "Order Medicines"},
    {"id": "my_orders", "title": "My Orders"},
    {"id": "speak", "title": "Speak to Team"},
]

_PENDING_INTENT_HANDOFF: dict[str, str] = {
    "order": (
        "Thank you! ✅ You're all set.\n\n"
        "Which product(s) would you like to order? Share product names or SKUs."
    ),
    "pricing": (
        "Thank you! ✅ You're all set.\n\n"
        "Please share the product name and quantity you need a quote for."
    ),
}

HOT_LEAD_MIN_SCORE = 80


def calculate_lead_score(lead: dict) -> int:
    """Return final lead score (client SOP)."""
    return score_lead(lead).score


def _normalize_qual_state(state: str | None) -> str:
    if not state or state in {_LEGACY_COMPANY, "COLLECT_VOLUME", "COLLECT_LICENSE"}:
        return COLLECT_COUNTRY
    if state == COLLECT_BIZ_TYPE:
        return COLLECT_BIZ_TYPE
    if state == QUAL_COMPLETE:
        return QUAL_COMPLETE
    return COLLECT_COUNTRY


def _is_plausible_country(candidate: str) -> bool:
    """Reject full sentences mistaken for a country name."""
    stripped = (candidate or "").strip()
    if len(stripped) < 2 or len(stripped) > 45:
        return False
    if len(stripped.split()) > 4:
        return False
    tier = classify_country(stripped)
    if tier == "missing" and len(stripped.split()) > 1:
        return False
    return True


def _extract_country(text: str) -> str | None:
    """Parse country from replies like 'USA', 'from UK', or 'Nairobi, Kenya'."""
    stripped = (text or "").strip()
    if not stripped:
        return None

    from_match = re.search(
        r"\bfrom\s+([A-Za-z][A-Za-z\s\-]{1,60})$",
        stripped,
        re.IGNORECASE,
    )
    if from_match:
        return from_match.group(1).strip()

    if "," in stripped:
        parts = [p.strip() for p in stripped.split(",", 1)]
        if len(parts) == 2 and len(parts[1]) >= 2:
            return parts[1]

    return stripped if len(stripped) >= 2 else None


def _extract_business_type(text: str) -> str:
    lowered = (text or "").lower()
    if any(k in lowered for k in ("independent buyer", "independent", "personal buyer")):
        return "independent_buyer"
    if any(k in lowered for k in ("distributor", "wholesale", "wholesaler", "resell", "bulk")):
        return "distributor"
    if any(k in lowered for k in ("doctor", "physician", "prescriber")):
        return "doctor"
    if any(
        k in lowered
        for k in ("pharmacy", "chemist", "drugstore", "clinic", "pharmacy chain", "retail pharmacy")
    ):
        return "pharmacy"
    if any(k in lowered for k in ("hospital", "medical center")):
        return "hospital"
    return "other"


async def _send_business_type_picker(phone: str) -> None:
    await send_interactive_list(
        phone,
        header_text="New Life Medicare",
        body_text="What type of business are you?",
        footer_text="Or type your business type",
        button_text="Select Type",
        rows=BIZ_TYPE_ROWS,
        section_title="Business Types",
    )


async def run_qualification_agent(
    message: str,
    session: dict,
    db: Session,
) -> tuple[str, dict, str]:
    """Run one turn of the qualification state machine.

    Returns (reply_text, updated_session, next_intent).
    """
    session = dict(session or {})
    session = enrich_session_from_message(session, message)
    text = (message or "").strip()
    state = _normalize_qual_state(session.get("qual_state"))
    session["qual_state"] = state
    phone = session.get("phone") or ""

    if state == COLLECT_COUNTRY:
        return await _handle_collect_country(text, session, phone)
    if state == COLLECT_BIZ_TYPE:
        return await _handle_collect_biz_type(text, session, db)
    if state == QUAL_COMPLETE:
        return await _handle_qual_complete(session, db, message)

    session["qual_state"] = COLLECT_COUNTRY
    session = await send_country_picker(phone, session)
    return country_prompt(), session, CONTINUE_QUAL


def _is_filler_reply(text: str) -> bool:
    return (text or "").lower().strip() in _FILLER_WORDS


def _is_generic_reply(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return (
        lowered in _FILLER_WORDS
        or lowered in MENU_OPTION_IDS
        or lowered == MAIN_MENU_ID
    )


async def _handle_collect_country(
    text: str, session: dict, phone: str
) -> tuple[str, dict, str]:
    session = await send_country_picker(phone, session)

    if session.get(SESSION_AWAITING_CUSTOM_COUNTRY):
        if not text or _is_generic_reply(text) or _is_filler_reply(text):
            return custom_country_prompt(), session, CONTINUE_QUAL
        country = _extract_country(text)
        if not country or not _is_plausible_country(country):
            return custom_country_prompt(), session, CONTINUE_QUAL
        session.pop(SESSION_AWAITING_CUSTOM_COUNTRY, None)
        return await _finalize_country(country, session, phone)

    if not text or _is_generic_reply(text) or _is_filler_reply(text):
        return country_prompt(reminded=bool(session.get("country_picker_sent"))), session, CONTINUE_QUAL

    key = text.strip().lower()
    if key in COUNTRY_BUTTON_IDS:
        resolved, follow_up = resolve_country_button(key)
        if follow_up:
            session[SESSION_AWAITING_CUSTOM_COUNTRY] = True
            return follow_up, session, CONTINUE_QUAL
        if resolved:
            return await _finalize_country(resolved, session, phone)

    country = _extract_country(text)
    if not country or not _is_plausible_country(country):
        return country_prompt(reminded=True), session, CONTINUE_QUAL

    return await _finalize_country(country, session, phone)


async def _finalize_country(
    country: str, session: dict, phone: str = ""
) -> tuple[str, dict, str]:
    if is_shipment_excluded_country(country):
        return SHIPMENT_EXCLUDED_REFUSAL, session, CONTINUE_QUAL

    session["country"] = country
    session["qual_state"] = COLLECT_BIZ_TYPE
    session.pop(SESSION_BIZ_TYPE_PICKER_SENT, None)
    if phone:
        await _send_business_type_picker(phone)
        session[SESSION_BIZ_TYPE_PICKER_SENT] = True
    return (
        "Great! Now select your business type from the list below 👇 "
        "or type it (e.g. *clinic*, *doctor*, *distributor*).",
        session,
        CONTINUE_QUAL,
    )


async def _handle_collect_biz_type(
    text: str,
    session: dict,
    db: Session,
) -> tuple[str, dict, str]:
    phone = normalize_phone(session.get("phone") or "")
    if phone and not session.get(SESSION_BIZ_TYPE_PICKER_SENT):
        await _send_business_type_picker(phone)
        session[SESSION_BIZ_TYPE_PICKER_SENT] = True

    if not text.strip() or _is_filler_reply(text) or _is_generic_reply(text):
        return (
            "Select your business type from the list above 👆 or type it "
            "(e.g. *clinic*, *doctor*, *distributor*).",
            session,
            CONTINUE_QUAL,
        )

    parsed = resolve_business_type_selection(text)
    if not parsed:
        return (
            "I didn't catch that. Tap *Select Type* above or type "
            "*clinic*, *doctor*, *distributor*, or *independent buyer*.",
            session,
            CONTINUE_QUAL,
        )

    session["business_type"] = _extract_business_type(parsed)
    session["buyer_type"] = map_business_type_to_buyer_type(session["business_type"], parsed)
    session.pop(SESSION_BIZ_TYPE_PICKER_SENT, None)
    session["qual_state"] = QUAL_COMPLETE
    return await _handle_qual_complete(session, db, parsed)


async def _apply_escalation_handoff(session: dict, reason: str) -> dict:
    """Mark human takeover and notify leads team (WhatsApp DMs)."""
    session["human_active"] = True
    session["escalation_reason"] = reason
    phone = session.get("phone") or ""
    if phone:
        await send_escalation_alert(phone, session, reason)
    return session


async def _handle_qual_complete(
    session: dict,
    db: Session,
    message: str = "",
) -> tuple[str, dict, str]:
    result = score_lead(session, message)
    session["lead_score"] = result.score
    session["lead_category"] = result.category
    session["lead_qualified"] = True
    session["manual_review_only"] = result.manual_review_only
    session["lifecycle_stage"] = "qualified"
    session.pop("qual_state", None)
    session["qual_completed_at"] = datetime.now(timezone.utc).isoformat()

    if not session.get("buyer_type"):
        session["buyer_type"] = map_business_type_to_buyer_type(
            session.get("business_type"), message
        )

    phone = normalize_phone(session.get("phone") or "")
    session["phone"] = phone
    upsert_lead_from_session(session, db)

    if result.disqualified:
        reply = (
            "Thank you for your interest. We're unable to process this request through "
            "our automated channel due to export compliance requirements. "
            "Our compliance team will review and contact you if applicable."
        )
        return reply, session, "escalate"

    if result.manual_review_only:
        reply = (
            "Thank you for the details. Your enquiry requires a compliance review by "
            "our specialist team. We'll contact you shortly."
        )
        session = await _apply_escalation_handoff(session, "manual_review")
        return reply, session, "escalate"

    if result.score >= HOT_LEAD_MIN_SCORE:
        reply = (
            "Thank you for the information! Based on your business profile, "
            "I'd like to connect you with our Senior Export Manager directly. "
            "They'll reach out to you shortly."
        )
        session = await _apply_escalation_handoff(session, "hot_lead")
        return reply, session, "escalate"

    pending = session.pop("pending_intent", None)
    if pending in _PENDING_INTENT_HANDOFF:
        return _PENDING_INTENT_HANDOFF[pending], session, pending

    next_intent = pending or "faq"
    reply = (
        "Thank you! ✅ You're all set.\n\n"
        "What would you like to do?\n"
        "• 💊 *Place an order* — new purchase\n"
        "• 📋 *My orders* — view past & pending orders\n"
        "• 💰 *Get pricing* — product quotes\n"
        "• ❓ *FAQs* — shipping, documents, timelines\n"
        "• 👤 *Speak to team* — connect with us"
    )

    if phone:
        try:
            await send_main_menu_list(phone)
        except Exception:
            logger.exception("send_main_menu_list failed user_ref=%s", user_ref(phone))

    return reply, session, next_intent


def _prompt_collect_country() -> str:
    return country_prompt()


def _prompt_collect_company() -> str:
    """Backward-compatible alias for FAQ/tests that referenced the old prompt."""
    return _prompt_collect_country()
