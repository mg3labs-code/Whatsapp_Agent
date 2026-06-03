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
from app.integrations.alerts import send_escalation_alert
from app.integrations.whatsapp import send_interactive_buttons
from app.messages.conversation_ui import MAIN_MENU_ID, MENU_OPTION_IDS

logger = logging.getLogger(__name__)

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
    {"id": "pricing", "title": "Get Pricing"},
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
    if any(k in lowered for k in ("pharmacy chain", "chain pharmacy", "retail chain")):
        return "pharmacy_chain"
    if any(k in lowered for k in ("hospital", "clinic", "medical center")):
        return "hospital"
    if any(k in lowered for k in ("distributor", "wholesale", "wholesaler", "resell", "bulk")):
        return "distributor"
    if any(k in lowered for k in ("doctor", "physician", "prescriber")):
        return "doctor"
    if any(k in lowered for k in ("pharmacy", "chemist", "drugstore")):
        return "pharmacy"
    return "other"


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

    if state == COLLECT_COUNTRY:
        return _handle_collect_country(text, session)
    if state == COLLECT_BIZ_TYPE:
        return await _handle_collect_biz_type(text, session, db)
    if state == QUAL_COMPLETE:
        return await _handle_qual_complete(session, db, message)

    session["qual_state"] = COLLECT_COUNTRY
    return _prompt_collect_country(), session, CONTINUE_QUAL


def _is_filler_reply(text: str) -> bool:
    return (text or "").lower().strip() in _FILLER_WORDS


def _is_generic_reply(text: str) -> bool:
    lowered = (text or "").lower().strip()
    return (
        lowered in _FILLER_WORDS
        or lowered in MENU_OPTION_IDS
        or lowered == MAIN_MENU_ID
    )


def _handle_collect_country(text: str, session: dict) -> tuple[str, dict, str]:
    if not text or _is_generic_reply(text) or _is_filler_reply(text):
        return _prompt_collect_country(), session, CONTINUE_QUAL

    country = _extract_country(text)
    if not country or not _is_plausible_country(country):
        return _prompt_collect_country(), session, CONTINUE_QUAL

    if is_shipment_excluded_country(country):
        return SHIPMENT_EXCLUDED_REFUSAL, session, CONTINUE_QUAL

    session["country"] = country
    session["qual_state"] = COLLECT_BIZ_TYPE
    return (
        "What type of business are you? (distributor, pharmacy/clinic, doctor, "
        "or independent buyer)",
        session,
        CONTINUE_QUAL,
    )


async def _handle_collect_biz_type(
    text: str,
    session: dict,
    db: Session,
) -> tuple[str, dict, str]:
    if (
        not text.strip()
        or _is_generic_reply(text)
        or _is_filler_reply(text)
        or len(text.strip()) < 3
    ):
        return (
            "What type of business are you? (distributor, pharmacy/clinic, doctor, "
            "or independent buyer)",
            session,
            CONTINUE_QUAL,
        )

    session["business_type"] = _extract_business_type(text)
    session["buyer_type"] = map_business_type_to_buyer_type(session["business_type"], text)
    session["qual_state"] = QUAL_COMPLETE
    return await _handle_qual_complete(session, db, text)


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
        "• 💊 *Order medicines* — share product names or a list\n"
        "• 💰 *Get pricing* — ask about any product\n"
        "• ❓ *FAQs* — shipping, documents, timelines\n\n"
        "Or reply *human* to speak with our team."
    )

    if phone:
        await send_interactive_buttons(
            phone,
            "Choose an option below:",
            _QUAL_COMPLETE_BUTTONS,
        )

    return reply, session, next_intent


def _prompt_collect_country() -> str:
    return (
        "Welcome to New Life Medicare! To provide accurate pricing and ensure "
        "compliance with export regulations, which country are you based in?"
    )


def _prompt_collect_company() -> str:
    """Backward-compatible alias for FAQ/tests that referenced the old prompt."""
    return _prompt_collect_country()
