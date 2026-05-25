"""Lead qualification agent — rule-based multi-turn state machine."""

from __future__ import annotations

import logging
import re
from decimal import Decimal

from sqlalchemy.orm import Session

from app.agents.lead_scoring import (
    enrich_session_from_message,
    map_business_type_to_buyer_type,
    score_lead,
)
from app.business.countries import (
    SHIPMENT_EXCLUDED_REFUSAL,
    is_shipment_excluded_country,
)
from app.db.models import Lead
from app.integrations.alerts import send_escalation_alert

logger = logging.getLogger(__name__)

COLLECT_COMPANY = "COLLECT_COMPANY"
COLLECT_COUNTRY = "COLLECT_COUNTRY"
COLLECT_BIZ_TYPE = "COLLECT_BIZ_TYPE"
COLLECT_VOLUME = "COLLECT_VOLUME"
COLLECT_LICENSE = "COLLECT_LICENSE"
QUAL_COMPLETE = "QUAL_COMPLETE"

CONTINUE_QUAL = "continue_qual"

_GENERIC_WORDS = frozenset({
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
})

HOT_LEAD_MIN_SCORE = 80

_NO_LICENSE_PHRASES = (
    "no",
    "nope",
    "none",
    "n/a",
    "na",
    "don't have",
    "do not have",
    "dont have",
    "not yet",
    "no license",
    "without license",
    "i don't",
)


def calculate_lead_score(lead: dict) -> int:
    """Return final lead score (client SOP)."""
    return score_lead(lead).score


def _extract_company(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) == 1:
        return lines[0]
    first_sentence = re.split(r"[.!?]", lines[0])[0].strip()
    return first_sentence or lines[0]


def _extract_business_type(text: str) -> str:
    lowered = (text or "").lower()
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


def _extract_volume_usd(text: str) -> float | None:
    raw = (text or "").strip().lower()
    if not raw:
        return None

    million = re.search(
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:million|mil\b|m\b)",
        raw,
    )
    if million:
        return float(million.group(1).replace(",", "")) * 1_000_000

    thousand = re.search(
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:thousand|k\b)",
        raw,
    )
    if thousand:
        return float(thousand.group(1).replace(",", "")) * 1_000

    money = re.search(r"\$?\s*(\d[\d,]*(?:\.\d+)?)", raw)
    if money:
        value = float(money.group(1).replace(",", ""))
        if "k" in raw and value < 1000:
            value *= 1000
        return value

    return None


def _extract_license_number(text: str) -> str | None:
    stripped = (text or "").strip()
    if not stripped:
        return None

    lowered = stripped.lower()
    if any(phrase in lowered for phrase in _NO_LICENSE_PHRASES):
        return None

    code_match = re.search(r"[A-Za-z0-9][A-Za-z0-9\-/]{3,}", stripped)
    if code_match:
        return stripped

    if re.search(r"\d", stripped):
        return stripped

    return None


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
    state = session.get("qual_state") or COLLECT_COMPANY
    session["qual_state"] = state

    if state == COLLECT_COMPANY:
        return _handle_collect_company(text, session)
    if state == COLLECT_COUNTRY:
        return _handle_collect_country(text, session)
    if state == COLLECT_BIZ_TYPE:
        return _handle_collect_biz_type(text, session)
    if state == COLLECT_VOLUME:
        return _handle_collect_volume(text, session)
    if state == COLLECT_LICENSE:
        return await _handle_collect_license(text, session, db)
    if state == QUAL_COMPLETE:
        return await _handle_qual_complete(session, db, message)

    session["qual_state"] = COLLECT_COMPANY
    return _prompt_collect_company(), session, CONTINUE_QUAL


def _is_generic_reply(text: str) -> bool:
    return (text or "").lower().strip() in _GENERIC_WORDS


def _handle_collect_company(text: str, session: dict) -> tuple[str, dict, str]:
    if not text or _is_generic_reply(text):
        return _prompt_collect_company(), session, CONTINUE_QUAL

    company = _extract_company(text)
    if len(company) < 3:
        return (
            "Please share your company name so we can continue.",
            session,
            CONTINUE_QUAL,
        )

    session["company"] = company
    session["qual_state"] = COLLECT_COUNTRY
    return "And which country are you based in?", session, CONTINUE_QUAL


def _handle_collect_country(text: str, session: dict) -> tuple[str, dict, str]:
    country = text.strip()
    if not country or _is_generic_reply(text):
        return "And which country are you based in?", session, CONTINUE_QUAL

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


def _handle_collect_biz_type(text: str, session: dict) -> tuple[str, dict, str]:
    if not text.strip() or _is_generic_reply(text):
        return (
            "What type of business are you? (distributor, pharmacy/clinic, doctor, "
            "or independent buyer)",
            session,
            CONTINUE_QUAL,
        )

    session["business_type"] = _extract_business_type(text)
    session["buyer_type"] = map_business_type_to_buyer_type(session["business_type"], text)
    session["qual_state"] = COLLECT_VOLUME
    return (
        "What is your typical order value in USD for one purchase? "
        "(e.g., $150, $500, or $2,000 — or share annual volume like $500,000)",
        session,
        CONTINUE_QUAL,
    )


def _handle_collect_volume(text: str, session: dict) -> tuple[str, dict, str]:
    volume = _extract_volume_usd(text)
    if volume is None:
        return (
            "Could you share an approximate USD amount? (e.g., $500 per order or $100,000/year)",
            session,
            CONTINUE_QUAL,
        )

    if volume >= 10_000:
        session["annual_volume_usd"] = volume
    else:
        session["order_value_usd"] = volume
    session["qual_state"] = COLLECT_LICENSE
    return (
        "Do you hold a pharmaceutical import/distribution license? "
        "If yes, please share the license number. (This field is optional)",
        session,
        CONTINUE_QUAL,
    )


async def _handle_collect_license(
    text: str,
    session: dict,
    db: Session,
) -> tuple[str, dict, str]:
    if not text.strip():
        return (
            "Do you hold a pharmaceutical import/distribution license? "
            "If yes, please share the license number. (This field is optional)",
            session,
            CONTINUE_QUAL,
        )

    session["license_number"] = _extract_license_number(text)
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

    if not session.get("buyer_type"):
        session["buyer_type"] = map_business_type_to_buyer_type(
            session.get("business_type"), message
        )

    phone = session.get("phone") or ""
    order_val = session.get("order_value_usd")
    lead = Lead(
        phone=phone,
        company=session.get("company"),
        country=session.get("country"),
        business_type=session.get("business_type"),
        buyer_type=session.get("buyer_type"),
        license_number=session.get("license_number"),
        annual_volume_usd=Decimal(str(session.get("annual_volume_usd") or 0)),
        order_value_usd=Decimal(str(order_val)) if order_val else None,
        lead_score=result.score,
        lead_category=result.category,
        lifecycle_stage="qualified",
        manual_review_only=result.manual_review_only,
    )
    db.add(lead)
    db.commit()

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

    reply = (
        "Thank you! I now have everything I need. Let me help with your query..."
    )
    next_intent = session.get("pending_intent") or "faq"
    return reply, session, next_intent


def _prompt_collect_company() -> str:
    return (
        "Welcome to New Life Medicare! To provide accurate pricing and ensure "
        "compliance with export regulations, may I get your company name and country?"
    )
