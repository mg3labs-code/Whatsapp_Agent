"""Restore qualified-buyer state from Postgres when Redis session is missing or expired."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import Lead
from app.session.manager import normalize_phone
from app.utils.security import user_ref

logger = logging.getLogger(__name__)

# In-progress qualification UI flags — safe to drop once lead is terminal.
QUAL_UI_KEYS: tuple[str, ...] = (
    "qual_state",
    "biz_type_picker_sent",
    "country_picker_sent",
    "awaiting_custom_country",
)

LIFECYCLE_DISQUALIFIED = "disqualified"


def phone_lookup_variants(phone: str) -> list[str]:
    """Canonical and legacy (+prefix) forms for leads.phone lookup."""
    normalized = normalize_phone(phone)
    if not normalized:
        return []
    variants = [normalized, f"+{normalized}"]
    return list(dict.fromkeys(variants))


def is_session_disqualified(session: dict | None) -> bool:
    session = session or {}
    return bool(session.get("disqualified")) or (
        session.get("lifecycle_stage") == LIFECYCLE_DISQUALIFIED
    )


def clear_stale_qualification_flags(session: dict) -> dict:
    """Drop mid-qualification prompts once the buyer is permanently terminal.

    Applies when ``lead_qualified`` or ``disqualified`` so stale ``qual_state``
    cannot reopen country/business-type collection.
    """
    session = dict(session or {})
    if not session.get("lead_qualified") and not is_session_disqualified(session):
        return session
    for key in QUAL_UI_KEYS:
        session.pop(key, None)
    if session.get("lead_qualified"):
        session.pop("pending_intent", None)
    return session


def mark_session_qualified(session: dict) -> dict:
    """Mark buyer qualified and clear mid-qual UI state.

    Keeps ``pending_intent`` so qualification completion can hand off to
    pricing/order in the same turn.
    """
    session = dict(session or {})
    session["lead_qualified"] = True
    session.pop("disqualified", None)
    for key in QUAL_UI_KEYS:
        session.pop(key, None)
    return session


def mark_session_disqualified(session: dict, country: str | None = None) -> dict:
    """Permanently lock this phone out of the automated channel."""
    session = dict(session or {})
    session["disqualified"] = True
    session["lead_qualified"] = False
    session["lifecycle_stage"] = LIFECYCLE_DISQUALIFIED
    session["manual_review_only"] = True
    if country:
        session["country"] = country
    for key in QUAL_UI_KEYS:
        session.pop(key, None)
    session.pop("pending_intent", None)
    return session


def lookup_lead_by_phone(db: Session, phone: str) -> Lead | None:
    """Return the most recent lead row for this phone (any stored format)."""
    for variant in phone_lookup_variants(phone):
        lead = (
            db.query(Lead)
            .filter(Lead.phone == variant)
            .order_by(Lead.created_at.desc(), Lead.id.desc())
            .first()
        )
        if lead:
            return lead
    return None


def hydrate_session_from_lead(session: dict, lead: Lead) -> dict:
    """Merge persisted lead fields into the live Redis session."""
    session = dict(session or {})
    stage = (lead.lifecycle_stage or "").strip().lower()
    if stage == LIFECYCLE_DISQUALIFIED:
        session = mark_session_disqualified(session, lead.country)
    else:
        session = mark_session_qualified(session)

    if lead.company:
        session["company"] = lead.company
    if lead.country:
        session["country"] = lead.country
    if lead.business_type:
        session["business_type"] = lead.business_type
    if lead.buyer_type:
        session["buyer_type"] = lead.buyer_type
    if lead.license_number is not None:
        session["license_number"] = lead.license_number
    if lead.annual_volume_usd is not None:
        session["annual_volume_usd"] = float(lead.annual_volume_usd)
    if lead.order_value_usd is not None:
        session["order_value_usd"] = float(lead.order_value_usd)
    if lead.lead_score is not None:
        session["lead_score"] = int(lead.lead_score)
    if lead.lead_category:
        session["lead_category"] = lead.lead_category
    if lead.lifecycle_stage:
        session["lifecycle_stage"] = lead.lifecycle_stage
    if lead.manual_review_only is not None:
        session["manual_review_only"] = bool(lead.manual_review_only)

    if lead.created_at and not session.get("qual_completed_at"):
        created = lead.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        session["qual_completed_at"] = created.isoformat()

    session["hydrated_from_lead"] = True
    return session


def hydrate_session_from_db(phone: str, session: dict, db: Session) -> dict:
    """If buyer is already in leads table, restore qualified or disqualified state."""
    session = dict(session or {})
    if session.get("lead_qualified") or is_session_disqualified(session):
        return clear_stale_qualification_flags(session)

    lead = lookup_lead_by_phone(db, phone)
    if not lead:
        return session

    hydrated = hydrate_session_from_lead(session, lead)
    logger.info(
        "Session hydrated from leads table user_ref=%s lead_id=%s lifecycle=%s",
        user_ref(phone),
        lead.id,
        lead.lifecycle_stage,
    )
    return hydrated
