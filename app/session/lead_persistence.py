"""Persist qualification results to leads table (normalized phone, upsert)."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy.orm import Session

from app.db.models import Lead
from app.session.lead_hydration import lookup_lead_by_phone
from app.session.manager import normalize_phone


def upsert_lead_from_session(session: dict, db: Session) -> Lead:
    """Insert or update lead row using canonical phone (digits only, no '+')."""
    phone = normalize_phone(session.get("phone") or "")
    if not phone:
        raise ValueError("phone required to upsert lead")

    order_val = session.get("order_value_usd")
    fields = {
        "company": session.get("company"),
        "country": session.get("country"),
        "business_type": session.get("business_type"),
        "buyer_type": session.get("buyer_type"),
        "license_number": session.get("license_number"),
        "annual_volume_usd": Decimal(str(session.get("annual_volume_usd") or 0)),
        "order_value_usd": Decimal(str(order_val)) if order_val else None,
        "lead_score": session.get("lead_score"),
        "lead_category": session.get("lead_category"),
        "lifecycle_stage": session.get("lifecycle_stage") or "qualified",
        "manual_review_only": bool(session.get("manual_review_only")),
    }

    existing = lookup_lead_by_phone(db, phone)
    if existing:
        existing.phone = phone
        for key, value in fields.items():
            setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing

    lead = Lead(phone=phone, **fields)
    db.add(lead)
    db.commit()
    db.refresh(lead)
    return lead
