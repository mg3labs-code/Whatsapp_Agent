"""Tests for Postgres lead hydration into Redis session."""

import pytest

from app.db.models import Base, Lead
from app.session.lead_hydration import (
    clear_stale_qualification_flags,
    hydrate_session_from_db,
    hydrate_session_from_lead,
    lookup_lead_by_phone,
    mark_session_disqualified,
    mark_session_qualified,
    phone_lookup_variants,
)
from app.session.manager import normalize_phone


@pytest.fixture
def lead_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_phone_lookup_variants():
    assert phone_lookup_variants("+919876543210") == ["919876543210", "+919876543210"]


def test_clear_stale_qualification_flags_only_when_qualified():
    unclean = {
        "lead_qualified": True,
        "qual_state": "COLLECT_BIZ_TYPE",
        "biz_type_picker_sent": True,
        "pending_intent": "pricing",
        "country": "Kenya",
    }
    cleaned = clear_stale_qualification_flags(unclean)
    assert cleaned["lead_qualified"] is True
    assert cleaned["country"] == "Kenya"
    assert cleaned.get("qual_state") is None
    assert cleaned.get("pending_intent") is None
    assert cleaned.get("biz_type_picker_sent") is None

    mid_qual = {"qual_state": "COLLECT_COUNTRY", "pending_intent": "order"}
    assert clear_stale_qualification_flags(mid_qual)["qual_state"] == "COLLECT_COUNTRY"


def test_mark_session_qualified_clears_mid_qual_flags():
    session = mark_session_qualified(
        {"qual_state": "COLLECT_BIZ_TYPE", "pending_intent": "pricing"}
    )
    assert session["lead_qualified"] is True
    assert session.get("qual_state") is None
    # pending_intent kept for same-turn handoff to pricing/order
    assert session.get("pending_intent") == "pricing"


def test_mark_session_disqualified_locks_and_clears_qual_ui():
    session = mark_session_disqualified(
        {"qual_state": "COLLECT_COUNTRY", "pending_intent": "pricing", "country_picker_sent": True},
        country="Iran",
    )
    assert session["disqualified"] is True
    assert session["lead_qualified"] is False
    assert session["country"] == "Iran"
    assert session["lifecycle_stage"] == "disqualified"
    assert session.get("qual_state") is None
    assert session.get("pending_intent") is None


def test_clear_stale_qualification_flags_for_disqualified():
    cleaned = clear_stale_qualification_flags(
        {
            "disqualified": True,
            "qual_state": "COLLECT_BIZ_TYPE",
            "country": "Iran",
        }
    )
    assert cleaned["disqualified"] is True
    assert cleaned.get("qual_state") is None


def test_lookup_lead_by_legacy_plus_phone(lead_db):
    lead_db.add(
        Lead(
            phone="+919876543210",
            company="Acme",
            country="Kenya",
            lead_score=55,
            lifecycle_stage="qualified",
        )
    )
    lead_db.commit()

    found = lookup_lead_by_phone(lead_db, "919876543210")
    assert found is not None
    assert found.company == "Acme"


def test_hydrate_session_from_lead_sets_qualified(lead_db):
    lead = Lead(
        phone="15550001111",
        company="MedEx",
        country="USA",
        business_type="distributor",
        buyer_type="distributor",
        lead_score=60,
        lead_category="WARM",
    )
    lead_db.add(lead)
    lead_db.commit()

    session = hydrate_session_from_lead(
        {"greeted": True, "qual_state": "COLLECT_COUNTRY"},
        lead,
    )
    assert session["lead_qualified"] is True
    assert session.get("qual_state") is None
    assert session["company"] == "MedEx"
    assert session["country"] == "USA"
    assert session["lead_score"] == 60
    assert session.get("hydrated_from_lead") is True


def test_hydrate_session_from_db_skips_when_already_qualified(lead_db):
    lead_db.add(Lead(phone="15550002222", company="X", country="UK", lead_score=50))
    lead_db.commit()

    session = hydrate_session_from_db(
        "15550002222",
        {
            "lead_qualified": True,
            "company": "Cached",
            "qual_state": "COLLECT_BIZ_TYPE",
        },
        lead_db,
    )
    assert session["company"] == "Cached"
    assert session.get("qual_state") is None


def test_hydrate_session_from_db_restores_expired_redis_session(lead_db):
    lead_db.add(
        Lead(
            phone=normalize_phone("+15550003333"),
            company="OldCo",
            country="India",
            business_type="pharmacy",
            lead_score=45,
        )
    )
    lead_db.commit()

    session = hydrate_session_from_db("15550003333", {"greeted": False}, lead_db)
    assert session["lead_qualified"] is True
    assert session["company"] == "OldCo"
    assert session.get("qual_state") is None


def test_hydrate_session_from_db_restores_disqualified_lead(lead_db):
    lead_db.add(
        Lead(
            phone="15550004444",
            country="Iran",
            lifecycle_stage="disqualified",
            manual_review_only=True,
            lead_score=0,
        )
    )
    lead_db.commit()

    session = hydrate_session_from_db("15550004444", {}, lead_db)
    assert session["disqualified"] is True
    assert session["lead_qualified"] is False
    assert session["country"] == "Iran"
    assert session["lifecycle_stage"] == "disqualified"
    assert session.get("qual_state") is None
