"""Tests for Postgres lead hydration into Redis session."""

import pytest

from app.db.models import Base, Lead
from app.session.lead_hydration import (
    hydrate_session_from_db,
    hydrate_session_from_lead,
    lookup_lead_by_phone,
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

    session = hydrate_session_from_lead({"greeted": True}, lead)
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
        {"lead_qualified": True, "company": "Cached"},
        lead_db,
    )
    assert session["company"] == "Cached"


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
