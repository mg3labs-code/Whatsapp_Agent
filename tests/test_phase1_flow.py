"""Phase 1 UX — country once, bulk list, quantity buttons, single checkout."""

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.order import (
    CART_MENU,
    COLLECT_CHECKOUT,
    COLLECT_QTY,
    CONFIRM_ORDER,
    run_order_agent,
)
from app.agents.qualification import run_qualification_agent
from app.db.models import Base, Product
from app.messages.onboarding import (
    checkout_prompt,
    looks_like_bulk_order,
    parse_bulk_order_lines,
    parse_checkout_oneline,
    resolve_country_button,
)


@pytest.fixture
def order_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    db.add(
        Product(
            product_name="Metformin 500mg",
            salt_name="Metformin",
            manufacturing_company="Gamma Pharma",
            expiry_date=date(2027, 1, 1),
            price_per_strip=0.95,
            is_restricted=False,
        )
    )
    db.add(
        Product(
            product_name="Amoxicillin 500mg",
            salt_name="Amoxicillin",
            manufacturing_company="Beta Pharma",
            expiry_date=date(2027, 6, 1),
            price_per_strip=1.10,
            is_restricted=False,
        )
    )
    db.commit()
    try:
        yield db
    finally:
        db.close()


def test_resolve_country_button_usa():
    country, follow_up = resolve_country_button("country_us")
    assert country == "United States"
    assert follow_up is None


def test_parse_bulk_order_lines():
    text = "Metformin 500mg - 100\nAmoxicillin 500mg - 200"
    lines = parse_bulk_order_lines(text)
    assert len(lines) == 2
    assert lines[0] == ("Metformin 500mg", 100)
    assert lines[1][1] == 200


def test_looks_like_bulk_order_comma_list():
    assert looks_like_bulk_order("Bacloheal, cenforce, artvigil") is True


def test_parse_checkout_oneline():
    parsed = parse_checkout_oneline("Jane Doe, Sydney, +61412345678", "Australia")
    assert parsed["contact"] == "Jane Doe (+61412345678)"
    assert parsed["city"] == "Sydney"
    assert parsed["country"] == "Australia"


def test_checkout_prompt_uses_known_country():
    prompt = checkout_prompt("Australia")
    assert "Australia" in prompt
    assert "Name, City, Phone" in prompt


@pytest.mark.asyncio
async def test_qualification_country_picker_once(order_db):
    session = {"phone": "+919876543210", "qual_state": "COLLECT_COUNTRY"}
    with patch(
        "app.agents.qualification.send_country_picker",
        new=AsyncMock(side_effect=lambda _p, s: {**s, "country_picker_sent": True}),
    ):
        reply, session, intent = await run_qualification_agent("hi", session, order_db)
    assert "select your country" in reply.lower()
    assert session.get("country_picker_sent") is True

    with patch(
        "app.agents.qualification.send_country_picker",
        new=AsyncMock(side_effect=lambda _p, s: s),
    ):
        reply2, _, _ = await run_qualification_agent("hi", session, order_db)
    assert "list above" in reply2.lower()


@pytest.mark.asyncio
async def test_qualification_country_button_sets_country(order_db):
    session = {"phone": "+919876543210", "qual_state": "COLLECT_COUNTRY", "country_picker_sent": True}
    with patch("app.agents.qualification.send_country_picker", new=AsyncMock(side_effect=lambda _p, s: s)):
        reply, session, _ = await run_qualification_agent("country_au", session, order_db)
    assert session["country"] == "Australia"
    assert session["qual_state"] == "COLLECT_BIZ_TYPE"
    assert "business" in reply.lower()


@pytest.mark.asyncio
async def test_order_checkout_reuses_qualification_country(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    session = {"phone": "+919876543210", "country": "Kenya"}

    await run_order_agent("order", session, order_db)
    _, session = await run_order_agent("Metformin 500mg", session, order_db)
    _, session = await run_order_agent("qty_100", session, order_db)
    assert session["order_state"] == CART_MENU

    _, session = await run_order_agent("checkout", session, order_db)
    assert session["order_state"] == COLLECT_CHECKOUT
    assert session["order_country"] == "Kenya"
    assert "Kenya" in checkout_prompt("Kenya")

    _, session = await run_order_agent("Jane Doe, Nairobi, +254700000000", session, order_db)
    assert session["order_state"] == CONFIRM_ORDER
    assert session["order_city"] == "Nairobi"
    assert session["order_contact"] == "Jane Doe (+254700000000)"


@pytest.mark.asyncio
async def test_order_bulk_list_adds_multiple_products(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    session = {"phone": "+919876543211", "country": "Kenya", "lead_qualified": True}

    await run_order_agent("order", session, order_db)
    with patch("app.agents.order.send_quantity_picker", new=AsyncMock(return_value=True)):
        reply, session = await run_order_agent(
            "Metformin 500mg - 100\nAmoxicillin 500mg - 200",
            session,
            order_db,
        )
    assert session["order_state"] == CART_MENU
    assert len(session["order_cart"]) == 2
    assert "Added to cart" in reply
