"""Dose-safe product/qty parsing and cart add pricing (production message styles)."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.order import (
    CART_MENU,
    COLLECT_QTY,
    COLLECT_SKU_CONFIRM,
    _extract_positive_int,
    run_order_agent,
)
from app.db.models import Base, Product
from app.messages.onboarding import (
    _parse_product_qty_segment,
    is_bare_order_qty,
    parse_order_line,
)


@pytest.mark.parametrize(
    "text, expect_name_contains, expect_qty",
    [
        ("Metformin 500mg - 100", "Metformin", 100),
        ("Metformin 500mg x 100", "Metformin", 100),
        ("Clenfit 40mg — 200", "Clenfit", 200),
        ("JGLUT 2000MG 30ML 350", "JGLUT", 350),
        ("Amoxicillin 500mg 200", "Amoxicillin", 200),
        ("Arkacan 100mcg", "Arkacan", None),
        ("Clenfit 40mg", "Clenfit", None),
        ("ARKACAN 100 MCG TAB 1X30", "ARKACAN", None),
        ("Arkacan 100", "Arkacan 100", None),  # ambiguous — do not assume qty
        ("Metformin 500mg", "Metformin", None),
    ],
)
def test_parse_order_line_dose_safe(text, expect_name_contains, expect_qty):
    name, qty = parse_order_line(text)
    assert expect_name_contains.lower() in name.lower()
    assert qty == expect_qty


@pytest.mark.parametrize(
    "text, expect",
    [
        ("100", 100),
        ("2,000", 2000),
        ("350 units", 350),
        ("100 strips", 100),
        ("Arkacan 100mcg", None),
        ("100mcg", None),
        ("40mg", None),
        ("", None),
    ],
)
def test_is_bare_order_qty(text, expect):
    assert is_bare_order_qty(text) == expect


@pytest.mark.parametrize(
    "text, expect",
    [
        ("100", 100),
        ("Metformin 500mg - 50", 50),
        ("Arkacan 100mcg", None),  # must NOT return 100 from dose
        ("Clenfit 40mg", None),
    ],
)
def test_extract_positive_int_never_steals_dose(text, expect):
    assert _extract_positive_int(text) == expect


def test_parse_product_qty_segment_matches_public_helper():
    assert _parse_product_qty_segment("Metformin 500mg - 100") == parse_order_line(
        "Metformin 500mg - 100"
    )


@pytest.fixture
def catalog_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    db.add_all(
        [
            Product(
                product_name="CLENFIT 40MG 1X10",
                salt_name="Clenbuterol",
                manufacturing_company="Test",
                expiry_date=date(2027, 1, 1),
                price_per_strip=0.64,
                is_restricted=False,
            ),
            Product(
                product_name="ARKACAN 100 MCG TAB 1X30",
                salt_name="Desmopressin",
                manufacturing_company="Test",
                expiry_date=date(2027, 1, 1),
                price_per_strip=1.25,
                is_restricted=False,
            ),
            Product(
                product_name="Metformin 500mg",
                salt_name="Metformin",
                manufacturing_company="Gamma Pharma",
                expiry_date=date(2027, 1, 1),
                price_per_strip=0.95,
                is_restricted=False,
            ),
        ]
    )
    db.commit()
    try:
        yield db
    finally:
        db.close()


@pytest.mark.asyncio
async def test_collect_qty_product_switch_does_not_assume_dose_as_qty(
    catalog_db, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    monkeypatch.setattr(
        "app.agents.order.send_interactive_buttons", AsyncMock(return_value=True)
    )

    session = {"phone": "+15550001111", "lead_qualified": True}
    _, session = await run_order_agent("CLENFIT 40MG 1X10", session, catalog_db)
    assert session["order_state"] == COLLECT_QTY

    reply, session = await run_order_agent("Arkacan 100mcg", session, catalog_db)
    assert session["order_state"] in {COLLECT_QTY, COLLECT_SKU_CONFIRM}
    assert session.get("order_cart") in (None, [],)
    # Must ask qty / confirm — not silently add Clenfit × 100
    assert "100" not in (session.get("order_cart") or [])
    cart = session.get("order_cart") or []
    assert cart == []
    assert "strip" in reply.lower() or "did you mean" in reply.lower() or "number only" in reply.lower()


@pytest.mark.asyncio
async def test_collect_qty_bare_number_adds_with_db_price(catalog_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    monkeypatch.setattr(
        "app.agents.order.send_interactive_buttons", AsyncMock(return_value=True)
    )

    session = {"phone": "+15550001112", "lead_qualified": True}
    _, session = await run_order_agent("CLENFIT 40MG 1X10", session, catalog_db)
    reply, session = await run_order_agent("100", session, catalog_db)

    assert session["order_state"] == CART_MENU
    line = session["order_cart"][0]
    assert line["quantity"] == 100 or line["qty"] == 100
    assert float(line["unit_price"]) == pytest.approx(0.64)
    assert "$64.00" in reply or "64.00" in reply


@pytest.mark.asyncio
async def test_yes_confirm_without_pending_qty_asks_quantity(catalog_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    monkeypatch.setattr(
        "app.agents.order.send_interactive_buttons", AsyncMock(return_value=True)
    )

    session = {"phone": "+15550001113", "lead_qualified": True}
    # Fuzzy / token path for short query
    reply, session = await run_order_agent("arkacan", session, catalog_db)
    if session.get("order_state") == COLLECT_SKU_CONFIRM:
        reply, session = await run_order_agent("yes", session, catalog_db)
        assert session["order_state"] == COLLECT_QTY
        assert "strip" in reply.lower() or "number only" in reply.lower()
        assert not session.get("order_cart")
    else:
        # Direct match still must ask qty when no explicit qty provided
        assert session["order_state"] == COLLECT_QTY
        assert "strip" in reply.lower() or "number only" in reply.lower()


@pytest.mark.asyncio
async def test_yes_confirm_with_explicit_qty_uses_db_price_not_zero(
    catalog_db, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    monkeypatch.setattr(
        "app.agents.order.send_interactive_buttons", AsyncMock(return_value=True)
    )

    session = {
        "phone": "+15550001114",
        "lead_qualified": True,
        "order_state": COLLECT_SKU_CONFIRM,
        "order_pending_sku": "PROD-0002",
        "order_pending_product_name": "ARKACAN 100 MCG TAB 1X30",
        "order_pending_qty": 120,
    }
    # Ensure sku id matches seeded row 2
    ark = (
        catalog_db.query(Product)
        .filter(Product.product_name == "ARKACAN 100 MCG TAB 1X30")
        .one()
    )
    session["order_pending_sku"] = f"PROD-{ark.id:04d}"

    reply, session = await run_order_agent("yes", session, catalog_db)
    assert session["order_state"] == CART_MENU
    line = session["order_cart"][0]
    assert line["qty"] == 120 or line["quantity"] == 120
    assert float(line["unit_price"]) == pytest.approx(1.25)
    assert "$0.00" not in reply
    assert "150.00" in reply or "$150" in reply
