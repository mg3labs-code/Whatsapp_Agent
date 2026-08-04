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
    _extract_order_status_ref,
    is_order_account_message,
    is_order_tracking_message,
    run_order_agent,
)
from app.agents.qualification import run_qualification_agent
from app.db.models import Base, Lead, Order, Product, ShippingRate
from app.messages.onboarding import (
    checkout_prompt,
    is_qty_instruction_echo,
    looks_like_bulk_order,
    parse_bulk_order_lines,
    parse_checkout_oneline,
    product_qty_prompt,
    resolve_country_button,
)


def _seed_kenya_shipping(db):
    """Wide weight brackets so checkout can reach CONFIRM_ORDER with a full total."""
    for shipping_type, rate in (("EMS", 45.0), ("LP", 28.0)):
        db.add(
            ShippingRate(
                country_name="KENYA",
                shipping_type=shipping_type,
                weight_from_g=0,
                weight_to_g=100_000,
                rate_usd=rate,
            )
        )
    db.commit()


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
    _seed_kenya_shipping(db)
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


def test_product_qty_prompt_asks_for_strips_number_only():
    prompt = product_qty_prompt("LASIX TAB 15S")
    assert "LASIX TAB 15S" in prompt
    assert "350" in prompt
    assert "strip" in prompt.lower()
    assert "full line" not in prompt.lower()
    assert "or send" not in prompt.lower()
    assert "number only" in prompt.lower()


def test_is_qty_instruction_echo():
    assert is_qty_instruction_echo("Full line") is True
    assert is_qty_instruction_echo("full line") is True
    assert is_qty_instruction_echo("quantity only") is True
    assert is_qty_instruction_echo("350") is False
    assert is_qty_instruction_echo("LASIX TAB 15S - 350") is False


@pytest.mark.asyncio
async def test_collect_qty_full_line_echo_stays_on_same_product(order_db, monkeypatch):
    """Screenshot bug: buyer types 'Full line' — must not jump to another SKU."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    order_db.add(
        Product(
            product_name="LASIX TAB 15S",
            salt_name="Furosemide",
            manufacturing_company="Sanofi",
            expiry_date=date(2027, 1, 1),
            price_per_strip=0.50,
            is_restricted=False,
        )
    )
    order_db.add(
        Product(
            product_name="CABERLEE 0.25MG TAB 1X4",
            salt_name="Cabergoline",
            manufacturing_company="Other",
            expiry_date=date(2027, 1, 1),
            price_per_strip=1.20,
            is_restricted=False,
        )
    )
    order_db.commit()

    session = {
        "phone": "+919876543299",
        "lead_qualified": True,
        "order_state": COLLECT_QTY,
        "order_product_name": "LASIX TAB 15S",
        "order_sku": "PROD-1",
    }
    reply, session = await run_order_agent("Full line", session, order_db)
    assert session["order_state"] == COLLECT_QTY
    assert session["order_product_name"] == "LASIX TAB 15S"
    assert "LASIX" in reply
    assert "CABERLEE" not in reply.upper()
    assert "350" in reply


def test_parse_trailing_qty():
    from app.messages.onboarding import _parse_product_qty_segment

    name, qty = _parse_product_qty_segment("JGLUT 2000MG 30ML 350")
    assert qty == 350
    assert "JGLUT" in name


def test_parse_checkout_oneline():
    parsed = parse_checkout_oneline(
        "Jane Doe, Sydney",
        "Australia",
        whatsapp_phone="+61412345678",
    )
    assert parsed["contact"] == "Jane Doe (+61412345678)"
    assert parsed["city"] == "Sydney"
    assert parsed["country"] == "Australia"
    # Legacy trailing phone is ignored; WhatsApp number wins.
    legacy = parse_checkout_oneline(
        "Jane Doe, Sydney, +61999999999",
        "Australia",
        whatsapp_phone="+61412345678",
    )
    assert legacy["contact"] == "Jane Doe (+61412345678)"


def test_checkout_prompt_uses_known_country():
    prompt = checkout_prompt("Australia")
    assert "Australia" in prompt
    assert "Name, City" in prompt
    assert "WhatsApp" in prompt
    assert "Name, City, Phone" not in prompt


def test_extract_order_status_ref_bare_date():
    assert _extract_order_status_ref("Order status of 20260608-2650") == "ORD-20260608-2650"


def test_is_order_tracking_message_typo_stus():
    assert is_order_tracking_message("20260608-2650 order stus") is True


def test_is_order_account_message_pending_payments():
    assert is_order_account_message("Total pending payments") is True


@pytest.mark.asyncio
async def test_order_status_does_not_start_cart(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    session = {"phone": "+919876543210", "lead_qualified": True}

    reply, session = await run_order_agent("order_status", session, order_db)
    assert "order" in reply.lower() or "couldn't find" in reply.lower()
    assert session.get("order_state") is None


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
        with patch("app.agents.qualification._send_business_type_picker", new=AsyncMock()):
            reply, session, _ = await run_qualification_agent("country_au", session, order_db)
    assert session["country"] == "Australia"
    assert session["qual_state"] == "COLLECT_BIZ_TYPE"
    assert session.get("biz_type_picker_sent") is True
    assert "business type" in reply.lower()


@pytest.mark.asyncio
async def test_qualification_freezes_first_allowed_country_on_list_retap(order_db):
    """Old WhatsApp country list re-tap must not overwrite the first allowed country."""
    session = {
        "phone": "+919876543210",
        "qual_state": "COLLECT_COUNTRY",
        "country_picker_sent": True,
    }
    with patch("app.agents.qualification.send_country_picker", new=AsyncMock(side_effect=lambda _p, s: s)):
        with patch("app.agents.qualification._send_business_type_picker", new=AsyncMock()):
            _, session, _ = await run_qualification_agent("country_uk", session, order_db)
            assert session["country"] == "United Kingdom"
            assert session["qual_state"] == "COLLECT_BIZ_TYPE"

            reply, session, intent = await run_qualification_agent(
                "country_us", session, order_db
            )

    assert intent == "continue_qual"
    assert session["country"] == "United Kingdom"
    assert session["qual_state"] == "COLLECT_BIZ_TYPE"
    assert "already set" in reply.lower()
    assert "united kingdom" in reply.lower()


@pytest.mark.asyncio
async def test_qualification_excluded_country_hard_locks_phone(order_db):
    """Restricted country permanently locks the phone; later UK tap cannot continue."""
    session = {
        "phone": "+919876543299",
        "qual_state": "COLLECT_COUNTRY",
        "country_picker_sent": True,
        "awaiting_custom_country": True,
    }
    with patch("app.agents.qualification.send_country_picker", new=AsyncMock(side_effect=lambda _p, s: s)):
        with patch(
            "app.agents.qualification.send_escalation_alert",
            new=AsyncMock(),
        ):
            reply, session, intent = await run_qualification_agent(
                "Iran", session, order_db
            )

    assert intent == "escalate"
    assert session.get("disqualified") is True
    assert session.get("lead_qualified") is False
    assert session["country"] == "Iran"
    assert session.get("qual_state") is None
    assert reply == ""
    queued = (session.get("escalation_buyer_reply") or "").lower()
    assert "unable" in queued or "compliance" in queued
    assert session.get("escalation_reason") == "excluded_country"

    leads = order_db.query(Lead).filter(Lead.phone == "919876543299").all()
    assert len(leads) == 1
    assert leads[0].lifecycle_stage == "disqualified"
    assert leads[0].country == "Iran"

    # Re-tap of an allowed country on the old list must stay locked.
    reply2, session2, intent2 = await run_qualification_agent(
        "country_uk", session, order_db
    )
    assert intent2 == "escalate"
    assert session2.get("disqualified") is True
    assert session2.get("country") == "Iran"
    assert "united kingdom" not in (session2.get("country") or "").lower()


@pytest.mark.asyncio
async def test_order_checkout_reuses_qualification_country(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    session = {"phone": "+919876543210", "country": "Kenya"}

    await run_order_agent("order", session, order_db)
    _, session = await run_order_agent("Metformin 500mg - 100", session, order_db)
    assert session["order_state"] == CART_MENU

    _, session = await run_order_agent("checkout", session, order_db)
    assert session["order_state"] == COLLECT_CHECKOUT
    assert session["order_country"] == "Kenya"
    assert "Kenya" in checkout_prompt("Kenya")

    _, session = await run_order_agent("Jane Doe, Nairobi", session, order_db)
    assert session["order_state"] in {CONFIRM_ORDER, "SHIPPING_CHOICE"}
    if session["order_state"] == "SHIPPING_CHOICE":
        _, session = await run_order_agent("express", session, order_db)
    assert session["order_state"] == CONFIRM_ORDER
    assert session["order_city"] == "Nairobi"
    assert session["order_contact"] == "Jane Doe (+919876543210)"


@pytest.mark.asyncio
async def test_pending_shipping_quote_no_wire_details(order_db, monkeypatch):
    """No DB rates → desk alert + wait; never send wire for product-only total."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    monkeypatch.setenv("STATIC_WIRE_ACCOUNT_NAME", "New Life Medicare Exports")
    monkeypatch.setenv("STATIC_WIRE_ACCOUNT_NUMBER", "123456789012")
    monkeypatch.setenv("STATIC_WIRE_BANK_NAME", "Example Bank Ltd")
    monkeypatch.setenv("STATIC_WIRE_BRANCH", "Mumbai Export Branch")
    monkeypatch.setenv("STATIC_WIRE_SWIFT_CODE", "EXAMPLGB")

    alerts: list[tuple] = []

    async def fake_quote_alert(session, *, order_ref=""):
        alerts.append((session.get("phone"), order_ref))
        return True

    monkeypatch.setattr(
        "app.agents.order.send_shipping_quote_alert",
        fake_quote_alert,
    )
    monkeypatch.setattr(
        "app.agents.order.send_message",
        AsyncMock(return_value=True),
    )

    session = {"phone": "+919876543299", "country": "Atlantis"}
    await run_order_agent("order", session, order_db)
    _, session = await run_order_agent("Metformin 500mg - 50", session, order_db)
    _, session = await run_order_agent("checkout", session, order_db)
    reply, session = await run_order_agent("Sam Buyer, Nowhere", session, order_db)

    assert "full amount" in reply.lower() or "export desk" in reply.lower()
    assert "wire transfer details have been sent" not in reply.lower()
    assert session.get("order_state") is None
    assert session.get("last_order_awaiting_shipping_quote") is True
    assert session.get("payment_method_chosen") is None
    assert alerts, "export desk should be alerted"

    rows = order_db.query(Order).all()
    assert rows
    assert rows[0].payment_status == "pending_shipping_quote"
    assert rows[0].shipping_type == "PENDING_QUOTE"


@pytest.mark.asyncio
async def test_order_restart_from_collect_qty(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    session = {
        "phone": "+919876543212",
        "lead_qualified": True,
        "order_state": COLLECT_QTY,
        "order_product_name": "JGLUT 2000MG 30ML",
        "order_sku": "PROD-1",
    }

    reply, session = await run_order_agent("I need new order", session, order_db)
    assert session.get("order_state") == "COLLECT_SKU"
    assert "strips" in reply.lower()
    assert "product name - strips" in reply.lower() or "send each product" in reply.lower()
    assert session.get("order_product_name") is None
    assert session.get("show_main_menu_after_reply") is not True


@pytest.mark.asyncio
async def test_order_cancel_from_collect_qty(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    session = {
        "phone": "+919876543212",
        "lead_qualified": True,
        "order_state": COLLECT_QTY,
        "order_product_name": "JGLUT 2000MG 30ML",
        "order_sku": "PROD-1",
        "faq_miss_count": 1,
        "clarification_count": 1,
    }

    reply, session = await run_order_agent("cancel", session, order_db)
    assert session.get("order_state") is None
    assert session.get("order_product_name") is None
    assert session.get("show_main_menu_after_reply") is True
    assert "faq_miss_count" not in session
    assert "clarification_count" not in session
    assert "order cancelled" in reply.lower()


@pytest.mark.asyncio
async def test_order_confirm_button_id(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    session = {
        "phone": "+919876543213",
        "lead_qualified": True,
        "country": "Kenya",
        "order_state": CONFIRM_ORDER,
        "order_cart": [
            {
                "sku": "PROD-1",
                "product_name": "Metformin 500mg",
                "quantity": 100,
                "unit_price": 0.95,
            }
        ],
        "order_country": "Kenya",
        "order_city": "Nairobi",
        "order_contact": "Jane Doe",
    }

    with patch("app.agents.order.send_interactive_buttons", new=AsyncMock()):
        with patch(
            "app.agents.order._commit_order",
            new=AsyncMock(return_value=("Order placed", session)),
        ):
            reply, _ = await run_order_agent("confirm", session, order_db)
    assert "placed" in reply.lower()


@pytest.mark.asyncio
async def test_my_orders_button_shows_grouped_history(order_db, monkeypatch):
    from datetime import datetime

    from app.db.models import Order

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    phone = "+919876543210"
    order_db.add(
        Order(
            phone=phone,
            sku="PROD-1",
            product_name="Metformin 500mg",
            quantity=100,
            country="Kenya",
            city="Nairobi",
            contact_name="Jane",
            order_ref="ORD-20260614-1001-L01",
            status="pending",
            payment_status="awaiting_payment",
            created_at=datetime(2026, 6, 14, 10, 0, 0),
        )
    )
    order_db.add(
        Order(
            phone=phone,
            sku="PROD-1",
            product_name="Metformin 500mg",
            quantity=50,
            country="Kenya",
            city="Nairobi",
            contact_name="Jane",
            order_ref="ORD-20260613-2002-L01",
            status="processing",
            payment_status="payment_received",
            created_at=datetime(2026, 6, 13, 10, 0, 0),
        )
    )
    order_db.commit()

    session = {"phone": phone, "lead_qualified": True}
    reply, session = await run_order_agent("my_orders", session, order_db)
    assert session.get("order_state") is None
    assert "My orders" in reply
    assert "Awaiting payment" in reply
    assert "Paid / processing" in reply
    assert "ORD-20260614-1001" in reply
    assert "ORD-20260613-2002" in reply


@pytest.mark.asyncio
async def test_order_bulk_list_adds_multiple_products(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    session = {"phone": "+919876543211", "country": "Kenya", "lead_qualified": True}

    await run_order_agent("order", session, order_db)
    reply, session = await run_order_agent(
        "Metformin 500mg - 100\nAmoxicillin 500mg - 200",
        session,
        order_db,
    )
    assert session["order_state"] == CART_MENU
    assert len(session["order_cart"]) == 2
    assert "Added to cart" in reply
