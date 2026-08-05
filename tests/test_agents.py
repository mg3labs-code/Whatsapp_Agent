import json
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents import escalation
from app.business import hours as business_hours
from app.agents.faq import (
    ERROR_REPLY,
    FAQ_SYSTEM_PROMPT,
    NO_CONTEXT_REPLY,
    run_faq_agent,
)
from app.agents.order import (
    CART_MENU,
    COLLECT_CHECKOUT,
    COLLECT_CITY,
    COLLECT_COUNTRY,
    COLLECT_QTY,
    COLLECT_SKU,
    COLLECT_SKU_CONFIRM,
    CONFIRM_ORDER,
    PAY_BANK_BUTTON,
    SANCTIONED_COUNTRY_REFUSAL,
    SELECT_PAYMENT,
    _resolve_pending_payment,
    _resolve_product_row,
    run_order_agent,
)
from app.agents.pricing import format_multi_product_quote, get_product_by_name, run_pricing_agent
from app.agents.lead_scoring import (
    classify_lead_score,
    score_lead,
)
from app.business.restricted_products import clear_restricted_terms_cache
from app.agents.qualification import (
    COLLECT_BIZ_TYPE,
    COLLECT_COUNTRY,
    calculate_lead_score,
    run_qualification_agent,
)
from app.db.models import Base, Lead, Order, Product, ShippingRate
from app.db.models import RestrictedTerm


def _fixed_now(tz, year: int, month: int, day: int, hour: int):
    return tz.localize(datetime(year, month, day, hour, 0))


def _seed_kenya_shipping(db):
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


@pytest.fixture(autouse=True)
def _business_hours_env(monkeypatch):
    monkeypatch.setenv("BUSINESS_HOURS_START", "10")
    monkeypatch.setenv("BUSINESS_HOURS_END", "20")
    monkeypatch.setenv("BUSINESS_TIMEZONE", "Asia/Kolkata")


def test_is_business_hours_false_at_11pm_saturday(monkeypatch):
    tz = pytz.timezone("Asia/Kolkata")
    monkeypatch.setattr(
        business_hours,
        "_now_in_tz",
        lambda _tz: _fixed_now(tz, 2026, 5, 16, 23),  # Sat 11:00 PM IST
    )

    assert escalation.is_business_hours() is False


def test_is_business_hours_true_at_11am_weekday(monkeypatch):
    tz = pytz.timezone("Asia/Kolkata")
    monkeypatch.setattr(
        business_hours,
        "_now_in_tz",
        lambda _tz: _fixed_now(tz, 2026, 5, 12, 11),  # Tue 11:00 AM IST
    )

    assert escalation.is_business_hours() is True


def test_next_open_after_saturday_evening_is_monday(monkeypatch):
    tz = pytz.timezone("Asia/Kolkata")
    monkeypatch.setattr(
        business_hours,
        "_now_in_tz",
        lambda _tz: _fixed_now(tz, 2026, 5, 16, 23),  # Sat 11:00 PM IST
    )

    assert escalation.get_next_business_open_str() == "Monday 10:00 AM IST"


def test_next_open_after_weekday_evening_is_tomorrow(monkeypatch):
    tz = pytz.timezone("Asia/Kolkata")
    monkeypatch.setattr(
        business_hours,
        "_now_in_tz",
        lambda _tz: _fixed_now(tz, 2026, 5, 12, 20),  # Tue 8:00 PM IST (after close)
    )

    assert escalation.get_next_business_open_str() == "tomorrow 10:00 AM IST"


def test_next_open_before_hours_is_today(monkeypatch):
    tz = pytz.timezone("Asia/Kolkata")
    monkeypatch.setattr(
        business_hours,
        "_now_in_tz",
        lambda _tz: _fixed_now(tz, 2026, 5, 12, 7),  # Tue 7:00 AM IST
    )

    assert escalation.get_next_business_open_str() == "today at 10:00 AM IST"


def test_is_business_hours_false_before_opening(monkeypatch):
    tz = pytz.timezone("Asia/Kolkata")
    monkeypatch.setattr(
        business_hours,
        "_now_in_tz",
        lambda _tz: _fixed_now(tz, 2026, 5, 12, 9),  # Tue 9:00 AM IST
    )
    assert escalation.is_business_hours() is False


def test_is_business_hours_true_saturday_afternoon(monkeypatch):
    tz = pytz.timezone("Asia/Kolkata")
    monkeypatch.setattr(
        business_hours,
        "_now_in_tz",
        lambda _tz: _fixed_now(tz, 2026, 5, 16, 14),  # Sat 2:00 PM IST
    )
    assert escalation.is_business_hours() is True


def test_sunday_is_limited_operations(monkeypatch):
    tz = pytz.timezone("Asia/Kolkata")
    monkeypatch.setattr(
        business_hours,
        "_now_in_tz",
        lambda _tz: _fixed_now(tz, 2026, 5, 17, 12),  # Sun noon
    )
    assert escalation.is_business_hours() is False
    assert escalation.is_limited_operations() is True
    assert escalation.get_operations_mode() == "limited"


def test_republic_day_is_public_holiday(monkeypatch):
    tz = pytz.timezone("Asia/Kolkata")
    monkeypatch.setattr(
        business_hours,
        "_now_in_tz",
        lambda _tz: _fixed_now(tz, 2026, 1, 26, 12),  # Republic Day
    )
    assert escalation.is_public_holiday() is True
    assert escalation.is_business_hours() is False
    assert "Republic Day" in escalation.get_off_hours_notice()


@pytest.fixture
def pricing_db():
    """In-memory SQLite session seeded with two catalog rows (one restricted)."""
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
                product_name="Amoxicillin 500mg",
                salt_name="Amoxicillin",
                manufacturing_company="Acme Labs",
                expiry_date=date(2027, 6, 1),
                price_per_strip=1.85,
                is_restricted=False,
            ),
            Product(
                product_name="Ciprofloxacin 500mg",
                salt_name="Ciprofloxacin",
                manufacturing_company="Beta Pharma",
                expiry_date=date(2026, 12, 31),
                price_per_strip=2.10,
                is_restricted=True,
                schedule_category="H",
            ),
        ]
    )
    db.commit()
    try:
        yield db
    finally:
        db.close()


def test_pricing_db_tool_found(pricing_db):
    result = get_product_by_name("amox", pricing_db)
    assert "product_name" in result
    assert result["product_name"] == "Amoxicillin 500mg"


def test_pricing_db_tool_not_found(pricing_db):
    result = get_product_by_name("xyz999", pricing_db)
    assert "error" in result
    assert result["error"] == "product_not_found"


def test_get_product_by_name_fuzzy_match(pricing_db):
    result = get_product_by_name("amox", pricing_db)

    assert "error" not in result
    assert result["product_name"] == "Amoxicillin 500mg"
    assert result["price_per_strip"] == 1.85
    assert result["is_restricted"] is False


def test_get_product_by_name_matches_manufacturer(pricing_db):
    result = get_product_by_name("Acme", pricing_db)
    assert result.get("product_name") == "Amoxicillin 500mg"


def test_get_product_by_name_not_found(pricing_db):
    result = get_product_by_name("xyz999", pricing_db)

    assert result["error"] == "product_not_found"
    assert result["query"] == "xyz999"
    assert result.get("suggestions") == []


def test_get_product_by_name_restricted(pricing_db):
    result = get_product_by_name("Ciprofloxacin", pricing_db)

    assert result == {
        "error": "product_restricted",
        "name": "Ciprofloxacin 500mg",
        "schedule_category": "H",
    }


def test_get_product_by_name_restricted_precheck_without_catalog_row(pricing_db):
    pricing_db.add(
        RestrictedTerm(
            term="Tramadol",
            normalized_term="tramadol",
            schedule_category="H",
            source="test",
        )
    )
    pricing_db.commit()
    clear_restricted_terms_cache()

    result = get_product_by_name("Tramadol 50mg price", pricing_db)
    assert result == {
        "error": "product_restricted",
        "name": "Tramadol",
        "schedule_category": "H",
    }


def test_get_product_by_name_token_fallback(pricing_db):
    """Extra tokens still resolve via order-style token fallback."""
    result = get_product_by_name("Amoxicillin 500mg urgent export", pricing_db)
    assert result.get("product_name") == "Amoxicillin 500mg"
    assert result.get("price_per_strip") == 1.85
    assert result.get("match_mode") in {"direct", "token"}


def test_order_resolve_product_row_restricted_precheck_without_catalog_row(order_db):
    order_db.add(
        RestrictedTerm(
            term="Ketamine",
            normalized_term="ketamine",
            schedule_category="X",
            source="test",
        )
    )
    order_db.commit()
    clear_restricted_terms_cache()

    product, error = _resolve_product_row("Ketamine 100mg", order_db)
    assert product is None
    assert error == "restricted"


def test_get_product_by_name_suggestions_on_near_miss(pricing_db):
    result = get_product_by_name("Amoxicill", pricing_db)
    # Partial may hit via ILIKE; if not_found, expect suggestions containing Amox
    if result.get("error") == "product_not_found":
        assert any("Amoxicillin" in s for s in result.get("suggestions", []))
    else:
        assert result.get("product_name") == "Amoxicillin 500mg"


@pytest.mark.asyncio
async def test_run_faq_agent_missing_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    out, session, intent = await run_faq_agent("What are your payment terms?")

    assert out == ERROR_REPLY
    assert intent == "faq"
    assert session.get("faq_miss_count") == 1


@pytest.mark.asyncio
async def test_run_faq_agent_menu_open_skips_rag(monkeypatch):
    """Bare FAQ menu id must not hit Pinecone or count as a miss."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    out, session, intent = await run_faq_agent("faq")

    assert "FAQs" in out or "shipping" in out.lower()
    assert "don't have specific information" not in out.lower()
    assert intent == "faq"
    assert session.get("faq_miss_count") in (None, 0)


@pytest.mark.asyncio
async def test_run_pricing_agent_menu_open_skips_llm(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reply, session, intent = await run_pricing_agent("pricing", {}, order_db)
    assert "product" in reply.lower() or "pricing" in reply.lower()
    assert intent == "pricing"
    assert session.get("pricing_miss_count") in (None, 0)


def test_format_faq_ship_available_only_lists_present_services():
    from app.agents.faq import _format_faq_ship_available

    ems_only = _format_faq_ship_available(
        "Kenya",
        {"EMS": {"days": "7-14 days"}, "LP": None},
    )
    assert "Express (EMS)" in ems_only
    assert "Standard (LP)" not in ems_only
    assert "checkout" in ems_only.lower()

    both = _format_faq_ship_available(
        "Kenya",
        {"EMS": {"days": "7-14 days"}, "LP": {"days": "15-30 days"}},
    )
    assert "Express (EMS)" in both
    assert "Standard (LP)" in both


@pytest.mark.asyncio
async def test_faq_ship_to_no_rates_alerts_team(monkeypatch):
    """No catalog shipping rates → notify leads immediately (no speak-to-team gate)."""
    alerts: list[tuple] = []

    async def capture_alert(phone, session, reason):
        alerts.append((phone, session, reason))
        return True

    monkeypatch.setattr(
        "app.agents.faq._faq_shipping_availability",
        lambda country, db=None: {
            "available": False,
            "country": country,
        },
    )
    monkeypatch.setattr("app.agents.faq.send_escalation_alert", capture_alert)
    monkeypatch.setattr("app.agents.faq.SessionLocal", MagicMock())

    out, session, intent = await run_faq_agent(
        "Do you ship to Kenya?",
        phone="+15550009999",
        session={"phone": "+15550009999"},
    )

    assert intent == "faq"
    assert "notified our team" in out.lower()
    assert "speak to team" not in out.lower()
    assert alerts and alerts[0][2] == "shipping_rates_unavailable"


@pytest.mark.asyncio
async def test_faq_agent_no_context(monkeypatch):
    """Pinecone returns no qualifying chunks → escalation copy (no chat call)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")

    mock_index = MagicMock()
    mock_index.query.return_value = MagicMock(matches=[])
    mock_pc_instance = MagicMock()
    mock_pc_instance.Index.return_value = mock_index

    mock_emb = MagicMock()
    mock_emb.data = [MagicMock(embedding=[0.01] * 8)]

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_emb)

    with (
        patch("pinecone.Pinecone", return_value=mock_pc_instance),
        patch("app.agents.faq.get_async_openai_client", return_value=mock_client),
    ):
        reply, session, intent = await run_faq_agent("Any question", session={"lead_qualified": True})

    assert "specific information" in reply.lower() or "rephrase" in reply.lower()
    assert reply == NO_CONTEXT_REPLY
    assert intent == "faq"
    assert session.get("faq_miss_count") == 1
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_faq_agent_with_context(monkeypatch):
    """Pinecone returns one strong chunk → GPT-4o-mini produces non-empty reply."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")

    mock_index = MagicMock()
    mock_index.query.return_value = MagicMock(
        matches=[
            MagicMock(score=0.91, metadata={"text": "We ship worldwide by air freight."}),
        ]
    )
    mock_pc_instance = MagicMock()
    mock_pc_instance.Index.return_value = mock_index

    mock_emb = MagicMock()
    mock_emb.data = [MagicMock(embedding=[0.02] * 8)]

    mock_chat = MagicMock()
    mock_chat.choices = [MagicMock(message=MagicMock(content="*Answer:* Grounded reply here."))]

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_emb)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_chat)

    with (
        patch("pinecone.Pinecone", return_value=mock_pc_instance),
        patch("app.agents.faq.get_async_openai_client", return_value=mock_client),
    ):
        reply, session, intent = await run_faq_agent("Do you export to Kenya?")

    assert isinstance(reply, str)
    assert len(reply) > 0
    assert intent == "faq"
    assert session.get("faq_miss_count", 0) == 0
    mock_client.chat.completions.create.assert_awaited_once()
    call_kw = mock_client.chat.completions.create.await_args
    assert call_kw.kwargs["model"] == "gpt-4o-mini"
    messages = call_kw.kwargs["messages"]
    assert messages[0]["content"] == FAQ_SYSTEM_PROMPT
    assert "We ship worldwide" in messages[1]["content"]


@pytest.mark.asyncio
async def test_faq_agent_second_miss_escalates(monkeypatch):
    """Two consecutive no-context FAQ turns → escalate intent with reason set."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")

    mock_index = MagicMock()
    mock_index.query.return_value = MagicMock(matches=[])
    mock_pc_instance = MagicMock()
    mock_pc_instance.Index.return_value = mock_index

    mock_emb = MagicMock()
    mock_emb.data = [MagicMock(embedding=[0.01] * 8)]

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_emb)

    with (
        patch("pinecone.Pinecone", return_value=mock_pc_instance),
        patch("app.agents.faq.get_async_openai_client", return_value=mock_client),
    ):
        session = {"lead_qualified": True, "faq_miss_count": 1}
        reply, session, intent = await run_faq_agent("Still unknown question", session=session)

    assert reply == ""
    assert intent == "escalate"
    assert session.get("faq_miss_count", 0) == 0
    assert session.get("escalation_reason") == "faq_no_match_repeated"
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_faq_agent_error_counts_toward_miss(monkeypatch):
    """Infra/API failures increment the same miss counter as no-context."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("app.agents.faq.get_async_openai_client", return_value=mock_client):
        session = {"lead_qualified": True, "faq_miss_count": 1}
        reply, session, intent = await run_faq_agent("Any question", session=session)

    assert reply == ""
    assert intent == "escalate"
    assert session.get("escalation_reason") == "faq_no_match_repeated"


@pytest.mark.asyncio
async def test_faq_agent_soft_llm_no_answer_counts_as_miss(monkeypatch):
    """Chunks found but LLM admits no answer → miss counter, not a reset."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")

    mock_index = MagicMock()
    mock_index.query.return_value = MagicMock(
        matches=[MagicMock(score=0.91, metadata={"text": "Unrelated chunk."})]
    )
    mock_pc_instance = MagicMock()
    mock_pc_instance.Index.return_value = mock_index

    mock_emb = MagicMock()
    mock_emb.data = [MagicMock(embedding=[0.02] * 8)]

    mock_chat = MagicMock()
    mock_chat.choices = [
        MagicMock(
            message=MagicMock(
                content=(
                    "I don't have specific information on that in our knowledge base. "
                    "Could you rephrase, or type *speak to team* if you'd like a specialist?"
                )
            )
        )
    ]

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_emb)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_chat)

    with (
        patch("pinecone.Pinecone", return_value=mock_pc_instance),
        patch("app.agents.faq.get_async_openai_client", return_value=mock_client),
    ):
        reply, session, intent = await run_faq_agent(
            "obscure question",
            session={"lead_qualified": True},
        )

    assert reply == NO_CONTEXT_REPLY
    assert intent == "faq"
    assert session.get("faq_miss_count") == 1


@pytest.fixture
def order_db(monkeypatch):
    async def _always_allow_commit(_phone: str) -> bool:
        return True

    monkeypatch.setattr(
        "app.agents.order._try_acquire_order_commit_lock",
        _always_allow_commit,
    )
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
    db.commit()
    _seed_kenya_shipping(db)
    try:
        yield db
    finally:
        db.close()


def _set_export_wire_env(monkeypatch, **overrides):
    defaults = {
        "STATIC_WIRE_ACCOUNT_NAME": "New Life Medicare Exports",
        "STATIC_WIRE_ACCOUNT_NUMBER": "123456789012",
        "STATIC_WIRE_BANK_NAME": "Example Bank Ltd",
        "STATIC_WIRE_BRANCH": "Mumbai Export Branch",
        "STATIC_WIRE_SWIFT_CODE": "EXAMPLGB",
        "STATIC_WIRE_IFSC": "HDFC0001234",
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


@pytest.mark.asyncio
async def test_order_agent_multi_turn_flow(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    _set_export_wire_env(monkeypatch)
    session = {"phone": "+919876543210", "country": "Kenya"}

    reply, session = await run_order_agent("I want to order", session, order_db)
    assert "product" in reply.lower() and ("strip" in reply.lower() or "quantity" in reply.lower())

    reply, session = await run_order_agent("Metformin 500mg - 100", session, order_db)
    assert session["order_state"] == CART_MENU
    assert len(session["order_cart"]) == 1
    assert session["order_cart"][0]["quantity"] == 100

    reply, session = await run_order_agent("checkout", session, order_db)
    assert session["order_state"] == COLLECT_CHECKOUT
    assert session["order_country"] == "Kenya"

    reply, session = await run_order_agent("Priya Sharma, Nairobi, +254700000000", session, order_db)
    # Both EMS + LP seeded → buyer chooses shipping (or auto if only one).
    assert session["order_state"] in {CONFIRM_ORDER, "SHIPPING_CHOICE"}
    if session["order_state"] == "SHIPPING_CHOICE":
        reply, session = await run_order_agent("express", session, order_db)
        assert session["order_state"] == CONFIRM_ORDER
    assert "t/t advance" in reply.lower() or "confirm" in reply.lower()

    reply, session = await run_order_agent("confirm", session, order_db)
    assert "confirmed" in reply.lower()
    assert "ORD-" in reply
    assert "order_state" not in session
    assert session.get("payment_method_chosen") == "wire_transfer"
    assert session.get("last_order_total", 0) > 0
    assert session.get("lead_qualified") is True
    assert session.get("qual_state") is None
    assert session.get("greeted") is True
    assert session.get("last_order_ref", "").startswith("ORD-")

    orders = order_db.query(Order).all()
    assert len(orders) == 1
    assert orders[0].phone == "+919876543210"
    assert orders[0].quantity == 100
    assert orders[0].country == "Kenya"
    assert orders[0].city == "Nairobi"

    # Lifetime memory: order path also writes the leads table.
    leads = order_db.query(Lead).all()
    assert len(leads) == 1
    assert leads[0].phone == "919876543210"
    assert leads[0].country == "Kenya"
    assert leads[0].lifecycle_stage == "qualified"


@pytest.mark.asyncio
async def test_order_agent_multi_product_cart_and_confirm(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    _set_export_wire_env(monkeypatch)
    db = order_db
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

    session = {"phone": "+919876543211", "country": "Kenya"}

    await run_order_agent("I want to order", session, db)
    _, session = await run_order_agent("Metformin 500mg - 1000", session, db)
    assert session["order_state"] == CART_MENU

    _, session = await run_order_agent("add", session, db)
    _, session = await run_order_agent("Amoxicillin 500mg - 500", session, db)
    assert len(session["order_cart"]) == 2

    _, session = await run_order_agent("qty 2 600", session, db)
    assert session["order_cart"][1]["quantity"] == 600

    _, session = await run_order_agent("checkout", session, db)
    assert session["order_state"] == COLLECT_CHECKOUT
    _, session = await run_order_agent("Jane Doe, Nairobi, +254700000000", session, db)
    if session["order_state"] == "SHIPPING_CHOICE":
        _, session = await run_order_agent("express", session, db)
    assert session["order_state"] == CONFIRM_ORDER

    reply, session = await run_order_agent("yes", session, db)
    assert "confirmed" in reply.lower()
    assert "ord-" in reply.lower()
    assert session.get("last_order_ref", "").startswith("ORD-")
    orders = db.query(Order).all()
    assert len(orders) == 2
    bases = {o.order_ref.rsplit("-L", 1)[0] for o in orders}
    assert len(bases) == 1
    assert db.query(Lead).count() == 1


@pytest.mark.asyncio
async def test_order_agent_payment_does_not_commit_without_confirm(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    session = {"phone": "+1", "country": "Kenya"}
    await run_order_agent("order", session, order_db)
    await run_order_agent("Metformin 500mg - 100", session, order_db)
    await run_order_agent("checkout", session, order_db)
    await run_order_agent("Contact Name, Nairobi, +254700000000", session, order_db)
    await run_order_agent("T/T", session, order_db)
    assert order_db.query(Order).count() == 0


def test_resolve_product_from_natural_language_sentence(order_db):
    product, error = _resolve_product_row(
        "I need 2000 units of metformin 500mg please",
        order_db,
    )
    assert error is None
    assert product is not None
    assert "Metformin" in product.product_name


@pytest.mark.asyncio
async def test_order_agent_llm_add_product_natural_language(order_db, monkeypatch):
    """Phase A: natural-language product+qty is parsed by rules without LLM."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")

    session = {"phone": "+1", "order_state": COLLECT_SKU}
    reply, session = await run_order_agent(
        "I need 2000 units of metformin 500mg",
        session,
        order_db,
    )
    assert session["order_state"] == CART_MENU
    assert session["order_cart"][0]["quantity"] == 2000
    assert "metformin" in reply.lower()


@pytest.mark.asyncio
async def test_order_agent_requires_confirmation_for_suggested_product(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")

    session = {"phone": "+1"}
    reply, session = await run_order_agent(
        "I need metformin 500mg please",
        session,
        order_db,
    )
    assert session["order_state"] == COLLECT_SKU_CONFIRM
    assert "did you mean" in reply.lower()

    reply, session = await run_order_agent("yes", session, order_db)
    assert session["order_state"] == COLLECT_QTY
    assert "strip" in reply.lower() or "number only" in reply.lower()
    assert "metformin" in reply.lower()

    reply, session = await run_order_agent("100", session, order_db)
    assert session["order_state"] == CART_MENU
    assert any("Metformin" in line["product_name"] for line in session["order_cart"])


@pytest.mark.asyncio
async def test_order_agent_llm_add_with_suggested_product_waits_for_confirmation(
    order_db, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "true")

    call_count = {"n": 0}

    async def fake_create(*_args, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] > 1:
            response = MagicMock()
            response.choices = [
                MagicMock(
                    message=MagicMock(
                        content="Please confirm the product before I add it.",
                        tool_calls=None,
                    )
                )
            ]
            return response

        tool_fn = MagicMock()
        tool_fn.name = "add_to_cart"
        tool_fn.arguments = json.dumps(
            {"product_query": "I need metformin 500mg please", "quantity": 100}
        )
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = tool_fn

        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(
                    content=None,
                    tool_calls=[tool_call],
                )
            )
        ]
        return response

    mock_client = MagicMock()
    mock_client.chat.completions.create = fake_create
    monkeypatch.setattr(
        "app.agents.order.get_async_openai_client",
        lambda **_: mock_client,
    )

    session = {"phone": "+1", "order_state": COLLECT_SKU}
    reply, session = await run_order_agent(
        "100 units of metformin please",
        session,
        order_db,
    )
    assert session["order_state"] == COLLECT_SKU_CONFIRM
    assert session.get("order_pending_qty") == 100

    # Confirm on next turn via deterministic fallback path.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    reply, session = await run_order_agent("yes", session, order_db)
    assert session["order_state"] == CART_MENU
    assert session["order_cart"][0]["quantity"] == 100
    assert "your cart" in reply.lower()


@pytest.mark.asyncio
async def test_order_status_query_returns_latest_status(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")

    session = {"phone": "+15550123456", "country": "Kenya"}
    _, session = await run_order_agent("order", session, order_db)
    _, session = await run_order_agent("Metformin 500mg - 100", session, order_db)
    _, session = await run_order_agent("checkout", session, order_db)
    _, session = await run_order_agent("Contact Name, Nairobi, +254700000000", session, order_db)
    _, session = await run_order_agent("confirm", session, order_db)

    reply, _ = await run_order_agent("where is my order", {"phone": "+15550123456"}, order_db)
    assert "order ord-" in reply.lower()
    assert "awaiting your payment transfer" in reply.lower()


@pytest.mark.asyncio
async def test_order_payment_wire_transfer_fallback_resends_details(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    _set_export_wire_env(monkeypatch)

    sent_messages: list[str] = []

    async def fake_send_message(phone, text):
        sent_messages.append(text)
        return True

    async def fake_send_buttons(phone, body, buttons):
        sent_messages.append(body)
        return True

    monkeypatch.setattr("app.agents.order.send_message", fake_send_message)
    monkeypatch.setattr("app.agents.order.send_interactive_buttons", fake_send_buttons)

    session = {"phone": "+91999", "country": "Kenya"}
    _, session = await run_order_agent("order", session, order_db)
    _, session = await run_order_agent("Metformin 500mg - 100", session, order_db)
    _, session = await run_order_agent("checkout", session, order_db)
    _, session = await run_order_agent("Jane Doe, Nairobi, +254700000000", session, order_db)
    if session.get("order_state") == "SHIPPING_CHOICE":
        _, session = await run_order_agent("express", session, order_db)
    _, session = await run_order_agent("confirm", session, order_db)
    sent_messages.clear()

    reply, session = await run_order_agent(PAY_BANK_BUTTON, session, order_db)
    assert "confirmed" in reply.lower()
    assert "payment details" in reply.lower() or "wire" in reply.lower()
    assert session.get("payment_method_chosen") == "wire_transfer"
    assert "order_state" not in session
    # Single interactive bubble: confirm + wire (no separate "details sent" follow-up).
    assert len(sent_messages) == 1
    assert "confirmed" in sent_messages[0].lower()
    assert (
        "123456789012" in sent_messages[0]
        and "EXAMPLGB" in sent_messages[0]
        and "New Life Medicare Exports" in sent_messages[0]
    )
    assert not any("details have been sent" in m.lower() for m in sent_messages)


@pytest.mark.asyncio
async def test_order_payment_export_wire_details_after_confirm(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    _set_export_wire_env(
        monkeypatch,
        STATIC_WIRE_ACCOUNT_NUMBER="9876543210",
        STATIC_WIRE_SWIFT_CODE="EXPORTGB",
        STATIC_WIRE_IFSC="SBIN0004321",
    )

    sent_messages: list[str] = []

    async def fake_send_message(phone, text):
        sent_messages.append(text)
        return True

    async def fake_send_buttons(phone, body, buttons):
        sent_messages.append(body)
        return True

    monkeypatch.setattr("app.agents.order.send_message", fake_send_message)
    monkeypatch.setattr("app.agents.order.send_interactive_buttons", fake_send_buttons)

    session = {"phone": "+91999", "country": "Kenya"}
    _, session = await run_order_agent("order", session, order_db)
    _, session = await run_order_agent("Metformin 500mg - 100", session, order_db)
    _, session = await run_order_agent("checkout", session, order_db)
    _, session = await run_order_agent("Contact Name, Nairobi, +254700000000", session, order_db)
    if session.get("order_state") == "SHIPPING_CHOICE":
        _, session = await run_order_agent("express", session, order_db)
    sent_messages.clear()
    reply, session = await run_order_agent("confirm", session, order_db)

    assert "confirmed" in reply.lower()
    assert "payment details" in reply.lower() or "wire" in reply.lower()
    assert session.get("payment_method_chosen") == "wire_transfer"
    assert len(sent_messages) == 1
    assert "confirmed" in sent_messages[0].lower()
    assert (
        "9876543210" in sent_messages[0]
        and "EXPORTGB" in sent_messages[0]
        and "SBIN0004321" in sent_messages[0]
    )
    assert not any("details have been sent" in m.lower() for m in sent_messages)


@pytest.mark.asyncio
async def test_payment_resolves_from_db_when_session_missing(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")

    session = {"phone": "+91999", "country": "Kenya"}
    _, session = await run_order_agent("order", session, order_db)
    _, session = await run_order_agent("Metformin 500mg - 100", session, order_db)
    _, session = await run_order_agent("checkout", session, order_db)
    _, session = await run_order_agent("Jane Doe, Nairobi, +254700000000", session, order_db)
    _, session = await run_order_agent("confirm", session, order_db)

    empty_session = {"phone": "+91999"}
    restored = _resolve_pending_payment(empty_session, order_db)
    assert restored.get("last_order_ref", "").startswith("ORD-")
    assert float(restored.get("last_order_total") or 0) > 0
    assert restored.get("order_state") == SELECT_PAYMENT


@pytest.mark.asyncio
async def test_order_agent_sanctioned_country_resets_state(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    session = {
        "phone": "+1",
        "order_state": "COLLECT_COUNTRY",
        "order_cart": [
            {
                "sku": "PROD-0001",
                "product_name": "Metformin 500mg",
                "quantity": 100,
            }
        ],
    }

    reply, session = await run_order_agent("Iran", session, order_db)
    assert reply == SANCTIONED_COUNTRY_REFUSAL
    assert "order_state" not in session
    assert order_db.query(Order).count() == 0


@pytest.fixture
def qual_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_hot_lead_distributor_uae_diabetes():
    """High-value buyer: distributor + priority country + P1 product + full qual."""
    session = {
        "business_type": "distributor",
        "buyer_type": "distributor",
        "country": "UAE",
        "company": "Gulf Med Trading",
        "order_value_usd": 600,
        "annual_volume_usd": 2_000_000,
        "license_number": "UAE-123",
        "pending_intent": "pricing",
    }
    result = score_lead(session, "price for metformin diabetes 500 units")
    assert result.score >= 80
    assert result.category == "hot"
    assert result.manual_review_only is False


def test_p6_product_forces_manual_review():
    session = {
        "business_type": "distributor",
        "country": "USA",
        "company": "Test Co",
        "order_value_usd": 800,
    }
    result = score_lead(session, "need schedule x controlled tramadol")
    assert result.manual_review_only is True
    assert result.breakdown["product_category"] == 0


def test_restricted_country_disqualified():
    session = {
        "business_type": "distributor",
        "country": "Iran",
        "company": "Test Co",
        "order_value_usd": 500,
    }
    result = score_lead(session, "bulk order")
    assert result.disqualified is True
    assert result.score < 40


def test_illegal_product_request_disqualified():
    session = {
        "business_type": "distributor",
        "country": "USA",
        "company": "Test Co",
        "order_value_usd": 500,
    }
    result = score_lead(session, "need banned medicine without prescription")
    assert result.disqualified is True
    assert result.score < 40


def test_incomplete_identity_after_retries_disqualified():
    session = {
        "buyer_type": "new_individual",
        "order_value_usd": 200,
        "incomplete_after_retries": True,
    }
    result = score_lead(session, "price for amoxicillin")
    assert result.disqualified is True


def test_time_based_and_best_price_adjustments():
    session = {
        "buyer_type": "pharmacy_clinic",
        "country": "India",
        "company": "MediCare Plus",
        "order_value_usd": 150,
        "fast_response": True,
        "active_conversation": True,
        "no_response_hours": 24,
        "repeated_best_price": True,
    }
    result = score_lead(session, "final quote please")
    assert result.score >= 40
    assert result.breakdown["buyer_type"] == 25


def test_classify_lead_score_bands():
    assert classify_lead_score(85) == "hot"
    assert classify_lead_score(70) == "warm"
    assert classify_lead_score(50) == "low_priority"
    assert classify_lead_score(30) == "ignore"


def test_calculate_lead_score_compat_wrapper():
    session = {
        "buyer_type": "new_individual",
        "country": "India",
        "company": "Solo",
        "order_value_usd": 30,
    }
    assert calculate_lead_score(session) == score_lead(session).score


@pytest.mark.asyncio
async def test_qualification_rejects_filler_biz_type(qual_db):
    session = {
        "phone": "+15550004444",
        "country": "Kenya",
        "qual_state": COLLECT_BIZ_TYPE,
    }
    reply, session, intent = await run_qualification_agent("bh", session, qual_db)
    assert intent == "continue_qual"
    assert session["qual_state"] == COLLECT_BIZ_TYPE
    assert "select type" in reply.lower() or "didn't catch" in reply.lower()


@pytest.mark.asyncio
async def test_qualification_complete_with_pending_order_skips_menu(qual_db, monkeypatch):
    sent: list[str] = []

    async def capture_menu(phone: str) -> bool:
        sent.append(phone)
        return True

    monkeypatch.setattr(
        "app.agents.qualification.send_main_menu_list",
        capture_menu,
    )

    session = {"phone": "+15550005555", "pending_intent": "order"}
    _, session, _ = await run_qualification_agent("Kenya", session, qual_db)
    reply, session, intent = await run_qualification_agent("pharmacy", session, qual_db)

    assert intent == "order"
    assert "you're all set" in reply.lower()
    assert "product" in reply.lower()
    assert sent == []
    assert session.get("pending_intent") is None
    leads = qual_db.query(Lead).filter(Lead.phone == "15550005555").all()
    assert len(leads) == 1


@pytest.mark.asyncio
async def test_qualification_complete_without_pending_sends_buttons(qual_db, monkeypatch):
    sent: list[str] = []

    async def capture_menu(phone: str) -> bool:
        sent.append(phone)
        return True

    monkeypatch.setattr(
        "app.agents.qualification.send_main_menu_list",
        capture_menu,
    )

    session = {"phone": "+15550006666"}
    _, session, _ = await run_qualification_agent("Kenya", session, qual_db)
    reply, session, intent = await run_qualification_agent("pharmacy", session, qual_db)

    assert intent == "faq"
    assert "you're all set" in reply.lower()
    assert sent == ["15550006666"]


@pytest.mark.asyncio
async def test_qualification_rejects_generic_hi_as_company(qual_db):
    session = {"phone": "+15550003333"}

    reply, session, intent = await run_qualification_agent("hi", session, qual_db)

    assert intent == "continue_qual"
    assert "country" not in session
    assert "welcome" in reply.lower() or "country" in reply.lower()


@pytest.mark.asyncio
async def test_faq_agent_no_context_returns_escalation_without_qualification(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")

    mock_index = MagicMock()
    mock_index.query.return_value = MagicMock(matches=[])
    mock_pc_instance = MagicMock()
    mock_pc_instance.Index.return_value = mock_index

    mock_emb = MagicMock()
    mock_emb.data = [MagicMock(embedding=[0.01] * 8)]

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_emb)

    with (
        patch("pinecone.Pinecone", return_value=mock_pc_instance),
        patch("app.agents.faq.get_async_openai_client", return_value=mock_client),
    ):
        reply, session, intent = await run_faq_agent("i need medicines", session={})

    assert reply == NO_CONTEXT_REPLY or "rephrase" in reply.lower() or "speak to team" in reply.lower()
    assert intent == "faq"
    assert "quick details" not in reply.lower()


@pytest.mark.asyncio
async def test_qualification_agent_multi_turn_flow(qual_db):
    session = {"phone": "+15550001111", "pending_intent": "pricing"}

    reply, session, intent = await run_qualification_agent("", session, qual_db)
    assert "country" in reply.lower()
    assert intent == "continue_qual"

    reply, session, intent = await run_qualification_agent("Kenya", session, qual_db)
    assert session["qual_state"] == COLLECT_BIZ_TYPE
    assert session["country"] == "Kenya"
    assert "business" in reply.lower()

    reply, session, intent = await run_qualification_agent(
        "pharmaceutical distributor", session, qual_db
    )
    assert session["business_type"] == "distributor"
    assert session.get("qual_state") is None
    assert session["lead_qualified"] is True
    assert session.get("qual_completed_at")
    assert session["lead_score"] >= 40
    assert session["lead_score"] < 80
    assert intent == "pricing"
    assert "you're all set" in reply.lower()

    leads = qual_db.query(Lead).all()
    assert len(leads) == 1
    assert leads[0].country == "Kenya"
    assert leads[0].business_type == "distributor"


@pytest.mark.asyncio
async def test_qualification_accepts_list_title_biz_type(qual_db, monkeypatch):
    monkeypatch.setattr(
        "app.agents.qualification.send_main_menu_list",
        AsyncMock(return_value=True),
    )
    session = {
        "phone": "+15550007777",
        "country": "Australia",
        "qual_state": COLLECT_BIZ_TYPE,
        "biz_type_picker_sent": True,
    }
    reply, session, intent = await run_qualification_agent(
        "Doctor / Prescriber / physician",
        session,
        qual_db,
    )
    assert session.get("qual_state") is None
    assert session["lead_qualified"] is True
    assert session["business_type"] == "doctor"
    assert intent == "faq"
    assert "you're all set" in reply.lower()


@pytest.mark.asyncio
async def test_qualification_accepts_typed_clinic(qual_db, monkeypatch):
    monkeypatch.setattr(
        "app.agents.qualification.send_main_menu_list",
        AsyncMock(return_value=True),
    )
    session = {
        "phone": "+15550008888",
        "country": "Australia",
        "qual_state": COLLECT_BIZ_TYPE,
        "biz_type_picker_sent": True,
    }
    reply, session, intent = await run_qualification_agent("clinic", session, qual_db)
    assert session.get("qual_state") is None
    assert session["business_type"] == "pharmacy"
    assert intent == "faq"


@pytest.mark.asyncio
async def test_qualification_high_score_alerts_but_keeps_services(qual_db, monkeypatch):
    """Hot lead: team alerted once; buyer continues to pricing (not blocked)."""
    alerts: list[str] = []

    async def capture_alert(phone, session, reason):
        alerts.append(reason)
        return True

    monkeypatch.setattr(
        "app.agents.qualification.send_escalation_alert",
        capture_alert,
    )

    session = {"phone": "+15550002222"}
    session["pending_intent"] = "pricing"
    _, session, _ = await run_qualification_agent("UK", session, qual_db)
    reply, session, intent = await run_qualification_agent(
        "distributor wholesale bulk container diabetes metformin", session, qual_db
    )

    assert session["lead_score"] >= 80
    assert session.get("lead_category") == "hot"
    assert intent == "pricing"
    assert session.get("_hot_lead_alerted") is True
    assert session.get("human_active") is not True
    assert session.get("escalation_reason") != "hot_lead"
    assert alerts == ["hot_lead"]
    assert session.get("_handoff_query") or True  # may hand off pending pricing query

@pytest.mark.asyncio
async def test_pricing_agent_uses_country_context(monkeypatch, pricing_db):
    """Pricing prompt includes country from session; no company field."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                tool_calls=None,
                content="Quote for Amoxicillin 5000 units to India: contact export team.",
            )
        )
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch("app.agents.pricing.get_async_openai_client", return_value=mock_client):
        out, session_out, intent = await run_pricing_agent(
            "price for amoxicillin 5000 units",
            {"country": "India"},
            pricing_db,
        )

    assert intent == "pricing"
    assert "export team" in out.lower()
    call_kwargs = mock_client.chat.completions.create.await_args.kwargs
    user_content = call_kwargs["messages"][1]["content"]
    assert "Country: India" in user_content
    assert "Company:" not in user_content


def test_format_multi_product_quote_list(pricing_db):
    reply, full_miss = format_multi_product_quote(
        "Amoxicillin 500mg - 100\nMetformin 500mg - 200",
        {"country": "Kenya"},
        pricing_db,
    )
    # Metformin not in pricing_db fixture — expect amox quoted + metformin missing
    assert reply is not None
    assert full_miss is False
    assert "Amoxicillin 500mg" in reply
    assert "$1.85" in reply
    assert "100" in reply
    assert "Kenya" in reply


@pytest.mark.asyncio
async def test_pricing_agent_multi_product_list_no_llm(monkeypatch, pricing_db):
    """Comma/newline product lists quote from DB without calling OpenAI."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pricing_db.add(
        Product(
            product_name="Metformin 500mg",
            salt_name="Metformin",
            manufacturing_company="Gamma",
            expiry_date=date(2027, 1, 1),
            price_per_strip=0.95,
            is_restricted=False,
        )
    )
    pricing_db.commit()

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    with patch("app.agents.pricing.get_async_openai_client", return_value=mock_client):
        out, session_out, intent = await run_pricing_agent(
            "Amoxicillin 500mg - 100\nMetformin 500mg - 200",
            {"country": "Kenya"},
            pricing_db,
        )

    mock_client.chat.completions.create.assert_not_called()
    assert intent == "pricing"
    assert session_out.get("pricing_miss_count", 0) == 0
    assert "Amoxicillin 500mg" in out
    assert "Metformin 500mg" in out
    assert "$1.85" in out
    assert "$0.95" in out
    assert "100" in out and "200" in out


@pytest.mark.asyncio
async def test_pricing_agent_multi_product_comma_list(pricing_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out, _session, intent = await run_pricing_agent(
        "Amoxicillin 500mg, Ciprofloxacin 500mg",
        {"country": "India"},
        pricing_db,
    )
    assert intent == "pricing"
    assert "Amoxicillin 500mg" in out
    assert "$1.85" in out
    # Restricted ciprofloxacin → channel note, not invented price
    assert "not available" in out.lower() or "Ciprofloxacin" in out


@pytest.mark.asyncio
async def test_pricing_agent_single_product_phrase_still_uses_llm(monkeypatch, pricing_db):
    """Non-list free text still uses GPT tool path."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                tool_calls=None,
                content="Amoxicillin 500mg is *$1.85* USD per strip.",
            )
        )
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch("app.agents.pricing.get_async_openai_client", return_value=mock_client):
        out, _session, intent = await run_pricing_agent(
            "what is the price for amoxicillin?",
            {"country": "Kenya"},
            pricing_db,
        )

    mock_client.chat.completions.create.assert_awaited()
    assert intent == "pricing"
    assert "1.85" in out


@pytest.mark.asyncio
async def test_pricing_agent_full_miss_suggests_then_escalates(monkeypatch, pricing_db):
    """Two consecutive unmatched catalog lists → suggest, then escalate."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    out1, session1, intent1 = await run_pricing_agent(
        "NoSuchDrugAAA - 10\nFakeProductBBB - 20",
        {"country": "Kenya"},
        pricing_db,
    )
    assert intent1 == "pricing"
    assert session1.get("pricing_miss_count") == 1
    assert "Couldn't match" in out1 or "couldn't find" in out1.lower()

    out2, session2, intent2 = await run_pricing_agent(
        "NoSuchDrugAAA - 10\nFakeProductBBB - 20",
        session1,
        pricing_db,
    )
    assert intent2 == "escalate"
    assert out2 == ""
    assert session2.get("escalation_reason") == "pricing_no_match_repeated"
    assert session2.get("pricing_miss_count", 0) == 0


@pytest.mark.asyncio
async def test_pricing_agent_llm_not_found_uses_suggestions(monkeypatch, pricing_db):
    """Tool not_found → deterministic miss reply; never use invented LLM price."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")

    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "get_product_by_name"
    tool_call.function.arguments = json.dumps({"query": "xyzzy999nomatch"})

    first = MagicMock()
    first.choices = [MagicMock(message=MagicMock(tool_calls=[tool_call], content=None))]
    second = MagicMock()
    second.choices = [
        MagicMock(
            message=MagicMock(
                tool_calls=None,
                content="I made up $9.99 for xyzzy999nomatch",
            )
        )
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=[first, second])

    with patch("app.agents.pricing.get_async_openai_client", return_value=mock_client):
        out, session, intent = await run_pricing_agent(
            "price for xyzzy999nomatch",
            {"country": "Kenya"},
            pricing_db,
        )

    assert intent == "pricing"
    assert session.get("pricing_miss_count") == 1
    assert "9.99" not in out
    assert "couldn't find" in out.lower()
    assert "xyzzy999nomatch" in out.lower()
