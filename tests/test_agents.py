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
    COLLECT_CITY,
    COLLECT_COUNTRY,
    COLLECT_QTY,
    SANCTIONED_COUNTRY_REFUSAL,
    run_order_agent,
)
from app.agents.pricing import get_product_by_name, run_pricing_agent
from app.agents.lead_scoring import (
    classify_lead_score,
    score_lead,
)
from app.agents.qualification import (
    COLLECT_BIZ_TYPE,
    COLLECT_COUNTRY as QUAL_COLLECT_COUNTRY,
    calculate_lead_score,
    run_qualification_agent,
)
from app.db.models import Base, Lead, Order, Product


def _fixed_now(tz, year: int, month: int, day: int, hour: int):
    return tz.localize(datetime(year, month, day, hour, 0))


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

    assert result == {"error": "product_not_found", "query": "xyz999"}


def test_get_product_by_name_restricted(pricing_db):
    result = get_product_by_name("Ciprofloxacin", pricing_db)

    assert result == {
        "error": "product_restricted",
        "name": "Ciprofloxacin 500mg",
        "schedule_category": "H",
    }


@pytest.mark.asyncio
async def test_run_faq_agent_missing_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    out = await run_faq_agent("Do you ship to Nigeria?")

    assert out == ERROR_REPLY


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
        out = await run_faq_agent("Any question")

    assert "connect you" in out.lower()
    assert out == NO_CONTEXT_REPLY
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
        out = await run_faq_agent("Do you export to Kenya?")

    assert isinstance(out, str)
    assert len(out) > 0
    mock_client.chat.completions.create.assert_awaited_once()
    call_kw = mock_client.chat.completions.create.await_args
    assert call_kw.kwargs["model"] == "gpt-4o-mini"
    messages = call_kw.kwargs["messages"]
    assert messages[0]["content"] == FAQ_SYSTEM_PROMPT
    assert "We ship worldwide" in messages[1]["content"]


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
    db.commit()
    try:
        yield db
    finally:
        db.close()


@pytest.mark.asyncio
async def test_order_agent_multi_turn_flow(order_db):
    session = {"phone": "+919876543210"}

    reply, session = await run_order_agent("I want to order", session, order_db)
    assert "which product" in reply.lower()

    reply, session = await run_order_agent("Metformin 500mg", session, order_db)
    assert session["order_state"] == COLLECT_QTY
    assert session["order_product_name"] == "Metformin 500mg"
    assert "how many units" in reply.lower()

    reply, session = await run_order_agent("0", session, order_db)
    assert session["order_state"] == COLLECT_QTY
    assert "minimum" in reply.lower() or "number" in reply.lower()

    reply, session = await run_order_agent("2000", session, order_db)
    assert session["order_state"] == COLLECT_COUNTRY
    assert session["order_qty"] == 2000

    reply, session = await run_order_agent("Kenya", session, order_db)
    assert session["order_state"] == COLLECT_CITY
    assert session["order_country"] == "Kenya"

    reply, session = await run_order_agent("Nairobi", session, order_db)
    assert "name and company" in reply.lower()

    reply, session = await run_order_agent("Priya Sharma, MedEx", session, order_db)
    assert "payment terms" in reply.lower()

    reply, session = await run_order_agent("T/T Advance", session, order_db)
    assert "order confirmed" in reply.lower()
    assert "ORD-" in reply
    assert "order_state" not in session

    orders = order_db.query(Order).all()
    assert len(orders) == 1
    assert orders[0].phone == "+919876543210"
    assert orders[0].quantity == 2000
    assert orders[0].country == "Kenya"
    assert orders[0].city == "Nairobi"


@pytest.mark.asyncio
async def test_order_agent_sanctioned_country_resets_state(order_db):
    session = {
        "phone": "+1",
        "order_state": "COLLECT_COUNTRY",
        "order_sku": "PROD-0001",
        "order_product_name": "Metformin 500mg",
        "order_qty": 100,
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
async def test_qualification_agent_multi_turn_flow(qual_db):
    session = {"phone": "+15550001111", "pending_intent": "pricing"}

    reply, session, intent = await run_qualification_agent("", session, qual_db)
    assert "welcome" in reply.lower()
    assert intent == "continue_qual"

    reply, session, intent = await run_qualification_agent("MedEx Distributors LLC", session, qual_db)
    assert session["qual_state"] == QUAL_COLLECT_COUNTRY
    assert session["company"] == "MedEx Distributors LLC"
    assert "country" in reply.lower()

    reply, session, intent = await run_qualification_agent("Kenya", session, qual_db)
    assert session["qual_state"] == COLLECT_BIZ_TYPE
    assert session["country"] == "Kenya"

    reply, session, intent = await run_qualification_agent(
        "pharmaceutical distributor", session, qual_db
    )
    assert session["business_type"] == "distributor"
    assert "volume" in reply.lower()

    reply, session, intent = await run_qualification_agent("$50,000", session, qual_db)
    assert session.get("annual_volume_usd") == 50_000.0 or session.get("order_value_usd")
    assert "license" in reply.lower()

    reply, session, intent = await run_qualification_agent("no", session, qual_db)
    assert session.get("qual_state") is None
    assert session["lead_qualified"] is True
    assert session["lead_score"] >= 40
    assert intent in {"pricing", "faq", "escalate"}
    assert "thank you" in reply.lower()

    leads = qual_db.query(Lead).all()
    assert len(leads) == 1
    assert leads[0].company == "MedEx Distributors LLC"
    assert leads[0].country == "Kenya"
    assert leads[0].business_type == "distributor"


@pytest.mark.asyncio
async def test_qualification_high_score_escalates(qual_db):
    session = {"phone": "+15550002222"}

    _, session, _ = await run_qualification_agent("NHS Supply Chain", session, qual_db)
    _, session, _ = await run_qualification_agent("UK", session, qual_db)
    _, session, _ = await run_qualification_agent("distributor wholesale", session, qual_db)
    _, session, _ = await run_qualification_agent("$2 million", session, qual_db)
    reply, session, intent = await run_qualification_agent("LIC-998877", session, qual_db)

    assert session["lead_score"] >= 80
    assert session.get("lead_category") == "hot"
    assert intent == "escalate"
    assert "senior export manager" in reply.lower()


@pytest.mark.asyncio
async def test_pricing_agent_asks_for_company(monkeypatch, pricing_db):
    """Session missing company → model may reply without tool; expect company ask."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                tool_calls=None,
                content=(
                    "I'd be happy to help with pricing. Could you please share your "
                    "*company name* and country so I can prepare an accurate quote?"
                ),
            )
        )
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch("app.agents.pricing.get_async_openai_client", return_value=mock_client):
        out = await run_pricing_agent("price for amoxicillin 5000 units", {}, pricing_db)

    assert "company" in out.lower()
