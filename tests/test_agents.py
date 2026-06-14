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
    PAY_CARD_BUTTON,
    SANCTIONED_COUNTRY_REFUSAL,
    SELECT_PAYMENT,
    _resolve_pending_payment,
    _resolve_product_row,
    run_order_agent,
)
from app.agents.pricing import get_product_by_name, run_pricing_agent
from app.agents.lead_scoring import (
    classify_lead_score,
    score_lead,
)
from app.agents.qualification import (
    COLLECT_BIZ_TYPE,
    COLLECT_COUNTRY,
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
        out = await run_faq_agent("Any question", session={"lead_qualified": True})

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
async def test_order_agent_multi_turn_flow(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    session = {"phone": "+919876543210", "country": "Kenya"}

    reply, session = await run_order_agent("I want to order", session, order_db)
    assert "product" in reply.lower() and "quantity" in reply.lower()

    reply, session = await run_order_agent("Metformin 500mg - 100", session, order_db)
    assert session["order_state"] == CART_MENU
    assert len(session["order_cart"]) == 1
    assert session["order_cart"][0]["quantity"] == 100

    reply, session = await run_order_agent("checkout", session, order_db)
    assert session["order_state"] == COLLECT_CHECKOUT
    assert session["order_country"] == "Kenya"

    reply, session = await run_order_agent("Priya Sharma, Nairobi, +254700000000", session, order_db)
    assert session["order_state"] == CONFIRM_ORDER
    assert "t/t advance" in reply.lower()
    assert "confirm" in reply.lower()

    reply, session = await run_order_agent("confirm", session, order_db)
    assert "order confirmed" in reply.lower()
    assert "ORD-" in reply
    assert session["order_state"] == SELECT_PAYMENT
    assert session.get("last_order_total", 0) > 0
    assert session.get("lead_qualified") is True
    assert session.get("greeted") is True
    assert session.get("last_order_ref", "").startswith("ORD-")

    orders = order_db.query(Order).all()
    assert len(orders) == 1
    assert orders[0].phone == "+919876543210"
    assert orders[0].quantity == 100
    assert orders[0].country == "Kenya"
    assert orders[0].city == "Nairobi"


@pytest.mark.asyncio
async def test_order_agent_multi_product_cart_and_confirm(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
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
    reply, session = await run_order_agent("LC", session, db)
    assert session["order_state"] == CONFIRM_ORDER

    reply, session = await run_order_agent("yes", session, db)
    assert "order confirmed" in reply.lower()
    assert session.get("last_order_ref", "").startswith("ORD-")
    orders = db.query(Order).all()
    assert len(orders) == 2
    bases = {o.order_ref.rsplit("-L", 1)[0] for o in orders}
    assert len(bases) == 1


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
    """Natural language product match still asks quantity first."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "true")

    call_count = {"n": 0}

    async def fake_create(*_args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] > 1:
            response = MagicMock()
            response.choices = [
                MagicMock(
                    message=MagicMock(
                        content="Added *Metformin 500mg* — *2000* units to your cart.",
                        tool_calls=None,
                    )
                )
            ]
            return response

        tool_fn = MagicMock()
        tool_fn.name = "add_to_cart"
        tool_fn.arguments = json.dumps(
            {"product_query": "metformin 500mg", "quantity": 2000}
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
        "I need 2000 units of metformin 500mg",
        session,
        order_db,
    )
    assert session["order_state"] == COLLECT_SKU_CONFIRM
    assert session.get("pending_product")
    assert "did you mean" in reply.lower()


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
    assert "quantity" in reply.lower()
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
    assert session.get("order_pending_qty") is None

    # Confirm on next turn via deterministic fallback path.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    reply, session = await run_order_agent("yes", session, order_db)
    assert session["order_state"] == COLLECT_QTY
    reply, session = await run_order_agent("100", session, order_db)
    assert session["order_state"] == CART_MENU
    assert any("Metformin" in line["product_name"] for line in session["order_cart"])
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
async def test_order_payment_bank_transfer_after_confirm(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")

    sent_messages: list[str] = []

    async def fake_send_message(phone, text):
        sent_messages.append(text)
        return True

    async def fake_send_buttons(phone, body, buttons):
        sent_messages.append(body)
        return True

    async def fake_create_va(order_ref, amount, customer_phone):
        return {
            "virtual_account_id": "va_test_1",
            "account_number": "1234567890",
            "ifsc": "CF0000001",
            "iban": "GB00TEST",
            "swift_code": "CFSWIFT",
            "amount": amount,
        }

    monkeypatch.setattr("app.agents.order.send_message", fake_send_message)
    monkeypatch.setattr("app.agents.order.send_interactive_buttons", fake_send_buttons)
    monkeypatch.setattr("app.agents.order.create_virtual_account", fake_create_va)

    session = {"phone": "+91999", "country": "Kenya"}
    await run_order_agent("order", session, order_db)
    await run_order_agent("Metformin 500mg - 100", session, order_db)
    await run_order_agent("checkout", session, order_db)
    await run_order_agent("Jane Doe, Nairobi, +254700000000", session, order_db)
    _, session = await run_order_agent("confirm", session, order_db)
    assert session["order_state"] == SELECT_PAYMENT

    reply, session = await run_order_agent(PAY_BANK_BUTTON, session, order_db)
    assert "bank transfer" in reply.lower()
    assert session.get("payment_method_chosen") == "bank_transfer"
    assert "order_state" not in session
    assert any("Payment Instructions" in msg for msg in sent_messages)
    assert order_db.query(Order).first().virtual_account_id == "va_test_1"


@pytest.mark.asyncio
async def test_order_payment_card_link_after_confirm(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")

    sent_messages: list[str] = []

    async def fake_send_message(phone, text):
        sent_messages.append(text)
        return True

    async def fake_send_buttons(phone, body, buttons):
        sent_messages.append(body)
        return True

    async def fake_create_link(order_ref, amount, customer_phone, customer_name=""):
        return {
            "link_url": "https://payments-test.cashfree.com/links/test123",
            "link_id": order_ref,
            "amount": amount,
        }

    monkeypatch.setattr("app.agents.order.send_message", fake_send_message)
    monkeypatch.setattr("app.agents.order.send_interactive_buttons", fake_send_buttons)
    monkeypatch.setattr("app.agents.order.create_payment_link", fake_create_link)

    session = {"phone": "+91999", "country": "Kenya"}
    await run_order_agent("order", session, order_db)
    await run_order_agent("Metformin 500mg - 100", session, order_db)
    await run_order_agent("checkout", session, order_db)
    await run_order_agent("Contact Name, Nairobi, +254700000000", session, order_db)
    _, session = await run_order_agent("confirm", session, order_db)

    reply, session = await run_order_agent(PAY_CARD_BUTTON, session, order_db)
    assert "payment link" in reply.lower()
    assert session.get("payment_method_chosen") == "card"
    assert any("payments-test.cashfree.com" in msg for msg in sent_messages)


@pytest.mark.asyncio
async def test_payment_resolves_from_db_when_session_missing(order_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")

    session = {"phone": "+91999", "country": "Kenya"}
    await run_order_agent("order", session, order_db)
    await run_order_agent("Metformin 500mg - 100", session, order_db)
    await run_order_agent("checkout", session, order_db)
    await run_order_agent("Jane Doe, Nairobi, +254700000000", session, order_db)
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
                "moq": 1,
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
    assert "business" in reply.lower()


@pytest.mark.asyncio
async def test_qualification_complete_with_pending_order_skips_menu(qual_db, monkeypatch):
    sent: list[tuple[str, list]] = []

    async def capture_buttons(phone: str, _body: str, buttons: list) -> bool:
        sent.append((phone, buttons))
        return True

    monkeypatch.setattr(
        "app.agents.qualification.send_interactive_buttons",
        capture_buttons,
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
    sent: list[tuple[str, list]] = []

    async def capture_buttons(phone: str, _body: str, buttons: list) -> bool:
        sent.append((phone, buttons))
        return True

    monkeypatch.setattr(
        "app.agents.qualification.send_interactive_buttons",
        capture_buttons,
    )

    session = {"phone": "+15550006666"}
    _, session, _ = await run_qualification_agent("Kenya", session, qual_db)
    reply, session, intent = await run_qualification_agent("pharmacy", session, qual_db)

    assert intent == "faq"
    assert sent
    assert {b["id"] for b in sent[0][1]} == {"order", "pricing", "speak"}


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
        out = await run_faq_agent("i need medicines", session={})

    assert out == NO_CONTEXT_REPLY or "connect you" in out.lower()
    assert "quick details" not in out.lower()


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
async def test_qualification_high_score_escalates(qual_db):
    session = {"phone": "+15550002222"}

    session["pending_intent"] = "pricing"
    _, session, _ = await run_qualification_agent("UK", session, qual_db)
    reply, session, intent = await run_qualification_agent(
        "distributor wholesale bulk container diabetes metformin", session, qual_db
    )

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
