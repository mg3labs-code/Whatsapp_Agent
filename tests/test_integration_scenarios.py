"""Task 5.5 — end-to-end scenario tests (local graph + agents; ngrok manual separate)."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import pytest
import pytz
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents import escalation as escalation_mod
from app.agents import router as router_mod
from app.agents.faq import NO_CONTEXT_REPLY, run_faq_agent
from app.messages.welcome import AI_DISCLOSURE_MESSAGE
from app.agents.lead_scoring import score_lead
from app.agents.qualification import run_qualification_agent
from app.business import hours as business_hours
from app.db.models import Base, GuardrailLog, Order, Product
from app.guardrails.check import (
    REFUSAL_RESTRICTED_PRODUCT,
    check_pre_guardrails,
)
from app.integrations import alerts as alerts_mod
from app.orchestrator import graph as graph_mod
from app.session import manager as session_manager

PHONE = "+919876543210"


def _fixed_now(tz, year: int, month: int, day: int, hour: int):
    return tz.localize(datetime(year, month, day, hour, 0))


@pytest.fixture
def integration_db():
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
            manufacturing_company="Beta Labs",
            expiry_date=date(2027, 6, 1),
            price_per_strip=1.20,
            is_restricted=False,
        )
    )
    db.commit()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def graph_env(monkeypatch, integration_db):
    """Redis + DB + no outbound WhatsApp / team alerts."""
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(session_manager, "_get_redis_client", lambda: fake_redis)

    sent_buyer: list[str] = []
    team_alerts: list[str] = []

    async def capture_buyer(phone: str, text: str) -> bool:
        sent_buyer.append(text)
        return True

    async def capture_team(text: str) -> bool:
        team_alerts.append(text)
        return True

    monkeypatch.setattr(graph_mod, "send_message", capture_buyer)
    monkeypatch.setattr("app.messages.welcome.send_message", capture_buyer)
    monkeypatch.setattr(
        "app.messages.welcome.send_interactive_list",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(graph_mod, "send_navigation_footer", AsyncMock(return_value=True))
    monkeypatch.setattr(alerts_mod, "send_leads_alert", capture_team)
    monkeypatch.setattr(alerts_mod, "send_order_team_alert", capture_team)

    def db_factory():
        def gen():
            try:
                yield integration_db
            finally:
                pass

        return gen()

    monkeypatch.setattr(graph_mod, "_get_db_generator", db_factory)

    log_session = sessionmaker(bind=integration_db.get_bind())
    monkeypatch.setattr("app.guardrails.check.SessionLocal", log_session)

    async def fake_pricing(message: str, session: dict, db):
        return "Quote: Metformin 500mg at $0.95/strip.", session

    async def fake_faq(message: str) -> str:
        return "We ship via DHL worldwide."

    monkeypatch.setattr(graph_mod, "run_pricing_agent", fake_pricing)
    monkeypatch.setattr(graph_mod, "run_faq_agent", fake_faq)

    return {"sent_buyer": sent_buyer, "team_alerts": team_alerts, "redis": fake_redis}


def _assert_disclosure_delivered(sent: list[str], session: dict) -> None:
    assert session.get("greeted") is True
    assert any("Hi! 👋 I'm the AI assistant for *New Life Medicare*" in msg for msg in sent)


async def _invoke(phone: str, text: str, mid: str, graph_env: dict) -> None:
    await graph_mod.compiled_graph.ainvoke(
        {
            "phone": phone,
            "message": text,
            "message_id": mid,
            "session": {},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )


# --- Scenarios A–F (docs/SCENARIOS.md) ---


@pytest.mark.asyncio
async def test_scenario_a_new_buyer_price_goes_to_qualify(graph_env, monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("pricing", 0.9)),
    )
    await _invoke(PHONE, "Hi, I need price for Amoxicillin 500mg", "a1", graph_env)
    session = await session_manager.get_session(PHONE)
    assert session.get("qual_state") or session.get("pending_intent") == "pricing"
    _assert_disclosure_delivered(graph_env["sent_buyer"], session)
    reply = graph_env["sent_buyer"][-1]
    reply_lower = reply.lower()
    assert "company" in reply_lower or "welcome" in reply_lower or "country" in reply_lower


@pytest.mark.asyncio
async def test_scenario_b_multi_turn_order(graph_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("order", 0.9)),
    )
    await session_manager.save_session(PHONE, {"lead_qualified": True})
    turns = [
        ("b1", "I want to order"),
        ("b2", "Metformin 500mg"),
        ("b3", "0"),
        ("b4", "2000"),
        ("b5", "done"),
        ("b6", "Kenya"),
        ("b7", "Nairobi"),
        ("b8", "Priya Sharma, MedEx"),
        ("b9", "T/T advance"),
        ("b10", "confirm"),
    ]
    for mid, text in turns:
        await _invoke(PHONE, text, mid, graph_env)
    assert any("order confirmed" in m.lower() for m in graph_env["sent_buyer"])
    session = await session_manager.get_session(PHONE)
    assert "order_state" not in session


@pytest.mark.asyncio
async def test_scenario_c_hot_lead_escalation(graph_env, monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("qualify", 0.9)),
    )
    await session_manager.save_session(
        PHONE,
        {"pending_intent": "pricing", "phone": PHONE},
    )
    steps = [
        "NHS Supply Chain",
        "United Kingdom",
        "hospital",
        "$2 million annual",
        "LIC-UK-99",
    ]
    for i, text in enumerate(steps, start=1):
        await _invoke(PHONE, text, f"c{i}", graph_env)

    session = await session_manager.get_session(PHONE)
    assert session.get("lead_score", 0) >= 80
    assert session.get("human_active") is True
    assert any(
        "senior export manager" in m.lower() or "connect" in m.lower()
        for m in graph_env["sent_buyer"]
    )
    assert any("ESCALATION" in a for a in graph_env["team_alerts"])


@pytest.mark.asyncio
async def test_scenario_d_faq_returns_answer(graph_env, monkeypatch):
    await session_manager.save_session(PHONE, {"lead_qualified": True})
    monkeypatch.setattr(
        graph_mod,
        "run_faq_agent",
        AsyncMock(return_value="We accept LC and TT payment terms."),
    )
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("faq", 0.9)),
    )
    await _invoke(PHONE, "what payment methods do you accept", "d1", graph_env)
    assert "payment" in graph_env["sent_buyer"][-1].lower()


@pytest.mark.asyncio
async def test_scenario_e_schedule_h_blocked(graph_env, integration_db):
    await _invoke(PHONE, "Do you sell Schedule H products?", "e1", graph_env)
    reply = graph_env["sent_buyer"][-1]
    session = await session_manager.get_session(PHONE)
    _assert_disclosure_delivered(graph_env["sent_buyer"], session)
    assert REFUSAL_RESTRICTED_PRODUCT in reply
    logs = integration_db.query(GuardrailLog).all()
    assert len(logs) >= 1
    assert logs[-1].reason == "restricted_product"


@pytest.mark.asyncio
async def test_scenario_f_human_keywords_escalate_without_llm(graph_env, monkeypatch):
    classify = AsyncMock(return_value=("faq", 0.99))
    monkeypatch.setattr(router_mod, "_classify_with_llm", classify)
    await _invoke(PHONE, "I need to speak to a real person please", "f1", graph_env)
    classify.assert_not_called()
    session = await session_manager.get_session(PHONE)
    assert session.get("human_active") is True


# --- Scenarios 7–15 ---


@pytest.mark.asyncio
async def test_scenario_7_off_hours_escalation_eta(monkeypatch):
    tz = pytz.timezone("Asia/Kolkata")
    monkeypatch.setattr(
        business_hours,
        "_now_in_tz",
        lambda _tz: _fixed_now(tz, 2026, 5, 16, 23),
    )
    monkeypatch.setattr(escalation_mod, "_now_in_tz", business_hours._now_in_tz)
    monkeypatch.setattr(escalation_mod, "send_escalation_alert", AsyncMock(return_value=True))

    reply, session = await escalation_mod.run_escalation_agent(
        "help",
        {"company": "Test Co", "phone": PHONE},
        "buyer_request",
        phone=PHONE,
    )
    assert "offline" in reply.lower() or "business hours" in reply.lower()
    assert session["human_active"] is True


def test_scenario_8_schedule_h_pre_guardrail():
    result = check_pre_guardrails("price for Schedule H antibiotics", {})
    assert result.blocked
    assert result.reason == "restricted_product"


def test_scenario_9_sanctioned_country_pre_guardrail():
    result = check_pre_guardrails("hello", {"country": "Iran"})
    assert result.blocked
    assert result.reason == "sanctioned_country"


@pytest.mark.asyncio
async def test_scenario_10_unqualified_price_qualify_first(graph_env, monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("pricing", 0.88)),
    )
    await _invoke(PHONE, "quote for 200 units metformin", "s10", graph_env)
    session = await session_manager.get_session(PHONE)
    assert session.get("pending_intent") == "pricing"
    assert session.get("qual_state") or "company" in graph_env["sent_buyer"][-1].lower()


@pytest.mark.asyncio
async def test_scenario_11_order_resume_mid_flow(graph_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    await session_manager.save_session(
        PHONE,
        {
            "lead_qualified": True,
            "order_state": "COLLECT_QTY",
            "order_product_name": "Metformin 500mg",
            "order_sku": "PROD-0001",
            "order_moq": 1,
        },
    )
    await _invoke(PHONE, "2000", "s11", graph_env)
    session = await session_manager.get_session(PHONE)
    assert session.get("order_state") == "CART_MENU" or session.get("order_cart")


def test_scenario_12_hospital_uk_2m_hot_score():
    session = {
        "business_type": "hospital",
        "country": "United Kingdom",
        "company": "NHS Supply Chain",
        "annual_volume_usd": 2_000_000,
        "license_number": "LIC-1",
        "pending_intent": "pricing",
    }
    result = score_lead(session, "price for bulk order")
    assert result.score >= 80
    assert result.category == "hot"


@pytest.mark.asyncio
async def test_scenario_12_qual_complete_escalates(integration_db):
    session = {"phone": PHONE, "pending_intent": "pricing"}
    for text in (
        "NHS Supply Chain",
        "United Kingdom",
        "hospital",
        "$2 million",
        "LIC-99",
    ):
        _, session, intent = await run_qualification_agent(text, session, integration_db)
    assert session["lead_score"] >= 80
    assert intent == "escalate"


@pytest.mark.asyncio
async def test_scenario_13_faq_empty_pinecone_team_message(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("PINECONE_API_KEY", "test")

    mock_embed = MagicMock()
    mock_embed.data = [MagicMock(embedding=[0.1] * 8)]

    mock_query = MagicMock()
    mock_query.matches = []

    with patch("app.agents.faq.get_async_openai_client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_embed)
        mock_client.chat.completions.create = AsyncMock()
        mock_client_cls.return_value = mock_client
        with patch(
            "app.agents.faq._pinecone_query_sync",
            return_value=mock_query,
        ):
            reply = await run_faq_agent(
                "obscure regulatory question xyz123",
                session={"lead_qualified": True},
            )
    assert reply == NO_CONTEXT_REPLY
    assert "team" in reply.lower()
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_scenario_14_human_keywords_no_llm():
    with patch.object(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("faq", 0.99)),
    ) as classify:
        intent, _ = await router_mod.classify_intent("please connect me to an agent", {})
    assert intent == "escalate"
    classify.assert_not_called()


@pytest.mark.asyncio
async def test_scenario_15_human_active_silent_drop(graph_env):
    await session_manager.save_session(PHONE, {"human_active": True, "lead_qualified": True})
    before = len(graph_env["sent_buyer"])
    await _invoke(PHONE, "any follow up message", "s15", graph_env)
    assert len(graph_env["sent_buyer"]) == before
