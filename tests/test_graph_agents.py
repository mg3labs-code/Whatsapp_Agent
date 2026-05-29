"""Orchestrator routing and real pricing / FAQ agent wiring."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import fakeredis
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from unittest.mock import AsyncMock

from app.db.models import Base, Order, Product
from app.orchestrator import graph as graph_mod
from app.session import manager as session_manager


@pytest.mark.asyncio
async def test_router_node_routes_unqualified_pricing_to_qualify(monkeypatch):
    monkeypatch.setattr(
        graph_mod,
        "classify_intent",
        AsyncMock(return_value=("qualify", {"pending_intent": "pricing"})),
    )
    out = await graph_mod.router_node(
        {
            "phone": "1",
            "message": "quote for 200 units metformin",
            "message_id": "m1",
            "session": {},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out == {"intent": "qualify", "session": {"pending_intent": "pricing"}}


@pytest.mark.asyncio
async def test_router_node_returns_pricing_when_qualified(monkeypatch):
    monkeypatch.setattr(
        graph_mod,
        "classify_intent",
        AsyncMock(return_value=("pricing", {"lead_qualified": True})),
    )
    out = await graph_mod.router_node(
        {
            "phone": "1",
            "message": "quote for 200 units metformin",
            "message_id": "m1",
            "session": {"lead_qualified": True},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out == {"intent": "pricing", "session": {"lead_qualified": True}}


@pytest.mark.asyncio
async def test_pricing_agent_node_calls_run_pricing_agent_and_closes_gen(monkeypatch):
    closed: list[str] = []

    async def fake_run(message: str, session: dict, db):
        assert message == "hi"
        assert session == {"company": "Acme", "phone": "+1", "last_agent": "pricing"}
        assert db is mock_db
        return "PRICE_REPLY"

    mock_db = MagicMock()

    def mock_factory():
        def mock_get_db():
            try:
                yield mock_db
            finally:
                closed.append("gen_finally")

        return mock_get_db()

    monkeypatch.setattr(graph_mod, "_get_db_generator", mock_factory)
    monkeypatch.setattr(graph_mod, "run_pricing_agent", fake_run)

    state: graph_mod.MessageState = {
        "phone": "+1",
        "message": "hi",
        "message_id": "x",
        "session": {"company": "Acme"},
        "intent": "pricing",
        "agent_response": None,
        "guardrail_blocked": False,
        "final_reply": None,
    }
    out = await graph_mod.pricing_agent_node(state)
    assert out == {
        "agent_response": "PRICE_REPLY",
        "session": {"company": "Acme", "phone": "+1", "last_agent": "pricing"},
    }
    assert closed == ["gen_finally"]


@pytest.mark.asyncio
async def test_faq_agent_node_calls_run_faq_agent(monkeypatch):
    async def fake_run(message: str, phone: str = "", session: dict | None = None) -> str:
        assert message == "what documents"
        assert phone == "+1"
        return "FAQ_REPLY"

    monkeypatch.setattr(graph_mod, "run_faq_agent", fake_run)
    state: graph_mod.MessageState = {
        "phone": "+1",
        "message": "what documents",
        "message_id": "x",
        "session": {},
        "intent": "faq",
        "agent_response": None,
        "guardrail_blocked": False,
        "final_reply": None,
    }
    out = await graph_mod.faq_agent_node(state)
    assert out == {
        "agent_response": "FAQ_REPLY",
        "session": {"phone": "+1", "last_agent": "faq"},
    }


@pytest.mark.asyncio
async def test_order_agent_node_updates_session(monkeypatch):
    closed: list[str] = []

    async def fake_run(message: str, session: dict, db):
        assert message == "Metformin"
        session = dict(session)
        session["order_state"] = "COLLECT_QTY"
        return "How many units?", session

    mock_db = MagicMock()

    def mock_factory():
        def mock_get_db():
            try:
                yield mock_db
            finally:
                closed.append("gen_finally")

        return mock_get_db()

    monkeypatch.setattr(graph_mod, "_get_db_generator", mock_factory)
    monkeypatch.setattr(graph_mod, "run_order_agent", fake_run)

    state: graph_mod.MessageState = {
        "phone": "+91999",
        "message": "Metformin",
        "message_id": "x",
        "session": {},
        "intent": "order",
        "agent_response": None,
        "guardrail_blocked": False,
        "final_reply": None,
    }
    out = await graph_mod.order_agent_node(state)
    assert out["agent_response"] == "How many units?"
    assert out["session"]["order_state"] == "COLLECT_QTY"
    assert out["session"]["phone"] == "+91999"
    assert closed == ["gen_finally"]


def test_route_after_qualify_mid_flow_goes_to_send_reply():
    state: graph_mod.MessageState = {
        "phone": "+1",
        "message": "Kenya",
        "message_id": "x",
        "session": {"qual_state": "COLLECT_COUNTRY"},
        "intent": None,
        "agent_response": "And which country?",
        "guardrail_blocked": False,
        "final_reply": None,
    }
    assert graph_mod._route_after_qualify(state) == "send_reply"


def test_route_after_qualify_complete_routes_to_pending_agent():
    state: graph_mod.MessageState = {
        "phone": "+1",
        "message": "no",
        "message_id": "x",
        "session": {"lead_qualified": True},
        "intent": "faq",
        "agent_response": "Thank you!",
        "guardrail_blocked": False,
        "final_reply": None,
    }
    assert graph_mod._route_after_qualify(state) == "faq"


@pytest.mark.asyncio
async def test_qualify_agent_node_redirects_intent_after_complete(monkeypatch):
    closed: list[str] = []

    async def fake_run(message: str, session: dict, db):
        return "Thank you!", {**session, "lead_qualified": True}, "pricing"

    mock_db = MagicMock()

    def mock_factory():
        def mock_get_db():
            try:
                yield mock_db
            finally:
                closed.append("gen_finally")

        return mock_get_db()

    monkeypatch.setattr(graph_mod, "_get_db_generator", mock_factory)
    monkeypatch.setattr(graph_mod, "run_qualification_agent", fake_run)

    state: graph_mod.MessageState = {
        "phone": "+91999",
        "message": "no",
        "message_id": "x",
        "session": {"qual_state": "COLLECT_LICENSE"},
        "intent": "qualify",
        "agent_response": None,
        "guardrail_blocked": False,
        "final_reply": None,
    }
    out = await graph_mod.qualify_agent_node(state)
    assert out["intent"] == "pricing"
    assert out["session"]["lead_qualified"] is True
    assert closed == ["gen_finally"]


@pytest.fixture
def graph_order_db():
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
async def test_scenario_b_multi_turn_order_via_graph(monkeypatch, graph_order_db):
    """docs/AGENTS.md Scenario B — order flow through compiled_graph + Redis session."""
    from app.agents import router as router_mod

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORDER_AGENT_USE_LLM", "false")
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("order", 0.9)),
    )
    phone = "+919876543210"
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(session_manager, "_get_redis_client", lambda: fake_redis)
    await session_manager.save_session(phone, {"lead_qualified": True})

    sent: list[tuple[str, str]] = []

    async def capture_send(to_phone: str, text: str) -> bool:
        sent.append((to_phone, text))
        return True

    monkeypatch.setattr(graph_mod, "send_message", capture_send)

    def db_factory():
        def gen():
            try:
                yield graph_order_db
            finally:
                pass

        return gen()

    monkeypatch.setattr(graph_mod, "_get_db_generator", db_factory)

    turns = [
        ("m1", "I want to order"),
        ("m2", "Metformin 500mg"),
        ("m3", "0"),
        ("m4", "2000"),
        ("m5", "done"),
        ("m6", "Kenya"),
        ("m7", "Nairobi"),
        ("m8", "Priya Sharma, MedEx"),
        ("m9", "T/T advance"),
        ("m10", "confirm"),
    ]

    for mid, text in turns:
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

    assert len(sent) == 10
    final_reply = sent[-1][1].lower()
    assert "order confirmed" in final_reply
    assert "ord-" in final_reply

    session = await session_manager.get_session(phone)
    assert "order_state" not in session
    assert graph_order_db.query(Order).count() == 1
