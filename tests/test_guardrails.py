"""Guardrail pre/post checks and DB logging."""

from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, GuardrailLog
from app.guardrails.check import (
    REFUSAL_CLINICAL_CONTENT,
    REFUSAL_RESTRICTED_PRODUCT,
    REFUSAL_SANCTIONED_COUNTRY,
    check_post_guardrails,
    check_pre_guardrails,
    log_guardrail,
)
from app.messages.welcome import AI_DISCLOSURE_MESSAGE
from app.orchestrator import graph as graph_mod


@pytest.fixture
def guardrail_db():
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


def test_pre_guardrails_blocks_sanctioned_session_country():
    result = check_pre_guardrails("hello", {"country": "Iran"})
    assert result.blocked is True
    assert result.reason == "sanctioned_country"
    assert result.refusal_message == REFUSAL_SANCTIONED_COUNTRY


def test_pre_guardrails_blocks_disqualified_flag():
    result = check_pre_guardrails("pricing", {"disqualified": True, "country": "United Kingdom"})
    assert result.blocked is True
    assert result.reason == "disqualified_lead"
    assert result.refusal_message == REFUSAL_SANCTIONED_COUNTRY


def test_pre_guardrails_blocks_sanctioned_country_case_insensitive():
    result = check_pre_guardrails("quote please", {"country": "PAKISTAN"})
    assert result.blocked is True
    assert result.reason == "sanctioned_country"


def test_pre_guardrails_blocks_hard_blocked_product():
    result = check_pre_guardrails("Do you sell Schedule H products?", {})
    assert result.blocked is True
    assert result.reason == "restricted_product"
    assert result.refusal_message == REFUSAL_RESTRICTED_PRODUCT


def test_pre_guardrails_blocks_schedule_h1_drug_name():
    result = check_pre_guardrails("What is your price for tramadol 50mg?", {})
    assert result.blocked is True
    assert result.reason == "restricted_product"


def test_pre_guardrails_blocks_schedule_x_drug_name():
    result = check_pre_guardrails("Need ketamine export quote", {})
    assert result.blocked is True
    assert result.reason == "restricted_product"


def test_pre_guardrails_passes_clean_message():
    result = check_pre_guardrails("price for metformin", {"country": "Kenya"})
    assert result.blocked is False


def test_post_guardrails_blocks_clinical_content():
    result = check_post_guardrails("The recommended dosage is 500mg twice daily.")
    assert result.blocked is True
    assert result.reason == "clinical_content"
    assert result.refusal_message == REFUSAL_CLINICAL_CONTENT


def test_post_guardrails_passes_business_reply():
    result = check_post_guardrails("We ship via DHL within 7-10 business days.")
    assert result.blocked is False


@pytest.mark.asyncio
async def test_log_guardrail_writes_row(guardrail_db, monkeypatch):
    monkeypatch.setattr(
        "app.guardrails.check.SessionLocal",
        sessionmaker(bind=guardrail_db.get_bind()),
    )

    await log_guardrail("+91999", "restricted_product", "pre", "schedule h query")

    row = guardrail_db.query(GuardrailLog).one()
    assert row.phone == "+91999"
    assert row.reason == "restricted_product"
    assert row.trigger_type == "pre"
    assert row.message_text == "schedule h query"


@pytest.mark.asyncio
async def test_log_guardrail_truncates_message_text(guardrail_db, monkeypatch):
    monkeypatch.setattr(
        "app.guardrails.check.SessionLocal",
        sessionmaker(bind=guardrail_db.get_bind()),
    )

    long_text = "x" * 300
    await log_guardrail("+91999", "clinical_content", "post", long_text)

    row = guardrail_db.query(GuardrailLog).one()
    assert len(row.message_text) == 200


@pytest.mark.asyncio
async def test_pre_guardrails_node_blocks_and_sets_final_reply(monkeypatch):
    logged: list[tuple] = []

    async def fake_log(phone, reason, stage, message_text=""):
        logged.append((phone, reason, stage, message_text))

    monkeypatch.setattr(graph_mod, "log_guardrail", fake_log)

    out = await graph_mod.pre_guardrails_node(
        {
            "phone": "+91999",
            "message": "narcotic products",
            "message_id": "m1",
            "session": {},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )

    assert out["guardrail_blocked"] is True
    assert out["final_reply"] == REFUSAL_RESTRICTED_PRODUCT
    assert logged == [("+91999", "restricted_product", "pre", "narcotic products")]


@pytest.mark.asyncio
async def test_post_guardrails_node_replaces_clinical_response(monkeypatch):
    logged: list[tuple] = []

    async def fake_log(phone, reason, stage, message_text=""):
        logged.append((phone, reason, stage))

    monkeypatch.setattr(graph_mod, "log_guardrail", fake_log)

    out = await graph_mod.post_guardrails_node(
        {
            "phone": "+91999",
            "message": "side effects?",
            "message_id": "m1",
            "session": {},
            "intent": "faq",
            "agent_response": "Common side effects include nausea.",
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )

    assert out["final_reply"] == REFUSAL_CLINICAL_CONTENT
    assert logged == [("+91999", "clinical_content", "post")]


def test_route_after_pre_guardrails_blocked_skips_router():
    assert graph_mod._route_after_pre_guardrails({"guardrail_blocked": True}) == "send_reply"
    assert graph_mod._route_after_pre_guardrails({"guardrail_blocked": False}) == "router"


def test_route_to_agent_guardrail_blocked_skips_agents():
    assert graph_mod._route_to_agent({"guardrail_blocked": True, "intent": "pricing"}) == "send_reply"


def test_route_to_agent_human_active_silent_drop():
    assert graph_mod._route_to_agent(
        {"guardrail_blocked": False, "session": {"human_active": True}}
    ) == "human_active"


@pytest.mark.asyncio
async def test_pre_guardrail_blocked_skips_agents_in_graph(monkeypatch):
    import fakeredis

    from app.session import manager as session_manager

    sent: list[str] = []

    async def capture_send(phone: str, text: str) -> bool:
        sent.append(text)
        return True

    async def fail_classify(*_args, **_kwargs):
        raise AssertionError("router should not run when pre-guardrail blocks")

    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(session_manager, "_get_redis_client", lambda: fake_redis)
    monkeypatch.setattr(graph_mod, "send_message", capture_send)
    monkeypatch.setattr("app.messages.welcome.send_message", capture_send)
    monkeypatch.setattr(
        "app.messages.conversation_ui.send_main_menu_list",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(graph_mod, "send_main_menu_list", AsyncMock(return_value=True))
    monkeypatch.setattr(graph_mod, "send_navigation_footer", AsyncMock(return_value=True))
    monkeypatch.setattr(graph_mod, "classify_intent", fail_classify)
    monkeypatch.setattr(graph_mod, "log_guardrail", AsyncMock())

    await graph_mod.compiled_graph.ainvoke(
        {
            "phone": "+91999111",
            "message": "price for ketamine",
            "message_id": "m1",
            "session": {},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )

    assert REFUSAL_RESTRICTED_PRODUCT in sent[-1]
    assert any("AI assistant" in msg or "AI sales assistant" in msg for msg in sent)
