"""Intent router — human keywords, qualify-before-pricing, LLM classification."""

from unittest.mock import AsyncMock

import pytest

from app.agents import router as router_mod
from app.orchestrator import graph as graph_mod


@pytest.mark.asyncio
async def test_human_keyword_escalate_async():
    intent, session = await router_mod.classify_intent("Please connect me to an agent", {})
    assert intent == "escalate"
    assert session == {}


@pytest.mark.asyncio
async def test_unqualified_pricing_routes_to_qualify(monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("pricing", 0.9)),
    )
    intent, session = await router_mod.classify_intent("quote for metformin", {})
    assert intent == "qualify"
    assert session["pending_intent"] == "pricing"


@pytest.mark.asyncio
async def test_unqualified_order_routes_to_qualify(monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("order", 0.85)),
    )
    intent, session = await router_mod.classify_intent("I want to buy 500 units", {})
    assert intent == "qualify"
    assert session["pending_intent"] == "order"


@pytest.mark.asyncio
async def test_unqualified_faq_routes_to_qualify(monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("faq", 0.9)),
    )
    intent, session = await router_mod.classify_intent("i need medicines", {})
    assert intent == "qualify"
    assert session["pending_intent"] == "faq"


@pytest.mark.asyncio
async def test_qualified_low_confidence_retries_faq_first(monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("faq", 0.35)),
    )
    intent, session = await router_mod.classify_intent(
        "something vague",
        {"lead_qualified": True},
    )
    assert intent == "faq"
    assert session["clarification_attempts"] == 1


@pytest.mark.asyncio
async def test_qualified_low_confidence_escalates_after_retries(monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("faq", 0.35)),
    )
    intent, _ = await router_mod.classify_intent(
        "something vague",
        {"lead_qualified": True, "clarification_attempts": 1},
    )
    assert intent == "escalate"


@pytest.mark.asyncio
async def test_qualified_high_confidence_returns_intent(monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("pricing", 0.9)),
    )
    intent, session = await router_mod.classify_intent(
        "price for amoxicillin",
        {"lead_qualified": True},
    )
    assert intent == "pricing"
    assert "pending_intent" not in session


@pytest.mark.asyncio
async def test_keyword_fallback_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    intent, session = await router_mod.classify_intent("what documents do you provide", {})
    assert intent == "qualify"
    assert session["pending_intent"] == "faq"


def test_parse_classifier_response_valid():
    intent, conf = router_mod._parse_classifier_response(
        '{"intent": "order", "confidence": 0.88}'
    )
    assert intent == "order"
    assert conf == 0.88


def test_parse_classifier_response_invalid_intent_defaults_faq():
    intent, conf = router_mod._parse_classifier_response(
        '{"intent": "unknown", "confidence": 0.9}'
    )
    assert intent == "faq"
    assert conf == 0.9


@pytest.mark.asyncio
async def test_graph_router_node_continues_order_state(monkeypatch):
    monkeypatch.setattr(
        graph_mod,
        "classify_intent",
        AsyncMock(return_value=("faq", {})),
    )
    out = await graph_mod.router_node(
        {
            "phone": "1",
            "message": "hello",
            "message_id": "m1",
            "session": {"order_state": "COLLECT_QTY", "lead_qualified": True},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out == {
        "intent": "order",
        "session": {"order_state": "COLLECT_QTY", "lead_qualified": True},
    }
    graph_mod.classify_intent.assert_not_called()


@pytest.mark.asyncio
async def test_graph_router_node_continues_qual_state(monkeypatch):
    monkeypatch.setattr(
        graph_mod,
        "classify_intent",
        AsyncMock(return_value=("pricing", {})),
    )
    out = await graph_mod.router_node(
        {
            "phone": "1",
            "message": "Kenya",
            "message_id": "m1",
            "session": {"qual_state": "COLLECT_COUNTRY"},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out["intent"] == "qualify"
    graph_mod.classify_intent.assert_not_called()


@pytest.mark.asyncio
async def test_graph_router_node_delegates_to_classify_intent(monkeypatch):
    mock_classify = AsyncMock(return_value=("qualify", {"pending_intent": "pricing"}))
    monkeypatch.setattr(graph_mod, "classify_intent", mock_classify)

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
    mock_classify.assert_awaited_once_with("quote for 200 units metformin", {})
