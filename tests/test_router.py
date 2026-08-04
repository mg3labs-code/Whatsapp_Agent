"""Intent router — human keywords, qualify-before-pricing, LLM classification."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents import router as router_mod
from app.orchestrator import graph as graph_mod


@pytest.mark.asyncio
async def test_human_keyword_escalate_async():
    intent, session = await router_mod.classify_intent("Please connect me to an agent", {})
    assert intent == "escalate"
    assert session.get("escalation_reason") == "speak_to_team"


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
async def test_menu_button_order_unqualified_without_llm():
    intent, session = await router_mod.classify_intent("order", {})
    assert intent == "qualify"
    assert session["pending_intent"] == "order"


@pytest.mark.asyncio
async def test_menu_button_speak_routes_to_escalate():
    intent, session = await router_mod.classify_intent("speak", {"lead_qualified": True})
    assert intent == "escalate"


@pytest.mark.asyncio
async def test_menu_button_pricing_qualified_without_llm():
    intent, session = await router_mod.classify_intent(
        "pricing",
        {"lead_qualified": True},
    )
    assert intent == "pricing"
    assert "pending_intent" not in session


@pytest.mark.asyncio
async def test_pure_greeting_routes_to_menu_refresh_without_llm(monkeypatch):
    mock_llm = AsyncMock()
    monkeypatch.setattr(router_mod, "_classify_with_llm", mock_llm)
    intent, session = await router_mod.classify_intent(
        "hi",
        {"lead_qualified": True},
    )
    assert intent == "menu_refresh"
    assert session.get("lead_qualified") is True
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_unqualified_pure_greeting_routes_to_menu_refresh(monkeypatch):
    mock_llm = AsyncMock()
    monkeypatch.setattr(router_mod, "_classify_with_llm", mock_llm)
    intent, _ = await router_mod.classify_intent("hello", {})
    assert intent == "menu_refresh"
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_greeting_with_pricing_still_uses_llm(monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("pricing", 0.9)),
    )
    intent, session = await router_mod.classify_intent(
        "hi, price for metformin",
        {"lead_qualified": True},
    )
    assert intent == "pricing"
    assert "pending_intent" not in session


@pytest.mark.asyncio
async def test_catalog_product_name_routes_to_pricing_without_llm(monkeypatch):
    """Bare product name (screenshot KLENSMART) → pricing, not FAQ."""
    from datetime import date

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db.models import Base, Product

    mock_llm = AsyncMock()
    monkeypatch.setattr(router_mod, "_classify_with_llm", mock_llm)

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    db.add(
        Product(
            product_name="KLENSMART 60MG",
            salt_name="Clenbuterol",
            manufacturing_company="TestCo",
            expiry_date=date(2027, 1, 1),
            price_per_strip=2.5,
            is_restricted=False,
        )
    )
    db.commit()
    try:
        intent, session = await router_mod.classify_intent(
            "• KLENSMART 60MG",
            {"lead_qualified": True},
            db=db,
        )
        assert intent == "pricing"
        mock_llm.assert_not_called()

        intent2, session2 = await router_mod.classify_intent(
            "Klensmart",
            {"lead_qualified": True},
            db=db,
        )
        assert intent2 == "pricing"
        assert session2.get("lead_qualified") is True
    finally:
        db.close()


@pytest.mark.asyncio
async def test_catalog_product_unqualified_goes_to_qualify(monkeypatch):
    from datetime import date

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db.models import Base, Product

    monkeypatch.setattr(router_mod, "_classify_with_llm", AsyncMock())

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    db.add(
        Product(
            product_name="Amoxicillin 500mg",
            salt_name="Amoxicillin",
            manufacturing_company="Acme",
            expiry_date=date(2027, 1, 1),
            price_per_strip=1.0,
            is_restricted=False,
        )
    )
    db.commit()
    try:
        intent, session = await router_mod.classify_intent(
            "Amoxicillin 500mg",
            {},
            db=db,
        )
        assert intent == "qualify"
        assert session["pending_intent"] == "pricing"
        assert "Amoxicillin" in session["pending_query"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_shipping_question_with_product_name_stays_faq(monkeypatch):
    """Do not force pricing when message is clearly a process/FAQ question."""
    from datetime import date

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db.models import Base, Product

    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("faq", 0.9)),
    )

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    db.add(
        Product(
            product_name="Metformin 500mg",
            salt_name="Metformin",
            manufacturing_company="Acme",
            expiry_date=date(2027, 1, 1),
            price_per_strip=1.0,
            is_restricted=False,
        )
    )
    db.commit()
    try:
        intent, _ = await router_mod.classify_intent(
            "do you ship Metformin 500mg to Kenya?",
            {"lead_qualified": True},
            db=db,
        )
        assert intent == "faq"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_qualified_never_routes_to_qualify_from_llm(monkeypatch):
    """Non-greeting qualify from LLM → FAQ (not re-qual). Greetings use menu_refresh."""
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("qualify", 0.9)),
    )
    intent, session = await router_mod.classify_intent(
        "just browsing generally",
        {"lead_qualified": True},
    )
    assert intent == "faq"
    assert session.get("lead_qualified") is True


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
async def test_unqualified_faq_routes_directly_to_faq(monkeypatch):
    monkeypatch.setattr(
        router_mod,
        "_classify_with_llm",
        AsyncMock(return_value=("faq", 0.9)),
    )
    intent, session = await router_mod.classify_intent("what are your shipping timelines", {})
    assert intent == "faq"
    assert "pending_intent" not in session


@pytest.mark.asyncio
async def test_unqualified_faq_menu_button_skips_qualification():
    intent, session = await router_mod.classify_intent("faq", {})
    assert intent == "faq"
    assert "pending_intent" not in session


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
    assert intent == "faq"
    assert "pending_intent" not in session


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
async def test_graph_router_node_cancel_routes_to_order():
    out = await graph_mod.router_node(
        {
            "phone": "1",
            "message": "cancel",
            "message_id": "m1",
            "session": {"lead_qualified": True},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out == {"intent": "order", "session": {"lead_qualified": True}}


@pytest.mark.asyncio
async def test_graph_router_node_main_menu_routes_to_menu_refresh():
    out = await graph_mod.router_node(
        {
            "phone": "1",
            "message": "main_menu",
            "message_id": "m1",
            "session": {"order_state": "COLLECT_QTY"},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out == {"intent": "menu_refresh", "session": {"order_state": "COLLECT_QTY"}}


@pytest.mark.asyncio
async def test_graph_router_node_free_text_menu_escapes_order_state():
    """Mid-order 'menu' opens main menu (same as Main Menu button)."""
    out = await graph_mod.router_node(
        {
            "phone": "1",
            "message": "menu",
            "message_id": "m1",
            "session": {"order_state": "COLLECT_QTY", "lead_qualified": True},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out["intent"] == "menu_refresh"


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
async def test_graph_router_skips_stale_qual_when_already_qualified(monkeypatch):
    """Bug fix: qualified buyers with leftover qual_state must reach pricing/order/FAQ."""
    mock_classify = AsyncMock(
        return_value=("pricing", {"lead_qualified": True, "pending_menu_ack": "pricing"})
    )
    monkeypatch.setattr(graph_mod, "classify_intent", mock_classify)

    def _fake_db_gen():
        yield MagicMock()

    monkeypatch.setattr(graph_mod, "_get_db_generator", _fake_db_gen)

    out = await graph_mod.router_node(
        {
            "phone": "1",
            "message": "pricing",
            "message_id": "m1",
            "session": {
                "lead_qualified": True,
                "qual_state": "COLLECT_BIZ_TYPE",
                "biz_type_picker_sent": True,
                "pending_intent": "pricing",
            },
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out["intent"] == "pricing"
    assert out["session"].get("qual_state") is None
    assert out["session"].get("pending_intent") is None
    mock_classify.assert_awaited_once()
    # Stale flags cleared before classify_intent sees the session.
    called_session = mock_classify.await_args.args[1]
    assert called_session.get("lead_qualified") is True
    assert called_session.get("qual_state") is None


@pytest.mark.asyncio
async def test_menu_refresh_clears_stale_qual_for_qualified_user(monkeypatch):
    monkeypatch.setattr(graph_mod, "send_main_menu_list", AsyncMock(return_value=True))
    monkeypatch.setattr(graph_mod, "send_message", AsyncMock(return_value=True))

    out = await graph_mod.menu_refresh_node(
        {
            "phone": "919999000111",
            "message": "main_menu",
            "message_id": "m1",
            "session": {
                "lead_qualified": True,
                "qual_state": "COLLECT_BIZ_TYPE",
                "order_state": "CART_MENU",
                "order_cart": [{"sku": "PROD-1"}],
            },
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    session = out["session"]
    assert session.get("lead_qualified") is True
    assert session.get("qual_state") is None
    assert session.get("order_state") is None
    assert session.get("order_cart") is None
    assert session.get("skip_welcome_compose") is True
    assert session.get("ai_disclosure_sent") is True
    assert session.get("suppress_nav_footer") is True
    graph_mod.send_main_menu_list.assert_awaited_once_with("919999000111")
    # Disclosure sent once for new Redis session (compliance).
    graph_mod.send_message.assert_awaited_once()
    assert "AI assistant" in graph_mod.send_message.await_args.args[1]


@pytest.mark.asyncio
async def test_menu_refresh_clears_unfinished_mid_qual(monkeypatch):
    """Unqualified buyers who open Main Menu must not stay stuck in qual_state."""
    monkeypatch.setattr(graph_mod, "send_main_menu_list", AsyncMock(return_value=True))
    monkeypatch.setattr(graph_mod, "send_message", AsyncMock(return_value=True))

    out = await graph_mod.menu_refresh_node(
        {
            "phone": "919999000222",
            "message": "menu",
            "message_id": "m2",
            "session": {
                "qual_state": "COLLECT_COUNTRY",
                "country_picker_sent": True,
                "pending_intent": "pricing",
            },
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    session = out["session"]
    assert session.get("qual_state") is None
    assert session.get("country_picker_sent") is None
    assert session.get("pending_intent") is None


@pytest.mark.asyncio
async def test_graph_router_speak_escapes_mid_qual():
    out = await graph_mod.router_node(
        {
            "phone": "1",
            "message": "speak to team",
            "message_id": "m1",
            "session": {"qual_state": "COLLECT_COUNTRY"},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out["intent"] == "escalate"
    assert out["session"].get("escalation_reason") == "speak_to_team"


@pytest.mark.asyncio
async def test_graph_router_faq_escapes_mid_qual(monkeypatch):
    monkeypatch.setattr(
        graph_mod,
        "classify_intent",
        AsyncMock(side_effect=AssertionError("should not classify")),
    )
    out = await graph_mod.router_node(
        {
            "phone": "1",
            "message": "faq",
            "message_id": "m1",
            "session": {"qual_state": "COLLECT_BIZ_TYPE", "country": "Kenya"},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out["intent"] == "faq"


@pytest.mark.asyncio
async def test_menu_refresh_skips_duplicate_disclosure(monkeypatch):
    monkeypatch.setattr(graph_mod, "send_main_menu_list", AsyncMock(return_value=True))
    monkeypatch.setattr(graph_mod, "send_message", AsyncMock(return_value=True))

    out = await graph_mod.menu_refresh_node(
        {
            "phone": "919999000111",
            "message": "hi",
            "message_id": "m2",
            "session": {
                "lead_qualified": True,
                "ai_disclosure_sent": True,
            },
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out["session"].get("ai_disclosure_sent") is True
    graph_mod.send_message.assert_not_called()
    graph_mod.send_main_menu_list.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_router_node_delegates_to_classify_intent(monkeypatch):
    mock_classify = AsyncMock(return_value=("qualify", {"pending_intent": "pricing"}))
    monkeypatch.setattr(graph_mod, "classify_intent", mock_classify)

    def _fake_db_gen():
        yield MagicMock()

    monkeypatch.setattr(graph_mod, "_get_db_generator", _fake_db_gen)

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
    mock_classify.assert_awaited_once()
    assert mock_classify.await_args.args[0] == "quote for 200 units metformin"
    assert mock_classify.await_args.kwargs.get("db") is not None
