"""AI disclosure at conversation start."""

from unittest.mock import AsyncMock

import pytest

from app.messages.welcome import (
    AI_DISCLOSURE_MESSAGE,
    GREETING_LIST_BODY,
    prepend_ai_disclosure,
    send_greeting_menu,
    should_send_ai_disclosure,
    should_send_greeting_buttons,
)
from app.orchestrator import graph as graph_mod


def test_should_send_ai_disclosure_new_session():
    assert should_send_ai_disclosure({}) is True


def test_should_send_ai_disclosure_already_sent():
    assert should_send_ai_disclosure({"ai_disclosure_sent": True}) is False


def test_should_send_ai_disclosure_skips_human_active():
    assert should_send_ai_disclosure({"human_active": True}) is False


def test_should_send_greeting_buttons_new_session():
    assert should_send_greeting_buttons({}) is True


def test_should_send_greeting_buttons_already_sent():
    assert should_send_greeting_buttons({"greeting_buttons_sent": True}) is False


@pytest.mark.asyncio
async def test_send_greeting_menu_sets_flags(monkeypatch):
    texts: list[str] = []
    lists: list[str] = []

    async def capture_text(phone: str, body: str) -> bool:
        texts.append(body)
        return True

    async def capture_menu(phone: str, **kwargs) -> bool:
        lists.append(kwargs.get("body", ""))
        return True

    monkeypatch.setattr("app.messages.welcome.send_message", capture_text)
    monkeypatch.setattr("app.messages.conversation_ui.send_main_menu_list", capture_menu)

    session, ok = await send_greeting_menu("+91999", {})
    assert ok is True
    assert session["greeting_buttons_sent"] is True
    assert session["ai_disclosure_sent"] is True
    assert len(texts) == 1
    assert "AI sales assistant" in texts[0]
    assert lists == [GREETING_LIST_BODY]


@pytest.mark.asyncio
async def test_greeting_node_sends_menu_once(monkeypatch):
    out = await graph_mod.greeting_node(
        {
            "phone": "+91999",
            "message": "hi",
            "message_id": "m1",
            "session": {},
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out["session"]["greeted"] is True
    assert out["greeting"] is True

    out2 = await graph_mod.greeting_node(
        {
            "phone": "+91999",
            "message": "hi again",
            "message_id": "m2",
            "session": out["session"],
            "intent": None,
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )
    assert out2["greeting"] is False


def test_prepend_ai_disclosure_merges_and_sets_flag():
    reply, session = prepend_ai_disclosure("Welcome! Company name?", {})

    assert session["ai_disclosure_sent"] is True
    assert reply.startswith(AI_DISCLOSURE_MESSAGE)
    assert "Welcome! Company name?" in reply


def test_prepend_ai_disclosure_only_once():
    first, session = prepend_ai_disclosure("First reply", {})
    second, session = prepend_ai_disclosure("Second reply", session)

    assert first.startswith(AI_DISCLOSURE_MESSAGE)
    assert second == "Second reply"
    assert "AI assistant" not in second


@pytest.mark.asyncio
async def test_send_reply_node_skips_disclosure_when_greeting_sent(monkeypatch):
    sent: list[str] = []

    monkeypatch.setattr(
        graph_mod,
        "send_message",
        AsyncMock(side_effect=lambda _p, text: sent.append(text) or True),
    )
    monkeypatch.setattr(graph_mod, "send_navigation_footer", AsyncMock(return_value=True))
    monkeypatch.setattr(graph_mod, "save_session", AsyncMock())

    await graph_mod.send_reply_node(
        {
            "phone": "+91999",
            "message": "hi",
            "message_id": "m1",
            "session": {
                "greeting_buttons_sent": True,
                "ai_disclosure_sent": True,
            },
            "intent": "qualify",
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": "May I get your company name?",
        }
    )

    assert sent == ["May I get your company name?"]
    assert "AI assistant" not in sent[0]


@pytest.mark.asyncio
async def test_send_reply_node_prepends_disclosure_once(monkeypatch):
    sent: list[str] = []
    saved: list[dict] = []
    sent_buttons: list[str] = []

    async def capture_send(phone: str, text: str) -> bool:
        sent.append(text)
        return True

    async def capture_save(phone: str, data: dict) -> None:
        saved.append(data)

    async def capture_menu(phone: str, **kwargs) -> bool:
        sent_buttons.append(phone)
        return True

    monkeypatch.setattr(graph_mod, "send_message", capture_send)
    monkeypatch.setattr(graph_mod, "send_main_menu_list", capture_menu)
    monkeypatch.setattr(graph_mod, "send_navigation_footer", AsyncMock(return_value=True))
    monkeypatch.setattr(graph_mod, "save_session", capture_save)

    state: graph_mod.MessageState = {
        "phone": "+91999",
        "message": "hi",
        "message_id": "m1",
        "session": {},
        "intent": "qualify",
        "agent_response": None,
        "guardrail_blocked": False,
        "final_reply": "Which country are you based in?",
        "greeting": True,
    }
    await graph_mod.send_reply_node(state)

    assert len(sent) == 1
    assert sent[0].startswith("Hi! 👋 I'm the AI assistant for *New Life Medicare*")
    assert "country" in sent[0].lower()
    assert saved[0].get("phone") == "91999"
    assert sent_buttons == ["+91999"]


@pytest.mark.asyncio
async def test_send_reply_node_skips_disclosure_when_already_sent(monkeypatch):
    sent: list[str] = []

    monkeypatch.setattr(
        graph_mod,
        "send_message",
        AsyncMock(side_effect=lambda _p, text: sent.append(text) or True),
    )
    monkeypatch.setattr(graph_mod, "send_navigation_footer", AsyncMock(return_value=True))
    monkeypatch.setattr(graph_mod, "save_session", AsyncMock())

    await graph_mod.send_reply_node(
        {
            "phone": "+91999",
            "message": "Kenya",
            "message_id": "m2",
            "session": {"ai_disclosure_sent": True, "qual_state": "COLLECT_BIZ_TYPE"},
            "intent": "qualify",
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": "What type of business are you?",
        }
    )

    assert sent == ["What type of business are you?"]
