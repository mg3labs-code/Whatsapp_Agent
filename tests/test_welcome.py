"""AI disclosure at conversation start."""

from unittest.mock import AsyncMock

import pytest

from app.messages.welcome import (
    AI_DISCLOSURE_MESSAGE,
    prepend_ai_disclosure,
    should_send_ai_disclosure,
)
from app.orchestrator import graph as graph_mod


def test_should_send_ai_disclosure_new_session():
    assert should_send_ai_disclosure({}) is True


def test_should_send_ai_disclosure_already_sent():
    assert should_send_ai_disclosure({"ai_disclosure_sent": True}) is False


def test_should_send_ai_disclosure_skips_human_active():
    assert should_send_ai_disclosure({"human_active": True}) is False


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
async def test_send_reply_node_prepends_disclosure_once(monkeypatch):
    sent: list[str] = []
    saved: list[dict] = []

    async def capture_send(phone: str, text: str) -> bool:
        sent.append(text)
        return True

    async def capture_save(phone: str, data: dict) -> None:
        saved.append(data)

    monkeypatch.setattr(graph_mod, "send_message", capture_send)
    monkeypatch.setattr(graph_mod, "save_session", capture_save)

    state: graph_mod.MessageState = {
        "phone": "+91999",
        "message": "hi",
        "message_id": "m1",
        "session": {},
        "intent": "qualify",
        "agent_response": None,
        "guardrail_blocked": False,
        "final_reply": "May I get your company name?",
    }
    await graph_mod.send_reply_node(state)

    assert len(sent) == 1
    assert sent[0].startswith(AI_DISCLOSURE_MESSAGE)
    assert "company name" in sent[0].lower()
    assert saved[0]["ai_disclosure_sent"] is True


@pytest.mark.asyncio
async def test_send_reply_node_skips_disclosure_when_already_sent(monkeypatch):
    sent: list[str] = []

    monkeypatch.setattr(
        graph_mod,
        "send_message",
        AsyncMock(side_effect=lambda _p, text: sent.append(text) or True),
    )
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
