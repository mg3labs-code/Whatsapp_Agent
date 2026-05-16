"""Escalation agent and WhatsApp team alerts."""

from unittest.mock import AsyncMock

import pytest

from app.agents import escalation as escalation_mod
from app.integrations import alerts as alerts_mod


@pytest.mark.asyncio
async def test_run_escalation_agent_in_hours_sets_human_active(monkeypatch):
    alerts: list[tuple] = []

    async def capture_alert(phone, session, reason):
        alerts.append((phone, session, reason))
        return True

    monkeypatch.setattr(escalation_mod, "is_business_hours", lambda: True)
    monkeypatch.setattr(escalation_mod, "send_escalation_alert", capture_alert)

    reply, session = await escalation_mod.run_escalation_agent(
        "I need a human",
        {"company": "Acme Pharma", "phone": "+91999"},
        "human_keywords",
        phone="+91999",
    )

    assert session["human_active"] is True
    assert "connecting you with our sales team" in reply.lower()
    assert "Acme Pharma" in reply
    assert "30–60 minutes" in reply
    assert alerts == [("+91999", session, "human_keywords")]


@pytest.mark.asyncio
async def test_run_escalation_agent_off_hours(monkeypatch):
    monkeypatch.setattr(escalation_mod, "is_business_hours", lambda: False)
    monkeypatch.setattr(
        escalation_mod,
        "get_next_business_open_str",
        lambda: "Monday 10:00 AM IST",
    )
    monkeypatch.setattr(
        escalation_mod,
        "send_escalation_alert",
        AsyncMock(return_value=True),
    )

    reply, session = await escalation_mod.run_escalation_agent(
        "help",
        {},
        "low_confidence",
        phone="+91999",
    )

    assert session["human_active"] is True
    assert "offline" in reply.lower()
    assert "Monday 10:00 AM IST" in reply


@pytest.mark.asyncio
async def test_send_leads_alert_uses_leads_env(monkeypatch):
    sent: list[tuple[str, str]] = []

    async def fake_send(phone: str, text: str) -> bool:
        sent.append((phone, text))
        return True

    monkeypatch.delenv("ESCALATION_PHONE_NUMBERS", raising=False)
    monkeypatch.setenv("LEADS_ALERT_PHONE_NUMBERS", "+911111111111,919222222222")
    monkeypatch.setattr(alerts_mod, "send_message", fake_send)

    ok = await alerts_mod.send_leads_alert("escalation test")

    assert ok is True
    assert len(sent) == 2
    assert sent[0][0] == "+911111111111"
    assert sent[1][0] == "919222222222"


@pytest.mark.asyncio
async def test_send_leads_alert_falls_back_to_legacy_escalation_env(monkeypatch):
    sent: list[tuple[str, str]] = []

    async def fake_send(phone: str, text: str) -> bool:
        sent.append((phone, text))
        return True

    monkeypatch.delenv("LEADS_ALERT_PHONE_NUMBERS", raising=False)
    monkeypatch.setenv("ESCALATION_PHONE_NUMBERS", "+91999")
    monkeypatch.setattr(alerts_mod, "send_message", fake_send)

    assert await alerts_mod.send_leads_alert("legacy") is True
    assert sent == [("+91999", "legacy")]


@pytest.mark.asyncio
async def test_send_order_team_alert_uses_order_env_only(monkeypatch):
    sent: list[tuple[str, str]] = []

    async def fake_send(phone: str, text: str) -> bool:
        sent.append((phone, text))
        return True

    monkeypatch.setenv("LEADS_ALERT_PHONE_NUMBERS", "+911111111111")
    monkeypatch.setenv("ORDER_ALERT_PHONE_NUMBERS", "+919333333333")
    monkeypatch.setattr(alerts_mod, "send_message", fake_send)

    ok = await alerts_mod.send_order_team_alert("order test")

    assert ok is True
    assert len(sent) == 1
    assert sent[0][0] == "+919333333333"


@pytest.mark.asyncio
async def test_send_order_alert_does_not_notify_leads_team(monkeypatch):
    sent: list[tuple[str, str]] = []

    async def fake_send(phone: str, text: str) -> bool:
        sent.append((phone, text))
        return True

    monkeypatch.setenv("LEADS_ALERT_PHONE_NUMBERS", "+911111111111,919222222222")
    monkeypatch.setenv("ORDER_ALERT_PHONE_NUMBERS", "919333333333")
    monkeypatch.setattr(alerts_mod, "send_message", fake_send)

    await alerts_mod.send_order_alert(
        {
            "order_ref": "ORD-1",
            "product_name": "Metformin",
            "quantity": 100,
            "city": "Nairobi",
            "country": "Kenya",
            "contact_name": "Priya",
            "phone": "+91888",
        }
    )

    assert len(sent) == 1
    assert sent[0][0] == "919333333333"
    assert "NEW ORDER" in sent[0][1]


@pytest.mark.asyncio
async def test_send_leads_alert_returns_false_when_no_numbers(monkeypatch):
    monkeypatch.delenv("LEADS_ALERT_PHONE_NUMBERS", raising=False)
    monkeypatch.delenv("ESCALATION_PHONE_NUMBERS", raising=False)
    assert await alerts_mod.send_leads_alert("x") is False


@pytest.mark.asyncio
async def test_send_escalation_alert_format(monkeypatch):
    captured: list[str] = []

    async def capture(text: str) -> bool:
        captured.append(text)
        return True

    monkeypatch.setattr(alerts_mod, "send_leads_alert", capture)

    await alerts_mod.send_escalation_alert(
        "+91888",
        {"company": "MedCo", "country": "UAE", "lead_score": 92},
        "hot_lead",
    )

    assert len(captured) == 1
    assert "ESCALATION ALERT" in captured[0]
    assert "+91888" in captured[0]
    assert "MedCo" in captured[0]
    assert "hot_lead" in captured[0]


@pytest.mark.asyncio
async def test_graph_escalation_agent_node(monkeypatch):
    from app.orchestrator import graph as graph_mod

    async def fake_run(message, session, reason, *, phone=""):
        return "Team is on it", {**session, "human_active": True}

    monkeypatch.setattr(graph_mod, "run_escalation_agent", fake_run)

    out = await graph_mod.escalation_agent_node(
        {
            "phone": "+91999",
            "message": "connect me to someone",
            "message_id": "m1",
            "session": {},
            "intent": "escalate",
            "agent_response": None,
            "guardrail_blocked": False,
            "final_reply": None,
        }
    )

    assert out["agent_response"] == "Team is on it"
    assert out["session"]["human_active"] is True
    assert out["session"]["phone"] == "+91999"
