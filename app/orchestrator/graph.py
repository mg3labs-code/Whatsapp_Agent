"""LangGraph orchestrator for WASA.

Wires up the full message pipeline per .cursorrules:
    load_session
      -> pre_guardrails
        -> (blocked) send_reply -> END
        -> (ok) router
             -> (human_active) human_active -> END
             -> (intent) <agent> -> post_guardrails -> send_reply -> END

Pricing, FAQ, order, and qualification agents are wired. Pre/post guardrails
block before agents (pre) and sanitize replies (post). Intent routing via
app/agents/router.classify_intent. Escalation via app/agents/escalation; team alerts via WhatsApp (LEADS_ALERT / ORDER_ALERT).
"""

from __future__ import annotations

import logging
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.faq import run_faq_agent
from app.agents.lead_scoring import enrich_session_from_message
from app.agents.order import run_order_agent
from app.agents.pricing import run_pricing_agent
from app.agents.escalation import run_escalation_agent
from app.agents.qualification import run_qualification_agent
from app.agents.router import classify_intent
from app.guardrails.check import (
    check_post_guardrails,
    check_pre_guardrails,
    log_guardrail,
)
from app.integrations.whatsapp import send_message
from app.session.manager import get_session, save_session
from app.utils.security import user_ref

logger = logging.getLogger(__name__)


def _get_db_generator():
    """Return the FastAPI-style DB generator. Patch this in tests to avoid DATABASE_URL."""
    from app.db.database import get_db

    return get_db()


class MessageState(TypedDict):
    phone: str
    message: str
    message_id: str
    session: dict
    intent: Optional[str]
    agent_response: Optional[str]
    guardrail_blocked: bool
    final_reply: Optional[str]


async def load_session_node(state: MessageState) -> dict:
    session = await get_session(state["phone"])
    session = enrich_session_from_message(session, state.get("message") or "")
    session["phone"] = state["phone"]
    return {"session": session}


async def pre_guardrails_node(state: MessageState) -> dict:
    result = check_pre_guardrails(
        state.get("message") or "",
        state.get("session") or {},
    )
    if result.blocked:
        await log_guardrail(
            state["phone"],
            result.reason,
            "pre",
            state.get("message") or "",
        )
        return {
            "guardrail_blocked": True,
            "final_reply": result.refusal_message,
        }
    return {"guardrail_blocked": False}


async def router_node(state: MessageState) -> dict:
    session = dict(state.get("session") or {})

    if session.get("order_state"):
        return {"intent": "order", "session": session}

    if session.get("qual_state"):
        return {"intent": "qualify", "session": session}

    intent, session = await classify_intent(
        state.get("message") or "",
        session,
    )
    return {"intent": intent, "session": session}


async def pricing_agent_node(state: MessageState) -> dict:
    gen = _get_db_generator()
    db = next(gen)
    session = dict(state.get("session") or {})
    if state.get("phone") and not session.get("phone"):
        session["phone"] = state["phone"]
    try:
        reply = await run_pricing_agent(
            state["message"],
            session,
            db,
        )
        return {"agent_response": reply}
    finally:
        gen.close()


async def faq_agent_node(state: MessageState) -> dict:
    reply = await run_faq_agent(
        state["message"],
        phone=state.get("phone") or "",
    )
    return {"agent_response": reply}


async def order_agent_node(state: MessageState) -> dict:
    gen = _get_db_generator()
    db = next(gen)
    session = dict(state.get("session") or {})
    if state.get("phone") and not session.get("phone"):
        session["phone"] = state["phone"]
    try:
        reply, updated_session = await run_order_agent(
            state["message"],
            session,
            db,
        )
        return {"agent_response": reply, "session": updated_session}
    finally:
        gen.close()


async def qualify_agent_node(state: MessageState) -> dict:
    gen = _get_db_generator()
    db = next(gen)
    session = dict(state.get("session") or {})
    if state.get("phone") and not session.get("phone"):
        session["phone"] = state["phone"]
    try:
        reply, updated_session, next_intent = await run_qualification_agent(
            state["message"],
            session,
            db,
        )
        result: dict = {"agent_response": reply, "session": updated_session}
        if next_intent != "continue_qual":
            result["intent"] = next_intent
        return result
    finally:
        gen.close()


def _escalation_reason(state: MessageState) -> str:
    session = state.get("session") or {}
    if session.get("escalation_reason"):
        return str(session["escalation_reason"])
    if session.get("lead_category") == "HOT":
        return "hot_lead"
    return "buyer_request"


async def escalation_agent_node(state: MessageState) -> dict:
    session = dict(state.get("session") or {})
    if state.get("phone") and not session.get("phone"):
        session["phone"] = state["phone"]
    reply, updated_session = await run_escalation_agent(
        state.get("message") or "",
        session,
        _escalation_reason(state),
        phone=state.get("phone") or "",
    )
    return {"agent_response": reply, "session": updated_session}


async def human_active_node(state: MessageState) -> dict:
    # No-op: when a human has taken over the conversation we silently drop.
    # SECURITY: hashed user ref in logs — not raw phone
    logger.info("human_active drop user_ref=%s", user_ref(state.get("phone")))
    return {}


async def post_guardrails_node(state: MessageState) -> dict:
    agent_response = state.get("agent_response")
    if not agent_response:
        return {}

    result = check_post_guardrails(agent_response)
    if result.blocked:
        await log_guardrail(
            state["phone"],
            result.reason,
            "post",
            agent_response,
        )
        return {"final_reply": result.refusal_message}

    return {"final_reply": agent_response}


async def send_reply_node(state: MessageState) -> dict:
    phone = state["phone"]
    final_reply = state.get("final_reply")

    if final_reply:
        try:
            await send_message(phone, final_reply)
        except Exception:
            logger.exception("send_message failed user_ref=%s", user_ref(phone))

    try:
        await save_session(phone, state.get("session") or {})
    except Exception:
        logger.exception("save_session failed user_ref=%s", user_ref(phone))

    return {}


def _route_after_pre_guardrails(state: MessageState) -> str:
    if state.get("guardrail_blocked"):
        return "send_reply"
    return "router"


def _route_to_agent(state: MessageState) -> str:
    if state.get("guardrail_blocked"):
        return "send_reply"

    session = state.get("session") or {}
    if session.get("human_active"):
        return "human_active"

    intent = state.get("intent")
    if intent in {"pricing", "faq", "order", "qualify", "escalate"}:
        return intent
    return "faq"


def _build_graph():
    graph = StateGraph(MessageState)

    graph.add_node("load_session", load_session_node)
    graph.add_node("pre_guardrails", pre_guardrails_node)
    graph.add_node("router", router_node)
    graph.add_node("pricing_agent", pricing_agent_node)
    graph.add_node("faq_agent", faq_agent_node)
    graph.add_node("order_agent", order_agent_node)
    graph.add_node("qualify_agent", qualify_agent_node)
    graph.add_node("escalation_agent", escalation_agent_node)
    graph.add_node("human_active", human_active_node)
    graph.add_node("post_guardrails", post_guardrails_node)
    graph.add_node("send_reply", send_reply_node)

    graph.set_entry_point("load_session")
    graph.add_edge("load_session", "pre_guardrails")

    graph.add_conditional_edges(
        "pre_guardrails",
        _route_after_pre_guardrails,
        {"send_reply": "send_reply", "router": "router"},
    )

    graph.add_conditional_edges(
        "router",
        _route_to_agent,
        {
            "send_reply": "send_reply",
            "human_active": "human_active",
            "pricing": "pricing_agent",
            "faq": "faq_agent",
            "order": "order_agent",
            "qualify": "qualify_agent",
            "escalate": "escalation_agent",
        },
    )

    for agent_node in (
        "pricing_agent",
        "faq_agent",
        "order_agent",
        "qualify_agent",
        "escalation_agent",
    ):
        graph.add_edge(agent_node, "post_guardrails")

    graph.add_edge("post_guardrails", "send_reply")
    graph.add_edge("send_reply", END)
    graph.add_edge("human_active", END)

    return graph.compile()


compiled_graph = _build_graph()
