"""LangGraph orchestrator for WASA.

Wires up the full message pipeline per .cursorrules:
    load_session
      -> greeting (one-time welcome + quick-reply buttons)
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

from datetime import datetime, timezone
import logging
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.faq import run_faq_agent
from app.agents.lead_scoring import enrich_session_from_message
from app.agents.order import (
    PAYMENT_BUTTON_IDS,
    SELECT_PAYMENT,
    is_order_tracking_message,
    run_order_agent,
)
from app.agents.pricing import run_pricing_agent
from app.agents.escalation import run_escalation_agent
from app.agents.qualification import run_qualification_agent
from app.agents.router import classify_intent
from app.guardrails.check import (
    check_post_guardrails,
    check_pre_guardrails,
    log_guardrail,
)
from app.db.models import Conversation
from app.integrations.whatsapp import (
    send_interactive_buttons,
    send_message,
    send_navigation_footer,
)
from app.messages.conversation_ui import (
    SESSION_SUPPRESS_NAV_FOOTER,
    apply_menu_selection_ack,
    is_main_menu_request,
    should_send_navigation_footer,
)
from app.session.lead_hydration import hydrate_session_from_db
from app.session.manager import get_session, save_session
from app.utils.security import user_ref

logger = logging.getLogger(__name__)


def _get_db_generator():
    """Return the FastAPI-style DB generator. Patch this in tests to avoid DATABASE_URL."""
    from app.db.database import get_db

    return get_db()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summarize_user_turn(message: str, session: dict) -> str:
    qual = session.get("qual_state")
    if qual:
        return f"Qualification step ({qual})"
    if session.get("order_state"):
        return f"Order step ({session.get('order_state')})"
    intent = session.get("pending_intent") or ""
    if intent:
        return f"User message (intent={intent})"
    return "User message"


def _summarize_bot_turn(session: dict) -> str:
    agent = session.get("last_agent", "unknown")
    qual = session.get("qual_state")
    if qual:
        return f"Qualification prompt ({qual})"
    if session.get("order_state"):
        return f"Order flow ({session.get('order_state')})"
    return f"{agent} agent response"


def _persist_conversation(phone: str, user_msg: str, bot_reply: str, session: dict) -> None:
    """Upsert conversation row with privacy-safe summaries (no full message bodies)."""
    gen = _get_db_generator()
    db = next(gen)
    phone_ref = user_ref(phone)
    try:
        conv = (
            db.query(Conversation)
            .filter(Conversation.phone_number == phone_ref)
            .order_by(Conversation.created_at.desc())
            .first()
        )
        if not conv:
            conv = Conversation(
                phone_number=phone_ref,
                session_id=phone_ref,
            )
            db.add(conv)

        messages = list(conv.messages or [])
        messages.append(
            {
                "role": "user",
                "summary": _summarize_user_turn(user_msg, session),
                "agent_role": "user",
                "ts": _now_iso(),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "summary": _summarize_bot_turn(session),
                "agent_role": session.get("last_agent", "unknown"),
                "ts": _now_iso(),
            }
        )
        # Keep last 100 summary entries per user
        conv.messages = messages[-100:]
        conv.current_agent = session.get("last_agent", "unknown")
        conv.lead_score = int(session.get("lead_score") or 0)
        conv.conversation_state = "active"
        if session.get("lead_qualified"):
            conv.conversation_state = "qualified"
        conv.updated_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        gen.close()


class MessageState(TypedDict):
    phone: str
    message: str
    message_id: str
    session: dict
    intent: Optional[str]
    agent_response: Optional[str]
    guardrail_blocked: bool
    final_reply: Optional[str]
    greeting: bool


WELCOME_MESSAGE = (
    "Hi! 👋 I'm the AI assistant for *New Life Medicare*\n"
    "I help with medicine orders, pricing & FAQs.\n"
    "Reply *human* anytime to reach our team."
)
WELCOME_BUTTONS = [
    {"id": "order", "title": "Order Medicines"},
    {"id": "pricing", "title": "Get Pricing"},
    {"id": "faq", "title": "Browse FAQs"},
]


async def load_session_node(state: MessageState) -> dict:
    from app.session.manager import normalize_phone

    phone = normalize_phone(state["phone"])
    session = await get_session(phone)
    if not session.get("lead_qualified"):
        gen = _get_db_generator()
        db = next(gen)
        try:
            session = hydrate_session_from_db(phone, session, db)
        finally:
            gen.close()
    session = enrich_session_from_message(session, state.get("message") or "")
    session["phone"] = phone
    return {"session": session, "phone": phone}


async def greeting_node(state: MessageState) -> dict:
    """Set one-time greeting flag for send_reply to prepend welcome message."""
    session = dict(state.get("session") or {})
    if state.get("phone") and not session.get("phone"):
        session["phone"] = state["phone"]
    if not session.get("greeted"):
        session["greeted"] = True
        return {"session": session, "greeting": True}
    return {"session": session, "greeting": False}


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


async def menu_refresh_node(state: MessageState) -> dict:
    """Resend main menu when buyer taps Main Menu (no agent logic change)."""
    session = dict(state.get("session") or {})
    await send_interactive_buttons(
        state["phone"],
        "How can I help you today? Select an option below 👇",
        WELCOME_BUTTONS,
    )
    session[SESSION_SUPPRESS_NAV_FOOTER] = True
    return {"session": session, "final_reply": None}


async def router_node(state: MessageState) -> dict:
    session = dict(state.get("session") or {})
    msg_key = (state.get("message") or "").strip().lower()

    if is_main_menu_request(state.get("message") or ""):
        return {"intent": "menu_refresh", "session": session}

    if is_order_tracking_message(state.get("message") or ""):
        return {"intent": "order", "session": session}

    if msg_key in PAYMENT_BUTTON_IDS or msg_key in {
        "bank transfer",
        "debit / credit card",
    } or session.get("order_state") == SELECT_PAYMENT:
        return {"intent": "order", "session": session}

    if session.get("order_state"):
        return {"intent": "order", "session": session}

    if session.get("qual_state"):
        return {"intent": "qualify", "session": session}

    intent, session = await classify_intent(
        state.get("message") or "",
        session,
    )
    return {"intent": intent, "session": session}


def _merge_prior_reply(state: MessageState, new_reply: str) -> str:
    prior = (state.get("agent_response") or "").strip()
    if not prior:
        return new_reply
    return f"{prior}\n\n{new_reply}"


async def pricing_agent_node(state: MessageState) -> dict:
    session = dict(state.get("session") or {})
    gen = _get_db_generator()
    db = next(gen)
    if state.get("phone") and not session.get("phone"):
        session["phone"] = state["phone"]
    session["last_agent"] = "pricing"
    try:
        reply = await run_pricing_agent(
            state["message"],
            session,
            db,
        )
        return {"agent_response": _merge_prior_reply(state, reply), "session": session}
    finally:
        gen.close()


async def faq_agent_node(state: MessageState) -> dict:
    session = dict(state.get("session") or {})
    if state.get("phone") and not session.get("phone"):
        session["phone"] = state["phone"]
    session["last_agent"] = "faq"
    reply = await run_faq_agent(
        state["message"],
        phone=state.get("phone") or "",
        session=session,
    )
    return {"agent_response": _merge_prior_reply(state, reply), "session": session}


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
        updated_session["last_agent"] = "order"
        return {
            "agent_response": _merge_prior_reply(state, reply),
            "session": updated_session,
        }
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
        updated_session["last_agent"] = "qualifier"
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
    updated_session["last_agent"] = "escalation"
    return {
        "agent_response": _merge_prior_reply(state, reply),
        "session": updated_session,
    }


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
    from app.session.manager import normalize_phone

    phone = normalize_phone(state["phone"])
    final_reply = state.get("final_reply")
    session = dict(state.get("session") or {})
    session["phone"] = phone

    if final_reply:
        final_reply, session = apply_menu_selection_ack(final_reply, session)
        if state.get("greeting"):
            final_reply = f"{WELCOME_MESSAGE}\n\n{final_reply}"
        try:
            await send_message(phone, final_reply)
        except Exception:
            logger.exception("send_message failed user_ref=%s", user_ref(phone))
        if state.get("greeting"):
            try:
                await send_interactive_buttons(phone, "Choose an option below:", WELCOME_BUTTONS)
            except Exception:
                logger.exception(
                    "send_interactive_buttons failed user_ref=%s",
                    user_ref(phone),
                )
        try:
            session["last_agent"] = str(state.get("intent") or session.get("last_agent") or "unknown")
            _persist_conversation(
                phone=phone,
                user_msg=state.get("message") or "",
                bot_reply=final_reply,
                session=session,
            )
        except Exception:
            logger.exception("persist_conversation failed user_ref=%s", user_ref(phone))

    if should_send_navigation_footer(session):
        try:
            await send_navigation_footer(phone)
        except Exception:
            logger.exception("send_navigation_footer failed user_ref=%s", user_ref(phone))

    session.pop(SESSION_SUPPRESS_NAV_FOOTER, None)

    try:
        await save_session(phone, session)
    except Exception:
        logger.exception("save_session failed user_ref=%s", user_ref(phone))

    return {"session": session}


def _route_after_pre_guardrails(state: MessageState) -> str:
    if state.get("guardrail_blocked"):
        return "send_reply"
    return "router"


def _after_qualify(state: MessageState) -> str:
    intent = state.get("intent", "")
    if intent == "continue_qual" or not intent:
        return "send_reply"
    if intent in {"order", "pricing", "faq", "escalate"}:
        return intent
    return "send_reply"


def _route_after_qualify(state: MessageState) -> str:
    """Backward-compatible alias for existing tests/imports."""
    return _after_qualify(state)


def _route_to_agent(state: MessageState) -> str:
    if state.get("guardrail_blocked"):
        return "send_reply"

    session = state.get("session") or {}
    if session.get("human_active"):
        return "human_active"

    intent = state.get("intent")
    if intent in {"pricing", "faq", "order", "qualify", "escalate", "menu_refresh"}:
        return intent
    return "faq"


def _build_graph():
    graph = StateGraph(MessageState)

    graph.add_node("load_session", load_session_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("pre_guardrails", pre_guardrails_node)
    graph.add_node("router", router_node)
    graph.add_node("menu_refresh", menu_refresh_node)
    graph.add_node("pricing_agent", pricing_agent_node)
    graph.add_node("faq_agent", faq_agent_node)
    graph.add_node("order_agent", order_agent_node)
    graph.add_node("qualify_agent", qualify_agent_node)
    graph.add_node("escalation_agent", escalation_agent_node)
    graph.add_node("human_active", human_active_node)
    graph.add_node("post_guardrails", post_guardrails_node)
    graph.add_node("send_reply", send_reply_node)

    graph.set_entry_point("load_session")
    graph.add_edge("load_session", "greeting")
    graph.add_edge("greeting", "pre_guardrails")

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
            "menu_refresh": "menu_refresh",
        },
    )

    graph.add_edge("menu_refresh", "send_reply")

    graph.add_conditional_edges(
        "qualify_agent",
        _after_qualify,
        {
            "send_reply": "post_guardrails",
            "order": "order_agent",
            "pricing": "pricing_agent",
            "faq": "faq_agent",
            "escalate": "escalation_agent",
        },
    )

    for agent_node in ("pricing_agent", "faq_agent", "order_agent", "escalation_agent"):
        graph.add_edge(agent_node, "post_guardrails")

    graph.add_edge("post_guardrails", "send_reply")
    graph.add_edge("send_reply", END)
    graph.add_edge("human_active", END)

    return graph.compile()


compiled_graph = _build_graph()
