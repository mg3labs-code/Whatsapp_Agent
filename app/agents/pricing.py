"""Pricing agent.

Uses GPT-4o function calling against a deterministic DB tool for single-product
quotes. Multi-product lists reuse the same bulk parsers + catalog resolve as the
order agent (DB prices only). Unmatched names get suggestions; repeated misses
escalate like FAQ.
"""

from __future__ import annotations

import json
import logging
import os

from langfuse import observe
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.business.restricted_products import match_restricted_term
from app.agents.escalation import SESSION_ESCALATION_BUYER_REPLY
from app.db.models import Product
from app.messages.onboarding import looks_like_bulk_order, parse_bulk_order_lines
from app.utils.tracing import get_async_openai_client, set_span_io

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS_PER_TURN = 3
PRICING_MISS_ESCALATE_AFTER = 2
CONTINUE_PRICING = "pricing"

PRICING_SYSTEM_PROMPT = (
    "You are a pharmaceutical export pricing specialist for New Life Medicare.\n"
    "Your job: understand free-text pricing questions, extract a clean catalog search "
    "string, look up the price with the database tool, and return a professional quote.\n"
    "Rules:\n"
    "- Always call the DB tool before quoting any price. Never guess or approximate prices.\n"
    "- Extract a short search query: trade/brand name or salt/generic (and strength if given). "
    "Drop filler like 'price of', 'how much', 'pls', 'quote for'.\n"
    "- If the first lookup fails, try once more with an alternate form "
    "(salt vs brand, or a shorter token). Do not invent a third guess.\n"
    "- If destination country is not in session context, ask the buyer for their country "
    "before quoting.\n"
    "- Default quote is *USD price per strip* from the catalog. Quantity is NOT required.\n"
    "- Only multiply price × quantity when the buyer asks for a total or gives a strip count.\n"
    "- Never invent or mention minimum order quantities.\n"
    "- Use *asterisks* for bold (WhatsApp format), not markdown **double asterisks**.\n"
    "- If product is restricted, say it's not available for export via this channel.\n"
    "- If product_not_found: share any suggestions from the tool ('Did you mean…'), ask the "
    "buyer to confirm the trade name / salt / SKU — never invent a price or product."
)

GET_PRODUCT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_product_by_name",
        "description": (
            "Look up a pharmaceutical product in the New Life Medicare catalog. "
            "Uses the same resolve path as ordering (full query, then token fallback). "
            "Pass a short trade name, salt/generic, manufacturer fragment, or SKU "
            "(PROD-123) — not the full buyer sentence. "
            "Returns USD price per strip, or not_found with suggestions, or restricted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Short catalog search string: product trade name, salt/generic, "
                        "manufacturer fragment, or SKU."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}


def lookup_product_ilike(query: str, db: Session) -> dict:
    """Core ILIKE match on trade name, salt, or manufacturer (no token fallback).

    Used by the order agent's resolve path. Pricing's public tool wraps resolve.
    """
    restricted_hit = match_restricted_term(query, db)
    if restricted_hit:
        return {
            "error": "product_restricted",
            "name": restricted_hit.term,
            "schedule_category": restricted_hit.schedule_category,
        }

    pattern = f"%{query}%"
    product = (
        db.query(Product)
        .filter(
            or_(
                Product.product_name.ilike(pattern),
                Product.salt_name.ilike(pattern),
                Product.manufacturing_company.ilike(pattern),
            )
        )
        .first()
    )

    if product is None:
        return {"error": "product_not_found", "query": query}

    if product.is_restricted:
        return {
            "error": "product_restricted",
            "name": product.product_name,
            "schedule_category": product.schedule_category,
        }

    return {
        "product_name": product.product_name,
        "salt_name": product.salt_name or "",
        "manufacturing_company": product.manufacturing_company or "",
        "expiry_date": product.expiry_date.isoformat() if product.expiry_date else None,
        "price_per_strip": float(product.price_per_strip),
        "is_restricted": product.is_restricted,
        "schedule_category": product.schedule_category,
    }


def get_product_by_name(query: str, db: Session) -> dict:
    """Pricing catalog tool: order-style resolve + suggestions on miss.

    Returns one of:
      - product dict (success) with optional match_mode
      - {"error": "product_not_found", "query", "suggestions"}
      - {"error": "product_restricted", "name", "schedule_category"}
    """
    # Lazy import: order imports lookup_product_ilike from this module.
    from app.agents.order import _resolve_product_match, _suggest_products

    text = (query or "").strip()
    if not text:
        return {"error": "product_not_found", "query": query, "suggestions": []}

    product, error, match_mode = _resolve_product_match(text, db)
    if error == "restricted":
        restricted_hit = match_restricted_term(text, db)
        if restricted_hit:
            return {
                "error": "product_restricted",
                "name": restricted_hit.term,
                "schedule_category": restricted_hit.schedule_category,
            }
        recovered = lookup_product_ilike(text, db)
        if recovered.get("error") == "product_restricted":
            return recovered
        return {
            "error": "product_restricted",
            "name": text,
            "schedule_category": None,
        }
    if product is None:
        return {
            "error": "product_not_found",
            "query": text,
            "suggestions": _suggest_products(text, db),
        }

    return {
        "product_name": product.product_name,
        "salt_name": product.salt_name or "",
        "manufacturing_company": product.manufacturing_company or "",
        "expiry_date": product.expiry_date.isoformat() if product.expiry_date else None,
        "price_per_strip": float(product.price_per_strip),
        "is_restricted": product.is_restricted,
        "schedule_category": product.schedule_category,
        "match_mode": match_mode,
    }


def _format_miss_reply(query: str, suggestions: list[str]) -> str:
    q = (query or "that product").strip() or "that product"
    parts = [f"I couldn't find *{q}* in our export catalog."]
    if suggestions:
        parts.append("\nDid you mean:\n• " + "\n• ".join(suggestions[:5]))
    parts.append(
        "\nPlease send the product trade name, salt/generic name, or SKU (e.g. PROD-123)."
    )
    return "".join(parts)


def _record_pricing_miss(
    session: dict, reply_for_first_miss: str
) -> tuple[str, str]:
    """Increment miss counter; escalate after consecutive catalog misses."""
    miss_count = int(session.get("pricing_miss_count", 0) or 0) + 1
    session["pricing_miss_count"] = miss_count
    if miss_count >= PRICING_MISS_ESCALATE_AFTER:
        session["pricing_miss_count"] = 0
        session["escalation_reason"] = "pricing_no_match_repeated"
        return "", "escalate"
    return reply_for_first_miss, CONTINUE_PRICING


def _clear_pricing_miss(session: dict) -> None:
    session["pricing_miss_count"] = 0


def _should_use_multi_product_quote(message: str) -> bool:
    """True for comma/newline product lists (same detection as order bulk)."""
    if not looks_like_bulk_order(message):
        return False
    lines = [item for item in parse_bulk_order_lines(message) if (item[0] or "").strip()]
    # Two or more products → always multi-quote. Single "Name - qty" also OK
    # (deterministic DB quote; avoids LLM for clear list-style lines).
    return len(lines) >= 1 and (
        len(lines) >= 2
        or "\n" in (message or "")
        or "," in (message or "")
        or ";" in (message or "")
        or (lines[0][1] is not None)
    )


def format_multi_product_quote(
    message: str,
    session: dict | None,
    db: Session,
) -> tuple[str | None, bool]:
    """Deterministic multi-line quote from catalog (no LLM).

    Returns (reply, full_miss). reply is None when not a bulk/list request.
    full_miss is True when every line failed to match (no quotes, only missing).
    Prices come only from DB via the same resolve path as the order agent.
    """
    if not _should_use_multi_product_quote(message):
        return None, False

    # Lazy import avoids circular import (order imports lookup_product_ilike).
    from app.agents.order import _resolve_product_match, _suggest_products

    lines = parse_bulk_order_lines(message)
    if not lines:
        return None, False

    country = ((session or {}).get("country") or "").strip()
    quoted: list[str] = []
    restricted: list[str] = []
    missing: list[str] = []
    seen_names: set[str] = set()

    for query, qty in lines:
        query = (query or "").strip()
        if not query:
            continue
        product, error, _mode = _resolve_product_match(query, db)
        if error == "restricted":
            name = (product.product_name if product else query).strip()
            restricted.append(name)
            continue
        if product is None:
            missing.append(query)
            continue

        name = product.product_name
        # Deduplicate exact catalog names in one message
        dedupe_key = name.lower()
        if dedupe_key in seen_names and qty is None:
            continue
        seen_names.add(dedupe_key)

        unit = float(product.price_per_strip or 0.0)
        line = f"• *{name}* — *${unit:.2f}* USD per strip"
        if qty is not None and qty >= 1:
            total = round(unit * qty, 2)
            line += f"\n  Qty *{qty}* → *${total:.2f}* USD"
        quoted.append(line)

    if not quoted and not restricted and not missing:
        return None, False

    parts: list[str] = ["💰 *Price quote* (catalog USD per strip)"]
    if country:
        parts.append(f"Destination: *{country}*")
    parts.append("")

    if quoted:
        parts.extend(quoted)
    if restricted:
        parts.append("")
        parts.append(
            "Not available for export via this channel:\n• " + "\n• ".join(restricted)
        )
    if missing:
        parts.append("")
        parts.append("Couldn't match:\n• " + "\n• ".join(missing))
        suggestions: list[str] = []
        for q in missing:
            suggestions.extend(_suggest_products(q, db))
        unique = list(dict.fromkeys(suggestions))[:5]
        if unique:
            parts.append("\nDid you mean:\n• " + "\n• ".join(unique))
        else:
            parts.append("Please check the product name/SKU and try again.")

    if quoted:
        parts.append("")
        parts.append(
            "Tap *Place an Order* when ready, or send another product name for a quote."
        )

    full_miss = bool(missing) and not quoted
    return "\n".join(parts).strip(), full_miss


@observe(name="pricing_agent", capture_input=False)
async def run_pricing_agent(
    message: str, session: dict, db: Session
) -> tuple[str, dict, str]:
    """Run the pricing agent on one buyer message.

    Multi-product lists → deterministic DB quotes (same parsers as order).
    Single-product free text → GPT-4o + get_product_by_name tool.

    Returns (reply_text, updated_session, next_intent). next_intent is
    ``pricing`` or ``escalate`` (after consecutive catalog misses).
    Never raises — falls back to a safe message on any error.
    """
    session = dict(session or {})
    country = session.get("country") or "(not provided)"
    # SECURITY: Langfuse input — no full message body
    set_span_io(
        input_data={
            "message_len": len(message or ""),
            "country": country,
        }
    )

    # Phase 3: list / multi-product quotes — no LLM required.
    try:
        multi, full_miss = format_multi_product_quote(message, session, db)
        if multi:
            if full_miss:
                reply, next_intent = _record_pricing_miss(session, multi)
                set_span_io(
                    output_data={
                        "reply_len": len(reply),
                        "agent": "pricing",
                        "mode": "multi_product_miss",
                        "next_intent": next_intent,
                    }
                )
                return reply, session, next_intent
            _clear_pricing_miss(session)
            set_span_io(
                output_data={
                    "reply_len": len(multi),
                    "agent": "pricing",
                    "mode": "multi_product",
                }
            )
            return multi, session, CONTINUE_PRICING
    except Exception:
        logger.exception("Multi-product pricing quote failed; falling back to LLM path")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set; pricing agent cannot run")
        session["escalation_reason"] = "pricing_outage"
        session[SESSION_ESCALATION_BUYER_REPLY] = (
            "I'm having trouble checking pricing right now. "
            "I've notified our team — they'll follow up with you."
        )
        set_span_io(
            output_data={
                "reply_len": 0,
                "agent": "pricing",
                "mode": "outage",
                "next_intent": "escalate",
            }
        )
        return "", session, "escalate"

    client = get_async_openai_client(api_key=api_key)

    messages = [
        {"role": "system", "content": PRICING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Buyer context:\n"
                f"- Country: {country}\n\n"
                f"Buyer message: {message}"
            ),
        },
    ]

    try:
        saw_tool = False
        had_hit = False
        last_miss_query = ""
        last_suggestions: list[str] = []

        for _ in range(MAX_TOOL_CALLS_PER_TURN):
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=[GET_PRODUCT_TOOL],
                tool_choice="auto",
            )
            assistant_msg = response.choices[0].message

            if not assistant_msg.tool_calls:
                if saw_tool and not had_hit:
                    miss_reply = _format_miss_reply(last_miss_query, last_suggestions)
                    reply, next_intent = _record_pricing_miss(session, miss_reply)
                    set_span_io(
                        output_data={
                            "reply_len": len(reply),
                            "agent": "pricing",
                            "mode": "llm_miss",
                            "next_intent": next_intent,
                        }
                    )
                    return reply, session, next_intent
                if had_hit:
                    _clear_pricing_miss(session)
                reply = assistant_msg.content or ""
                set_span_io(
                    output_data={
                        "reply_len": len(reply),
                        "agent": "pricing",
                        "mode": "llm",
                    }
                )
                return reply, session, CONTINUE_PRICING

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_msg.tool_calls
                    ],
                }
            )

            for tool_call in assistant_msg.tool_calls:
                tool_result = _execute_tool_call(tool_call, db)
                saw_tool = True
                err = tool_result.get("error")
                if err == "product_not_found":
                    last_miss_query = str(tool_result.get("query") or last_miss_query)
                    last_suggestions = list(tool_result.get("suggestions") or [])
                elif err == "product_restricted":
                    # Catalog hit (restricted) — not a name miss.
                    had_hit = True
                    _clear_pricing_miss(session)
                elif "product_name" in tool_result and "price_per_strip" in tool_result:
                    had_hit = True
                    _clear_pricing_miss(session)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result),
                    }
                )

        if saw_tool and not had_hit:
            miss_reply = _format_miss_reply(last_miss_query, last_suggestions)
            reply, next_intent = _record_pricing_miss(session, miss_reply)
            set_span_io(
                output_data={
                    "reply_len": len(reply),
                    "agent": "pricing",
                    "mode": "llm_miss",
                    "next_intent": next_intent,
                }
            )
            return reply, session, next_intent

        final = await client.chat.completions.create(model="gpt-4o", messages=messages)
        if had_hit:
            _clear_pricing_miss(session)
        reply = final.choices[0].message.content or ""
        set_span_io(
            output_data={"reply_len": len(reply), "agent": "pricing", "mode": "llm"}
        )
        return reply, session, CONTINUE_PRICING
    except Exception:
        # SECURITY: log agent name only — not message content
        logger.exception("Pricing agent failed")
        session["escalation_reason"] = "pricing_error"
        session[SESSION_ESCALATION_BUYER_REPLY] = (
            "I'm having trouble checking pricing right now. "
            "I've notified our team — they'll follow up with you."
        )
        return "", session, "escalate"


def _execute_tool_call(tool_call, db: Session) -> dict:
    """Dispatch a single LLM tool call to its DB function."""
    name = tool_call.function.name
    if name != "get_product_by_name":
        return {"error": "unknown_tool", "name": name}

    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {"error": "invalid_tool_arguments"}

    query = args.get("query", "")
    if not query:
        return {"error": "missing_query"}

    return get_product_by_name(query, db)
