"""Pricing agent.

Uses GPT-4o function calling against a single deterministic DB tool
(`get_product_by_name`). The LLM is never trusted with prices — it can
only quote what the DB returns (USD per strip from catalog).
"""

from __future__ import annotations

import json
import logging
import os

from langfuse import observe
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.models import Product
from app.utils.tracing import get_async_openai_client, set_span_io

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS_PER_TURN = 3

PRICING_SYSTEM_PROMPT = (
    "You are a pharmaceutical export pricing specialist for New Life Medicare.\n"
    "Your job: extract product name and quantity from the buyer's message, look up the price using\n"
    "the database tool, and return a professional formatted price quote.\n"
    "Rules:\n"
    "- Always call the DB tool before quoting any price. Never guess prices.\n"
    "- If company or country not in session context, ask for them before quoting.\n"
    "- The tool returns a single *USD price per strip* from the catalog. Quote that exact value.\n"
    "- If the buyer asks for totals, multiply *USD price per strip* by their quantity (use clear math).\n"
    "- Use *asterisks* for bold (WhatsApp format), not markdown **double asterisks**.\n"
    "- If product is restricted, say it's not available for export via this channel.\n"
    "- If product not found, say you'll check and confirm — never invent a price."
)

GET_PRODUCT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_product_by_name",
        "description": (
            "Look up a pharmaceutical product in the New Life Medicare catalog. "
            "Matches product trade name, salt/generic name, or manufacturing company (partial, case-insensitive). "
            "Returns USD price per strip and restriction flag."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Product name, generic/salt name, or manufacturer fragment to search.",
                }
            },
            "required": ["query"],
        },
    },
}


def get_product_by_name(query: str, db: Session) -> dict:
    """Fuzzy-lookup a product by trade name, salt/generic, or manufacturer.

    Returns one of:
      - product dict (success)
      - {"error": "product_not_found", "query": ...}
      - {"error": "product_restricted", "name": ...}
    """
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


@observe(name="pricing_agent", capture_input=False)
async def run_pricing_agent(message: str, session: dict, db: Session) -> str:
    """Run the pricing agent on one buyer message.

    Returns the final assistant reply (WhatsApp-formatted string).
    Never raises — falls back to a safe "let me check" message on any error.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set; pricing agent cannot run")
        return (
            "I'm having trouble checking pricing right now. "
            "Let me confirm with our team and get back to you shortly."
        )

    company = session.get("company") or "(not provided)"
    country = session.get("country") or "(not provided)"
    # SECURITY: Langfuse input — no full message body
    set_span_io(
        input_data={
            "message_len": len(message),
            "company": company,
            "country": country,
        }
    )

    client = get_async_openai_client(api_key=api_key)

    messages = [
        {"role": "system", "content": PRICING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Buyer context:\n"
                f"- Company: {company}\n"
                f"- Country: {country}\n\n"
                f"Buyer message: {message}"
            ),
        },
    ]

    try:
        for _ in range(MAX_TOOL_CALLS_PER_TURN):
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=[GET_PRODUCT_TOOL],
                tool_choice="auto",
            )
            assistant_msg = response.choices[0].message

            if not assistant_msg.tool_calls:
                reply = assistant_msg.content or ""
                set_span_io(output_data={"reply_len": len(reply), "agent": "pricing"})
                return reply

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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result),
                    }
                )

        final = await client.chat.completions.create(model="gpt-4o", messages=messages)
        reply = final.choices[0].message.content or ""
        set_span_io(output_data={"reply_len": len(reply), "agent": "pricing"})
        return reply
    except Exception:
        # SECURITY: log agent name only — not message content
        logger.exception("Pricing agent failed")
        return (
            "I'm having trouble pulling pricing right now. "
            "Let me check with our team and confirm shortly."
        )


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
