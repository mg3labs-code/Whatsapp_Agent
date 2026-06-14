"""Order collection agent — LLM + DB tools with multi-product cart."""

from __future__ import annotations

import json
import logging
import os
import random
import re
from datetime import datetime
from typing import Any

from langfuse import observe
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.agents.pricing import get_product_by_name
from app.session.lead_hydration import phone_lookup_variants
from app.business.countries import (
    SHIPMENT_EXCLUDED_REFUSAL,
    is_shipment_excluded_country,
)
from app.db.models import Order, Product
from app.integrations.alerts import send_order_alert
from app.integrations.cashfree import (
    create_card_checkout,
    create_virtual_account,
    get_card_payment_text,
    get_payment_instructions_text,
)
from app.integrations.indiapost import (
    extract_tracking_number,
    fetch_tracking_bundle,
    is_indiapost_configured,
    lookup_tracking_message,
)
from app.integrations.whatsapp import send_interactive_buttons, send_message
from app.messages.onboarding import (
    BULK_LIST_PROMPT,
    QTY_BUTTON_IDS,
    QTY_BUTTON_MAP,
    checkout_prompt,
    looks_like_bulk_order,
    parse_bulk_order_lines,
    parse_checkout_oneline,
    send_quantity_picker,
)
from app.utils.tracing import get_async_openai_client, set_span_io

logger = logging.getLogger(__name__)

ORDER_MODEL = "gpt-4o-mini"
MAX_TOOL_CALLS_PER_TURN = 8

COLLECT_SKU = "COLLECT_SKU"
COLLECT_SKU_CONFIRM = "COLLECT_SKU_CONFIRM"
COLLECT_QTY = "COLLECT_QTY"
CART_MENU = "CART_MENU"
COLLECT_COUNTRY = "COLLECT_COUNTRY"
COLLECT_CITY = "COLLECT_CITY"
COLLECT_CONTACT = "COLLECT_CONTACT"
COLLECT_CHECKOUT = "COLLECT_CHECKOUT"
CONFIRM_ORDER = "CONFIRM_ORDER"
SELECT_PAYMENT = "SELECT_PAYMENT"
PAYMENT_METHOD = "T/T Advance"
ORDER_COMPLETE = "ORDER_COMPLETE"

PAY_BANK_BUTTON = "pay_bank"
PAY_CARD_BUTTON = "pay_card"
PAYMENT_BUTTON_IDS = frozenset({PAY_BANK_BUTTON, PAY_CARD_BUTTON})

PAYMENT_OPTION_BUTTONS = [
    {"id": PAY_BANK_BUTTON, "title": "Bank Transfer"},
    {"id": PAY_CARD_BUTTON, "title": "Debit / Credit Card"},
]

POST_PAYMENT_BUTTONS = [
    {"id": "new_order", "title": "New Order"},
    {"id": "order_status", "title": "Order Status"},
    {"id": "speak", "title": "Speak to Team"},
]

SANCTIONED_COUNTRY_REFUSAL = SHIPMENT_EXCLUDED_REFUSAL

ORDER_SESSION_KEYS = (
    "order_state",
    "order_cart",
    "order_sku",
    "order_product_name",
    "order_qty",
    "order_country",
    "order_city",
    "order_contact",
    "order_ref",
    "order_pending_sku",
    "order_pending_product_name",
    "order_pending_qty",
    "pending_product",
    "order_bulk_queue",
    "order_qty_custom",
)

ORDER_SYSTEM_PROMPT = (
    "You are the order placement assistant for New Life Medicare B2B WhatsApp exports.\n"
    "Use the provided tools to manage the buyer's cart and checkout. Never invent products or prices.\n"
    "Rules:\n"
    "- Interpret natural language (e.g. 'I need 2000 metformin', 'remove the amoxicillin', "
    "'ship to Nairobi Kenya', 'I'm done adding').\n"
    "- Always call lookup_product before add_to_cart; use the exact catalog name from lookup.\n"
    "- add_to_cart requires a positive integer quantity (parse '2k' as 2000, 'two thousand' as 2000).\n"
    "- update_cart_line / remove_from_cart: use line_number OR product_query.\n"
    "- proceed_to_checkout only when the buyer wants to finish adding products and the cart is non-empty.\n"
    "- At checkout, reuse session.country as order_country when present — do not re-ask country.\n"
    "- Collect shipping contact in one message (name, city, phone) when phase is COLLECT_CHECKOUT.\n"
    "- set_shipping / set_contact: extract from natural phrases.\n"
    f"- Payment is always {PAYMENT_METHOD}; never ask for other payment terms.\n"
    "- confirm_order ONLY when the buyer clearly confirms (yes, confirm, place order) and phase is CONFIRM_ORDER.\n"
    "- If lookup fails, show suggestions from the tool and ask for a clearer product name.\n"
    "- Use *single asterisks* for bold (WhatsApp). Be concise and professional.\n"
    "- Do not commit the order without confirm_order after shipping and contact are collected."
)

ORDER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_product",
            "description": (
                "Search catalog by product trade name, salt/generic, manufacturer fragment, or PROD-#### SKU. "
                "Call this before adding to cart."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short product search string extracted from the buyer message.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Add or merge a product line (same SKU merges quantities).",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_query": {"type": "string"},
                    "quantity": {
                        "type": "integer",
                        "description": "Unit count (positive integer).",
                    },
                },
                "required": ["product_query", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_cart_line",
            "description": "Change quantity on a cart line by line number and/or product name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line_number": {"type": "integer"},
                    "product_query": {"type": "string"},
                    "quantity": {"type": "integer"},
                },
                "required": ["quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_from_cart",
            "description": "Remove a cart line by line number and/or product name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line_number": {"type": "integer"},
                    "product_query": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_cart",
            "description": "Show current cart contents and checkout phase.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "proceed_to_checkout",
            "description": "Buyer finished adding products; move to shipping details.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_shipping",
            "description": "Set destination country and/or city (port of entry).",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                    "city": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_contact",
            "description": "Buyer name and company for the order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact": {"type": "string"},
                },
                "required": ["contact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_order",
            "description": (
                "Place the order after buyer confirmation. Only when review was shown and buyer agreed."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _product_sku(product: Product) -> str:
    return f"PROD-{product.id:04d}"


def _get_cart(session: dict) -> list[dict[str, Any]]:
    cart = session.get("order_cart")
    if isinstance(cart, list):
        return cart
    return []


def _set_cart(session: dict, cart: list[dict[str, Any]]) -> None:
    session["order_cart"] = cart


def _migrate_legacy_single_line_session(session: dict) -> None:
    if _get_cart(session):
        return
    sku = session.get("order_sku")
    if not sku:
        return
    qty = int(session.get("order_qty") or 0)
    if qty <= 0:
        return
    _set_cart(
        session,
        [
            {
                "sku": sku,
                "product_name": session.get("order_product_name") or sku,
                "quantity": qty,
                "qty": qty,
                "unit_price": float(session.get("order_unit_price") or 0.0),
            }
        ],
    )
    for key in ("order_sku", "order_product_name", "order_qty"):
        session.pop(key, None)
    if session.get("order_state") in {COLLECT_SKU, COLLECT_QTY, None}:
        session["order_state"] = CART_MENU


def _clear_order_session(session: dict) -> None:
    for key in ORDER_SESSION_KEYS:
        session.pop(key, None)


def _ensure_order_started(session: dict) -> None:
    if not session.get("order_state"):
        session["order_state"] = COLLECT_SKU


def _set_pending_product(
    session: dict,
    *,
    sku: str,
    product_name: str,
    qty: int | None = None,
) -> None:
    session["order_pending_sku"] = sku
    session["order_pending_product_name"] = product_name
    if qty is None:
        session.pop("order_pending_qty", None)
    else:
        session["order_pending_qty"] = int(qty)
    session["pending_product"] = {
        "sku": sku,
        "name": product_name,
    }


def _clear_pending_product(session: dict) -> None:
    for key in ("order_pending_sku", "order_pending_product_name", "order_pending_qty"):
        session.pop(key, None)
    session.pop("pending_product", None)


def _item_qty(item: dict[str, Any]) -> int:
    return int(item.get("qty", item.get("quantity", 0)) or 0)


def _prefill_order_country(session: dict) -> str | None:
    """Copy qualification country into order checkout — never ask twice."""
    if session.get("order_country"):
        return str(session["order_country"])
    known = (session.get("country") or "").strip()
    if known:
        session["order_country"] = known
        return known
    return None


def _get_bulk_queue(session: dict) -> list[str]:
    queue = session.get("order_bulk_queue")
    return list(queue) if isinstance(queue, list) else []


def _set_bulk_queue(session: dict, queue: list[str]) -> None:
    if queue:
        session["order_bulk_queue"] = queue
    else:
        session.pop("order_bulk_queue", None)


async def _start_quantity_step(
    session: dict,
    product: Product,
    *,
    phone: str,
) -> tuple[str, dict]:
    session["order_sku"] = _product_sku(product)
    session["order_product_name"] = product.product_name
    session["order_unit_price"] = float(product.price_per_strip or 0.0)
    session["order_state"] = COLLECT_QTY
    session.pop("order_qty_custom", None)
    if phone:
        await send_quantity_picker(phone, product.product_name)
    return (
        f"Found: *{product.product_name}*\nSelect quantity from the list, or type a number.",
        session,
    )


async def _process_bulk_order(
    text: str,
    session: dict,
    db: Session,
    phone: str,
) -> tuple[str, dict]:
    lines = parse_bulk_order_lines(text)
    if not lines:
        return BULK_LIST_PROMPT, session

    added: list[str] = []
    failed: list[str] = []
    pending: list[str] = []

    for query, qty in lines:
        product, error, match_mode = _resolve_product_match(query, db)
        if error == "restricted":
            failed.append(query)
            continue
        if product is None:
            failed.append(query)
            continue
        if qty is None or match_mode == "token":
            pending.append(query)
            continue
        unit_price = float(product.price_per_strip or 0.0)
        _add_line_to_cart(session, _product_sku(product), product.product_name, qty, unit_price)
        added.append(f"{product.product_name} × {qty}")

    if pending:
        first_query = pending[0]
        _set_bulk_queue(session, pending[1:])
        product, error, _ = _resolve_product_match(first_query, db)
        if product is not None and error != "restricted":
            return await _start_quantity_step(session, product, phone=phone)

    if added:
        session["order_state"] = CART_MENU
        reply_lines = ["✅ Added to cart:", *[f"• {line}" for line in added]]
        if failed:
            reply_lines.append("\nCouldn't match: " + ", ".join(failed))
        reply_lines.append(f"\n{_format_cart_lines(_get_cart(session))}")
        return "\n".join(reply_lines), session

    suggestions: list[str] = []
    for query, _ in lines[:3]:
        suggestions.extend(_suggest_products(query, db))
    reply = "I couldn't match those products. Please check names or SKUs."
    if suggestions:
        reply += "\n\nDid you mean:\n• " + "\n• ".join(dict.fromkeys(suggestions)[:5])
    return reply, session


async def _continue_bulk_queue(session: dict, db: Session, phone: str) -> tuple[str, dict] | None:
    queue = _get_bulk_queue(session)
    if not queue:
        return None
    next_query = queue[0]
    product, error, _ = _resolve_product_match(next_query, db)
    if product is None or error == "restricted":
        queue.pop(0)
        _set_bulk_queue(session, queue)
        if not queue:
            session["order_state"] = CART_MENU
            return (
                f"Couldn't match *{next_query}*. Skipped.\n\n{_format_cart_lines(_get_cart(session))}",
                session,
            )
        return await _continue_bulk_queue(session, db, phone)
    return await _start_quantity_step(session, product, phone=phone)


def _format_money(amount: float) -> str:
    return f"${amount:,.2f}"


def _cart_total(cart: list[dict[str, Any]]) -> float:
    total = 0.0
    for item in cart:
        qty = _item_qty(item)
        unit_price = float(item.get("unit_price") or 0.0)
        total += qty * unit_price
    return round(total, 2)


def _format_cart_lines(cart: list[dict[str, Any]]) -> str:
    if not cart:
        return "🛒 *Your cart:*\n_Empty cart._"
    lines = ["🛒 *Your cart:*"]
    total = 0.0
    for idx, item in enumerate(cart, start=1):
        name = item.get("product_name") or item.get("sku")
        qty = _item_qty(item)
        unit_price = float(item.get("unit_price") or 0.0)
        line_total = qty * unit_price
        total += line_total
        lines.append(f"{idx}. {name} × {qty} = {_format_money(line_total)}")
    lines.append("─────────────────")
    lines.append(f"Total: {_format_money(total)}")
    lines.append("")
    lines.append("Add more products or type *checkout* to proceed.")
    return "\n".join(lines)


def _format_order_review(session: dict) -> str:
    return (
        "REVIEW:\n"
        f"cart:\n{_format_cart_lines(_get_cart(session))}\n"
        f"ship_to: {session.get('order_city', '')}, {session.get('order_country', '')}\n"
        f"contact: {session.get('order_contact', '')}\n"
        f"payment: {PAYMENT_METHOD}"
    )


def _session_snapshot(session: dict) -> dict[str, Any]:
    return {
        "phase": session.get("order_state", COLLECT_SKU),
        "cart": _get_cart(session),
        "country": session.get("order_country") or session.get("country"),
        "city": session.get("order_city"),
        "contact": session.get("order_contact"),
        "payment": PAYMENT_METHOD,
        "pending_product": session.get("order_product_name"),
        "pending_suggested_product": session.get("order_pending_product_name"),
    }


_SKIP_PRODUCT_TOKENS = frozenset(
    {
        "order",
        "orders",
        "want",
        "need",
        "buy",
        "purchase",
        "place",
        "like",
        "please",
        "units",
        "unit",
        "the",
        "for",
        "and",
    }
)


def _lookup_product_query(query: str, db: Session) -> tuple[Product | None, str | None]:
    text = (query or "").strip()
    if not text:
        return None, "not_found"

    sku_match = re.fullmatch(r"PROD-(\d+)", text, re.IGNORECASE)
    if sku_match:
        product = db.query(Product).filter(Product.id == int(sku_match.group(1))).first()
        if product is None:
            return None, "not_found"
        if product.is_restricted:
            return None, "restricted"
        return product, None

    result = get_product_by_name(text, db)
    if result.get("error") == "product_restricted":
        return None, "restricted"
    if result.get("error") != "product_not_found":
        product = (
            db.query(Product)
            .filter(Product.product_name == result["product_name"])
            .first()
        )
        if product:
            return product, None

    return None, "not_found"


def _product_search_tokens(text: str) -> list[str]:
    raw_tokens = re.split(r"[\s,;]+", text)
    tokens: list[str] = []
    for token in raw_tokens:
        cleaned = token.strip(".,!?()")
        if len(cleaned) < 3:
            continue
        if cleaned.lower() in _SKIP_PRODUCT_TOKENS:
            continue
        tokens.append(cleaned)
    return sorted(set(tokens), key=len, reverse=True)


def _resolve_product_match(
    query: str, db: Session
) -> tuple[Product | None, str | None, str]:
    """Return (product, error, match_mode) where match_mode is direct|token|none."""
    text = (query or "").strip()
    if not text:
        return None, "not_found", "none"

    product, err = _lookup_product_query(text, db)
    if product is not None or err == "restricted":
        return product, err, "direct"

    for token in _product_search_tokens(text):
        product, err = _lookup_product_query(token, db)
        if product is not None:
            return product, err, "token"
        if err == "restricted":
            return None, "restricted", "token"

    return None, "not_found", "none"


def _resolve_product_row(query: str, db: Session) -> tuple[Product | None, str | None]:
    product, error, _ = _resolve_product_match(query, db)
    return product, error


def _suggest_products(query: str, db: Session) -> list[str]:
    terms = [t for t in re.split(r"[\s,;]+", (query or "")) if len(t) >= 3]
    search = terms[0] if terms else (query or "")[:40]
    pattern = f"%{search[:40]}%"
    rows = (
        db.query(Product.product_name)
        .filter(
            Product.is_restricted.is_(False),
            or_(
                Product.product_name.ilike(pattern),
                Product.salt_name.ilike(pattern),
            ),
        )
        .limit(5)
        .all()
    )
    return [name for (name,) in rows]


def _find_cart_line(cart: list[dict], line_number: int | None, product_query: str | None) -> int | None:
    if line_number is not None:
        if 1 <= line_number <= len(cart):
            return line_number - 1
        return None
    if product_query:
        q = product_query.lower()
        for idx, item in enumerate(cart):
            name = (item.get("product_name") or "").lower()
            sku = (item.get("sku") or "").lower()
            if q in name or q in sku or name in q:
                return idx
    return None


def _add_line_to_cart(
    session: dict,
    sku: str,
    product_name: str,
    qty: int,
    unit_price: float,
) -> None:
    cart = _get_cart(session)
    for item in cart:
        if item.get("sku") == sku:
            merged = _item_qty(item) + qty
            item["quantity"] = merged
            item["qty"] = merged
            item["product_name"] = product_name
            item["unit_price"] = unit_price
            _set_cart(session, cart)
            return
    cart.append(
        {
            "sku": sku,
            "product_name": product_name,
            "quantity": qty,
            "qty": qty,
            "unit_price": unit_price,
        }
    )
    _set_cart(session, cart)


def _tool_lookup_product(args: dict, db: Session) -> dict:
    query = (args.get("query") or "").strip()
    product, error, match_mode = _resolve_product_match(query, db)
    if error == "restricted":
        return {"error": "product_restricted", "query": query}
    if product is None:
        return {
            "error": "product_not_found",
            "query": query,
            "suggestions": _suggest_products(query, db),
        }
    return {
        "product_name": product.product_name,
        "sku": _product_sku(product),
        "salt_name": product.salt_name or "",
        "match_mode": match_mode,
    }


def _tool_add_to_cart(args: dict, session: dict, db: Session) -> dict:
    qty = args.get("quantity")
    try:
        qty = int(qty)
    except (TypeError, ValueError):
        return {"error": "invalid_quantity"}
    if qty < 1:
        return {"error": "invalid_quantity", "message": "Quantity must be a positive integer."}

    product_query = (args.get("product_query") or "").strip()
    product, error, match_mode = _resolve_product_match(product_query, db)
    if error == "restricted":
        return {"error": "product_restricted"}
    if product is None:
        return {
            "error": "product_not_found",
            "suggestions": _suggest_products(args.get("product_query", ""), db),
        }
    if match_mode == "token":
        _set_pending_product(
            session,
            sku=_product_sku(product),
            product_name=product.product_name,
            qty=qty,
        )
        session["order_state"] = COLLECT_SKU_CONFIRM
        return {
            "error": "needs_product_confirmation",
            "candidate": product.product_name,
            "query": product_query,
            "quantity": qty,
        }

    unit_price = float(product.price_per_strip or 0.0)
    _clear_pending_product(session)
    _add_line_to_cart(
        session,
        _product_sku(product),
        product.product_name,
        qty,
        unit_price,
    )
    session["order_state"] = CART_MENU
    for key in ("order_sku", "order_product_name", "order_qty", "order_unit_price"):
        session.pop(key, None)
    return {
        "ok": True,
        "cart": _get_cart(session),
        "added": {
            "product_name": product.product_name,
            "quantity": qty,
            "unit_price": unit_price,
            "line_total": round(qty * unit_price, 2),
        },
    }


def _tool_update_cart_line(args: dict, session: dict) -> dict:
    cart = _get_cart(session)
    if not cart:
        return {"error": "empty_cart"}
    try:
        qty = int(args.get("quantity"))
    except (TypeError, ValueError):
        return {"error": "invalid_quantity"}
    idx = _find_cart_line(
        cart,
        args.get("line_number"),
        args.get("product_query"),
    )
    if idx is None:
        return {"error": "line_not_found", "cart": cart}
    cart[idx]["quantity"] = qty
    cart[idx]["qty"] = qty
    _set_cart(session, cart)
    session["order_state"] = CART_MENU
    return {"ok": True, "cart": cart}


def _tool_remove_from_cart(args: dict, session: dict) -> dict:
    cart = _get_cart(session)
    if not cart:
        return {"error": "empty_cart"}
    idx = _find_cart_line(
        cart,
        args.get("line_number"),
        args.get("product_query"),
    )
    if idx is None:
        return {"error": "line_not_found", "cart": cart}
    removed = cart.pop(idx)
    _set_cart(session, cart)
    session["order_state"] = COLLECT_SKU if not cart else CART_MENU
    return {"ok": True, "removed": removed, "cart": cart}


def _tool_view_cart(session: dict) -> dict:
    return {"cart": _get_cart(session), "phase": session.get("order_state", COLLECT_SKU)}


def _tool_proceed_to_checkout(session: dict) -> dict:
    if not _get_cart(session):
        return {"error": "empty_cart"}
    _prefill_order_country(session)
    session["order_state"] = COLLECT_CHECKOUT
    return {"ok": True, "phase": COLLECT_CHECKOUT, "next": "collect_checkout"}


def _tool_set_shipping(args: dict, session: dict) -> dict:
    country = (args.get("country") or "").strip()
    city = (args.get("city") or "").strip()
    if not country:
        country = (_prefill_order_country(session) or "").strip()
    if country:
        if is_shipment_excluded_country(country):
            _clear_order_session(session)
            return {"error": "sanctioned_country", "message": SANCTIONED_COUNTRY_REFUSAL}
        session["order_country"] = country
    if city:
        session["order_city"] = city

    if session.get("order_country") and session.get("order_city"):
        session["order_state"] = COLLECT_CONTACT
        return {"ok": True, "phase": COLLECT_CONTACT, "next": "collect_contact"}
    if session.get("order_country"):
        session["order_state"] = COLLECT_CITY
        return {"ok": True, "phase": COLLECT_CITY, "next": "collect_city"}
    session["order_state"] = COLLECT_CHECKOUT
    return {"ok": True, "phase": COLLECT_CHECKOUT, "next": "collect_checkout"}


def _tool_set_contact(args: dict, session: dict) -> dict:
    contact = (args.get("contact") or "").strip()
    if len(contact) < 3:
        return {"error": "contact_too_short"}
    if not _get_cart(session):
        return {"error": "empty_cart"}
    session["order_contact"] = contact
    session["order_state"] = CONFIRM_ORDER
    return {"ok": True, "phase": CONFIRM_ORDER, "review": _format_order_review(session)}


async def _tool_confirm_order(session: dict, db: Session) -> dict:
    if session.get("order_state") != CONFIRM_ORDER:
        return {"error": "wrong_phase", "phase": session.get("order_state")}
    required = ("order_country", "order_city", "order_contact")
    missing = [k for k in required if not session.get(k)]
    if missing:
        return {"error": "incomplete", "missing": missing}
    reply, updated = await _commit_order(session, db)
    session.clear()
    session.update(updated)
    return {"ok": True, "committed": True, "final_reply": reply}


async def _commit_order(session: dict, db: Session) -> tuple[str, dict]:
    cart = _get_cart(session)
    if not cart:
        session["order_state"] = COLLECT_SKU
        return "Your cart is empty. Which product would you like to add?", session

    order_ref = session.get("order_ref") or (
        f"ORD-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    )
    phone = session.get("phone") or ""
    contact = session.get("order_contact") or ""
    order_total = _cart_total(cart)
    if order_total <= 0:
        order_total = 1.0

    for idx, item in enumerate(cart, start=1):
        db.add(
            Order(
                phone=phone,
                sku=item["sku"],
                product_name=item.get("product_name"),
                quantity=int(item["quantity"]),
                country=session["order_country"],
                city=session["order_city"],
                contact_name=session["order_contact"],
                order_ref=f"{order_ref}-L{idx:02d}",
                status="pending",
                payment_status="awaiting_payment",
            )
        )
    db.commit()

    await send_order_alert(
        {
            "order_ref": order_ref,
            "phone": phone,
            "city": session.get("order_city"),
            "country": session.get("order_country"),
            "contact_name": session.get("order_contact"),
            "payment_method": PAYMENT_METHOD,
            "lines": [
                {
                    "product_name": item.get("product_name"),
                    "sku": item.get("sku"),
                    "quantity": item.get("quantity"),
                }
                for item in cart
            ],
        }
    )

    _clear_order_session(session)
    session.pop("cart", None)
    session.pop("pending_product", None)
    session["lead_qualified"] = True
    session["greeted"] = True
    session["last_order_ref"] = order_ref
    session["last_order_total"] = order_total
    session["last_order_contact"] = contact
    session["order_state"] = SELECT_PAYMENT
    if phone:
        await send_interactive_buttons(
            phone,
            (
                f"✅ *Order {order_ref} confirmed!*\n"
                f"Total: {_format_money(order_total)}\n\n"
                "Please choose how you'd like to pay the full amount:"
            ),
            PAYMENT_OPTION_BUTTONS,
        )
    reply = (
        f"✅ *Order Confirmed!*\n"
        f"Order Ref: {order_ref}\n"
        f"Total: {_format_money(order_total)}\n\n"
        "Choose *Bank Transfer* or *Debit / Credit Card* to complete payment."
    )
    return reply, session


def _latest_order_by_phone(db: Session, phone: str, order_ref: str | None = None) -> Order | None:
    variants = phone_lookup_variants(phone)
    if not variants:
        return None
    q = db.query(Order).filter(Order.phone.in_(variants))
    if order_ref:
        q = q.filter(Order.order_ref.ilike(f"{order_ref}%"))
    return q.order_by(Order.created_at.desc(), Order.id.desc()).first()


def _orders_for_base_ref(db: Session, phone: str, base_ref: str) -> list[Order]:
    variants = phone_lookup_variants(phone)
    if not variants:
        return []
    return (
        db.query(Order)
        .filter(Order.phone.in_(variants), Order.order_ref.ilike(f"{base_ref}%"))
        .order_by(Order.order_ref.asc())
        .all()
    )


def _order_total_from_lines(db: Session, lines: list[Order]) -> float:
    total = 0.0
    for line in lines:
        sku = line.sku or ""
        match = re.fullmatch(r"PROD-(\d+)", sku, re.IGNORECASE)
        if not match:
            continue
        product = db.query(Product).filter(Product.id == int(match.group(1))).first()
        if product is None:
            continue
        total += float(product.price_per_strip or 0) * int(line.quantity or 0)
    return round(total, 2)


def _resolve_pending_payment(session: dict, db: Session) -> dict:
    """Restore payment context from session or latest awaiting_payment order in DB."""
    session = dict(session or {})
    if session.get("last_order_ref") and float(session.get("last_order_total") or 0) > 0:
        return session

    phone = session.get("phone") or ""
    latest = _latest_order_by_phone(db, phone)
    if latest is None:
        return session

    status = (latest.payment_status or latest.status or "").strip().lower()
    if status not in {"awaiting_payment", "pending", ""}:
        return session

    base_ref = (latest.order_ref or "").rsplit("-L", 1)[0]
    if not base_ref:
        return session

    lines = _orders_for_base_ref(db, phone, base_ref)
    total = _order_total_from_lines(db, lines)
    if total <= 0:
        total = float(session.get("last_order_total") or 1.0)

    session["last_order_ref"] = base_ref
    session["last_order_total"] = total
    session["order_state"] = SELECT_PAYMENT
    if latest.contact_name and not session.get("last_order_contact"):
        session["last_order_contact"] = latest.contact_name
    return session


def _update_orders_virtual_account(
    db: Session, base_ref: str, virtual_account_id: str | None
) -> None:
    if not virtual_account_id:
        return
    db.query(Order).filter(Order.order_ref.ilike(f"{base_ref}%")).update(
        {"virtual_account_id": virtual_account_id},
        synchronize_session=False,
    )
    db.commit()


async def _send_post_payment_buttons(phone: str, body: str) -> None:
    if phone:
        await send_interactive_buttons(phone, body, POST_PAYMENT_BUTTONS)


async def _handle_bank_transfer(
    session: dict, db: Session, order_ref: str, amount: float, phone: str
) -> tuple[str, dict]:
    contact = session.get("last_order_contact") or session.get("order_contact") or ""
    va_details = await create_virtual_account(order_ref, amount, phone)
    _update_orders_virtual_account(db, order_ref, va_details.get("virtual_account_id"))

    order_payload = {
        "order_ref": order_ref,
        "total_amount": amount,
        "contact_name": contact,
    }
    instructions = get_payment_instructions_text(order_payload, va_details)
    session.pop("order_state", None)
    session["payment_method_chosen"] = "bank_transfer"

    if phone:
        await send_message(phone, instructions)
        await _send_post_payment_buttons(
            phone,
            f"Bank transfer details sent for *{order_ref}*.\n"
            "Share your UTR/reference once you've paid.",
        )

    return (
        f"Bank transfer details for *{order_ref}* ({_format_money(amount)}) have been sent. "
        "Please complete the transfer and reply with your UTR/reference number.",
        session,
    )


async def _handle_card_payment(
    session: dict, order_ref: str, amount: float, phone: str
) -> tuple[str, dict]:
    contact = session.get("last_order_contact") or session.get("order_contact") or ""
    link_details = await create_card_checkout(order_ref, amount, phone, contact)
    link_url = link_details.get("link_url")

    session.pop("order_state", None)
    session["payment_method_chosen"] = "card"

    if not link_url:
        return (
            "Sorry, we couldn't generate a card payment link right now. "
            "Please choose *Bank Transfer* or contact our team for assistance.",
            session,
        )

    card_text = get_card_payment_text(order_ref, amount, link_url)
    if phone:
        await send_message(phone, card_text)
        await _send_post_payment_buttons(
            phone,
            f"Secure card payment link sent for *{order_ref}*.",
        )

    return (
        f"Secure payment link for *{order_ref}* ({_format_money(amount)}) has been sent. "
        "Open the link to pay by debit or credit card.",
        session,
    )


async def _handle_payment_selection(
    message: str, session: dict, db: Session
) -> tuple[str, dict]:
    text = (message or "").strip().lower()
    session = _resolve_pending_payment(session, db)
    order_ref = session.get("last_order_ref")
    amount = float(session.get("last_order_total") or 0.0)
    phone = session.get("phone") or ""

    if not order_ref:
        return (
            "I couldn't find a recent order to pay for. "
            "Please place an order first or share your order reference (e.g. ORD-20260603-2439).",
            session,
        )

    if text == PAY_BANK_BUTTON or "bank transfer" in text:
        return await _handle_bank_transfer(session, db, order_ref, amount, phone)
    if text == PAY_CARD_BUTTON or "debit" in text or "credit card" in text or text == "card":
        return await _handle_card_payment(session, order_ref, amount, phone)

    return (
        f"Order *{order_ref}* total: {_format_money(amount)}.\n\n"
        "Please choose a payment method using the buttons below:\n"
        "• *Bank Transfer* — domestic or international wire\n"
        "• *Debit / Credit Card* — secure Cashfree checkout link",
        session,
    )


def _is_payment_button_message(text: str, session: dict) -> bool:
    lowered = (text or "").strip().lower()
    if lowered in PAYMENT_BUTTON_IDS:
        return True
    if session.get("order_state") == SELECT_PAYMENT:
        return True
    if lowered in {"bank transfer", "debit / credit card"}:
        return True
    if "bank transfer" in lowered or "debit" in lowered or "credit card" in lowered:
        return True
    return False


async def _try_payment_actions(
    message: str, session: dict, db: Session
) -> tuple[str, dict] | None:
    text = (message or "").strip().lower()
    state = session.get("order_state")

    if _is_payment_button_message(message, session):
        return await _handle_payment_selection(message, session, db)

    if text == "new_order":
        session.pop("order_state", None)
        session.pop("last_order_ref", None)
        session.pop("last_order_total", None)
        session.pop("payment_method_chosen", None)
        session["order_state"] = COLLECT_SKU
        return "Starting a new order. Which product would you like to add?", session

    return None


def _extract_order_status_ref(text: str) -> str | None:
    match = re.search(r"\b(ORD-[0-9][0-9\-]*)\b", (text or "").upper())
    return match.group(1) if match else None


def is_order_tracking_message(text: str) -> bool:
    """True for order status / AWB tracking queries (route to order agent)."""
    lowered = (text or "").lower().strip()
    return (
        lowered == "order_status"
        or lowered == "order status"
        or "order status" in lowered
        or "track" in lowered
        or "where is my order" in lowered
        or "where is my shipment" in lowered
        or "awb" in lowered
        or bool(_extract_order_status_ref(text))
        or bool(extract_tracking_number(text))
    )


def _is_order_status_query(text: str) -> bool:
    return is_order_tracking_message(text)


def _format_order_summary_for_status(
    order,
    base_ref: str,
    *,
    shipment_summary: dict | None = None,
) -> str:
    """Order/payment lines shown below live shipment tracking."""
    payment_status = (order.payment_status or "").strip().lower()
    order_status = (order.status or "processing").strip().lower()
    lines = [f"📋 *Order {base_ref}*"]

    if payment_status in {"awaiting_payment", "payment_received"}:
        lines.append(f"Payment: {payment_status.replace('_', ' ').title()}")
        lines.append(_status_message(payment_status))
        return "\n".join(lines)

    if shipment_summary:
        # Shipment block above is source of truth — avoid contradicting "delivered".
        if not shipment_summary.get("is_delivered"):
            if payment_status == "shipped" or order_status == "shipped":
                lines.append("Shipment: Dispatched — see tracking above.")
            elif order_status == "processing":
                lines.append("Shipment: Being prepared.")
        return "\n".join(lines)

    display_status = payment_status if payment_status else order_status
    lines.append(f"Status: {display_status.replace('_', ' ').title()}")
    lines.append(_status_message(display_status))
    return "\n".join(lines)


async def _build_order_status_reply(
    order,
    base_ref: str,
    *,
    tracking_number: str,
) -> str:
    """Shipment tracking first (AWB, status, location), then order/payment summary."""
    tracking_msg = None
    shipment_summary = None
    if tracking_number:
        tracking_msg, shipment_summary = await fetch_tracking_bundle(tracking_number)

    order_summary = _format_order_summary_for_status(
        order,
        base_ref,
        shipment_summary=shipment_summary,
    )

    if tracking_msg:
        return f"{tracking_msg}\n\n{order_summary}"

    if tracking_number:
        return (
            f"{order_summary}\n\n"
            "_Live India Post tracking is temporarily unavailable — "
            "we'll update you when the shipment is booked._"
        )
    return order_summary


def _status_message(status: str, eta: str = "") -> str:
    mapping = {
        "awaiting_payment": "Awaiting your payment transfer.",
        "payment_received": "Payment received ✅ — processing your order.",
        "processing": "Being prepared for shipment.",
        "shipped": f"On the way! Expected delivery: {eta or 'To be shared soon.'}",
        "delivered": "Delivered ✅",
    }
    return mapping.get(status, "We are reviewing your order and will update you shortly.")


def _execute_order_tool(name: str, args: dict, session: dict, db: Session) -> dict:
    if name == "lookup_product":
        return _tool_lookup_product(args, db)
    if name == "add_to_cart":
        return _tool_add_to_cart(args, session, db)
    if name == "update_cart_line":
        return _tool_update_cart_line(args, session)
    if name == "remove_from_cart":
        return _tool_remove_from_cart(args, session)
    if name == "view_cart":
        return _tool_view_cart(session)
    if name == "proceed_to_checkout":
        return _tool_proceed_to_checkout(session)
    if name == "set_shipping":
        return _tool_set_shipping(args, session)
    if name == "set_contact":
        return _tool_set_contact(args, session)
    return {"error": "unknown_tool", "name": name}


async def _execute_order_tool_async(
    name: str, args: dict, session: dict, db: Session
) -> dict:
    if name == "confirm_order":
        return await _tool_confirm_order(session, db)
    return _execute_order_tool(name, args, session, db)


def _phase_hint(session: dict) -> str:
    phase = session.get("order_state", COLLECT_SKU)
    hints = {
        COLLECT_SKU: "Collect products for the cart (natural language OK).",
        COLLECT_SKU_CONFIRM: "Confirm suggested product before quantity/cart actions.",
        COLLECT_QTY: "Pending quantity for a product — use add_to_cart with quantity.",
        CART_MENU: "Cart building — add, edit, remove, or proceed_to_checkout.",
        COLLECT_COUNTRY: "Collect shipping country (skip if session.country is set).",
        COLLECT_CITY: "Collect city/port of entry.",
        COLLECT_CONTACT: "Collect buyer name and company.",
        COLLECT_CHECKOUT: "Collect name, city, phone in one buyer message.",
        CONFIRM_ORDER: "Show review; confirm_order only after explicit buyer confirmation.",
    }
    return hints.get(phase, "Order flow active.")


@observe(name="order_agent", capture_input=False)
async def _run_order_llm(message: str, session: dict, db: Session) -> tuple[str, dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    set_span_io(
        input_data={
            "message_len": len(message or ""),
            "phase": session.get("order_state"),
            "cart_lines": len(_get_cart(session)),
        }
    )

    client = get_async_openai_client(api_key=api_key)
    user_content = (
        f"Phase hint: {_phase_hint(session)}\n"
        f"Session snapshot: {json.dumps(_session_snapshot(session), default=str)}\n\n"
        f"Buyer message: {message or '(empty)'}"
    )
    messages: list[dict] = [
        {"role": "system", "content": ORDER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for _ in range(MAX_TOOL_CALLS_PER_TURN):
        response = await client.chat.completions.create(
            model=ORDER_MODEL,
            messages=messages,
            tools=ORDER_TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        assistant_msg = response.choices[0].message

        if not assistant_msg.tool_calls:
            reply = (assistant_msg.content or "").strip()
            if not reply:
                reply = (
                    "I'm here to help with your order. "
                    "Tell me which products and quantities you need."
                )
            set_span_io(output_data={"reply_len": len(reply), "agent": "order"})
            return reply, session

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
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await _execute_order_tool_async(
                tool_call.function.name,
                args,
                session,
                db,
            )
            if result.get("error") == "sanctioned_country":
                return result.get("message", SANCTIONED_COUNTRY_REFUSAL), session
            if result.get("committed"):
                return result.get("final_reply", ""), session
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                }
            )

    final = await client.chat.completions.create(
        model=ORDER_MODEL, messages=messages, temperature=0
    )
    reply = (final.choices[0].message.content or "").strip()
    set_span_io(output_data={"reply_len": len(reply), "agent": "order"})
    return reply, session


# --- Rule-based fallback (no API key / LLM failure) ---

_ORDER_INTENT_MARKERS = ("order", "buy", "purchase", "place an order", "want to order")
_CONFIRM_MARKERS = frozenset(
    {"confirm", "confirmed", "yes", "y", "ok", "okay", "proceed", "place order"}
)
_ADD_MARKERS = frozenset({"add", "more", "another", "+"})
_DONE_MARKERS = frozenset({"done", "checkout", "review", "proceed", "finish", "ship"})
_EDIT_MARKERS = frozenset({"edit", "cart", "list", "update"})


def _extract_positive_int(text: str) -> int | None:
    match = re.search(r"\d[\d,]*", (text or "").replace(",", ""))
    if not match:
        return None
    try:
        return int(match.group().replace(",", ""))
    except ValueError:
        return None


def _parse_remove_command(text: str) -> int | None:
    match = re.search(r"\b(?:remove|delete)\s*#?(\d+)\b", text.lower())
    return int(match.group(1)) if match else None


def _parse_qty_command(text: str) -> tuple[int, int] | None:
    match = re.search(
        r"\b(?:qty|quantity|units?)\s*#?(\d+)\s+(\d[\d,]*)\b",
        (text or "").lower(),
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2).replace(",", ""))


def _normalize_menu_action(text: str) -> str | None:
    lowered = (text or "").strip().lower()
    if not lowered:
        return None
    first = lowered.split()[0]
    if first in _CONFIRM_MARKERS or lowered in _CONFIRM_MARKERS:
        return "confirm"
    if first in _ADD_MARKERS or lowered in _ADD_MARKERS:
        return "add"
    if first in _DONE_MARKERS or lowered in _DONE_MARKERS:
        return "done"
    if first in _EDIT_MARKERS or lowered in _EDIT_MARKERS:
        return "edit"
    return None


def _try_cart_edit_commands(text: str, session: dict) -> tuple[str, dict] | None:
    cart = _get_cart(session)
    if not cart:
        return None
    line_no = _parse_remove_command(text)
    if line_no is not None:
        result = _tool_remove_from_cart({"line_number": line_no}, session)
        if result.get("error"):
            return f"Line {line_no} not found.", session
        return f"Removed line {line_no}. Cart:\n{_format_cart_lines(_get_cart(session))}", session
    qty_edit = _parse_qty_command(text)
    if qty_edit is not None:
        line_no, qty = qty_edit
        result = _tool_update_cart_line({"line_number": line_no, "quantity": qty}, session)
        if result.get("error"):
            return f"Could not update line {line_no}.", session
        return f"Updated line {line_no} to {qty} units.", session
    return None


async def _run_order_rules(
    message: str, session: dict, db: Session
) -> tuple[str, dict]:
    text = (message or "").strip()
    phone = session.get("phone") or ""
    if phone and _is_order_status_query(text):
        requested_ref = _extract_order_status_ref(text)
        awb_from_text = extract_tracking_number(text)
        latest = _latest_order_by_phone(db, phone, requested_ref)
        if not latest and awb_from_text:
            tracking_only = await lookup_tracking_message(awb_from_text)
            if tracking_only:
                return tracking_only, session
            return (
                f"I couldn't find an order linked to AWB *{awb_from_text}* yet. "
                "Please check the number or contact our team.",
                session,
            )
        if not latest:
            return (
                "I couldn't find any recent order for this number yet. "
                "Please share your order reference (e.g., ORD-12345) or AWB number.",
                session,
            )
        ref = latest.order_ref or "ORD-UNKNOWN"
        base_ref = ref.split("-L")[0]
        tracking_number = awb_from_text or (latest.tracking_number or "")
        reply = await _build_order_status_reply(
            latest,
            base_ref,
            tracking_number=tracking_number,
        )
        return reply, session

    state = session.get("order_state") or COLLECT_SKU
    session["order_state"] = state

    edit_result = _try_cart_edit_commands(text, session)
    if edit_result and state in {CART_MENU, CONFIRM_ORDER}:
        return edit_result

    if state == COLLECT_SKU_CONFIRM:
        pending_sku = session.get("order_pending_sku")
        pending_name = session.get("order_pending_product_name")
        pending_qty = session.get("order_pending_qty")
        if not pending_sku or not pending_name:
            _clear_pending_product(session)
            session["order_state"] = COLLECT_SKU
            return "Which product would you like to add? (name or SKU)", session

        if _normalize_menu_action(text) == "confirm":
            if pending_qty:
                _add_line_to_cart(session, pending_sku, pending_name, int(pending_qty), 0.0)
                _clear_pending_product(session)
                session["order_state"] = CART_MENU
                return (
                    f"Added to cart.\n\n{_format_cart_lines(_get_cart(session))}\n\n"
                    "Reply *checkout* when finished adding products.",
                    session,
                )
            product = None
            if pending_sku:
                sku_match = re.fullmatch(r"PROD-(\d+)", pending_sku, re.IGNORECASE)
                if sku_match:
                    product = db.query(Product).filter(Product.id == int(sku_match.group(1))).first()
            if product is None and pending_name:
                product = (
                    db.query(Product)
                    .filter(Product.product_name == pending_name)
                    .first()
                )
            _clear_pending_product(session)
            if product is None:
                session["order_state"] = COLLECT_SKU
                return BULK_LIST_PROMPT, session
            return await _start_quantity_step(session, product, phone=phone)

        _clear_pending_product(session)
        session["order_state"] = COLLECT_SKU
        if not text:
            return BULK_LIST_PROMPT, session
        # Treat non-confirm text as a fresh product query.
        state = COLLECT_SKU

    if state == COLLECT_SKU:
        if _is_order_status_query(text):
            return await _run_order_rules(message, session, db)
        if not text or any(m in text.lower() for m in _ORDER_INTENT_MARKERS):
            return BULK_LIST_PROMPT, session
        if looks_like_bulk_order(text):
            return await _process_bulk_order(text, session, db, phone)
        product, error, match_mode = _resolve_product_match(text, db)
        if error == "restricted":
            return (
                "I'm unable to assist with that product through this channel. "
                "Please contact our medical compliance team directly.",
                session,
            )
        if product is None:
            suggestions = _suggest_products(text, db)
            reply = "I couldn't find that product. Please try the product name or SKU."
            if suggestions:
                reply += "\n\nDid you mean:\n• " + "\n• ".join(suggestions)
            return reply, session
        if match_mode == "token":
            _set_pending_product(
                session,
                sku=_product_sku(product),
                product_name=product.product_name,
            )
            session["order_state"] = COLLECT_SKU_CONFIRM
            return (
                f"Did you mean *{product.product_name}*?\n"
                "Reply *yes* to continue or share the exact product name/SKU.",
                session,
            )
        return await _start_quantity_step(session, product, phone=phone)

    if state == COLLECT_QTY:
        key = text.strip().lower()
        if key == "qty_custom":
            session["order_qty_custom"] = True
            return "Type the quantity you need (e.g. 750):", session

        qty: int | None = QTY_BUTTON_MAP.get(key)
        if qty is None:
            qty = _extract_positive_int(text)
        if qty is None or qty < 1:
            return "Select a quantity from the list or type a positive number.", session

        session.pop("order_qty_custom", None)
        result = _tool_add_to_cart(
            {
                "product_query": session.get("order_product_name", ""),
                "quantity": qty,
            },
            session,
            db,
        )
        if result.get("error"):
            return "Could not add to cart. Please try again.", session

        continued = await _continue_bulk_queue(session, db, phone)
        if continued:
            return continued

        session["order_state"] = CART_MENU
        return _format_cart_lines(_get_cart(session)), session

    if state == CART_MENU:
        action = _normalize_menu_action(text)
        if action == "add":
            session["order_state"] = COLLECT_SKU
            return BULK_LIST_PROMPT, session
        if action == "done" or (text or "").strip().lower() == "checkout":
            result = _tool_proceed_to_checkout(session)
            if result.get("error"):
                return "Your cart is empty.", session
            country = _prefill_order_country(session) or ""
            return checkout_prompt(country), session
        if not text:
            return _format_cart_lines(_get_cart(session)), session
        return "Reply *checkout* to proceed, or *add* for more products.", session

    if state == COLLECT_CHECKOUT:
        parsed = parse_checkout_oneline(text, _prefill_order_country(session))
        if not parsed:
            country = _prefill_order_country(session) or "your country"
            return (
                f"Please send all details in one message:\n"
                f"*Name, City, Phone*\n\n"
                f"Shipping country: *{country}*"
            ), session
        country = parsed.get("country") or _prefill_order_country(session) or ""
        if country and is_shipment_excluded_country(country):
            _clear_order_session(session)
            return SANCTIONED_COUNTRY_REFUSAL, session
        if country:
            session["order_country"] = country
        session["order_city"] = parsed["city"]
        session["order_contact"] = parsed["contact"]
        session["order_state"] = CONFIRM_ORDER
        return (
            "Please review your order:\n"
            f"{_format_cart_lines(_get_cart(session))}\n\n"
            f"Ship to: *{session.get('order_city')}*, *{session.get('order_country')}*\n"
            f"Contact: *{session.get('order_contact')}*\n"
            f"Payment: *{PAYMENT_METHOD}*\n\n"
            "Reply *CONFIRM* to place the order."
        ), session

    if state == COLLECT_COUNTRY:
        if is_shipment_excluded_country(text):
            _clear_order_session(session)
            return SANCTIONED_COUNTRY_REFUSAL, session
        session["order_country"] = text
        session["order_state"] = COLLECT_CITY
        return "Which city or port of entry?", session

    if state == COLLECT_CITY:
        session["order_city"] = text
        session["order_state"] = COLLECT_CONTACT
        return "Your name and company for this order?", session

    if state == COLLECT_CONTACT:
        if len(text) < 3:
            return "Please share your name and company.", session
        session["order_contact"] = text
        session["order_state"] = CONFIRM_ORDER
        return (
            "Please review your order:\n"
            f"{_format_cart_lines(_get_cart(session))}\n\n"
            f"Payment method: *{PAYMENT_METHOD}*\n\n"
            "Reply *CONFIRM* to place the order."
        ), session

    if state == CONFIRM_ORDER:
        if _normalize_menu_action(text) == "confirm":
            return await _commit_order(session, db)
        return (
            "Please reply *CONFIRM* to place the order, or describe cart changes."
        ), session

    session["order_state"] = COLLECT_SKU
    return BULK_LIST_PROMPT, session


async def run_order_agent(message: str, session: dict, db: Session) -> tuple[str, dict]:
    """Run one turn of the order agent (LLM + tools, with rule fallback)."""
    session = dict(session or {})
    _migrate_legacy_single_line_session(session)
    _ensure_order_started(session)
    state = session.get("order_state") or COLLECT_SKU
    text = (message or "").strip()

    payment_result = await _try_payment_actions(message, session, db)
    if payment_result is not None:
        return payment_result

    # Deterministic UX guards regardless of LLM mode.
    if _is_order_status_query(text):
        return await _run_order_rules(message, session, db)
    if state in {
        COLLECT_SKU,
        COLLECT_QTY,
        COLLECT_SKU_CONFIRM,
        COLLECT_CHECKOUT,
        CONFIRM_ORDER,
    }:
        return await _run_order_rules(message, session, db)
    if state == CART_MENU and (
        not text
        or text.lower() in {"done", "checkout", "add"}
        or _normalize_menu_action(text) in {"done", "add"}
    ):
        return await _run_order_rules(message, session, db)

    use_llm = os.getenv("ORDER_AGENT_USE_LLM", "true").lower() in {"1", "true", "yes"}
    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            return await _run_order_llm(message, session, db)
        except Exception:
            logger.exception("Order LLM agent failed; using rule fallback")

    return await _run_order_rules(message, session, db)
