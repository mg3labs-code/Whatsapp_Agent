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
from app.business.countries import (
    SHIPMENT_EXCLUDED_REFUSAL,
    is_shipment_excluded_country,
)
from app.db.models import Order, Product
from app.integrations.alerts import send_order_alert
from app.integrations.whatsapp import send_interactive_buttons
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
COLLECT_PAYMENT = "COLLECT_PAYMENT"
CONFIRM_ORDER = "CONFIRM_ORDER"
ORDER_COMPLETE = "ORDER_COMPLETE"

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
    "order_payment",
    "order_ref",
    "order_pending_sku",
    "order_pending_product_name",
    "order_pending_qty",
    "pending_product",
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
    "- set_shipping / set_contact / set_payment_terms: extract from natural phrases.\n"
    "- confirm_order ONLY when the buyer clearly confirms (yes, confirm, place order) and phase is CONFIRM_ORDER.\n"
    "- If lookup fails, show suggestions from the tool and ask for a clearer product name.\n"
    "- Use *single asterisks* for bold (WhatsApp). Be concise and professional.\n"
    "- Do not commit the order without confirm_order after payment terms are collected."
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
            "name": "set_payment_terms",
            "description": "Payment terms (e.g. T/T Advance, LC, 30-day net). Shows order review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payment_terms": {"type": "string"},
                },
                "required": ["payment_terms"],
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


def _format_money(amount: float) -> str:
    return f"${amount:,.2f}"


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
        f"payment: {session.get('order_payment', '')}"
    )


def _session_snapshot(session: dict) -> dict[str, Any]:
    return {
        "phase": session.get("order_state", COLLECT_SKU),
        "cart": _get_cart(session),
        "country": session.get("order_country"),
        "city": session.get("order_city"),
        "contact": session.get("order_contact"),
        "payment": session.get("order_payment"),
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
    session["order_state"] = COLLECT_COUNTRY
    return {"ok": True, "phase": COLLECT_COUNTRY, "next": "collect_country"}


def _tool_set_shipping(args: dict, session: dict) -> dict:
    country = (args.get("country") or "").strip()
    city = (args.get("city") or "").strip()
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
    session["order_state"] = COLLECT_COUNTRY
    return {"ok": True, "phase": COLLECT_COUNTRY, "next": "collect_country"}


def _tool_set_contact(args: dict, session: dict) -> dict:
    contact = (args.get("contact") or "").strip()
    if len(contact) < 3:
        return {"error": "contact_too_short"}
    session["order_contact"] = contact
    session["order_state"] = COLLECT_PAYMENT
    return {"ok": True, "phase": COLLECT_PAYMENT, "next": "collect_payment"}


def _tool_set_payment_terms(args: dict, session: dict) -> dict:
    terms = (args.get("payment_terms") or "").strip()
    if not terms:
        return {"error": "missing_payment_terms"}
    if not _get_cart(session):
        return {"error": "empty_cart"}
    session["order_payment"] = terms
    session["order_state"] = CONFIRM_ORDER
    return {"ok": True, "phase": CONFIRM_ORDER, "review": _format_order_review(session)}


async def _tool_confirm_order(session: dict, db: Session) -> dict:
    if session.get("order_state") != CONFIRM_ORDER:
        return {"error": "wrong_phase", "phase": session.get("order_state")}
    required = ("order_country", "order_city", "order_contact", "order_payment")
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
                payment_terms=session["order_payment"],
                order_ref=f"{order_ref}-L{idx:02d}",
                status="awaiting_payment",
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
            "payment_terms": session.get("order_payment"),
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

    city = session.get("order_city", "")
    country = session.get("order_country", "")
    contact = session.get("order_contact", "")
    payment = session.get("order_payment", "")
    product_lines = "\n".join(
        f"• {item.get('product_name')} — {_item_qty(item)} units ({item.get('sku')})"
        for item in cart
    )
    _clear_order_session(session)
    session.pop("cart", None)
    session.pop("pending_product", None)
    session["lead_qualified"] = True
    session["greeted"] = True
    session["last_order_ref"] = order_ref
    phone = session.get("phone") or ""
    if phone:
        await send_interactive_buttons(
            phone,
            (
                f"✅ *Order {order_ref} confirmed!*\n"
                "Our sales team will send your proforma invoice within 24 hours."
            ),
            [
                {"id": "new_order", "title": "New Order"},
                {"id": "order_status", "title": "Order Status"},
                {"id": "speak", "title": "Speak to Team"},
            ],
        )
    reply = (
        f"✅ *Order Confirmed!*\n"
        f"Order Ref: {order_ref}\n"
        "Our sales team will send your proforma invoice within 24 hours."
    )
    return reply, session


def _extract_order_status_ref(text: str) -> str | None:
    match = re.search(r"\b(ORD-[0-9][0-9\-]*)\b", (text or "").upper())
    return match.group(1) if match else None


def _is_order_status_query(text: str) -> bool:
    lowered = (text or "").lower()
    return (
        "order status" in lowered
        or "where is my order" in lowered
        or bool(_extract_order_status_ref(text))
    )


def _status_message(status: str, eta: str = "") -> str:
    mapping = {
        "awaiting_payment": "Awaiting your payment transfer.",
        "payment_received": "Payment received ✅ — processing your order.",
        "processing": "Being prepared for shipment.",
        "shipped": f"On the way! Expected delivery: {eta or 'To be shared soon.'}",
        "delivered": "Delivered ✅",
    }
    return mapping.get(status, "We are reviewing your order and will update you shortly.")


def _latest_order_by_phone(db: Session, phone: str, order_ref: str | None = None) -> Order | None:
    q = db.query(Order).filter(Order.phone == phone)
    if order_ref:
        q = q.filter(Order.order_ref.ilike(f"{order_ref}%"))
    return q.order_by(Order.created_at.desc(), Order.id.desc()).first()


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
    if name == "set_payment_terms":
        return _tool_set_payment_terms(args, session)
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
        COLLECT_COUNTRY: "Collect shipping country.",
        COLLECT_CITY: "Collect city/port of entry.",
        COLLECT_CONTACT: "Collect buyer name and company.",
        COLLECT_PAYMENT: "Collect payment terms, then show review.",
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
        latest = _latest_order_by_phone(db, phone, requested_ref)
        if not latest:
            return (
                "I couldn't find any recent order for this number yet. "
                "Please share your order reference (e.g., ORD-12345).",
                session,
            )
        ref = latest.order_ref or "ORD-UNKNOWN"
        base_ref = ref.split("-L")[0]
        status = (latest.status or "processing").strip().lower()
        return (
            f"📦 *Order {base_ref}*\n"
            f"Status: {status}\n"
            f"{_status_message(status)}",
            session,
        )

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
                    "Reply *done* when finished adding products.",
                    session,
                )
            session["order_sku"] = pending_sku
            session["order_product_name"] = pending_name
            _clear_pending_product(session)
            session["order_state"] = COLLECT_QTY
            return (
                f"How many units of {session['order_product_name']}?",
                session,
            )

        _clear_pending_product(session)
        session["order_state"] = COLLECT_SKU
        if not text:
            return "Please share the exact product name or SKU.", session
        # Treat non-confirm text as a fresh product query.
        state = COLLECT_SKU

    if state == COLLECT_SKU:
        if not text:
            return "Which product would you like to add? (name or SKU)", session
        product, error, match_mode = _resolve_product_match(text, db)
        if error == "restricted":
            return (
                "I'm unable to assist with that product through this channel. "
                "Please contact our medical compliance team directly.",
                session,
            )
        if product is None:
            if any(m in text.lower() for m in _ORDER_INTENT_MARKERS):
                return "Which product would you like to add? (name or SKU)", session
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
        _clear_pending_product(session)
        session["order_sku"] = _product_sku(product)
        session["order_product_name"] = product.product_name
        session["order_unit_price"] = float(product.price_per_strip or 0.0)
        session["pending_product"] = {
            "name": product.product_name,
            "sku": _product_sku(product),
        }
        session["order_state"] = COLLECT_QTY
        return (
            f"Found: *{product.product_name}*\n"
            "How many units?",
            session,
        )

    if state == COLLECT_QTY:
        qty = _extract_positive_int(text)
        if qty is None or qty < 1:
            return "Please enter a positive number of units (e.g. 1000).", session
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
        return _format_cart_lines(_get_cart(session)), session

    if state == CART_MENU:
        action = _normalize_menu_action(text)
        if action == "add":
            session["order_state"] = COLLECT_SKU
            return "Which product would you like to add?", session
        if action == "done" or (text or "").strip().lower() == "checkout":
            result = _tool_proceed_to_checkout(session)
            if result.get("error"):
                return "Your cart is empty.", session
            return "Which country should we ship to?", session
        if not text:
            return _format_cart_lines(_get_cart(session)), session
        return "Reply *done* to checkout, or *add* for another product.", session

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
        session["order_state"] = COLLECT_PAYMENT
        return "Preferred payment terms? (T/T Advance, LC, or 30-day net)", session

    if state == COLLECT_PAYMENT:
        result = _tool_set_payment_terms({"payment_terms": text}, session)
        if result.get("error"):
            return "Please provide payment terms.", session
        return (
            "Please review your order:\n"
            f"{_format_cart_lines(_get_cart(session))}\n\n"
            "Reply *CONFIRM* to place the order."
        ), session

    if state == CONFIRM_ORDER:
        if _normalize_menu_action(text) == "confirm":
            return await _commit_order(session, db)
        return (
            "Please reply *CONFIRM* to place the order, or describe cart changes."
        ), session

    session["order_state"] = COLLECT_SKU
    return "Which product would you like to add?", session


async def run_order_agent(message: str, session: dict, db: Session) -> tuple[str, dict]:
    """Run one turn of the order agent (LLM + tools, with rule fallback)."""
    session = dict(session or {})
    _migrate_legacy_single_line_session(session)
    _ensure_order_started(session)
    state = session.get("order_state") or COLLECT_SKU
    text = (message or "").strip()

    # Deterministic UX guards regardless of LLM mode.
    if _is_order_status_query(text):
        return await _run_order_rules(message, session, db)
    if state in {COLLECT_SKU, COLLECT_QTY, COLLECT_SKU_CONFIRM}:
        return await _run_order_rules(message, session, db)

    use_llm = os.getenv("ORDER_AGENT_USE_LLM", "true").lower() in {"1", "true", "yes"}
    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            return await _run_order_llm(message, session, db)
        except Exception:
            logger.exception("Order LLM agent failed; using rule fallback")

    return await _run_order_rules(message, session, db)
