"""Order collection agent — rule-based multi-turn state machine."""

from __future__ import annotations

import logging
import random
import re
from datetime import datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.agents.pricing import get_product_by_name
from app.business.countries import (
    SHIPMENT_EXCLUDED_REFUSAL,
    is_shipment_excluded_country,
)
from app.db.models import Order, Product
from app.integrations.alerts import send_order_alert

logger = logging.getLogger(__name__)

COLLECT_SKU = "COLLECT_SKU"
COLLECT_QTY = "COLLECT_QTY"
COLLECT_COUNTRY = "COLLECT_COUNTRY"
COLLECT_CITY = "COLLECT_CITY"
COLLECT_CONTACT = "COLLECT_CONTACT"
COLLECT_PAYMENT = "COLLECT_PAYMENT"
ORDER_COMPLETE = "ORDER_COMPLETE"

DEFAULT_MOQ = 1

SANCTIONED_COUNTRY_REFUSAL = SHIPMENT_EXCLUDED_REFUSAL

_ORDER_INTENT_MARKERS = ("order", "buy", "purchase", "place an order", "want to order")

ORDER_SESSION_KEYS = (
    "order_state",
    "order_sku",
    "order_product_name",
    "order_moq",
    "order_qty",
    "order_country",
    "order_city",
    "order_contact",
    "order_payment",
)


def _product_sku(product: Product) -> str:
    return f"PROD-{product.id:04d}"


def _find_product(query: str, db: Session) -> tuple[Product | None, str | None]:
    """Look up a product by name fragment or PROD-#### SKU."""
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
    if result.get("error") == "product_not_found":
        return None, "not_found"

    product = (
        db.query(Product)
        .filter(Product.product_name == result["product_name"])
        .first()
    )
    if product is None:
        return None, "not_found"
    return product, None


def _suggest_products(query: str, db: Session) -> list[str]:
    pattern = f"%{(query or '')[:40]}%"
    rows = (
        db.query(Product.product_name)
        .filter(
            Product.is_restricted.is_(False),
            or_(
                Product.product_name.ilike(pattern),
                Product.salt_name.ilike(pattern),
            ),
        )
        .limit(3)
        .all()
    )
    return [name for (name,) in rows]


def _extract_positive_int(text: str) -> int | None:
    match = re.search(r"\d[\d,]*", (text or "").replace(",", ""))
    if not match:
        return None
    try:
        return int(match.group().replace(",", ""))
    except ValueError:
        return None


def _is_sanctioned_country(country: str) -> bool:
    return is_shipment_excluded_country(country)


def _clear_order_session(session: dict) -> None:
    for key in ORDER_SESSION_KEYS:
        session.pop(key, None)


async def run_order_agent(message: str, session: dict, db: Session) -> tuple[str, dict]:
    """Run one turn of the order state machine.

    Returns (reply_text, updated_session).
    """
    session = dict(session or {})
    text = (message or "").strip()
    state = session.get("order_state") or COLLECT_SKU
    session["order_state"] = state

    if state == COLLECT_SKU:
        return _handle_collect_sku(text, session, db)
    if state == COLLECT_QTY:
        return _handle_collect_qty(text, session)
    if state == COLLECT_COUNTRY:
        return _handle_collect_country(text, session)
    if state == COLLECT_CITY:
        return _handle_collect_city(text, session)
    if state == COLLECT_CONTACT:
        return _handle_collect_contact(text, session)
    if state == COLLECT_PAYMENT:
        return await _handle_collect_payment(text, session, db)
    if state == ORDER_COMPLETE:
        return await _handle_order_complete(session, db)

    session["order_state"] = COLLECT_SKU
    return _prompt_collect_sku(), session


def _handle_collect_sku(text: str, session: dict, db: Session) -> tuple[str, dict]:
    if not text:
        return _prompt_collect_sku(), session

    product, error = _find_product(text, db)
    if error == "restricted":
        return (
            "I'm unable to assist with that product through this channel. "
            "Please contact our medical compliance team directly.",
            session,
        )
    if product is None:
        lowered = text.lower()
        if any(marker in lowered for marker in _ORDER_INTENT_MARKERS):
            return _prompt_collect_sku(), session

        suggestions = _suggest_products(text, db)
        reply = (
            "I couldn't find that product. Could you try the full product name or SKU "
            "(e.g., PROD-0001)?"
        )
        if suggestions:
            reply += "\n\nDid you mean:\n• " + "\n• ".join(suggestions)
        return reply, session

    session["order_sku"] = _product_sku(product)
    session["order_product_name"] = product.product_name
    session["order_moq"] = DEFAULT_MOQ
    session["order_state"] = COLLECT_QTY
    moq = session["order_moq"]
    return (
        f"How many units of {session['order_product_name']}? (Minimum: {moq} units)",
        session,
    )


def _handle_collect_qty(text: str, session: dict) -> tuple[str, dict]:
    qty = _extract_positive_int(text)
    moq = int(session.get("order_moq") or DEFAULT_MOQ)

    if qty is None:
        return (
            "Please enter a number (e.g., 1000 or 5000).",
            session,
        )
    if qty < moq:
        return (
            f"The minimum order for {session.get('order_product_name', 'this product')} "
            f"is {moq} units. How many units would you like?",
            session,
        )

    session["order_qty"] = qty
    session["order_state"] = COLLECT_COUNTRY
    return "Which country should we ship to?", session


def _handle_collect_country(text: str, session: dict) -> tuple[str, dict]:
    country = text.strip()
    if not country:
        return "Which country should we ship to?", session

    if _is_sanctioned_country(country):
        _clear_order_session(session)
        return SANCTIONED_COUNTRY_REFUSAL, session

    session["order_country"] = country
    session["order_state"] = COLLECT_CITY
    return "Which city or port of entry?", session


def _handle_collect_city(text: str, session: dict) -> tuple[str, dict]:
    city = text.strip()
    if not city:
        return "Which city or port of entry?", session

    session["order_city"] = city
    session["order_state"] = COLLECT_CONTACT
    return "Your name and company for this order?", session


def _handle_collect_contact(text: str, session: dict) -> tuple[str, dict]:
    contact = text.strip()
    if len(contact) < 3:
        return "Please share your name and company (at least a few characters).", session

    session["order_contact"] = contact
    session["order_state"] = COLLECT_PAYMENT
    return (
        "Preferred payment terms? (T/T Advance, Letter of Credit, or 30-day net)",
        session,
    )


async def _handle_collect_payment(text: str, session: dict, db: Session) -> tuple[str, dict]:
    terms = text.strip()
    if not terms:
        return (
            "Preferred payment terms? (T/T Advance, Letter of Credit, or 30-day net)",
            session,
        )

    session["order_payment"] = terms
    session["order_state"] = ORDER_COMPLETE
    return await _handle_order_complete(session, db)


async def _handle_order_complete(session: dict, db: Session) -> tuple[str, dict]:
    order_ref = f"ORD-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    phone = session.get("phone") or ""

    order = Order(
        phone=phone,
        sku=session["order_sku"],
        product_name=session.get("order_product_name"),
        quantity=session["order_qty"],
        country=session["order_country"],
        city=session["order_city"],
        contact_name=session["order_contact"],
        payment_terms=session["order_payment"],
        order_ref=order_ref,
        status="pending",
    )
    db.add(order)
    db.commit()

    order_dict = {
        "order_ref": order_ref,
        "product_name": session.get("order_product_name"),
        "quantity": session.get("order_qty"),
        "city": session.get("order_city"),
        "country": session.get("order_country"),
        "contact_name": session.get("order_contact"),
        "phone": phone,
    }
    await send_order_alert(order_dict)

    product_name = session.get("order_product_name", "")
    qty = session.get("order_qty", "")
    city = session.get("order_city", "")
    country = session.get("order_country", "")
    contact = session.get("order_contact", "")
    payment = session.get("order_payment", "")

    _clear_order_session(session)

    reply = (
        "✅ *Order Confirmed!*\n"
        "📋 *Order Summary:*\n"
        f"• Product: {product_name}\n"
        f"• Quantity: {qty} units\n"
        f"• Ship to: {city}, {country}\n"
        f"• Contact: {contact}\n"
        f"• Payment: {payment}\n"
        f"• Order Ref: {order_ref}\n"
        "Our sales team will contact you within 24 hours with the proforma invoice. Thank you!"
    )
    return reply, session


def _prompt_collect_sku() -> str:
    return (
        "I'll help you place your order! Which product would you like to order? "
        "(product name or SKU)"
    )
