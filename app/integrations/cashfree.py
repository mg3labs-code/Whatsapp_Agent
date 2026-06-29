"""Cashfree helpers: payment links, VA creation, webhooks, overdue tracking."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import Order
from app.integrations.alerts import send_critical_error_alert, send_order_team_alert
from app.integrations.whatsapp import send_message
from app.session.manager import normalize_phone

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 12.0
OVERDUE_HOURS = 48
DEFAULT_API_VERSION = "2025-01-01"


def _cashfree_headers() -> dict[str, str]:
    app_id = os.getenv("CASHFREE_APP_ID", "")
    secret = os.getenv("CASHFREE_SECRET_KEY", "")
    env = os.getenv("CASHFREE_ENV", "sandbox").strip().lower()
    api_version = os.getenv("CASHFREE_API_VERSION", DEFAULT_API_VERSION).strip()
    base = (
        "https://sandbox.cashfree.com"
        if env != "production"
        else "https://api.cashfree.com"
    )
    return {
        "x-client-id": app_id,
        "x-client-secret": secret,
        "x-api-version": api_version,
        "content-type": "application/json",
        "accept": "application/json",
        "_base": base,
    }


def _base_order_ref(order_ref: str) -> str:
    if "-L" in (order_ref or ""):
        return order_ref.rsplit("-L", 1)[0]
    return order_ref


async def create_virtual_account(
    order_ref: str, amount: float, customer_phone: str
) -> dict[str, Any]:
    """Create VA (domestic) and include international remittance details if available."""
    headers = _cashfree_headers()
    base = headers.pop("_base")
    payload = {
        "reference_id": order_ref,
        "customer_details": {"customer_phone": customer_phone},
        "amount": float(amount),
        "remarks": f"Payment for {order_ref}",
    }
    out: dict[str, Any] = {
        "virtual_account_id": None,
        "account_number": None,
        "ifsc": None,
        "iban": None,
        "swift_code": None,
        "amount": float(amount),
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        # Domestic VA (Autocollect).
        try:
            domestic_url = f"{base}/pg/links/autocollect"
            domestic = await client.post(domestic_url, headers=headers, json=payload)
            if domestic.status_code < 400:
                data = domestic.json()
                out["virtual_account_id"] = (
                    data.get("virtual_account_id")
                    or data.get("va_id")
                    or data.get("id")
                )
                out["account_number"] = (
                    data.get("account_number")
                    or data.get("bank_details", {}).get("account_number")
                )
                out["ifsc"] = data.get("ifsc") or data.get("bank_details", {}).get("ifsc")
            else:
                logger.warning("Cashfree domestic VA creation failed: HTTP %s", domestic.status_code)
        except Exception:
            logger.exception("Cashfree domestic VA creation error")

        # International collection details (best effort).
        try:
            intl_url = f"{base}/pg/international/collections/accounts"
            intl = await client.get(intl_url, headers=headers)
            if intl.status_code < 400:
                rows = intl.json()
                row = rows[0] if isinstance(rows, list) and rows else {}
                out["iban"] = row.get("iban") or row.get("account_iban")
                out["swift_code"] = row.get("swift_code") or row.get("swift")
        except Exception:
            logger.exception("Cashfree international details fetch error")

    return out


def _sanitize_link_id(order_ref: str) -> str:
    """Cashfree link_id allows alphanumeric, hyphen, underscore."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", (order_ref or "").strip())
    return cleaned[:50] or "order"


def _cashfree_checkout_mode() -> str:
    """links | orders | auto — auto tries payment link then orders API."""
    return os.getenv("CASHFREE_PAYMENT_MODE", "auto").strip().lower()


def _checkout_base_url() -> str:
    """Origin Cashfree JS checkout validates — must be a Website-whitelisted domain."""
    explicit = os.getenv("CASHFREE_CHECKOUT_BASE_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    return os.getenv("BASE_URL", "").strip().rstrip("/")


def _checkout_page_url(payment_session_id: str) -> str | None:
    base_url = _checkout_base_url()
    if not base_url or not payment_session_id:
        return None
    return f"{base_url}/payment/checkout?session_id={quote(payment_session_id, safe='')}"


async def create_order_checkout(
    order_ref: str,
    amount: float,
    customer_phone: str,
    customer_name: str = "",
) -> dict[str, Any]:
    """Create Cashfree PG order + payment session; buyer pays via /payment/checkout."""
    headers = _cashfree_headers()
    base = headers.pop("_base")
    order_id = _sanitize_link_id(order_ref)
    currency = os.getenv("CASHFREE_LINK_CURRENCY", "INR").strip().upper() or "INR"
    base_url = os.getenv("BASE_URL", "").strip().rstrip("/")

    customer: dict[str, str] = {
        "customer_id": order_id[:45],
        "customer_phone": customer_phone.lstrip("+"),
    }
    if customer_name:
        customer["customer_name"] = customer_name

    payload: dict[str, Any] = {
        "order_id": order_id,
        "order_amount": round(float(amount), 2),
        "order_currency": currency,
        "customer_details": customer,
        "order_tags": {
            "order_ref": order_ref,
            "phone": customer_phone.lstrip("+"),
        },
    }
    if base_url:
        payload["order_meta"] = {
            "notify_url": f"{base_url}/webhook/cashfree",
            "return_url": f"{base_url}/payment/return?order_id={{order_id}}",
        }

    out: dict[str, Any] = {
        "link_id": order_id,
        "link_url": None,
        "amount": round(float(amount), 2),
        "currency": currency,
        "payment_session_id": None,
        "checkout_mode": "orders",
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        try:
            response = await client.post(f"{base}/pg/orders", headers=headers, json=payload)
            if response.status_code < 400:
                data = response.json()
                session_id = data.get("payment_session_id")
                out["payment_session_id"] = session_id
                out["link_url"] = _checkout_page_url(str(session_id or ""))
            else:
                logger.warning(
                    "Cashfree create order failed: HTTP %s body=%s",
                    response.status_code,
                    response.text[:300],
                )
        except Exception:
            logger.exception("Cashfree create order error")

    return out


async def create_payment_link(
    order_ref: str,
    amount: float,
    customer_phone: str,
    customer_name: str = "",
) -> dict[str, Any]:
    """Create a Cashfree payment link for debit/credit card checkout."""
    headers = _cashfree_headers()
    base = headers.pop("_base")
    link_id = _sanitize_link_id(order_ref)
    currency = os.getenv("CASHFREE_LINK_CURRENCY", "INR").strip().upper() or "INR"
    base_url = os.getenv("BASE_URL", "").strip().rstrip("/")
    expiry = datetime.now(timezone.utc) + timedelta(hours=48)

    payload: dict[str, Any] = {
        "link_id": link_id,
        "link_amount": round(float(amount), 2),
        "link_currency": currency,
        "link_purpose": f"Payment for {order_ref} - New Life Medicare",
        "link_expiry_time": expiry.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
        "link_partial_payments": False,
        "link_auto_reminders": True,
        "customer_details": {
            "customer_phone": customer_phone.lstrip("+"),
        },
        "link_notes": {
            "order_ref": order_ref,
            "phone": customer_phone.lstrip("+"),
        },
        "link_notify": {
            "send_sms": False,
            "send_email": False,
        },
    }
    if customer_name:
        payload["customer_details"]["customer_name"] = customer_name
    if base_url:
        payload["link_meta"] = {
            "notify_url": f"{base_url}/webhook/cashfree",
            "return_url": f"{base_url}/payment/return",
            "upi_intent": False,
        }

    out: dict[str, Any] = {
        "link_id": link_id,
        "link_url": None,
        "amount": round(float(amount), 2),
        "currency": currency,
        "checkout_mode": "payment_link",
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        try:
            response = await client.post(
                f"{base}/pg/links",
                headers=headers,
                json=payload,
            )
            if response.status_code < 400:
                data = response.json()
                out["link_url"] = data.get("link_url") or data.get("payment_link")
            else:
                logger.warning(
                    "Cashfree payment link creation failed: HTTP %s body=%s",
                    response.status_code,
                    response.text[:300],
                )
        except Exception:
            logger.exception("Cashfree payment link creation error")

    return out


async def create_card_checkout(
    order_ref: str,
    amount: float,
    customer_phone: str,
    customer_name: str = "",
) -> dict[str, Any]:
    """Card/UPI checkout URL — Payment Links API or Orders API (see CASHFREE_PAYMENT_MODE)."""
    mode = _cashfree_checkout_mode()

    if mode == "orders":
        return await create_order_checkout(order_ref, amount, customer_phone, customer_name)

    link_out = await create_payment_link(order_ref, amount, customer_phone, customer_name)
    if link_out.get("link_url") or mode == "links":
        return link_out

    logger.info(
        "Payment link unavailable for %s — falling back to Cashfree Orders API",
        order_ref,
    )
    order_out = await create_order_checkout(order_ref, amount, customer_phone, customer_name)
    if order_out.get("link_url"):
        return order_out
    return link_out


def get_card_payment_text(order_ref: str, amount: float, link_url: str) -> str:
    """WhatsApp copy for debit/credit card payment via Cashfree link."""
    currency = os.getenv("CASHFREE_LINK_CURRENCY", "INR").strip().upper() or "INR"
    amount_label = f"₹{amount:,.2f}" if currency == "INR" else f"${amount:,.2f} {currency}"
    return (
        "💳 *Pay by Debit / Credit Card*\n"
        f"Order: {order_ref}\n"
        f"Amount: {amount_label}\n"
        "─────────────────\n"
        "Tap the secure link below to complete payment:\n"
        f"{link_url}\n\n"
        "You can pay with Visa, Mastercard, UPI, or netbanking.\n"
        "_Link expires in 48 hours._\n"
        "We will confirm your payment automatically in this chat."
    )


def get_payment_instructions_text(order: dict, va_details: dict) -> str:
    """Build WhatsApp payment instructions text for buyer."""
    order_ref = order.get("order_ref", "N/A")
    total = float(va_details.get("amount") or order.get("total_amount") or 0)
    account_number = va_details.get("account_number") or "Will be shared shortly"
    ifsc = va_details.get("ifsc") or "Will be shared shortly"
    iban = va_details.get("iban") or "Will be shared shortly"
    swift = va_details.get("swift_code") or "Will be shared shortly"

    return (
        "💳 *Payment Instructions*\n"
        f"Order: {order_ref}\n"
        f"Amount: ${total:,.2f} USD\n"
        "─────────────────\n"
        "Bank Transfer (India):\n"
        f"Account: {account_number}\n"
        f"IFSC: {ifsc}\n"
        f"Reference: {order_ref}\n"
        "─────────────────\n"
        "International Wire:\n"
        f"Account: {iban}\n"
        f"SWIFT: {swift}\n"
        f"Reference: {order_ref}\n\n"
        "Funds reflect in 1–3 business days.\n"
        "Reply with your UTR/reference once transferred."
    )


def _cashfree_hmac_b64(secret: str, message: bytes) -> str:
    return base64.b64encode(
        hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()
    ).decode("utf-8")


def _cashfree_hmac_hex(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify_cashfree_webhook_signature(
    raw_body: bytes,
    *,
    webhook_signature: str | None = None,
    webhook_timestamp: str | None = None,
    legacy_signature: str | None = None,
) -> bool:
    """Verify Cashfree webhook (2023-08-01 / 2025-01-01 and legacy headers)."""
    client_secret = os.getenv("CASHFREE_SECRET_KEY", "").strip()
    webhook_secret = (os.getenv("CASHFREE_WEBHOOK_SECRET", "") or client_secret).strip()
    sig = (webhook_signature or legacy_signature or "").strip()
    ts = (webhook_timestamp or "").strip()

    if not client_secret and not webhook_secret:
        logger.warning("Cashfree secrets not set — skipping webhook signature check")
        return True

    if not sig:
        return False

    secrets = [s for s in (client_secret, webhook_secret) if s]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_secrets: list[str] = []
    for s in secrets:
        if s not in seen:
            seen.add(s)
            unique_secrets.append(s)

    if ts:
        # Official: HMAC-SHA256 base64 over (timestamp + raw body) — no separator.
        signed_std = ts.encode("utf-8") + raw_body
        for secret in unique_secrets:
            if hmac.compare_digest(_cashfree_hmac_b64(secret, signed_std), sig):
                return True
        # Some older samples used timestamp + "." + body; keep for compatibility.
        signed_dot = f"{ts}.".encode("utf-8") + raw_body
        for secret in unique_secrets:
            if hmac.compare_digest(_cashfree_hmac_b64(secret, signed_dot), sig):
                return True

    for secret in unique_secrets:
        if hmac.compare_digest(_cashfree_hmac_hex(secret, raw_body), sig):
            return True
        if hmac.compare_digest(_cashfree_hmac_b64(secret, raw_body), sig):
            return True

    return False


def _notes_order_ref(link_notes: dict | None, link_id: str = "") -> str:
    notes = link_notes or {}
    order_ref = notes.get("order_ref") or ""
    if order_ref:
        return _base_order_ref(str(order_ref))
    if link_id and link_id.upper().startswith("ORD"):
        return _base_order_ref(link_id)
    return ""


def _notes_phone(link_notes: dict | None, customer: dict | None) -> str:
    notes = link_notes or {}
    phone = notes.get("phone") or (customer or {}).get("customer_phone") or ""
    if phone:
        return normalize_phone(str(phone))
    return ""


def _parse_payment_link_event(payload: dict) -> dict[str, Any]:
    """Parse PAYMENT_LINK_EVENT (Cashfree 2025-01-01 schema)."""
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    link_notes = data.get("link_notes") if isinstance(data.get("link_notes"), dict) else {}
    customer = data.get("customer_details") if isinstance(data.get("customer_details"), dict) else {}
    order = data.get("order") if isinstance(data.get("order"), dict) else {}
    payment = data.get("payment") if isinstance(data.get("payment"), dict) else {}

    link_id = str(data.get("link_id") or "")
    order_ref = _notes_order_ref(link_notes, link_id)
    buyer_phone = _notes_phone(link_notes, customer)
    link_status = str(data.get("link_status") or "").upper()
    txn_status = str(order.get("transaction_status") or payment.get("payment_status") or "").upper()

    amount = float(
        order.get("order_amount")
        or data.get("link_amount_paid")
        or data.get("link_amount")
        or payment.get("payment_amount")
        or 0
    )
    payment_id = (
        str(order.get("order_id") or "")
        or str(payment.get("cf_payment_id") or "")
        or str(order.get("transaction_id") or "")
    )

    if link_status == "PAID" or txn_status in {"SUCCESS", "PAID"}:
        payment_status = "payment_received"
    elif link_status in {"EXPIRED"}:
        payment_status = "payment_expired"
    elif link_status in {"CANCELLED"} or txn_status in {"FAILED", "CANCELLED"}:
        payment_status = "payment_failed"
    else:
        payment_status = "awaiting_payment"

    return {
        "event_type": "PAYMENT_LINK_EVENT",
        "order_ref": order_ref,
        "buyer_phone": buyer_phone,
        "payment_status": payment_status,
        "link_status": link_status,
        "amount": amount,
        "payment_id": payment_id,
        "utr": payment.get("bank_reference") or payment.get("utr") or "",
        "virtual_account_id": None,
        "remitter_name": customer.get("customer_name") or "Customer",
        "payment_time": payload.get("event_time") or "",
        "status": "process",
    }


def handle_cashfree_webhook(payload: dict) -> dict[str, Any]:
    """Normalize Cashfree webhook payloads into one internal shape."""
    data = payload or {}
    event_type = str(
        data.get("type") or data.get("event") or data.get("event_type") or ""
    ).strip()

    if event_type == "PAYMENT_LINK_EVENT":
        event = _parse_payment_link_event(data)
        if not event.get("order_ref"):
            event["status"] = "ignored"
        return event

    body = data.get("data") if isinstance(data.get("data"), dict) else data
    link_notes = body.get("link_notes") if isinstance(body.get("link_notes"), dict) else {}
    order_tags = body.get("order_tags") if isinstance(body.get("order_tags"), dict) else {}
    order_block = body.get("order") if isinstance(body.get("order"), dict) else {}

    order_ref = (
        link_notes.get("order_ref")
        or order_tags.get("order_ref")
        or body.get("order_ref")
        or body.get("reference_id")
        or order_block.get("order_id")
        or body.get("order_id")
        or body.get("merchant_order_id")
        or body.get("link_id")
        or body.get("vAccountId")
        or ""
    )
    if not order_ref:
        va_ref = body.get("virtual_account_reference") or body.get("remarks") or ""
        match = re.search(r"ORD-\d{8}-\d{4}", str(va_ref))
        if match:
            order_ref = match.group(0)

    customer = body.get("customer_details") if isinstance(body.get("customer_details"), dict) else {}
    buyer_phone = _notes_phone(link_notes, customer) or str(body.get("phone") or "")

    normalized = {
        "event_type": event_type,
        "order_ref": _base_order_ref(str(order_ref)) if order_ref else "",
        "buyer_phone": normalize_phone(buyer_phone) if buyer_phone else "",
        "virtual_account_id": body.get("virtual_account_id") or body.get("va_id"),
        "payment_id": body.get("payment_id") or body.get("cf_payment_id") or body.get("utr"),
        "utr": body.get("utr") or body.get("bank_reference") or body.get("reference_number"),
        "amount": float(body.get("amount") or body.get("payment_amount") or 0),
        "currency": body.get("currency") or "INR",
        "remitter_name": body.get("remitter_name") or body.get("payer_name") or body.get("payerName") or "Unknown",
        "payment_time": body.get("payment_time") or body.get("paid_at") or "",
        "payment_status": "payment_received",
        "status": "process",
    }

    bank_success = {
        "virtual_account.credited",
        "INTERNATIONAL_PAYMENT_COLLECTED",
        "VIRTUAL_ACCOUNT_CREDITED",
    }
    card_success = {"PAYMENT_SUCCESS_WEBHOOK", "LINK_PAYMENT_RECEIVED", "PAYMENT_SUCCESS"}
    settlement_events = {
        "SETTLEMENT_SUCCESS",
        "SETTLEMENT_INITIATED",
        "SETTLEMENT_FAILED",
        "SETTLEMENT_REVERSED",
    }

    if event_type in settlement_events:
        normalized["payment_status"] = "settlement_update"
        normalized["settlement_status"] = event_type
    elif event_type in bank_success or event_type in card_success:
        normalized["payment_status"] = "payment_received"
    elif event_type in {"PAYMENT_FAILED_WEBHOOK"}:
        normalized["payment_status"] = "payment_failed"
    else:
        normalized["status"] = "ignored"

    if not normalized["order_ref"]:
        normalized["status"] = "ignored"
    return normalized


def buyer_message_for_payment_status(
    payment_status: str,
    *,
    order_ref: str,
    amount: float,
    payment_id: str = "",
) -> str | None:
    currency = os.getenv("CASHFREE_LINK_CURRENCY", "INR").strip().upper() or "INR"
    amount_label = f"₹{amount:,.2f}" if currency == "INR" else f"${amount:,.2f} {currency}"

    if payment_status == "payment_received":
        return (
            "✅ *Payment received!*\n\n"
            f"Order  : {order_ref}\n"
            f"Amount : {amount_label}\n"
            f"Ref    : `{payment_id or 'N/A'}`\n\n"
            "Your order is confirmed and being processed.\n"
            "Settlement to our business account may take 1–2 business days."
        )
    if payment_status == "payment_failed":
        return (
            "❌ *Payment unsuccessful*\n\n"
            f"Order: {order_ref}\n\n"
            "You can try again using the same link, or reply *new order* to start fresh."
        )
    if payment_status == "payment_expired":
        return (
            "⏰ *Payment link expired*\n\n"
            f"Order: {order_ref}\n\n"
            "Reply in this chat and we can generate a new payment link."
        )
    if payment_status == "settlement_update":
        return None
    return None


def _parse_payment_time(raw: str) -> datetime | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


async def process_cashfree_webhook_event(event: dict[str, Any], db: Session) -> None:
    """Apply webhook event to DB and notify buyer + ops team."""
    if event.get("status") == "ignored":
        return

    try:
        payment_status = event.get("payment_status") or "awaiting_payment"
        order_ref = (event.get("order_ref") or "").strip()
        if not order_ref:
            return

        order = (
            db.query(Order)
            .filter(Order.order_ref.ilike(f"{order_ref}%"))
            .order_by(Order.created_at.desc(), Order.id.desc())
            .first()
        )
        if not order:
            logger.warning("Cashfree webhook order not found: %s", event)
            return

        base_ref = (order.order_ref or "").rsplit("-L", 1)[0]
        buyer_phone = event.get("buyer_phone") or order.phone
        amount = float(event.get("amount") or 0)
        payment_id = str(event.get("payment_id") or "")
        utr = str(event.get("utr") or "")
        paid_at = _parse_payment_time(str(event.get("payment_time") or ""))

        if payment_status == "payment_received":
            q = db.query(Order).filter(Order.order_ref.ilike(f"{base_ref}%"))
            q.update(
                {
                    "payment_status": "payment_received",
                    "status": "payment_received",
                    "utr_number": utr or Order.utr_number,
                    "payment_id": payment_id or Order.payment_id,
                    "payment_received_at": paid_at or datetime.utcnow(),
                    "virtual_account_id": event.get("virtual_account_id") or Order.virtual_account_id,
                },
                synchronize_session=False,
            )
            db.commit()
            logger.info("Payment marked received order_ref=%s amount=%s", base_ref, amount)

            buyer_msg = buyer_message_for_payment_status(
                "payment_received",
                order_ref=base_ref,
                amount=amount,
                payment_id=payment_id or utr,
            )
            if buyer_phone and buyer_msg:
                await send_message(buyer_phone, buyer_msg)
            await send_order_team_alert(
                f"💰 Payment confirmed — {base_ref} — {amount:,.2f} "
                f"from {event.get('remitter_name') or 'Customer'} — Ref: {payment_id or utr or 'N/A'}"
            )
            return

        if payment_status in {"payment_failed", "payment_expired"}:
            db.query(Order).filter(Order.order_ref.ilike(f"{base_ref}%")).update(
                {"payment_status": payment_status},
                synchronize_session=False,
            )
            db.commit()
            buyer_msg = buyer_message_for_payment_status(
                payment_status,
                order_ref=base_ref,
                amount=amount,
                payment_id=payment_id,
            )
            if buyer_phone and buyer_msg:
                await send_message(buyer_phone, buyer_msg)
            return

        if payment_status == "settlement_update":
            settlement_status = event.get("settlement_status") or event.get("event_type")
            await send_order_team_alert(
                f"🏦 Settlement update — {base_ref} — {settlement_status} — amount {amount:,.2f}"
            )
    except Exception as exc:
        logger.exception("Cashfree payment processing failed")
        await send_critical_error_alert("Payment processing", str(exc))


async def poll_cashfree_payment_status(order_ref: str) -> dict[str, Any]:
    """Poll Cashfree for order payment status (fallback when webhook missed)."""
    headers = _cashfree_headers()
    base = headers.pop("_base")
    url = f"{base}/pg/orders/{order_ref}/payments"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        try:
            response = await client.get(url, headers=headers)
            if response.status_code >= 400:
                return {"status": "unknown", "order_ref": order_ref}
            rows = response.json()
            if not isinstance(rows, list) or not rows:
                return {"status": "pending", "order_ref": order_ref}
            paid = next(
                (r for r in rows if str(r.get("payment_status", "")).upper() in {"SUCCESS", "PAID"}),
                None,
            )
            if not paid:
                return {"status": "pending", "order_ref": order_ref}
            return {
                "status": "received",
                "order_ref": order_ref,
                "payment_id": paid.get("cf_payment_id") or paid.get("payment_id"),
                "utr": paid.get("bank_reference") or paid.get("utr"),
                "amount": float(paid.get("payment_amount") or 0),
                "currency": paid.get("payment_currency") or "INR",
                "payment_time": paid.get("payment_time") or "",
                "remitter_name": paid.get("payment_message") or "Unknown",
            }
        except Exception:
            logger.exception("Cashfree poll failed for order_ref=%s", order_ref)
            return {"status": "unknown", "order_ref": order_ref}


def _group_base_order_refs(rows: list[Order]) -> list[str]:
    refs: set[str] = set()
    for row in rows:
        if row.order_ref:
            refs.add(_base_order_ref(row.order_ref))
    return sorted(refs)


def _latest_order_for_ref(db: Session, base_ref: str) -> Order | None:
    return (
        db.query(Order)
        .filter(Order.order_ref.ilike(f"{base_ref}%"))
        .order_by(Order.created_at.desc(), Order.id.desc())
        .first()
    )


async def check_overdue_payments() -> None:
    """Check awaiting-payment orders >48h, poll Cashfree, remind buyer if overdue."""
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(hours=OVERDUE_HOURS)
        overdue_rows = (
            db.query(Order)
            .filter(Order.payment_status == "awaiting_payment", Order.created_at <= cutoff)
            .all()
        )
        if not overdue_rows:
            return

        for base_ref in _group_base_order_refs(overdue_rows):
            latest = _latest_order_for_ref(db, base_ref)
            if not latest:
                continue
            polled = await poll_cashfree_payment_status(base_ref)
            if polled.get("status") == "received":
                paid_at = datetime.utcnow()
                q = db.query(Order).filter(Order.order_ref.ilike(f"{base_ref}%"))
                q.update(
                    {
                        "payment_status": "payment_received",
                        "status": "payment_received",
                        "utr_number": polled.get("utr"),
                        "payment_id": polled.get("payment_id"),
                        "payment_received_at": paid_at,
                    },
                    synchronize_session=False,
                )
                db.commit()
                await send_message(
                    latest.phone,
                    f"✅ Payment of ${float(polled.get('amount') or 0):,.2f} received! Order is being processed.",
                )
                await send_order_team_alert(
                    f"💰 Payment confirmed — {base_ref} — ${float(polled.get('amount') or 0):,.2f} "
                    f"from {polled.get('remitter_name') or 'Unknown'} — UTR: {polled.get('utr') or 'N/A'}"
                )
                continue

            # genuinely overdue: reminder + owner alert
            await send_message(
                latest.phone,
                f"Friendly reminder: payment for *{base_ref}* is still pending. "
                "Please share your UTR/reference once transferred.",
            )
            await send_order_team_alert(
                f"⏰ Overdue payment alert — {base_ref} (>{OVERDUE_HOURS}h) still awaiting payment."
            )
    except Exception:
        logger.exception("check_overdue_payments failed")
    finally:
        db.close()


def start_overdue_scheduler() -> None:
    """Start 6-hour recurring overdue-payment checker (best effort)."""
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
    except Exception:
        logger.warning("APScheduler not installed; overdue payment job disabled")
        return

    scheduler = AsyncIOScheduler(timezone=timezone.utc)
    scheduler.add_job(
        lambda: asyncio.create_task(check_overdue_payments()),
        "interval",
        hours=6,
        id="cashfree_overdue_payments",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Cashfree overdue payment scheduler started (interval=6h)")
