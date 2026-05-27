"""Cashfree helpers: VA creation, webhook normalization, overdue tracking."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import Order
from app.integrations.alerts import send_order_team_alert
from app.integrations.whatsapp import send_message

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 12.0
OVERDUE_HOURS = 48


def _cashfree_headers() -> dict[str, str]:
    app_id = os.getenv("CASHFREE_APP_ID", "")
    secret = os.getenv("CASHFREE_SECRET_KEY", "")
    env = os.getenv("CASHFREE_ENV", "sandbox").strip().lower()
    base = (
        "https://sandbox.cashfree.com"
        if env != "production"
        else "https://api.cashfree.com"
    )
    return {
        "x-client-id": app_id,
        "x-client-secret": secret,
        "x-api-version": "2023-08-01",
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


def handle_cashfree_webhook(payload: dict) -> dict[str, Any]:
    """Normalize domestic/international Cashfree webhook payload into one shape."""
    data = payload or {}
    event_type = (
        str(data.get("event") or data.get("event_type") or data.get("type") or "").strip()
    )
    body = data.get("data") if isinstance(data.get("data"), dict) else data

    order_ref = (
        body.get("order_ref")
        or body.get("reference_id")
        or body.get("order_id")
        or body.get("merchant_order_id")
        or ""
    )
    if not order_ref:
        va_ref = body.get("virtual_account_reference") or body.get("remarks") or ""
        match = re.search(r"ORD-\d{8}-\d{4}", str(va_ref))
        if match:
            order_ref = match.group(0)

    normalized = {
        "event_type": event_type,
        "order_ref": _base_order_ref(str(order_ref)) if order_ref else "",
        "virtual_account_id": body.get("virtual_account_id") or body.get("va_id"),
        "payment_id": body.get("payment_id") or body.get("cf_payment_id"),
        "utr": body.get("utr") or body.get("bank_reference") or body.get("reference_number"),
        "amount": float(body.get("amount") or body.get("payment_amount") or 0),
        "currency": body.get("currency") or "INR",
        "remitter_name": body.get("remitter_name") or body.get("payer_name") or "Unknown",
        "payment_time": body.get("payment_time") or body.get("paid_at") or "",
        "status": "received",
    }

    if event_type not in {"virtual_account.credited", "INTERNATIONAL_PAYMENT_COLLECTED"}:
        normalized["status"] = "ignored"
    return normalized


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
