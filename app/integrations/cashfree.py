"""Payment helpers: export wire instructions and overdue tracking."""

from __future__ import annotations

import asyncio
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
from app.integrations.alerts import send_order_team_alert
from app.integrations.whatsapp import send_message

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


EXPORT_WIRE_ENV = {
    "account_name": "STATIC_WIRE_ACCOUNT_NAME",
    "account_number": "STATIC_WIRE_ACCOUNT_NUMBER",
    "bank_name": "STATIC_WIRE_BANK_NAME",
    "branch": "STATIC_WIRE_BRANCH",
    "swift_code": "STATIC_WIRE_SWIFT_CODE",
    "ifsc": "STATIC_WIRE_IFSC",
}

REQUIRED_EXPORT_WIRE_FIELDS = (
    "account_name",
    "account_number",
    "bank_name",
    "branch",
    "swift_code",
)

_SWIFT_PATTERN = re.compile(r"^[A-Z0-9]{8}([A-Z0-9]{3})?$")
_IFSC_PATTERN = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")


def load_export_wire_details() -> dict[str, str]:
    """Read export wire details from env (evaluated on each call)."""
    return {
        field: (os.getenv(env_key, "") or "").strip()
        for field, env_key in EXPORT_WIRE_ENV.items()
    }


def missing_export_wire_fields(details: dict[str, str] | None = None) -> list[str]:
    """Return required field names that are empty."""
    data = details if details is not None else load_export_wire_details()
    return [field for field in REQUIRED_EXPORT_WIRE_FIELDS if not data.get(field)]


def validate_export_wire_details(details: dict[str, str] | None = None) -> list[str]:
    """Return human-readable validation errors for configured wire details."""
    data = details if details is not None else load_export_wire_details()
    errors: list[str] = []

    swift = (data.get("swift_code") or "").upper()
    if swift and not _SWIFT_PATTERN.match(swift):
        errors.append("swift_code must be 8 or 11 alphanumeric characters")

    ifsc = (data.get("ifsc") or "").upper()
    if ifsc and not _IFSC_PATTERN.match(ifsc):
        errors.append("ifsc must be an 11-character IFSC (e.g. HDFC0001234)")

    account_number = data.get("account_number") or ""
    if account_number and not re.fullmatch(r"\d{6,18}", account_number):
        errors.append("account_number must be 6–18 digits")

    return errors


def is_export_wire_configured() -> bool:
    """True when all required export wire env vars are set and pass validation."""
    details = load_export_wire_details()
    return not missing_export_wire_fields(details) and not validate_export_wire_details(details)


def get_static_payment_details_text(
    order_ref: str, amount: float, currency: str
) -> str:
    """Build export wire instructions for the buyer from environment variables."""
    currency_code = (currency or "").strip().upper()
    if currency_code == "USD":
        symbol = "$"
    elif currency_code == "INR":
        symbol = "₹"
    else:
        raise ValueError(f"Unsupported currency: {currency_code!r}")

    details = load_export_wire_details()
    missing = missing_export_wire_fields(details)
    validation_errors = validate_export_wire_details(details)
    if missing or validation_errors:
        logger.warning(
            "Export wire details unavailable order_ref=%s missing=%s validation=%s",
            order_ref,
            missing,
            validation_errors,
        )
        return (
            "*Payment details (International wire)*\n"
            f"Amount: {symbol}{amount:,.2f} {currency_code}\n"
            f"Reference: {order_ref}\n\n"
            "Our team will share international wire transfer details with you directly.\n"
            "Our team will confirm receipt manually within 24 hours."
        )

    lines = [
        "*Payment details (International wire)*",
        f"Amount: {symbol}{amount:,.2f} {currency_code}",
        f"Beneficiary: {details['account_name']}",
        f"Account: {details['account_number']}",
        f"Bank: {details['bank_name']}",
        f"Branch: {details['branch']}",
        f"SWIFT: {details['swift_code'].upper()}",
    ]
    if details.get("ifsc"):
        lines.append(f"IFSC: {details['ifsc'].upper()}")
    lines.extend(
        [
            f"Reference: {order_ref}",
            "",
            "Please mention the reference exactly as above.",
            "Our team will confirm receipt manually within 24 hours.",
        ]
    )
    return "\n".join(lines)


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
    """Remind buyers and alert ops for awaiting-payment orders older than the cutoff."""
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
