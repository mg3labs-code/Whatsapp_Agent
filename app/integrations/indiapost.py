"""India Post external integrations — auth, tracking, webhooks (Phase 1)."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.db.models import Order
from app.integrations.whatsapp import send_message

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 15.0
TRACKING_NUMBER_RE = re.compile(r"\b([A-Z]{2}\d{9}[A-Z]{2})\b", re.IGNORECASE)

_token_cache: dict[str, Any] = {"access_token": "", "expires_at": 0.0}
_token_lock = asyncio.Lock()


def is_indiapost_configured() -> bool:
    return bool(
        os.getenv("INDIAPOST_USERNAME", "").strip()
        and os.getenv("INDIAPOST_PASSWORD", "").strip()
    )


def _api_base() -> str:
    explicit = os.getenv("INDIAPOST_API_BASE", "").strip().rstrip("/")
    if explicit:
        return explicit
    env = os.getenv("INDIAPOST_ENV", "sandbox").strip().lower()
    if env == "production":
        return "https://cept.gov.in/beextcustomer"
    return "https://test.cept.gov.in/beextcustomer"


def extract_tracking_number(text: str) -> str | None:
    match = TRACKING_NUMBER_RE.search((text or "").upper())
    return match.group(1).upper() if match else None


async def _login() -> str:
    username = os.getenv("INDIAPOST_USERNAME", "").strip()
    password = os.getenv("INDIAPOST_PASSWORD", "").strip()
    url = f"{_api_base()}/v1/access/login"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(
            url,
            json={"username": username, "password": password},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        if response.status_code >= 400:
            logger.warning(
                "India Post login failed: HTTP %s body=%s",
                response.status_code,
                response.text[:300],
            )
            return ""
        payload = response.json()
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        token = str(data.get("access_token") or "")
        expires_in = int(data.get("expires_in") or 900)
        if not token:
            logger.warning("India Post login returned no access_token")
            return ""
        _token_cache["access_token"] = token
        _token_cache["expires_at"] = time.time() + max(60, expires_in - 60)
        return token


async def get_access_token() -> str:
    if not is_indiapost_configured():
        return ""
    if _token_cache["access_token"] and time.time() < float(_token_cache["expires_at"]):
        return str(_token_cache["access_token"])
    async with _token_lock:
        if _token_cache["access_token"] and time.time() < float(_token_cache["expires_at"]):
            return str(_token_cache["access_token"])
        return await _login()


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


async def track_bulk(tracking_numbers: list[str]) -> dict[str, Any]:
    """Track up to 50 consignments via India Post bulk API."""
    numbers = [n.upper() for n in tracking_numbers if n][:50]
    if not numbers:
        return {"success": False, "data": []}

    token = await get_access_token()
    if not token:
        return {"success": False, "data": [], "error": "auth_failed"}

    url = f"{_api_base()}/v1/tracking/bulk"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        try:
            response = await client.post(
                url,
                headers=_auth_headers(token),
                json={"bulk": numbers},
            )
            if response.status_code == 401:
                _token_cache["expires_at"] = 0
                token = await get_access_token()
                if token:
                    response = await client.post(
                        url,
                        headers=_auth_headers(token),
                        json={"bulk": numbers},
                    )
            if response.status_code >= 400:
                logger.warning(
                    "India Post bulk tracking failed: HTTP %s body=%s",
                    response.status_code,
                    response.text[:300],
                )
                return {"success": False, "data": [], "error": "api_error"}
            return response.json()
        except Exception:
            logger.exception("India Post bulk tracking error")
            return {"success": False, "data": [], "error": "network_error"}


async def track_single(tracking_number: str) -> dict[str, Any] | None:
    """Track one consignment via GET /v1/tracking/{trackingNumber}."""
    number = (tracking_number or "").strip().upper()
    if not number:
        return None

    token = await get_access_token()
    if not token:
        return None

    url = f"{_api_base()}/v1/tracking/{number}"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        try:
            response = await client.get(url, headers=_auth_headers(token))
            if response.status_code == 401:
                _token_cache["expires_at"] = 0
                token = await get_access_token()
                if token:
                    response = await client.get(url, headers=_auth_headers(token))
            if response.status_code >= 400:
                logger.warning(
                    "India Post tracking failed for %s: HTTP %s",
                    number,
                    response.status_code,
                )
                return None
            payload = response.json()
            if isinstance(payload.get("data"), dict):
                return payload["data"]
            return payload if isinstance(payload, dict) else None
        except Exception:
            logger.exception("India Post single tracking error for %s", number)
            return None


def _delivery_status(row: dict) -> str:
    del_status = row.get("del_status")
    if isinstance(del_status, dict):
        return str(del_status.get("del_status") or "").strip().lower()
    booking = row.get("booking_details") if isinstance(row.get("booking_details"), dict) else {}
    events = row.get("tracking_details") if isinstance(row.get("tracking_details"), list) else []
    if events:
        last_event = str(events[-1].get("event") or "").lower()
        if "delivered" in last_event:
            return "delivered"
    current = str(row.get("currentStatus") or row.get("current_status") or "").lower()
    if current:
        return current
    if booking.get("delivery_confirmed_on"):
        return "delivered"
    return "in_transit"


def format_tracking_message(
    tracking_number: str,
    row: dict[str, Any],
    *,
    order_ref: str = "",
) -> str:
    """Build WhatsApp-friendly tracking summary from bulk or single API row."""
    booking = row.get("booking_details") if isinstance(row.get("booking_details"), dict) else {}
    events = row.get("tracking_details") if isinstance(row.get("tracking_details"), list) else []
    history = row.get("history") if isinstance(row.get("history"), list) else []

    article = str(
        booking.get("article_number")
        or row.get("trackingNumber")
        or row.get("tracking_number")
        or tracking_number
    ).upper()
    status = _delivery_status(row)
    origin = booking.get("origin_pincode") or row.get("origin") or ""
    destination = (
        booking.get("destination_pincode")
        or row.get("destination")
        or booking.get("delivery_location")
        or ""
    )
    eta = str(row.get("estimatedDelivery") or row.get("estimated_delivery") or "").strip()

    lines = ["📦 *Shipment tracking*"]
    if order_ref:
        lines.append(f"Order: {order_ref}")
    lines.append(f"AWB: {article}")
    lines.append(f"Status: {status.replace('_', ' ').title() or 'In transit'}")
    if origin or destination:
        lines.append(f"Route: {origin or '?'} → {destination or '?'}")
    if eta:
        lines.append(f"Est. delivery: {eta}")

    timeline = events or history
    if timeline:
        latest = timeline[-1]
        when = latest.get("date") or latest.get("timestamp") or ""
        where = latest.get("office") or latest.get("location") or ""
        event = latest.get("event") or latest.get("status") or ""
        if event:
            lines.append("─────────────────")
            lines.append(f"Latest: {event}")
            if where:
                lines.append(f"At: {where}")
            if when:
                lines.append(f"When: {when}")

    return "\n".join(lines)


async def lookup_tracking_message(
    tracking_number: str,
    *,
    order_ref: str = "",
) -> str | None:
    """Fetch India Post tracking and return WhatsApp text, or None if unavailable."""
    if not is_indiapost_configured():
        return None

    bulk = await track_bulk([tracking_number])
    rows = bulk.get("data") if isinstance(bulk.get("data"), list) else []
    if rows and isinstance(rows[0], dict):
        return format_tracking_message(tracking_number, rows[0], order_ref=order_ref)

    single = await track_single(tracking_number)
    if single:
        return format_tracking_message(tracking_number, single, order_ref=order_ref)
    return None


def _map_event_to_payment_status(event_code: str, event_description: str) -> str | None:
    code = (event_code or "").upper()
    desc = (event_description or "").lower()
    if code in {"ID", "ITEM_DELIVERED"} or "delivered" in desc:
        return "delivered"
    if code in {"ITEM_BOOK", "IB"} or "booked" in desc:
        return "processing"
    if any(k in desc for k in ("dispatch", "transit", "received", "bag")):
        return "shipped"
    return None


async def process_indiapost_webhook_event(payload: dict[str, Any], db: Session) -> None:
    """Apply India Post tracking webhook — update order and notify buyer."""
    article = str(payload.get("article_number") or "").strip().upper()
    if not article:
        logger.info("India Post webhook ignored — no article_number")
        return

    order = (
        db.query(Order)
        .filter(Order.tracking_number == article)
        .order_by(Order.created_at.desc())
        .first()
    )
    if not order:
        logger.info("India Post webhook: no order for AWB %s", article)
        return

    new_status = _map_event_to_payment_status(
        str(payload.get("event_code") or ""),
        str(payload.get("event_description") or ""),
    )
    if new_status == "delivered":
        order.payment_status = "delivered"
        order.status = "delivered"
    elif new_status == "shipped":
        order.payment_status = "shipped"
        order.status = "shipped"
    elif new_status == "processing":
        order.status = "processing"
    db.commit()

    phone = order.phone
    if not phone:
        return

    event_desc = str(payload.get("event_description") or payload.get("event_code") or "Update")
    order_ref = (order.order_ref or "").split("-L")[0]
    msg = (
        f"📦 *Shipment update — {order_ref or article}*\n"
        f"AWB: {article}\n"
        f"Status: {event_desc}\n"
        f"Location: {payload.get('event_office_name') or payload.get('destination_city') or '—'}"
    )
    await send_message(phone, msg)
