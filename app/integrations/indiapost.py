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


def _shipment_status_label(status: str) -> str:
    normalized = (status or "").strip().lower().replace("_", " ")
    if normalized in {"not delivered", "not_delivered"}:
        return "Not delivered — in transit"
    if normalized == "delivered":
        return "Delivered ✅"
    if normalized in {"in transit", "intransit"}:
        return "In transit"
    return normalized.title() or "In transit"


def parse_tracking_summary(tracking_number: str, row: dict[str, Any]) -> dict[str, Any]:
    """Extract structured shipment fields from an India Post API row."""
    booking = row.get("booking_details") if isinstance(row.get("booking_details"), dict) else {}
    events = row.get("tracking_details") if isinstance(row.get("tracking_details"), list) else []
    history = row.get("history") if isinstance(row.get("history"), list) else []
    timeline = events or history

    article = str(
        booking.get("article_number")
        or row.get("trackingNumber")
        or row.get("tracking_number")
        or tracking_number
    ).upper()
    status = _delivery_status(row)
    latest_event = ""
    latest_when = ""
    latest_where = ""
    if timeline:
        latest = timeline[-1]
        latest_event = str(latest.get("event") or latest.get("status") or "").strip()
        latest_when = str(latest.get("date") or latest.get("timestamp") or "").strip()
        latest_where = str(latest.get("office") or latest.get("location") or "").strip()

    location = (
        latest_where
        or str(booking.get("delivery_location") or "").strip()
        or str(row.get("event_office_name") or "").strip()
        or str(row.get("destination") or "").strip()
    )
    origin = str(booking.get("origin_pincode") or row.get("origin") or "").strip()
    destination = str(
        booking.get("destination_pincode")
        or booking.get("delivery_location")
        or row.get("destination")
        or ""
    ).strip()
    eta = str(row.get("estimatedDelivery") or row.get("estimated_delivery") or "").strip()

    return {
        "awb": article,
        "status": status,
        "status_label": _shipment_status_label(status),
        "location": location,
        "latest_event": latest_event,
        "latest_when": latest_when,
        "origin": origin,
        "destination": destination,
        "eta": eta,
        "is_delivered": status == "delivered" or "delivered" in latest_event.lower(),
    }


def format_tracking_message(
    tracking_number: str,
    row: dict[str, Any],
    *,
    order_ref: str = "",
) -> str:
    """Build WhatsApp-friendly shipment block (AWB + status + location first)."""
    summary = parse_tracking_summary(tracking_number, row)
    lines = ["📦 *Shipment tracking*"]
    lines.append(f"AWB: {summary['awb']}")
    lines.append(f"Status: {summary['status_label']}")
    if summary["location"]:
        lines.append(f"📍 Location: {summary['location']}")
    if summary["origin"] or summary["destination"]:
        lines.append(f"Route: {summary['origin'] or '?'} → {summary['destination'] or '?'}")
    if summary["eta"]:
        lines.append(f"Est. delivery: {summary['eta']}")
    if summary["latest_event"]:
        lines.append("─────────────────")
        lines.append(f"Latest update: {summary['latest_event']}")
        if summary["latest_when"]:
            lines.append(f"When: {summary['latest_when']}")
    return "\n".join(lines)


async def _fetch_tracking_row(tracking_number: str) -> dict[str, Any] | None:
    bulk = await track_bulk([tracking_number])
    rows = bulk.get("data") if isinstance(bulk.get("data"), list) else []
    if rows and isinstance(rows[0], dict):
        return rows[0]
    return await track_single(tracking_number)


async def lookup_tracking_message(
    tracking_number: str,
    *,
    order_ref: str = "",
) -> str | None:
    """Fetch India Post tracking and return WhatsApp text, or None if unavailable."""
    _ = order_ref  # order summary is composed separately in order agent
    if not is_indiapost_configured():
        return None
    row = await _fetch_tracking_row(tracking_number)
    if row:
        return format_tracking_message(tracking_number, row)
    return None


async def lookup_tracking_summary(tracking_number: str) -> dict[str, Any] | None:
    """Fetch structured India Post tracking fields, or None if unavailable."""
    if not is_indiapost_configured():
        return None
    row = await _fetch_tracking_row(tracking_number)
    if row:
        return parse_tracking_summary(tracking_number, row)
    return None


async def fetch_tracking_bundle(
    tracking_number: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """One API round-trip — returns (whatsapp_text, structured_summary)."""
    if not is_indiapost_configured():
        return None, None
    row = await _fetch_tracking_row(tracking_number)
    if not row:
        return None, None
    summary = parse_tracking_summary(tracking_number, row)
    return format_tracking_message(tracking_number, row), summary


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
