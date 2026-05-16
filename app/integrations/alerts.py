"""Team alerts via WhatsApp DM — role-based recipient lists (Option A)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from app.integrations.whatsapp import send_message
from app.utils.security import user_ref

logger = logging.getLogger(__name__)

# Primary env keys (comma-separated E.164 numbers)
LEADS_ALERT_ENV = "LEADS_ALERT_PHONE_NUMBERS"
ORDER_ALERT_ENV = "ORDER_ALERT_PHONE_NUMBERS"
# Legacy fallback for leads/escalations only
LEGACY_ESCALATION_ENV = "ESCALATION_PHONE_NUMBERS"


def _parse_phone_list(*env_keys: str) -> list[str]:
    """First non-empty env var wins; values are comma-separated phone numbers."""
    for key in env_keys:
        raw = os.getenv(key, "")
        phones: list[str] = []
        for part in raw.split(","):
            part = part.strip()
            if part:
                phones.append(part)
        if phones:
            return phones
    return []


def leads_alert_recipients() -> list[str]:
    """Sales / support — escalations and lead-related alerts."""
    return _parse_phone_list(LEADS_ALERT_ENV, LEGACY_ESCALATION_ENV)


def order_alert_recipients() -> list[str]:
    """Export / order desk — new order notifications only."""
    return _parse_phone_list(ORDER_ALERT_ENV)


async def _send_to_recipients(phones: list[str], text: str, *, list_name: str) -> bool:
    """Send text to each number. Never raises."""
    if not phones:
        logger.warning("%s not set; team alert skipped", list_name)
        return False

    try:
        all_ok = True
        for phone in phones:
            ok = await send_message(phone, text)
            if not ok:
                all_ok = False
                # SECURITY: hashed recipient ref in logs
                logger.error(
                    "Team alert failed for %s recipient_ref=%s",
                    list_name,
                    user_ref(phone),
                )
        return all_ok
    except Exception:
        logger.exception("send_to_recipients failed (%s)", list_name)
        return False


async def send_leads_alert(text: str) -> bool:
    """Escalations / lead alerts → LEADS_ALERT_PHONE_NUMBERS (fallback: ESCALATION_PHONE_NUMBERS)."""
    return await _send_to_recipients(
        leads_alert_recipients(),
        text,
        list_name=LEADS_ALERT_ENV,
    )


async def send_order_team_alert(text: str) -> bool:
    """New orders → ORDER_ALERT_PHONE_NUMBERS only."""
    return await _send_to_recipients(
        order_alert_recipients(),
        text,
        list_name=ORDER_ALERT_ENV,
    )


async def send_escalation_alert(phone: str, session: dict, reason: str) -> bool:
    session = session or {}
    message = (
        "🚨 *ESCALATION ALERT*\n"
        f"Phone: {phone}\n"
        f"Company: {session.get('company', 'Unknown')}\n"
        f"Country: {session.get('country', 'Unknown')}\n"
        f"Score: {session.get('lead_score', 'N/A')}\n"
        f"Category: {session.get('lead_category', 'N/A')}\n"
        f"Reason: {reason}\n"
        f"Time: {datetime.now(timezone.utc).isoformat()}"
    )
    return await send_leads_alert(message)


async def send_order_alert(order: dict) -> bool:
    order = order or {}
    message = (
        "📦 *NEW ORDER*\n"
        f"Ref: {order.get('order_ref', 'N/A')}\n"
        f"Product: {order.get('product_name', 'N/A')}\n"
        f"Qty: {order.get('quantity', 'N/A')}\n"
        f"Ship to: {order.get('city', '')}, {order.get('country', '')}\n"
        f"Contact: {order.get('contact_name', 'N/A')}\n"
        f"Phone: {order.get('phone', 'N/A')}"
    )
    return await send_order_team_alert(message)
