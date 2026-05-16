"""Meta WhatsApp Cloud API sender.

Sends outbound text messages via Meta Graph API v18. Implements a single
retry on HTTP 429 (rate limit) and is defensive — never raises to the
caller, so a transient Meta outage cannot crash the webhook background task.
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from app.utils.security import user_ref

logger = logging.getLogger(__name__)

META_API_VERSION = "v18.0"
META_GRAPH_BASE = "https://graph.facebook.com"
REQUEST_TIMEOUT_SECONDS = 10.0
RATE_LIMIT_RETRY_DELAY_SECONDS = 2


async def send_message(phone: str, text: str) -> bool:
    """Send a WhatsApp text message via Meta Cloud API.

    Args:
        phone: Buyer's phone number. Leading '+' is stripped automatically.
        text: Message body. WhatsApp formatting (*bold*, _italic_) supported.

    Returns:
        True on 2xx response, False otherwise. Never raises.
    """
    token = os.getenv("WHATSAPP_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

    if not token or not phone_number_id:
        logger.error("WhatsApp credentials missing; cannot send message")
        return False

    normalized_phone = phone.lstrip("+")
    url = f"{META_GRAPH_BASE}/{META_API_VERSION}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "messaging_product": "whatsapp",
        "to": normalized_phone,
        "type": "text",
        "text": {"body": text},
    }

    try:
        # SECURITY: bounded HTTP timeout on all Meta API calls
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=body)
            logger.info(
                "WhatsApp send user_ref=%s: HTTP %s",
                user_ref(normalized_phone),
                response.status_code,
            )

            if response.status_code == 429:
                await asyncio.sleep(RATE_LIMIT_RETRY_DELAY_SECONDS)
                response = await client.post(url, headers=headers, json=body)
                logger.info(
                    "WhatsApp retry user_ref=%s: HTTP %s",
                    user_ref(normalized_phone),
                    response.status_code,
                )

            return 200 <= response.status_code < 300
    except Exception:
        # SECURITY: hashed user ref in logs — not raw phone
        logger.exception("WhatsApp send failed user_ref=%s", user_ref(normalized_phone))
        return False
