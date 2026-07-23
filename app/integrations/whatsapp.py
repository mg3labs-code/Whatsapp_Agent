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
MAX_WHATSAPP_TEXT_LEN = 4000


def _split_whatsapp_text(text: str, max_len: int = MAX_WHATSAPP_TEXT_LEN) -> list[str]:
    """Split long text at newlines before max_len; hard-split if no newline fits."""
    chunks: list[str] = []
    rest = text
    while len(rest) > max_len:
        newline_at = rest.rfind("\n", 0, max_len)
        if newline_at > 0:
            chunks.append(rest[:newline_at])
            rest = rest[newline_at + 1 :]
        else:
            chunks.append(rest[:max_len])
            rest = rest[max_len:]
    if rest:
        chunks.append(rest)
    return chunks


async def send_message(phone: str, text: str) -> bool:
    """Send a WhatsApp text message via Meta Cloud API.

    Args:
        phone: Buyer's phone number. Leading '+' is stripped automatically.
        text: Message body. WhatsApp formatting (*bold*, _italic_) supported.

    Returns:
        True on 2xx response, False otherwise. Never raises.
    """
    if len(text) > MAX_WHATSAPP_TEXT_LEN:
        chunks = _split_whatsapp_text(text)
        normalized_phone = phone.lstrip("+")
        logger.warning(
            "WhatsApp message split into %s chunks (len=%s) user_ref=%s",
            len(chunks),
            len(text),
            user_ref(normalized_phone),
        )
        for chunk in chunks:
            if not await send_message(phone, chunk):
                return False
        return True

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


async def _post_whatsapp_payload(phone: str, payload: dict) -> bool:
    """Shared POST helper for WhatsApp Cloud API message payloads."""
    token = os.getenv("WHATSAPP_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

    if not token or not phone_number_id:
        logger.warning("WhatsApp credentials not set; skipping interactive send")
        return False

    normalized_phone = phone.lstrip("+")
    url = f"{META_GRAPH_BASE}/{META_API_VERSION}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {**payload, "to": normalized_phone}

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return True
    except Exception:
        logger.exception(
            "WhatsApp interactive send failed user_ref=%s",
            user_ref(normalized_phone),
        )
        return False


async def send_interactive_list(
    phone: str,
    *,
    header_text: str,
    body_text: str,
    footer_text: str,
    button_text: str,
    rows: list[dict],
    section_title: str = "Services",
) -> bool:
    """Send a WhatsApp list message (View Options style, up to 10 rows)."""
    list_rows = [
        {
            "id": row["id"],
            "title": row["title"][:24],
            **(
                {"description": row["description"][:72]}
                if row.get("description")
                else {}
            ),
        }
        for row in rows[:10]
    ]
    payload = {
        "messaging_product": "whatsapp",
        "type": "interactive",
        "interactive": {
            "type": "list",
            "header": {"type": "text", "text": header_text[:60]},
            "body": {"text": body_text[:1024]},
            "footer": {"text": footer_text[:60]},
            "action": {
                "button": button_text[:20],
                "sections": [
                    {
                        "title": section_title[:24],
                        "rows": list_rows,
                    }
                ],
            },
        },
    }
    return await _post_whatsapp_payload(phone, payload)


async def send_interactive_buttons(
    phone: str,
    body_text: str,
    buttons: list[dict],
) -> bool:
    """Send a WhatsApp interactive message with up to 3 quick reply buttons.

    Args:
        phone: Buyer's phone number. Leading '+' is stripped automatically.
        body_text: Message body shown above the buttons.
        buttons: List of {"id": str, "title": str} (max 3; titles truncated to 20 chars).

    Returns:
        True on 2xx response, False otherwise. Never raises.
    """
    payload = {
        "messaging_product": "whatsapp",
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text[:1024]},
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {
                            "id": btn["id"],
                            "title": btn["title"][:20],
                        },
                    }
                    for btn in buttons[:3]
                ]
            },
        },
    }
    return await _post_whatsapp_payload(phone, payload)


async def send_navigation_footer(phone: str) -> bool:
    """Send 'Anything else?' with a Main Menu quick-reply button."""
    from app.messages.conversation_ui import MAIN_MENU_BUTTON, NAV_FOOTER_BODY

    return await send_interactive_buttons(phone, NAV_FOOTER_BODY, MAIN_MENU_BUTTON)
