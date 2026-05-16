"""Meta WhatsApp webhook payload parser.

Meta sends many webhook event types to the same endpoint:
- Inbound text messages (handled here)
- Message status updates (delivered/read/failed)
- Template message notifications
- Account/business profile updates

Only inbound text messages contain entry[].changes[].value.messages[].
Anything else (status updates, template events, etc.) lacks the
"messages" key and should be silently ignored — `parse_meta_payload`
returns None in those cases so the webhook handler can drop them.
"""

from __future__ import annotations


def parse_meta_payload(payload: dict) -> dict | None:
    """Extract phone, text, message_id from a Meta Cloud API webhook payload.

    Returns:
        {"phone": str, "text": str, "message_id": str} for inbound text messages.
        None for any non-message webhook (status updates, template events, etc.)
        or malformed payloads. Callers should treat None as "ignore silently".
    """
    try:
        message = payload["entry"][0]["changes"][0]["value"]["messages"][0]
        return {
            "phone": message["from"],
            "text": message["text"]["body"],
            "message_id": message["id"],
        }
    except (KeyError, IndexError, TypeError):
        return None
