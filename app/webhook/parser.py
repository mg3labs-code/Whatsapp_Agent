"""Meta WhatsApp webhook payload parser.

Meta sends many webhook event types to the same endpoint:
- Inbound text messages (handled here)
- Interactive button/list replies (handled here)
- Message status updates (delivered/read/failed)
- Template message notifications
- Account/business profile updates

Only inbound user messages contain entry[].changes[].value.messages[].
Anything else (status updates, template events, etc.) lacks the
"messages" key and should be silently ignored — `parse_meta_payload`
returns None in those cases so the webhook handler can drop them.
"""

from __future__ import annotations


def _extract_message_text(message: dict) -> str | None:
    """Return normalized text for routing, or None if unsupported."""
    msg_type = message.get("type")

    if msg_type == "text":
        return message["text"]["body"]

    if msg_type == "interactive":
        interactive = message.get("interactive") or {}
        interactive_type = interactive.get("type")

        if interactive_type == "button_reply":
            button_reply = interactive.get("button_reply") or {}
            return button_reply.get("id") or button_reply.get("title")

        if interactive_type == "list_reply":
            list_reply = interactive.get("list_reply") or {}
            return (
                list_reply.get("id")
                or list_reply.get("title")
                or list_reply.get("description")
            )

    return None


def parse_meta_payload(payload: dict) -> dict | None:
    """Extract phone, text, message_id from a Meta Cloud API webhook payload.

    Returns:
        {"phone": str, "text": str, "message_id": str} for inbound text or
        interactive button_reply / list_reply messages.
        None for any non-message webhook (status updates, template events, etc.)
        or malformed payloads. Callers should treat None as "ignore silently".
    """
    try:
        message = payload["entry"][0]["changes"][0]["value"]["messages"][0]
        text = _extract_message_text(message)
        if not text:
            return None
        return {
            "phone": message["from"],
            "text": text,
            "message_id": message["id"],
        }
    except (KeyError, IndexError, TypeError):
        return None
