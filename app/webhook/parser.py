"""Meta WhatsApp webhook payload parser.

Meta sends many webhook event types to the same endpoint:
- Inbound text messages (handled here)
- Interactive button/list replies (handled here)
- Message status updates (delivered/read/failed)
- Template message notifications
- Account/business profile updates

Only inbound user messages contain entry[].changes[].value.messages[].
Anything else (status updates, template events, etc.) lacks the
"messages" key and should be silently ignored — parsers return empty/None
so the webhook handler can drop them.
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


def _message_timestamp(message: dict) -> int:
    try:
        return int(message.get("timestamp") or 0)
    except (TypeError, ValueError):
        return 0


def parse_meta_messages(payload: dict) -> list[dict]:
    """Extract every inbound text / interactive message in a Meta payload.

    Returns items oldest-first by WhatsApp ``timestamp``, then ``message_id``.
    Empty list for status-only, malformed, or unsupported payloads.
    """
    results: list[dict] = []
    try:
        for entry in payload.get("entry") or []:
            for change in entry.get("changes") or []:
                value = (change or {}).get("value") or {}
                for message in value.get("messages") or []:
                    if not isinstance(message, dict):
                        continue
                    text = _extract_message_text(message)
                    phone = message.get("from")
                    message_id = message.get("id")
                    if not text or not phone or not message_id:
                        continue
                    results.append(
                        {
                            "phone": phone,
                            "text": text,
                            "message_id": message_id,
                            "timestamp": _message_timestamp(message),
                        }
                    )
    except (AttributeError, TypeError):
        return []

    results.sort(key=lambda item: (item["timestamp"], item["message_id"]))
    return results


def parse_meta_payload(payload: dict) -> dict | None:
    """First (oldest) inbound message, or None if the payload has none.

    Prefer ``parse_meta_messages`` when a payload may contain several messages.
    """
    items = parse_meta_messages(payload)
    return items[0] if items else None
