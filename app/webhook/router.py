import asyncio
import hashlib
import hmac
import json
import logging
import os

from fastapi import APIRouter, BackgroundTasks, Request, Response
from fastapi.responses import PlainTextResponse

from app.db.database import get_db
from app.integrations.indiapost import process_indiapost_webhook_event
from app.orchestrator.graph import compiled_graph
from app.session.manager import _get_redis_client, get_session, save_session
from app.utils.security import user_ref
from app.utils.tracing import flush_langfuse, message_trace_context
from app.webhook.parser import parse_meta_payload

DEDUP_TTL_SECONDS = 86400
LOCK_TTL_SECONDS = 30
LOCK_RETRY_COUNT = 10
LOCK_RETRY_DELAY_SECONDS = 0.1

logger = logging.getLogger(__name__)

webhook_router = APIRouter()


def _verify_meta_signature(raw_body: bytes, signature_header: str) -> bool:
    """Verify Meta X-Hub-Signature-256 (HMAC-SHA256 over raw request body)."""
    app_secret = os.getenv("WHATSAPP_APP_SECRET", "").strip()
    if not app_secret or not signature_header:
        return False

    header = signature_header.strip()
    if not header.startswith("sha256="):
        return False

    expected_sig = header[len("sha256=") :]
    computed_sig = hmac.new(
        app_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(computed_sig, expected_sig)


@webhook_router.get("/webhook")
async def verify_webhook(request: Request) -> Response:
    """Meta webhook verification handshake.

    Meta sends a GET with hub.mode=subscribe, hub.verify_token, and hub.challenge.
    We must echo back the challenge as PLAIN TEXT (not JSON) when the token matches.
    """
    mode = request.query_params.get("hub.mode")
    verify_token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    expected_token = os.getenv("WEBHOOK_VERIFY_TOKEN")
    if mode == "subscribe" and verify_token and verify_token == expected_token:
        return PlainTextResponse(str(challenge))
    return Response(status_code=403)


@webhook_router.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks) -> Response:
    """Inbound message receiver.

    CRITICAL: Returns HTTP 200 INSTANTLY before any processing.
    All real work happens in a background task so Meta never sees latency
    (and never retries due to a slow response).
    """
    # SECURITY: always 200 to Meta — never expose parse/validation errors in HTTP status
    try:
        raw = await request.body()
    except Exception:
        logger.warning("Webhook body read failed")
        return Response(status_code=200)

    signature_header = (
        request.headers.get("x-hub-signature-256")
        or request.headers.get("X-Hub-Signature-256")
        or ""
    )
    if not _verify_meta_signature(raw, signature_header):
        logger.warning(
            "Meta webhook signature mismatch — ignoring payload "
            "(has_sig=%s, body_len=%s)",
            bool(signature_header),
            len(raw),
        )
        return Response(status_code=200)

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        logger.warning("Webhook JSON parse failed")
        return Response(status_code=200)

    background_tasks.add_task(process_message, payload)
    return Response(status_code=200)


@webhook_router.post("/webhook/cashfree")
async def receive_cashfree_webhook() -> Response:
    """Acknowledge legacy Cashfree webhook retries — integration removed."""
    logger.info("Cashfree webhook received but integration has been removed")
    return Response(status_code=200)


async def _process_indiapost_payload(payload: dict) -> None:
    db_gen = get_db()
    db = next(db_gen)
    try:
        await process_indiapost_webhook_event(payload, db)
    except Exception:
        logger.exception("India Post webhook processing failed")
    finally:
        db_gen.close()


@webhook_router.post("/webhook/indiapost")
async def receive_indiapost_webhook(
    request: Request, background_tasks: BackgroundTasks
) -> Response:
    """India Post shipment tracking event webhooks (register URL with India Post)."""
    try:
        payload = await request.json()
    except Exception:
        logger.warning("India Post webhook JSON parse failed")
        return Response(status_code=200)

    if not isinstance(payload, dict):
        return Response(status_code=200)

    background_tasks.add_task(_process_indiapost_payload, payload)
    return Response(status_code=200)


async def _is_duplicate(message_id: str, client) -> bool:
    """Atomic dedup via SET NX — returns True if message_id was already seen."""
    key = f"wasa:msgid:{message_id}"
    try:
        was_new = await client.set(key, "1", ex=DEDUP_TTL_SECONDS, nx=True)
        return was_new is None  # None means key existed = duplicate
    except Exception:
        logger.exception("Dedup check failed for message_id=%s; processing anyway", message_id)
        return False


async def process_message(payload: dict) -> None:
    """Background pipeline for a single inbound webhook payload."""
    try:
        parsed = parse_meta_payload(payload)
        if parsed is None:
            return

        from app.session.manager import normalize_phone

        phone = normalize_phone(parsed["phone"])
        text = parsed["text"]
        message_id = parsed["message_id"]

        from app.utils.request_context import set_request_id
        from app.utils.tracing import hash_user_id

        set_request_id(hash_user_id(phone), message_id)

        client = _get_redis_client()

        if await _is_duplicate(message_id, client):
            logger.info("Dropping duplicate message_id=%s", message_id)
            return

        lock_key = f"wasa:lock:{phone}"
        acquired = False
        for _ in range(LOCK_RETRY_COUNT):
            acquired = await client.set(lock_key, "1", ex=LOCK_TTL_SECONDS, nx=True)
            if acquired:
                break
            await asyncio.sleep(LOCK_RETRY_DELAY_SECONDS)

        if not acquired:
            logger.warning(
                "Could not acquire lock for phone; skipping message_id=%s",
                message_id,
            )
            return

        try:
            session = await get_session(phone)

            state = {
                "phone": phone,
                "message": text,
                "message_id": message_id,
                "session": session,
                "intent": None,
                "agent_response": None,
                "guardrail_blocked": False,
                "final_reply": None,
            }

            with message_trace_context(
                trace_name="whatsapp_message",
                phone=phone,
                message_id=message_id,
                feature="orchestrator",
            ):
                result = await compiled_graph.ainvoke(state)
            updated = (result or {}).get("session")
            if updated:
                try:
                    await save_session(phone, updated)
                except Exception:
                    logger.exception(
                        "Backup session save failed user_ref=%s message_id=%s",
                        user_ref(phone),
                        message_id,
                    )
        finally:
            await client.delete(lock_key)
    except Exception:
        logger.exception("process_message failed")
    finally:
        flush_langfuse()
