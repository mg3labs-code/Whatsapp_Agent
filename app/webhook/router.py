import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, BackgroundTasks, Request, Response
from fastapi.responses import PlainTextResponse

from app.orchestrator.graph import compiled_graph
from app.session.manager import _get_redis_client, get_session
from app.utils.tracing import flush_langfuse, message_trace_context
from app.webhook.parser import parse_meta_payload

DEDUP_TTL_SECONDS = 86400
LOCK_TTL_SECONDS = 30
LOCK_RETRY_COUNT = 10
LOCK_RETRY_DELAY_SECONDS = 0.1

logger = logging.getLogger(__name__)

webhook_router = APIRouter()


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
        payload = await request.json()
    except Exception:
        logger.warning("Webhook JSON parse failed")
        return Response(status_code=200)

    background_tasks.add_task(process_message, payload)
    return Response(status_code=200)


async def process_message(payload: dict) -> None:
    """Background pipeline for a single inbound webhook payload."""
    try:
        parsed = parse_meta_payload(payload)
        if parsed is None:
            return

        phone = parsed["phone"]
        text = parsed["text"]
        message_id = parsed["message_id"]

        if await _is_duplicate(message_id):
            logger.info("Dropping duplicate message_id=%s", message_id)
            return

        lock_key = f"wasa:lock:{phone}"
        async with _phone_lock(lock_key, ttl=LOCK_TTL_SECONDS) as acquired:
            if not acquired:
                logger.warning(
                    "Could not acquire lock for phone; skipping message_id=%s",
                    message_id,
                )
                return

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
                await compiled_graph.ainvoke(state)
    except Exception:
        # SECURITY: log message_id / user_ref only — not message body or raw phone
        logger.exception("Orchestrator processing failed")
    finally:
        flush_langfuse()


def _dedup_key(message_id: str) -> str:
    return f"wasa:msgid:{message_id}"


async def _is_duplicate(message_id: str) -> bool:
    """Atomic dedup via SET NX — one key per message_id with its own TTL.

    Meta retries webhook deliveries when it doesn't get a 200 fast enough,
    so the same message_id can arrive multiple times concurrently.

    Returns True if message_id was already processed; False (and records it) if new.
    """
    try:
        client = _get_redis_client()
        was_new = await client.set(
            _dedup_key(message_id),
            "1",
            ex=DEDUP_TTL_SECONDS,
            nx=True,
        )
        return was_new is None
    except Exception:
        logger.exception("Dedup check failed for message_id=%s; processing anyway", message_id)
        return False


@asynccontextmanager
async def _phone_lock(key: str, ttl: int = LOCK_TTL_SECONDS):
    """Per-phone Redis lock — at most one pipeline active per phone at a time."""
    client = _get_redis_client()
    acquired = False
    for _ in range(LOCK_RETRY_COUNT):
        acquired = await client.set(key, "1", ex=ttl, nx=True)
        if acquired:
            break
        await asyncio.sleep(LOCK_RETRY_DELAY_SECONDS)
    try:
        yield acquired
    finally:
        if acquired:
            await client.delete(key)
