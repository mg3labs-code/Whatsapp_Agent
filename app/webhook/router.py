import logging
import os

from fastapi import APIRouter, BackgroundTasks, Request, Response
from fastapi.responses import PlainTextResponse

from app.orchestrator.graph import compiled_graph
from app.session.manager import _get_redis_client, get_session
from app.utils.tracing import flush_langfuse, message_trace_context
from app.webhook.parser import parse_meta_payload

DEDUP_SET_KEY = "wasa:processed_ids"
DEDUP_TTL_SECONDS = 86400

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


async def _is_duplicate(message_id: str) -> bool:
    """Redis-backed dedup against the wasa:processed_ids set.

    Meta retries webhook deliveries when it doesn't get a 200 fast enough,
    so the same message_id can arrive multiple times. We track seen IDs
    in a single Redis set with a 24h sliding TTL.

    Returns True if message_id was already processed; False (and records it) if new.
    """
    try:
        client = _get_redis_client()
        already_seen = await client.sismember(DEDUP_SET_KEY, message_id)
        if already_seen:
            return True
        await client.sadd(DEDUP_SET_KEY, message_id)
        await client.expire(DEDUP_SET_KEY, DEDUP_TTL_SECONDS)
        return False
    except Exception:
        logger.exception("Dedup check failed for message_id=%s; processing anyway", message_id)
        return False
