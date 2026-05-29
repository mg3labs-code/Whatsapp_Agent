import asyncio
from datetime import datetime
import hashlib
import hmac
import logging
import os

from fastapi import APIRouter, BackgroundTasks, Request, Response
from fastapi.responses import PlainTextResponse

from app.db.database import get_db
from app.db.models import Order
from app.integrations.alerts import send_order_team_alert
from app.integrations.cashfree import handle_cashfree_webhook
from app.integrations.whatsapp import send_message
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


def _verify_cashfree_signature(raw_body: bytes, signature: str) -> bool:
    secret = os.getenv("CASHFREE_WEBHOOK_SECRET", "")
    if not secret or not signature:
        return False
    digest = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature.strip())


def _parse_payment_time(raw: str) -> datetime | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@webhook_router.post("/webhook/cashfree")
async def receive_cashfree_webhook(request: Request) -> Response:
    raw = await request.body()
    signature = request.headers.get("X-Cashfree-Signature", "")
    if not _verify_cashfree_signature(raw, signature):
        return Response(status_code=401)

    try:
        payload = await request.json()
    except Exception:
        return Response(status_code=400)

    event = handle_cashfree_webhook(payload)
    if event.get("status") == "ignored":
        return Response(status_code=200)

    db_gen = get_db()
    db = next(db_gen)
    try:
        order = None
        va_id = event.get("virtual_account_id")
        order_ref = (event.get("order_ref") or "").strip()
        if va_id:
            order = (
                db.query(Order)
                .filter(Order.virtual_account_id == va_id)
                .order_by(Order.created_at.desc(), Order.id.desc())
                .first()
            )
        if not order and order_ref:
            order = (
                db.query(Order)
                .filter(Order.order_ref.ilike(f"{order_ref}%"))
                .order_by(Order.created_at.desc(), Order.id.desc())
                .first()
            )
        if not order:
            logger.warning("Cashfree webhook order not found: %s", event)
            return Response(status_code=200)

        base_ref = (order.order_ref or "").rsplit("-L", 1)[0]
        amount = float(event.get("amount") or 0)
        remitter = event.get("remitter_name") or "Unknown"
        utr = event.get("utr") or ""
        payment_time = event.get("payment_time") or ""
        paid_at = _parse_payment_time(str(payment_time))

        q = db.query(Order).filter(Order.order_ref.ilike(f"{base_ref}%"))
        q.update(
            {
                "payment_status": "payment_received",
                "status": "payment_received",
                "utr_number": utr,
                "payment_id": event.get("payment_id"),
                "payment_received_at": paid_at,
                "virtual_account_id": va_id or Order.virtual_account_id,
            },
            synchronize_session=False,
        )
        db.commit()

        await send_message(
            order.phone,
            f"✅ Payment of ${amount:,.2f} received! Order is being processed.",
        )
        await send_order_team_alert(
            f"💰 Payment confirmed — {base_ref} — ${amount:,.2f} from {remitter} — UTR: {utr or 'N/A'}"
        )
        return Response(status_code=200)
    except Exception:
        logger.exception("Cashfree webhook processing failed")
        return Response(status_code=200)
    finally:
        db_gen.close()


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
