import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from sqlalchemy import text

from app.db.database import engine
from app.integrations.cashfree import start_overdue_scheduler
from app.session.manager import ping_redis, redis_configured, redis_key_stats
from app.utils.tracing import flush_langfuse
from app.webhook.router import webhook_router

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="WASA - WhatsApp AI Sales Agent", version="1.0.0")


@app.on_event("startup")
async def startup_event() -> None:
    """Verify DB connectivity. Run schema migrations manually (alembic upgrade head)."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("DB connected")
        if redis_configured():
            if await ping_redis():
                stats = await redis_key_stats()
                logger.info("Redis connected db_size=%s session_keys=%s", stats["db_size"], stats["session_key_count"])
            else:
                logger.error("Redis ping failed — sessions will not persist (check REDIS_URL on app service)")
        else:
            logger.error("REDIS_URL not set on app service — sessions will not persist")
        start_overdue_scheduler()
    except Exception as exc:
        # SECURITY: do not expose connection strings or env details in error messages
        logger.exception("Database startup check failed")
        raise RuntimeError("Database startup check failed") from exc


@app.on_event("shutdown")
async def shutdown_event() -> None:
    flush_langfuse()


@app.get("/health")
async def health() -> dict[str, str]:
    # SECURITY: no env vars, DB URLs, or internal connection details
    return {"status": "ok", "version": "1.0.0"}


@app.get("/health/redis")
async def health_redis() -> dict:
    """Diagnose Redis connectivity and whether session keys exist (no secret values)."""
    if not redis_configured():
        return {
            "status": "misconfigured",
            "redis_configured": False,
            "hint": "Set REDIS_URL on the WASA app service to ${{Redis.REDIS_URL}} from Railway.",
        }
    if not await ping_redis():
        return {
            "status": "unreachable",
            "redis_configured": True,
            "ping_ok": False,
            "hint": "App cannot reach Redis. Use the private REDIS_URL from the Redis plugin.",
        }
    stats = await redis_key_stats()
    return {
        "status": "ok",
        "redis_configured": True,
        "ping_ok": True,
        **stats,
        "key_patterns": {
            "sessions": "session:{phone} — TTL 24h, written after each bot reply",
            "dedup": "wasa:msgid:{id} — TTL 24h, one per WhatsApp message",
            "locks": "wasa:lock:{phone} — TTL 30s, deleted after each message",
        },
    }


app.include_router(webhook_router, prefix="")
