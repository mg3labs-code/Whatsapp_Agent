import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from sqlalchemy import text

from app.db.database import engine
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


app.include_router(webhook_router, prefix="")
