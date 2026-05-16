import asyncio
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from sqlalchemy import text

from app.db.database import create_tables, engine
from app.db.migrate import is_railway_production, run_alembic_upgrade_head
from app.utils.tracing import flush_langfuse
from app.webhook.router import webhook_router

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="WASA - WhatsApp AI Sales Agent", version="1.0.0")

_MIGRATION_TIMEOUT = 60  # seconds
_TABLE_CREATION_TIMEOUT = 30  # seconds


@app.on_event("startup")
async def startup_event() -> None:
    loop = asyncio.get_event_loop()
    try:
        if is_railway_production():
            logger.info("Running Alembic migrations (timeout: %ds)", _MIGRATION_TIMEOUT)
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, run_alembic_upgrade_head),
                    timeout=_MIGRATION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Alembic migrations timed out after %ds — database may be "
                    "unreachable or a migration is stuck",
                    _MIGRATION_TIMEOUT,
                )
                raise RuntimeError(
                    f"Alembic migrations did not complete within {_MIGRATION_TIMEOUT}s"
                )
        else:
            logger.info("Creating tables (timeout: %ds)", _TABLE_CREATION_TIMEOUT)
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, create_tables),
                    timeout=_TABLE_CREATION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Table creation timed out after %ds — database may be unreachable",
                    _TABLE_CREATION_TIMEOUT,
                )
                raise RuntimeError(
                    f"Table creation did not complete within {_TABLE_CREATION_TIMEOUT}s"
                )

        def _check_connection() -> None:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))

        await loop.run_in_executor(None, _check_connection)
        logger.info("DB connected")
    except RuntimeError:
        raise
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
