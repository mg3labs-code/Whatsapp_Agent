"""Alembic migration helpers for production deploy."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.db.database import engine

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_railway_production() -> bool:
    return bool(os.getenv("RAILWAY_ENVIRONMENT"))


def run_alembic_upgrade_head() -> None:
    """Apply all pending Alembic revisions (uses DATABASE_URL from env)."""
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is not set; cannot run migrations")

    cfg = Config(str(_PROJECT_ROOT / "alembic.ini"))
    logger.info("Running Alembic upgrade head (Railway production)")
    # Release app pool connections so Alembic can acquire migration locks.
    engine.dispose()
    os.environ["ALEMBIC_SKIP_FILE_CONFIG"] = "1"
    try:
        command.upgrade(cfg, "head")
    finally:
        os.environ.pop("ALEMBIC_SKIP_FILE_CONFIG", None)
    logger.info("Alembic migrations complete")
