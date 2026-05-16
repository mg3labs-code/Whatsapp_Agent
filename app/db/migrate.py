"""Alembic migration helpers for production deploy."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config

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
    command.upgrade(cfg, "head")
    logger.info("Alembic migrations complete")
