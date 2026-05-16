"""Security helpers — safe references for logs and traces."""

from __future__ import annotations

from app.utils.tracing import hash_user_id


def user_ref(phone: str | None) -> str:
    """Hashed user reference safe for application logs (never raw phone)."""
    return hash_user_id(phone or "")
