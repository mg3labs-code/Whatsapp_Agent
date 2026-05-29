import json
import logging
import os

import redis.asyncio as redis

from app.utils.security import user_ref

logger = logging.getLogger(__name__)


def normalize_phone(phone: str) -> str:
    """Canonical session key (Meta sends digits without '+')."""
    return (phone or "").lstrip("+").strip()


def _redis_url() -> str:
    """Read at call time so dotenv in main.py is applied before first use."""
    for env_key in ("REDIS_URL", "REDIS_PRIVATE_URL", "REDISCLOUD_URL"):
        value = (os.getenv(env_key) or "").strip()
        if value:
            return value
    return ""


def redis_configured() -> bool:
    return bool(_redis_url())


def _get_redis_client() -> redis.Redis:
    url = _redis_url()
    if not url:
        raise RuntimeError(
            "REDIS_URL is not set (set REDIS_URL or REDIS_PRIVATE_URL on the app service)"
        )
    kwargs: dict = {"decode_responses": True}
    if url.startswith("rediss://"):
        kwargs["ssl_cert_reqs"] = None
    return redis.from_url(url, **kwargs)


async def ping_redis() -> bool:
    """Return True if Redis accepts PING."""
    try:
        client = _get_redis_client()
        return (await client.ping()) is True
    except Exception:
        logger.exception("Redis ping failed")
        return False


async def redis_key_stats() -> dict:
    """Counts keys by prefix — safe for /health/redis (no values)."""
    client = _get_redis_client()
    session_keys: list[str] = []
    wasa_keys: list[str] = []
    async for key in client.scan_iter(match="session:*", count=200):
        session_keys.append(key)
        if len(session_keys) >= 20:
            break
    async for key in client.scan_iter(match="wasa:*", count=200):
        wasa_keys.append(key)
        if len(wasa_keys) >= 20:
            break
    db_size = await client.dbsize()
    return {
        "db_size": db_size,
        "session_key_count": len(session_keys),
        "wasa_key_count": len(wasa_keys),
        "session_key_samples": session_keys[:5],
        "wasa_key_samples": wasa_keys[:5],
    }


async def get_session(phone: str) -> dict:
    key = f"session:{normalize_phone(phone)}"
    try:
        client = _get_redis_client()
        raw_data = await client.get(key)
        if not raw_data:
            return {"greeted": False}
        data = json.loads(raw_data)
        if isinstance(data, dict):
            data["phone"] = normalize_phone(phone)
            return data
        return {"greeted": False}
    except (redis.RedisError, json.JSONDecodeError):
        logger.exception("Failed to get session user_ref=%s", user_ref(phone))
        return {"greeted": False}


async def save_session(phone: str, data: dict) -> None:
    key = f"session:{normalize_phone(phone)}"
    try:
        client = _get_redis_client()
        payload = dict(data or {})
        payload["phone"] = normalize_phone(phone)
        await client.set(key, json.dumps(payload), ex=86400)
        logger.info(
            "Session saved key=%s user_ref=%s qual_state=%s lead_qualified=%s",
            key,
            user_ref(phone),
            payload.get("qual_state"),
            payload.get("lead_qualified"),
        )
    except (redis.RedisError, TypeError, ValueError):
        logger.exception("Failed to save session user_ref=%s", user_ref(phone))
        raise RuntimeError(f"Failed to save session user_ref={user_ref(phone)}") from None


async def delete_session(phone: str) -> None:
    key = f"session:{normalize_phone(phone)}"
    try:
        client = _get_redis_client()
        await client.delete(key)
    except redis.RedisError:
        logger.exception("Failed to delete session user_ref=%s", user_ref(phone))
        raise RuntimeError(f"Failed to delete session user_ref={user_ref(phone)}") from None
