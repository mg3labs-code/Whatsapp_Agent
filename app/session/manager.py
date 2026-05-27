import json
import logging
import os

import redis.asyncio as redis

from app.utils.security import user_ref

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL")


def _get_redis_client() -> redis.Redis:
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is not set")
    return redis.from_url(REDIS_URL, decode_responses=True)


async def get_session(phone: str) -> dict:
    key = f"session:{phone}"
    try:
        client = _get_redis_client()
        raw_data = await client.get(key)
        if not raw_data:
            return {"greeted": False}
        data = json.loads(raw_data)
        if isinstance(data, dict):
            return data
        return {"greeted": False}
    except (redis.RedisError, json.JSONDecodeError) as exc:
        # SECURITY: hashed user ref in logs — not raw phone
        logger.exception("Failed to get session user_ref=%s", user_ref(phone))
        return {"greeted": False}


async def save_session(phone: str, data: dict) -> None:
    key = f"session:{phone}"
    try:
        client = _get_redis_client()
        payload = json.dumps(data)
        await client.set(key, payload, ex=86400)
    except (redis.RedisError, TypeError, ValueError) as exc:
        logger.exception("Failed to save session user_ref=%s", user_ref(phone))
        raise RuntimeError(f"Failed to save session user_ref={user_ref(phone)}") from exc


async def delete_session(phone: str) -> None:
    key = f"session:{phone}"
    try:
        client = _get_redis_client()
        await client.delete(key)
    except redis.RedisError as exc:
        logger.exception("Failed to delete session user_ref=%s", user_ref(phone))
        raise RuntimeError(f"Failed to delete session user_ref={user_ref(phone)}") from exc
