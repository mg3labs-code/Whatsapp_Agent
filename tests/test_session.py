import fakeredis
import pytest

from app.session import manager


@pytest.mark.asyncio
async def test_save_and_get_session(monkeypatch):
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(manager, "_get_redis_client", lambda: fake_redis)

    await manager.save_session("1234567890", {"company": "TestCo"})
    data = await manager.get_session("1234567890")

    assert data == {"company": "TestCo"}


@pytest.mark.asyncio
async def test_empty_session(monkeypatch):
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(manager, "_get_redis_client", lambda: fake_redis)

    data = await manager.get_session("0000000000")

    assert data == {}


@pytest.mark.asyncio
async def test_delete_session(monkeypatch):
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(manager, "_get_redis_client", lambda: fake_redis)

    await manager.save_session("5555555555", {"company": "DeleteMe"})
    await manager.delete_session("5555555555")
    data = await manager.get_session("5555555555")

    assert data == {}
