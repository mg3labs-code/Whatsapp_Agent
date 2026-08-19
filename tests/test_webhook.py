import time
from contextlib import nullcontext
from unittest.mock import AsyncMock

import fakeredis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.orchestrator import graph as orchestrator_graph
from app.session import manager as session_manager
from app.webhook import router as webhook_router_module
from app.webhook.parser import parse_meta_messages, parse_meta_payload
from app.webhook.router import webhook_router


META_TEXT_MESSAGE_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "1234567890",
            "changes": [
                {
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {
                            "display_phone_number": "15551234567",
                            "phone_number_id": "PHONE_NUMBER_ID",
                        },
                        "contacts": [
                            {
                                "profile": {"name": "Test Buyer"},
                                "wa_id": "919876543210",
                            }
                        ],
                        "messages": [
                            {
                                "from": "919876543210",
                                "id": "wamid.TEST_MSG_ID_123",
                                "timestamp": "1715500000",
                                "text": {"body": "Hi, I need price for Amoxicillin 500mg"},
                                "type": "text",
                            }
                        ],
                    },
                }
            ],
        }
    ],
}


META_STATUS_UPDATE_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "1234567890",
            "changes": [
                {
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {
                            "display_phone_number": "15551234567",
                            "phone_number_id": "PHONE_NUMBER_ID",
                        },
                        "statuses": [
                            {
                                "id": "wamid.STATUS_ID_456",
                                "status": "delivered",
                                "timestamp": "1715500001",
                                "recipient_id": "919876543210",
                            }
                        ],
                    },
                }
            ],
        }
    ],
}


META_BUTTON_REPLY_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "1234567890",
            "changes": [
                {
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {
                            "display_phone_number": "15551234567",
                            "phone_number_id": "PHONE_NUMBER_ID",
                        },
                        "contacts": [
                            {
                                "profile": {"name": "Test Buyer"},
                                "wa_id": "919876543210",
                            }
                        ],
                        "messages": [
                            {
                                "from": "919876543210",
                                "id": "wamid.BUTTON_REPLY_789",
                                "timestamp": "1715500002",
                                "type": "interactive",
                                "interactive": {
                                    "type": "button_reply",
                                    "button_reply": {
                                        "id": "order",
                                        "title": "Order Medicines",
                                    },
                                },
                            }
                        ],
                    },
                }
            ],
        }
    ],
}


def test_parse_text_message():
    result = parse_meta_payload(META_TEXT_MESSAGE_PAYLOAD)
    assert result is not None
    assert result["phone"] == "919876543210"
    assert result["text"] == "Hi, I need price for Amoxicillin 500mg"
    assert result["message_id"] == "wamid.TEST_MSG_ID_123"


META_LIST_REPLY_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "1234567890",
            "changes": [
                {
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "messages": [
                            {
                                "from": "919876543210",
                                "id": "wamid.LIST_REPLY_321",
                                "timestamp": "1715500003",
                                "type": "interactive",
                                "interactive": {
                                    "type": "list_reply",
                                    "list_reply": {
                                        "id": "pricing",
                                        "title": "Get Pricing",
                                    },
                                },
                            }
                        ],
                    },
                }
            ],
        }
    ],
}


def test_parse_list_reply_uses_row_id_as_text():
    result = parse_meta_payload(META_LIST_REPLY_PAYLOAD)
    assert result is not None
    assert result["text"] == "pricing"
    assert result["message_id"] == "wamid.LIST_REPLY_321"


def test_parse_button_reply_uses_button_id_as_text():
    result = parse_meta_payload(META_BUTTON_REPLY_PAYLOAD)
    assert result is not None
    assert result["phone"] == "919876543210"
    assert result["text"] == "order"
    assert result["message_id"] == "wamid.BUTTON_REPLY_789"


def test_parse_status_update_returns_none():
    result = parse_meta_payload(META_STATUS_UPDATE_PAYLOAD)
    assert result is None


def test_parse_empty_payload_returns_none():
    assert parse_meta_payload({}) is None


def test_parse_malformed_payload_returns_none():
    assert parse_meta_payload({"entry": [{"changes": []}]}) is None


def test_parse_meta_messages_orders_by_timestamp():
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.SECOND",
                                    "timestamp": "1715500002",
                                    "text": {"body": "100"},
                                    "type": "text",
                                },
                                {
                                    "from": "919876543210",
                                    "id": "wamid.FIRST",
                                    "timestamp": "1715500001",
                                    "text": {"body": "Metformin 500mg"},
                                    "type": "text",
                                },
                            ]
                        }
                    }
                ]
            }
        ],
    }
    items = parse_meta_messages(payload)
    assert [item["text"] for item in items] == ["Metformin 500mg", "100"]
    assert [item["message_id"] for item in items] == ["wamid.FIRST", "wamid.SECOND"]
    first = parse_meta_payload(payload)
    assert first is not None
    assert first["message_id"] == "wamid.FIRST"


def _build_test_app() -> FastAPI:
    """Minimal app with only the webhook router — no DB startup event."""
    app = FastAPI()
    app.include_router(webhook_router)
    return app


def _patch_redis_and_orchestrator(monkeypatch) -> tuple[AsyncMock, object]:
    """Shared setup: fake redis + mocked orchestrator. Returns (ainvoke mock, redis)."""
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(session_manager, "_get_redis_client", lambda: fake_redis)
    monkeypatch.setattr(webhook_router_module, "_get_redis_client", lambda: fake_redis)
    monkeypatch.setattr(webhook_router_module, "flush_langfuse", lambda: None)
    monkeypatch.setattr(
        webhook_router_module,
        "message_trace_context",
        lambda **_kwargs: nullcontext(),
    )
    # Unit tests focus on routing/dedup, not HMAC — accept unsigned test payloads.
    monkeypatch.setattr(
        webhook_router_module,
        "_verify_meta_signature",
        lambda _raw, _sig: True,
    )

    mock_invoke = AsyncMock(return_value={})
    monkeypatch.setattr(orchestrator_graph.compiled_graph, "ainvoke", mock_invoke)
    return mock_invoke, fake_redis


def test_get_webhook_verify_succeeds_with_correct_token(monkeypatch):
    monkeypatch.setenv("WEBHOOK_VERIFY_TOKEN", "secret-token-123")
    client = TestClient(_build_test_app())

    response = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "secret-token-123",
            "hub.challenge": "1234567890",
        },
    )

    assert response.status_code == 200
    assert response.text == "1234567890"
    assert response.headers["content-type"].startswith("text/plain")


def test_get_webhook_verify_fails_with_wrong_token(monkeypatch):
    monkeypatch.setenv("WEBHOOK_VERIFY_TOKEN", "secret-token-123")
    client = TestClient(_build_test_app())

    response = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "WRONG-TOKEN",
            "hub.challenge": "1234567890",
        },
    )

    assert response.status_code == 403


def test_post_webhook_returns_200_quickly(monkeypatch):
    _patch_redis_and_orchestrator(monkeypatch)
    client = TestClient(_build_test_app())

    start = time.perf_counter()
    response = client.post("/webhook", json=META_TEXT_MESSAGE_PAYLOAD)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert response.status_code == 200
    assert elapsed_ms < 100, f"Webhook took {elapsed_ms:.1f}ms (>100ms)"


def test_post_webhook_invalid_json_returns_200(monkeypatch):
    """SECURITY: Meta always gets 200 even when JSON body is invalid."""
    _patch_redis_and_orchestrator(monkeypatch)
    client = TestClient(_build_test_app())

    response = client.post(
        "/webhook",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200


def test_dedup_drops_duplicate(monkeypatch):
    mock_invoke, _redis = _patch_redis_and_orchestrator(monkeypatch)
    client = TestClient(_build_test_app())

    r1 = client.post("/webhook", json=META_TEXT_MESSAGE_PAYLOAD)
    r2 = client.post("/webhook", json=META_TEXT_MESSAGE_PAYLOAD)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert mock_invoke.call_count == 1


def test_batched_messages_processed_oldest_first(monkeypatch):
    mock_invoke, _redis = _patch_redis_and_orchestrator(monkeypatch)
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": "wamid.LATER",
                                    "timestamp": "1715500099",
                                    "text": {"body": "checkout"},
                                    "type": "text",
                                },
                                {
                                    "from": "919876543210",
                                    "id": "wamid.EARLIER",
                                    "timestamp": "1715500001",
                                    "text": {"body": "Metformin 500mg - 100"},
                                    "type": "text",
                                },
                            ]
                        }
                    }
                ]
            }
        ],
    }
    client = TestClient(_build_test_app())
    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert mock_invoke.call_count == 2
    texts = [call.args[0]["message"] for call in mock_invoke.call_args_list]
    assert texts == ["Metformin 500mg - 100", "checkout"]


def test_lock_busy_queues_then_drains(monkeypatch):
    import asyncio

    monkeypatch.setattr(webhook_router_module, "LOCK_RETRY_COUNT", 1)
    monkeypatch.setattr(webhook_router_module, "LOCK_RETRY_DELAY_SECONDS", 0)
    mock_invoke, fake_redis = _patch_redis_and_orchestrator(monkeypatch)
    phone = "919876543210"
    asyncio.run(fake_redis.set(f"wasa:lock:{phone}", "1"))

    client = TestClient(_build_test_app())
    response = client.post("/webhook", json=META_TEXT_MESSAGE_PAYLOAD)
    assert response.status_code == 200
    assert mock_invoke.call_count == 0
    queued = asyncio.run(fake_redis.llen(f"wasa:inbox:{phone}"))
    assert queued == 1

    asyncio.run(fake_redis.delete(f"wasa:lock:{phone}"))
    asyncio.run(webhook_router_module._drain_phone_inbox(phone, fake_redis))
    assert mock_invoke.call_count == 1
    assert mock_invoke.call_args.args[0]["message_id"] == "wamid.TEST_MSG_ID_123"


def test_post_webhook_rejects_unsigned_payload(monkeypatch):
    """SECURITY: missing/invalid Meta signature → 200 but no orchestrator work."""
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(session_manager, "_get_redis_client", lambda: fake_redis)
    monkeypatch.setattr(webhook_router_module, "_get_redis_client", lambda: fake_redis)
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "test-app-secret")

    mock_invoke = AsyncMock(return_value={})
    monkeypatch.setattr(orchestrator_graph.compiled_graph, "ainvoke", mock_invoke)

    client = TestClient(_build_test_app())
    response = client.post("/webhook", json=META_TEXT_MESSAGE_PAYLOAD)

    assert response.status_code == 200
    assert mock_invoke.call_count == 0


def test_indiapost_webhook_disabled_for_release():
    """Inbound India Post webhooks are off for v1 — never process payloads."""
    client = TestClient(_build_test_app())
    response = client.post(
        "/webhook/indiapost",
        json={"article_number": "EB123"},
        headers={"Authorization": "Bearer anything"},
    )
    assert response.status_code == 404
