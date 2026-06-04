import base64
import hashlib
import hmac
import json

import pytest

from app.integrations.cashfree import (
    handle_cashfree_webhook,
    verify_cashfree_webhook_signature,
)


def test_verify_webhook_signature_2025_format(monkeypatch):
    secret = "test_secret_key"
    monkeypatch.setenv("CASHFREE_SECRET_KEY", secret)
    monkeypatch.delenv("CASHFREE_WEBHOOK_SECRET", raising=False)

    body = b'{"type":"PAYMENT_LINK_EVENT","data":{"link_id":"ORD-20250603-1234"}}'
    timestamp = "1617695238078"
    signed = timestamp.encode() + body
    signature = base64.b64encode(
        hmac.new(secret.encode(), signed, hashlib.sha256).digest()
    ).decode()

    assert verify_cashfree_webhook_signature(
        body,
        webhook_signature=signature,
        webhook_timestamp=timestamp,
    )


def test_parse_payment_link_event_paid():
    payload = {
        "type": "PAYMENT_LINK_EVENT",
        "data": {
            "link_id": "ORD-20250603-1234",
            "link_status": "PAID",
            "link_amount": "115.00",
            "link_amount_paid": "115.00",
            "link_notes": {
                "order_ref": "ORD-20250603-1234",
                "phone": "919876543210",
            },
            "customer_details": {"customer_phone": "919876543210", "customer_name": "Jane"},
            "order": {
                "order_amount": "115.00",
                "order_id": "cf_order_abc",
                "transaction_status": "SUCCESS",
            },
        },
    }
    event = handle_cashfree_webhook(payload)
    assert event["status"] == "process"
    assert event["payment_status"] == "payment_received"
    assert event["order_ref"] == "ORD-20250603-1234"
    assert event["buyer_phone"] == "+919876543210"
    assert event["payment_id"] == "cf_order_abc"


def test_verify_webhook_signature_legacy_hex(monkeypatch):
    secret = "webhook_only_secret"
    monkeypatch.setenv("CASHFREE_SECRET_KEY", secret)
    monkeypatch.delenv("CASHFREE_WEBHOOK_SECRET", raising=False)

    body = b'{"type":"PAYMENT_SUCCESS_WEBHOOK","data":{}}'
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    assert verify_cashfree_webhook_signature(body, legacy_signature=digest)


def test_parse_payment_link_event_expired():
    payload = {
        "type": "PAYMENT_LINK_EVENT",
        "data": {
            "link_id": "ORD-20250603-9999",
            "link_status": "EXPIRED",
            "link_notes": {"order_ref": "ORD-20250603-9999", "phone": "91999"},
        },
    }
    event = handle_cashfree_webhook(payload)
    assert event["payment_status"] == "payment_expired"
