import pytest

from app.integrations.indiapost import (
    extract_tracking_number,
    format_tracking_message,
    is_indiapost_configured,
)


def test_extract_tracking_number():
    assert extract_tracking_number("Track EB126023474IN please") == "EB126023474IN"
    assert extract_tracking_number("no awb here") is None


def test_format_tracking_message_bulk_shape():
    row = {
        "booking_details": {
            "article_number": "EB126023474IN",
            "origin_pincode": "560001",
            "destination_pincode": "560040",
            "delivery_location": "Vijayanagar S.O",
        },
        "tracking_details": [
            {"date": "2025-08-13", "office": "Vijayanagar S.O", "event": "Item Delivered"},
        ],
        "del_status": {"del_status": "delivered"},
    }
    msg = format_tracking_message("EB126023474IN", row, order_ref="ORD-20260604-4720")
    assert "EB126023474IN" in msg
    assert "Delivered" in msg
    assert "ORD-20260604-4720" in msg


def test_is_indiapost_configured(monkeypatch):
    monkeypatch.delenv("INDIAPOST_USERNAME", raising=False)
    monkeypatch.delenv("INDIAPOST_PASSWORD", raising=False)
    assert is_indiapost_configured() is False
    monkeypatch.setenv("INDIAPOST_USERNAME", "user")
    monkeypatch.setenv("INDIAPOST_PASSWORD", "pass")
    assert is_indiapost_configured() is True
