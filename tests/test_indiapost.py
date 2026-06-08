import pytest

from app.integrations.indiapost import (
    extract_tracking_number,
    format_tracking_message,
    is_indiapost_configured,
    parse_tracking_summary,
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
    msg = format_tracking_message("EB126023474IN", row)
    assert "EB126023474IN" in msg
    assert "Delivered" in msg
    assert "Vijayanagar S.O" in msg
    assert msg.index("AWB:") < msg.index("Status:")


def test_format_tracking_message_not_delivered_first():
    row = {
        "booking_details": {"article_number": "EB126023474IN", "delivery_location": "Vijayanagar S.O"},
        "del_status": {"del_status": "not delivered"},
    }
    msg = format_tracking_message("EB126023474IN", row)
    summary = parse_tracking_summary("EB126023474IN", row)
    assert "Not delivered" in msg
    assert summary["location"] == "Vijayanagar S.O"
    assert msg.startswith("📦 *Shipment tracking*")
    assert "AWB: EB126023474IN" in msg


def test_is_order_tracking_message_detects_awb():
    from app.agents.order import is_order_tracking_message

    assert is_order_tracking_message("EB126023474IN")
    assert is_order_tracking_message("Order status - EB126023474IN")
    assert is_order_tracking_message("order_status")
    assert not is_order_tracking_message("Metformin 500mg")


def test_is_indiapost_configured(monkeypatch):
    monkeypatch.delenv("INDIAPOST_USERNAME", raising=False)
    monkeypatch.delenv("INDIAPOST_PASSWORD", raising=False)
    assert is_indiapost_configured() is False
    monkeypatch.setenv("INDIAPOST_USERNAME", "user")
    monkeypatch.setenv("INDIAPOST_PASSWORD", "pass")
    assert is_indiapost_configured() is True
