import pytest

from app.integrations.cashfree import (
    get_static_payment_details_text,
    is_export_wire_configured,
    load_export_wire_details,
    missing_export_wire_fields,
    validate_export_wire_details,
)


def _sample_details(**overrides):
    base = {
        "account_name": "New Life Medicare Exports",
        "account_number": "123456789012",
        "bank_name": "Example Bank Ltd",
        "branch": "Mumbai Export Branch",
        "swift_code": "EXAMPLGB",
        "ifsc": "HDFC0001234",
    }
    base.update(overrides)
    return base


def test_missing_export_wire_fields_detects_required_gaps():
    missing = missing_export_wire_fields(_sample_details(swift_code=""))
    assert "swift_code" in missing


def test_validate_export_wire_details_rejects_bad_swift():
    errors = validate_export_wire_details(_sample_details(swift_code="BAD"))
    assert any("swift_code" in err for err in errors)


def test_validate_export_wire_details_rejects_bad_ifsc():
    errors = validate_export_wire_details(_sample_details(ifsc="123"))
    assert any("ifsc" in err for err in errors)


def test_validate_export_wire_details_allows_empty_ifsc():
    assert validate_export_wire_details(_sample_details(ifsc="")) == []


def test_get_static_payment_details_text_includes_export_fields(monkeypatch):
    for field, env_key in {
        "account_name": "STATIC_WIRE_ACCOUNT_NAME",
        "account_number": "STATIC_WIRE_ACCOUNT_NUMBER",
        "bank_name": "STATIC_WIRE_BANK_NAME",
        "branch": "STATIC_WIRE_BRANCH",
        "swift_code": "STATIC_WIRE_SWIFT_CODE",
        "ifsc": "STATIC_WIRE_IFSC",
    }.items():
        monkeypatch.setenv(env_key, _sample_details()[field])

    text = get_static_payment_details_text("ORD-20260712-1001", 1500.0, "USD")
    assert "Beneficiary: New Life Medicare Exports" in text
    assert "Account: 123456789012" in text
    assert "Bank: Example Bank Ltd" in text
    assert "Branch: Mumbai Export Branch" in text
    assert "SWIFT: EXAMPLGB" in text
    assert "IFSC: HDFC0001234" in text
    assert "Reference: ORD-20260712-1001" in text


def test_get_static_payment_details_text_fallback_when_incomplete(monkeypatch):
    monkeypatch.delenv("STATIC_WIRE_ACCOUNT_NAME", raising=False)
    monkeypatch.delenv("STATIC_WIRE_ACCOUNT_NUMBER", raising=False)
    monkeypatch.delenv("STATIC_WIRE_BANK_NAME", raising=False)
    monkeypatch.delenv("STATIC_WIRE_BRANCH", raising=False)
    monkeypatch.delenv("STATIC_WIRE_SWIFT_CODE", raising=False)

    text = get_static_payment_details_text("ORD-20260712-1001", 1500.0, "USD")
    assert "Our team will share international wire transfer details" in text
    assert not is_export_wire_configured()


def test_is_export_wire_configured(monkeypatch):
    for field, env_key in {
        "account_name": "STATIC_WIRE_ACCOUNT_NAME",
        "account_number": "STATIC_WIRE_ACCOUNT_NUMBER",
        "bank_name": "STATIC_WIRE_BANK_NAME",
        "branch": "STATIC_WIRE_BRANCH",
        "swift_code": "STATIC_WIRE_SWIFT_CODE",
    }.items():
        monkeypatch.setenv(env_key, _sample_details()[field])

    assert is_export_wire_configured()
    assert load_export_wire_details()["swift_code"] == "EXAMPLGB"
