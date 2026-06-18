"""Tests for app.business.shipping (mocked DB — no PostgreSQL required)."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.business.shipping import (
    _resolve_country,
    calculate_cart_weight,
    get_shipping_options,
)


def _mock_db_for_weights(
    *,
    product_weight_g: float | None,
    box_weight_g: float = 150.0,
) -> MagicMock:
    """Return a MagicMock session for product + box_specs lookups."""

    def _execute(statement, params=None):
        sql = str(statement)
        result = MagicMock()
        if "products" in sql and "weight_g" in sql:
            result.fetchone.return_value = (product_weight_g,)
            return result
        if "box_specs" in sql:
            result.fetchone.return_value = (box_weight_g,)
            return result
        result.fetchone.return_value = None
        result.fetchall.return_value = []
        return result

    db = MagicMock()
    db.execute.side_effect = _execute
    return db


def _mock_db_for_shipping_rates(rows: list[tuple[str, float | None]]) -> MagicMock:
    """Mock shipping_rates query; fuzzy distinct-country query returns empty."""

    def _execute(statement, params=None):
        sql = str(statement)
        result = MagicMock()
        if "shipping_rates" in sql and "shipping_type" in sql:
            result.fetchall.return_value = rows
            return result
        if "DISTINCT" in sql.upper() and "country_name" in sql:
            result.fetchall.return_value = []
            return result
        result.fetchall.return_value = []
        result.fetchone.return_value = None
        return result

    db = MagicMock()
    db.execute.side_effect = _execute
    return db


def test_calculate_cart_weight_with_known_weights():
    db = _mock_db_for_weights(product_weight_g=15.0, box_weight_g=150.0)
    cart = [{"sku": "PROD-001", "product_name": "AMLIP 10MG", "qty": 50}]

    result = calculate_cart_weight(cart, db)

    assert result is not None
    assert result["total_product_g"] == 750
    # 50 units → box band 31–60 → box_no "9"
    assert result["box_no"] == "9"
    assert result["box_weight_g"] == 150
    assert result["total_shipment_g"] == 900
    assert result["items_missing_weight"] == []


def test_calculate_cart_weight_missing_weight_uses_default():
    db = _mock_db_for_weights(product_weight_g=0.0, box_weight_g=64.0)
    cart = [{"sku": "PROD-002", "product_name": "UNKNOWN PRODUCT", "qty": 3}]

    result = calculate_cart_weight(cart, db)

    assert result is not None
    assert result["total_product_g"] == 60  # 3 × 20g default
    assert "UNKNOWN PRODUCT" in result["items_missing_weight"]


def test_get_shipping_options_both_available():
    db = _mock_db_for_shipping_rates([("EMS", 42.0), ("LP", 28.0)])

    result = get_shipping_options("KENYA", 1400, db)

    assert result["available"] is True
    assert result["EMS"]["rate_usd"] == 42.0
    assert result["LP"]["rate_usd"] == 28.0


def test_get_shipping_options_lp_null_at_heavy_weight():
    db = _mock_db_for_shipping_rates([("EMS", 55.0), ("LP", None)])

    result = get_shipping_options("KENYA", 2500, db)

    assert result["available"] is True
    assert result["LP"] is None
    assert result["EMS"]["rate_usd"] == 55.0


def test_get_shipping_options_country_not_found():
    db = _mock_db_for_shipping_rates([])

    result = get_shipping_options("ATLANTIS", 1000, db)

    assert result["available"] is False
    assert result["EMS"] is None
    assert result["LP"] is None


def test_country_alias_resolution():
    assert _resolve_country("UAE") == "UNITED ARAB EMIRATES"
    assert _resolve_country("UK") == "UNITED KINGDOM"
    assert _resolve_country("usa") == "UNITED STATES OF AMERICA"
    assert _resolve_country("KENYA") == "KENYA"
