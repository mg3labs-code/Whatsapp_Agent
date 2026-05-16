"""Unit tests for product import helpers (no DB)."""

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Product
from scripts.import_product_price_list import (
    _category_from_schedule_header,
    _find_existing,
    _match_schedule_category,
    _parse_date,
    _parse_decimal,
    _prune_orphan_null_expiry_duplicates,
    _resolve_headers,
    _row_to_payload,
    import_workbook,
    load_schedule_terms_by_category,
)

_CATALOG_SAMPLE = Path(__file__).resolve().parent.parent / "data" / "import_samples" / "catalog.xlsx"
_SCHEDULE_SAMPLE = Path(__file__).resolve().parent.parent / "data" / "import_samples" / "schedule_hx.xlsx"


def _empty_schedule() -> dict[str, set[str]]:
    return {"X": set(), "H": set(), "H1": set()}


@pytest.fixture
def import_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()


def test_category_from_schedule_header():
    assert _category_from_schedule_header("Schedule X drugs:") == "X"
    assert _category_from_schedule_header("Schedule H drugs:") == "H"
    assert _category_from_schedule_header("Schedule H1 drugs:") == "H1"


def test_resolve_headers_catalog_sample_headers():
    header = (
        "PRODUCT NAME",
        "SALTNAME / GENERIC NAME",
        "MANUFACTURING COMPANY",
        "EXPIRY DATE",
        "USD PRICE PER STRIP",
    )
    m = _resolve_headers(list(header))
    assert m["product_name"] == 0
    assert m["expiry_date"] == 3


def test_resolve_headers_standard_sheet():
    header = [
        "Product name",
        "Saltname/Generic name",
        "Manufacturing company",
        "Expiry date",
        "USD price per strip",
    ]
    m = _resolve_headers(header)
    assert m["product_name"] == 0
    assert m["salt_name"] == 1
    assert m["manufacturing_company"] == 2
    assert m["expiry_date"] == 3
    assert m["price_per_strip"] == 4


def test_parse_decimal_currency_stripped():
    assert _parse_decimal("$12.50") == Decimal("12.50")


def test_parse_date_datetime_object():
    assert _parse_date(datetime(2028, 1, 1, 0, 0, 0)) == date(2028, 1, 1)


def test_parse_date_dd_mm_yyyy_string():
    assert _parse_date("01-01-2028") == date(2028, 1, 1)


def test_parse_date_datetime_string_with_time():
    assert _parse_date("2028-01-01 00:00:00") == date(2028, 1, 1)


def test_parse_date_excel_serial():
    from openpyxl.utils.datetime import to_excel

    serial = to_excel(datetime(2028, 1, 1))
    assert _parse_date(serial) == date(2028, 1, 1)


@pytest.mark.skipif(not _SCHEDULE_SAMPLE.is_file(), reason="sample schedule_hx.xlsx not present")
def test_load_schedule_terms_all_three_columns():
    by_cat = load_schedule_terms_by_category(_SCHEDULE_SAMPLE)
    assert len(by_cat["X"]) >= 10
    assert len(by_cat["H"]) >= 10
    assert len(by_cat["H1"]) >= 5
    assert "tramadol" in by_cat["H"] or "tramadol hydrochloride" in by_cat["H"]
    assert "ketamine" in by_cat["X"]


def test_row_to_payload_schedule_h_match():
    colmap = {
        "product_name": 0,
        "salt_name": 1,
        "manufacturing_company": 2,
        "expiry_date": 3,
        "price_per_strip": 4,
    }
    schedule = {"X": set(), "H": {"tramadol"}, "H1": set()}
    row = ("Tab A", "Tramadol HCl", "Co Y", datetime(2027, 1, 15), 1.25)
    p = _row_to_payload(row, colmap, 2, schedule)
    assert p is not None
    assert p["is_restricted"] is True
    assert p["schedule_category"] == "H"


def test_match_schedule_category_priority_x_over_h():
    schedule = {"X": {"ketamine"}, "H": {"ketamine"}, "H1": set()}
    assert _match_schedule_category("Ketamine tabs", "Ketamine HCl", schedule) == "X"


def test_match_schedule_category_priority_h1_over_h():
    schedule = {"X": set(), "H": {"codeine"}, "H1": {"codeine"}}
    assert _match_schedule_category("Tabs", "Codeine phosphate", schedule) == "H1"


def test_row_to_payload_not_restricted_without_schedule_match():
    colmap = {"product_name": 0, "salt_name": 1, "manufacturing_company": 2, "expiry_date": 3, "price_per_strip": 4}
    row = ("Tab A", "Paracetamol", "Co Y", "01-01-2028", 1.25)
    p = _row_to_payload(row, colmap, 2, _empty_schedule())
    assert p["is_restricted"] is False
    assert p["schedule_category"] is None
    assert p["expiry_date"] == date(2028, 1, 1)


def test_row_to_payload_skips_empty_name():
    colmap = {"product_name": 0, "price_per_strip": 1}
    assert _row_to_payload(("", 5), colmap, 3, _empty_schedule()) is None


def test_row_to_payload_logs_unparsed_expiry():
    colmap = {"product_name": 0, "expiry_date": 1, "price_per_strip": 2}
    issues: list[tuple[int, str, object, str]] = []
    p = _row_to_payload(
        ("Tab A", "not-a-date", 1.0),
        colmap,
        5,
        _empty_schedule(),
        expiry_parse_issues=issues,
    )
    assert p is not None
    assert p["expiry_date"] is None
    assert len(issues) == 1
    assert issues[0][0] == 5


def test_find_existing_distinguishes_expiry_date(import_db):
    db = import_db
    db.add_all(
        [
            Product(
                product_name="CENFORCE 100MG 1X10",
                salt_name="SILDENAFIL",
                manufacturing_company="CENTURION LABS",
                expiry_date=date(2028, 1, 12),
                price_per_strip=Decimal("1.00"),
            ),
            Product(
                product_name="CENFORCE 100MG 1X10",
                salt_name="SILDENAFIL",
                manufacturing_company="CENTURION LABS",
                expiry_date=date(2028, 1, 9),
                price_per_strip=Decimal("1.10"),
            ),
        ]
    )
    db.commit()
    payload = {
        "product_name": "CENFORCE 100MG 1X10",
        "salt_name": "SILDENAFIL",
        "manufacturing_company": "CENTURION LABS",
        "expiry_date": date(2028, 1, 9),
        "price_per_strip": Decimal("1.20"),
    }
    found = _find_existing(db, payload)
    assert found is not None
    assert found.expiry_date == date(2028, 1, 9)


def test_prune_orphan_null_expiry_duplicates(import_db):
    db = import_db
    db.add(
        Product(
            product_name="VIDALISTA 10MG 1X10",
            salt_name="TADALAFIL",
            manufacturing_company="CENTURION LABS",
            expiry_date=None,
            price_per_strip=Decimal("1.00"),
        )
    )
    db.add(
        Product(
            product_name="VIDALISTA 10MG 1X10",
            salt_name="TADALAFIL",
            manufacturing_company="CENTURION LABS",
            expiry_date=date(2028, 1, 12),
            price_per_strip=Decimal("1.10"),
        )
    )
    db.commit()
    pruned = _prune_orphan_null_expiry_duplicates(db)
    db.commit()
    remaining = db.query(Product).filter(Product.product_name == "VIDALISTA 10MG 1X10").all()
    assert pruned == 1
    assert len(remaining) == 1
    assert remaining[0].expiry_date == date(2028, 1, 12)


@pytest.mark.skipif(not _CATALOG_SAMPLE.is_file(), reason="sample catalog.xlsx not present")
def test_catalog_sample_all_expiries_parse():
    _, _, skipped, stats, _ = import_workbook(_CATALOG_SAMPLE, dry_run=True, schedule_path=None)
    assert skipped == 0
    assert stats["unparsed"] == 0
    assert stats["parsed"] == 351
    assert stats["missing"] == 0


@pytest.mark.skipif(
    not _CATALOG_SAMPLE.is_file() or not _SCHEDULE_SAMPLE.is_file(),
    reason="sample workbooks not present",
)
def test_catalog_with_schedule_dry_run_assigns_categories():
    _, _, skipped, stats, _ = import_workbook(
        _CATALOG_SAMPLE,
        dry_run=True,
        schedule_path=_SCHEDULE_SAMPLE,
    )
    assert skipped == 0
    assert stats["parsed"] == 351
