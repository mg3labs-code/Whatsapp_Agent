"""Optional DB seed.

Legacy sample products were removed — catalog is loaded from Excel via
`python -m scripts.import_product_price_list` (see docs/PRODUCT_IMPORT.md).

This module keeps `create_tables()` + `seed_products()` for environments that
expect a no-op or future tiny fixtures.
"""

from dotenv import load_dotenv

from app.db.database import SessionLocal, create_tables

load_dotenv()

SAMPLE_PRODUCTS: list[dict] = []


def seed_products() -> None:
    """Create tables if missing. Inserts nothing by default (empty catalog)."""
    create_tables()
    db = SessionLocal()
    try:
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    seed_products()
