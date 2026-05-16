from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Lead, Product

TEST_DB_URL = "sqlite:///./test.db"
ENGINE = create_engine(TEST_DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)


def _setup_test_db():
    Base.metadata.drop_all(bind=ENGINE)
    Base.metadata.create_all(bind=ENGINE)


def test_product_insert():
    _setup_test_db()
    db = SessionLocal()
    try:
        product = Product(
            product_name="Test Product",
            salt_name="Test Salt",
            manufacturing_company="Test Mfg",
            price_per_strip=1.00,
            is_restricted=False,
        )
        db.add(product)
        db.commit()

        inserted = db.query(Product).filter(Product.product_name == "Test Product").first()
        assert inserted is not None
        assert inserted.price_per_strip == 1.00

        db.delete(inserted)
        db.commit()
    finally:
        db.close()


def test_lead_insert():
    _setup_test_db()
    db = SessionLocal()
    try:
        lead = Lead(
            phone="+911234567890",
            company="TestCo",
            country="India",
            business_type="distributor",
            lead_score=75,
        )
        db.add(lead)
        db.commit()

        inserted = db.query(Lead).filter(Lead.phone == "+911234567890").first()
        assert inserted is not None
        assert inserted.lead_score == 75
    finally:
        db.close()
