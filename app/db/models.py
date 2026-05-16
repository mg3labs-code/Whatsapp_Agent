from datetime import datetime

from sqlalchemy import Boolean, Column, Date, DateTime, Integer, Numeric, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Product(Base):
    """Catalog row from price-list import (Excel). No SKU / volume-discount columns."""

    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    product_name = Column(String, nullable=False)
    salt_name = Column(String(512), nullable=True)
    manufacturing_company = Column(String(512), nullable=True)
    expiry_date = Column(Date, nullable=True)
    price_per_strip = Column(Numeric(10, 2), nullable=False)
    is_restricted = Column(Boolean, default=False)
    schedule_category = Column(String(8), nullable=True)  # X, H, or H1 when restricted
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Product {self.product_name!r}>"


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True)
    phone = Column(String, nullable=False)
    company = Column(String)
    country = Column(String)
    business_type = Column(String)
    buyer_type = Column(String)
    license_number = Column(String)
    annual_volume_usd = Column(Numeric(12, 2))
    order_value_usd = Column(Numeric(12, 2))
    lead_score = Column(Integer)
    lead_category = Column(String)
    lifecycle_stage = Column(String, default="qualified")
    manual_review_only = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Lead phone={self.phone!r}>"


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    phone = Column(String, nullable=False)
    sku = Column(String, nullable=False)
    product_name = Column(String)
    quantity = Column(Integer, nullable=False)
    country = Column(String, nullable=False)
    city = Column(String, nullable=False)
    contact_name = Column(String, nullable=False)
    payment_terms = Column(String, nullable=False)
    status = Column(String, default="pending")
    order_ref = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Order order_ref={self.order_ref!r}>"


class GuardrailLog(Base):
    __tablename__ = "guardrail_logs"

    id = Column(Integer, primary_key=True)
    phone = Column(String, nullable=False)
    trigger_type = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    message_text = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<GuardrailLog trigger_type={self.trigger_type!r}>"
