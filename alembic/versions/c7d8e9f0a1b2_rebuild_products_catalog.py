"""Rebuild products table: catalog columns only (no SKU, no volume discounts).

Revision ID: c7d8e9f0a1b2
Revises: a3f8c1d2e4b5
Create Date: 2026-05-13

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c7d8e9f0a1b2"
down_revision: Union[str, Sequence[str], None] = "a3f8c1d2e4b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("products")
    op.create_table(
        "products",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("product_name", sa.String(), nullable=False),
        sa.Column("salt_name", sa.String(length=512), nullable=True),
        sa.Column("manufacturing_company", sa.String(length=512), nullable=True),
        sa.Column("expiry_date", sa.Date(), nullable=True),
        sa.Column("price_per_strip", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("is_restricted", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("products")
    op.create_table(
        "products",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sku", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("salt_name", sa.String(length=512), nullable=True),
        sa.Column("manufacturing_company", sa.String(length=512), nullable=True),
        sa.Column("expiry_date", sa.Date(), nullable=True),
        sa.Column("price_per_unit", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("discount_200_units", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("discount_500_units", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("is_restricted", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sku"),
    )
