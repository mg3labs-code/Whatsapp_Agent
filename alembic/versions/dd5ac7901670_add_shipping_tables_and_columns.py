"""add_shipping_tables_and_columns

Revision ID: dd5ac7901670
Revises: h3c4d5e6f7a8
Create Date: 2026-06-16 10:58:18.667291

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dd5ac7901670'
down_revision: Union[str, Sequence[str], None] = 'h3c4d5e6f7a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "box_specs" not in tables:
        op.create_table(
            "box_specs",
            sa.Column("box_no", sa.Text(), nullable=False),
            sa.Column("box_type", sa.Text(), nullable=False),
            sa.Column("weight_g", sa.Numeric(8, 2), nullable=False),
            sa.Column("height_cm", sa.Text(), nullable=True),
            sa.Column("length_cm", sa.Text(), nullable=True),
            sa.Column("breadth_cm", sa.Text(), nullable=True),
            sa.Column("max_strips", sa.Text(), nullable=True),
            sa.Column("max_tubes", sa.Text(), nullable=True),
            sa.Column("max_vials", sa.Text(), nullable=True),
            sa.Column("max_bottles", sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint("box_no"),
        )

    if "shipping_rates" not in tables:
        op.create_table(
            "shipping_rates",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("country_name", sa.Text(), nullable=False),
            sa.Column("shipping_type", sa.Text(), nullable=False),
            sa.Column("weight_from_g", sa.Integer(), nullable=False),
            sa.Column("weight_to_g", sa.Integer(), nullable=False),
            sa.Column("rate_usd", sa.Numeric(8, 2), nullable=True),
            sa.CheckConstraint("shipping_type IN ('EMS','LP')", name="ck_shipping_rates_type"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "country_name",
                "shipping_type",
                "weight_from_g",
                name="uq_shipping_rates_country_type_weight_from",
            ),
        )

    inspector = sa.inspect(bind)
    shipping_indexes = {idx["name"] for idx in inspector.get_indexes("shipping_rates")}
    if "idx_shipping_lookup" not in shipping_indexes:
        op.execute(
            "CREATE INDEX idx_shipping_lookup "
            "ON shipping_rates (UPPER(country_name), shipping_type, weight_from_g)"
        )

    product_columns = {c["name"] for c in inspector.get_columns("products")}
    if "weight_g" not in product_columns:
        op.add_column(
            "products",
            sa.Column("weight_g", sa.Numeric(8, 2), nullable=True, server_default=sa.text("0")),
        )
    if "weight_source" not in product_columns:
        op.add_column(
            "products",
            sa.Column("weight_source", sa.String(), nullable=True, server_default=sa.text("'unknown'")),
        )

    order_columns = {c["name"] for c in inspector.get_columns("orders")}
    if "total_weight_g" not in order_columns:
        op.add_column("orders", sa.Column("total_weight_g", sa.Numeric(10, 2), nullable=True))
    if "box_no" not in order_columns:
        op.add_column("orders", sa.Column("box_no", sa.Text(), nullable=True))
    if "shipping_type" not in order_columns:
        op.add_column("orders", sa.Column("shipping_type", sa.Text(), nullable=True))
    if "shipping_cost_usd" not in order_columns:
        op.add_column("orders", sa.Column("shipping_cost_usd", sa.Numeric(10, 2), nullable=True))
    if "shipping_days" not in order_columns:
        op.add_column("orders", sa.Column("shipping_days", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "orders" in tables:
        order_columns = {c["name"] for c in inspector.get_columns("orders")}
        if "shipping_days" in order_columns:
            op.drop_column("orders", "shipping_days")
        if "shipping_cost_usd" in order_columns:
            op.drop_column("orders", "shipping_cost_usd")
        if "shipping_type" in order_columns:
            op.drop_column("orders", "shipping_type")
        if "box_no" in order_columns:
            op.drop_column("orders", "box_no")
        if "total_weight_g" in order_columns:
            op.drop_column("orders", "total_weight_g")

    if "products" in tables:
        product_columns = {c["name"] for c in inspector.get_columns("products")}
        if "weight_source" in product_columns:
            op.drop_column("products", "weight_source")
        if "weight_g" in product_columns:
            op.drop_column("products", "weight_g")

    tables = set(sa.inspect(bind).get_table_names())
    if "shipping_rates" in tables:
        shipping_indexes = {idx["name"] for idx in sa.inspect(bind).get_indexes("shipping_rates")}
        if "idx_shipping_lookup" in shipping_indexes:
            op.drop_index("idx_shipping_lookup", table_name="shipping_rates")
        op.drop_table("shipping_rates")

    tables = set(sa.inspect(bind).get_table_names())
    if "box_specs" in tables:
        op.drop_table("box_specs")
