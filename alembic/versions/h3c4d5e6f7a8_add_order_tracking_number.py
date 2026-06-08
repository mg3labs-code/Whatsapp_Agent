"""Add tracking_number to orders for India Post AWB linkage.

Revision ID: h3c4d5e6f7a8
Revises: g2b3c4d5e6f7
Create Date: 2026-06-06
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "h3c4d5e6f7a8"
down_revision: Union[str, Sequence[str], None] = "g2b3c4d5e6f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("orders")}
    if "tracking_number" not in columns:
        op.add_column("orders", sa.Column("tracking_number", sa.String(), nullable=True))
        op.create_index("ix_orders_tracking_number", "orders", ["tracking_number"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("orders")}
    if "tracking_number" in columns:
        op.drop_index("ix_orders_tracking_number", table_name="orders")
        op.drop_column("orders", "tracking_number")
