"""Add order payment tracking columns; drop payment_terms (T/T Advance only).

Revision ID: g2b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2026-05-29
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "g2b3c4d5e6f7"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("orders")}

    if "payment_status" not in columns:
        op.add_column(
            "orders",
            sa.Column(
                "payment_status",
                sa.String(),
                nullable=True,
                server_default="awaiting_payment",
            ),
        )
    if "virtual_account_id" not in columns:
        op.add_column("orders", sa.Column("virtual_account_id", sa.String(), nullable=True))
    if "utr_number" not in columns:
        op.add_column("orders", sa.Column("utr_number", sa.String(), nullable=True))
    if "payment_id" not in columns:
        op.add_column("orders", sa.Column("payment_id", sa.String(), nullable=True))
    if "payment_received_at" not in columns:
        op.add_column("orders", sa.Column("payment_received_at", sa.DateTime(), nullable=True))

    if "payment_terms" in columns:
        op.drop_column("orders", "payment_terms")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("orders")}

    if "payment_terms" not in columns:
        op.add_column(
            "orders",
            sa.Column("payment_terms", sa.String(), nullable=False, server_default="T/T Advance"),
        )

    for col in (
        "payment_received_at",
        "payment_id",
        "utr_number",
        "virtual_account_id",
        "payment_status",
    ):
        if col in columns:
            op.drop_column("orders", col)
