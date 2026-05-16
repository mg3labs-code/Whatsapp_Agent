"""Add client SOP lead scoring fields to leads table.

Revision ID: e8f9a0b1c2d3
Revises: d4e5f6a7b8c9
Create Date: 2026-05-15

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e8f9a0b1c2d3"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("leads", sa.Column("buyer_type", sa.String(), nullable=True))
    op.add_column("leads", sa.Column("order_value_usd", sa.Numeric(precision=12, scale=2), nullable=True))
    op.add_column("leads", sa.Column("lead_category", sa.String(), nullable=True))
    op.add_column("leads", sa.Column("lifecycle_stage", sa.String(), nullable=True))
    op.add_column("leads", sa.Column("manual_review_only", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("leads", "manual_review_only")
    op.drop_column("leads", "lifecycle_stage")
    op.drop_column("leads", "lead_category")
    op.drop_column("leads", "order_value_usd")
    op.drop_column("leads", "buyer_type")
