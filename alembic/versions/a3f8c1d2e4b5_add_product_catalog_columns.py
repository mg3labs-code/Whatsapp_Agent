"""Add optional catalog columns from price-list imports.

Revision ID: a3f8c1d2e4b5
Revises: 81bdeacc0c49
Create Date: 2026-05-13

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a3f8c1d2e4b5"
down_revision: Union[str, Sequence[str], None] = "81bdeacc0c49"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("products", sa.Column("salt_name", sa.String(length=512), nullable=True))
    op.add_column(
        "products",
        sa.Column("manufacturing_company", sa.String(length=512), nullable=True),
    )
    op.add_column("products", sa.Column("expiry_date", sa.Date(), nullable=True))


def downgrade() -> None:
    op.drop_column("products", "expiry_date")
    op.drop_column("products", "manufacturing_company")
    op.drop_column("products", "salt_name")
