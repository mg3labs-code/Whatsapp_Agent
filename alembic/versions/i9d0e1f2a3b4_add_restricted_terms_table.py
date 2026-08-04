"""Add restricted_terms table for pre-check blocking.

Revision ID: i9d0e1f2a3b4
Revises: b2c3d4e5f6a7
Create Date: 2026-08-04
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "i9d0e1f2a3b4"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "restricted_terms",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("term", sa.String(length=512), nullable=False),
        sa.Column("normalized_term", sa.String(length=512), nullable=False),
        sa.Column("schedule_category", sa.String(length=8), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False, server_default="manual"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_term", name="uq_restricted_terms_normalized_term"),
    )
    op.create_index(
        op.f("ix_restricted_terms_normalized_term"),
        "restricted_terms",
        ["normalized_term"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_restricted_terms_normalized_term"), table_name="restricted_terms")
    op.drop_table("restricted_terms")
