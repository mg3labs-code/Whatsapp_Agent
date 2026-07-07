"""Add unique constraint on leads.phone for atomic upserts.

Revision ID: b2c3d4e5f6a7
Revises: dd5ac7901670
Create Date: 2026-06-30
"""

from typing import Sequence, Union

from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "dd5ac7901670"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint("uq_leads_phone", "leads", ["phone"])


def downgrade() -> None:
    op.drop_constraint("uq_leads_phone", "leads", type_="unique")
