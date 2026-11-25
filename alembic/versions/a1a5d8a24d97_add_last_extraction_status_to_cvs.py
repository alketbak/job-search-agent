"""add last extraction status to cvs

Revision ID: a1a5d8a24d97
Revises: 33ab03db8711
Create Date: 2026-11-17 22:44:46.348102

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a1a5d8a24d97'
down_revision: str | None = '33ab03db8711'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # NOTE: hand-edited - see the same note in
    # 33ab03db8711_add_category_to_candidate_skills.py; autogenerate again
    # proposed dropping the raw-SQL HNSW indexes, which is unrelated to this
    # revision's actual change and intentionally not applied.
    op.add_column('cvs', sa.Column('last_extraction_status', sa.String(length=16), nullable=True))
    op.add_column('cvs', sa.Column('last_extraction_failure_reason', sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column('cvs', 'last_extraction_failure_reason')
    op.drop_column('cvs', 'last_extraction_status')
