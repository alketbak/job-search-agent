"""add job profile scalar fields to jobs

Revision ID: 817171521b52
Revises: a1a5d8a24d97
Create Date: 2026-11-17 23:55:54.065916

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '817171521b52'
down_revision: str | None = 'a1a5d8a24d97'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # NOTE: hand-edited - see the same note in 33ab03db8711; autogenerate
    # again proposed dropping the raw-SQL HNSW indexes, unrelated here too.
    op.add_column('jobs', sa.Column('seniority', sa.String(length=64), nullable=True))
    op.add_column('jobs', sa.Column('remote', sa.String(length=16), nullable=True))
    op.add_column('jobs', sa.Column('employment_type', sa.String(length=64), nullable=True))
    op.add_column('jobs', sa.Column('salary_min', sa.Float(), nullable=True))
    op.add_column('jobs', sa.Column('salary_max', sa.Float(), nullable=True))
    op.add_column('jobs', sa.Column('salary_currency', sa.String(length=8), nullable=True))
    op.add_column('jobs', sa.Column('years_experience_min', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('jobs', 'years_experience_min')
    op.drop_column('jobs', 'salary_currency')
    op.drop_column('jobs', 'salary_max')
    op.drop_column('jobs', 'salary_min')
    op.drop_column('jobs', 'employment_type')
    op.drop_column('jobs', 'remote')
    op.drop_column('jobs', 'seniority')
