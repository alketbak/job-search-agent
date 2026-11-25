"""add category to candidate_skills

Revision ID: 33ab03db8711
Revises: fe8411e537e3
Create Date: 2026-11-17 22:36:52.934525

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '33ab03db8711'
down_revision: str | None = 'fe8411e537e3'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # NOTE: hand-edited - autogenerate also proposed dropping the
    # ix_cv_chunks_embedding_hnsw / ix_job_embeddings_embedding_hnsw indexes
    # because they were created via raw SQL in the initial migration (ORM
    # metadata doesn't know about them) and so look "removed" by comparison.
    # They're intentional (pgvector HNSW with a specific opclass that
    # op.create_index can't express) - left alone here.
    op.add_column(
        'candidate_skills',
        sa.Column('category', sa.String(length=32), nullable=False, server_default='other'),
    )


def downgrade() -> None:
    op.drop_column('candidate_skills', 'category')
