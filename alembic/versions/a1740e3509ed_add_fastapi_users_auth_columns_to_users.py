"""add fastapi-users auth columns to users

Revision ID: a1740e3509ed
Revises: fb3e7e665203
Create Date: 2026-11-18 20:04:00.646060

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a1740e3509ed'
down_revision: str | None = 'fb3e7e665203'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # NOTE: autogenerate always proposes dropping ix_cv_chunks_embedding_hnsw
    # and ix_job_embeddings_embedding_hnsw here - a known false positive
    # (docs/DEVELOPMENT.md: they're raw-SQL-created, ORM metadata doesn't know about
    # them). Stripped, same as every prior migration that hit this.
    #
    # hashed_password gets a server_default of '!' - never a valid bcrypt
    # hash, so it can't match any real password check - purely so the
    # pre-existing `default@jobradar.local` bootstrap row (created before
    # auth existed, see database/seed.py) satisfies the NOT NULL
    # constraint without being a row anyone could actually log in as.
    op.add_column('users', sa.Column('hashed_password', sa.String(length=1024), nullable=False, server_default='!'))
    op.add_column('users', sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()))
    op.add_column('users', sa.Column('is_superuser', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('users', sa.Column('is_verified', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.alter_column('users', 'email',
               existing_type=sa.VARCHAR(length=256),
               type_=sa.String(length=320),
               existing_nullable=False)


def downgrade() -> None:
    op.alter_column('users', 'email',
               existing_type=sa.String(length=320),
               type_=sa.VARCHAR(length=256),
               existing_nullable=False)
    op.drop_column('users', 'is_verified')
    op.drop_column('users', 'is_superuser')
    op.drop_column('users', 'is_active')
    op.drop_column('users', 'hashed_password')
