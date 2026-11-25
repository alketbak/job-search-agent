"""add status to pipeline_runs

Revision ID: 5f78ada7cab8
Revises: b30b11ce6ab6
Create Date: 2026-11-18 20:40:10.735251

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '5f78ada7cab8'
down_revision: str | None = 'b30b11ce6ab6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Known autogenerate false-positive stripped (docs/DEVELOPMENT.md) - the two
    # HNSW indexes again.
    #
    # Every pre-existing row was written by the old fully-synchronous
    # pipeline, which only ever inserted a PipelineRun after the whole
    # run completed - there's no real "pending"/"running" history to
    # recover for them, so they all backfill to 'success' regardless of
    # whether `errors` happens to be non-empty (this column is for
    # tracking a background job's progress going forward, not a
    # retroactive audit of past runs' outcomes).
    op.add_column('pipeline_runs', sa.Column('status', sa.String(length=16), nullable=True))
    op.execute("UPDATE pipeline_runs SET status = 'success' WHERE status IS NULL")
    op.alter_column('pipeline_runs', 'status', nullable=False)


def downgrade() -> None:
    op.drop_column('pipeline_runs', 'status')
