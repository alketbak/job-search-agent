"""add consent_given_at to cvs

Revision ID: b30b11ce6ab6
Revises: a1740e3509ed
Create Date: 2026-11-18 20:18:01.935294

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b30b11ce6ab6'
down_revision: str | None = 'a1740e3509ed'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Known autogenerate false-positive stripped (docs/DEVELOPMENT.md) - these two
    # HNSW indexes are raw-SQL-created and ORM metadata doesn't know about
    # them.
    #
    # Nullable first, backfilled from the row's own `uploaded_at` (the
    # best available evidence of when an older CV was actually
    # submitted, since consent wasn't tracked separately before this),
    # then tightened to NOT NULL - new rows always get a real value from
    # api/router.py::upload_cv.
    op.add_column('cvs', sa.Column('consent_given_at', sa.DateTime(), nullable=True))
    op.execute('UPDATE cvs SET consent_given_at = uploaded_at WHERE consent_given_at IS NULL')
    op.alter_column('cvs', 'consent_given_at', nullable=False)


def downgrade() -> None:
    op.drop_column('cvs', 'consent_given_at')
