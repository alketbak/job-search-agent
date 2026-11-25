"""add owner_user_id to jobs

Jobs a user adds by hand are private to that user; NULL keeps the existing
shared-job behavior for everything that came from a source connector.

Hand-edited: autogenerate's drop/create of the two pgvector HNSW indexes
was removed (they are created via raw SQL in the initial migration, so ORM
metadata doesn't know they exist).

Revision ID: e58182b45eaa
Revises: 5f78ada7cab8
Create Date: 2026-11-25 00:34:21.781002

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e58182b45eaa'
down_revision: str | None = '5f78ada7cab8'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FK_NAME = 'fk_jobs_owner_user_id_users'


def upgrade() -> None:
    op.add_column('jobs', sa.Column('owner_user_id', sa.String(length=36), nullable=True))
    op.create_index(op.f('ix_jobs_owner_user_id'), 'jobs', ['owner_user_id'], unique=False)
    op.create_foreign_key(_FK_NAME, 'jobs', 'users', ['owner_user_id'], ['id'], ondelete='CASCADE')
    # Jobs added by hand before this migration were shared; make each one
    # private to the account tracking it. Skipped if more than one account
    # tracks the job (it then stays shared rather than guessing an owner).
    op.execute(
        """
        UPDATE jobs SET owner_user_id = (
            SELECT MIN(a.user_id) FROM applications a WHERE a.job_id = jobs.id
        )
        WHERE source = 'manual'
          AND (SELECT COUNT(DISTINCT a.user_id) FROM applications a WHERE a.job_id = jobs.id) = 1
        """
    )


def downgrade() -> None:
    op.drop_constraint(_FK_NAME, 'jobs', type_='foreignkey')
    op.drop_index(op.f('ix_jobs_owner_user_id'), table_name='jobs')
    op.drop_column('jobs', 'owner_user_id')
