"""add application status lifecycle

Revision ID: fb3e7e665203
Revises: 817171521b52
Create Date: 2026-11-18 13:49:00.565901

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'fb3e7e665203'
down_revision: str | None = '817171521b52'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # NOTE: hand-edited - see the same note in 33ab03db8711; autogenerate
    # again proposed dropping the raw-SQL HNSW indexes, unrelated here.
    # server_default added on the three new NOT NULL columns so this is
    # safe to run against a table that already has rows.
    op.add_column(
        'applications',
        sa.Column(
            'status',
            sa.Enum(
                'INTERESTED', 'SAVED', 'APPLIED', 'SCREENING', 'INTERVIEW',
                'TECHNICAL_ASSESSMENT', 'OFFER', 'REJECTED', 'WITHDRAWN',
                name='applicationstatus', native_enum=False,
            ),
            nullable=False,
            server_default='INTERESTED',
        ),
    )
    op.add_column(
        'applications',
        sa.Column('status_changed_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    op.add_column('applications', sa.Column('salary_offered', sa.Float(), nullable=True))
    op.add_column(
        'applications',
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    op.alter_column('applications', 'applied_at', existing_type=postgresql.TIMESTAMP(), nullable=True)
    op.create_index(op.f('ix_applications_status'), 'applications', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_applications_status'), table_name='applications')
    op.alter_column('applications', 'applied_at', existing_type=postgresql.TIMESTAMP(), nullable=False)
    op.drop_column('applications', 'created_at')
    op.drop_column('applications', 'salary_offered')
    op.drop_column('applications', 'status_changed_at')
    op.drop_column('applications', 'status')
