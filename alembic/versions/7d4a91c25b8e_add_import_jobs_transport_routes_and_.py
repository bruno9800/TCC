"""add import_jobs, transport_routes and academic_events.source

Revision ID: 7d4a91c25b8e
Revises: 486495197093
Create Date: 2026-07-08

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7d4a91c25b8e'
down_revision: Union[str, Sequence[str], None] = '486495197093'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'academic_events',
        sa.Column('source', sa.String(length=20), nullable=False, server_default='manual'),
    )
    # Tudo que existe hoje veio dos scripts de seed — marca como 'seed' para que a
    # primeira importação via admin possa substituí-los (só 'manual' é preservado).
    op.execute("UPDATE academic_events SET source = 'seed'")

    op.create_table(
        'transport_routes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('semester', sa.String(length=10), nullable=False),
        sa.Column('shift', sa.String(length=20), nullable=False),
        sa.Column('bus_label', sa.String(length=50), nullable=False),
        sa.Column('route_description', sa.String(length=300), nullable=False),
        sa.Column('section_title', sa.String(length=300), nullable=True),
        sa.Column('effective_date', sa.Date(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'transport_route_stops',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('route_id', sa.Integer(), nullable=False),
        sa.Column('seq', sa.Integer(), nullable=False),
        sa.Column('time', sa.String(length=10), nullable=True),
        sa.Column('location', sa.String(length=300), nullable=False),
        sa.ForeignKeyConstraint(['route_id'], ['transport_routes.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'import_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('import_type', sa.String(length=20), nullable=False),
        sa.Column('course_id', sa.Integer(), nullable=True),
        sa.Column('semester', sa.String(length=10), nullable=True),
        sa.Column('filename', sa.String(length=300), nullable=False),
        sa.Column('storage_path', sa.String(length=500), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=True),
        sa.Column('stats', sa.JSON(), nullable=True),
        sa.Column('warnings', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('created_by_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('applied_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ),
        sa.ForeignKeyConstraint(['created_by_id'], ['admin_users.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('import_jobs')
    op.drop_table('transport_route_stops')
    op.drop_table('transport_routes')
    op.drop_column('academic_events', 'source')
