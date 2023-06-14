"""Remove unique constraint for url for version update

Revision ID: b2d256343ab8
Revises: 6f61b5a3d9a8
Create Date: 2023-05-13 19:38:49.076517

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b2d256343ab8'
down_revision = '6f61b5a3d9a8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('ix_houses_url', table_name='houses')
    op.create_index(op.f('ix_houses_url'), 'houses', ['url'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_houses_url'), table_name='houses')
    op.create_index('ix_houses_url', 'houses', ['url'], unique=False)
    # ### end Alembic commands ###