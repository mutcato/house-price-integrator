"""Add age field

Revision ID: 68a1728bca9b
Revises: 82c8c312f8e0
Create Date: 2023-04-06 01:47:05.685586

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "68a1728bca9b"
down_revision = "82c8c312f8e0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("houses", sa.Column("age", sa.Integer(), nullable=True))
    op.create_index(op.f("ix_houses_age"), "houses", ["age"], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f("ix_houses_age"), table_name="houses")
    op.drop_column("houses", "age")
    # ### end Alembic commands ###
