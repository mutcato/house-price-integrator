"""Add is_active field into unique_house combine index

Revision ID: 6c7fc3b575dd
Revises: 17a457b1b0b3
Create Date: 2023-05-16 03:05:01.610131

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "6c7fc3b575dd"
down_revision = "17a457b1b0b3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint("unique_house", "houses", type_="unique")
    op.create_unique_constraint(
        "unique_house", "houses", ["internal_id", "data_source", "version", "is_active"]
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint("unique_house", "houses", type_="unique")
    op.create_unique_constraint(
        "unique_house", "houses", ["internal_id", "data_source", "version"]
    )
    # ### end Alembic commands ###
