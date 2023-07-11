import logging
import math
import os

import pytest
from pytest_postgresql import factories
from pytest_postgresql.janitor import DatabaseJanitor
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import sessionmaker

from database import Base
from integrator import House
from models import Attribute, Base, House, HouseAttribute

logger = logging.getLogger("offices")


# def test_create_house(postgresql):
#     from models import House, DataSource, RealtyType

#     # Create a new House object
#     house = House(
#         internal_id=1234,
#         data_source=DataSource.HEPSI.value,
#         url="http://example.com/house/1234",
#         room=2,
#         living_room=1,
#         floor="2",
#         total_floor=5,
#         bathroom=1,
#         net_sqm=80,
#         gross_sqm=100,
#         age=10,
#         heating="central",
#         fuel="natural gas",
#         usage="residential",
#         credit="available",
#         deposit="5000 TL",
#         furnished="unfurnished",
#         latitude=41.0082,
#         longitude=28.9784,
#         realty_type=RealtyType.DAIRE.value,
#         currency="TL",
#         district="Kadikoy",
#         county="Istanbul",
#         city="Istanbul",
#         price=300000,
#         predicted_price=320000,
#     )

#     # Add the House object to the database
#     postgresql.add(house)
#     postgresql.commit()

#     # Retrieve the House object from the database
#     house = postgresql.query(House).filter_by(internal_id=1234).one()

#     # Assert that the object was added correctly
#     assert house.internal_id == 1234
#     assert house.data_source == DataSource.HEPSI.value
#     assert house.url == "http://example.com/house/1234"
#     assert house.room == 2
#     assert house.living_room == 1
#     assert house.floor == "2"
#     assert house.total_floor == 5
#     assert house.bathroom == 1
#     assert house.net_sqm == 80
#     assert house.gross_sqm == 100
#     assert house.age == 10
#     assert house.heating == "central"
#     assert house.fuel == "natural gas"
#     assert house.usage == "residential"
#     assert house.credit == "available"
#     assert house.deposit == "5000 TL"
#     assert house.furnished == "unfurnished"
#     assert house.latitude == 41.0082
#     assert house.longitude == 28.9784
#     assert house.realty_type == RealtyType.DAIRE.value
#     assert house.currency == "TL"
#     assert house.district == "Kadikoy"
#     assert house.county == "Istanbul"
#     assert house.city == "Istanbul"
#     assert house.price == 300000
#     assert house.predicted_price == 320000

"""
TODO: DB ile test yapılacak
* update methodu versiyonlama şeklinde değiştirilecek ve testi yazılacak
   id, data_source ve fiyat aynı ise update yapacak
   id, data_source ve fiyat farklı ise versiyon numarası bir artırılarak yeni kayıt ekleyecek. eski kaydın is_last_version alanı False yapılacak. Yeni kaydın is_last_version alanı True olacak.
"""


postgresql_noproc = factories.postgresql_noproc(
    host="127.0.0.1",
    port=5432,
    user="postgres",
)
postgres_client = factories.postgresql("postgresql_noproc")

from psycopg import sql
from psycopg.rows import dict_row

import alembic
from alembic.config import Config


def test_tables_count(postgres_client):
    os.environ["TESTING"] = "1"
    config = Config("alembic.ini")
    alembic.command.upgrade(config, "head")
    yield
    alembic.command.downgrade(config, "base")
    with postgres_client.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            sql.SQL(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_type = 'houses'"
                " AND table_schema NOT IN ('pg_catalog', 'information_schema')"
            )
        )
        result = cursor.fetchone()
        breakpoint()
        print(result)
    assert result["count"] == 0
