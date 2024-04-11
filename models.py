from datetime import datetime
from enum import Enum as EnumType

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
)

# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase, relationship


# Base = declarative_base()
class Base(DeclarativeBase):
    __abstract__ = True

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)


class ListingCategory(EnumType):
    SALE = "satilik"
    RENTAL = "kiralik"


class DataSource(EnumType):
    HEPSI = "hepsiemlak.com"
    SAHIBINDEN = "sahibinden.com"


class RealtyType(EnumType):
    DAIRE = "daire"
    VILLA = "villa"
    MUSTAKIL_EV = "müstakil ev"
    RESIDENCE = "residence"
    CIFTLIK_EVI = "çiftlik evi"
    YAZLIK = "yazlık"
    PREFABRIK = "prefabrik"
    KOSK = "köşk"
    YALI_DAIRESI = "yalı dairesi"
    YALI = "yalı"
    BINA = "bina"
    KOY_EVI = "köy evi"
    KOOPERATIF = "kooperatif"
    DAG_EVI = "dağ evi"
    LOFT_DAIRE = "loft daire"
    BUNGALOV = "bungalov"


class House(Base):
    __tablename__ = "houses"
    __table_args__ = (
        # this can be db.PrimaryKeyConstraint if you want it to be a primary key
        UniqueConstraint("internal_id", "data_source", "version", "is_active", name="unique_house"),
    )

    internal_id = Column(Integer, index=True)
    data_source = Column(String, index=True, default=DataSource.HEPSI.value)
    url = Column(String, index=True, nullable=False)
    room = Column(SmallInteger, index=True, default=0)
    living_room = Column(SmallInteger, index=True, default=0)
    floor = Column(String, index=True)
    total_floor = Column(Integer, index=True)
    bathroom = Column(SmallInteger, default=1)
    net_sqm = Column(Integer, index=True)
    gross_sqm = Column(Integer, index=True)
    age = Column(Integer, index=True)
    heating = Column(String, index=True)
    fuel = Column(String, index=True)
    usage = Column(String, index=True)
    credit = Column(String, index=True)
    deposit = Column(String, index=True)
    furnished = Column(String, index=True)
    version = Column(SmallInteger, default=1)
    is_last_version = Column(Boolean, default=True)
    latitude = Column(Numeric, index=True)
    longitude = Column(Numeric, index=True)
    created_at = Column(DateTime, index=True)
    updated_at = Column(DateTime, index=True)
    realty_type = Column(String, index=True, default=RealtyType.DAIRE.value)
    currency = Column(String, index=True)
    district = Column(String, index=True)
    county = Column(String, index=True)
    city = Column(String, index=True)
    price = Column(Integer, index=True)
    predicted_price = Column(Integer, index=True, default=None, nullable=True)
    predicted_rental_price = Column(Integer, index=True, default=None, nullable=True)
    created_at = Column(DateTime, index=True)
    updated_at = Column(DateTime, index=True)
    inserted_at = Column(DateTime, index=True, default=datetime.now)
    is_active = Column(Boolean, default=True)
    listing_category = Column(String, index=True, default=ListingCategory.SALE.value)
    _attribute = relationship(
        "Attribute", backref="House", secondary="house_attributes"
    )

    def __repr__(self):
        return f"House(internal_id={self.internal_id}, data_source={self.data_source}, version={self.version}, is_last_version={self.is_last_version}, is_active={self.is_active} price={self.price}, currency={self.currency}, predicted_price={self.predicted_price})"

    def __str__(self):
        return f"House(internal_id={self.internal_id}, data_source={self.data_source}, version={self.version}, is_last_version={self.is_last_version}, is_active={self.is_active} price={self.price}, currency={self.currency}, predicted_price={self.predicted_price})"


class Attribute(Base):
    __tablename__ = "attributes"

    name = Column(String, index=True)
    value = Column(String, index=True)
    _house = relationship("House", backref="Attribute", secondary="house_attributes")

    def __repr__(self):
        return f"Attributes(name={self.name}, value={self.value})"

    def __str__(self):
        return f"Attributes(name={self.name}, value={self.value})"


class HouseAttribute(Base):
    __tablename__ = "house_attributes"

    house_id = Column(Integer, ForeignKey("houses.id"), index=True)
    attribute_id = Column(Integer, ForeignKey("attributes.id"), index=True)

    def __repr__(self):
        return f"HouseAttribute(house={self.house}, attribute={self.attribute})"

    def __str__(self):
        return f"HouseAttribute(house={self.house}, attribute={self.attribute})"
