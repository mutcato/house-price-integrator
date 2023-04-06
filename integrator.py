import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Iterator, Optional, Union

from sqlalchemy.exc import IntegrityError, NoResultFound

from database import session
from models import Attribute, DataSource
from models import House as HouseModel
from models import HouseAttribute, ListingCategory, RealtyType
from settings import config as settings

files: Iterator = os.scandir(settings["DATA_ROOT"])


def camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


@dataclass
class House:
    currency: str
    district: str
    county: str
    city: str
    room: int
    living_room: int = field(metadata={"alias": "livingRoom"})
    price: int
    floor: dict
    heating: str
    fuel: dict
    usage: dict
    credit: dict
    deposit: str
    balcony: str
    furnished: str
    parking: str
    attributes: list
    total_floor: int = field(init=False, default_factory=int)
    internal_id: int = field(metadata={"alias": "realtyId"}, default_factory=int)
    url: str = field(default="", metadata={"alias": "detailUrl"})
    bathroom: int = field(metadata={"alias": "bathRoom"}, default_factory=int)
    map_location: dict = field(
        default="", metadata={"alias": "mapLocation"}, repr=False
    )
    latitude: float = field(default=0.0)
    longitude: float = field(default=0.0)
    created_at: datetime = field(default="", metadata={"alias": "createdDate"})
    updated_at: datetime = field(default="", metadata={"alias": "updatedDate"})
    sqm: dict = field(default="", metadata={"alias": "sqm"}, repr=False)
    age: int = field(default_factory=int)
    realty_type: str = field(default="", metadata={"alias": "subCategory"})
    listing_category: str = field(default="", metadata={"alias": "redirectLink"})

    def __post_init__(self):
        self.district = self.district["name"].encode("utf-8").decode("utf-8").lower()
        self.county = self.county["name"].encode("utf-8").decode("utf-8").lower()
        self.city = self.city["name"].encode("utf-8").decode("utf-8").lower()
        self.room = sum(self.room)
        self.living_room = sum(self.living_room)
        self.latitude = self.map_location["lat"]
        self.longitude = self.map_location["lon"]
        self.created_at = datetime.strptime(
            self.created_at.split(".")[0], "%Y-%m-%dT%H:%M:%S"
        )
        self.updated_at = datetime.strptime(
            self.updated_at.split(".")[0], "%Y-%m-%dT%H:%M:%S"
        )
        self.net_sqm = self.sqm["netSqm"]
        self.gross_sqm = sum(self.sqm["grossSqm"])
        _temp = self.floor
        self.floor = _temp["name"]
        self.total_floor = _temp["count"]
        self.heating = self.heating["name"].encode("utf-8").decode("utf-8").lower()
        self.fuel = self.sanitize(self.fuel, "name")
        self.usage = self.sanitize(self.usage, "name")
        self.credit = self.sanitize(self.credit, "name")
        self.deposit = int(self.deposit["amount"])
        self.realty_type = RealtyType(self.sanitize(self.realty_type, "typeName"))
        self.listing_category = ListingCategory(
            self.sanitize(self.listing_category, "linkedCategoryUrl")
        )
        self.attributes = self.get_attributes(self.attributes)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        result = {}
        for k, v in data.items():
            for field in fields(cls):
                if field.name == k or not hasattr(field, "metadata"):
                    result[field.name] = v
                    break
                else:
                    if field.metadata.get("alias") == k:
                        result[field.name] = v
                        break
        return cls(**result)

    def __str__(self) -> str:
        return (
            f"{self.internal_id}-{self.listing_category.value}-{self.realty_type.value}"
        )

    def get_attributes(self, data: Dict[str, Any]):
        result = []
        for category, attr_list in data.items():
            for attribute in attr_list:
                result.append(
                    f"{camel_to_snake(category)}_{attribute['name'].encode('utf-8').decode('utf-8').lower()}"
                )
        return result

    def sanitize(self, data: Dict[Any, Any], *args) -> Optional[Union[str, int]]:
        if not data:
            return None

        for arg in args:
            result = data.get(arg)
            if type(result) not in [list, dict]:
                if type(result) is str:
                    return result.encode("utf-8").decode("utf-8").lower()
                else:
                    return result
            data = result

    def to_dict(self):
        return {
            "internal_id": self.internal_id,
            "url": self.url,
            "price": self.price,
            "currency": self.currency,
            "district": self.district,
            "county": self.county,
            "city": self.city,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "room": self.room,
            "living_room": self.living_room,
            "bathroom": self.bathroom,
            "net_sqm": self.net_sqm,
            "gross_sqm": self.gross_sqm,
            "floor": self.floor,
            "total_floor": self.total_floor,
            "heating": self.heating,
            "fuel": self.fuel,
            "usage": self.usage,
            "credit": self.credit,
            "deposit": self.deposit,
            "balcony": self.balcony,
            "furnished": self.furnished,
            "parking": self.parking,
            "age": self.age,
            "realty_type": self.realty_type.value,
            "listing_category": self.listing_category.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def save(self):
        house = HouseModel(**self.to_dict())
        for attr in self.attributes:
            attribute = Attribute(name=attr, value=1)
            attribute._house.append(house)
            house._attribute.append(attribute)
            session.add(attribute)
        session.add(house)
        session.commit()

    def save_or_update(self):
        house = session.query(HouseModel).filter_by(
            internal_id=self.internal_id, data_source=DataSource.HEPSI.value
        )
        if house.count() == 0:
            self.save()
        elif house.count() == 1:
            house.update(self.to_dict())
            session.commit()
        else:
            raise Exception("Duplicate house")
        return


for file in files:
    with open(file.path, "r") as f:
        data = json.loads(f.read())["realtyDetail"]
        house = House.from_dict(data)
        house.save_or_update()
        # Neden next file'a geçmiyor?
        print("bitti")
