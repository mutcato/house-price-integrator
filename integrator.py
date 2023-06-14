import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, Iterator, Optional, Union

from database import session
from models import Attribute, DataSource
from models import House as HouseModel, HouseAttribute
from models import ListingCategory, RealtyType
from settings import config as settings
from utils import get_traceback

logger = logging.getLogger("django")


def follow():
    """Follow the data directory for new files."""
    while True:
        files: Iterator = os.scandir(settings["DATA_ROOT"])
        _files = [file for file in files]
        if not _files:
            time.sleep(180)  # Sleep briefly
            continue
        yield _files


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
    floor: dict
    heating: str
    fuel: dict
    usage: dict
    credit: dict
    deposit: str
    furnished: str
    price: int = field(default_factory=int)
    _attributes: list = field(default_factory=list, metadata={"alias": "attributes"})
    total_floor: int = field(init=False, default_factory=int)
    internal_id: int = field(metadata={"alias": "realtyId"}, default_factory=int)
    url: str = field(default="", metadata={"alias": "detailUrl"})
    bathroom: int = field(metadata={"alias": "bathRoom"}, default_factory=int)
    _map_location: dict = field(
        default="", metadata={"alias": "mapLocation"}, repr=False
    )
    latitude: float = field(default=0.0)
    longitude: float = field(default=0.0)
    created_at: datetime = field(default="", metadata={"alias": "createdDate"})
    updated_at: datetime = field(default="", metadata={"alias": "updatedDate"})
    _sqm: dict = field(default="", metadata={"alias": "sqm"}, repr=False)
    age: int = field(default_factory=int)
    realty_type: str = field(default="", metadata={"alias": "subCategory"})
    listing_category: str = field(default="", metadata={"alias": "redirectLink"})
    version: int = field(default=1)
    is_last_version: bool = field(default=True)

    def __post_init__(self):
        self.district = self.district["name"].encode("utf-8").decode("utf-8").lower()
        self.county = self.county["name"].encode("utf-8").decode("utf-8").lower()
        self.city = self.city["name"].encode("utf-8").decode("utf-8").lower()
        self.room = sum(self.room)
        self.living_room = sum(self.living_room)
        self.latitude = self._map_location["lat"]
        self.longitude = self._map_location["lon"]
        self.created_at = datetime.strptime(
            self.created_at.split(".")[0], "%Y-%m-%dT%H:%M:%S"
        )
        self.updated_at = datetime.strptime(
            self.updated_at.split(".")[0], "%Y-%m-%dT%H:%M:%S"
        )
        self.net_sqm = self._sqm["netSqm"]
        self.gross_sqm = sum(self._sqm["grossSqm"])
        _temp = self.floor
        self.floor = _temp["name"] if _temp else None
        self.total_floor = _temp["count"] if _temp else None
        self.heating = (
            self.heating["name"].encode("utf-8").decode("utf-8").lower()
            if self.heating
            else None
        )
        self.fuel = self.sanitize(self.fuel, "name")
        self.usage = self.sanitize(self.usage, "name")
        self.credit = self.sanitize(self.credit, "name")
        self.deposit = int(self.deposit["amount"]) if self.deposit else None
        self.realty_type = RealtyType(self.sanitize(self.realty_type, "typeName"))
        self.listing_category = ListingCategory(
            self.sanitize(self.listing_category, "linkedCategoryUrl")
        )
        self._attributes = (
            self.get_attributes(self._attributes) if self._attributes else None
        )

    def __setattr__(self, name, value):
        if name == "price":
            assert (
                value < 2_147_483_647
            ), f"value of {name} can't be higher than integer limit 2_147_483_647 : {value}"
        self.__dict__[name] = value

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
                    f"{camel_to_snake(category)}_{attribute['name'].encode('utf-8').decode('utf-8').lower()}".replace(
                        " ", "_"
                    )
                    .replace("-", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "_")
                    .replace("'", "")
                    .replace("`", "")
                    .replace(":", "")
                    .replace(";", "")
                    .replace(".", "")
                    .replace(",", "")
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
        result = {}
        for key, item in self.__dict__.items():
            if not key.startswith("_"):
                _field = getattr(self, key)
                if hasattr(_field, "value"):
                    # Enum field case (e.g. ListingCategory) -> get value of enum field
                    result[key] = _field.value
                else:
                    result[key] = item

        return result

    def save(self):
        house = HouseModel(**self.to_dict())
        if self._attributes is None:
            session.add(house)
            session.commit()
            return

        for attr in self._attributes:
            attributes = session.query(Attribute).filter_by(name=attr)
            number_of_attrs = attributes.count()
            if number_of_attrs > 1:
                error_message = f"Duplicate attribute: {attr}"
                logger.error(error_message)
                raise Exception(error_message)
            elif number_of_attrs == 0:
                attribute = Attribute(name=attr, value=1)
            else:
                attribute = attributes.first()
            # attribute._house.append(house)
            house._attribute.append(attribute)
            session.add(attribute)
        session.add(house)
        session.commit()

    def save_or_update(self):
        existing_houses = session.query(HouseModel).filter_by(
            internal_id=self.internal_id,
            data_source=DataSource.HEPSI.value,
            is_last_version=True,
        )
        if existing_houses.count() == 0:
            self.save()
            logger.info(f"House {self.internal_id} saved")
        elif existing_houses.count() == 1 and existing_houses.first().price == self.price:
            existing_houses.update(self.to_dict())
            session.commit()
            logger.info(f"House {self.internal_id} updated")
        elif existing_houses.count() == 1 and existing_houses.first().price != self.price:
            existing_house = existing_houses.first()
            self.version = existing_house.version + 1
            self.save()
            existing_house.is_last_version = False
            existing_house.is_active = False
            session.commit()
            
            logger.info(
                f"House {self.internal_id} updated. Old price: {existing_house.price}, new price: {self.price}, version: {self.version}"
            )
        elif existing_houses.count() > 1:
            for existing_house in existing_houses:
                session.query(HouseAttribute).filter_by(house_id=existing_house.id).delete()
            logger.warning(f"Duplicate houses deleted: {[h for h in existing_houses]}")
            existing_houses.delete()
        else:
            logger.error(f"THERE IS AN UNEXPECTED ERROR: {[h for h in existing_houses]}")
        return


def main_coroutine():
    _files = follow()  # generator of files
    for files in _files:
        for file in files:
            try:
                with open(file.path, "r") as f:
                    data = json.loads(f.read())["realtyDetail"]
                    house = House.from_dict(data)
                    house.save_or_update()
                os.remove(file.path)
                logger.info(f"File {file.name} removed")
            except AssertionError as e:
                logger.error(
                    f"Error on {file.name} integration. Error: {get_traceback(e)}"
                )
                os.remove(file.path)
                session.rollback()
            except Exception as e:
                logger.error(
                    f"Error on {file.name} integration. Error: {get_traceback(e)}"
                )
                session.rollback()
            finally:
                session.close()
                continue


if __name__ == "__main__":
    main_coroutine()
