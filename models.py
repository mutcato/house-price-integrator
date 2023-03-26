from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class House(Base):
    __tablename__ = "houses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    internal_id = Column(Integer, index=True)
    data_source = Column(String, index=True)
    url = Column(String, unique=True,index=True)
    version = Column(Integer, default=1)
    is_last_version = Column(Boolean, default=True)
    price = Column(Integer, index=True)
    currency = Column(String, index=True)
    predicted_price = Column(Integer, index=True, default=None, nullable=True)
    predicted_rental_price = Column(Integer, index=True, default=None, nullable=True)
    created_at = Column(String, index=True)
    updated_at = Column(String, index=True)
    inserted_at = Column(String, index=True)
    is_active = Column(Boolean, default=True)
    attribute = relationship("Attribute", back_populates="houses", secondary="house_attributes")

    def __repr__(self):
        return f"House(internal_id={self.internal_id}, data_source={self.data_source}, price={self.price}, currency={self.currency}, predicted_price={self.predicted_price})"
    
    def __str__(self):
        return f"House(internal_id={self.internal_id}, data_source={self.data_source}, price={self.price}, currency={self.currency}, predicted_price={self.predicted_price})"
    

class Attribute(Base):
    __tablename__ = "attributes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, index=True)
    value = Column(String, index=True)
    house = relationship("House", back_populates="attributes", secondary="house_attributes")

    def __repr__(self):
        return f"Attributes(name={self.name}, value={self.value})"
    
    def __str__(self):
        return f"Attributes(name={self.name}, value={self.value})"
    
class HouseAttribute(Base):
    __tablename__ = "house_attributes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    house_id = Column(Integer, ForeignKey("houses.id"))
    attribute_id = Column(Integer, ForeignKey("attributes.id"))


    def __repr__(self):
        return f"HouseAttribute(house={self.house}, attribute={self.attribute})"
    
    def __str__(self):
        return f"HouseAttribute(house={self.house}, attribute={self.attribute})"