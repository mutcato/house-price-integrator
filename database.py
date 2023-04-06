from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import settings

config = settings.config
SQLALCHEMY_DATABASE_URL = f"postgresql://{config['DATABASE_USER']}:{config['DATABASE_PASSWORD']}@{config['DATABASE_HOST']}/{config['DATABASE_NAME']}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

Base = declarative_base()

# conn = psycopg2.connect(
#     database=config["DATABASE_NAME"],
#     user=config["DATABASE_USER"],
#     password=config["DATABASE_PASSWORD"],
#     host=config["DATABASE_HOST"],
# )

# cur = conn.cursor()
