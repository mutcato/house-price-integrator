import argparse
import csv
import logging
from datetime import datetime

from sqlalchemy.sql import text

from database import session
from models import *
from settings import config as settings

logger = logging.getLogger("offices")


parser = argparse.ArgumentParser(description="Script for export csv files.")
parser.add_argument("--listing", type=str, help="Listing category", default="satilik")
parser.add_argument(
    "--export_path", type=str, help="Export path", default=settings["EXPORT_PATH"]
)
parser.add_argument("--price_min", type=str, help="Minimum price", default=100)
args = parser.parse_args()


class Flattener:
    qs = session.query(Attribute.name).all()

    @property
    def attributes(self):
        for attr in self.qs:
            yield attr[0]

    def get_sql(self):
        select_attributes = ",".join(
            [
                f"""MAX(CASE WHEN attr.name = '{attr}' THEN attr.value ELSE NULL END) AS {attr}"""
                for attr in self.attributes
            ]
        )
        statement = f"""
            SELECT 
                h.*,
                {select_attributes}
            FROM 
                houses h
                LEFT JOIN house_attributes ha ON h.id = ha.house_id
                LEFT JOIN attributes attr ON ha.attribute_id = attr.id
            WHERE h.listing_category = '{args.listing}' AND h.price > {args.price_min} AND h.currency = 'TL' AND h.is_active = true
            GROUP BY 
                h.id
            ORDER BY inserted_at DESC;
        """
        return statement

    def format(self):
        rows = session.execute(text(self.get_sql()))
        columns = rows.keys()._keys
        result = [dict(zip(columns, row)) for row in rows]
        file = f"""{args.export_path}/{str(datetime.today().date())}-sanitized-{args.listing}.csv"""
        with open(file, "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, columns)
            dict_writer.writeheader()
            dict_writer.writerows(result)


"""
TODO: csv oluştururken her seferinde en baştan oluşturmasın.
Günlük olarak alsın, önceki gün kaldığı yerden devam etsin.
Drive'a yollasın.
"""


if __name__ == "__main__":
    logger.info(f"{args.listing} export started")
    flatter = Flattener()
    flatter.format()
    attrs = flatter.attributes
    logger.info(f"{args.listing} export ended")
