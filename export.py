import csv
from datetime import datetime

from sqlalchemy.sql import text

from database import session
from models import *
from settings import config as settings


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
            WHERE h.listing_category = 'satilik' AND h.price > 100000 AND h.currency = 'TL'
            GROUP BY 
                h.id
            ORDER BY inserted_at DESC;
        """
        return statement

    def format(self):
        rows = session.execute(text(self.get_sql()))
        columns = rows.keys()._keys
        result = [dict(zip(columns, row)) for row in rows]
        file = f"""{settings["EXPORT_PATH"]}/{str(datetime.today().date())}-sanitized-satilik.csv"""
        with open(file, "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, columns)
            dict_writer.writeheader()
            dict_writer.writerows(result)


"""
TODO: csv oluştururken her seferinde en baştan oluşturmasın.
Günlük olarak alsın, önceki gün kaldığı yerden devam etsin.
Drive'a yollasın.

TODO: Manuel yapılan işler:
100.000 TL'de ucuz evleri elle siliyorsun.
Para birimi TL dışında olan evleri siliyorsun. Kur dönüştürme yok.
"""


if __name__ == "__main__":
    flatter = Flattener()
    flatter.format()
    attrs = flatter.attributes
