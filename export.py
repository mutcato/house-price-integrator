import csv
from database import session
from models import *
from sqlalchemy.sql import text
from datetime import datetime
from settings import config as settings

class Flattener:
    qs = session.query(Attribute.name).all()
    
    @property
    def attributes(self):
        for attr in self.qs:
            yield attr[0] 

    def get_sql(self):
        select_attributes = ",".join([f"""MAX(CASE WHEN ta.name = '{attr}' THEN ta.value ELSE NULL END) AS {attr}""" for attr in self.attributes])
        statement = f"""
            SELECT 
                th.*,
                {select_attributes}
            FROM 
                houses th
                LEFT JOIN house_attributes tha ON th.id = tha.house_id
                LEFT JOIN attributes ta ON tha.attribute_id = ta.id
            GROUP BY 
                th.id, th.url, th.district, th.city
            ORDER BY inserted_at DESC;
        """
        return statement

    def format(self):
        rows = session.execute(text(self.get_sql()))
        columns = rows.keys()._keys
        result = [dict(zip(columns, row)) for row in rows]
        file = f"""{settings["EXPORT_PATH"]}/{str(datetime.today().date())}.csv"""
        with open(file, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, columns)
            dict_writer.writeheader()
            dict_writer.writerows(result)


if __name__ == "__main__":
    flatter = Flattener()
    flatter.format()
    attrs = flatter.attributes
