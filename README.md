# house-price-integrator
Inserts scraped houses into data storage

### To run PG Admin in development environment
docker run -p 5050:80 -e "PGADMIN_DEFAULT_EMAIL=example@example.com" -e "PGADMIN_DEFAULT_PASSWORD=123456" -d dpage/pgadmin4

### Alembic commands
alembic revision --autogenerate -m "Add Type and Category fields"
alembic upgrade head