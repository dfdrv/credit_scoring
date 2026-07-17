import os

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATA_FULL_PATH = r"C:/repositories/shift/home-credit-default-risk"
DATABASE_NAME = "home_credit_default_risk"
DB_HOST = "127.0.0.1"
DB_PORT = 5432

DB_ARGS = {
    "database": DATABASE_NAME,
    "host": DB_HOST,
    "port": DB_PORT,
    "user": DB_USER,
    "password": DB_PASSWORD,
}