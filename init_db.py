"""
Runs schema.sql to create scout_db and its tables.
Usage: python db/init_db.py
"""
import pymysql
from db_connect import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL_CA

SCHEMA_PATH = "schema.sql"

def run_schema():
    ssl_args = {"ssl": {"ca": DB_SSL_CA}} if DB_SSL_CA else {}
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,   # connect directly to defaultdb — no CREATE DATABASE needed
        autocommit=True,
        **ssl_args,
    )
    with open(SCHEMA_PATH, "r") as f:
        sql_script = f.read()

    with conn.cursor() as cur:
        for statement in sql_script.split(";"):
            statement = statement.strip()
            if statement:
                cur.execute(statement)
    conn.close()
    print("scout_db schema created successfully.")

if __name__ == "__main__":
    run_schema()