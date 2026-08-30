"""
SCOUT — MySQL connection utilities.
Password contains '@' (garvit@123) — never put it raw inside a connection URI string.
Use pymysql.connect() with the password as a plain kwarg (no encoding needed),
or urllib.parse.quote_plus() if you build a SQLAlchemy URI string.
"""

import os
import pymysql
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

# --- Config (move to .env in production, don't hardcode) ---
DB_HOST = os.getenv("SCOUT_DB_HOST", "localhost")
DB_PORT = int(os.getenv("SCOUT_DB_PORT", 3306))
DB_USER = os.getenv("SCOUT_DB_USER", "root")
DB_PASSWORD = os.getenv("SCOUT_DB_PASSWORD", "garvit@123")
DB_NAME = os.getenv("SCOUT_DB_NAME", "scout_db")
DB_SSL_CA = os.getenv("SCOUT_DB_SSL_CA")  # path to Aiven's ca.pem, if using SSL


def get_pymysql_connection():
    """Raw connection — good for simple inserts/queries, cursors."""
    ssl_args = {"ssl": {"ca": DB_SSL_CA}} if DB_SSL_CA else {}
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        **ssl_args,
    )


def get_sqlalchemy_engine():
    """
    SQLAlchemy engine — needed for pandas.to_sql()/read_sql().
    The password IS part of a URI string here, so '@' must be percent-encoded
    with quote_plus, otherwise SQLAlchemy misreads the host section.
    """
    safe_password = quote_plus(DB_PASSWORD)
    uri = f"mysql+pymysql://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    connect_args = {"ssl": {"ca": DB_SSL_CA}} if DB_SSL_CA else {}
    return create_engine(uri, connect_args=connect_args)


if __name__ == "__main__":
    conn = get_pymysql_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT DATABASE(), VERSION();")
        print(cur.fetchone())
    conn.close()
    print("pymysql connection OK")

    engine = get_sqlalchemy_engine()
    with engine.connect() as c:
        print("SQLAlchemy engine OK:", engine.url)