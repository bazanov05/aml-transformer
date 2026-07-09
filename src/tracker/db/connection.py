import psycopg_pool
import os
from dotenv import load_dotenv

load_dotenv()

pool = None     


def init_pool():
    global pool
    conninfo = f"host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')} dbname={os.getenv('DB_NAME')} user={os.getenv('DB_USER')} password={os.getenv('DB_PASSWORD')}"
    pool = psycopg_pool.ConnectionPool(conninfo=conninfo)


def close_pool():
    global pool
    pool.close()


def get_db():
    with pool.connection() as conn: # borrow connection from pool
        yield conn      # give this conn to FastAPI, get_db() freezes here 
    