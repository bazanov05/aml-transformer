import psycopg_pool
import os
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

pool = None     


def init_pool():
    global pool
    host = (os.getenv('DB_HOST') or '127.0.0.1').strip()
    
    if host == 'localhost':
        host = '127.0.0.1'
        
    port = (os.getenv('DB_PORT') or '5432').strip()
    dbname = (os.getenv('DB_NAME') or 'aml_db').strip()
    user = (os.getenv('DB_USER') or 'aml_user').strip()
    password = (os.getenv('DB_PASSWORD') or 'aml_password').strip()
    
    conninfo = f"host={host} port={port} dbname={dbname} user={user} password={password} connect_timeout=10"
    
    try:
        pool = psycopg_pool.ConnectionPool(conninfo=conninfo, min_size=1, max_size=5, timeout=30)
        print("Connected with PostgreSQL!")
    except Exception as e:
        print(f"Error with connecting to DB: {e}")
        raise RuntimeError("Cannot init connection pool.") from e


def close_pool():
    global pool
    pool.close()


def get_db():
    with pool.connection() as conn: # borrow connection from pool
        conn.row_factory = dict_row   # now quieries will return dict instead of tuple
        yield conn      # give this conn to FastAPI, get_db() freezes here 
    