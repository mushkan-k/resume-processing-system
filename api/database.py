"""Database connection helper for the API"""
import mysql.connector
from mysql.connector import pooling
from pathlib import Path
from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

db_pool = pooling.MySQLConnectionPool(
    pool_name="api_pool",
    pool_size=5,
    host=os.getenv("MYSQL_HOST", "127.0.0.1"),
    port=int(os.getenv("MYSQL_PORT", "3305")),
    database=os.getenv("MYSQL_DATABASE", "resume_processing"),
    user=os.getenv("MYSQL_USER", "resume_user"),
    password=os.getenv("MYSQL_PASSWORD", "resume_password"),
    charset="utf8mb4"
)


def get_db():
    """Get a connection from the pool"""
    conn = db_pool.get_connection()
    try:
        yield conn
    finally:
        conn.close()