"""Database connection – context manager used by MCP server tools."""

import sys
import mysql.connector
from contextlib import contextmanager
from mcp_server.config import Config


@contextmanager
def get_db_connection():
    """
    Yields a mysql.connector connection.
    Auto-commits on success, rolls back on exception, always closes.
    """
    conn = None
    try:
        conn = mysql.connector.connect(
            host     = Config.MYSQL_HOST,
            port     = Config.MYSQL_PORT,
            database = Config.MYSQL_DATABASE,
            user     = Config.MYSQL_USER,
            password = Config.MYSQL_PASSWORD,
            charset  = "utf8mb4",
        )
        print(f"[MCP-DB] Connected to {Config.MYSQL_DATABASE}", file=sys.stderr)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[MCP-DB ERROR] {e}", file=sys.stderr)
        raise
    finally:
        if conn:
            conn.close()