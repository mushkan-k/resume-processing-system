"""MCP Server Configuration – reads from project-root .env"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


class Config:
    MYSQL_HOST     = os.getenv("MYSQL_HOST",     "127.0.0.1")
    MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3305"))
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "resume_processing")
    MYSQL_USER     = os.getenv("MYSQL_USER",     "resume_user")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "resume_password")