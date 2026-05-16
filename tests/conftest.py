"""Pytest bootstrap — load .env before app imports that need DATABASE_URL."""

from dotenv import load_dotenv

load_dotenv()
