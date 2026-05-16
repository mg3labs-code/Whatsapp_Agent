import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set; cannot initialise database engine"
    )

# connect_timeout (passed via connect_args) caps how long a single TCP connection
# attempt to Postgres will block before raising an OperationalError, preventing
# the engine from hanging indefinitely when the database is unreachable.
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    echo=False,
    connect_args={"connect_timeout": 10},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
