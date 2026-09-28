import psycopg

from rag_engine.config import settings


def connect(url: str | None = None, **kwargs) -> psycopg.Connection:
    """Open a Postgres connection; defaults to the Engine's own database."""
    return psycopg.connect(url or settings.database_url, **kwargs)


def ping(url: str | None = None) -> bool:
    try:
        with connect(url, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.Error:
        return False
