import urllib.request

import psycopg
import pytest

from rag_engine import db
from rag_engine.config import settings

pytestmark = pytest.mark.integration

DEMO_TABLES = ["clusters", "nodes", "deployments", "pods", "incidents", "alerts", "oncall_logs"]


@pytest.fixture
def readonly_conn():
    with psycopg.connect(settings.sql_database_urls["k8s_ops"]) as conn:
        yield conn


def test_engine_database_is_reachable():
    assert db.ping()


def test_readonly_can_read_every_demo_table(readonly_conn):
    for table in DEMO_TABLES:
        count = readonly_conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert count > 0, table


def test_readonly_cannot_write(readonly_conn):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        readonly_conn.execute("DELETE FROM clusters")


def test_readonly_cannot_create_tables(readonly_conn):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        readonly_conn.execute("CREATE TABLE sneaky (id int)")


def test_readonly_cannot_connect_to_engine_database():
    readonly_url = settings.sql_database_urls["k8s_ops"].rsplit("/", 1)[0] + "/rag_engine"
    with pytest.raises(psycopg.OperationalError):
        psycopg.connect(readonly_url, connect_timeout=2)


def test_qdrant_is_ready():
    with urllib.request.urlopen(f"{settings.qdrant_url}/readyz", timeout=2) as response:
        assert response.status == 200
