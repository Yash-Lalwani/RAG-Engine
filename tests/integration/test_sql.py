import time

import pytest

from rag_engine.cache.store import cache
from rag_engine.models import EngineError
from rag_engine.sql.executor import run_sql
from rag_engine.sql.schema import describe_schema

pytestmark = pytest.mark.integration

DB = "k8s_ops"
TABLES = ["clusters", "nodes", "deployments", "pods", "incidents", "alerts", "oncall_logs"]


def test_pg_sleep_times_out():
    started = time.perf_counter()
    with pytest.raises(EngineError, match="timed out"):
        run_sql("SELECT pg_sleep(600) FROM clusters LIMIT 1", DB, TABLES)
    assert time.perf_counter() - started < 10


def test_a_select_cannot_switch_off_its_own_timeout():
    with pytest.raises(EngineError, match="timed out"):
        run_sql(
            "SELECT set_config('statement_timeout', '0', false), pg_sleep(600) FROM clusters LIMIT 1",
            DB, TABLES,
        )


def test_non_allowlisted_tables_fail_in_the_safety_check():
    with pytest.raises(EngineError, match="not an allowed table"):
        run_sql("SELECT * FROM pg_roles", DB, TABLES)
    with pytest.raises(EngineError, match="not an allowed table"):
        run_sql("SELECT * FROM nodes", DB, ["clusters"])


def test_the_read_only_role_blocks_what_the_allowlist_would_let_through():
    with pytest.raises(EngineError, match="permission denied"):
        run_sql("SELECT * FROM pg_authid", DB, ["pg_authid"])


def test_cte_query_works():
    result = run_sql(
        "WITH p1 AS (SELECT * FROM incidents WHERE severity = 'P1') SELECT count(*) AS n FROM p1",
        DB, TABLES,
    )
    assert result.columns == ["n"] and result.rows[0]["n"] > 0 and not result.truncated


def test_limit_is_added_and_large_limits_are_truncated():
    default = run_sql("SELECT alert_id, fired_at FROM alerts", DB, TABLES)
    assert default.row_count == 200 and not default.truncated and default.sql.endswith("LIMIT 200")
    assert isinstance(default.rows[0]["fired_at"], str)
    big = run_sql("SELECT alert_id FROM alerts LIMIT 5000", DB, TABLES)
    assert big.row_count == 200 and big.truncated


def test_results_are_cached():
    before = cache.stats().get("sql_result", {}).get("hits", 0)
    sql = "SELECT count(*) AS n FROM clusters WHERE environment = 'production'"
    assert run_sql(sql, DB, TABLES) == run_sql(sql.replace(" ", "  "), DB, TABLES)
    assert cache.stats()["sql_result"]["hits"] == before + 1


def test_schema_lists_only_allowed_tables_and_their_foreign_keys():
    schema = describe_schema(DB, ("clusters", "nodes"))
    assert "clusters(cluster_id integer" in schema and "nodes(" in schema
    assert "pods(" not in schema
    assert "REFERENCES clusters(cluster_id)" in schema


def test_unknown_database_and_missing_table_are_clear_errors():
    with pytest.raises(EngineError, match="not configured"):
        run_sql("SELECT 1 FROM clusters", "nope", TABLES)
    with pytest.raises(EngineError, match="not found"):
        describe_schema(DB, ("clusters", "no_such_table"))
