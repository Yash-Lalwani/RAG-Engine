import pytest

from rag_engine.models import EngineError
from rag_engine.sql.safety import validate_sql

ALLOWED = ["clusters", "nodes", "incidents"]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM clusters WHERE environment = 'production'",
        "SELECT * FROM incidents WHERE rca_summary ILIKE '%update%' OR rca_summary LIKE '%drop table%'",
        "WITH p1 AS (SELECT * FROM incidents WHERE severity = 'P1') SELECT count(*) FROM p1",
        "WITH a AS (SELECT * FROM clusters), b AS (SELECT * FROM a) SELECT * FROM b",
        "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t WHERE n < 5) SELECT * FROM t",
        "SELECT cluster_id FROM clusters UNION SELECT cluster_id FROM nodes",
        "SELECT c.name, count(*) FROM clusters c JOIN nodes n ON n.cluster_id = c.cluster_id GROUP BY 1",
        "SELECT * FROM public.clusters",
        "SELECT * FROM CLUSTERS",
        "select * from clusters;",
    ],
)
def test_valid_read_only_queries_pass(sql):
    assert validate_sql(sql, ALLOWED)


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("SELECT 1 FROM clusters; DROP TABLE clusters", "Exactly one"),
        ("DELETE FROM clusters", "Only SELECT"),
        ("UPDATE clusters SET name = 'x'", "Only SELECT"),
        ("INSERT INTO clusters (name) VALUES ('x')", "Only SELECT"),
        ("DROP TABLE clusters", "Only SELECT"),
        ("WITH d AS (DELETE FROM clusters RETURNING *) SELECT * FROM d", "change data"),
        ("SELECT * INTO copy_of_clusters FROM clusters", "INTO"),
        ("SELECT * FROM clusters FOR UPDATE", "Locking"),
        ("SELECT * FROM users", "not an allowed table"),
        ("SELECT * FROM (SELECT * FROM users) u", "not an allowed table"),
        ("SELECT * FROM pg_catalog.pg_roles", "not an allowed table"),
        ("SELECT * FROM pg_roles", "not an allowed table"),
        ("SELECT * FROM information_schema.tables", "not an allowed table"),
        ("SELECT * FROM other_schema.clusters", "not an allowed table"),
        ("SELECT * FROM k8s_ops.public.clusters", "not an allowed table"),
        ("WITH pg_roles AS (SELECT * FROM pg_roles) SELECT * FROM pg_roles", "not an allowed table"),
        ("SELECT * FROM generate_series(1, 1000000000)", "not an allowed table"),
        ("SELEC nme FRM clusters", "could not be parsed"),
        ("", "Exactly one"),
    ],
)
def test_unsafe_or_invalid_queries_are_rejected(sql, reason):
    with pytest.raises(EngineError, match=reason):
        validate_sql(sql, ALLOWED)


def test_limit_is_added_when_missing_and_kept_when_present():
    assert validate_sql("SELECT name FROM clusters", ALLOWED).endswith("LIMIT 200")
    assert validate_sql("SELECT name FROM clusters LIMIT 5", ALLOWED).endswith("LIMIT 5")
    union = validate_sql("SELECT 1 FROM clusters UNION SELECT 2 FROM nodes", ALLOWED)
    assert union.endswith("LIMIT 200")


def test_output_is_normalized_but_keeps_string_case():
    sql = validate_sql("select   name from clusters where environment='Prod'", ALLOWED)
    assert sql == "SELECT name FROM clusters WHERE environment = 'Prod' LIMIT 200"
