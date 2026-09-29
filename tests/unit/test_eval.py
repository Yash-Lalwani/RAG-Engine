import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import diff  # noqa: E402
from metrics import (  # noqa: E402
    average,
    forbidden_hits,
    recall_at_k,
    reciprocal_rank,
    sql_execution_match,
)
from run_eval import cost_usd, load_goldens  # noqa: E402

from rag_engine.sql.safety import validate_sql  # noqa: E402

DEMO_TABLES = ["clusters", "nodes", "deployments", "pods", "incidents", "alerts", "oncall_logs"]


def test_recall_counts_documents_not_chunks():
    retrieved = ["a.md", "a.md", "b.md", "x.md", "y.md", "z.md", "c.md"]
    assert recall_at_k(retrieved, ["a.md", "c.md"], k=5) == 0.5  # c.md is the 6th document
    assert recall_at_k(retrieved, ["b.md"]) == 1.0
    assert recall_at_k(retrieved, []) is None


def test_reciprocal_rank():
    assert reciprocal_rank(["x.md", "x.md", "a.md"], ["a.md"]) == 0.5
    assert reciprocal_rank(["x.md"], ["a.md"]) == 0.0
    assert reciprocal_rank(["a.md"], []) is None


def test_sql_match_ignores_column_names_and_allows_extra_columns():
    reference = [{"name": "cluster-1"}]
    assert sql_execution_match([{"cluster": "cluster-1", "pods": 99}], reference)
    assert not sql_execution_match([{"cluster": "cluster-2", "pods": 99}], reference)
    assert not sql_execution_match([{"c": "cluster-1"}, {"c": "cluster-2"}], reference)
    assert sql_execution_match([{"count": 392.0}], [{"count": 392}])
    assert sql_execution_match([{"a": 2}, {"a": 1}], [{"x": 1}, {"x": 2}])


def test_forbidden_hits_and_average():
    assert forbidden_hits("Use your KubeConfig file", ["kubeconfig", "token"]) == ["kubeconfig"]
    assert average([1.0, None, 0.5]) == 0.75 and average([None]) is None


def test_cost_uses_the_most_specific_model_price():
    usage = {"gpt-4o-mini-2024-07-18": {"prompt": 1_000_000, "completion": 0},
             "gpt-4o-2024-08-06": {"prompt": 0, "completion": 1_000_000}}
    assert cost_usd(usage) == pytest.approx(0.15 + 10.00)


def test_goldens_are_valid():
    goldens = load_goldens()
    assert len(goldens) == 40
    for golden in goldens:
        if golden.intent in ("sql", "hybrid"):
            assert golden.reference_sql and validate_sql(golden.reference_sql, DEMO_TABLES)
        if golden.intent in ("rag", "hybrid"):
            assert golden.sources, golden.id


def test_diff_reports_summary_deltas_and_moved_goldens():
    def run(run_id, recall, faithfulness):
        row = {"id": "q-001", "recall_at_5": recall, "mrr": 1.0, "sql_match": None,
               "ragas": {"faithfulness": faithfulness}}
        summary = {m: None for m in diff.METRICS} | {"recall_at_5": recall, "faithfulness": faithfulness}
        return {"run_id": run_id, "profiles": {"all": {"summary": summary, "rows": [row]}}}

    old, new = run("old", 0.5, 0.9), run("new", 1.0, 0.95)
    assert "1.0 (+0.500)" in "\n".join(diff.summary_deltas(old, new))
    assert diff.changed_goldens(old, new) == ["- all q-001 recall_at_5: 0.5 -> 1.0"]
