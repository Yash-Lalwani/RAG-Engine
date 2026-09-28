import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from rag_engine.graph import builder, nodes
from rag_engine.models import (
    CollectionSettings,
)


def start(core, intent="rag", **settings):
    core.intent = intent
    graph = builder.build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = {
        "query_id": "q1", "caller": "dev", "created_at": "2026-01-01T00:00:00+00:00",
        "collection_id": "c", "question": "How many pods crash?", "search_query": "How many pods crash?",
        "options": {}, "filters": {"team": "a"}, "retries": 0, "warnings": [],
        "settings": CollectionSettings(
            sql_database="k8s_ops", sql_allowed_tables=["pods"], **settings
        ).model_dump(),
    }
    return graph, config, graph.invoke(state, config)


def test_rag_path_runs_search_generate_verify_and_records_metadata(core):
    graph, config, final = start(core, "rag")
    assert final["status"] == "completed" and "answer from 1 chunks" in final["final_answer"]
    assert "run_sql" not in core.calls and "generate_sql" not in core.calls
    assert core.calls["search"][0]["filters"] == {"team": "a"}
    assert final["verification"] is not None and final["search_info"]["crag_action"] == "correct"
    assert {"route_intent", "retrieve", "generate", "verify", "finalize"} <= set(final["timings_ms"])


def test_hybrid_never_runs_sql_before_approval(core):
    graph, config, paused = start(core, "hybrid")
    assert paused["__interrupt__"][0].value == {"sql": "SELECT COUNT(*) FROM pods LIMIT 200", "explanation": "Counts pods."}
    assert "run_sql" not in core.calls and "generate_answer" not in core.calls
    assert graph.get_state(config).next == ("approve",)


def test_hybrid_approved_uses_rows_and_chunks(core):
    graph, config, _ = start(core, "hybrid")
    final = graph.invoke(Command(resume={"approved": True}), config)
    assert len(core.calls["run_sql"]) == 1
    generated = core.calls["generate_answer"][0]
    assert generated["sql_result"].row_count == 60 and len(generated["sql_result"].rows) == 60
    assert len(generated["chunks"]) == 1 and "verify_citations" in core.calls
    assert "sql_results" in core.calls["verify_citations"][0]["passage_ids"]
    assert final["status"] == "completed" and final["sql_row_count"] == 60


def test_hybrid_rejected_answers_from_documents_only(core):
    graph, config, _ = start(core, "hybrid")
    final = graph.invoke(Command(resume={"approved": False}), config)
    assert "run_sql" not in core.calls
    assert core.calls["generate_answer"][0]["sql_result"] is None
    assert final["status"] == "completed" and "not approved" in final["warnings"][-1]


def test_sql_rejected_finishes_with_query_not_approved(core):
    graph, config, _ = start(core, "sql")
    assert "search" not in core.calls
    final = graph.invoke(Command(resume={"approved": False}), config)
    assert "run_sql" not in core.calls and "generate_answer" not in core.calls
    assert final["status"] == "completed" and final["final_answer"] == nodes.NOT_APPROVED


def test_sql_approved_answers_from_rows_without_verification(core):
    graph, config, _ = start(core, "sql")
    final = graph.invoke(Command(resume={"approved": True}), config)
    assert core.calls["generate_answer"][0]["chunks"] == []
    assert "verify_citations" not in core.calls and final["status"] == "completed"


@pytest.mark.parametrize("intent", ["sql", "hybrid"])
def test_unsafe_sql_is_never_offered_for_approval(core, intent):
    core.sql = "DELETE FROM pods"
    graph, config, final = start(core, intent)
    assert "__interrupt__" not in final and "run_sql" not in core.calls
    if intent == "sql":
        assert final["status"] == "error" and "Only SELECT" in final["message"]
    else:
        assert final["status"] == "completed" and "Answered from documents only" in final["warnings"][-1]


@pytest.mark.parametrize("intent", ["sql", "hybrid"])
def test_sql_execution_failure(core, intent):
    core.sql_fails = True
    graph, config, _ = start(core, intent)
    final = graph.invoke(Command(resume={"approved": True}), config)
    if intent == "sql":
        assert final["status"] == "error" and "timed out" in final["message"]
    else:
        assert final["status"] == "completed" and "timed out" in final["warnings"][-1]


def test_self_rag_retry_keeps_the_better_answer_and_answers_the_original_question(core):
    core.scores = [0.4, 0.8]
    graph, config, final = start(core, "rag", self_rag=True)
    assert [c["query"] for c in core.calls["search"]] == ["How many pods crash?", "better query"]
    assert {c["question"] for c in core.calls["generate_answer"]} == {"How many pods crash?"}
    assert final["self_rag"] == {"first_score": 0.4, "retried": True, "retry_score": 0.8,
                                 "kept": "retry", "retry_query": "better query"}
    assert final["chunks"][0]["id"] == "chunk-better query"


def test_self_rag_keeps_the_first_answer_when_the_retry_is_not_better(core):
    core.scores = [0.5, 0.5]
    graph, config, final = start(core, "rag", self_rag=True)
    assert final["self_rag"]["kept"] == "first" and len(core.calls["self_check"]) == 2
    assert final["chunks"][0]["id"] == "chunk-How many pods crash?"


def test_self_rag_does_not_retry_a_good_answer_or_when_off(core):
    core.scores = [0.95]
    _, _, final = start(core, "rag", self_rag=True)
    assert len(core.calls["search"]) == 1 and final["self_rag"] == {"first_score": 0.95}
    core.calls.clear()
    start(core, "rag", self_rag=False)
    assert "self_check" not in core.calls


def test_router_failure_falls_back_to_documents(core, monkeypatch):
    def broken(*args):
        raise RuntimeError("OpenAI timeout")

    monkeypatch.setattr(nodes, "classify_intent", broken)
    _, _, final = start(core, "sql")
    assert final["intent"] == "rag" and "Intent routing failed" in final["warnings"][0]


def test_undeclared_state_keys_are_refused():
    with pytest.raises(RuntimeError, match="undeclared state keys"):
        nodes.tracked("bad", lambda state: {"metadata": {}})({})
