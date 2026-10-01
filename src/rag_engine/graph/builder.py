"""The ask() graph: which node runs after which. Built once, at engine.setup()."""

from functools import lru_cache

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from rag_engine.config import settings
from rag_engine.grading.self_rag import should_retry
from rag_engine.graph import nodes
from rag_engine.graph.state import AskState
from rag_engine.models import CollectionSettings

NODES = {
    "route_intent": nodes.route_intent_node,
    "retrieve": nodes.retrieve_node,
    "generate": nodes.generate_node,
    "self_check": nodes.self_check_node,
    "rewrite": nodes.rewrite_node,
    "generate_sql": nodes.generate_sql_node,
    "validate_sql": nodes.validate_sql_node,
    "approve": nodes.approve_node,
    "execute_sql": nodes.execute_sql_node,
    "verify": nodes.verify_node,
    "finalize": nodes.finalize_node,
}


def build_graph(checkpointer: BaseCheckpointSaver):
    """
    rag:    retrieve -> generate -> [self_check -> (rewrite -> retrieve -> generate -> self_check)]
            -> verify -> finalize
    sql:    generate_sql -> validate_sql -> approve (pause) -> execute_sql -> generate -> finalize
    hybrid: retrieve -> generate_sql -> validate_sql -> approve (pause) -> execute_sql -> generate
            -> verify -> finalize
    """
    builder = StateGraph(AskState)
    for name, node in NODES.items():
        builder.add_node(name, nodes.tracked(name, node))

    builder.add_edge(START, "route_intent")
    builder.add_conditional_edges("route_intent", after_route_intent, ["retrieve", "generate_sql"])
    builder.add_conditional_edges("retrieve", after_retrieve, ["generate", "generate_sql"])
    builder.add_edge("generate_sql", "validate_sql")
    builder.add_conditional_edges("validate_sql", after_validate_sql, ["approve", "generate", "finalize"])
    builder.add_conditional_edges("approve", after_approve, ["execute_sql", "generate", "finalize"])
    builder.add_conditional_edges("execute_sql", after_execute_sql, ["generate", "finalize"])
    builder.add_conditional_edges("generate", after_generate, ["self_check", "verify", "finalize"])
    builder.add_conditional_edges("self_check", after_self_check, ["rewrite", "verify"])
    builder.add_conditional_edges("rewrite", after_rewrite, ["retrieve", "verify"])
    builder.add_edge("verify", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


@lru_cache
def get_graph():
    """The graph with a Postgres checkpointer (used only while a run waits for SQL approval).
    A small pool replaces connections the database has closed (idle timeouts, restarts)."""
    pool = ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        check=ConnectionPool.check_connection,
        open=True,
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return build_graph(checkpointer)


def after_route_intent(state: AskState) -> str:
    return "generate_sql" if state["intent"] == "sql" else "retrieve"


def after_retrieve(state: AskState) -> str:
    return "generate_sql" if state["intent"] == "hybrid" else "generate"


def after_validate_sql(state: AskState) -> str:
    return _without_sql(state) if state.get("sql_error") else "approve"


def after_approve(state: AskState) -> str:
    return "execute_sql" if state.get("sql_approved") else _without_sql(state)


def after_execute_sql(state: AskState) -> str:
    return _without_sql(state) if state.get("sql_error") else "generate"


def after_generate(state: AskState) -> str:
    if state["intent"] == "sql":
        return "finalize"
    if state["intent"] == "rag" and _settings(state).self_rag:
        return "self_check"
    return "verify"


def after_self_check(state: AskState) -> str:
    check = state.get("answer_check")
    threshold = _settings(state).self_rag_threshold
    if check and should_retry(check["score"], threshold, state.get("retries", 0)):
        return "rewrite"
    return "verify"


def after_rewrite(state: AskState) -> str:
    return "retrieve" if state.get("first_attempt") else "verify"


def _without_sql(state: AskState) -> str:
    """Where to go when SQL failed or was rejected: sql ends, hybrid answers from documents."""
    return "finalize" if state["intent"] == "sql" else "generate"


def _settings(state: AskState) -> CollectionSettings:
    return CollectionSettings.model_validate(state["settings"])
