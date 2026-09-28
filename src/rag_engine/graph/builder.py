import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from rag_engine.config import settings
from rag_engine.graph.nodes import (
    execute_sql,
    finalize,
    generate_answer,
    generate_sql_node,
    request_sql_approval,
    retrieve_rag,
    route_intent,
)
from rag_engine.graph.state import GraphState


def _get_checkpointer() -> PostgresSaver:
    conn = psycopg.connect(settings.database_url, autocommit=True)
    saver = PostgresSaver(conn=conn)
    saver.setup()
    return saver


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("route_intent", route_intent)
    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_node("generate_sql_node", generate_sql_node)
    builder.add_node("request_sql_approval", request_sql_approval)
    builder.add_node("execute_sql", execute_sql)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "route_intent")
    builder.add_conditional_edges(
        "route_intent",
        lambda s: s.get("intent", "rag"),
        {"sql": "generate_sql_node", "rag": "generate_answer", "hybrid": "retrieve_rag"},
    )
    builder.add_edge("retrieve_rag", "generate_sql_node")
    builder.add_edge("generate_sql_node", "request_sql_approval")
    builder.add_edge("request_sql_approval", "execute_sql")
    builder.add_edge("execute_sql", "generate_answer")
    builder.add_edge("generate_answer", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=_get_checkpointer())
