import json

from langgraph.types import interrupt

from rag_engine.generation.generator import _generate
from rag_engine.graph.state import GraphState
from rag_engine.guardrails.spotlight import build_spotlighted_context
from rag_engine.llm import generate
from rag_engine.retrieval.search import _retrieve
from rag_engine.routing.intent_router import classify_intent
from rag_engine.sql import executor
from rag_engine.sql.text_to_sql import generate_sql


def route_intent(state: GraphState) -> dict:
    """LLM-based intent router for sql/rag/hybrid."""
    intent = classify_intent(state["question"])
    return {"intent": intent}


def retrieve_rag(state: GraphState) -> dict:
    chunks = _retrieve(state["question"], flags=state.get("flags", {}))
    return {
        "retrieved_chunks": chunks,
        "spotlighted_context": build_spotlighted_context(chunks),
    }


def generate_sql_node(state: GraphState) -> dict:
    result = generate_sql(state["question"])
    return {
        "generated_sql": result["sql"],
        "sql_explanation": result["explanation"],
    }


def request_sql_approval(state: GraphState) -> dict:
    approval = interrupt({
        "type": "sql_approval_required",
        "sql": state["generated_sql"],
        "explanation": state["sql_explanation"],
    })
    return {"sql_approved": approval.get("approved", False)}


def execute_sql(state: GraphState) -> dict:
    """Execute approved SQL and store results."""
    if not state.get("sql_approved"):
        return {"sql_rows": [], "final_answer": "SQL query was not approved."}

    sql = state.get("generated_sql", "")
    try:
        rows = executor.execute_sql(sql)
        return {"sql_rows": rows}
    except Exception as exc:
        return {"sql_rows": [], "final_answer": f"SQL execution failed: {exc}"}


def generate_answer(state: GraphState) -> dict:
    intent = state.get("intent", "rag")

    if intent == "sql":
        rows = state.get("sql_rows", [])
        if not rows:
            return {
                "final_answer": state.get("final_answer", "No results."),
                "sources": ["database query"],
                "confidence": 0.9,
            }
        answer = f"Query results:\n```\n{json.dumps(rows, indent=2, default=str)}\n```"
        return {
            "final_answer": answer,
            "sources": ["database query"],
            "confidence": 0.9,
        }

    if intent == "hybrid":
        return _generate_hybrid_answer(state)

    flags = state.get("flags", {})
    chunks = _retrieve(state["question"], flags=flags)
    response = _generate(state["question"], chunks, flags=flags)
    chunk_previews = [chunk.model_dump() for chunk in response.metadata.retrieved_chunks]

    return {
        "final_answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "cache_hit": response.cache_hit,
        "chunk_previews": chunk_previews,
        "metadata": response.metadata.model_dump(),
        "reflection_iterations": response.metadata.reflection_iterations,
        "refined_question": response.metadata.refined_question,
    }


def _generate_hybrid_answer(state: GraphState) -> dict:
    rows = state.get("sql_rows", [])
    rag_context = state.get("spotlighted_context", "")

    sql_section = ""
    if rows:
        sql_section = (
            f"=== Database Query Results ===\n```\n{json.dumps(rows, indent=2, default=str)}\n```\n"
        )

    rag_section = f"=== Retrieved Documents ===\n{rag_context}\n" if rag_context else ""

    system = (
        "You are an AI assistant. Synthesize database query results and "
        "retrieved documents into a single coherent answer. Cite sources using "
        "[database query] for SQL results and [source_name] for documents."
    )
    user_msg = f"{sql_section}{rag_section}\n\nQuestion: {state['question']}"

    result = generate(system, user_msg)
    return {
        "final_answer": result["text"],
        "sources": ["database query"] + [c.source for c in state.get("retrieved_chunks", [])],
        "confidence": 0.85,
    }


def finalize(state: GraphState) -> dict:
    return {}
