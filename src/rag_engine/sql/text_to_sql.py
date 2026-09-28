import json

from rag_engine.cache.store import query_cache
from rag_engine.config import settings
from rag_engine.llm import generate
from rag_engine.sql.schema import describe_schema


def generate_sql(question: str) -> dict:
    cached = query_cache.get_sql_generation(question)
    if cached is not None:
        return {"sql": cached, "explanation": "Loaded from SQL generation cache."}

    schema = describe_schema(settings.database_url)
    system = (
        "You are a SQL expert. Given a database schema and a question, "
        "generate a valid PostgreSQL SELECT query. Return JSON with keys: sql, explanation."
    )
    user = f"{schema}\n\nQuestion: {question}\n\nReturn only the JSON."
    result = generate(system, user, model=settings.llm_model_strong, temperature=0.0)
    text = result["text"].strip()

    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    data = json.loads(text)
    payload = {
        "sql": data.get("sql", ""),
        "explanation": data.get("explanation", ""),
    }

    query_cache.set_sql_generation(question, payload["sql"])
    return payload
