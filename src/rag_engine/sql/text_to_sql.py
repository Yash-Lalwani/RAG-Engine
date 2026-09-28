"""generate_sql(): question -> one SELECT over the allowlisted tables. Executes nothing."""

from rag_engine import llm
from rag_engine.cache.keys import SQL_GEN_TIER, SQL_GEN_TTL, sql_generation_key
from rag_engine.cache.store import cache
from rag_engine.config import settings
from rag_engine.generation import prompts
from rag_engine.models import EngineError, SqlDraft
from rag_engine.sql.safety import validate_sql
from rag_engine.sql.schema import describe_schema


def generate_sql(
    question: str, database: str, allowed_tables: list[str], domain_description: str = ""
) -> SqlDraft:
    if not allowed_tables:
        raise EngineError("No SQL tables are allowed for this collection")
    key = sql_generation_key(database, allowed_tables, question)
    cached = cache.get(SQL_GEN_TIER, key)
    if cached is not None:
        return SqlDraft.model_validate_json(cached)

    schema = describe_schema(database, tuple(sorted(allowed_tables)))
    raw = llm.generate_structured(
        prompts.sql_system(domain_description),
        f"Schema:\n{schema}\n\nQuestion: {question}",
        SqlDraft,
        model=settings.llm_model_strong,
    )
    draft = SqlDraft(sql=raw.sql.strip().rstrip(";").strip(), explanation=raw.explanation.strip())

    try:
        validate_sql(draft.sql, allowed_tables)
    except EngineError:
        return draft  # returned for the validation step to report; never cached
    cache.set(SQL_GEN_TIER, key, draft.model_dump_json(), SQL_GEN_TTL)
    return draft
