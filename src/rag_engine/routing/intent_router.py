"""Intent router: should a question be answered from documents (rag), SQL (sql) or both (hybrid)?"""

from typing import Literal

from pydantic import BaseModel

from rag_engine import llm
from rag_engine.cache.keys import INTENT_TIER, INTENT_TTL, intent_key
from rag_engine.cache.store import cache
from rag_engine.config import settings as config
from rag_engine.generation import prompts
from rag_engine.models import CollectionSettings

Intent = Literal["rag", "sql", "hybrid"]


class _Route(BaseModel):
    intent: Intent


def classify_intent(question: str, collection_id: str, settings: CollectionSettings) -> Intent:
    """Collections without a SQL database always get "rag", without an LLM call."""
    if settings.sql_database is None or not settings.sql_allowed_tables:
        return "rag"
    key = intent_key(collection_id, settings.sql_allowed_tables, question)
    cached = cache.get(INTENT_TIER, key)
    if cached in ("rag", "sql", "hybrid"):
        return cached

    route = llm.generate_structured(
        prompts.router_system(settings.domain_description, settings.sql_allowed_tables),
        question,
        _Route,
        model=config.llm_model_small,
    )
    cache.set(INTENT_TIER, key, route.intent, INTENT_TTL)
    return route.intent
