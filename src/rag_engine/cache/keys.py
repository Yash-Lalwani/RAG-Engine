import hashlib
import json
from typing import Any

EMBEDDING_TIER = "embedding"
EMBEDDING_TTL = 7 * 24 * 3600
INTENT_TIER = "intent"
INTENT_TTL = 24 * 3600
SQL_GEN_TIER = "sql_gen"
SQL_GEN_TTL = 24 * 3600
SQL_RESULT_TIER = "sql_result"
SQL_RESULT_TTL = 15 * 60
ANSWER_TIER = "answer"
ANSWER_TTL = 3600


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def embedding_key(model: str, text: str) -> str:
    return f"emb:{_digest(model, text)}"


def sql_generation_key(database: str, allowed_tables: list[str], question: str) -> str:
    return f"sqlgen:{_digest(database, ','.join(sorted(allowed_tables)), question.strip())}"


def sql_result_key(database: str, validated_sql: str) -> str:
    """validated_sql is sqlglot's normalized output, so spacing differences share a key."""
    return f"sqlres:{_digest(database, validated_sql)}"


def intent_key(collection_id: str, allowed_tables: list[str], question: str) -> str:
    return f"intent:{_digest(collection_id, ','.join(sorted(allowed_tables)), question.strip())}"


def answer_key(
    collection_id: str, version: int, question: str, effective_options: dict[str, Any]
) -> str:
    """The collection version is part of the key, so any ingest or delete invalidates answers."""
    options = json.dumps(effective_options, sort_keys=True, default=str)
    return f"answer:{_digest(collection_id, str(version), question.strip(), options)}"
