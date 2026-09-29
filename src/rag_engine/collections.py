"""Collection and document records in Postgres. Qdrant is handled by the callers."""

import logging
import re
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from rag_engine import db
from rag_engine.models import Collection, CollectionSettings, DocumentRecord, EngineError

logger = logging.getLogger(__name__)

COLLECTION_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")

_CREATE_COLLECTIONS = """
CREATE TABLE IF NOT EXISTS collections (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    settings    JSONB NOT NULL,
    version     INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    doc_id        TEXT NOT NULL,
    source_name   TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    chunk_count   INTEGER NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}',
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (collection_id, doc_id)
)
"""

_SELECT_COLLECTIONS = """
SELECT c.*, count(d.id) AS document_count
FROM collections c LEFT JOIN documents d ON d.collection_id = c.id
"""


def ensure_schema() -> None:
    with db.connect() as conn:
        conn.execute(_CREATE_COLLECTIONS)
        conn.execute(_CREATE_DOCUMENTS)


def check_collection_id(collection_id: str) -> None:
    if not COLLECTION_ID_PATTERN.fullmatch(collection_id):
        raise EngineError(
            "collection_id must be 1-64 characters of lowercase letters, digits, '-' or '_', "
            "starting with a letter or digit"
        )


def create(
    collection_id: str, name: str, description: str, settings: CollectionSettings
) -> Collection:
    check_collection_id(collection_id)
    with db.connect(row_factory=dict_row) as conn:
        row = conn.execute(
            "INSERT INTO collections (id, name, description, settings) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO NOTHING RETURNING *, 0 AS document_count",
            (collection_id, name, description, Jsonb(settings.model_dump())),
        ).fetchone()
    if row is None:
        raise EngineError(f"Collection {collection_id!r} already exists")
    return _to_collection(row)


def get(collection_id: str) -> Collection:
    with db.connect(row_factory=dict_row) as conn:
        row = conn.execute(
            _SELECT_COLLECTIONS + "WHERE c.id = %s GROUP BY c.id", (collection_id,)
        ).fetchone()
    if row is None:
        raise EngineError(f"Collection {collection_id!r} does not exist")
    return _to_collection(row)


def list_all() -> list[Collection]:
    with db.connect(row_factory=dict_row) as conn:
        rows = conn.execute(_SELECT_COLLECTIONS + "GROUP BY c.id ORDER BY c.id").fetchall()
    return [_to_collection(row) for row in rows]


def update_settings(collection_id: str, settings: CollectionSettings) -> Collection:
    with db.connect() as conn:
        conn.execute(
            "UPDATE collections SET settings = %s WHERE id = %s",
            (Jsonb(settings.model_dump()), collection_id),
        )
    return get(collection_id)


def delete(collection_id: str) -> None:
    """Deletes the collection row; its document rows go with it (ON DELETE CASCADE)."""
    with db.connect() as conn:
        conn.execute("DELETE FROM collections WHERE id = %s", (collection_id,))


def get_document(collection_id: str, doc_id: str) -> DocumentRecord | None:
    with db.connect(row_factory=dict_row) as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE collection_id = %s AND doc_id = %s",
            (collection_id, doc_id),
        ).fetchone()
    return DocumentRecord(**row) if row else None


def list_documents(collection_id: str) -> list[DocumentRecord]:
    with db.connect(row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE collection_id = %s ORDER BY doc_id", (collection_id,)
        ).fetchall()
    return [DocumentRecord(**row) for row in rows]


def save_document(
    collection_id: str,
    doc_id: str,
    source_name: str,
    content_hash: str,
    chunk_count: int,
    metadata: dict[str, Any],
) -> None:
    """Insert or replace the document row and bump the collection version, in one transaction."""
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (collection_id, doc_id, source_name, content_hash, chunk_count, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (collection_id, doc_id) DO UPDATE SET
                source_name = EXCLUDED.source_name,
                content_hash = EXCLUDED.content_hash,
                chunk_count = EXCLUDED.chunk_count,
                metadata = EXCLUDED.metadata,
                ingested_at = now()
            """,
            (collection_id, doc_id, source_name, content_hash, chunk_count, Jsonb(metadata)),
        )
        _bump_version(conn, collection_id)


def delete_document(collection_id: str, doc_id: str) -> None:
    with db.connect() as conn:
        conn.execute(
            "DELETE FROM documents WHERE collection_id = %s AND doc_id = %s",
            (collection_id, doc_id),
        )
        _bump_version(conn, collection_id)


def _bump_version(conn, collection_id: str) -> None:
    conn.execute("UPDATE collections SET version = version + 1 WHERE id = %s", (collection_id,))


def _to_collection(row: dict[str, Any]) -> Collection:
    return Collection(**{**row, "settings": stored_settings(row["id"], row["settings"])})


def stored_settings(collection_id: str, stored: dict[str, Any]) -> CollectionSettings:
    """Settings saved by an older version may contain keys that no longer exist; skip those
    (with a warning) so the collection still loads. Unknown values of known keys still fail."""
    unknown = sorted(set(stored) - set(CollectionSettings.model_fields))
    if unknown:
        logger.warning("Collection %r: ignoring stored settings that no longer exist: %s", collection_id, unknown)
    return CollectionSettings.model_validate({k: v for k, v in stored.items() if k not in unknown})
