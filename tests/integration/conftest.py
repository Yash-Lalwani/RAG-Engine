import base64
import re
import uuid
import zlib

import pytest

from rag_engine import engine
from rag_engine.retrieval import embeddings, vector_store
from rag_engine.retrieval.vector_store import dense_vector_size


def fake_dense(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words vectors, so tests need no OpenAI key."""
    size = dense_vector_size()
    vectors = []
    for text in texts:
        vector = [0.0] * size
        vector[0] = 0.001
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(word.encode()) % size] += 1.0
        vectors.append(vector)
    return vectors


@pytest.fixture(scope="session", autouse=True)
def engine_ready():
    engine.setup()


@pytest.fixture(autouse=True)
def no_openai(monkeypatch):
    monkeypatch.setattr(embeddings, "embed_dense", fake_dense)


@pytest.fixture
def make_collection():
    created = []

    def _make(settings: dict | None = None) -> str:
        collection_id = f"test-{uuid.uuid4().hex[:8]}"
        engine.create_collection(collection_id, "Test", settings={"rerank": False, **(settings or {})})
        created.append(collection_id)
        return collection_id

    yield _make
    for collection_id in created:
        if collection_id in {c.id for c in engine.list_collections()}:
            engine.delete_collection(collection_id)


@pytest.fixture
def ingest_text():
    def _ingest(collection_id: str, doc_id: str, text: str, metadata: dict | None = None):
        return engine.ingest_document(
            collection_id,
            content_base64=base64.b64encode(text.encode()).decode(),
            filename=f"{doc_id}.md",
            doc_id=doc_id,
            metadata=metadata,
        )

    return _ingest


@pytest.fixture
def point_count():
    return vector_store.count_points
