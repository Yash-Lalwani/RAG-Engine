import pytest

from rag_engine import engine
from rag_engine.models import EngineError
from rag_engine.retrieval import vector_store

pytestmark = pytest.mark.integration

MODES = ["dense", "sparse", "hybrid"]
POD_TEXT = "# Pods\n\nA pod is the smallest deployable unit. Pods run containers on nodes."


@pytest.mark.parametrize("mode", MODES)
def test_search_never_returns_another_collections_chunks(make_collection, mode, ingest_text):
    a, b = make_collection(), make_collection()
    ingest_text(a, "a-doc", POD_TEXT)
    ingest_text(b, "b-doc", POD_TEXT)

    for collection_id, own_doc in [(a, "a-doc"), (b, "b-doc")]:
        result = engine.search(collection_id, "what is a pod", options={"search_mode": mode})
        assert result.chunks, mode
        assert {chunk.doc_id for chunk in result.chunks} == {own_doc}
        assert result.info.mode == mode


@pytest.mark.parametrize("mode", MODES)
def test_metadata_filters_only_return_matching_chunks(make_collection, mode, ingest_text):
    cid = make_collection({"filterable_fields": ["team"]})
    for team in ["alpha", "beta", "gamma"]:
        ingest_text(cid, f"{team}-doc", POD_TEXT + f" Owned by team {team}.", {"team": team})

    options = {"search_mode": mode}
    one = engine.search(cid, "pod", filters={"team": "beta"}, options=options)
    assert one.chunks and {c.metadata["team"] for c in one.chunks} == {"beta"}

    many = engine.search(cid, "pod", filters={"team": ["alpha", "gamma"]}, options=options)
    assert many.chunks and {c.metadata["team"] for c in many.chunks} <= {"alpha", "gamma"}


def test_filter_on_field_not_in_filterable_fields_is_rejected(make_collection, ingest_text):
    cid = make_collection({"filterable_fields": ["team"]})
    ingest_text(cid, "doc", POD_TEXT, {"team": "alpha", "owner": "sam"})
    with pytest.raises(EngineError, match="filterable_fields"):
        engine.search(cid, "pod", filters={"owner": "sam"})


def test_adding_a_filterable_field_creates_its_index(make_collection):
    cid = make_collection()
    engine.update_collection(cid, {"filterable_fields": ["region"]})
    schema = vector_store.get_client().get_collection(vector_store.QDRANT_COLLECTION).payload_schema
    assert "metadata.region" in schema
    assert engine.update_collection(cid, {"top_k": 3}).settings.filterable_fields == ["region"]


def test_reingest_unchanged_then_changed_content(make_collection, ingest_text, point_count):
    cid = make_collection()
    long_text = "\n\n".join(f"## Section {i}\n\n" + "Pods and nodes. " * 60 for i in range(6))

    first = ingest_text(cid, "doc", long_text)
    version = engine.list_collections()
    version = next(c.version for c in version if c.id == cid)
    assert first.status == "ingested" and first.chunk_count > 1
    assert point_count(cid, "doc") == first.chunk_count

    again = ingest_text(cid, "doc", long_text)
    assert again.status == "unchanged" and again.chunk_count == first.chunk_count
    assert next(c.version for c in engine.list_collections() if c.id == cid) == version

    changed = ingest_text(cid, "doc", "# Services\n\nA service exposes pods on the network.")
    assert changed.status == "replaced" and changed.chunk_count == 1
    assert point_count(cid, "doc") == 1
    assert next(c.version for c in engine.list_collections() if c.id == cid) == version + 1
    texts = [c.text for c in engine.search(cid, "pods nodes section", top_k=5).chunks]
    assert texts and all("Section" not in text for text in texts)


def test_metadata_only_change_is_reingested(make_collection, ingest_text):
    cid = make_collection({"filterable_fields": ["team"]})
    assert ingest_text(cid, "doc", POD_TEXT, {"team": "alpha"}).status == "ingested"
    assert ingest_text(cid, "doc", POD_TEXT, {"team": "alpha"}).status == "unchanged"
    version = next(c.version for c in engine.list_collections() if c.id == cid)

    assert ingest_text(cid, "doc", POD_TEXT, {"team": "beta"}).status == "replaced"
    assert next(c.version for c in engine.list_collections() if c.id == cid) == version + 1
    assert engine.search(cid, "pod", filters={"team": "beta"}).chunks
    assert not engine.search(cid, "pod", filters={"team": "alpha"}).chunks
    assert engine.list_documents(cid)[0].metadata == {"team": "beta"}


def test_delete_document_and_collection(make_collection, ingest_text, point_count):
    cid = make_collection()
    ingest_text(cid, "keep", POD_TEXT)
    ingest_text(cid, "drop", POD_TEXT)

    engine.delete_document(cid, "drop")
    assert point_count(cid, "drop") == 0 and point_count(cid, "keep") > 0
    assert [d.doc_id for d in engine.list_documents(cid)] == ["keep"]
    with pytest.raises(EngineError, match="does not exist"):
        engine.delete_document(cid, "drop")

    engine.delete_collection(cid)
    assert point_count(cid) == 0
    with pytest.raises(EngineError, match="does not exist"):
        engine.list_documents(cid)


def test_ingest_from_file_path_inside_ingest_dir(make_collection, tmp_path, monkeypatch):
    from rag_engine.config import settings

    monkeypatch.setattr(settings, "ingest_dir", str(tmp_path))
    (tmp_path / "pods.html").write_text("<html><body><h1>Pods</h1><p>Pods run containers.</p></body></html>")
    cid = make_collection()
    result = engine.ingest_document(cid, file_path=str(tmp_path / "pods.html"))
    assert result.status == "ingested" and result.doc_id == "pods.html"


def test_search_with_rerank_scores_every_returned_chunk(make_collection, ingest_text):
    cid = make_collection({"rerank": True, "top_k": 2})
    ingest_text(cid, "pods", POD_TEXT)
    ingest_text(cid, "ingress", "# Ingress\n\nIngress routes external HTTP traffic to services.")
    result = engine.search(cid, "what is the smallest deployable unit")
    assert result.info.reranked and result.info.warnings == []
    assert result.chunks[0].doc_id == "pods"
    assert all(c.rerank_score is not None for c in result.chunks)


def test_standalone_rerank_puts_the_relevant_passage_first():
    passages = [
        {"id": "1", "text": "Ingress exposes HTTP routes from outside the cluster."},
        {"id": "2", "text": "A Pod is the smallest deployable unit of computing in Kubernetes."},
        {"id": "3", "text": "The weather today is sunny."},
    ]
    ranked = engine.rerank("what is a pod", passages, top_k=2)
    assert len(ranked) == 2
    assert ranked[0].id == "2"
    assert ranked[0].score > ranked[1].score


def test_unknown_collection_is_a_clear_error():
    with pytest.raises(EngineError, match="does not exist"):
        engine.search("no-such-collection", "pod")
