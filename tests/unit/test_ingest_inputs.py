import base64

import pytest

from rag_engine.config import settings
from rag_engine.ingestion import pipeline
from rag_engine.ingestion.pipeline import check_metadata, document_hash, read_input
from rag_engine.models import EngineError


@pytest.fixture
def ingest_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ingest_dir", str(tmp_path))
    return tmp_path


def test_reads_file_inside_ingest_dir(ingest_dir):
    (ingest_dir / "doc.md").write_text("hello")
    assert read_input(str(ingest_dir / "doc.md"), None, None) == (b"hello", "doc.md")


def test_rejects_file_outside_ingest_dir(ingest_dir, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "secret.txt"
    outside.write_text("secret")
    with pytest.raises(EngineError, match="inside the ingest directory"):
        read_input(str(outside), None, None)


def test_rejects_path_traversal(ingest_dir):
    with pytest.raises(EngineError, match="inside the ingest directory"):
        read_input(str(ingest_dir / ".." / "x.txt"), None, None)


def test_base64_input_uses_only_the_file_name():
    encoded = base64.b64encode(b"hello").decode()
    assert read_input(None, encoded, "../../notes.md") == (b"hello", "notes.md")


@pytest.mark.parametrize(
    "args",
    [(None, None, None), ("a.md", "aGk=", "a.md"), (None, "aGk=", None), (None, "not base64!", "a.md")],
)
def test_bad_input_combinations(args):
    with pytest.raises(EngineError):
        read_input(*args)


def test_size_limit(monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_DOCUMENT_BYTES", 3)
    with pytest.raises(EngineError, match="larger than"):
        read_input(None, base64.b64encode(b"hello").decode(), "a.md")


def test_filterable_metadata_must_be_simple():
    check_metadata({"team": "alpha", "tags": ["x", "y"]}, ["team"])
    with pytest.raises(EngineError):
        check_metadata({"team": ["alpha"]}, ["team"])
    with pytest.raises(EngineError):
        check_metadata({"url": 5}, [])


def test_document_hash_covers_content_and_metadata():
    base = document_hash(b"hello", {"team": "alpha", "year": 2024})
    assert base == document_hash(b"hello", {"year": 2024, "team": "alpha"})
    assert base != document_hash(b"hello", {"team": "beta", "year": 2024})
    assert base != document_hash(b"hello", {})
    assert base != document_hash(b"hello!", {"team": "alpha", "year": 2024})


def test_document_hash_rejects_non_json_metadata():
    with pytest.raises(EngineError, match="JSON"):
        document_hash(b"hello", {"when": object()})


def test_comma_heavy_text_files_are_not_mistaken_for_csv():
    from rag_engine.ingestion.chunker import chunk_document
    from rag_engine.ingestion.parser import parse_document

    text = "Cast\n\nAlice, Bob, Carol, Dan\nEve, Frank, Grace, Heidi\n\nA film about a pilot, a poet, and a plan.\n"
    chunks = chunk_document(parse_document(text.encode(), "cast.txt"))
    assert chunks and "pilot" in " ".join(c["text"] for c in chunks)
