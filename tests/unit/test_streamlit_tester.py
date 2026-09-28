"""Smoke test for the temporary Streamlit tester (deleted together with dev/streamlit_tester.py)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag_engine import engine
from rag_engine.models import (
    AskResult,
    Collection,
    CollectionSettings,
    DeleteResult,
    SearchChunk,
    SearchInfo,
    SearchResult,
    Source,
)

APP = Path(__file__).resolve().parents[2] / "dev" / "streamlit_tester.py"


class FakeEngine:
    def __init__(self):
        self.collections: dict[str, Collection] = {}

    def setup(self):
        pass

    def list_collections(self):
        return list(self.collections.values())

    def create_collection(self, collection_id, name, description="", settings=None):
        collection = Collection(id=collection_id, name=name, description=description,
                                settings=CollectionSettings.from_dict(settings), created_at=datetime.now(UTC))
        self.collections[collection_id] = collection
        return collection

    def update_collection(self, collection_id, settings):
        old = self.collections[collection_id]
        self.collections[collection_id] = old.model_copy(update={"settings": old.settings.merged(settings)})
        return self.collections[collection_id]

    def delete_collection(self, collection_id):
        del self.collections[collection_id]
        return DeleteResult(collection_id=collection_id)

    def list_documents(self, collection_id):
        return []

    def search(self, collection_id, query, filters=None, options=None):
        chunk = SearchChunk(id="c1", doc_id="d", source="pods.md", chunk_index=0,
                            text="Pods run containers.", fused_score=0.03, rerank_score=5.0, grade="relevant")
        return SearchResult(collection_id=collection_id, query=query, chunks=[chunk],
                            info=SearchInfo(mode=options["search_mode"], crag_action="correct"))

    def ask(self, collection_id, question, options=None, *, caller):
        return AskResult(status="pending_sql", query_id="q1", intent="sql", sql="SELECT 1",
                         sql_explanation="Returns one.", message="Review the SQL.")

    def approve_sql(self, query_id, approve, *, caller):
        if caller != "dev":
            return AskResult(status="error", query_id=query_id, message="This query belongs to another caller")
        return AskResult(status="completed", query_id=query_id, intent="sql", answer="There is 1 row. [1]",
                         sources=[Source(number=1, chunk_id="sql_results", source="SQL query results")],
                         rows_preview=[{"n": 1}])


@pytest.fixture
def app(monkeypatch):
    fake = FakeEngine()
    for name in ["setup", "list_collections", "create_collection", "update_collection",
                 "delete_collection", "list_documents", "search", "ask", "approve_sql"]:
        monkeypatch.setattr(engine, name, getattr(fake, name))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    return at, fake


def click(at: AppTest, label: str) -> AppTest:
    next(b for b in at.button if b.label == label).click()
    return at.run()


def test_every_page_loads(app):
    at, _ = app
    for page in ["Collections", "Ingest", "Search", "Ask"]:
        at.radio(key="page").set_value(page).run()
        assert not at.exception, page


def test_create_edit_and_delete_a_collection(app):
    at, fake = app
    at.text_input(key="new_id").set_value("demo")
    at.text_area(key="new_settings").set_value('{"top_k": 3}')
    click(at, "Create")
    assert not at.exception and fake.collections["demo"].settings.top_k == 3
    assert at.selectbox(key="collection").value == "demo"

    at.text_area(key="settings_demo").set_value('{"top_k": 4}')
    click(at, "Save settings")
    assert fake.collections["demo"].settings.top_k == 4

    at.checkbox(key="confirm_delete").check().run()
    click(at, "Delete collection")
    assert fake.collections == {} and not at.exception


def test_search_and_ask_with_sql_approval(app):
    at, fake = app
    fake.create_collection("demo", "Demo")
    at.run()

    at.radio(key="page").set_value("Search").run()
    at.text_input(key="search_query").set_value("what is a pod").run()
    click(at, "Search")
    assert not at.exception and at.dataframe[0].value["source"][0] == "pods.md"

    at.radio(key="page").set_value("Ask").run()
    at.text_area(key="ask_question").set_value("How many rows?").run()
    click(at, "Ask")
    assert at.code[0].value == "SELECT 1"

    at.text_input(key="caller").set_value("astra").run()
    click(at, "Approve SQL")
    assert "another caller" in at.error[0].value and at.code[0].value == "SELECT 1"

    at.text_input(key="caller").set_value("dev").run()
    click(at, "Approve SQL")
    assert not at.exception and "There is 1 row" in at.markdown[0].value
