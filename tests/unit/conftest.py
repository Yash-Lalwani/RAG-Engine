import pytest

from rag_engine.grading.citations import verify_citations as real_verify
from rag_engine.graph import nodes
from rag_engine.models import (
    CitedAnswer,
    EngineError,
    SearchChunk,
    SearchInfo,
    SearchResult,
    SelfCheck,
    SqlDraft,
    SqlResult,
    Statement,
)


@pytest.fixture(autouse=True)
def no_guardrail_models(monkeypatch):
    """Unit tests never load the G2/G6 models; tests that need scores override these."""
    from rag_engine.guardrails import input_checks, output_checks

    monkeypatch.setattr(input_checks, "injection_score", lambda text: 0.0)
    monkeypatch.setattr(input_checks, "toxicity_score", lambda text: 0.0)
    monkeypatch.setattr(output_checks, "toxicity_score", lambda text: 0.0)


@pytest.fixture
def make_chunk():
    def _make(chunk_id: str, text: str = "some text", source: str = "doc.md") -> SearchChunk:
        return SearchChunk(
            id=chunk_id, doc_id=source, source=source, chunk_index=0, text=text, fused_score=0.1
        )

    return _make


@pytest.fixture
def fake_structured(monkeypatch):
    """Replace llm.generate_structured with a function returning the given object (or raising)."""
    from rag_engine import llm

    calls = []

    def _install(result):
        def fake(system, user, schema, model=None, temperature=0.0):
            calls.append({"system": system, "user": user, "schema": schema, "model": model})
            if isinstance(result, Exception):
                raise result
            return result(schema) if callable(result) else result

        monkeypatch.setattr(llm, "generate_structured", fake)
        return calls

    return _install


class FakeCore:
    """Records every call to the core functions the nodes use."""

    def __init__(self, intent="rag", sql="SELECT count(*) FROM pods", scores=(0.9,), sql_fails=False):
        self.intent, self.sql, self.scores, self.sql_fails = intent, sql, list(scores), sql_fails
        self.calls: dict[str, list] = {}

    def record(self, name, **kwargs):
        self.calls.setdefault(name, []).append(kwargs)

    def classify_intent(self, question, collection_id, settings):
        self.record("classify_intent", question=question)
        return self.intent

    def search(self, collection_id, query, filters=None, options=None):
        self.record("search", query=query, filters=filters, options=options)
        chunk = SearchChunk(id=f"chunk-{query}", doc_id="doc", source="doc.md", chunk_index=0,
                            text=f"text for {query}", fused_score=0.1)
        return SearchResult(collection_id=collection_id, query=query, chunks=[chunk],
                            info=SearchInfo(mode="hybrid", crag_action="correct"))

    def generate_answer(self, question, chunks, sql_result=None, domain_description=""):
        self.record("generate_answer", question=question, chunks=chunks, sql_result=sql_result)
        ids = [c.id for c in chunks] + (["sql_results"] if sql_result else [])
        return CitedAnswer(statements=[Statement(text=f"answer from {len(chunks)} chunks", chunk_ids=ids)],
                           insufficient_context=False)

    def self_check(self, question, answer, chunks):
        self.record("self_check", question=question)
        return SelfCheck(groundedness=1.0, completeness=1.0, score=self.scores.pop(0), reason="missing detail")

    def rewrite_query(self, question, reason, domain_description=""):
        self.record("rewrite_query", question=question)
        return "better query"

    def generate_sql(self, question, database, allowed_tables, domain_description=""):
        self.record("generate_sql", question=question)
        return SqlDraft(sql=self.sql, explanation="Counts pods.")

    def run_sql(self, sql, database, allowed_tables):
        self.record("run_sql", sql=sql)
        if self.sql_fails:
            raise EngineError("The SQL query timed out after 5 s")
        rows = [{"n": i} for i in range(60)]
        return SqlResult(sql=sql, columns=["n"], rows=rows, row_count=60, truncated=False)

    def verify_citations(self, statements, passages, strict=False):
        self.record("verify_citations", strict=strict, passage_ids=[p.id for p in passages])
        return real_verify([s.model_copy(update={"chunk_ids": []}) for s in statements], passages, strict)


@pytest.fixture
def core(monkeypatch):
    fake = FakeCore()
    for name in ["classify_intent", "search", "generate_answer", "self_check", "rewrite_query",
                 "generate_sql", "run_sql", "verify_citations"]:
        monkeypatch.setattr(nodes, name, getattr(fake, name))
    return fake
