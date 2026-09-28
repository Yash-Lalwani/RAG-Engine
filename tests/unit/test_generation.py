from rag_engine.generation.generator import (
    NO_CONTEXT_TEXT,
    SQL_SOURCE_ID,
    citation_passages,
    generate_answer,
    render_answer,
)
from rag_engine.models import CitedAnswer, Statement


def test_labels_map_back_to_real_ids_and_unknown_labels_are_dropped(fake_structured, make_chunk):
    chunks = [make_chunk("id-a", "A."), make_chunk("id-b", "B.")]
    calls = fake_structured(
        CitedAnswer(
            statements=[
                Statement(text=" First. ", chunk_ids=["c2", "c9", "c2"]),
                Statement(text="Second.", chunk_ids=["c1"]),
                Statement(text="  ", chunk_ids=[]),
            ],
            insufficient_context=False,
        )
    )
    answer = generate_answer("q", chunks, domain_description="Kubernetes docs")
    assert [(s.text, s.chunk_ids) for s in answer.statements] == [
        ("First.", ["id-b"]), ("Second.", ["id-a"])
    ]
    assert '<document id="c1" source="doc.md">' in calls[0]["user"]
    assert "Kubernetes docs" in calls[0]["system"]


def test_sql_rows_are_citable(fake_structured, make_chunk):
    calls = fake_structured(
        CitedAnswer(statements=[Statement(text="3 clusters.", chunk_ids=["sql"])], insufficient_context=False)
    )
    answer = generate_answer("q", [], sql_rows=[{"count": 3}])
    assert answer.statements[0].chunk_ids == [SQL_SOURCE_ID]
    assert '<sql_results id="sql">' in calls[0]["user"]
    assert citation_passages([], [{"count": 3}])[0].id == SQL_SOURCE_ID


def test_no_context_skips_the_llm(fake_structured):
    calls = fake_structured(RuntimeError("must not be called"))
    answer = generate_answer("q", [])
    assert calls == [] and answer.insufficient_context
    assert answer.statements[0].text == NO_CONTEXT_TEXT


def test_render_numbers_each_cited_chunk_once(make_chunk):
    chunks = [make_chunk("id-a", source="a.md"), make_chunk("id-b", source="b.md")]
    statements = [
        Statement(text="Intro.", chunk_ids=[]),
        Statement(text="One.", chunk_ids=["id-b"]),
        Statement(text="Two.", chunk_ids=["id-a", "id-b"]),
    ]
    text, sources = render_answer(statements, chunks)
    assert text == "Intro. One. [1] Two. [2][1]"
    assert [(s.number, s.source) for s in sources] == [(1, "b.md"), (2, "a.md")]
