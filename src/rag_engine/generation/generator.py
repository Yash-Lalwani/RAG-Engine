"""generate_answer(): a structured, cited answer from spotlighted chunks (and SQL rows)."""

from rag_engine import llm
from rag_engine.config import settings
from rag_engine.generation import prompts
from rag_engine.guardrails.spotlight import spotlight_documents, spotlight_rows
from rag_engine.models import CitedAnswer, Passage, SearchChunk, Source, SqlResult, Statement

SQL_SOURCE_ID = "sql_results"
LLM_ROWS = 50  # rows shown to the LLM; the total count is always included
NO_CONTEXT_TEXT = "The available documents do not contain enough information to answer this question."


def generate_answer(
    question: str,
    chunks: list[SearchChunk],
    sql_result: SqlResult | None = None,
    domain_description: str = "",
) -> CitedAnswer:
    if not chunks and sql_result is None:
        return CitedAnswer(
            statements=[Statement(text=NO_CONTEXT_TEXT, chunk_ids=[])], insufficient_context=True
        )

    labels = {f"c{n}": chunk.id for n, chunk in enumerate(chunks, start=1)}
    context = spotlight_documents([(f"c{n}", c.source, c.text) for n, c in enumerate(chunks, start=1)])
    if sql_result is not None:
        labels["sql"] = SQL_SOURCE_ID
        context += "\n\n" + _sql_block(sql_result)

    raw = llm.generate_structured(
        prompts.answer_system(domain_description),
        f"{context}\n\nQuestion: {question}",
        CitedAnswer,
        model=settings.llm_model_strong,
    )
    statements = [
        Statement(text=s.text.strip(), chunk_ids=_known_ids(s.chunk_ids, labels))
        for s in raw.statements
        if s.text.strip()
    ]
    return CitedAnswer(statements=statements, insufficient_context=raw.insufficient_context)


def render_answer(
    statements: list[Statement], chunks: list[SearchChunk]
) -> tuple[str, list[Source]]:
    """Answer text with [n] markers, plus the numbered source list (one number per cited chunk)."""
    numbers: dict[str, int] = {}
    parts = []
    for statement in statements:
        for chunk_id in statement.chunk_ids:
            numbers.setdefault(chunk_id, len(numbers) + 1)
        markers = "".join(f"[{numbers[chunk_id]}]" for chunk_id in statement.chunk_ids)
        parts.append(f"{statement.text} {markers}".rstrip())

    by_id = {chunk.id: chunk for chunk in chunks}
    sources = [_source(number, chunk_id, by_id.get(chunk_id)) for chunk_id, number in numbers.items()]
    return " ".join(parts), sources


def citation_passages(
    chunks: list[SearchChunk], sql_result: SqlResult | None = None
) -> list[Passage]:
    """The passages a generated answer may cite, in the form verify_citations expects."""
    passages = [Passage(id=chunk.id, text=chunk.text) for chunk in chunks]
    if sql_result is not None:
        passages.append(Passage(id=SQL_SOURCE_ID, text=_sql_block(sql_result)))
    return passages


def _sql_block(result: SqlResult) -> str:
    """The query, up to LLM_ROWS rows, and the total count when more rows exist."""
    total = None
    if result.row_count > LLM_ROWS or result.truncated:
        total = f"{result.row_count}{'+' if result.truncated else ''}"
    return spotlight_rows("sql", result.sql, result.rows[:LLM_ROWS], total)


def _known_ids(chunk_labels: list[str], labels: dict[str, str]) -> list[str]:
    """Map prompt labels (c1, sql) to real ids, dropping labels the model made up."""
    ids = []
    for label in chunk_labels:
        real_id = labels.get(label.strip())
        if real_id and real_id not in ids:
            ids.append(real_id)
    return ids


def _source(number: int, chunk_id: str, chunk: SearchChunk | None) -> Source:
    if chunk_id == SQL_SOURCE_ID:
        return Source(number=number, chunk_id=chunk_id, source="SQL query results")
    if chunk is None:
        return Source(number=number, chunk_id=chunk_id, source="unknown")
    return Source(
        number=number,
        chunk_id=chunk_id,
        doc_id=chunk.doc_id,
        source=chunk.source,
        url=chunk.url,
        chunk_index=chunk.chunk_index,
    )
