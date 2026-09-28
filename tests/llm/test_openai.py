"""Real OpenAI calls (a fraction of a cent per run). The chain test also needs Docker."""

import base64
import uuid

import pytest

from rag_engine import engine, llm
from rag_engine.generation.generator import citation_passages, generate_answer, render_answer
from rag_engine.grading.citations import verify_citations
from rag_engine.grading.crag import grade_chunks
from rag_engine.models import CitedAnswer, Passage, SearchChunk, Statement

pytestmark = pytest.mark.llm


def chunk(chunk_id: str, text: str) -> SearchChunk:
    return SearchChunk(id=chunk_id, doc_id=chunk_id, source=f"{chunk_id}.md", chunk_index=0, text=text, fused_score=0.0)


def test_structured_output_and_usage_tracking():
    with llm.track_usage() as usage:
        answer = llm.generate_structured(
            "Answer with one statement.", "Say hello.", CitedAnswer, model="gpt-4o-mini"
        )
    assert isinstance(answer, CitedAnswer) and answer.statements
    assert usage.calls == 1 and usage.total_tokens > 0


def test_crag_tells_relevant_from_irrelevant():
    graded = grade_chunks(
        "What is a Kubernetes Pod?",
        [chunk("pods", "A Pod is the smallest deployable unit of computing in Kubernetes."),
         chunk("bread", "Mix flour, water and yeast, then bake the dough for 40 minutes.")],
    )
    assert graded[0].grade == "relevant" and graded[1].grade == "irrelevant"


def test_verification_removes_an_unsupported_statement():
    passages = [Passage(id="p", text="A Pod is the smallest deployable unit in Kubernetes.")]
    statements = [
        Statement(text="A Pod is the smallest deployable unit in Kubernetes.", chunk_ids=["p"]),
        Statement(text="Pods were invented in 1995 by NASA.", chunk_ids=["p"]),
    ]
    result = verify_citations(statements, passages)
    assert [s.text for s in result.statements] == [statements[0].text]
    assert result.removed_count == 1


def test_search_generate_verify_chain():
    cid = f"test-{uuid.uuid4().hex[:8]}"
    engine.setup()
    engine.create_collection(cid, "LLM test", settings={"rerank": False})
    try:
        text = ("# Rolling back\n\nTo roll back a Deployment to the previous revision, run "
                "kubectl rollout undo deployment/<name>. Use --to-revision to pick a specific revision.")
        engine.ingest_document(
            cid, content_base64=base64.b64encode(text.encode()).decode(), filename="rollback.md"
        )
        found = engine.search(cid, "How do I roll back a deployment?")
        assert found.info.crag_action == "correct"

        answer = generate_answer("How do I roll back a deployment?", found.chunks)
        verified = verify_citations(answer.statements, citation_passages(found.chunks))
        rendered, sources = render_answer(verified.statements, found.chunks)
        assert "rollout undo" in rendered and "[1]" in rendered
        assert verified.all_supported and sources[0].source == "rollback.md"
    finally:
        engine.delete_collection(cid)
