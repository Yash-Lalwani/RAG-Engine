from rag_engine.grading.citations import verify_citations
from rag_engine.models import Passage, Statement

PASSAGES = [
    Passage(id="p-pods", text="A Pod is the smallest deployable unit in Kubernetes."),
    Passage(id="p-ingress", text="Ingress exposes HTTP routes."),
]
STATEMENTS = [
    Statement(text="Here is what the docs say.", chunk_ids=[]),
    Statement(text="A Pod is the smallest deployable unit.", chunk_ids=["p-pods"]),
    Statement(text="Pods are billed per second.", chunk_ids=["p-pods"]),
]


def judged(*items):
    return lambda schema: schema(
        checks=[{"statement": i, "supported": ok, "reason": "because"} for i, ok in items]
    )


def test_default_mode_removes_the_unsupported_statement(fake_structured):
    fake_structured(judged((1, True), (2, False)))
    result = verify_citations(STATEMENTS, PASSAGES)
    assert [s.text for s in result.statements] == [STATEMENTS[0].text, STATEMENTS[1].text]
    assert result.removed_count == 1 and not result.all_supported
    assert [c.index for c in result.failing] == [2]


def test_strict_mode_flags_instead_of_removing(fake_structured):
    fake_structured(judged((1, True), (2, False)))
    result = verify_citations(STATEMENTS, PASSAGES, strict=True)
    assert result.statements == STATEMENTS and result.removed_count == 0
    assert not result.all_supported and [c.text for c in result.failing] == [STATEMENTS[2].text]


def test_uncited_statements_are_not_checked(fake_structured):
    calls = fake_structured(judged((1, True), (2, True)))
    result = verify_citations(STATEMENTS, PASSAGES)
    assert result.checks[0].supported is None and result.all_supported and result.checked == 2
    assert "Here is what the docs say" not in calls[0]["user"]


def test_citing_a_passage_that_was_not_supplied_fails_without_the_llm(fake_structured):
    calls = fake_structured(judged())
    result = verify_citations([Statement(text="X.", chunk_ids=["missing"])], PASSAGES)
    assert calls == [] and result.removed_count == 1 and "not provided" in result.failing[0].reason


def test_verification_failure_keeps_statements_marked_not_verified(fake_structured):
    fake_structured(RuntimeError("OpenAI timeout"))
    result = verify_citations(STATEMENTS, PASSAGES)
    assert result.statements == STATEMENTS and result.removed_count == 0
    assert not result.all_supported and result.checked == 0
    assert "OpenAI timeout" in result.warnings[0]


def test_passages_get_short_labels_in_the_prompt(fake_structured):
    calls = fake_structured(judged((1, True), (2, True)))
    verify_citations(STATEMENTS, PASSAGES)
    assert '<document id="p1"' in calls[0]["user"] and "p-pods" not in calls[0]["user"]
