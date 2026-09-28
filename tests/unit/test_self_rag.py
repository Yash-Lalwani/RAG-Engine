from rag_engine.grading import self_rag


def test_score_is_the_lower_of_groundedness_and_completeness(fake_structured, make_chunk):
    fake_structured(lambda schema: schema(groundedness=0.9, completeness=0.4, reason="misses part"))
    check = self_rag.self_check("q", "answer", [make_chunk("a")])
    assert check.score == 0.4 and check.groundedness == 0.9 and check.reason == "misses part"


def test_scores_are_clamped(fake_structured, make_chunk):
    fake_structured(lambda schema: schema(groundedness=1.4, completeness=-0.2, reason=""))
    check = self_rag.self_check("q", "answer", [make_chunk("a")])
    assert (check.groundedness, check.completeness, check.score) == (1.0, 0.0, 0.0)


def test_at_most_one_retry_and_only_below_threshold():
    assert self_rag.should_retry(0.5, 0.7, retries_done=0)
    assert not self_rag.should_retry(0.5, 0.7, retries_done=1)
    assert not self_rag.should_retry(0.7, 0.7, retries_done=0)


def test_retry_answer_must_score_strictly_higher():
    assert self_rag.keep_retry(0.5, 0.6)
    assert not self_rag.keep_retry(0.6, 0.6)
    assert not self_rag.keep_retry(0.6, 0.4)


def test_rewrite_falls_back_to_the_original_question(monkeypatch):
    from rag_engine import llm

    monkeypatch.setattr(llm, "generate_text", lambda *a, **k: ["  "])
    assert self_rag.rewrite_query("what is a pod", "too vague") == "what is a pod"


def test_rewrite_uses_the_domain_and_strips_quotes(monkeypatch):
    from rag_engine import llm

    seen = {}

    def fake_text(system, user, model=None, temperature=0.0, n=1):
        seen["system"] = system
        return [' "kubectl rollout undo deployment" ']

    monkeypatch.setattr(llm, "generate_text", fake_text)
    rewritten = self_rag.rewrite_query("roll back?", "vague", domain_description="Kubernetes docs")
    assert rewritten == "kubectl rollout undo deployment" and "Kubernetes docs" in seen["system"]
