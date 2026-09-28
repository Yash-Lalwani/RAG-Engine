import pytest

from rag_engine.retrieval.fusion import rrf_fuse


def test_rrf_scores_match_hand_computed_values():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "d"]], rrf_k=60))

    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 62)


def test_rrf_orders_by_score_and_rewards_agreement():
    order = [item for item, _ in rrf_fuse([["a", "b", "c"], ["b", "d"]], rrf_k=60)]
    assert order == ["b", "a", "d", "c"]


def test_rrf_k_changes_scores():
    assert dict(rrf_fuse([["a"]], rrf_k=10))["a"] == pytest.approx(1 / 11)


def test_rrf_single_list_keeps_order():
    assert [item for item, _ in rrf_fuse([["x", "y", "z"]])] == ["x", "y", "z"]


def test_rrf_empty_input():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
