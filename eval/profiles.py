"""Evaluation profiles: each one is just a set of per-call options for engine.ask()."""

_BASE = {"search_mode": "dense", "top_k": 5, "rerank": False, "hyde": False,
         "crag": False, "crag_web_fallback": False, "self_rag": False}
_HYBRID_RERANK = {**_BASE, "search_mode": "hybrid", "rerank": True}

PROFILES: dict[str, dict] = {
    "dense": _BASE,
    "sparse": {**_BASE, "search_mode": "sparse"},
    "hybrid": {**_BASE, "search_mode": "hybrid"},
    "hybrid+rerank": _HYBRID_RERANK,
    "+hyde": {**_HYBRID_RERANK, "hyde": True},
    "+crag": {**_HYBRID_RERANK, "crag": True, "crag_web_fallback": True},
    "+self_rag": {**_HYBRID_RERANK, "self_rag": True},
    "all": {**_HYBRID_RERANK, "hyde": True, "crag": True, "crag_web_fallback": True, "self_rag": True},
}

# RRF only changes the fused order, so the sweep uses plain hybrid search (no rerank, no grading).
RRF_SWEEP_PROFILE = PROFILES["hybrid"]
RRF_K_VALUES = [10, 60, 120]
