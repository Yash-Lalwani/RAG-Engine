"""Metrics computed without an LLM: retrieval quality, SQL execution match, forbidden words."""

from statistics import mean
from typing import Any


def unique_in_order(sources: list[str]) -> list[str]:
    """Chunks become documents: keep each source once, at its best rank."""
    return list(dict.fromkeys(sources))


def recall_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float | None:
    """Share of the relevant documents that appear in the top k retrieved documents."""
    if not relevant:
        return None
    top = set(unique_in_order(retrieved)[:k])
    return len(top & set(relevant)) / len(set(relevant))


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float | None:
    """1 / rank of the first relevant document (0 if none was retrieved)."""
    if not relevant:
        return None
    for rank, source in enumerate(unique_in_order(retrieved), start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def sql_execution_match(generated: list[dict[str, Any]], reference: list[dict[str, Any]]) -> bool:
    """True if the generated SQL returned the reference rows. Column names and order may differ,
    and generated rows may carry extra columns (e.g. a count next to the name that was asked for)."""
    if len(generated) != len(reference):
        return False
    remaining = [set(_normalize(v) for v in row.values()) for row in generated]
    for row in reference:
        wanted = {_normalize(v) for v in row.values()}
        match = next((r for r in remaining if wanted <= r), None)
        if match is None:
            return False
        remaining.remove(match)
    return True


def forbidden_hits(answer: str, forbidden: list[str]) -> list[str]:
    return [word for word in forbidden if word.lower() in answer.lower()]


def average(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(mean(known), 3) if known else None


def _normalize(value: Any) -> str:
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.2f}"
    return str(value).strip()
