"""verify_citations(): check that every cited statement is supported by the passages it cites."""

import logging

from langsmith import traceable
from pydantic import BaseModel

from rag_engine import llm
from rag_engine.config import settings
from rag_engine.generation import prompts
from rag_engine.guardrails.spotlight import neutralize, spotlight_documents
from rag_engine.models import Passage, Statement, StatementCheck, VerificationResult

logger = logging.getLogger(__name__)


class _Check(BaseModel):
    statement: int
    supported: bool
    reason: str


class _Checks(BaseModel):
    checks: list[_Check]


@traceable(name="verify_citations")
def verify_citations(
    statements: list[Statement], passages: list[Passage], strict: bool = False
) -> VerificationResult:
    """Default mode removes unsupported statements; strict mode keeps them and flags the result."""
    by_id = {p.id: p for p in passages}
    checks: dict[int, StatementCheck] = {}
    to_judge: list[int] = []
    for index, statement in enumerate(statements):
        missing = [cid for cid in statement.chunk_ids if cid not in by_id]
        if not statement.chunk_ids:
            checks[index] = _check(index, statement, None, "No citations; not checked")
        elif missing:
            checks[index] = _check(
                index, statement, False, f"Cited passage(s) not provided: {', '.join(missing)}"
            )
        else:
            to_judge.append(index)

    warnings = []
    if to_judge:
        try:
            checks.update(_judge(statements, to_judge, by_id))
        except Exception as exc:
            logger.warning("Citation verification failed: %s", exc)
            warnings.append(f"Citation verification failed: {exc}")
        for index in to_judge:
            if index not in checks:
                checks[index] = _check(index, statements[index], None, "Not verified")

    ordered = [checks[index] for index in range(len(statements))]
    failing = [c for c in ordered if c.supported is False]
    cited = [c for c in ordered if statements[c.index].chunk_ids]
    kept = statements if strict else [s for s, c in zip(statements, ordered, strict=True) if c.supported is not False]
    return VerificationResult(
        strict=strict,
        all_supported=all(c.supported is True for c in cited),
        checked=sum(c.supported is not None for c in cited),
        removed_count=len(statements) - len(kept),
        statements=kept,
        checks=ordered,
        failing=failing,
        warnings=warnings,
    )


def _judge(
    statements: list[Statement], indexes: list[int], by_id: dict[str, Passage]
) -> dict[int, StatementCheck]:
    """One small-model call for all statements. Passages get short labels (p1, p2, ...)."""
    labels: dict[str, str] = {}
    for index in indexes:
        for chunk_id in statements[index].chunk_ids:
            labels.setdefault(chunk_id, f"p{len(labels) + 1}")

    documents = spotlight_documents([(label, "", by_id[cid].text) for cid, label in labels.items()])
    lines = [
        f"S{index}: {neutralize(statements[index].text)} "
        f"(cites: {', '.join(labels[cid] for cid in statements[index].chunk_ids)})"
        for index in indexes
    ]
    result = llm.generate_structured(
        prompts.VERIFY_SYSTEM,
        f"{documents}\n\nStatements (return the number after S as `statement`):\n" + "\n".join(lines),
        _Checks,
        model=settings.llm_model_small,
    )
    wanted = set(indexes)
    return {
        c.statement: _check(c.statement, statements[c.statement], c.supported, c.reason)
        for c in result.checks
        if c.statement in wanted
    }


def _check(index: int, statement: Statement, supported: bool | None, reason: str) -> StatementCheck:
    return StatementCheck(
        index=index, text=statement.text, chunk_ids=statement.chunk_ids, supported=supported, reason=reason
    )
