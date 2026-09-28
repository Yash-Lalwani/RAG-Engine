"""G6: output moderation (toxicity model) and PII redaction (regex) on final answers."""

import logging
import re

from rag_engine.config import settings
from rag_engine.guardrails.input_checks import toxicity_score
from rag_engine.models import AskResult, Statement, StatementCheck

logger = logging.getLogger(__name__)

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
US_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
PHONE = re.compile(r"(?<![\w-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w-])")
CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def guard_answer(result: AskResult) -> AskResult:
    """Block a toxic answer, otherwise redact PII from the answer, its statements and checks."""
    if result.status != "completed" or not result.answer:
        return result
    warnings = []
    try:
        if toxicity_score(result.answer) >= settings.output_toxicity_threshold:
            return AskResult(
                status="blocked",
                query_id=result.query_id,
                intent=result.intent,
                message="The answer was blocked by output moderation",
                metadata=result.metadata,
            )
    except Exception as exc:
        logger.warning("Output moderation failed: %s", exc)
        warnings.append(f"Output moderation (G6) unavailable, continued without it: {exc}")

    updates = {
        "answer": redact_pii(result.answer),
        "statements": _redact_statements(result.statements),
        "metadata": result.metadata.model_copy(
            update={"warnings": result.metadata.warnings + warnings}
        ),
    }
    if result.verification:
        updates["verification"] = result.verification.model_copy(
            update={
                "statements": _redact_statements(result.verification.statements),
                "checks": _redact_checks(result.verification.checks),
                "failing": _redact_checks(result.verification.failing),
            }
        )
    return result.model_copy(update=updates)


def redact_pii(text: str) -> str:
    text = EMAIL.sub("[REDACTED_EMAIL]", text)
    text = US_SSN.sub("[REDACTED_SSN]", text)
    text = CARD_CANDIDATE.sub(_redact_card, text)
    return PHONE.sub("[REDACTED_PHONE]", text)


def _redact_card(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[REDACTED_CARD]" if luhn_valid(digits) else match.group(0)


def luhn_valid(digits: str) -> bool:
    """Card numbers pass the Luhn checksum; most random digit strings do not."""
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char)
        if position % 2 == 1:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def _redact_statements(statements: list[Statement]) -> list[Statement]:
    return [s.model_copy(update={"text": redact_pii(s.text)}) for s in statements]


def _redact_checks(checks: list[StatementCheck]) -> list[StatementCheck]:
    return [c.model_copy(update={"text": redact_pii(c.text)}) for c in checks]
