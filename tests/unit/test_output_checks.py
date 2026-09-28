from rag_engine.guardrails import output_checks
from rag_engine.guardrails.output_checks import guard_answer, luhn_valid, redact_pii
from rag_engine.models import AskResult, Statement, StatementCheck, VerificationResult

PII_TEXT = (
    "Mail jane.doe@example.com or call 415-555-0132 or +1 (415) 555-0199. "
    "SSN 123-45-6789. Card 4111 1111 1111 1111."
)


def test_pii_including_ssn_is_redacted():
    assert redact_pii(PII_TEXT) == (
        "Mail [REDACTED_EMAIL] or call [REDACTED_PHONE] or [REDACTED_PHONE]. "
        "SSN [REDACTED_SSN]. Card [REDACTED_CARD]."
    )


def test_ordinary_numbers_are_left_alone():
    text = ("Pod 10.0.0.12 has 3 replicas; incident 12345; order 1234 5678 9012 3456; "
            "port 8080; 7621 pods; date 2026-01-15; version 1.29.3")
    assert redact_pii(text) == text


def test_luhn():
    assert luhn_valid("4111111111111111") and luhn_valid("5500005555555559")
    assert not luhn_valid("1234567890123456")


def completed(text: str) -> AskResult:
    statement = Statement(text=text, chunk_ids=["c1"])
    check = StatementCheck(index=0, text=text, chunk_ids=["c1"], supported=False, reason="x")
    return AskResult(
        status="completed", query_id="q", intent="rag", answer=text + " [1]", statements=[statement],
        verification=VerificationResult(strict=True, all_supported=False, checked=1, removed_count=0,
                                        statements=[statement], checks=[check], failing=[check]),
    )


def test_guard_answer_redacts_answer_statements_and_verification():
    result = guard_answer(completed("The on-call contact is ops@example.com, SSN 123-45-6789."))
    expected = "The on-call contact is [REDACTED_EMAIL], SSN [REDACTED_SSN]."
    assert result.answer == expected + " [1]" and result.statements[0].text == expected
    v = result.verification
    assert v.statements[0].text == v.checks[0].text == v.failing[0].text == expected


def test_toxic_answers_are_blocked(monkeypatch):
    monkeypatch.setattr(output_checks, "toxicity_score", lambda text: 0.95)
    result = guard_answer(completed("something awful"))
    assert result.status == "blocked" and result.answer == "" and "moderation" in result.message


def test_moderation_failure_still_redacts_and_warns(monkeypatch):
    def broken(text):
        raise OSError("model missing")

    monkeypatch.setattr(output_checks, "toxicity_score", broken)
    result = guard_answer(completed("Call 415-555-0132."))
    assert result.answer.startswith("Call [REDACTED_PHONE].") and "unavailable" in result.metadata.warnings[0]


def test_non_completed_results_are_untouched():
    pending = AskResult(status="pending_sql", query_id="q", sql="SELECT 1")
    assert guard_answer(pending) is pending
