"""G1: basic validation and obvious injection phrases. G2: prompt-injection and toxicity models."""

import logging
import re
from functools import lru_cache

from rag_engine.config import settings
from rag_engine.models import Blocked

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
INJECTION_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
TOXICITY_MODEL = "unitary/unbiased-toxic-roberta"
# The toxicity model also scores mentions of identity groups ("female", "muslim", ...);
# only these labels mean the text itself is harmful.
HARMFUL_LABELS = {
    "toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit",
}

# Only obvious, complete phrases. Subtle cases are left to the G2 model.
INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(ignore|disregard|forget)\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)\b",
        r"\bforget\s+(all\s+)?your\s+(instructions|rules)\b",
        r"\b(reveal|show|print|repeat)\s+(me\s+)?your\s+(system\s+)?(prompt|instructions)\b",
        r"\byou\s+are\s+now\s+(a|an)\b",
        r"\bnew\s+instructions\s*:",
        r"\boverride\s+(your|the|all)\s+(previous\s+)?(instructions|rules)\b",
        r"<\s*script\b",
        r"\bjavascript\s*:",
        r"\bon(error|load|click|mouseover|focus|submit)\s*=",
    ]
]


def guard_input(text: str) -> list[str]:
    """Run G1 then G2 on a question or search query. Raises Blocked with a short reason;
    returns warnings when the G2 models could not run (the request then continues)."""
    reason = check_question(text)
    if reason:
        raise Blocked(reason)
    reason, warnings = scan_question(text)
    if reason:
        raise Blocked(reason)
    return warnings


def check_question(text: str) -> str | None:
    """G1. Return a short reason if the text must be blocked, else None."""
    text = text.strip()
    if len(text) > MAX_QUESTION_CHARS:
        return f"The question is longer than {MAX_QUESTION_CHARS} characters"
    if re.fullmatch(r"[\W_]+", text):
        return "The question must contain words"
    if any(pattern.search(text) for pattern in INJECTION_PATTERNS):
        return "The question looks like a prompt-injection attempt"
    return None


def scan_question(text: str) -> tuple[str | None, list[str]]:
    """G2. Returns (block reason or None, warnings)."""
    try:
        if injection_score(text) >= settings.prompt_injection_threshold:
            return "The question looks like a prompt-injection attempt", []
        if toxicity_score(text) >= settings.toxicity_threshold:
            return "The question was blocked as toxic", []
    except Exception as exc:
        logger.warning("Input scan failed: %s", exc)
        return None, [f"Input scan (G2) unavailable, continued without it: {exc}"]
    return None, []


def injection_score(text: str) -> float:
    scores = _scores(INJECTION_MODEL, text)
    return scores.get("INJECTION", 0.0)


def toxicity_score(text: str) -> float:
    scores = _scores(TOXICITY_MODEL, text)
    return max((score for label, score in scores.items() if label in HARMFUL_LABELS), default=0.0)


def _scores(model: str, text: str) -> dict[str, float]:
    return {result["label"]: result["score"] for result in _classifier(model)([text])[0]}


@lru_cache
def _classifier(model: str):
    """Load a Hugging Face text classifier once (downloaded on first use)."""
    from transformers import pipeline

    return pipeline("text-classification", model=model, top_k=None, truncation=True, max_length=512)
