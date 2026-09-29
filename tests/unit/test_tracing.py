import os

import pytest

from rag_engine import tracing
from rag_engine.config import settings


@pytest.fixture(autouse=True)
def restore(monkeypatch):
    yield
    monkeypatch.setattr(settings, "langsmith_tracing", False)
    tracing.configure_tracing()


@pytest.mark.parametrize(("flag", "key", "expected"), [(True, "", False), (False, "lsv2-x", False), (True, "lsv2-x", True)])
def test_tracing_needs_the_flag_and_a_key(monkeypatch, flag, key, expected):
    monkeypatch.setattr(settings, "langsmith_tracing", flag)
    monkeypatch.setattr(settings, "langsmith_api_key", key)
    assert tracing.configure_tracing() is expected
    assert os.environ["LANGSMITH_TRACING"] == ("true" if expected else "false")


def test_tag_current_run_is_harmless_outside_a_traced_function():
    tracing.tag_current_run(["caller:dev"], {"collection_id": "k8s-demo"})
