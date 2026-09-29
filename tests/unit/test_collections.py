import pytest
from pydantic import ValidationError

from rag_engine.collections import stored_settings


def test_removed_settings_are_ignored_with_a_warning(caplog):
    settings = stored_settings("k8s-demo", {"top_k": 3, "contextualize_chunks": False})
    assert settings.top_k == 3
    assert "contextualize_chunks" in caplog.text


def test_invalid_values_of_known_settings_still_fail():
    with pytest.raises(ValidationError):
        stored_settings("k8s-demo", {"search_mode": "fuzzy"})
