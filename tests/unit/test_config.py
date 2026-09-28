import pytest

from rag_engine.config import Settings, parse_api_keys, parse_sql_databases


def test_parse_api_keys_maps_key_to_caller():
    assert parse_api_keys("dev:k1, layer:k2 ,astra:k3") == {"k1": "dev", "k2": "layer", "k3": "astra"}


def test_parse_api_keys_empty_means_no_callers():
    assert parse_api_keys("") == {}


@pytest.mark.parametrize("raw", ["dev", "dev:", ":k1"])
def test_parse_api_keys_rejects_malformed_entries(raw):
    with pytest.raises(ValueError):
        parse_api_keys(raw)


def test_parse_api_keys_rejects_shared_key():
    with pytest.raises(ValueError):
        parse_api_keys("dev:same,layer:same")


def test_parse_sql_databases_keeps_equals_signs_in_url():
    raw = "k8s_ops=postgresql://ro:ro@db:5432/k8s_ops?sslmode=disable,other=postgresql://x/y"
    assert parse_sql_databases(raw) == {
        "k8s_ops": "postgresql://ro:ro@db:5432/k8s_ops?sslmode=disable",
        "other": "postgresql://x/y",
    }


def test_parse_sql_databases_rejects_malformed_entry():
    with pytest.raises(ValueError):
        parse_sql_databases("k8s_ops")


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("ENGINE_API_KEYS", "dev:abc")
    monkeypatch.setenv("SQL_DATABASES", "k8s_ops=postgresql://ro@db/k8s_ops")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "7")

    settings = Settings(_env_file=None)

    assert settings.api_key_callers == {"abc": "dev"}
    assert settings.sql_database_urls == {"k8s_ops": "postgresql://ro@db/k8s_ops"}
    assert settings.rate_limit_per_minute == 7
