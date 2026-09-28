from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    llm_model_strong: str = "gpt-4o"
    llm_model_small: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    qdrant_url: str = "http://localhost:6333"
    database_url: str = "postgresql://rag:rag@localhost:5432/rag_engine"
    sql_databases: str = "k8s_ops=postgresql://readonly:readonly@localhost:5432/k8s_ops"

    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""

    engine_api_keys: str = ""
    rate_limit_per_minute: int = 100
    daily_token_budget: int = 100_000

    prompt_injection_threshold: float = 0.75
    toxicity_threshold: float = 0.75
    output_toxicity_threshold: float = 0.5

    tavily_api_key: str = ""
    voyage_api_key: str = ""
    voyage_model: str = "rerank-2.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-engine"

    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000
    docling_device: str = "auto"

    @property
    def api_key_callers(self) -> dict[str, str]:
        return parse_api_keys(self.engine_api_keys)

    @property
    def sql_database_urls(self) -> dict[str, str]:
        return parse_sql_databases(self.sql_databases)


def parse_api_keys(raw: str) -> dict[str, str]:
    """Parse "dev:key1,layer:key2" into {"key1": "dev", "key2": "layer"} (key -> caller)."""
    callers: dict[str, str] = {}
    for entry in _split_entries(raw):
        caller, sep, key = entry.partition(":")
        caller, key = caller.strip(), key.strip()
        if not sep or not caller or not key:
            raise ValueError(f"ENGINE_API_KEYS entry must look like 'caller:key', got {entry!r}")
        if key in callers:
            raise ValueError(f"ENGINE_API_KEYS uses the same key for {callers[key]!r} and {caller!r}")
        callers[key] = caller
    return callers


def parse_sql_databases(raw: str) -> dict[str, str]:
    """Parse "k8s_ops=postgresql://..." (comma-separated) into {"k8s_ops": "postgresql://..."}."""
    databases: dict[str, str] = {}
    for entry in _split_entries(raw):
        name, sep, url = entry.partition("=")
        name, url = name.strip(), url.strip()
        if not sep or not name or not url:
            raise ValueError(f"SQL_DATABASES entry must look like 'name=url', got {entry!r}")
        databases[name] = url
    return databases


def _split_entries(raw: str) -> list[str]:
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


settings = Settings()
