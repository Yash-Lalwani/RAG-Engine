import re
from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)


class EngineError(Exception):
    """Bad input or a missing resource. The message is meant for the caller."""


class Blocked(EngineError):
    """A guardrail stopped the request. The message is the short reason shown to the caller."""


class AuthError(EngineError):
    """Missing or invalid API key."""


def validation_message(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in e['loc']) or 'value'}: {e['msg']}" for e in error.errors()
    )


class CollectionSettings(BaseModel):
    """Per-collection defaults. Per-call options are merged on top with `merged()`."""

    model_config = ConfigDict(extra="forbid")

    search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = Field(5, ge=1, le=50)
    fetch_k: int = Field(20, ge=1, le=200)
    rrf_k: int = Field(60, ge=1)
    rerank: bool = True
    reranker: Literal["local", "voyage"] = "local"
    hyde: bool = False
    crag: bool = True
    crag_threshold: float = Field(0.5, ge=0.0, le=1.0)
    crag_web_fallback: bool = False
    self_rag: bool = False
    self_rag_threshold: float = Field(0.7, ge=0.0, le=1.0)
    citation_mode: Literal["verify", "strict"] = "verify"
    domain_description: str = ""
    sql_database: str | None = None
    sql_allowed_tables: list[str] = Field(default_factory=list)
    filterable_fields: list[str] = Field(default_factory=list)

    @field_validator("filterable_fields")
    @classmethod
    def _simple_field_names(cls, fields: list[str]) -> list[str]:
        for field in fields:
            if not re.fullmatch(r"[A-Za-z0-9_]+", field):
                raise ValueError(f"filterable field {field!r} may only use letters, digits and _")
        return fields

    @model_validator(mode="after")
    def _fetch_at_least_top_k(self) -> "CollectionSettings":
        if self.fetch_k < self.top_k:
            raise ValueError(f"fetch_k ({self.fetch_k}) must be at least top_k ({self.top_k})")
        return self

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "CollectionSettings":
        try:
            return cls.model_validate(values or {})
        except ValidationError as error:
            raise EngineError(f"Invalid settings: {validation_message(error)}") from None

    def merged(self, overrides: dict[str, Any] | None) -> "CollectionSettings":
        return CollectionSettings.from_dict({**self.model_dump(), **(overrides or {})})


class Collection(BaseModel):
    id: str
    name: str
    description: str = ""
    settings: CollectionSettings
    version: int = 0
    created_at: datetime
    document_count: int = 0


class DocumentRecord(BaseModel):
    collection_id: str
    doc_id: str
    source_name: str
    content_hash: str
    chunk_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime


class IngestResult(BaseModel):
    collection_id: str
    doc_id: str
    status: Literal["ingested", "unchanged", "replaced"]
    chunk_count: int


class DeleteResult(BaseModel):
    collection_id: str
    doc_id: str | None = None
    deleted: bool = True


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded"]
    postgres: bool
    qdrant: bool
    redis: Literal["ok", "not configured", "error"]


class Passage(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankedPassage(Passage):
    score: float


class SearchChunk(BaseModel):
    id: str
    doc_id: str
    source: str
    chunk_index: int
    text: str
    fused_score: float
    rerank_score: float | None = None
    grade: str | None = None
    grade_score: float | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchInfo(BaseModel):
    mode: str
    reranked: bool = False
    hyde_used: bool = False
    crag_action: str = "not_run"
    web_used: bool = False
    insufficient_context: bool = False
    candidates: int = 0
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class SearchResult(BaseModel):
    collection_id: str
    query: str
    chunks: list[SearchChunk]
    info: SearchInfo


class Statement(BaseModel):
    text: str
    chunk_ids: list[str]  # ids of supporting chunks; empty only for non-factual text


class CitedAnswer(BaseModel):
    statements: list[Statement]
    insufficient_context: bool


class Source(BaseModel):
    number: int
    chunk_id: str
    doc_id: str | None = None
    source: str
    url: str | None = None
    chunk_index: int | None = None


class SelfCheck(BaseModel):
    groundedness: float
    completeness: float
    score: float
    reason: str


class StatementCheck(BaseModel):
    index: int
    text: str
    chunk_ids: list[str]
    supported: bool | None  # None: not checked (no citations) or not verified (check failed)
    reason: str


class VerificationResult(BaseModel):
    strict: bool
    all_supported: bool
    checked: int
    removed_count: int
    statements: list[Statement]
    checks: list[StatementCheck]
    failing: list[StatementCheck]
    warnings: list[str] = Field(default_factory=list)


class SqlDraft(BaseModel):
    sql: str
    explanation: str


class SqlResult(BaseModel):
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChunkPreview(BaseModel):
    id: str
    doc_id: str
    source: str
    text: str  # first 300 characters
    fused_score: float
    rerank_score: float | None = None
    grade: str | None = None
    grade_score: float | None = None
    url: str | None = None


class SelfRagInfo(BaseModel):
    first_score: float | None = None
    retry_score: float | None = None
    retried: bool = False
    kept: Literal["first", "retry"] = "first"
    retry_query: str | None = None


class AskMetadata(BaseModel):
    cache_hit: bool = False
    cache: dict[str, dict[str, int]] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    search: SearchInfo | None = None
    self_rag: SelfRagInfo | None = None
    sql_row_count: int | None = None
    sql_truncated: bool = False
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    warnings: list[str] = Field(default_factory=list)


class AskResult(BaseModel):
    status: Literal["completed", "pending_sql", "blocked", "error"]
    query_id: str
    intent: Literal["rag", "sql", "hybrid"] | None = None
    answer: str = ""
    statements: list[Statement] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    chunks: list[ChunkPreview] = Field(default_factory=list)
    sql: str | None = None
    sql_explanation: str | None = None
    rows_preview: list[dict[str, Any]] = Field(default_factory=list)
    verification: VerificationResult | None = None
    insufficient_context: bool = False
    message: str | None = None
    metadata: AskMetadata = Field(default_factory=AskMetadata)
