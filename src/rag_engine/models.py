import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class EngineError(Exception):
    """Bad input or a missing resource. The message is meant for the caller."""


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
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchInfo(BaseModel):
    mode: str
    reranked: bool = False
    rerank_error: str | None = None
    hyde_used: bool = False
    crag_action: str = "not_run"
    web_used: bool = False
    candidates: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)


class SearchResult(BaseModel):
    collection_id: str
    query: str
    chunks: list[SearchChunk]
    info: SearchInfo


class RetrievedChunkPreview(BaseModel):
    text: str
    source: str
    score: float = 0.0


class ResponseMetadata(BaseModel):
    route: str = "rag"
    retrieved_chunks: list[RetrievedChunkPreview] = Field(default_factory=list)
    cache_hit: bool = False
    reflection_iterations: int = 0
    reflection_score: float | None = None
    refined_question: str | None = None


class PendingSQLBlock(BaseModel):
    sql: str
    query_id: str
    explanation: str = ""


class ChatResponse(BaseModel):
    answer: str = Field(..., min_length=0)
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    pending_sql: PendingSQLBlock | None = None
    cache_hit: bool = False
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float = 0.0


class CRAGEvaluation(BaseModel):
    relevance_score: float = 0.0
    relevance_label: str = ""
    confidence: float = 0.0
    reasoning: str = ""


class ReflectionResult(BaseModel):
    """Self-RAG reflection on a generated answer."""

    reflection_score: float = 0.0
    needs_regeneration: bool = False
    refined_question: str = ""
    reasoning: str = ""
