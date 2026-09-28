from pydantic import BaseModel, Field


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
