"""Download every local model at image build time (run by the Dockerfile), so the server never
downloads anything when it starts. Unlike engine.warm_up(), any failure here fails the build."""

import os
from pathlib import Path

from fastembed import SparseTextEmbedding

from rag_engine.guardrails import input_checks
from rag_engine.ingestion.chunker import chunk_document
from rag_engine.ingestion.parser import parse_document
from rag_engine.models import Passage
from rag_engine.retrieval import embeddings, rerank


def main() -> None:
    rerank.rerank("warm up", [Passage(id="w", text="warm up")])  # cross-encoder
    embeddings.embed_sparse_query("warm up")  # BM25
    _add_missing_bm25_files()
    input_checks.injection_score("warm up")  # prompt-injection classifier
    input_checks.toxicity_score("warm up")  # toxicity classifier
    chunk_document(parse_document(b"# Warm up\n\nLoads the chunker's tokenizer.\n", "warm-up.md"))
    print("All local models downloaded.")


def _add_missing_bm25_files() -> None:
    """fastembed's offline check (HF_HUB_OFFLINE=1 at runtime) needs every file BM25 declares, but
    the BM25 repository does not ship some of them ("mock.file", "tamil.txt"). Empty placeholders
    satisfy the check; an empty stop-word list only affects Tamil text."""
    bm25 = next(m for m in SparseTextEmbedding.list_supported_models() if m["model"] == "Qdrant/bm25")
    declared = [bm25["model_file"], *bm25["additional_files"]]
    for snapshot in Path(os.environ["FASTEMBED_CACHE_PATH"]).glob("models--Qdrant--bm25/snapshots/*"):
        for name in declared:
            (snapshot / name).touch(exist_ok=True)


if __name__ == "__main__":
    main()
