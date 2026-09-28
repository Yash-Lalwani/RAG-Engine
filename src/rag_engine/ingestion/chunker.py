from functools import lru_cache

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument


@lru_cache
def _chunker() -> HybridChunker:
    return HybridChunker()


def chunk_document(doc: DoclingDocument) -> list[dict]:
    """Split a parsed document into chunks: {"text": ..., "page_number": int | None}."""
    chunks = []
    for chunk in _chunker().chunk(doc):
        text = chunk.text.strip()
        if not text:
            continue
        page_number = None
        items = getattr(chunk.meta, "doc_items", None)
        if items and items[0].prov:
            page_number = items[0].prov[0].page_no
        chunks.append({"text": text, "page_number": page_number})
    return chunks
