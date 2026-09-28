import logging

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument

logger = logging.getLogger(__name__)


def chunk_document(doc: DoclingDocument, source_name: str) -> list[dict]:
    chunks = []
    for chunk in HybridChunker().chunk(doc):
        meta = {"text": chunk.text, "source": source_name}
        if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
            items = chunk.meta.doc_items
            if items and hasattr(items[0], "prov") and items[0].prov:
                meta["page_number"] = items[0].prov[0].page_no
        chunks.append(meta)
    logger.info("Processed %d chunks from %s", len(chunks), source_name)
    return chunks
