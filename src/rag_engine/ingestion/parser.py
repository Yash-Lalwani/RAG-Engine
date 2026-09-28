from functools import lru_cache
from io import BytesIO

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from rag_engine.config import settings
from rag_engine.models import EngineError


@lru_cache
def _converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice(settings.docling_device.lower())
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def parse_document(data: bytes, filename: str) -> DoclingDocument:
    """Parse raw bytes with Docling. The filename's extension tells Docling the format."""
    try:
        return _converter().convert(DocumentStream(name=filename, stream=BytesIO(data))).document
    except Exception as exc:
        raise EngineError(f"Could not parse {filename!r}: {exc}") from exc
