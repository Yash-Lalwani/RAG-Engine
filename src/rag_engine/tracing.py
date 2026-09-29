"""LangSmith tracing. On only when LANGSMITH_TRACING is true AND LANGSMITH_API_KEY is set."""

import os
import warnings
from typing import Any

from langsmith import get_current_run_tree
from langsmith import utils as langsmith_utils

from rag_engine.config import settings

# When the tracer serializes an OpenAI structured-output response, Pydantic warns about the
# `parsed` field on every LLM call. The trace is correct; the warning is only noise.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)


def tracing_enabled() -> bool:
    return settings.langsmith_tracing and bool(settings.langsmith_api_key)


def configure_tracing() -> bool:
    """Export the settings for LangSmith (used by @traceable and the LangGraph graph) or force
    tracing off. Returns whether tracing is on."""
    enabled = tracing_enabled()
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    langsmith_utils.get_env_var.cache_clear()  # LangSmith caches environment reads
    return enabled


def tag_current_run(tags: list[str], metadata: dict[str, Any]) -> None:
    """Add tags and metadata to the run of the @traceable function that is running now."""
    run = get_current_run_tree()
    if run is not None:
        run.add_tags(tags)
        run.add_metadata(metadata)
