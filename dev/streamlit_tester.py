"""Temporary manual tester for the Engine. It calls rag_engine.engine directly (not MCP) and is
deleted before the final push. Run from the repository root:

    uv run streamlit run dev/streamlit_tester.py
"""

import base64
import json
from typing import Any

import streamlit as st

from rag_engine import engine
from rag_engine.models import AskResult, Collection, CollectionSettings, EngineError

PAGES = ["Collections", "Ingest", "Search", "Ask"]


@st.cache_resource
def setup_engine() -> bool:
    engine.setup()
    return True


def call(function, *args, **kwargs) -> Any:
    """Run an engine call. Bad input (EngineError) is shown as a message, not a traceback."""
    try:
        with st.spinner("Working..."):
            return function(*args, **kwargs)
    except EngineError as error:
        st.error(str(error))
        return None


def parse_json(label: str, text: str) -> dict | None:
    if not text.strip():
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        st.error(f"{label} is not valid JSON: {error}")
        return None
    if not isinstance(value, dict):
        st.error(f"{label} must be a JSON object")
        return None
    return value


def flash(message: str, kind: str = "success") -> None:
    """Show a message after the next st.rerun()."""
    st.session_state["flash"] = (kind, message)


def show_flash() -> None:
    kind, message = st.session_state.pop("flash", (None, None))
    if message:
        getattr(st, kind)(message)


def current_collection() -> Collection | None:
    selected = st.session_state.get("collection")
    return next((c for c in engine.list_collections() if c.id == selected), None)


def collections_page() -> None:
    st.title("Collections")
    st.dataframe(
        [
            {"id": c.id, "name": c.name, "documents": c.document_count, "version": c.version,
             "description": c.description}
            for c in engine.list_collections()
        ],
        hide_index=True,
    )

    with st.form("create_collection"):
        st.subheader("Create a collection")
        collection_id = st.text_input("Collection id", key="new_id")
        name = st.text_input("Name", key="new_name")
        description = st.text_input("Description", key="new_description")
        settings_text = st.text_area("Settings (JSON, optional)", "{}", key="new_settings")
        if st.form_submit_button("Create"):
            settings = parse_json("Settings", settings_text)
            if settings is not None and call(
                engine.create_collection, collection_id, name or collection_id, description, settings
            ):
                st.session_state["select_collection"] = collection_id
                flash(f"Created {collection_id}")
                st.rerun()

    collection = current_collection()
    if collection is None:
        return
    st.subheader(f"Settings of {collection.id}")
    settings_text = st.text_area(
        "Settings (JSON)",
        json.dumps(collection.settings.model_dump(), indent=2),
        height=420,
        key=f"settings_{collection.id}",
    )
    if st.button("Save settings"):
        settings = parse_json("Settings", settings_text)
        if settings is not None and call(engine.update_collection, collection.id, settings):
            flash("Settings saved")
            st.rerun()

    with st.expander("Delete this collection"):
        confirm = st.checkbox(f"Yes, delete {collection.id} and all its documents", key="confirm_delete")
        if st.button("Delete collection", disabled=not confirm):
            if call(engine.delete_collection, collection.id):
                flash(f"Deleted {collection.id}")
                st.rerun()


def ingest_page() -> None:
    st.title("Ingest")
    collection = current_collection()
    if collection is None:
        st.info("Create or select a collection first.")
        return

    files = st.file_uploader("Documents", accept_multiple_files=True)
    metadata_text = st.text_area("Metadata for every file (JSON, optional)", "{}", key="ingest_metadata")
    if st.button("Ingest", disabled=not files):
        metadata = parse_json("Metadata", metadata_text)
        if metadata is not None:
            st.dataframe([ingest_one(collection.id, file, metadata) for file in files], hide_index=True)

    st.subheader("Documents")
    documents = engine.list_documents(collection.id)
    st.dataframe(
        [
            {"doc_id": d.doc_id, "source": d.source_name, "chunks": d.chunk_count,
             "metadata": json.dumps(d.metadata), "ingested_at": d.ingested_at}
            for d in documents
        ],
        hide_index=True,
    )
    if documents:
        doc_id = st.selectbox("Document to delete", [d.doc_id for d in documents], key="delete_doc")
        if st.button("Delete document") and call(engine.delete_document, collection.id, doc_id):
            flash(f"Deleted {doc_id}")
            st.rerun()


def ingest_one(collection_id: str, file, metadata: dict) -> dict:
    try:
        with st.spinner(f"Ingesting {file.name}..."):
            result = engine.ingest_document(
                collection_id,
                content_base64=base64.b64encode(file.getvalue()).decode(),
                filename=file.name,
                metadata=metadata,
            )
        return {"file": file.name, "status": result.status, "chunks": result.chunk_count, "error": ""}
    except EngineError as error:
        return {"file": file.name, "status": "failed", "chunks": 0, "error": str(error)}


def option_widgets(settings: CollectionSettings, page: str) -> dict[str, Any]:
    """Per-call options, pre-filled from the collection's settings."""
    left, middle, right = st.columns(3)
    modes = ["hybrid", "dense", "sparse"]
    options: dict[str, Any] = {
        "search_mode": left.selectbox("Search mode", modes, modes.index(settings.search_mode), key=f"{page}_mode"),
        "top_k": left.number_input("top_k", 1, 50, settings.top_k, key=f"{page}_top_k"),
        "rerank": middle.checkbox("Rerank", settings.rerank, key=f"{page}_rerank"),
        "hyde": middle.checkbox("HyDE", settings.hyde, key=f"{page}_hyde"),
        "crag": right.checkbox("CRAG", settings.crag, key=f"{page}_crag"),
        "crag_web_fallback": right.checkbox("CRAG web fallback", settings.crag_web_fallback, key=f"{page}_web"),
    }
    if page == "ask":
        options["self_rag"] = middle.checkbox("Self-RAG", settings.self_rag, key="ask_self_rag")
        citation_modes = ["verify", "strict"]
        options["citation_mode"] = right.selectbox(
            "Citation mode", citation_modes, citation_modes.index(settings.citation_mode), key="ask_citation"
        )
    return options


def search_page() -> None:
    st.title("Search")
    collection = current_collection()
    if collection is None:
        st.info("Create or select a collection first.")
        return
    query = st.text_input("Query", key="search_query")
    options = option_widgets(collection.settings, "search")
    filters = parse_json("Filters", st.text_input("Filters (JSON, optional)", key="search_filters"))
    if not st.button("Search", disabled=not query.strip()) or filters is None:
        return

    result = call(engine.search, collection.id, query, filters=filters or None, options=options)
    if result is None:
        return
    info = result.info
    st.caption(
        f"mode={info.mode} · reranked={info.reranked} · CRAG={info.crag_action} · HyDE={info.hyde_used}"
        f" · web={info.web_used} · insufficient_context={info.insufficient_context}"
        f" · candidates={info.candidates} · {info.timings_ms.get('total_ms', 0):.0f} ms"
    )
    for warning in info.warnings:
        st.warning(warning)
    st.dataframe(
        [
            {"source": c.source, "chunk": c.chunk_index, "fused": round(c.fused_score, 4),
             "rerank": c.rerank_score, "grade": c.grade, "grade_score": c.grade_score, "text": c.text[:300]}
            for c in result.chunks
        ],
        hide_index=True,
    )
    with st.expander("Timings and full result"):
        st.json(result.model_dump())


def ask_page() -> None:
    st.title("Ask")
    collection = current_collection()
    if collection is None:
        st.info("Create or select a collection first.")
        return
    question = st.text_area("Question", key="ask_question")
    options = option_widgets(collection.settings, "ask")
    filters = parse_json("Filters", st.text_input("Filters (JSON, optional)", key="ask_filters"))
    if st.button("Ask", disabled=not question.strip()) and filters is not None:
        if filters:
            options["filters"] = filters
        st.session_state["ask_result"] = call(
            engine.ask, collection.id, question, options, caller=st.session_state["caller"]
        )

    result: AskResult | None = st.session_state.get("ask_result")
    if result is not None:
        show_result(result)


def decide_sql(result: AskResult, approve: bool) -> None:
    decided = call(engine.approve_sql, result.query_id, approve, caller=st.session_state["caller"])
    if decided is not None and decided.status == "error":
        flash(decided.message or "Approval failed", "error")  # keep showing the pending SQL
    elif decided is not None:
        st.session_state["ask_result"] = decided
    st.rerun()


def show_result(result: AskResult) -> None:
    st.caption(f"status={result.status} · intent={result.intent} · query_id={result.query_id}")
    if result.sql:
        st.code(result.sql, language="sql")
        st.caption(result.sql_explanation or "")
    if result.status == "pending_sql":
        st.info(result.message)
        approve, reject = st.columns(2)
        if approve.button("Approve SQL"):
            decide_sql(result, True)
        if reject.button("Reject SQL"):
            decide_sql(result, False)
        return
    if result.status == "error":
        st.error(result.message)
        return

    st.subheader("Answer")
    st.markdown(result.answer or "_(empty)_")
    for source in result.sources:
        chunk = f" · chunk {source.chunk_index}" if source.chunk_index is not None else ""
        st.markdown(f"[{source.number}] {source.url or source.source}{chunk}")
    if result.message:
        st.info(result.message)
    for warning in result.metadata.warnings:
        st.warning(warning)

    if result.verification:
        v = result.verification
        st.caption(
            f"Verification: all_supported={v.all_supported} · checked={v.checked}"
            f" · removed={v.removed_count} · strict={v.strict}"
        )
        for failing in v.failing:
            st.error(f"Unsupported: {failing.text} ({failing.reason})")
    if result.rows_preview:
        st.subheader("SQL rows")
        st.dataframe(result.rows_preview, hide_index=True)
    if result.chunks:
        st.subheader("Retrieved chunks")
        st.dataframe([c.model_dump() for c in result.chunks], hide_index=True)
    with st.expander("Metadata: timings, cache, tokens, Self-RAG"):
        st.json(result.metadata.model_dump())


def main() -> None:
    st.set_page_config(page_title="RAG-Engine tester", layout="wide")
    setup_engine()
    page = st.sidebar.radio("Page", PAGES, key="page")
    collection_ids = [c.id for c in engine.list_collections()]
    if "select_collection" in st.session_state:  # set by "Create"; applied before the widget exists
        st.session_state["collection"] = st.session_state.pop("select_collection")
    if st.session_state.get("collection") not in collection_ids:
        st.session_state.pop("collection", None)
    if collection_ids:
        st.sidebar.selectbox("Collection", collection_ids, key="collection")
    st.sidebar.text_input("Caller", value="dev", key="caller")
    show_flash()
    {"Collections": collections_page, "Ingest": ingest_page, "Search": search_page, "Ask": ask_page}[page]()


main()
