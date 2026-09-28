"""Create the k8s-demo collection and ingest the Kubernetes docs. Safe to re-run.

Run from the repository root: uv run python scripts/seed_demo.py
"""

import logging
from pathlib import Path

from rag_engine import engine
from rag_engine.models import EngineError

COLLECTION_ID = "k8s-demo"
DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "k8s_docs"
DEMO_TABLES = ["clusters", "nodes", "deployments", "pods", "incidents", "alerts", "oncall_logs"]
SETTINGS = {
    "domain_description": "Kubernetes documentation (concepts, tasks, tutorials and API reference)",
    "sql_database": "k8s_ops",
    "sql_allowed_tables": DEMO_TABLES,
    "filterable_fields": ["section"],
}


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    engine.setup()

    if COLLECTION_ID in {c.id for c in engine.list_collections()}:
        engine.update_collection(COLLECTION_ID, SETTINGS)
    else:
        engine.create_collection(COLLECTION_ID, "Kubernetes demo", settings=SETTINGS)

    files = sorted(p for p in DOCS_DIR.iterdir() if p.is_file() and not p.name.startswith("."))
    counts: dict[str, int] = {}
    for number, path in enumerate(files, start=1):
        section = path.name.split("__")[0]
        try:
            result = engine.ingest_document(
                COLLECTION_ID, file_path=str(path), metadata={"section": section}
            )
            status, detail = result.status, f"{result.chunk_count:4d} chunks"
        except EngineError as error:
            status, detail = "failed", str(error)
        counts[status] = counts.get(status, 0) + 1
        print(f"[{number:2d}/{len(files)}] {status:9s} {detail}  {path.name}")

    print("Done:", ", ".join(f"{count} {status}" for status, count in sorted(counts.items())))


if __name__ == "__main__":
    main()
