# RAG-Engine

A reusable advanced RAG engine exposed as an MCP server. Work in progress.

## Development

```bash
cp .env.example .env                      # then fill in your keys
uv sync                                   # install
docker compose up -d qdrant postgres      # Qdrant + Postgres (rag_engine and k8s_ops databases)
uv run pytest                             # unit tests
uv run pytest -m integration              # tests that need the Docker services
```

## Data

- `data/k8s_docs/` contains 47 pages from the [Kubernetes documentation](https://kubernetes.io/docs/),
  © The Kubernetes Authors, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  They are used only as the `k8s-demo` test collection and evaluation dataset.
- `data/sql/001_k8s_ops.sql` is synthetic Kubernetes operations data used as the Text2SQL demo database.
