.PHONY: install infra test test-integration lint format

install:
	uv sync

infra:
	docker compose up -d qdrant postgres

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check .

format:
	uv run ruff format .
