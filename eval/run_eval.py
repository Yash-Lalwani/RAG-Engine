"""Run the evaluation: every profile over every golden question, the rrf_k sweep, and Ragas.

    uv run python eval/run_eval.py                       # full run
    uv run python eval/run_eval.py --limit 5             # first 5 goldens only (to check the cost)
    uv run python eval/run_eval.py --profiles dense,all --no-ragas

Writes eval/results/<timestamp>.json and eval/results/summary.md and prints the summary.
Before the first run, build the noise corpus and seed k8s-demo (see the README).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))  # eval/ is a folder of scripts
from metrics import (  # noqa: E402
    average,
    forbidden_hits,
    recall_at_k,
    reciprocal_rank,
    sql_execution_match,
)
from profiles import PROFILES, RRF_K_VALUES, RRF_SWEEP_PROFILE  # noqa: E402

from rag_engine import engine  # noqa: E402
from rag_engine.config import settings  # noqa: E402
from rag_engine.llm import track_usage  # noqa: E402
from rag_engine.retrieval import vector_store  # noqa: E402
from rag_engine.sql.executor import run_sql  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLDENS = ROOT / "eval" / "goldens.yaml"
RESULTS = ROOT / "eval" / "results"
DEFAULT_COLLECTION = "k8s-demo"
# USD per 1M tokens (input, output). List prices at the time of writing; check before relying on them.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}
RAGAS_MODEL = "gpt-4o-mini"
RAGAS_COMMAND = [
    "uv", "run", "--isolated", "--no-project", "--python", "3.12", "--quiet",
    "--with", "ragas==0.4.3", "--with", "openai<3.4", "--with", "langchain-community<0.4",
    "python", str(ROOT / "eval" / "ragas_score.py"),
]


class Golden(BaseModel):
    id: str
    question: str
    intent: Literal["rag", "sql", "hybrid", "web"]
    feature: str
    sources: list[str] = []
    reference: str
    reference_sql: str | None = None
    forbidden_keywords: list[str] = []


def load_goldens(path: Path = GOLDENS) -> list[Golden]:
    goldens = [Golden.model_validate(item) for item in yaml.safe_load(path.read_text())]
    ids = [g.id for g in goldens]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate golden ids")
    known = set(os.listdir(ROOT / "data" / "k8s_docs"))
    missing = [s for g in goldens for s in g.sources if s not in known]
    if missing:
        raise ValueError(f"Golden sources not in data/k8s_docs: {missing}")
    return goldens


def cost_usd(by_model: dict[str, dict[str, int]]) -> float:
    total = 0.0
    for model, tokens in by_model.items():
        prices = next((PRICES[name] for name in sorted(PRICES, key=len, reverse=True) if model.startswith(name)), (0, 0))
        total += (tokens["prompt"] * prices[0] + tokens["completion"] * prices[1]) / 1_000_000
    return total


def evaluate_golden(
    collection_id: str, golden: Golden, options: dict, sql_settings: tuple[str, list[str]]
) -> dict[str, Any]:
    """Ask one question with one profile's options; approve SQL like a user would."""
    started = time.perf_counter()
    with track_usage() as usage:
        result = engine.ask(collection_id, golden.question, options, caller="eval")
        if result.status == "pending_sql":
            result = engine.approve_sql(result.query_id, True, caller="eval")
    sources = [chunk.source for chunk in result.chunks]
    texts = vector_store.get_texts([c.id for c in result.chunks if not c.id.startswith("web-")])
    contexts = [texts.get(chunk.id, chunk.text) for chunk in result.chunks]
    if result.sql and result.rows_preview:
        contexts.append(f"SQL: {result.sql}\nRows: {json.dumps(result.rows_preview, default=str)}")
    search = result.metadata.search

    row: dict[str, Any] = {
        "id": golden.id,
        "feature": golden.feature,
        "intent": golden.intent,
        "status": result.status,
        "routed_intent": result.intent,
        "answer": result.answer or result.message or "",
        "sources": sources,
        "contexts": contexts,
        "recall_at_5": recall_at_k(sources, golden.sources) if golden.intent in ("rag", "hybrid") else None,
        "mrr": reciprocal_rank(sources, golden.sources) if golden.intent in ("rag", "hybrid") else None,
        "web_used": bool(search and search.web_used),
        "abstained": result.insufficient_context or bool(search and search.insufficient_context),
        "forbidden_hits": forbidden_hits(result.answer, golden.forbidden_keywords),
        "sql": result.sql,
        "sql_match": None,
        "latency_s": round(time.perf_counter() - started, 2),
        "tokens": usage.prompt_tokens + usage.completion_tokens,
        "cost_usd": round(cost_usd(usage.by_model), 5),
        "warnings": result.metadata.warnings,
    }
    if golden.reference_sql:
        database, tables = sql_settings
        reference_rows = run_sql(golden.reference_sql, database, tables).rows
        row["sql_match"] = bool(result.rows_preview) and sql_execution_match(result.rows_preview, reference_rows)
    return row


def run_profiles(
    collection_id: str, goldens: list[Golden], names: list[str], sql_settings
) -> dict[str, list[dict]]:
    results = {}
    for name in names:
        print(f"\n== profile {name}")
        rows = []
        for golden in goldens:
            row = evaluate_golden(collection_id, golden, PROFILES[name], sql_settings)
            print(f"  {golden.id} {row['status']:10s} recall={row['recall_at_5']} "
                  f"sql_match={row['sql_match']} {row['latency_s']}s ${row['cost_usd']}")
            rows.append(row)
        results[name] = rows
    return results


def run_rrf_sweep(collection_id: str, goldens: list[Golden]) -> list[dict]:
    """Retrieval only: how rrf_k changes recall@5 and MRR for plain hybrid search."""
    sweep = []
    retrievable = [g for g in goldens if g.sources]
    for rrf_k in RRF_K_VALUES:
        recalls, mrrs = [], []
        for golden in retrievable:
            result = engine.search(collection_id, golden.question, options={**RRF_SWEEP_PROFILE, "rrf_k": rrf_k})
            sources = [chunk.source for chunk in result.chunks]
            recalls.append(recall_at_k(sources, golden.sources))
            mrrs.append(reciprocal_rank(sources, golden.sources))
        sweep.append({"rrf_k": rrf_k, "recall_at_5": average(recalls), "mrr": average(mrrs)})
        print(f"  rrf_k={rrf_k}: recall@5={sweep[-1]['recall_at_5']} MRR={sweep[-1]['mrr']}")
    return sweep


def run_ragas(goldens: list[Golden], results: dict[str, list[dict]]) -> dict[str, Any]:
    """Score answers in Ragas' own environment and attach the scores to the result rows."""
    references = {g.id: g.reference for g in goldens}
    samples = [
        {"id": f"{profile}|{row['id']}", "question": next(g.question for g in goldens if g.id == row["id"]),
         "answer": row["answer"], "contexts": row["contexts"], "reference": references[row["id"]]}
        for profile, rows in results.items() for row in rows
        if row["status"] == "completed" and row["intent"] != "web" and row["answer"]
    ]
    if not samples:
        return {"prompt": 0, "completion": 0}
    with tempfile.TemporaryDirectory() as folder:
        samples_path, scores_path = Path(folder) / "samples.json", Path(folder) / "scores.json"
        samples_path.write_text(json.dumps(samples))
        print(f"\n== Ragas: scoring {len(samples)} answers with {RAGAS_MODEL} (isolated environment)")
        subprocess.run([*RAGAS_COMMAND, str(samples_path), str(scores_path)], check=True,
                       env={**os.environ, "OPENAI_API_KEY": settings.openai_api_key, "PYTHONWARNINGS": "ignore"})
        output = json.loads(scores_path.read_text())
    for profile, rows in results.items():
        for row in rows:
            row["ragas"] = output["scores"].get(f"{profile}|{row['id']}", {})
    return output["tokens"]


def rescore(run_path: Path) -> None:
    """Fill in Ragas scores that are missing in a finished run (e.g. after rate-limit errors)."""
    run = json.loads(run_path.read_text())
    goldens = {g.id: g for g in load_goldens()}
    metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    missing = {
        name: [r for r in profile["rows"] if r["intent"] != "web" and r["status"] == "completed"
               and any(r.get("ragas", {}).get(m) is None for m in metrics)]
        for name, profile in run["profiles"].items()
    }
    print(f"Answers with missing Ragas scores: {sum(len(rows) for rows in missing.values())}")
    tokens = run_ragas(list(goldens.values()), missing)
    for profile in run["profiles"].values():
        profile["summary"] = summarize(profile["rows"], profile["options"])
    extra = cost_usd({RAGAS_MODEL: tokens})
    run["cost_usd"]["ragas"] = round(run["cost_usd"]["ragas"] + extra, 4)
    run["cost_usd"]["total"] = round(run["cost_usd"]["answers"] + run["cost_usd"]["ragas"], 4)
    run_path.write_text(json.dumps(run, indent=1, default=str))
    if run["collection"] == DEFAULT_COLLECTION:
        (RESULTS / "summary.md").write_text(markdown(run))
    print(markdown(run))


def summarize(rows: list[dict], options: dict) -> dict[str, Any]:
    def ragas(metric):
        return average([row.get("ragas", {}).get(metric) for row in rows])

    web_rows = [r for r in rows if r["intent"] == "web"]
    web_ok = [r["web_used"] if options.get("crag_web_fallback") else r["abstained"] for r in web_rows]
    sql_rows = [r["sql_match"] for r in rows if r["sql_match"] is not None]
    return {
        "recall_at_5": average([r["recall_at_5"] for r in rows]),
        "mrr": average([r["mrr"] for r in rows]),
        "faithfulness": ragas("faithfulness"),
        "answer_relevancy": ragas("answer_relevancy"),
        "context_precision": ragas("context_precision"),
        "context_recall": ragas("context_recall"),
        "sql_match": round(sum(sql_rows) / len(sql_rows), 3) if sql_rows else None,
        "web_handled": round(sum(web_ok) / len(web_ok), 3) if web_ok else None,
        "forbidden_hits": sum(len(r["forbidden_hits"]) for r in rows),
        "not_completed": sum(r["status"] != "completed" for r in rows),
        "latency_s": average([r["latency_s"] for r in rows]),
        "tokens_per_question": round(sum(r["tokens"] for r in rows) / len(rows)) if rows else 0,
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
    }


def markdown(run: dict[str, Any]) -> str:
    columns = ["recall_at_5", "mrr", "faithfulness", "answer_relevancy", "context_precision",
               "context_recall", "sql_match", "web_handled", "forbidden_hits", "not_completed",
               "latency_s", "tokens_per_question", "cost_usd"]
    lines = [
        f"# Evaluation summary ({run['run_id']})",
        "",
        f"Collection `{run['collection']}`, {run['goldens']} golden questions. "
        f"Total cost ${run['cost_usd']['total']} (answers ${run['cost_usd']['answers']}, "
        f"Ragas ${run['cost_usd']['ragas']}).",
        "",
        "| profile | " + " | ".join(columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for name, profile in run["profiles"].items():
        values = ["–" if profile["summary"][c] is None else str(profile["summary"][c]) for c in columns]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    if run["rrf_sweep"]:
        lines += ["", "rrf_k sweep (plain hybrid search, retrieval only):", "",
                  "| rrf_k | recall_at_5 | mrr |", "|---|---|---|"]
        lines += [f"| {s['rrf_k']} | {s['recall_at_5']} | {s['mrr']} |" for s in run["rrf_sweep"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", default=",".join(PROFILES), help="comma-separated profile names")
    parser.add_argument("--limit", type=int, help="only the first N goldens")
    parser.add_argument("--no-ragas", action="store_true")
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--trace", action="store_true", help="send the runs to LangSmith")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--rescore", type=Path, help="only fill in missing Ragas scores of this run file")
    args = parser.parse_args()
    if args.rescore:
        rescore(args.rescore)
        return
    collection_id = args.collection

    settings.langsmith_tracing = settings.langsmith_tracing and args.trace
    engine.setup()
    goldens = load_goldens()[: args.limit]
    collection = next((c for c in engine.list_collections() if c.id == collection_id), None)
    if collection is None:
        raise SystemExit(f"{collection_id} does not exist: run scripts/seed_demo.py first")
    noise = sum(d.metadata.get("section") == "noise" for d in engine.list_documents(collection_id))
    print(f"{collection_id}: {collection.document_count} documents ({noise} noise), {len(goldens)} goldens")
    sql_settings = (collection.settings.sql_database, collection.settings.sql_allowed_tables)

    names = [n.strip() for n in args.profiles.split(",")]
    results = run_profiles(collection_id, goldens, names, sql_settings)
    sweep = [] if args.no_sweep else (print("\n== rrf_k sweep") or run_rrf_sweep(collection_id, goldens))
    ragas_tokens = {"prompt": 0, "completion": 0} if args.no_ragas else run_ragas(goldens, results)

    answers_cost = sum(r["cost_usd"] for rows in results.values() for r in rows)
    ragas_cost = cost_usd({RAGAS_MODEL: ragas_tokens})
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = {
        "run_id": run_id,
        "collection": collection_id,
        "goldens": len(goldens),
        "profiles": {n: {"options": PROFILES[n], "summary": summarize(results[n], PROFILES[n]), "rows": results[n]}
                     for n in names},
        "rrf_sweep": sweep,
        "cost_usd": {"answers": round(answers_cost, 4), "ragas": round(ragas_cost, 4),
                     "total": round(answers_cost + ragas_cost, 4)},
        "ragas_tokens": ragas_tokens,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{run_id}.json").write_text(json.dumps(run, indent=1, default=str))
    written = f"eval/results/{run_id}.json"
    if collection_id == DEFAULT_COLLECTION:  # experiments on other collections keep the main summary
        (RESULTS / "summary.md").write_text(markdown(run))
        written += " and eval/results/summary.md"
    print("\n" + markdown(run))
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
