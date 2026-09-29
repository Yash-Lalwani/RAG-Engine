"""Compare two evaluation runs.

    uv run python eval/diff.py eval/results/<old>.json eval/results/<new>.json
"""

import json
import sys

METRICS = ["recall_at_5", "mrr", "faithfulness", "answer_relevancy", "context_precision",
           "context_recall", "sql_match", "web_handled", "latency_s", "cost_usd"]
CHANGE = 0.2  # report a golden when one of its scores moves at least this much


def summary_deltas(old: dict, new: dict) -> list[str]:
    lines = ["| profile | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for name in new["profiles"]:
        if name not in old["profiles"]:
            continue
        before, after = old["profiles"][name]["summary"], new["profiles"][name]["summary"]
        cells = []
        for metric in METRICS:
            if before.get(metric) is None or after.get(metric) is None:
                cells.append("–")
            else:
                delta = after[metric] - before[metric]
                cells.append(f"{after[metric]} ({delta:+.3f})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def changed_goldens(old: dict, new: dict) -> list[str]:
    lines = []
    for name, profile in new["profiles"].items():
        before_rows = {r["id"]: r for r in old["profiles"].get(name, {}).get("rows", [])}
        for row in profile["rows"]:
            before = before_rows.get(row["id"])
            if before is None:
                continue
            for metric, value in _scores(row).items():
                old_value = _scores(before).get(metric)
                if value is not None and old_value is not None and abs(value - old_value) >= CHANGE:
                    lines.append(f"- {name} {row['id']} {metric}: {old_value} -> {value}")
    return lines


def _scores(row: dict) -> dict:
    return {"recall_at_5": row.get("recall_at_5"), "mrr": row.get("mrr"),
            "sql_match": None if row.get("sql_match") is None else float(row["sql_match"]),
            **{k: v for k, v in row.get("ragas", {}).items()}}


def main(old_path: str, new_path: str) -> None:
    old, new = json.load(open(old_path)), json.load(open(new_path))
    print(f"{old['run_id']} -> {new['run_id']}\n")
    print("\n".join(summary_deltas(old, new)))
    changes = changed_goldens(old, new)
    print(f"\nGoldens that moved by {CHANGE} or more:" if changes else "\nNo golden moved by 0.2 or more.")
    print("\n".join(changes))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
