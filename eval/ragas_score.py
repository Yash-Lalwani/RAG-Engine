"""Score answers with Ragas. Runs in its own environment (Ragas 0.4.3 needs openai<3.4 and
langchain-community<0.4), started by run_eval.py:

    uv run --isolated --no-project --with ragas==0.4.3 --with "openai<3.4" \
        --with "langchain-community<0.4" python eval/ragas_score.py samples.json scores.json

samples.json: [{"id", "question", "answer", "contexts", "reference"}, ...]
scores.json:  {"scores": {id: {metric: value}}, "tokens": {"prompt": n, "completion": n}}
"""

import json
import math
import sys
import warnings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

warnings.filterwarnings("ignore")
JUDGE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
METRIC_NAMES = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}


def main(samples_path: str, scores_path: str) -> None:
    samples = json.load(open(samples_path))
    dataset = EvaluationDataset.from_list([
        {"user_input": s["question"], "response": s["answer"],
         "retrieved_contexts": s["contexts"] or [""], "reference": s["reference"]}
        for s in samples
    ])
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        # A per-request timeout: without one, a hanging request waits for the whole job timeout.
        llm=LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, timeout=60, max_retries=3)),
        embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL, timeout=60, max_retries=3)),
        token_usage_parser=get_token_usage_for_openai,
        # Few parallel jobs: the default 16 hits OpenAI's per-minute limits and times out.
        run_config=RunConfig(max_workers=4, timeout=180),
        show_progress=False,
    )
    rows = result.to_pandas().to_dict(orient="records")
    scores = {
        sample["id"]: {short: _clean(row.get(long)) for long, short in METRIC_NAMES.items()}
        for sample, row in zip(samples, rows, strict=True)
    }
    usage = result.total_tokens()
    json.dump({"scores": scores, "tokens": {"prompt": usage.input_tokens, "completion": usage.output_tokens}},
              open(scores_path, "w"), indent=1)


def _clean(value) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(float(value), 3)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
