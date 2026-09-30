#!/usr/bin/env python3
"""Compute the oracle (best-possible) reranking once, independent of any classifier.

Retrieval (BM25/E5 top-N) and the ground-truth gold entities
(`metadata.relevance_ratings`) are both static — they don't depend on which
estimator/classifier is later used to rerank. So instead of recomputing an
oracle upper bound inside every classifier pipeline run, this script runs
*once* per (dataset, retriever/index, top_n, top_k) combination:

1. Retrieve the top-N candidate pool for every query (no LLM/ProbLog involved).
2. Compute the oracle ranking: reorder that same pool so every gold entity
   comes first (a stable partition). Since P@K/R@K/F1@K/NDCG@K/MRR all treat
   relevance as binary, no reranker restricted to this candidate pool can
   ever score higher than this ordering on any of those metrics.
3. Persist per-record oracle results + an aggregated metrics summary to disk.

The resulting files are a static baseline: any number of later classifier/
reranker runs (which share the same retriever/top_n) can be compared against
it directly, without re-running this script.

Example:
  python scripts/compute_oracle_ranking.py \\
      --data-file llm_bayesian_reasoning/data/preprocessed_data/parsed_test.jsonl \\
      --index-path llm_bayesian_reasoning/data/index_data/bm25_index \\
      --index-documents llm_bayesian_reasoning/data/index_data/documents.jsonl \\
      --top-n 50 --top-k 10
"""

import argparse
import json
import logging
from pathlib import Path

from llm_bayesian_reasoning.pipeline.config import RetrieverType
from llm_bayesian_reasoning.pipeline.metrics import (
    compute_record_metrics,
    oracle_ranking,
)
from llm_bayesian_reasoning.retrievers.base_retriever import BaseRetriever
from llm_bayesian_reasoning.retrievers.factory import build_or_load_retriever

logger = logging.getLogger("compute_oracle_ranking")


def _load_queries_with_ground_truth(
    data_file: Path,
    limit: int | None = None,
) -> dict[int | str, dict]:
    """Load {id: {"query": str, "ground_truth": list[str]}} from a preprocessed JSONL.

    Records without `metadata.relevance_ratings` are skipped (there is no
    gold set to compute an oracle against).
    """
    records: dict[int | str, dict] = {}
    with data_file.open(encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(records) >= limit:
                break
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            rid = doc.get("id")
            if rid is None:
                continue

            meta = doc.get("metadata", {})
            rel = meta.get("relevance_ratings") or {}
            if not isinstance(rel, dict) or not rel:
                logger.debug("Record %s has no relevance_ratings — skipping", rid)
                continue

            query = doc.get("query") or doc.get("original_query") or ""
            if not query:
                logger.debug("Record %s has no query — skipping", rid)
                continue

            records[rid] = {"query": query, "ground_truth": list(rel.keys())}

    return records


def compute_oracle_ranking_for_dataset(
    records: dict[int | str, dict],
    retriever: BaseRetriever,
    top_n: int,
    top_k: int,
) -> dict[int | str, dict]:
    """Run retrieval once per query and compute the oracle ranking/metrics.

    Returns a mapping record_id -> result dict (same shape written to disk).
    """
    results: dict[int | str, dict] = {}
    for rid, record in records.items():
        query = record["query"]
        relevant = set(record["ground_truth"])

        retrieval_result = retriever.retrieve(query, top_k=top_n)
        documents = retrieval_result.documents
        retrieved_entities = [document.title for document in documents]
        retrieval_scores = {document.title: document.score for document in documents}

        oracle_ranked_entities = oracle_ranking(retrieved_entities, relevant)
        num_gold_in_pool = sum(1 for e in retrieved_entities if e in relevant)

        record_metrics: dict[str, float] = {}
        record_metrics.update(
            compute_record_metrics(
                retrieved_entities, relevant, top_k, prefix="retrieval"
            )
        )
        record_metrics.update(
            compute_record_metrics(
                oracle_ranked_entities, relevant, top_k, prefix="oracle"
            )
        )

        results[rid] = {
            "query": query,
            "ground_truth": list(relevant),
            "retrieved_entities": retrieved_entities,
            "retrieval_scores": retrieval_scores,
            "oracle_ranked_entities": oracle_ranked_entities,
            "num_gold_in_pool": num_gold_in_pool,
            "reranked_pool_size": len(retrieved_entities),
            "metric_top_k": top_k,
            "record_metrics": record_metrics,
        }
        logger.debug("Record %s: num_gold_in_pool=%d", rid, num_gold_in_pool)

    return results


def _aggregate_metrics(results: dict[int | str, dict]) -> dict[str, float]:
    """Average per-record `record_metrics` across all evaluated records."""
    sums: dict[str, float] = {}
    n = 0
    for record_result in results.values():
        for key, value in record_result["record_metrics"].items():
            sums[key] = sums.get(key, 0.0) + value
        n += 1

    if n == 0:
        return {"num_evaluated": 0}

    averaged = {key: value / n for key, value in sums.items()}
    averaged["num_evaluated"] = float(n)
    return averaged


def _write_results(results: dict[int | str, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rid, record_result in results.items():
            f.write(json.dumps({"id": rid, **record_result}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("llm_bayesian_reasoning/data/preprocessed_data/parsed_test.jsonl"),
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("llm_bayesian_reasoning/data/index_data/bm25_index"),
    )
    parser.add_argument(
        "--index-documents",
        type=Path,
        default=Path("llm_bayesian_reasoning/data/index_data/documents.jsonl"),
    )
    parser.add_argument(
        "--retriever-type",
        type=str,
        choices=[e.value for e in RetrieverType],
        default=RetrieverType.BM25.value,
    )
    parser.add_argument(
        "--retriever-model-name",
        type=str,
        default="intfloat/e5-base-v2",
        help="Dense retriever model identifier (only used for E5).",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--index-limit",
        type=int,
        default=None,
        help="Limit documents indexed when building.",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of dataset records processed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write per-record oracle results JSONL. "
        "Defaults to llm_bayesian_reasoning/results/oracle/oracle_<retriever>_n<top_n>_k<top_k>.jsonl",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Path to write the aggregated metrics summary JSON. "
        "Defaults to <output>_metrics.json next to --output.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    retriever_type = RetrieverType(args.retriever_type)
    output_path = args.output or Path(
        "llm_bayesian_reasoning/results/oracle"
        f"/oracle_{retriever_type.value}_n{args.top_n}_k{args.top_k}.jsonl"
    )
    metrics_output_path = args.metrics_output or output_path.with_name(
        f"{output_path.stem}_metrics.json"
    )

    logger.info(
        "Loading or building %s index (index_path=%s)",
        retriever_type.value,
        args.index_path,
    )
    retriever = build_or_load_retriever(
        documents_path=args.index_documents,
        index_path=args.index_path,
        retriever_type=retriever_type,
        batch_size=args.batch_size,
        limit=args.index_limit,
        retriever_model_name=args.retriever_model_name,
    )

    logger.info("Loading queries + ground truth from %s", args.data_file)
    records = _load_queries_with_ground_truth(args.data_file, limit=args.limit)
    logger.info("Loaded %d records with ground truth", len(records))

    results = compute_oracle_ranking_for_dataset(
        records, retriever=retriever, top_n=args.top_n, top_k=args.top_k
    )

    num_gold_missing_from_pool = sum(
        1 for r in results.values() if r["num_gold_in_pool"] == 0
    )
    metrics = _aggregate_metrics(results)

    _write_results(results, output_path)

    summary = {
        "data_file": str(args.data_file),
        "retriever_type": retriever_type.value,
        "index_path": str(args.index_path),
        "top_n": args.top_n,
        "top_k": args.top_k,
        "num_records_total": len(results),
        "num_records_gold_missing_from_pool": num_gold_missing_from_pool,
        "metrics": metrics,
    }
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "Done. %d records evaluated (%d with gold entirely missing from the top-%d pool).",
        len(results),
        num_gold_missing_from_pool,
        args.top_n,
    )
    logger.info("Oracle results JSONL: %s", output_path)
    logger.info("Aggregated metrics: %s", metrics_output_path)


if __name__ == "__main__":
    main()
