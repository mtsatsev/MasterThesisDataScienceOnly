from llm_bayesian_reasoning.metrics.ranking_metrics import (
    f1_at_k,
    ndcg_at_k,
    oracle_ranking,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from llm_bayesian_reasoning.pipeline.score_pool import QueryResult
from llm_bayesian_reasoning.records.query_record import QueryRecord

METRICS_AT_K = {
    "P": precision_at_k,
    "R": recall_at_k,
    "F1": f1_at_k,
    "NDCG": ndcg_at_k,
}


def evaluate_results(
    results: list[QueryResult], records: list[QueryRecord], ks: list[int]
) -> dict[str, float]:
    """
    Return each metric of the retrieval, reranked and oracle orders, averaged.

    Args:
        results (list[QueryResult]):
            One result per evaluated query.
        records (list[QueryRecord]):
            The records holding the relevant entities, a superset of the results.
        ks (list[int]):
            The cutoffs, each at least 1.

    Returns:
        dict[str, float]: The mean of each metric over the results, keyed as
            "<ordering>_<metric>@<k>" or "<ordering>_MRR".
    """
    if not ks:
        raise ValueError("At least one cutoff is needed")
    for k in ks:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
    if not results:
        raise ValueError("There are no results to evaluate")

    relevant_entities = {
        record.query_id: record.relevant_entities for record in records
    }
    query_ids = set()
    sums: dict[str, float] = {}
    for result in results:
        if result.query_id in query_ids:
            raise ValueError(f"Results repeat query id {result.query_id}")
        query_ids.add(result.query_id)
        if result.query_id not in relevant_entities:
            raise ValueError(f"No record has query id {result.query_id}")
        relevant = relevant_entities[result.query_id]

        orderings = {
            "retrieval": result.retrieved,
            "reranked": [scored.entity for scored in result.reranked],
            "oracle": oracle_ranking(result.retrieved, relevant),
        }
        for ordering, ranked in orderings.items():
            values = {
                f"{ordering}_{name}@{k}": metric(ranked, relevant, k)
                for name, metric in METRICS_AT_K.items()
                for k in ks
            }
            values[f"{ordering}_MRR"] = reciprocal_rank(ranked, relevant)
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + value

    return {key: total / len(results) for key, total in sums.items()}
