"""Retrieval evaluation metrics: P@K, R@K, F1@K, NDCG@K, MRR."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    ranked_k = ranked[:k]
    if not ranked_k:
        return 0.0
    return sum(1 for e in ranked_k if e in relevant) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    ranked_k = ranked[:k]
    return sum(1 for e in ranked_k if e in relevant) / len(relevant)


def f1_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    p = precision_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    ranked_k = ranked[:k]
    dcg = sum(
        (1.0 if e in relevant else 0.0) / math.log2(i + 2)
        for i, e in enumerate(ranked_k)
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, e in enumerate(ranked):
        if e in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_record_metrics(
    ranked: list[str],
    relevant: set[str],
    top_k: int,
) -> dict[str, float]:
    """Compute retrieval metrics for a single record.

    Args:
        ranked: Ordered list of predicted entity names.
        relevant: Set of ground-truth entity names.
        top_k: The K used for the final reranked output.

    Returns:
        Dict of metric_name -> float for this record.
    """
    return {
        "P@1": precision_at_k(ranked, relevant, 1),
        f"P@{top_k}": precision_at_k(ranked, relevant, top_k),
        "R@1": recall_at_k(ranked, relevant, 1),
        f"R@{top_k}": recall_at_k(ranked, relevant, top_k),
        "F1@1": f1_at_k(ranked, relevant, 1),
        f"F1@{top_k}": f1_at_k(ranked, relevant, top_k),
        "NDCG@1": ndcg_at_k(ranked, relevant, 1),
        f"NDCG@{top_k}": ndcg_at_k(ranked, relevant, top_k),
        "MRR": reciprocal_rank(ranked, relevant),
    }


def compute_metrics(
    results: dict,
    ground_truth: dict[int | str, list[str]],
    top_k: int,
) -> dict[str, float]:
    """Compute averaged retrieval metrics over all evaluated records.

    Args:
        results: Pipeline output mapping record_id -> {"ranked_entities": [...], ...}.
        ground_truth: Mapping record_id -> list of relevant entity names.
        top_k: The K used for the final reranked output.

    Returns:
        Dict of metric_name -> averaged value.
    """
    metric_sums: dict[str, float] = {}

    n = 0
    total_atoms = 0
    total_entity_pairs = 0

    for record_id, record_result in results.items():
        if record_id not in ground_truth:
            logger.debug("No ground truth for record %s — skipping metrics", record_id)
            continue

        relevant = set(ground_truth[record_id])
        ranked = record_result["ranked_entities"]

        record_m = compute_record_metrics(ranked, relevant, top_k)
        for k, v in record_m.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + v

        num_atoms = record_result.get("num_atoms", 0)
        num_entities = len(record_result.get("scores", {}))
        total_atoms += num_atoms * num_entities
        total_entity_pairs += num_entities

        n += 1

    if n == 0:
        logger.warning("No records had matching ground truth — all metrics are 0")
        return {"num_evaluated": 0}

    averaged = {k: v / n for k, v in metric_sums.items()}
    averaged["num_evaluated"] = float(n)
    averaged["avg_atoms_per_query_entity"] = (
        total_atoms / total_entity_pairs if total_entity_pairs else 0.0
    )
    averaged["total_llm_forward_passes"] = float(total_atoms)
    return averaged
