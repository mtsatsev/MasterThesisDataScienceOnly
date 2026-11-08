import math


def _check_inputs(relevant: list[str], k: int = 1) -> None:
    """Reject a cutoff below 1 and an empty relevant list."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not relevant:
        raise ValueError("There must be at least one relevant entity")


def _hits_at_k(ranked: list[str], relevant: list[str], k: int) -> int:
    """Return how many of the first k ranked entities are relevant."""
    relevant_set = set(relevant)
    return sum(1 for entity in ranked[:k] if entity in relevant_set)


def precision_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    """
    Return the share of the first k positions that hold a relevant entity.

    Args:
        ranked (list[str]):
            The entities, best first.
        relevant (list[str]):
            The relevant entities of the query.
        k (int):
            The cutoff, at least 1.

    Returns:
        float: The number of hits divided by k.
    """
    _check_inputs(relevant, k)
    return _hits_at_k(ranked, relevant, k) / k


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    """
    Return the share of relevant entities found in the first k positions.

    Args:
        ranked (list[str]):
            The entities, best first.
        relevant (list[str]):
            The relevant entities of the query, retrieved or not.
        k (int):
            The cutoff, at least 1.

    Returns:
        float: The number of hits divided by the number of relevant entities.
    """
    _check_inputs(relevant, k)
    return _hits_at_k(ranked, relevant, k) / len(set(relevant))


def f1_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    """
    Return the harmonic mean of precision and recall at k.

    Args:
        ranked (list[str]):
            The entities, best first.
        relevant (list[str]):
            The relevant entities of the query.
        k (int):
            The cutoff, at least 1.

    Returns:
        float: The F1 score, or 0 when there is no hit.
    """
    precision = precision_at_k(ranked, relevant, k)
    recall = recall_at_k(ranked, relevant, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ndcg_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    """
    Return the discounted gain of the first k positions over the best possible.

    Args:
        ranked (list[str]):
            The entities, best first.
        relevant (list[str]):
            The relevant entities of the query.
        k (int):
            The cutoff, at least 1.

    Returns:
        float: The normalised discounted cumulative gain, with binary relevance.
    """
    _check_inputs(relevant, k)
    relevant_set = set(relevant)
    dcg = sum(
        1 / math.log2(position + 2)
        for position, entity in enumerate(ranked[:k])
        if entity in relevant_set
    )
    ideal_dcg = sum(
        1 / math.log2(position + 2) for position in range(min(len(relevant_set), k))
    )
    return dcg / ideal_dcg


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    """
    Return one over the rank of the first relevant entity, over the whole ranking.

    Args:
        ranked (list[str]):
            The entities, best first.
        relevant (list[str]):
            The relevant entities of the query.

    Returns:
        float: The reciprocal rank, or 0 when no relevant entity is ranked.
    """
    _check_inputs(relevant)
    relevant_set = set(relevant)
    for position, entity in enumerate(ranked, start=1):
        if entity in relevant_set:
            return 1 / position
    return 0.0


def oracle_ranking(pool: list[str], relevant: list[str]) -> list[str]:
    """
    Return the best possible order of a pool: relevant entities first.

    Both groups keep their pool order, and relevant entities outside the pool are
    not added.

    Args:
        pool (list[str]):
            The retrieved entities, in retrieval order.
        relevant (list[str]):
            The relevant entities of the query.

    Returns:
        list[str]: A new list with the same entities, relevant ones first.
    """
    _check_inputs(relevant)
    relevant_set = set(relevant)
    return [entity for entity in pool if entity in relevant_set] + [
        entity for entity in pool if entity not in relevant_set
    ]
