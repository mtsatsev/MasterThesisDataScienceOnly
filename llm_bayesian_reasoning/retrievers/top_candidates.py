import numpy as np

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate


def check_top_n(top_n: int) -> None:
    """Reject a pool size below 1, before any scoring work is done."""
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}")


def top_candidates(
    scores: np.ndarray, titles: list[str], texts: list[str], top_n: int
) -> list[Candidate]:
    """
    Return the top_n documents by score, best first, ties in index order.

    Args:
        scores (np.ndarray):
            The score of every indexed document.
        titles (list[str]):
            The entity name of every indexed document.
        texts (list[str]):
            The text of every indexed document.
        top_n (int):
            The number of documents to return.

    Returns:
        list[Candidate]: The documents, with their score as retrieval score.
    """
    best = np.argsort(-scores, kind="stable")[:top_n]
    return [
        Candidate(
            entity=titles[index],
            text=texts[index],
            retrieval_score=float(scores[index]),
        )
        for index in best
    ]
