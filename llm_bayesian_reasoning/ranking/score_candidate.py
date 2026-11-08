from llm_bayesian_reasoning.estimators.atom_estimator import AtomEstimator, Candidate
from llm_bayesian_reasoning.logic.formula import evaluate_formula
from llm_bayesian_reasoning.ranking.rerank import ScoredCandidate
from llm_bayesian_reasoning.records.query_record import QueryRecord


def score_candidate(
    record: QueryRecord, candidate: Candidate, estimator: AtomEstimator
) -> ScoredCandidate:
    """
    Return the candidate with its atom scores and the probability of the formula.

    Args:
        record (QueryRecord):
            The query, with its atoms and formula.
        candidate (Candidate):
            The retrieved entity to score.
        estimator (AtomEstimator):
            The estimator that scores all atoms of the candidate in one call.

    Returns:
        ScoredCandidate: The scored candidate.
    """
    atom_probabilities = estimator.estimate_atoms(record.atoms, candidate)
    try:
        atom_names = {atom.name for atom in record.atoms}
        for name in atom_names - atom_probabilities.keys():
            raise ValueError(f"Estimator did not score atom {name!r}")
        for name in atom_probabilities.keys() - atom_names:
            raise ValueError(f"Estimator scored unknown atom {name!r}")
        probability = evaluate_formula(record.formula, atom_probabilities)
    except ValueError as error:
        raise ValueError(f"Entity {candidate.entity!r}: {error}") from error
    return ScoredCandidate(
        entity=candidate.entity,
        probability=probability,
        atom_probabilities=atom_probabilities,
    )
