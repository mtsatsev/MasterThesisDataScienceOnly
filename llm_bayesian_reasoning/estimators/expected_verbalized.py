import math

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.language_model import LanguageModel
from llm_bayesian_reasoning.records.atoms import Atom

SCORE_TOKENS = [f" {score}" for score in range(11)]
QUESTION = "On a scale from 0 to 10, how likely is it that the statement is true?"


def _softmax(values: list[float]) -> list[float]:
    """Return the softmax of the values, shifted by their maximum to stay finite."""
    largest = max(values)
    exponentials = [math.exp(value - largest) for value in values]
    total = sum(exponentials)
    return [exponential / total for exponential in exponentials]


def _build_prompt(atom: Atom, candidate: Candidate) -> str:
    """Return the context block, entity, statement and the 0-10 question."""
    document = candidate.text.strip()
    context = f"Context:\n{document}\n\n" if document else ""
    statement = atom.text.replace("{X}", repr(candidate.entity))
    return (
        f"{context}Entity: {candidate.entity!r}\n"
        f"Statement: {statement}\n\n{QUESTION}\nAnswer:"
    )


class ExpectedVerbalizedEstimator:
    """Ask for a 0-10 confidence and read it out as an expected value."""

    def __init__(self, language_model: LanguageModel):
        """
        Args:
            language_model (LanguageModel):
                The model that gives next-token log-probabilities.
        """
        self.language_model = language_model

    def estimate_atoms(
        self, atoms: list[Atom], candidate: Candidate
    ) -> dict[str, float]:
        """
        Return the probability of each atom for the candidate, keyed by atom name.

        The probability is sum(k / 10 * p(k)) over the scores 0 to 10, with p
        renormalized over those eleven answers.

        Args:
            atoms (list[Atom]):
                The atoms of the query.
            candidate (Candidate):
                The entity and its document.

        Returns:
            dict[str, float]: The probability that each atom holds.
        """
        prompts = [_build_prompt(atom, candidate) for atom in atoms]
        logprobs = self.language_model.next_token_logprobs(prompts, SCORE_TOKENS)
        probabilities = {}
        for atom, token_logprobs in zip(atoms, logprobs):
            weights = _softmax([token_logprobs[token] for token in SCORE_TOKENS])
            probabilities[atom.name] = sum(
                score / 10 * weight for score, weight in enumerate(weights)
            )
        return probabilities
