import math

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.language_model import LanguageModel
from llm_bayesian_reasoning.records.atoms import Atom

LABEL_TOKENS = [" A", " B", " C"]
QUESTION_WITH_CONTEXT = (
    "Based on the context, the statement is:\n"
    "A) supported\nB) contradicted\nC) not mentioned"
)
QUESTION_WITHOUT_CONTEXT = "The statement is:\nA) true\nB) false\nC) unknown"


def _softmax(values: list[float]) -> list[float]:
    """Return the softmax of the values, shifted by their maximum to stay finite."""
    largest = max(values)
    exponentials = [math.exp(value - largest) for value in values]
    total = sum(exponentials)
    return [exponential / total for exponential in exponentials]


def _build_prompt(atom: Atom, candidate: Candidate) -> str:
    """Return the context block, entity, statement and the three-way question."""
    document = candidate.text.strip()
    statement = atom.text.replace("{X}", repr(candidate.entity))
    if document:
        context = f"Context:\n{document}\n\n"
        question = QUESTION_WITH_CONTEXT
    else:
        context = ""
        question = QUESTION_WITHOUT_CONTEXT
    return (
        f"{context}Entity: {candidate.entity!r}\n"
        f"Statement: {statement}\n\n{question}\nAnswer:"
    )


class ThreeWayEstimator:
    """Ask if the statement is supported, contradicted or not mentioned (A/B/C)."""

    def __init__(
        self,
        language_model: LanguageModel,
        temperature: float = 1.0,
        unknown_weight: float = 0.0,
    ):
        """
        Args:
            language_model (LanguageModel):
                The model that gives next-token log-probabilities.
            temperature (float):
                The divisor of the three log-probabilities, positive and finite.
            unknown_weight (float):
                The share of p(not mentioned) counted as true, in [0, 1].
        """
        if not (math.isfinite(temperature) and temperature > 0):
            raise ValueError(
                f"temperature must be positive and finite, got {temperature}"
            )
        if not 0.0 <= unknown_weight <= 1.0:
            raise ValueError(f"unknown_weight must be in [0, 1], got {unknown_weight}")
        self.language_model = language_model
        self.temperature = temperature
        self.unknown_weight = unknown_weight

    def estimate_atoms(
        self, atoms: list[Atom], candidate: Candidate
    ) -> dict[str, float]:
        """
        Return the probability of each atom for the candidate, keyed by atom name.

        The probability is p(supported) + unknown_weight * p(not mentioned), from a
        softmax over the A/B/C log-probabilities divided by the temperature.

        Args:
            atoms (list[Atom]):
                The atoms of the query.
            candidate (Candidate):
                The entity and its document.

        Returns:
            dict[str, float]: The probability that each atom holds.
        """
        prompts = [_build_prompt(atom, candidate) for atom in atoms]
        logprobs = self.language_model.next_token_logprobs(prompts, LABEL_TOKENS)
        probabilities = {}
        for atom, token_logprobs in zip(atoms, logprobs):
            supported, _, unknown = _softmax(
                [token_logprobs[token] / self.temperature for token in LABEL_TOKENS]
            )
            probabilities[atom.name] = min(
                supported + self.unknown_weight * unknown, 1.0
            )
        return probabilities
