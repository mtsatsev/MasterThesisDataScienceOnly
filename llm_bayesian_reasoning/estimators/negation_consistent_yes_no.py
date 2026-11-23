import math

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.language_model import LanguageModel
from llm_bayesian_reasoning.estimators.yes_no import (
    NO_TOKEN,
    YES_TOKEN,
    build_yes_no_prompt,
)
from llm_bayesian_reasoning.records.atoms import Atom


def _sigmoid(logit: float) -> float:
    """Return 1 / (1 + e^-logit) without overflowing for large logits."""
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    return math.exp(logit) / (1 + math.exp(logit))


class NegationConsistentYesNoEstimator:
    """Ask the yes/no question for the atom and its negation, averaged as logits."""

    def __init__(self, language_model: LanguageModel, temperature: float = 1.0):
        """
        Args:
            language_model (LanguageModel):
                The model that gives next-token log-probabilities.
            temperature (float):
                The divisor of each yes/no gap, positive and finite.
        """
        if not (math.isfinite(temperature) and temperature > 0):
            raise ValueError(
                f"temperature must be positive and finite, got {temperature}"
            )
        self.language_model = language_model
        self.temperature = temperature

    def estimate_atoms(
        self, atoms: list[Atom], candidate: Candidate
    ) -> dict[str, float]:
        """
        Return the probability of each atom for the candidate, keyed by atom name.

        With gap = log p(" yes") - log p(" no"), the probability is
        sigmoid((gap(atom) - gap(negation)) / (2 * temperature)).

        Args:
            atoms (list[Atom]):
                The atoms of the query.
            candidate (Candidate):
                The entity and its document.

        Returns:
            dict[str, float]: The probability that each atom holds.
        """
        prompts = []
        for atom in atoms:
            prompts.append(build_yes_no_prompt(atom.text, candidate))
            prompts.append(build_yes_no_prompt(atom.negated_text, candidate))

        logprobs = self.language_model.next_token_logprobs(
            prompts, [YES_TOKEN, NO_TOKEN]
        )
        gaps = [
            token_logprobs[YES_TOKEN] - token_logprobs[NO_TOKEN]
            for token_logprobs in logprobs
        ]
        return {
            atom.name: _sigmoid(
                (gaps[2 * index] - gaps[2 * index + 1]) / (2 * self.temperature)
            )
            for index, atom in enumerate(atoms)
        }
