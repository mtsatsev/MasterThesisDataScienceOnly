import math

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.language_model import LanguageModel
from llm_bayesian_reasoning.records.atoms import Atom

YES_TOKEN = " yes"
NO_TOKEN = " no"
QUESTION = "Is the statement supported by the context?"


def _sigmoid(logit: float) -> float:
    """Return 1 / (1 + e^-logit) without overflowing for large logits."""
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    return math.exp(logit) / (1 + math.exp(logit))


def build_yes_no_prompt(statement_template: str, candidate: Candidate) -> str:
    """
    Return the context block, entity, statement and question, one per line.

    Args:
        statement_template (str):
            An atom text or negated text, with {X} for the entity.
        candidate (Candidate):
            The entity and its document.

    Returns:
        str: The prompt, ending in "Answer:".
    """
    statement = statement_template.replace("{X}", repr(candidate.entity)).strip()
    lines = []
    document = candidate.text.strip()
    if document:
        lines += ["Context:", document, ""]
    lines += [
        f"Entity: {candidate.entity!r}",
        f"Statement: {statement}",
        "",
        QUESTION,
        "Answer:",
    ]
    return "\n".join(lines)


class YesNoEstimator:
    """Compare " yes" and " no" after asking if the context supports the atom."""

    def __init__(self, language_model: LanguageModel, temperature: float = 1.0):
        """
        Args:
            language_model (LanguageModel):
                The model that gives next-token log-probabilities.
            temperature (float):
                The divisor of the yes/no gap, positive and finite.
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

        The probability is sigmoid((log p(" yes") - log p(" no")) / temperature).

        Args:
            atoms (list[Atom]):
                The atoms of the query.
            candidate (Candidate):
                The entity and its document.

        Returns:
            dict[str, float]: The probability that each atom holds.
        """
        prompts = [build_yes_no_prompt(atom.text, candidate) for atom in atoms]
        logprobs = self.language_model.next_token_logprobs(
            prompts, [YES_TOKEN, NO_TOKEN]
        )
        return {
            atom.name: _sigmoid(
                (token_logprobs[YES_TOKEN] - token_logprobs[NO_TOKEN])
                / self.temperature
            )
            for atom, token_logprobs in zip(atoms, logprobs)
        }
