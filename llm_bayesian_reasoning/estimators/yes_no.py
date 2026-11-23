from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.language_model import LanguageModel
from llm_bayesian_reasoning.estimators.prompts import (
    context_block,
    fill_entity,
    question_prompt,
)
from llm_bayesian_reasoning.estimators.scoring import check_temperature, sigmoid
from llm_bayesian_reasoning.records.atoms import Atom

YES_TOKEN = " yes"
NO_TOKEN = " no"
QUESTION = "Is the statement supported by the context?"


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
    statement = fill_entity(statement_template, candidate).strip()
    return question_prompt(context_block(candidate), candidate, statement, QUESTION)


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
        check_temperature(temperature)
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
            atom.name: sigmoid(
                (token_logprobs[YES_TOKEN] - token_logprobs[NO_TOKEN])
                / self.temperature
            )
            for atom, token_logprobs in zip(atoms, logprobs)
        }
