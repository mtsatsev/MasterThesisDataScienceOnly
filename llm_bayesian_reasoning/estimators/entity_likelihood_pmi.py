import re

from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.language_model import LanguageModel
from llm_bayesian_reasoning.estimators.scoring import check_temperature, sigmoid
from llm_bayesian_reasoning.records.atoms import Atom

PROMPT_TEMPLATE = "Name something for which the following is true: {}.\nAnswer:"
BASELINE_PROPERTY = "it exists"
DISAMBIGUATION_SUFFIX = re.compile(r"\s*\([^)]*\)$")


class EntityLikelihoodPMIEstimator:
    """Score how much the atom property raises the likelihood of the entity name,
    from the model's memory only."""

    def __init__(self, language_model: LanguageModel, temperature: float = 1.0):
        """
        Args:
            language_model (LanguageModel):
                The model that gives continuation log-probabilities.
            temperature (float):
                The divisor of the likelihood gain, positive and finite.
        """
        check_temperature(temperature)
        self.language_model = language_model
        self.temperature = temperature

    def estimate_atoms(
        self, atoms: list[Atom], candidate: Candidate
    ) -> dict[str, float]:
        """
        Return the probability of each atom for the candidate, keyed by atom name.

        The entity name, without a trailing disambiguation suffix, is scored after
        "Name something for which the following is true: <property>." for each
        atom and for the vacuous property "it exists"; the probability is
        sigmoid((sum specific - sum baseline) / temperature). The document is
        not used.

        Args:
            atoms (list[Atom]):
                The atoms of the query.
            candidate (Candidate):
                The entity; its document is ignored.

        Returns:
            dict[str, float]: The probability that each atom holds.
        """
        name = " " + DISAMBIGUATION_SUFFIX.sub("", candidate.entity)
        properties = [BASELINE_PROPERTY] + [
            atom.text.replace("{X}", "it") for atom in atoms
        ]
        token_logprobs = self.language_model.continuation_token_logprobs(
            [(PROMPT_TEMPLATE.format(text), name) for text in properties]
        )
        baseline = sum(token_logprobs[0])
        return {
            atom.name: sigmoid(
                (sum(token_logprobs[index + 1]) - baseline) / self.temperature
            )
            for index, atom in enumerate(atoms)
        }
