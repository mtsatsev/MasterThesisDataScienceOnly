from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.language_model import LanguageModel
from llm_bayesian_reasoning.estimators.prompts import context_block
from llm_bayesian_reasoning.estimators.scoring import check_temperature, sigmoid
from llm_bayesian_reasoning.records.atoms import Atom


class PMIEvidenceGainEstimator:
    """Score how much the document raises the likelihood of the atom statement."""

    def __init__(self, language_model: LanguageModel, temperature: float = 1.0):
        """
        Args:
            language_model (LanguageModel):
                The model that gives continuation log-probabilities.
            temperature (float):
                The divisor of the evidence gain, positive and finite.
        """
        check_temperature(temperature)
        self.language_model = language_model
        self.temperature = temperature

    def estimate_atoms(
        self, atoms: list[Atom], candidate: Candidate
    ) -> dict[str, float]:
        """
        Return the probability of each atom for the candidate, keyed by atom name.

        Only the statement text after the entity is scored, once after
        "Fact: <entity>" with the document in front and once without; the
        probability is sigmoid((sum with - sum without) / temperature).

        Args:
            atoms (list[Atom]):
                The atoms of the query.
            candidate (Candidate):
                The entity and its document.

        Returns:
            dict[str, float]: The probability that each atom holds.
        """
        context = context_block(candidate)
        requests = []
        for atom in atoms:
            before, _, after = atom.text.partition("{X}")
            head = f"Fact: {before}{candidate.entity!r}"
            requests.append((context + head, after))
            requests.append((head, after))

        token_logprobs = self.language_model.continuation_token_logprobs(requests)
        return {
            atom.name: sigmoid(
                (sum(token_logprobs[2 * index]) - sum(token_logprobs[2 * index + 1]))
                / self.temperature
            )
            for index, atom in enumerate(atoms)
        }
