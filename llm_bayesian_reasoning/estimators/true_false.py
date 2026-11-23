from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.language_model import LanguageModel
from llm_bayesian_reasoning.estimators.prompts import fill_entity
from llm_bayesian_reasoning.estimators.scoring import sigmoid
from llm_bayesian_reasoning.records.atoms import Atom

TRUE_TOKEN = " True"
FALSE_TOKEN = " False"
QUESTION = "Is this phrase true or false?"


def _build_prompt(atom: Atom, candidate: Candidate) -> str:
    """
    Return the document, then the atom statement followed by the question.

    Args:
        atom (Atom):
            The atom to evaluate.
        candidate (Candidate):
            The candidate document and entity.

    Returns:
        str: The prompt combining the document and the atom statement with the question.
    """
    statement = fill_entity(atom.text, candidate).strip()
    if statement.endswith((".", "?", "!")):
        statement = statement[:-1]
    statement = f"{statement}. {QUESTION}"

    document = candidate.text.strip()
    if document:
        return f"{document}\n{statement}"
    return statement


class TrueFalseEstimator:
    """The baseline: compare " True" and " False" after asking if the atom holds."""

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

        The probability is the softmax of " True" over " True" and " False".

        Args:
            atoms (list[Atom]):
                The atoms of the query.
            candidate (Candidate):
                The entity and its document.

        Returns:
            dict[str, float]: The probability that each atom holds.
        """
        prompts = [_build_prompt(atom, candidate) for atom in atoms]
        logprobs = self.language_model.next_token_logprobs(
            prompts, [TRUE_TOKEN, FALSE_TOKEN]
        )
        return {
            atom.name: sigmoid(token_logprobs[TRUE_TOKEN] - token_logprobs[FALSE_TOKEN])
            for atom, token_logprobs in zip(atoms, logprobs)
        }
