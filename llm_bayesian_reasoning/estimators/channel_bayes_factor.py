from llm_bayesian_reasoning.estimators.atom_estimator import Candidate
from llm_bayesian_reasoning.estimators.language_model import LanguageModel
from llm_bayesian_reasoning.estimators.prompts import fill_entity, question_prompt
from llm_bayesian_reasoning.estimators.scoring import check_temperature, sigmoid
from llm_bayesian_reasoning.records.atoms import Atom

YES_TOKEN = " yes"
NO_TOKEN = " no"
PRIOR_QUESTION = "Is the statement true?"
ARTICLE_HEADER = "Wikipedia article:"


class ChannelBayesFactorEstimator:
    """Add a yes/no prior to how much better the atom than its negation explains
    the document."""

    def __init__(
        self,
        language_model: LanguageModel,
        temperature: float = 1.0,
        max_document_tokens: int = 512,
    ):
        """
        Args:
            language_model (LanguageModel):
                The model that gives log-probabilities and cuts text to tokens.
            temperature (float):
                The divisor of the document evidence, positive and finite.
            max_document_tokens (int):
                How many leading document tokens are scored, at least 1.
        """
        check_temperature(temperature)
        if max_document_tokens < 1:
            raise ValueError(
                f"max_document_tokens must be at least 1, got {max_document_tokens}"
            )
        self.language_model = language_model
        self.temperature = temperature
        self.max_document_tokens = max_document_tokens

    def estimate_atoms(
        self, atoms: list[Atom], candidate: Candidate
    ) -> dict[str, float]:
        """
        Return the probability of each atom for the candidate, keyed by atom name.

        The probability is sigmoid(prior + evidence / temperature), where the prior
        is the yes/no gap without the document and the evidence is the summed
        log-probability of the document lead after the atom fact minus after the
        negated fact.

        Args:
            atoms (list[Atom]):
                The atoms of the query.
            candidate (Candidate):
                The entity and its document.

        Returns:
            dict[str, float]: The probability that each atom holds.
        """
        prior_prompts = [
            question_prompt(
                "", candidate, fill_entity(atom.text, candidate), PRIOR_QUESTION
            )
            for atom in atoms
        ]
        prior_logprobs = self.language_model.next_token_logprobs(
            prior_prompts, [YES_TOKEN, NO_TOKEN]
        )
        priors = [
            token_logprobs[YES_TOKEN] - token_logprobs[NO_TOKEN]
            for token_logprobs in prior_logprobs
        ]

        evidence = [0.0] * len(atoms)
        document = candidate.text.strip()
        if document:
            lead = "\n" + self.language_model.truncate_to_tokens(
                document, self.max_document_tokens
            )
            requests = []
            for atom in atoms:
                for template in (atom.text, atom.negated_text):
                    fact = (
                        f"Fact: {fill_entity(template, candidate)}.\n\n{ARTICLE_HEADER}"
                    )
                    requests.append((fact, lead))
            token_logprobs = self.language_model.continuation_token_logprobs(requests)
            evidence = [
                sum(token_logprobs[2 * index]) - sum(token_logprobs[2 * index + 1])
                for index in range(len(atoms))
            ]

        return {
            atom.name: sigmoid(priors[index] + evidence[index] / self.temperature)
            for index, atom in enumerate(atoms)
        }
