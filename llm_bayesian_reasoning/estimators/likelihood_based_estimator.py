import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llm_bayesian_reasoning.estimators.base import BaseEstimator
from llm_bayesian_reasoning.estimators.scoring_inputs import (
    ContrastiveScoringInput,
    PositiveScoringInput,
    ScoringInput,
)
from llm_bayesian_reasoning.problog_models.problog_models import ProblogAtom

LABEL_IGNORE = -100


class LikelihoodBasedEstimator(BaseEstimator):
    """
    LLM Likelihood Estimator using Contrastive Tokens.

    Args:
        model_name (str): Name of the pre-trained model to use.
        device (str): Device to run the model on.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
        contrastive_temperature: float = 1.0,
    ):
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        self.contrastive_temperature = contrastive_temperature

    def _conditional_loss(self, atom: ProblogAtom, entity: str) -> float:
        """
        Compute the conditional loss of an atom being true for a given entity.

        Args:
            atom (ProblogAtom): The atomic statement.
            entity (str): The entity to evaluate the atom against.

        Returns:
            float: The conditional loss of the atom being true for the entity.
        """
        ctx = self.tokenizer(atom.to_prompt_with_context(entity), return_tensors="pt")
        hypothesis = self.tokenizer(
            atom.to_prompt(entity).strip(),
            return_tensors="pt",
            add_special_tokens=False,
        )
        model_device = next(self.model.parameters()).device
        input_ids = torch.cat([ctx["input_ids"], hypothesis["input_ids"]], dim=1).to(
            model_device
        )
        labels = input_ids.clone()
        labels[:, : ctx["input_ids"].size(1)] = LABEL_IGNORE
        with torch.no_grad():
            loss = self.model(input_ids=input_ids, labels=labels).loss.item()
        return loss

    def prob_contrastive(
        self, atom: ProblogAtom, negated_atom: ProblogAtom, entity: str
    ) -> float:
        """
        Compute the contrastive probability of an atom being true for a given entity.

        Args:
            atom (ProblogAtom): The atomic statement.
            entity (str): The entity to evaluate the atom against.

        Returns:
            float: The contrastive probability of the atom being true for the entity.
        """
        pos_loss = self._conditional_loss(atom, entity)
        neg_loss = self._conditional_loss(negated_atom, entity)
        delta = (neg_loss - pos_loss) / max(1e-8, float(self.contrastive_temperature))
        return torch.sigmoid(torch.tensor(delta)).item()

    def perplexity(self, atom: ProblogAtom, entity: str) -> float:
        """
        Compute the perplexity of an atom being true for a given entity.

        Args:
            atom (ProblogAtom): The atomic statement.
            entity (str): The entity to evaluate the atom against.

        Returns:
            float: The perplexity of the atom being true for the entity.
        """
        pos_loss = self._conditional_loss(atom, entity)
        return torch.exp(-torch.tensor(pos_loss)).item()

    def score_probability(
        self,
        predicates: list[ScoringInput],
        entity: str,
    ) -> list[ProblogAtom]:
        """
        Score a list of predicates and return their probabilities of being True.

        Args:
            predicates (list[ScoringInput]): List of scoring inputs to score.
            entity (str): The entity to replace the placeholder '{X}'.

        Returns:
            list[ProblogAtom]: List of scored predicates with their probabilities.
        """
        scored_predicates = []
        if not predicates:
            return scored_predicates

        for inp in predicates:
            if isinstance(inp, ContrastiveScoringInput):
                probability = self.prob_contrastive(
                    inp.atom,
                    inp.negated_atom,
                    entity,
                )
                scored_predicates.append(
                    ProblogAtom(
                        atom=inp.atom.atom,
                        probability=probability,
                        context=inp.atom.context,
                    )
                )
            elif isinstance(inp, PositiveScoringInput):
                probability = self.perplexity(inp.atom, entity)
                scored_predicates.append(
                    ProblogAtom(
                        atom=inp.atom.atom,
                        probability=probability,
                        context=inp.atom.context,
                    )
                )
        return scored_predicates
