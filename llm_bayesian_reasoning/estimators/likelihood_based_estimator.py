import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_bayesian_reasoning.estimators.base import BaseEstimator
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
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        contrastive_temperature: float = 1.0,
    ):
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        self.contrastive_temperature = contrastive_temperature

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "microsoft/phi-2",
        device: str = "cuda",
        contrastive_temperature: float = 1.0,
        **kwargs,
    ) -> "LikelihoodBasedEstimator":
        """Load a pre-trained model and tokenizer to create a LikelihoodBasedEstimator.

        Args:
            model_name (str): HuggingFace model identifier.
            device (str): Target device (cuda/cpu).
            contrastive_temperature (float): Temperature for contrastive scoring.
            **kwargs: Additional kwargs for from_pretrained (e.g., quantization_config).
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", **kwargs
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            contrastive_temperature=contrastive_temperature,
        )

    def _conditional_loss(self, atom: ProblogAtom, entity: str) -> float:
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
        predicates: list[ProblogAtom] | list[tuple[ProblogAtom, ProblogAtom]],
        entity: str,
    ) -> list[ProblogAtom]:
        scored_predicates = []
        if not predicates:
            return scored_predicates

        if isinstance(predicates[0], tuple):
            for atom, negated_atom in predicates:
                probability = self.prob_contrastive(atom, negated_atom, entity)
                scored_predicates.append(
                    ProblogAtom(
                        atom=atom.atom,
                        probability=probability,
                        context=atom.context,
                    )
                )
        elif isinstance(predicates[0], ProblogAtom):
            for atom in predicates:
                probability = self.perplexity(atom, entity)
                scored_predicates.append(
                    ProblogAtom(
                        atom=atom.atom,
                        probability=probability,
                        context=atom.context,
                    )
                )
        return scored_predicates
