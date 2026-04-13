import logging

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

logger = logging.getLogger(__name__)


class TrueFalseLLMEstimator(BaseEstimator):
    """
    LLM Likelihood Estimator using True/False tokens.

    Args:
        model: Pretrained causal language model
        tokenizer: Corresponding tokenizer
        device: Device to run on (cuda/cpu)
        true_token: Token representing True (default: " True")
        false_token: Token representing False (default: " False")
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
        true_token: str = " True",
        false_token: str = " False",
    ) -> None:
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        self.true_token = true_token
        self.false_token = false_token

    def _build_true_false_prompt(self, predicate: ProblogAtom, entity: str) -> str:
        statement = predicate.to_prompt(entity).strip()
        if statement.endswith((".", "?", "!")):
            statement = statement[:-1]
        statement = statement + ". Is this phrase true or false?"

        if predicate.context:
            return predicate.context.strip() + "\n" + statement

        return statement

    def get_probability_for_prompt(self, prompt: str) -> tuple[float, float]:
        """
        Get the probabilities of True and False tokens for a given prompt.
        This is done by estimating the likelihood of the next token being " True" or " False".
        Then we use the softmax over these two logits to get the probabilities.
        We are particularly interested in the probability of being True. However,
        we return both probabilities for completeness.

        Args:
            prompt (str): The input prompt. This should be a question or statement
                that can be answered with True or False. The question is created using
                the predicate/fact.

        Returns:
            tuple[float, float]: Probabilities of True and False tokens.
        """
        try:
            # NOTE: Don't use .to(self.device) with device_map="auto"
            # Let accelerate handle device placement automatically
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # Move directly to model's device (detected automatically)
            inputs = {
                k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()
            }
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]

            true_token_ids = self.tokenizer.encode(
                self.true_token, add_special_tokens=False
            )
            false_token_ids = self.tokenizer.encode(
                self.false_token, add_special_tokens=False
            )

            if not true_token_ids or not false_token_ids:
                logger.debug(
                    "True/False tokenization produced empty ids for tokens: %r / %r",
                    self.true_token,
                    self.false_token,
                )
                return 0.0, 0.0

            true_token_id = true_token_ids[0]
            false_token_id = false_token_ids[0]

            # Gather logits for the first token of each label (approximation for multi-token labels)
            true_logit = float(logits[true_token_id].item())
            false_logit = float(logits[false_token_id].item())

            probs = torch.nn.functional.softmax(
                torch.tensor([true_logit, false_logit]), dim=0
            )
            t_prob, f_prob = float(probs[0].item()), float(probs[1].item())

            logger.debug(
                "Prompt truncated=%r, true_id=%d, false_id=%d, true_logit=%.4f, false_logit=%.4f, probs=(%.4f, %.4f)",
                (prompt[:200] + "...") if len(prompt) > 200 else prompt,
                true_token_id,
                false_token_id,
                true_logit,
                false_logit,
                t_prob,
                f_prob,
            )

            return t_prob, f_prob
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to compute token probabilities: %s", exc)
            return 0.0, 0.0

    def score_probability(
        self,
        predicates: list[ScoringInput],
        entity: str,
    ) -> list[ProblogAtom]:
        """
        Score a list of predicates and return their probabilities of being True.

        Args:
            predicates (list[ProblogAtom]): List of predicates to score.
            entity (str): The entity to replace the placeholder '{X}'.

        Returns:
            dict[str, float]: Dictionary mapping predicates to their probabilities of being True.
        """
        results = []
        for predicate in predicates:
            if isinstance(predicate, ContrastiveScoringInput):
                raise ValueError(
                    "TrueFalseLLMEstimator does not support contrastive scoring inputs"
                )

            positive_input = predicate
            if isinstance(positive_input, PositiveScoringInput):
                prompt = self._build_true_false_prompt(positive_input.atom, entity)
                t_prob, f_prob = self.get_probability_for_prompt(prompt)
                if t_prob == 0.0 and f_prob == 0.0:
                    logger.debug(
                        "LLM returned zero probs for prompt sample: %r",
                        (prompt[:200] + "...") if len(prompt) > 200 else prompt,
                    )
                results.append(
                    ProblogAtom(
                        atom=positive_input.atom.atom,
                        probability=t_prob,
                        context=positive_input.atom.context,
                    )
                )
        return results
