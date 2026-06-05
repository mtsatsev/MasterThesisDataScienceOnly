import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llm_bayesian_reasoning.estimators.base import BaseEstimator
from llm_bayesian_reasoning.problog_models.problog_models import ProblogAtom

LABEL_IGNORE = -100


class LikelihoodBasedYesNoEstimator(BaseEstimator):
    """Contrastive likelihood estimator that compares ` yes` vs ` no` continuations.

    This estimator keeps the same core idea as the statement-level contrastive
    likelihood approach: hold the prefix fixed, score two competing
    continuations, and convert the loss difference into a probability-like
    value.

    The difference is the prompt framing. Instead of comparing the likelihood of
    a positive statement against the likelihood of a negated statement, this
    class builds a QA-style prompt and compares the answer continuations
    ``" yes"`` and ``" no"``.

    The prompt structure is:

    ``Context:``
    ``<optional context>``
    ``Entity: '<entity>'``
    ``Statement: <atom statement>``
    ``Is the statement supported by the context?``
    ``Answer:``

    Mini example:

    If the atom is ``{X} is a science fiction film`` and the entity is
    ``Blade Runner``, then the estimator may build a prefix like:

    ``Context:``
    ``Blade Runner is a 1982 science fiction film directed by Ridley Scott.``
    ````
    ``Entity: 'Blade Runner'``
    ``Statement: 'Blade Runner' is a science fiction film``
    ````
    ``Is the statement supported by the context?``
    ``Answer:``

    It then scores the two candidate continuations:

    - ``" yes"``
    - ``" no"``

    and computes:

    ``sigmoid((loss_no - loss_yes) / temperature)``

    Lower loss for ``" yes"`` produces a probability above ``0.5``.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
        contrastive_temperature: float = 1.0,
        positive_continuation: str = " yes",
        negative_continuation: str = " no",
    ):
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        self.contrastive_temperature = contrastive_temperature
        self.positive_continuation = positive_continuation
        self.negative_continuation = negative_continuation

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "microsoft/phi-2",
        device: str = "cuda",
        contrastive_temperature: float = 1.0,
        positive_continuation: str = " yes",
        negative_continuation: str = " no",
        **kwargs,
    ) -> "LikelihoodBasedYesNoEstimator":
        """Load a pretrained causal LM and tokenizer for yes/no contrastive scoring.

        Args:
            model_name: Hugging Face model identifier.
            device: Requested runtime device.
            contrastive_temperature: Temperature used in the contrastive sigmoid.
            positive_continuation: Continuation treated as the positive answer.
            negative_continuation: Continuation treated as the negative answer.
            **kwargs: Additional kwargs passed to ``from_pretrained``.

        Returns:
            An initialized ``LikelihoodBasedYesNoEstimator``.

        Mini example:

        ``LikelihoodBasedYesNoEstimator.from_pretrained("microsoft/phi-2")``
        loads the same model family used by the other LM estimators, but with a
        yes/no answer-comparison scoring rule.
        """
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", **kwargs
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            contrastive_temperature=contrastive_temperature,
            positive_continuation=positive_continuation,
            negative_continuation=negative_continuation,
        )

    def _build_answer_prefix(self, atom: ProblogAtom, entity: str) -> str:
        """Build the fixed prefix used before scoring ``yes`` and ``no``.

        Args:
            atom: Predicate to be evaluated.
            entity: Concrete entity substituted into the atom text.

        Returns:
            A QA-style prefix ending in ``Answer:``.

        Mini example:

        With context:

        ``Context:``
        ``Blade Runner is a 1982 science fiction film directed by Ridley Scott.``
        ````
        ``Entity: 'Blade Runner'``
        ``Statement: 'Blade Runner' is a science film``
        ````
        ``Is the statement supported by the context?``
        ``Answer:``

        This prefix is held fixed while the estimator scores the continuations
        ``" yes"`` and ``" no"``.
        """
        context = atom.context.strip() if atom.context else ""
        statement = atom.to_prompt(entity).strip()

        prefix_parts: list[str] = []
        if context:
            prefix_parts.extend(["Context:", context, ""])
        prefix_parts.extend(
            [
                f"Entity: {entity!r}",
                f"Statement: {statement}",
                "",
                "Is the statement supported by the context?",
                "Answer:",
            ]
        )
        return "\n".join(prefix_parts)

    def _conditional_loss_for_continuation(
        self,
        prefix: str,
        continuation: str,
    ) -> float:
        """Score a chosen continuation under a fixed prefix with teacher forcing.

        Args:
            prefix: Fixed prompt prefix.
            continuation: Candidate continuation appended to the prefix.

        Returns:
            Mean language-model loss over the continuation tokens only.

        This method does not use free generation. Instead, it appends the
        continuation to the prefix, masks the prefix tokens with ``LABEL_IGNORE``,
        and asks the model how likely the continuation tokens are.

        Mini example:

        If ``prefix`` ends with ``Answer:``, calling:

        - ``_conditional_loss_for_continuation(prefix, " yes")``
        - ``_conditional_loss_for_continuation(prefix, " no")``

        measures which answer continuation is more likely after the same prompt.
        """
        prefix_inputs = self.tokenizer(prefix, return_tensors="pt")
        continuation_inputs = self.tokenizer(
            continuation,
            return_tensors="pt",
            add_special_tokens=False,
        )

        model_device = next(self.model.parameters()).device
        input_ids = torch.cat(
            [prefix_inputs["input_ids"], continuation_inputs["input_ids"]], dim=1
        ).to(model_device)
        attention_mask = torch.cat(
            [
                prefix_inputs["attention_mask"],
                continuation_inputs["attention_mask"],
            ],
            dim=1,
        ).to(model_device)
        labels = input_ids.clone()
        labels[:, : prefix_inputs["input_ids"].size(1)] = LABEL_IGNORE

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        return float(outputs.loss.item())

    def prob_yes_no(self, atom: ProblogAtom, entity: str) -> float:
        """Compute the contrastive yes/no probability for a predicate.

        Args:
            atom: Predicate to score.
            entity: Concrete entity substituted into the atom text.

        Returns:
            A probability-like score for the positive answer continuation.

        The computation is:

        - build the shared prefix
        - compute ``loss_yes`` for ``self.positive_continuation``
        - compute ``loss_no`` for ``self.negative_continuation``
        - return ``sigmoid((loss_no - loss_yes) / temperature)``

        Mini example:

        If ``loss_yes = 8.4`` and ``loss_no = 12.6``, then the returned
        probability is high, because ``yes`` is the more likely continuation.
        """
        prefix = self._build_answer_prefix(atom, entity)
        yes_loss = self._conditional_loss_for_continuation(
            prefix,
            self.positive_continuation,
        )
        no_loss = self._conditional_loss_for_continuation(
            prefix,
            self.negative_continuation,
        )
        delta = (no_loss - yes_loss) / max(
            1e-8,
            float(self.contrastive_temperature),
        )
        return float(torch.sigmoid(torch.tensor(delta)).item())

    def score_probability(
        self,
        predicates: list[ProblogAtom] | list[tuple[ProblogAtom, ProblogAtom]],
        entity: str,
    ) -> list[ProblogAtom]:
        """Score predicates and return ``ProblogAtom`` objects with probabilities.

        Args:
            predicates: Either plain atoms or tuples. If tuples are provided,
                only the first atom is used because this estimator constructs its
                own yes/no contrast inside the prompt rather than relying on an
                explicit negated atom.
            entity: Concrete entity substituted into the atom text.

        Returns:
            A list of ``ProblogAtom`` objects with the estimated probability in
            the ``probability`` field.

        Mini example:

        Given one atom such as ``{X} is a science fiction film``, this method
        returns a new ``ProblogAtom`` with the same ``atom`` text and a
        probability derived from the contrast between ``" yes"`` and ``" no"``.
        """
        scored_predicates: list[ProblogAtom] = []
        if not predicates:
            return scored_predicates

        for predicate in predicates:
            atom = predicate[0] if isinstance(predicate, tuple) else predicate
            probability = self.prob_yes_no(atom, entity)
            scored_predicates.append(
                ProblogAtom(
                    atom=atom.atom,
                    probability=probability,
                    context=atom.context,
                )
            )
        return scored_predicates
