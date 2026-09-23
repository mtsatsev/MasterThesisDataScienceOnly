import math

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llm_bayesian_reasoning.estimators.base import BaseEstimator
from llm_bayesian_reasoning.problog_models.problog_models import ProblogAtom


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

    def _answer_token_id(self, continuation: str) -> int:
        token_ids = self.tokenizer(continuation, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(
                f"Continuation {continuation!r} must be a single token, got {token_ids}"
            )
        return token_ids[0]

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

        All prompts for one entity share the same context prefix, so that
        prefix is encoded once and its KV cache is reused for every atom. Since
        ``" yes"`` and ``" no"`` are single tokens, one forward pass yields both
        log-probabilities, and ``loss_no - loss_yes`` reduces to
        ``logit_yes - logit_no`` at the last prompt position.
        """
        atoms = [p[0] if isinstance(p, tuple) else p for p in predicates]
        if not atoms:
            return []

        yes_id = self._answer_token_id(self.positive_continuation)
        no_id = self._answer_token_id(self.negative_continuation)
        prompts = [
            self.tokenizer(self._build_answer_prefix(atom, entity))["input_ids"]
            for atom in atoms
        ]
        # Longest shared token prefix, leaving at least one token per atom.
        shared = min(len(ids) for ids in prompts) - 1
        for ids in prompts[1:]:
            shared = next((i for i in range(shared) if ids[i] != prompts[0][i]), shared)

        model_device = next(self.model.parameters()).device
        temperature = max(1e-8, float(self.contrastive_temperature))
        cache = DynamicCache()
        scored_predicates: list[ProblogAtom] = []
        with torch.no_grad():
            if shared > 0:
                self.model(
                    input_ids=torch.tensor([prompts[0][:shared]], device=model_device),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
            for atom, ids in zip(atoms, prompts):
                logits = self.model(
                    input_ids=torch.tensor([ids[shared:]], device=model_device),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                ).logits[0, -1]
                cache.crop(shared)
                delta = float(logits[yes_id] - logits[no_id]) / temperature
                scored_predicates.append(
                    ProblogAtom(
                        atom=atom.atom,
                        probability=1.0 / (1.0 + math.exp(-delta)),
                        context=atom.context,
                    )
                )
        return scored_predicates
