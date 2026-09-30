"""Alternative atom estimators built on differences of LM log-likelihoods.

Every estimator here turns one or more log-probabilities into a logit ``z`` and
returns ``sigmoid(z)`` as the ProbLog fact probability. Raw log-likelihoods mix
truth with fluency (short, common or copied phrasings are "likely"); each
estimator cancels that by subtracting a second log-likelihood computed under
matched conditions, and they differ in *what* is subtracted.

``contrastive_temperature`` from the config is used as a generic temperature
``T`` that divides the logit.
"""

import math
import re

import torch
from transformers import DynamicCache

from llm_bayesian_reasoning.estimators.base import BaseEstimator
from llm_bayesian_reasoning.estimators.likelihood_based_yes_no_estimator import (
    LikelihoodBasedYesNoEstimator,
)
from llm_bayesian_reasoning.problog_models.problog_models import ProblogAtom


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logit(p: float, eps: float = 1e-12) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _with_probability(atom: ProblogAtom, probability: float) -> ProblogAtom:
    return ProblogAtom(atom=atom.atom, probability=probability, context=atom.context)


def _positive_atoms(
    predicates: list[ProblogAtom] | list[tuple[ProblogAtom, ProblogAtom]],
) -> list[ProblogAtom]:
    return [p[0] if isinstance(p, tuple) else p for p in predicates]


def _require_pairs(
    predicates: list[ProblogAtom] | list[tuple[ProblogAtom, ProblogAtom]],
    name: str,
) -> list[tuple[ProblogAtom, ProblogAtom]]:
    if any(not isinstance(p, tuple) for p in predicates):
        raise ValueError(f"{name} needs (atom, negated_atom) pairs")
    return list(predicates)


def _context_block(atom: ProblogAtom) -> str:
    return f"Context:\n{atom.context.strip()}\n\n" if atom.context else ""


class _LogProbEstimator(BaseEstimator):
    """Shared plumbing: a temperature and summed continuation log-probabilities."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        contrastive_temperature: float = 1.0,
    ):
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        self.temperature = max(1e-8, float(contrastive_temperature))

    def _continuation_logprobs(
        self, prefix: str, continuations: list[str]
    ) -> list[float]:
        """Return ``sum_t log p(token_t | prefix, earlier tokens)`` per continuation.

        The prefix is encoded once; each continuation reuses its KV cache.
        """
        model_device = next(self.model.parameters()).device
        prefix_ids = self.tokenizer(prefix, return_tensors="pt")["input_ids"].to(
            model_device
        )
        prefix_len = prefix_ids.shape[1]
        cache = DynamicCache()
        scores: list[float] = []
        with torch.no_grad():
            last = self.model(
                input_ids=prefix_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            ).logits[0, -1]
            first_logprobs = torch.log_softmax(last.float(), dim=-1)
            for text in continuations:
                ids = self.tokenizer(
                    text, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(model_device)
                total = float(first_logprobs[ids[0, 0]])
                if ids.shape[1] > 1:
                    logits = (
                        self.model(
                            input_ids=ids[:, :-1], past_key_values=cache, use_cache=True
                        )
                        .logits[0]
                        .float()
                    )
                    cache.crop(prefix_len)
                    total += float(
                        torch.log_softmax(logits, dim=-1)
                        .gather(1, ids[0, 1:, None])
                        .sum()
                    )
                scores.append(total)
        return scores


class PMIEvidenceGainEstimator(_LogProbEstimator):
    """Evidence gain: how much the document raises the statement's likelihood.

    Paper: Holtzman et al. (2021), "Surface Form Competition: Why the Highest
    Probability Answer Isn't Always Right", EMNLP. They score an answer by
    domain-conditional PMI, ``log p(answer | input) - log p(answer | domain
    premise)``, where the domain premise is the prompt template without the
    specific input. Subtracting it removes how "easy" the answer string is on
    its own, which is exactly what makes raw perplexity useless as a truth score.

    Here the input is the retrieved document and the answer is the atom:

        z = log p(" is a German film" | doc, "Fact: 'X'")
          - log p(" is a German film" |      "Fact: 'X'")

    Only the text after the entity is scored: the document nearly always
    contains the entity name, so scoring the name would reward copying.
    Without retrieved context both terms are identical and p = 0.5.
    """

    def score_probability(self, predicates, entity):
        scored = []
        for atom in _positive_atoms(predicates):
            before, _, after = atom.atom.partition("{X}")
            head = f"Fact: {before}{entity!r}"
            (with_doc,) = self._continuation_logprobs(
                _context_block(atom) + head, [after]
            )
            (without_doc,) = self._continuation_logprobs(head, [after])
            z = (with_doc - without_doc) / self.temperature
            scored.append(_with_probability(atom, _sigmoid(z)))
        return scored


class ChannelBayesFactorEstimator(_LogProbEstimator):
    """Channel scoring: which hypothesis better explains the document (Bayes rule).

    Paper: Min et al. (2022), "Noisy Channel Language Model Prompting for
    Few-Shot Text Classification", ACL. Instead of the direct model
    p(label | input) they score the channel model p(input | label), which by
    Bayes' rule is proportional to the posterior, and found it more robust to
    label imbalance and prompt wording.

    Here the two "labels" are the atom and its negated atom, and the input is
    the document lead (first ``max_context_tokens`` tokens). Their log-ratio
    is a Bayes factor, added to a prior in log-odds space:

        log BF           = log p(doc | "Fact: a") - log p(doc | "Fact: not a")
        prior            = logit_yes - logit_no for "Is the statement true?",
                           asked without the document (parametric knowledge)
        logit P(a | doc) = prior + log BF / T

    ``T`` tempers only the evidence: the sum runs over hundreds of document
    tokens, so the raw factor is usually overconfident. Document tokens that
    are unrelated to the atom get nearly equal probability under both
    hypotheses and cancel in the difference. Needs (atom, negated_atom) pairs.
    """

    max_context_tokens = 512

    def _lead(self, text: str) -> str:
        ids = self.tokenizer(text.strip(), add_special_tokens=False)["input_ids"]
        return self.tokenizer.decode(ids[: self.max_context_tokens])

    def _prior_logit(self, atom: ProblogAtom, entity: str) -> float:
        prompt = (
            f"Entity: {entity!r}\nStatement: {atom.to_prompt(entity)}\n\n"
            "Is the statement true?\nAnswer:"
        )
        yes, no = self._continuation_logprobs(prompt, [" yes", " no"])
        return yes - no

    def score_probability(self, predicates, entity):
        scored = []
        for atom, negated in _require_pairs(predicates, type(self).__name__):
            log_bf = 0.0
            if atom.context:
                doc = "\n" + self._lead(atom.context)
                (pos,) = self._continuation_logprobs(
                    f"Fact: {atom.to_prompt(entity)}.\n\nWikipedia article:", [doc]
                )
                (neg,) = self._continuation_logprobs(
                    f"Fact: {negated.to_prompt(entity)}.\n\nWikipedia article:",
                    [doc],
                )
                log_bf = pos - neg
            z = self._prior_logit(atom, entity) + log_bf / self.temperature
            scored.append(_with_probability(atom, _sigmoid(z)))
        return scored


class ExpectedVerbalizedEstimator(_LogProbEstimator):
    """Verbalized confidence, read out as an expectation over a 0-10 scale.

    Papers: Tian et al. (2023), "Just Ask for Calibration: Strategies for
    Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with
    Human Feedback", EMNLP: RLHF-tuned models (such as Llama-3.1-Instruct)
    state their confidence in words or numbers better calibrated than their
    token probabilities.
    Liu et al. (2023), "G-Eval: NLG Evaluation using GPT-4 with Better Human
    Alignment", EMNLP: instead of taking the single most likely score token,
    use the probability-weighted average over all score tokens.

    One prompt, 11 candidate answers " 0" ... " 10". Their probabilities are
    renormalized over those 11 options and the result is
    ``sum_k (k / 10) * p(k)``. That is already a probability, so no
    temperature is applied.
    """

    def score_probability(self, predicates, entity):
        options = [f" {k}" for k in range(11)]
        scored = []
        for atom in _positive_atoms(predicates):
            prompt = (
                f"{_context_block(atom)}Entity: {entity!r}\n"
                f"Statement: {atom.to_prompt(entity)}\n\n"
                "On a scale from 0 to 10, how likely is it that the statement is "
                "true?\nAnswer:"
            )
            logprobs = torch.tensor(self._continuation_logprobs(prompt, options))
            weights = torch.softmax(logprobs, dim=0).tolist()
            expected = sum(k / 10 * w for k, w in enumerate(weights))
            scored.append(_with_probability(atom, min(max(expected, 0.0), 1.0)))
        return scored


class ThreeWayEstimator(_LogProbEstimator):
    """Supported / contradicted / not mentioned, as a 3-way categorical.

    Background: the three-way label set of natural language inference
    (entailment / contradiction / neutral), Bowman et al. (2015), "A large
    annotated corpus for learning natural language inference", EMNLP, and the
    SUPPORTS / REFUTES / NOT ENOUGH INFO labels of fact verification,
    Thorne et al. (2018), "FEVER: a Large-scale Dataset for Fact Extraction
    and VERification", NAACL.

    The model picks " A" / " B" / " C"; a softmax over the three
    log-probabilities (divided by ``T``) gives p_sup, p_con and p_unk. The
    returned probability is ``p_sup + unknown_weight * p_unk``:

    - ``unknown_weight = 0`` (default): closed world. "Not mentioned" counts as
      false, so ProbLog's NOT turns it into true.
    - ``unknown_weight = 1``: "not mentioned" counts as true, so NOT needs an
      explicit contradiction (open world for negated atoms, but lenient for
      positive ones). Treating the two polarities differently would need the
      formula, which estimators do not see.
    """

    unknown_weight = 0.0

    def score_probability(self, predicates, entity):
        scored = []
        for atom in _positive_atoms(predicates):
            if atom.context:
                question = (
                    "Based on the context, the statement is:\n"
                    "A) supported\nB) contradicted\nC) not mentioned"
                )
            else:
                question = "The statement is:\nA) true\nB) false\nC) unknown"
            prompt = (
                f"{_context_block(atom)}Entity: {entity!r}\n"
                f"Statement: {atom.to_prompt(entity)}\n\n{question}\nAnswer:"
            )
            logprobs = torch.tensor(
                self._continuation_logprobs(prompt, [" A", " B", " C"])
            )
            p_sup, _p_con, p_unk = torch.softmax(
                logprobs / self.temperature, dim=0
            ).tolist()
            probability = min(p_sup + self.unknown_weight * p_unk, 1.0)
            scored.append(_with_probability(atom, probability))
        return scored


class NegationConsistentYesNoEstimator(LikelihoodBasedYesNoEstimator):
    """Yes/no on the atom and on its negation, averaged in logit space.

    Paper: Burns et al. (2023), "Discovering Latent Knowledge in Language
    Models Without Supervision", ICLR. Their Contrast-Consistent Search uses
    the constraint p(a) = 1 - p(not a) to find truth without labels, through a
    probe on hidden states (the probe is not implemented here).

    This applies the same constraint to the output tokens: ask the existing
    yes/no question for ``a`` and for ``not a`` and average the two views,

        z = (z_yes/no(a) - z_yes/no(not a)) / 2

    An "always yes" bias raises both terms equally and cancels. Needs
    (atom, negated_atom) pairs; reuses the parent's prompt and scoring as is.
    """

    def score_probability(self, predicates, entity):
        pairs = _require_pairs(predicates, type(self).__name__)
        flat = [a for pair in pairs for a in pair]
        probs = [a.probability for a in super().score_probability(flat, entity)]
        return [
            _with_probability(atom, _sigmoid((_logit(p_pos) - _logit(p_neg)) / 2))
            for (atom, _), p_pos, p_neg in zip(pairs, probs[0::2], probs[1::2])
        ]


class EntityLikelihoodPMIEstimator(_LogProbEstimator):
    """Parametric membership: how much the property raises the entity name's likelihood.

    Same domain-conditional PMI as Holtzman et al. (2021, EMNLP; see
    ``PMIEvidenceGainEstimator``), in the reverse direction: the model
    *generates the entity* given the property, and the same template with a
    vacuous property ("it exists") is the domain premise:

        z = log p(" Less Than Zero" | "... true: it is a film.")
          - log p(" Less Than Zero" | "... true: it exists.")

    Subtracting the baseline removes popularity (famous titles are likely
    after any prompt). Uses only the model's memory; retrieved context is
    ignored. The Wikipedia disambiguation suffix, e.g. "(film)", is stripped
    because it would give the category away.
    """

    template = "Name something for which the following is true: {}.\nAnswer:"

    def score_probability(self, predicates, entity):
        name = " " + re.sub(r"\s*\([^)]*\)$", "", entity)
        (baseline,) = self._continuation_logprobs(
            self.template.format("it exists"), [name]
        )
        scored = []
        for atom in _positive_atoms(predicates):
            statement = atom.atom.replace("{X}", "it")
            (specific,) = self._continuation_logprobs(
                self.template.format(statement), [name]
            )
            z = (specific - baseline) / self.temperature
            scored.append(_with_probability(atom, _sigmoid(z)))
        return scored
