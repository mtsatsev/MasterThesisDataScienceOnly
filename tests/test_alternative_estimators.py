"""Tests for the alternative estimators on a tiny random GPT-2 (CPU, no real model)."""

import math
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from llm_bayesian_reasoning.estimators import alternative_estimators as alt
from llm_bayesian_reasoning.problog_models.problog_models import ProblogAtom


@pytest.fixture(scope="module")
def lm():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=tokenizer.vocab_size)
    return GPT2LMHeadModel(config).eval(), tokenizer


def _reference_logprob(model, tokenizer, prefix: str, continuation: str) -> float:
    prefix_ids = tokenizer(prefix)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prefix_ids + cont_ids])
    with torch.no_grad():
        logprobs = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
    return sum(
        float(logprobs[len(prefix_ids) + i - 1, tok]) for i, tok in enumerate(cont_ids)
    )


def test_continuation_logprobs_match_uncached_reference(lm):
    model, tokenizer = lm
    estimator = alt.PMIEvidenceGainEstimator(model, tokenizer, device="cpu")
    prefix = "Fact: 'Blade Runner'"
    continuations = [" is a science fiction film", " yes", " is not a German film"]
    got = estimator._continuation_logprobs(prefix, continuations)
    expected = [_reference_logprob(model, tokenizer, prefix, c) for c in continuations]
    assert got == pytest.approx(expected, abs=1e-4)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def test_pmi_scores_document_gain_on_text_after_entity(lm):
    model, tokenizer = lm
    estimator = alt.PMIEvidenceGainEstimator(
        model, tokenizer, device="cpu", contrastive_temperature=0.5
    )
    doc = "Blade Runner is a 1982 science fiction film directed by Ridley Scott."
    atom = ProblogAtom(atom="{X} is a science fiction film", context=doc)
    [scored] = estimator.score_probability([atom], "Blade Runner")
    head, tail = "Fact: 'Blade Runner'", " is a science fiction film"
    gain = _reference_logprob(
        model, tokenizer, f"Context:\n{doc}\n\n{head}", tail
    ) - _reference_logprob(model, tokenizer, head, tail)
    assert scored.atom == atom.atom
    assert scored.probability == pytest.approx(_sigmoid(gain / 0.5), abs=1e-4)


def test_channel_combines_prior_and_tempered_bayes_factor(lm):
    model, tokenizer = lm
    estimator = alt.ChannelBayesFactorEstimator(
        model, tokenizer, device="cpu", contrastive_temperature=0.5
    )
    doc = "Blade Runner is a 1982 science fiction film directed by Ridley Scott."
    atom = ProblogAtom(atom="{X} is a science fiction film", context=doc)
    negated = ProblogAtom(atom="{X} is not a science fiction film", context=doc)
    [scored] = estimator.score_probability([(atom, negated)], "Blade Runner")

    def ref(prefix, continuation):
        return _reference_logprob(model, tokenizer, prefix, continuation)

    prior_prompt = (
        "Entity: 'Blade Runner'\nStatement: 'Blade Runner' is a science fiction film"
        "\n\nIs the statement true?\nAnswer:"
    )
    prior = ref(prior_prompt, " yes") - ref(prior_prompt, " no")
    log_bf = ref(
        "Fact: 'Blade Runner' is a science fiction film.\n\nWikipedia article:",
        "\n" + doc,
    ) - ref(
        "Fact: 'Blade Runner' is not a science fiction film.\n\nWikipedia article:",
        "\n" + doc,
    )
    assert scored.probability == pytest.approx(_sigmoid(prior + log_bf / 0.5), abs=1e-4)


def test_verbalized_returns_expected_score_over_0_to_10(lm):
    model, tokenizer = lm
    estimator = alt.ExpectedVerbalizedEstimator(model, tokenizer, device="cpu")
    doc = "Blade Runner is a 1982 science fiction film."
    atom = ProblogAtom(atom="{X} is a science fiction film", context=doc)
    [scored] = estimator.score_probability([atom], "Blade Runner")
    prompt = (
        f"Context:\n{doc}\n\nEntity: 'Blade Runner'\n"
        "Statement: 'Blade Runner' is a science fiction film\n\n"
        "On a scale from 0 to 10, how likely is it that the statement is true?\n"
        "Answer:"
    )
    logprobs = torch.tensor(
        [_reference_logprob(model, tokenizer, prompt, f" {k}") for k in range(11)]
    )
    weights = torch.softmax(logprobs, dim=0)
    expected = sum(k / 10 * float(w) for k, w in enumerate(weights))
    assert scored.probability == pytest.approx(expected, abs=1e-4)


def test_three_way_returns_tempered_supported_probability(lm):
    model, tokenizer = lm
    estimator = alt.ThreeWayEstimator(
        model, tokenizer, device="cpu", contrastive_temperature=0.5
    )
    doc = "Blade Runner is a 1982 science fiction film."
    atom = ProblogAtom(atom="{X} is a German film", context=doc)
    [scored] = estimator.score_probability([atom], "Blade Runner")
    prompt = (
        f"Context:\n{doc}\n\nEntity: 'Blade Runner'\n"
        "Statement: 'Blade Runner' is a German film\n\n"
        "Based on the context, the statement is:\n"
        "A) supported\nB) contradicted\nC) not mentioned\nAnswer:"
    )
    logprobs = torch.tensor(
        [_reference_logprob(model, tokenizer, prompt, o) for o in (" A", " B", " C")]
    )
    p_sup = float(torch.softmax(logprobs / 0.5, dim=0)[0])
    assert scored.probability == pytest.approx(p_sup, abs=1e-4)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def test_negation_consistent_averages_yes_no_logits_of_atom_and_negation(lm):
    from llm_bayesian_reasoning.estimators.likelihood_based_yes_no_estimator import (
        LikelihoodBasedYesNoEstimator,
    )

    model, tokenizer = lm
    doc = "Blade Runner is a 1982 science fiction film."
    atom = ProblogAtom(atom="{X} is a German film", context=doc)
    negated = ProblogAtom(atom="{X} is not a German film", context=doc)
    estimator = alt.NegationConsistentYesNoEstimator(model, tokenizer, device="cpu")
    [scored] = estimator.score_probability([(atom, negated)], "Blade Runner")
    yes_no = LikelihoodBasedYesNoEstimator(model, tokenizer, device="cpu")
    p_pos, p_neg = (
        a.probability for a in yes_no.score_probability([atom, negated], "Blade Runner")
    )
    expected = _sigmoid((_logit(p_pos) - _logit(p_neg)) / 2)
    assert scored.atom == atom.atom
    assert scored.probability == pytest.approx(expected, abs=1e-4)


def test_entity_likelihood_pmi_strips_disambiguation_and_subtracts_baseline(lm):
    model, tokenizer = lm
    estimator = alt.EntityLikelihoodPMIEstimator(
        model, tokenizer, device="cpu", contrastive_temperature=0.5
    )
    atom = ProblogAtom(atom="{X} is a film", context="ignored document")
    [scored] = estimator.score_probability([atom], "Less Than Zero (film)")
    template = "Name something for which the following is true: {}.\nAnswer:"
    specific = _reference_logprob(
        model, tokenizer, template.format("it is a film"), " Less Than Zero"
    )
    baseline = _reference_logprob(
        model, tokenizer, template.format("it exists"), " Less Than Zero"
    )
    assert scored.probability == pytest.approx(
        _sigmoid((specific - baseline) / 0.5), abs=1e-4
    )


@pytest.mark.parametrize(
    ("estimator_type", "cls_name"),
    [
        ("PMIEvidenceGain", "PMIEvidenceGainEstimator"),
        ("ChannelBayesFactor", "ChannelBayesFactorEstimator"),
        ("ExpectedVerbalized", "ExpectedVerbalizedEstimator"),
        ("ThreeWay", "ThreeWayEstimator"),
        ("NegationConsistentYesNo", "NegationConsistentYesNoEstimator"),
        ("EntityLikelihoodPMI", "EntityLikelihoodPMIEstimator"),
    ],
)
def test_factory_builds_alternative_estimators_with_temperature_above_one(
    lm, estimator_type, cls_name
):
    from llm_bayesian_reasoning.estimators.factory import (
        create_estimator_from_components,
    )
    from llm_bayesian_reasoning.pipeline.config import EstimatorConfig

    model, tokenizer = lm
    config = EstimatorConfig(
        estimator_type=estimator_type, device="cpu", contrastive_temperature=4.0
    )
    estimator = create_estimator_from_components(config, model, tokenizer)
    assert type(estimator).__name__ == cls_name


@pytest.mark.parametrize(
    "estimator_type", ["ChannelBayesFactor", "NegationConsistentYesNo"]
)
def test_suite_builds_negation_pairs_for_paired_estimators(estimator_type):
    import importlib.util
    from pathlib import Path

    from llm_bayesian_reasoning.pipeline.config import EstimatorType

    path = Path(__file__).parents[1] / "scripts" / "run_experiment_suite.py"
    spec = importlib.util.spec_from_file_location("run_experiment_suite", path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    atom = ProblogAtom(atom="{X} is a film")
    negated = ProblogAtom(atom="{X} is not a film")
    record = {"query": "films", "atoms": [atom], "negated_atoms": [negated]}
    built = suite._build_variant_atoms(record, EstimatorType(estimator_type))
    assert built == [(atom, negated)]


def test_run_pipeline_loads_negation_pairs_for_paired_estimators(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    from llm_bayesian_reasoning.pipeline.config import EstimatorType

    path = Path(__file__).parents[1] / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline", path)
    run_pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_pipeline)
    record = {
        "id": 1,
        "query": "films",
        "parsed": {
            "atoms": ["{x} is a film"],
            "negated_atoms": ["{X} is not a film"],
            "logical query": "({x} is a film)",
        },
    }
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(json.dumps(record) + "\n")
    data, _ = run_pipeline._load_preprocessed(
        data_file, EstimatorType.NEGATION_CONSISTENT_YES_NO
    )
    [(atom, negated)] = data[1]["atoms"]
    assert (atom.atom, negated.atom) == ("{X} is a film", "{X} is not a film")
