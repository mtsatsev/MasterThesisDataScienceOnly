from __future__ import annotations

from typing import Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llm_bayesian_reasoning.estimators.base import BaseEstimator
from llm_bayesian_reasoning.estimators.likelihood_based_estimator import (
    LikelihoodBasedEstimator,
)
from llm_bayesian_reasoning.estimators.true_false_lm_estimator import (
    TrueFalseLLMEstimator,
)
from llm_bayesian_reasoning.pipeline.config import EstimatorConfig, EstimatorType


def _common_model_kwargs(estimator_config: EstimatorConfig) -> dict[str, Any]:
    common_kwargs: dict[str, Any] = {}
    if getattr(estimator_config, "quantization", None) is not None:
        common_kwargs["quantization_config"] = estimator_config.quantization
    return common_kwargs


def load_model_and_tokenizer_from_config(
    estimator_config: EstimatorConfig,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load shared model resources for one or more estimator variants."""
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        estimator_config.model_name
    )
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        estimator_config.model_name,
        device_map="auto",
        **_common_model_kwargs(estimator_config),
    )
    return model, tokenizer


def create_estimator_from_components(
    estimator_config: EstimatorConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> BaseEstimator:
    """Create an estimator wrapper from shared model resources."""
    etype = estimator_config.estimator_type

    if etype == EstimatorType.TRUE_FALSE_LLM:
        return TrueFalseLLMEstimator(
            model=model,
            tokenizer=tokenizer,
            device=estimator_config.device,
            true_token=estimator_config.true_token,
            false_token=estimator_config.false_token,
        )

    if etype == EstimatorType.LIKELIHOOD_BASED_PERPLEXITY:
        return LikelihoodBasedEstimator(
            model=model,
            tokenizer=tokenizer,
            device=estimator_config.device,
            contrastive_temperature=estimator_config.contrastive_temperature,
        )

    if etype == EstimatorType.LIKELIHOOD_BASED_CONTRASTIVE:
        return LikelihoodBasedEstimator(
            model=model,
            tokenizer=tokenizer,
            device=estimator_config.device,
            contrastive_temperature=estimator_config.contrastive_temperature,
        )

    raise ValueError(f"Unsupported estimator type: {etype}")


def create_estimator_from_config(estimator_config: EstimatorConfig) -> BaseEstimator:
    """Instantiate an estimator from an EstimatorConfig.

    Supports True/False token estimator and likelihood-based estimators.
    """
    model, tokenizer = load_model_and_tokenizer_from_config(estimator_config)
    return create_estimator_from_components(estimator_config, model, tokenizer)
