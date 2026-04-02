from __future__ import annotations

from typing import Any

from llm_bayesian_reasoning.estimators.base import BaseEstimator
from llm_bayesian_reasoning.estimators.likelihood_based_estimator import (
    LikelihoodBasedEstimator,
)
from llm_bayesian_reasoning.estimators.true_false_lm_estimator import (
    TrueFalseLLMEstimator,
)
from llm_bayesian_reasoning.pipeline.config import EstimatorConfig, EstimatorType


def create_estimator_from_config(estimator_config: EstimatorConfig) -> BaseEstimator:
    """Instantiate an estimator from an EstimatorConfig.

    Supports True/False token estimator and likelihood-based estimators.
    """
    etype = estimator_config.estimator_type
    common_kwargs: dict[str, Any] = {}

    # Pass quantization config through if present
    if getattr(estimator_config, "quantization", None) is not None:
        common_kwargs["quantization_config"] = estimator_config.quantization

    if etype == EstimatorType.TRUE_FALSE_LLM:
        return TrueFalseLLMEstimator.from_pretrained(
            model_name=estimator_config.model_name,
            device=estimator_config.device,
            true_token=estimator_config.true_token,
            false_token=estimator_config.false_token,
            **common_kwargs,
        )

    if etype == EstimatorType.LIKELIHOOD_BASED_PERPLEXITY:
        return LikelihoodBasedEstimator.from_pretrained(
            model_name=estimator_config.model_name,
            device=estimator_config.device,
            **common_kwargs,
        )

    if etype == EstimatorType.LIKELIHOOD_BASED_CONTRASTIVE:
        return LikelihoodBasedEstimator.from_pretrained(
            model_name=estimator_config.model_name,
            device=estimator_config.device,
            contrastive_temperature=estimator_config.contrastive_temperature,
            **common_kwargs,
        )

    raise ValueError(f"Unsupported estimator type: {etype}")
