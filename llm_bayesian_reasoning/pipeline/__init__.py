from llm_bayesian_reasoning.pipeline.config import EstimatorConfig, PipelineConfig
from llm_bayesian_reasoning.pipeline.metrics import compute_metrics

__all__ = [
    "EstimatorConfig",
    "PipelineConfig",
    "build_or_load_index",
    "compute_metrics",
    "evaluate_problog",
    "run_pipeline",
]


def build_or_load_index(*args, **kwargs):
    from llm_bayesian_reasoning.pipeline.pipeline import build_or_load_index as _impl

    return _impl(*args, **kwargs)


def run_pipeline(*args, **kwargs):
    from llm_bayesian_reasoning.pipeline.pipeline import run_pipeline as _impl

    return _impl(*args, **kwargs)


def evaluate_problog(program: str) -> float:
    from llm_bayesian_reasoning.pipeline.logic_backends import evaluate_problog_program

    return evaluate_problog_program(program)
