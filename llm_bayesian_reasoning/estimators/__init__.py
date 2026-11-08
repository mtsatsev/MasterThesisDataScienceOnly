from importlib import import_module

_MODULES = {
    "BaseEstimator": "base",
    "LikelihoodBasedEstimator": "likelihood_based_estimator",
    "LikelihoodBasedYesNoEstimator": "likelihood_based_yes_no_estimator",
    "TrueFalseLLMEstimator": "true_false_lm_estimator",
}

__all__ = list(_MODULES)


def __getattr__(name: str):
    """Import an estimator class on first use, so the package does not load transformers."""
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
