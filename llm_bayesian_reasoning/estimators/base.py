from logging import getLogger

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llm_bayesian_reasoning.estimators.scoring_inputs import ScoringInput
from llm_bayesian_reasoning.problog_models.problog_models import ProblogAtom

logger = getLogger(__name__)


class BaseEstimator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
    ):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    def score_probability(
        self,
        predicates: list[ScoringInput],
        entity: str,
    ) -> list[ProblogAtom]:
        raise NotImplementedError("Subclasses must implement this method.")
