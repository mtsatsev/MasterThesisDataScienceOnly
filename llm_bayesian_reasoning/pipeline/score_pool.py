from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_bayesian_reasoning.estimators.atom_estimator import AtomEstimator, Candidate
from llm_bayesian_reasoning.ranking.rerank import Probability, ScoredCandidate, rerank
from llm_bayesian_reasoning.ranking.score_candidate import score_candidate
from llm_bayesian_reasoning.records.query_record import QueryRecord

CONTENT_FREE_CANDIDATE = Candidate(entity="N/A", text="N/A", retrieval_score=0.0)


class QueryResult(BaseModel):
    """The scored and reranked pool of one query, with its content-free scores."""

    model_config = ConfigDict(extra="forbid")

    query_id: int
    retrieved: list[str]
    reranked: list[ScoredCandidate]
    content_free_scores: dict[str, Probability] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pool(self) -> Self:
        """Reject pools whose lists or atoms do not match."""
        retrieved = set()
        for entity in self.retrieved:
            if entity in retrieved:
                raise ValueError(f"Entity {entity!r} is retrieved more than once")
            retrieved.add(entity)

        reranked = set()
        for scored in self.reranked:
            if scored.entity in reranked:
                raise ValueError(f"Entity {scored.entity!r} is reranked more than once")
            if scored.entity not in retrieved:
                raise ValueError(
                    f"Entity {scored.entity!r} is reranked but not retrieved"
                )
            if scored.atom_probabilities.keys() != self.content_free_scores.keys():  # type: ignore
                raise ValueError(
                    f"Entity {scored.entity!r} has other atoms than the "
                    "content-free scores"
                )
            reranked.add(scored.entity)

        for entity in retrieved - reranked:
            raise ValueError(f"Entity {entity!r} is retrieved but not reranked")
        return self


def score_pool(
    record: QueryRecord, candidates: list[Candidate], estimator: AtomEstimator
) -> QueryResult:
    """
    Return the scored and reranked pool of one query.

    Args:
        record (QueryRecord):
            The query, with its atoms and formula.
        candidates (list[Candidate]):
            The retrieved candidates, in retrieval order.
        estimator (AtomEstimator):
            The estimator that scores the atoms of each candidate.

    Returns:
        QueryResult: The pool, with the atoms also scored on a content-free input.
    """
    scored_candidates = [
        score_candidate(record, candidate, estimator) for candidate in candidates
    ]
    content_free = score_candidate(record, CONTENT_FREE_CANDIDATE, estimator)
    return QueryResult(
        query_id=record.query_id,
        retrieved=[candidate.entity for candidate in candidates],
        reranked=rerank(scored_candidates),
        content_free_scores=content_free.atom_probabilities,
    )
