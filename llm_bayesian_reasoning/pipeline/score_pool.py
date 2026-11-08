from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_bayesian_reasoning.ranking.rerank import Probability, ScoredCandidate


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
