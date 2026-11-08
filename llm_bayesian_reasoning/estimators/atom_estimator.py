from pydantic import BaseModel, ConfigDict, Field


class Candidate(BaseModel):
    """A retrieved entity with its text and retrieval score, before any scoring."""

    model_config = ConfigDict(extra="forbid")

    entity: str = Field(min_length=1)
    text: str
    retrieval_score: float = Field(allow_inf_nan=False)
